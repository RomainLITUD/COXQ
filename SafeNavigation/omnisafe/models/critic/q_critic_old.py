# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of Q Critic."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.func import vmap, functional_call
from functorch import make_functional_with_buffers
from torch.utils._pytree import tree_map

from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network
from torch.func import functional_call, stack_module_state, vmap

class QCritic(Critic):
    """Implementation of Q Critic.

    A Q-function approximator that uses a multi-layer perceptron (MLP) to map observation-action
    pairs to Q-values. This class is an inherit class of :class:`Critic`. You can design your own
    Q-function approximator by inheriting this class or :class:`Critic`.

    The Q critic network has two modes:

    .. hint::
        - ``use_obs_encoder = False``: The input of the network is the concatenation of the
            observation and action.
        - ``use_obs_encoder = True``: The input of the network is the concatenation of the output of
            the observation encoder and action.

    For example, in :class:`DDPG`, the action is not directly concatenated with the observation, but
    is concatenated with the output of the observation encoder.

    .. note::
        The Q critic network contains multiple critics, and the output of the network :meth`forward`
        is a list of Q-values. If you want to get the single Q-value of a specific critic, you need
        to use the index to get it.

    Args:
        obs_space (OmnisafeSpace): observation space.
        act_space (OmnisafeSpace): action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
        num_critics (int, optional): Number of critics. Defaults to 1.
        use_obs_encoder (bool, optional): Whether to use observation encoder, only used in q critic.
            Defaults to False.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 1,
        use_obs_encoder: bool = False,
        quantile: bool = False,
    ) -> None:
        """Initialize an instance of :class:`QCritic`."""
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes,
            activation,
            weight_initialization_mode,
            num_critics,
            use_obs_encoder,
        )
        self.quantile = quantile
        self.net_lst: list[nn.Sequential] = []
        for idx in range(self._num_critics):
            if self._use_obs_encoder:
                obs_encoder = build_mlp_network(
                    [self._obs_dim, hidden_sizes[0]],
                    activation=activation,
                    output_activation=activation,
                    weight_initialization_mode=weight_initialization_mode,
                )
                if self.quantile:
                    net = build_mlp_network(
                        [hidden_sizes[0] + self._act_dim] + hidden_sizes[1:] + [25],
                        activation=activation,
                        weight_initialization_mode=weight_initialization_mode,
                    )
                else:
                    net = build_mlp_network(
                        [hidden_sizes[0] + self._act_dim] + hidden_sizes[1:] + [1],
                        activation=activation,
                        weight_initialization_mode=weight_initialization_mode,
                    )
                critic = nn.Sequential(obs_encoder, net)
            else:
                if self.quantile:
                    net = build_mlp_network(
                        [self._obs_dim + self._act_dim, *hidden_sizes, 25],
                        activation=activation,
                        weight_initialization_mode=weight_initialization_mode,
                    )
                else:
                    net = build_mlp_network(
                        [self._obs_dim + self._act_dim, *hidden_sizes, 1],
                        activation=activation,
                        weight_initialization_mode=weight_initialization_mode,
                    )
                critic = nn.Sequential(net)
            self.net_lst.append(critic)
            self.add_module(f'critic_{idx}', critic)

    def critic_forward(self, critic, obs, act, use_obs_encoder):
        if use_obs_encoder:
            obs_enc = critic[0](obs)
            inp = torch.cat([obs_enc, act], dim=-1)
            return torch.squeeze(critic[1](inp), -1)
        else:
            inp = torch.cat([obs, act], dim=-1)
            return torch.squeeze(critic(inp), -1)

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Forward function.

        As a multi-critic network, the output of the network is a list of Q-values. If you want to
        use it as a single-critic network, you only need to set the ``num_critics`` parameter to 1
        when initializing the network, and then use the index 0 to get the Q-value.

        Args:
            obs (torch.Tensor): Observation from environments.
            act (torch.Tensor): Action from actor .

        Returns:
            A list of Q critic values of action and observation pair.
        """

        """
        res = []
        for critic in self.net_lst:
            if self._use_obs_encoder:
                obs_encode = critic[0](obs)
                res.append(torch.squeeze(critic[1](torch.cat([obs_encode, act], dim=-1)), -1))
            else:
                res.append(torch.squeeze(critic(torch.cat([obs, act], dim=-1)), -1))
        return res
        """
        x = torch.cat([obs, act], dim=-1)
        params_b, buffers_b = stack_module_state(self.net_lst)
        proto = self.net_lst[0]  # structure prototype

        def one_q(params, buffers, x_):
            return functional_call(proto, (params, buffers), (x_,)).squeeze(-1)  # [B]

        q_all = vmap(one_q, in_dims=(0, 0, None))(params_b, buffers_b, x)        # [N, B]
        return list(q_all.unbind(0))
