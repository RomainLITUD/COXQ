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
"""Implementation of Lagrange."""

from __future__ import annotations

import torch


class Lagrange:
    """Base class for Lagrangian-base Algorithms.

    This class implements the Lagrange multiplier update and the Lagrange loss.

    ..  note::
        Any traditional policy gradient algorithm can be converted to a Lagrangian-based algorithm
        by inheriting from this class and implementing the :meth:`_loss_pi` method.

    Examples:
        >>> from omnisafe.common.lagrange import Lagrange
        >>> def loss_pi(self, data):
        ...     # implement your own loss function here
        ...     return loss

    You can also inherit this class to implement your own Lagrangian-based algorithm, with any
    policy gradient method you like in OmniSafe.

    Examples:
        >>> from omnisafe.common.lagrange import Lagrange
        >>> class CustomAlgo:
        ...     def __init(self) -> None:
        ...         # initialize your own algorithm here
        ...         super().__init__()
        ...         # initialize the Lagrange multiplier
        ...         self.lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    Args:
        cost_limit (float): The cost limit.
        lagrangian_multiplier_init (float): The initial value of the Lagrange multiplier.
        lambda_lr (float): The learning rate of the Lagrange multiplier.
        lambda_optimizer (str): The optimizer for the Lagrange multiplier.
        lagrangian_upper_bound (float or None, optional): The upper bound of the Lagrange multiplier.
            Defaults to None.

    Attributes:
        cost_limit (float): The cost limit.
        lambda_lr (float): The learning rate of the Lagrange multiplier.
        lagrangian_upper_bound (float, optional): The upper bound of the Lagrange multiplier.
            Defaults to None.
        lagrangian_multiplier (torch.nn.Parameter): The Lagrange multiplier.
        lambda_range_projection (torch.nn.ReLU): The projection function for the Lagrange multiplier.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        cost_limit: float,
        lagrangian_multiplier_init: float,
        lambda_lr: float,
        lambda_optimizer: str,
        lagrangian_upper_bound: float | None = None,
    ) -> None:
        """Initialize an instance of :class:`Lagrange`."""
        self.cost_limit: float = cost_limit
        self.lambda_lr: float = lambda_lr
        self.lagrangian_upper_bound: float | None = lagrangian_upper_bound

        init_value = max(lagrangian_multiplier_init, 0.01)
        self.lagrangian_multiplier: torch.nn.Parameter = torch.nn.Parameter(
            torch.as_tensor(init_value),
            requires_grad=True,
        )

        self.delta_fly: torch.nn.Parameter = torch.nn.Parameter(
            torch.as_tensor(4.),
            requires_grad=True,
        )

        self.auto_truncation: torch.nn.Parameter = torch.nn.Parameter(
            torch.as_tensor(0.2),
            requires_grad=True,
        )

        self.auto_truncation_reward: torch.nn.Parameter = torch.nn.Parameter(
            torch.as_tensor(5.),
            requires_grad=True,
        )


        self.lambda_range_projection: torch.nn.ReLU = torch.nn.ReLU()
        # fetch optimizer from PyTorch optimizer package
        assert hasattr(
            torch.optim,
            lambda_optimizer,
        ), f'Optimizer={lambda_optimizer} not found in torch.'
        torch_opt = getattr(torch.optim, lambda_optimizer)
        self.lambda_optimizer: torch.optim.Optimizer = torch_opt(
            [
                self.lagrangian_multiplier,
            ],
            lr=lambda_lr,
        )

        self.delta_optimizer: torch.optim.Optimizer = torch_opt(
            [
                self.delta_fly,
            ],
            lr=5e-4,
        )

        self.truncation_optimizer: torch.optim.Optimizer = torch_opt(
            [
                self.auto_truncation,
            ],
            lr=1e-3,
        )

        self.truncation_reward_optimizer: torch.optim.Optimizer = torch_opt(
            [
                self.auto_truncation_reward,
            ],
            lr=1e-2,
        )



    def compute_lambda_loss(self, mean_ep_cost: float) -> torch.Tensor:
        """Penalty loss for Lagrange multiplier.

        .. note::
            ``mean_ep_cost`` is obtained from ``self.logger.get_stats('EpCosts')[0]``, which is
            already averaged across MPI processes.

        Args:
            mean_ep_cost (float): mean episode cost.

        Returns:
            Penalty loss for Lagrange multiplier.
        """
        return -self.lagrangian_multiplier * (mean_ep_cost - self.cost_limit/10.)
    
    def compute_delta_loss(self, train_cost: float) -> torch.Tensor:
        # return self.delta_fly * (train_cost - self.cost_limit/10.)
        return self.delta_fly * (train_cost - 2.5)

    def update_lagrange_multiplier(self, Jc: float) -> None:
        r"""Update Lagrange multiplier (lambda).

        We update the Lagrange multiplier by minimizing the penalty loss, which is defined as:

        .. math::

            \lambda ^{'} = \lambda + \eta \cdot (J_C - J_C^*)

        where :math:`\lambda` is the Lagrange multiplier, :math:`\eta` is the learning rate,
        :math:`J_C` is the mean episode cost, and :math:`J_C^*` is the cost limit.

        Args:
            Jc (float): mean episode cost.
        """
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(Jc)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(
            0.0,
            100., #self.lagrangian_upper_bound,
        )  # enforce: lambda in [0, inf]

    def update_delta_fly(self, Jc: float) -> None:
        self.delta_optimizer.zero_grad()
        delta_loss = self.compute_delta_loss(Jc)
        delta_loss.backward()
        self.delta_optimizer.step()
        self.delta_fly.data.clamp_(
            0.05,
            6.0,
        )  # enforce: lambda in [0, inf]

    def update_auto_truncation(self, predict_cost):
        self.truncation_optimizer.zero_grad()
        truncation_loss = self.auto_truncation * (1. - predict_cost) # test_cost higher --> truncate more
        truncation_loss.backward()
        self.truncation_optimizer.step()
        self.auto_truncation.data.clamp_(
            0.01, 30.,
        )

    def update_auto_truncation_reward(self, test_reward, predict_reward):
        self.truncation_reward_optimizer.zero_grad()
        truncation_loss = self.auto_truncation_reward * (test_reward - predict_reward) # test_cost higher --> truncate less
        truncation_loss.backward()
        self.truncation_reward_optimizer.step()
        self.auto_truncation_reward.data.clamp_(
            1.2,
            50.0,
        )



class Lagrange_exp:

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        cost_limit: float,
        lagrangian_multiplier_init: float,
        lambda_lr: float,
        lambda_optimizer: str,
        lagrangian_upper_bound: float | None = None,
    ) -> None:
        """Initialize an instance of :class:`Lagrange`."""
        self.cost_limit: float = cost_limit
        self.lambda_lr: float = lambda_lr
        self.lagrangian_upper_bound: float | None = lagrangian_upper_bound

        self.delta_fly: torch.nn.Parameter = torch.nn.Parameter(
            torch.as_tensor(4.),
            requires_grad=True,
        )

        init_value = lagrangian_multiplier_init
        self.lagrangian_multiplier: torch.nn.Parameter = torch.nn.Parameter(
            torch.as_tensor(init_value),
            requires_grad=True,
        )
        self.lambda_range_projection: torch.nn.ReLU = torch.nn.ReLU()
        # fetch optimizer from PyTorch optimizer package
        assert hasattr(
            torch.optim,
            lambda_optimizer,
        ), f'Optimizer={lambda_optimizer} not found in torch.'
        torch_opt = getattr(torch.optim, lambda_optimizer)
        self.lambda_optimizer: torch.optim.Optimizer = torch_opt(
            [
                self.lagrangian_multiplier,
            ],
            lr=lambda_lr,
        )

        self.delta_optimizer: torch.optim.Optimizer = torch_opt(
            [
                self.delta_fly,
            ],
            lr=1e-4,
        )

    def compute_lambda_loss(self, mean_ep_cost: float) -> torch.Tensor:
        return -self.lagrangian_multiplier.exp() * (mean_ep_cost - self.cost_limit/10.)
    
    def compute_delta_loss(self, train_cost: float) -> torch.Tensor:
        return self.delta_fly * (train_cost - self.cost_limit/10.)


    def update_lagrange_multiplier(self, Jc: float) -> None:
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(Jc)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        #self.lagrangian_multiplier.data.clamp_(
        #    -self.lagrangian_upper_bound,
        #    self.lagrangian_upper_bound,
        #)  # enforce: lambda in [0, inf]

    def update_delta_fly(self, Jc: float) -> None:
        self.delta_optimizer.zero_grad()
        delta_loss = self.compute_delta_loss(Jc)
        delta_loss.backward()
        self.delta_optimizer.step()
        self.delta_fly.data.clamp_(
            0.01,
            6.0,
        )  # enforce: lambda in [0, inf]
