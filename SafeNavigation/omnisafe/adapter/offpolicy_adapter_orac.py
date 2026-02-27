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
"""OffPolicy Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import torch

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils.config import Config

from torch.distributions import Normal


class OffPolicyAdapter(OnlineAdapter):

    _current_obs: torch.Tensor
    _ep_ret: torch.Tensor
    _ep_cost: torch.Tensor
    _ep_len: torch.Tensor

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize a instance of :class:`OffPolicyAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self._current_obs, _ = self.reset()
        self._max_ep_len: int = 1000
        self._reset_log()
        self.nb_conflicts = 0

    def eval_policy(  # pylint: disable=too-many-locals
        self,
        episode: int,
        agent: ConstraintActorQCritic,
        logger: Logger,
    ) -> None:
        """Rollout the environment with deterministic agent action.

        Args:
            episode (int): Number of episodes.
            agent (ConstraintActorCritic): Agent.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        for _ in range(episode):
            ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
            obs, _ = self._eval_env.reset()
            # print(self._env.goal.pos)
            obs = obs.to(self._device)

            done = False
            while not done:
                act = agent.step(obs, deterministic=True)
                obs, reward, cost, terminated, truncated, info = self._eval_env.step(act)
                obs, reward, cost, terminated, truncated = (
                    torch.as_tensor(x, dtype=torch.float32, device=self._device)
                    for x in (obs, reward, cost, terminated, truncated)
                )
                ep_ret += info.get('original_reward', reward).cpu()
                ep_cost += info.get('original_cost', cost).cpu()
                ep_len += 1
                done = bool(terminated[0].item()) or bool(truncated[0].item())

            logger.store(
                {
                    'Metrics/TestEpRet': ep_ret,
                    'Metrics/TestEpCost': ep_cost,
                    'Metrics/TestEpLen': ep_len,
                },
            )

    def rollout(  # pylint: disable=too-many-locals
        self,
        rollout_step: int,
        agent: ConstraintActorQCritic,
        buffer: VectorOffPolicyBuffer,
        logger: Logger,
        use_rand_action: bool,
        lam = 1.0,
        cost_limit = 1.0,
        delta = 4.0,
        exploration = False,
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        .. warning::
            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            rollout_step (int): Number of rollout steps.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor, reward critic,
                and cost critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            use_rand_action (bool): Whether to use random action.
        """
        for _ in range(rollout_step):
            if use_rand_action:
                act = (torch.rand(self.action_space.shape) * 2 - 1).unsqueeze(0).to(self._device)  # type: ignore
            else:
                if not exploration:
                    act = agent.step(self._current_obs, deterministic=False)
                else:
                    mu_pre_tanh, std = agent.step_raw(self._current_obs)
                    pre_tanh_mu_T = mu_pre_tanh.clone().requires_grad_(True)

                    def Q_reward_UB(x):
                        tanh_mu_T = torch.tanh(x)                         # → (B, n_a)
                        qs = agent.reward_critic(self._current_obs, tanh_mu_T)
                        q_reward = torch.stack(qs, -1)
                        Q_UB = q_reward.mean(-1) + 2. * torch.abs(q_reward[:,0] - q_reward[:,1])

                        return Q_UB
                    
                    def Q_cost_LB(x):
                        tanh_mu_T = torch.tanh(x)                         # → (B, n_a)
                        qs = agent.cost_critic(self._current_obs, tanh_mu_T)
                        q_cost = torch.stack(qs, -1)[:, :16] # (B, 16, nc)
                        Q_LB = q_cost.mean(-1) - 1. * q_cost.std(-1)
                        return Q_LB.mean(-1)
                    
                    
                    J_UB = torch.autograd.functional.jacobian(Q_reward_UB,
                                                        pre_tanh_mu_T,
                                                        create_graph=False,
                                                        vectorize=True)
                    
                    J_LB = torch.autograd.functional.jacobian(Q_cost_LB,
                                                        pre_tanh_mu_T,
                                                        create_graph=False,
                                                        vectorize=True)


                    g_r = J_UB.diagonal(dim1=0, dim2=1).transpose(0,1)
                    g_c = J_LB.diagonal(dim1=0, dim2=1).transpose(0,1)

                    var = std**2
                    with torch.no_grad():
                        q_cost = agent.cost_critic(self._current_obs, torch.tanh(pre_tanh_mu_T))
                        q_mean_raw = torch.stack(q_cost, -1)
                        q_cost = q_mean_raw[:, :16] # (B, 16, nc)
                        Q_LB_value = q_cost.mean(-1) - 1. * q_cost.std(-1)

                    lam_in = (lam - torch.clamp(10. * (1. - Q_LB_value.mean()), max=lam.item())).detach()

                    # conflict = conflict_assert(g_r, -g_c, var, lam_in)
                    # above = int((-q_mean_raw.mean() > 1.).item())
                    # self.nb_conflicts += int(bool(conflict) and bool(above))
                    grad = g_r - lam_in * g_c

                    denom = torch.sum(grad**2 * var, -1)

                    mu_C = delta * var * grad / (torch.sqrt(denom[:,None])+1e-5)

                    mu_E = pre_tanh_mu_T +mu_C

                    act = torch.tanh(Normal(mu_E, std).rsample()).detach()

            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            self._log_value(reward=reward, cost=cost, info=info)
            real_next_obs = next_obs.clone()
            for idx, done in enumerate(torch.logical_or(terminated, truncated)):
                if done:
                    if 'final_observation' in info:
                        real_next_obs[idx] = info['final_observation'][idx]
                    self._log_metrics(logger, idx)
                    self._reset_log(idx)

            buffer.store(
                obs=self._current_obs,
                act=act,
                reward=reward,
                cost=cost,
                done=torch.logical_and(terminated, torch.logical_xor(terminated, truncated)),
                next_obs=real_next_obs,
            )

            self._current_obs = next_obs

    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: dict[str, Any],
    ) -> None:
        """Log value.

        .. note::
            OmniSafe uses :class:`RewardNormalizer` wrapper, so the original reward and cost will
            be stored in ``info['original_reward']`` and ``info['original_cost']``.

        Args:
            reward (torch.Tensor): The immediate step reward.
            cost (torch.Tensor): The immediate step cost.
            info (dict[str, Any]): Some information logged by the environment.
        """
        self._ep_ret += info.get('original_reward', reward).cpu()
        self._ep_cost += info.get('original_cost', cost).cpu()
        self._ep_len += 1

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        """Log metrics, including ``EpRet``, ``EpCost``, ``EpLen``.

        Args:
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            idx (int): The index of the environment.
        """
        if hasattr(self._env, 'spec_log'):
            self._env.spec_log(logger)
        logger.store(
            {
                'Metrics/EpRet': self._ep_ret[idx],
                'Metrics/EpCost': self._ep_cost[idx],
                'Metrics/EpLen': self._ep_len[idx],
            },
        )

    def _reset_log(self, idx: int | None = None) -> None:
        """Reset the episode return, episode cost and episode length.

        Args:
            idx (int or None, optional): The index of the environment. Defaults to None
                (single environment).
        """
        if idx is None:
            self._ep_ret = torch.zeros(self._env.num_envs)
            self._ep_cost = torch.zeros(self._env.num_envs)
            self._ep_len = torch.zeros(self._env.num_envs)
        else:
            self._ep_ret[idx] = 0.0
            self._ep_cost[idx] = 0.0
            self._ep_len[idx] = 0.0
