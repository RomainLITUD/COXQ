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

def sigmadot(u, v, var):
    return (u * var * v).sum(dim=-1)


def sigma_mgda_cone_two(
    g_r,   # (B, A) reward-UCB ascent gradient
    g_c,   # (B, A) cost-LCB descent gradient (already negated)
    var,   # (B, A) diag Σ entries (>=0)
    lam,       # scalar λ
    eps = 1e-5,
    ):
    v = g_r + lam * g_c

    s_rr = sigmadot(g_r, g_r, var) + eps
    s_cc = sigmadot(g_c, g_c, var) + eps
    s_rc = sigmadot(g_r, g_c, var)
    v_r  = sigmadot(v,   g_r, var)
    v_c  = sigmadot(v,   g_c, var)

    m_r = v_r < 0.0
    m_c = v_c < 0.0

    # Project to each boundary
    v_Hr = v - (v_r / s_rr).unsqueeze(-1) * g_r
    v_Hc = v - (v_c / s_cc).unsqueeze(-1) * g_c

    # Project to intersection (solve 2x2 Gram system)
    det   = s_rr * s_cc - s_rc * s_rc + eps
    alpha = ( s_cc * v_r - s_rc * v_c) / det
    beta  = (-s_rc * v_r + s_rr * v_c) / det
    v_both = v - alpha.unsqueeze(-1) * g_r - beta.unsqueeze(-1) * g_c

    v_star = torch.where(
        (m_r & m_c).unsqueeze(-1), v_both,
        torch.where((m_r & ~m_c).unsqueeze(-1), v_Hr,
        torch.where((~m_r & m_c).unsqueeze(-1), v_Hc, v))
    )
    return v_star


def cox_explore(
    g1_ucb,     # (B,A)  ∇_a(μ1 + β1 σ1) at a=μ   (reward-UCB)
    g2_lcb,     # (B,A)  ∇_a(μ2 - β2 σ2) at a=μ   (cost-LCB)
    grad_cost,  # (B,A)  h = ∇_a Q2^{cap} (mean or conservative)
    q2_cap_mu,  # (B,)    Q2^{cap}(μ)
    var,        # (B,A)   diag Σ (>=0)
    lam,            # scalar λ (weights = (1, -λ))
    tau,            # scalar cost threshold
    delta,          # scalar KL radius
    eps: float = 1e-5,
):
    # Improvement grads (reward up, cost down)
    g_r = g1_ucb
    g_c = -g2_lcb

    # 1) conflict-free direction in Σ-metric
    v_aligned = sigma_mgda_cone_two(g_r, g_c, var, lam, eps=eps)     # (B,A)

    v_naive   = g_r + lam * g_c
    slack = tau - q2_cap_mu
    use_naive = (slack > 0.).unsqueeze(-1)
    v_star = torch.where(use_naive, v_naive, v_aligned)

    # 2) KL step size along that direction
    denom   = sigmadot(v_star, v_star, var) + eps                 # (B,)
    delta_t = torch.as_tensor(delta, dtype=denom.dtype, device=denom.device)
    eta_kl  = delta_t/torch.sqrt(denom)                      # (B,)

    # 3) Cap violation slope along the ray and available slack
    # tau_t = torch.as_tensor(tau, dtype=denom.dtype, device=denom.device)
    r     = tau - q2_cap_mu                                     # (B,)
    s     = sigmadot(grad_cost, v_star, var)                      # (B,)

    # If s > 0 the step increases cost; shrink to keep violation as small as possible
    eta_cap_upper  = torch.clamp(r / (s + eps), min=0.0)             # (B,) used only when s>0
    eta_when_s_pos = torch.minimum(eta_kl, eta_cap_upper)            # (B,)
    eta            = torch.where(s > 0.0, eta_when_s_pos, eta_kl)

    # 4) Final mean shift along v*: Δa = η Σ v*
    delta_a = eta.unsqueeze(-1) * (var * v_star)                  # (B,A)
    return delta_a


class OffPolicyAdapter(OnlineAdapter):
    """OffPolicy Adapter for OmniSafe.

    :class:`OffPolicyAdapter` is used to adapt the environment to the off-policy training.

    .. note::
        Off-policy training need to update the policy before finish the episode,
        so the :class:`OffPolicyAdapter` will store the current observation in ``_current_obs``.
        After update the policy, the agent will *remember* the current observation and
        use it to interact with the environment.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

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
        self.onpolicy_cost = torch.zeros(1280).to(self._device)

    '''
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
    '''

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
            obs, _ = self._env.reset()
            obs = obs.to(self._device)

            done = False
            while not done:
                act = agent.step(obs, deterministic=True)
                obs, reward, cost, terminated, truncated, info = self._env.step(act)
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
                        Q_UB = q_reward.mean(-1) + 3. * q_reward.std(-1)
                        return Q_UB.mean(-1)
                    
                    def Q_cost_LB(x):
                        tanh_mu_T = torch.tanh(x)                         # → (B, n_a)
                        qs = agent.cost_critic(self._current_obs, tanh_mu_T)
                        q_cost = torch.stack(qs, -1)[:, 16:]
                        Q_LB = q_cost.mean(-1) - 3. * q_cost.std(-1)
                        return Q_LB.mean(-1)
                    
                    def Q_cost_mean(x):
                        tanh_mu_T = torch.tanh(x)                         # → (B, n_a)
                        qs = agent.cost_critic(self._current_obs, tanh_mu_T)
                        q_cost = torch.stack(qs, -1)
                        return q_cost.mean(-1).mean(-1)
                    
                    J_UB = torch.autograd.functional.jacobian(Q_reward_UB,
                                                        pre_tanh_mu_T,
                                                        create_graph=False,
                                                        vectorize=True)
                    
                    J_LB = torch.autograd.functional.jacobian(Q_cost_LB,
                                                        pre_tanh_mu_T,
                                                        create_graph=False,
                                                        vectorize=True)
                    
                    J_mean = torch.autograd.functional.jacobian(Q_cost_mean,
                                                        pre_tanh_mu_T,
                                                        create_graph=False,
                                                        vectorize=True)


                    g_r = J_UB.diagonal(dim1=0, dim2=1).transpose(0,1)
                    g_c = J_LB.diagonal(dim1=0, dim2=1).transpose(0,1)
                    g_m = J_mean.diagonal(dim1=0, dim2=1).transpose(0,1)
                   
                    var = std**2
                    with torch.no_grad():
                        q_cost = agent.cost_critic(self._current_obs, torch.tanh(pre_tanh_mu_T))
                        q_mean_raw = torch.stack(q_cost, -1)
                        q_mean = q_mean_raw[:, 16:].mean(-1).mean(-1)

                        #q_cost = q_mean_raw[:, :16]

                        #Q_LB_value = q_cost.mean(-1) - 3. * q_cost.std(-1)
                        q_mean_explore = q_mean_raw.mean(-1).mean(-1)

                    lam_in = (lam - torch.clamp(1. * (1. - q_mean.mean()), max=lam.item())).detach()

                    delta_a = cox_explore(g_r, g_c, g_m, q_mean_explore, var,
                                          lam_in, cost_limit, delta)

                    mu_E = pre_tanh_mu_T + delta_a

                    act = torch.tanh(Normal(mu_E, std).rsample()).detach()

            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            self.onpolicy_cost = torch.concatenate([self.onpolicy_cost[:-1], cost])

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
