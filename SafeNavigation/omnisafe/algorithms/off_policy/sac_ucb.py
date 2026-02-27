import torch

from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.common.lagrange import Lagrange
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class SACUCB(SAC):

    def _init_model(self) -> None:
        self._actor_critic = ConstraintActorQCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
            quantile=False,
        ).to(self._device)

    def _init(self,
              ) -> None:
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')

    def _update(self) -> None:
        for _ in range(self._cfgs.algo_cfgs.update_iters):
            data = self._buf.sample_batch()
            self._update_count += 1
            obs, act, reward, cost, done, next_obs = (
                data['obs'],
                data['act'],
                data['reward'],
                data['cost'],
                data['done'],
                data['next_obs'],
            )

            self._update_reward_critic(obs, act, reward, done, next_obs)
            if self._cfgs.algo_cfgs.use_cost:
                self._update_cost_critic(obs, act, cost, done, next_obs)

            if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
                self._update_actor(obs)
                self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=False)
            next_logp = self._actor_critic.actor.log_prob(next_action)
            next_q_r = self._actor_critic.target_reward_critic(
                next_obs,
                next_action,
            )

            next_q_r = torch.stack(next_q_r, -1)
            # (next_q_r.shape)

            next_q_value_r = torch.amin(next_q_r, -1) - next_logp * self._alpha
            target_q_value_r = reward + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_r

        current_q = self._actor_critic.reward_critic(obs, action)
        current_q = torch.stack(current_q, -1)

        loss = nn.functional.mse_loss(current_q[..., 0], target_q_value_r) + nn.functional.mse_loss(
            current_q[..., 1],
            target_q_value_r,
        )

        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss.backward()
        self._actor_critic.reward_critic_optimizer.step()
        self._logger.store(
            {
                'Loss/Loss_reward_critic': loss.mean().item(),
                'Value/reward_critic': current_q.mean().item(),
            },
        )


    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:

        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=True)
            next_q_cost = self._actor_critic.target_cost_critic(next_obs, next_action)

            next_q_cost = torch.amax(torch.stack(next_q_cost, -1), -1)

            target_q_value_c = cost + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_cost

        q_value_c = self._actor_critic.cost_critic(obs, action)
        current_q_c = torch.stack(q_value_c, -1)

        loss_c = nn.functional.mse_loss(current_q_c[..., 0], target_q_value_c) + nn.functional.mse_loss(
            current_q_c[..., 1],
            target_q_value_c,
        )

        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss_c.backward()
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store(
            {
                'Loss/Loss_cost_critic': loss_c.mean().item(),
                'Value/cost_critic': current_q_c.mean().item(),
            },
        )


    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
    ) -> None:

        loss_pi = self._loss_pi(obs)
        self._actor_critic.actor_optimizer.zero_grad()
        loss_pi.backward()
        self._actor_critic.actor_optimizer.step()
        self._logger.store(
            {
                'Loss/Loss_pi': loss_pi.mean().item(),
            },
        )

        if self._cfgs.algo_cfgs.auto_alpha:
            with torch.no_grad():
                action = self._actor_critic.actor.predict(obs, deterministic=False)
                log_prob = self._actor_critic.actor.log_prob(action)
            alpha_loss = -self._log_alpha * (log_prob + self._target_entropy).mean()

            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()
            self._logger.store(
                {
                    'Loss/alpha_loss': alpha_loss.mean().item(),
                },
            )

        self._logger.store(
            {
                'Value/alpha': self._alpha,
            },
        )

        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        
        self._lagrange.update_lagrange_multiplier(Jc)
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
            },
        )


    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)
        q_reward = self._actor_critic.reward_critic(obs, action)
        q_reward = torch.amin(torch.stack(q_reward, -1), -1)

        loss_r = self._alpha * log_prob - q_reward

        q_cost = self._actor_critic.cost_critic(obs, action)
        q_cost = torch.amax(torch.stack(q_cost, -1), -1)

        lam = self._lagrange.lagrangian_multiplier.item()
        
        return (loss_r + lam * q_cost).mean() / (1 + lam)


    def _log_when_not_update(self) -> None:
        super()._log_when_not_update()
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
            },
        )
