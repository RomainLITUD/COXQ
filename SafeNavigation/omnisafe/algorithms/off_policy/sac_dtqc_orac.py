import torch

from torch.nn.utils.clip_grad import clip_grad_norm_
from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.common.lagrange import Lagrange_exp, Lagrange
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
import time

@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class SACDtqc(SAC):

    def _init_model(self) -> None:
        self._actor_critic = ConstraintActorQCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
            quantile=True,
        ).to(self._device)

    def _init(self,
              ) -> None:
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)
        self.convex = self._cfgs.algo_cfgs.convex
        self.delta = self._cfgs.algo_cfgs.delta
        self.budget = self._cfgs.algo_cfgs.budget
        self.exploration = self._cfgs.algo_cfgs.exploration
        self.tail_reward = self._cfgs.algo_cfgs.tail_reward
        self.tail_cost = self._cfgs.algo_cfgs.tail_cost

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')

    def learn(self) -> tuple[float, float, float]:
        self._logger.log('INFO: Start training')
        start_time = time.time()
        step = 0
        for epoch in range(self._epochs):
            self._epoch = epoch
            rollout_time = 0.0
            update_time = 0.0
            epoch_time = time.time()

            for sample_step in range(
                epoch * self._samples_per_epoch,
                (epoch + 1) * self._samples_per_epoch,
            ):
                step = sample_step * self._update_cycle * self._cfgs.train_cfgs.vector_env_nums

                rollout_start = time.time()
                # set noise for exploration
                if self._cfgs.algo_cfgs.use_exploration_noise:
                    self._actor_critic.actor.noise = self._cfgs.algo_cfgs.exploration_noise

                # collect data from environment
                if self.exploration and self._epoch > self._cfgs.algo_cfgs.warmup_epochs:
                    set_exploration = False
                else:
                    set_exploration = False

                self._env.rollout(
                    rollout_step=self._update_cycle,
                    agent=self._actor_critic,
                    buffer=self._buf,
                    logger=self._logger,
                    use_rand_action=(step <= self._cfgs.algo_cfgs.start_learning_steps),
                    lam = self._lagrange.lagrangian_multiplier.clone().detach(),
                    cost_limit = 1.,
                    delta = 4.,
                    exploration = set_exploration,
                )
                rollout_time += time.time() - rollout_start

                # update parameters
                update_start = time.time()
                if step > self._cfgs.algo_cfgs.start_learning_steps:
                    self._update()
                # if we haven't updated the network, log 0 for the loss
                else:
                    self._log_when_not_update()
                update_time += time.time() - update_start

            eval_start = time.time()
            self._env.eval_policy(
                episode=self._cfgs.train_cfgs.eval_episodes,
                agent=self._actor_critic,
                logger=self._logger,
            )
            eval_time = time.time() - eval_start

            self._logger.store({'Time/Update': update_time})
            self._logger.store({'Time/Rollout': rollout_time})
            self._logger.store({'Time/Evaluate': eval_time})

            if (
                step > self._cfgs.algo_cfgs.start_learning_steps
                and self._cfgs.model_cfgs.linear_lr_decay
            ):
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': step + 1,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': self._actor_critic.actor_scheduler.get_last_lr()[0],
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()
        self._env.close()

        return ep_ret, ep_cost, ep_len

    def _update(self) -> None:
        for _ in range(self._cfgs.algo_cfgs.update_iters):
            data = self._buf.sample_batch()
            self._update_count += 1
            obs, act, reward, cost, done, next_obs = (
                data['obs'],
                data['act'],
                data['reward'],
                -data['cost'],
                data['done'],
                data['next_obs'],
            )

            # cost = -cost
            #for _ in range(10):
            self._update_reward_critic(obs, act, reward, done, next_obs)
            if self._cfgs.algo_cfgs.use_cost:
                self._update_cost_critic(obs, act, cost, done, next_obs)
            self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

            if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
                self._update_actor(obs, act)
                # self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)


    def quantile_huber_loss(self, current_quantiles_in,  # (B, nq, nc)
                            target_quantiles):
        
        current_quantiles = current_quantiles_in.swapaxes(-1, -2)
        n_quantiles = current_quantiles.shape[-1]

        cum_prob = (torch.arange(n_quantiles, dtype=current_quantiles.dtype,
                                device=current_quantiles.device) + 0.5) / n_quantiles

        pairwise_delta = target_quantiles[..., None, None, :] - current_quantiles[..., None] #(B, nc, nq, nqt)

        abs_delta = pairwise_delta.abs()

        # Huber: quadratic for |δ|≤1, linear beyond
        huber_loss = torch.where(abs_delta > 1,
                                abs_delta - 0.5,
                                0.5 * pairwise_delta.pow(2))

        indicator = torch.where(pairwise_delta < 0.,
                                1.,
                                0.).detach()

        # quantile weighting
        loss = (cum_prob[..., None] - indicator).abs() * huber_loss
        return loss.mean(dim=tuple(range(1, loss.ndim)))


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
            next_q = self._actor_critic.target_reward_critic(
                next_obs,
                next_action,
            )

            next_q_r, _ = torch.concatenate(next_q, -1).sort() # (B, nq*nc)
            next_q_value_r = next_q_r - next_logp[:, None] * self._alpha
            target_q_value_r = reward[:,None] + self._cfgs.algo_cfgs.gamma * (1 - done[:,None]) * next_q_value_r

        current_q = self._actor_critic.reward_critic(obs, action)
        current_q = torch.stack(current_q, -1)
        loss = self.quantile_huber_loss(current_q, target_q_value_r)
        loss = loss.mean()

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

            next_quantiles_cost, _ = torch.concatenate(next_q_cost, -1).sort()
            target_q_value_c = cost[:,None] + self._cfgs.algo_cfgs.gamma * (1 - done[:,None]) * next_quantiles_cost
        q_value_c = self._actor_critic.cost_critic(obs, action)
        current_q_c = torch.stack(q_value_c, -1)

        loss_c = self.quantile_huber_loss(current_q_c, target_q_value_c)
        loss_c = loss_c.mean()

        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss_c.backward()
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store(
            {
                'Loss/Loss_cost_critic': loss_c.mean().item(),
                'Value/cost_critic': -current_q_c.mean().item(),
            },
        )


    def _update_actor(# pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
        act: torch.Tensor
    ) -> None:

        loss_pi, cost_pred, _ = self._loss_pi(obs, act)
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

        Jc = cost_pred.detach() # self._logger.get_stats('Metrics/EpCost')[0]
        # Jc = self._logger.get_stats('Metrics/EpCost')[0]/4.
        # print(Jc)
        if self._epoch > self._cfgs.algo_cfgs.warmup_epochs:
            self._lagrange.update_lagrange_multiplier(Jc)
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
            },
        )


    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor
    ) -> torch.Tensor:
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)
        q_reward = self._actor_critic.reward_critic(obs, action)
        q_reward = torch.concatenate(q_reward, -1).mean(-1)

        with torch.no_grad():
            q_current_cost = self._actor_critic.cost_critic(obs, act)
            q_current_cost = torch.stack(q_current_cost, -1)[:, :13]
            q_current_cost = -q_current_cost.mean(-1).mean(-1)

        lam = self._lagrange.lagrangian_multiplier.item()
        q_cost = self._actor_critic.cost_critic(obs, action)
        q_cost = torch.stack(q_cost, -1)[:, :13]

        q_cost = -q_cost.mean(-1).mean(-1)

        rect = torch.clamp(self.convex * (self._lagrange.cost_limit/10 - q_current_cost.mean()), max=lam)

        actor_loss = self._alpha * log_prob - q_reward + (lam-rect).detach()*q_cost
        return actor_loss.mean(), q_current_cost.mean(), 0

    def _log_when_not_update(self) -> None:
        super()._log_when_not_update()
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
            },
        )
