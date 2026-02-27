import torch

from torch.nn.utils.clip_grad import clip_grad_norm_
from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.common.lagrange import Lagrange_exp, Lagrange
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.models.actor_critic.actor_q_critic import MOActorQCritic
import time

@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class COPQ(SAC):

    def _init_model(self) -> None:
        self._actor_critic = MOActorQCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
            quantile=False,
            output_dimension = 2,
        ).to(self._device)

    def _init(self,
              ) -> None:
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)
        self.convex = self._cfgs.algo_cfgs.convex
        self.delta = self._cfgs.algo_cfgs.delta
        self.budget = self._cfgs.algo_cfgs.budget
        self.exploration = False

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
                    set_exploration = True
                else:
                    set_exploration = False

                self._env.rollout(
                    rollout_step=self._update_cycle,
                    agent=self._actor_critic,
                    buffer=self._buf,
                    logger=self._logger,
                    use_rand_action=(step <= self._cfgs.algo_cfgs.start_learning_steps),
                    lam = self._lagrange.lagrangian_multiplier.clone().detach(),
                    cost_limit = self.budget,
                    delta = self._lagrange.delta_fly.clone().detach(),
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

            self._update_critic(obs, act, reward, cost, done, next_obs)
            self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

            if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
                self._update_actor(obs, act)

    def _update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=False)
            next_logp = self._actor_critic.actor.log_prob(next_action)
            next_q = self._actor_critic.target_critic(
                next_obs,
                next_action,
            )

            next_q = torch.stack(next_q, -1) # (B, 2, nc)
            # print(next_q.shape)
            q_mean = next_q.mean(dim=-1, keepdim=True)
            qm = next_q - q_mean
            q_cov = qm @ qm.transpose(-1, -2) / 3
            q_cov = 0.5 * (q_cov + q_cov.transpose(-1, -2)) + 1e-4* torch.eye(2, device=q_cov.device, dtype=q_cov.dtype)

            r_c = torch.stack([cost, reward], -1).squeeze()

            l11 = torch.sqrt(torch.clamp(q_cov[:, 0, 0], min=1e-6)) 
            l21 = q_cov[:, 0, 1] / l11
            l22 = torch.sqrt(torch.clamp(q_cov[:, 1, 1] - l21 ** 2, min=1e-6))

            lam = self._lagrange.lagrangian_multiplier.data.item() + 1e-3
            #v1 = lam/((lam**2+1.)**0.5)
            #v2 = 1./((lam**2+1.)**0.5)

            # q_current_cost = -next_q[:, 0].mean(-1) + next_q[:, 0].std(-1)*2/3

            # rect = torch.clamp(self.convex * (self._lagrange.cost_limit/10 - q_current_cost), max=lam)

            # llam = (lam-rect).detach()

            # v1 = llam/((llam**2+1.)**0.5)
            # v2 = 1./((llam**2+1.)**0.5)

            cholesky_term = torch.stack([l11, (l21+l22)], -1)

            ent_term = torch.stack([torch.zeros_like(next_logp), next_logp * self._alpha], -1) # (B, 2)
            next_q_value_r = next_q.mean(-1) - cholesky_term
            target_q_value_r = r_c + self._cfgs.algo_cfgs.gamma * (1 - done[:,None]) * (next_q_value_r - ent_term)

        current_q = self._actor_critic.critic(obs, action)
        current_q = torch.stack(current_q, -1)

        current_reward = current_q[:,1]
        current_cost = current_q[:,0]
        loss_reward = torch.nn.functional.mse_loss(current_reward, target_q_value_r[:, 1, None])
        loss_cost = torch.nn.functional.mse_loss(current_cost, target_q_value_r[:, 0, None])
        loss = loss_reward.mean() + loss_cost.mean()

        self._actor_critic.critic_optimizer.zero_grad()
        loss.backward()
        self._actor_critic.critic_optimizer.step()
        self._logger.store(
            {
                'Loss/Loss_reward_critic': loss_reward.mean().item(),
                'Value/reward_critic': current_reward.mean().item(),
                'Loss/Loss_cost_critic': loss_cost.mean().item(),
                'Value/cost_critic': -current_cost.mean().item(),
            },
        )

    def _update_actor(  # pylint: disable=too-many-arguments
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
        # print(Jc)
        if self._epoch > self._cfgs.algo_cfgs.warmup_epochs:
            self._lagrange.update_lagrange_multiplier(Jc)
            #cost_test = self._logger.get_stats('Metrics/TestEpCost')[0]
            #reward_test = self._logger.get_stats('Metrics/TestEpRet')[0]
            #self._lagrange.update_auto_truncation(cost_test/10., Jc)
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
        next_q = self._actor_critic.critic(obs, action)
        next_q = torch.stack(next_q, -1) # (B, 2, nc)
        q_mean = next_q.mean(dim=-1, keepdim=True)
        qm = next_q - q_mean
        q_cov = qm @ qm.transpose(-1, -2) / 3
        q_cov = 0.5 * (q_cov + q_cov.transpose(-1, -2)) + 1e-4* torch.eye(2, device=q_cov.device, dtype=q_cov.dtype)

        l11 = torch.sqrt(torch.clamp(q_cov[:, 0, 0], min=1e-6)) 
        l21 = q_cov[:, 0, 1] / l11
        l22 = torch.sqrt(torch.clamp(q_cov[:, 1, 1] - l21 ** 2, min=1e-6))

        lam = self._lagrange.lagrangian_multiplier.item() + 1e-3
        #v1 = lam/((lam**2+1.)**0.5)
        #v2 = 1./((lam**2+1.)**0.5)

        with torch.no_grad():
            q_current = self._actor_critic.critic(obs, act)
            q_current = torch.stack(q_current, -1)

            q_current_cost = -q_current[:, 0].mean(-1) + q_current[:, 0].std(-1)*2/3

        rect = torch.clamp(self.convex * (self._lagrange.cost_limit/10 - q_current_cost), max=lam)

        # llam = (lam-rect).detach()

        # v1 = llam/((llam**2+1.)**0.5)
        # v2 = 1./((llam**2+1.)**0.5)

        q_reward = next_q[:,1].mean(-1) - (l21 + l22) #/(2**0.5)
        q_cost = -next_q[:,0].mean(-1) + l11 #/(2**0.5)
        actor_loss = self._alpha * log_prob - q_reward + (lam-rect).detach()*q_cost
        # actor_loss = self._alpha * log_prob - q_reward + lam*q_cost
        return actor_loss.mean(), q_current_cost.mean(), 0 #-q_current_cost_raw.mean()
        # return actor_loss.mean(), q_cost.mean(), 0


    def _log_when_not_update(self) -> None:
        super()._log_when_not_update()
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
            },
        )

        # self._logger.store(
        #     {
        #         'Metrics/Delta': self._lagrange.delta_fly.data.item(),
        #     },
        # )
