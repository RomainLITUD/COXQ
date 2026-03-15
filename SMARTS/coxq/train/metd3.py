import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union, Tuple, List

import numpy as np
import torch as th
#from torch.func import vmap
from gymnasium import spaces
from torch.nn import functional as F
import time
import sys
import pathlib
import io
import copy
from copy import deepcopy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs

from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update

from train.custom_buffer import DEDictReplayBuffer
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps

from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl

from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise

from contrib_policy.network import TD3Policy

SelfDEOffPolicyAlgorithm = TypeVar("SelfDEOffPolicyAlgorithm", bound="DEOffPolicyAlgorithm")

SelfDETD3 = TypeVar("SelfDETD3", bound="DETD3")


def sigmadot(u, v, var):
    return (u * var * v).sum(dim=-1)


def sigma_mgda_cone_two(
    g_r,   # (B, A) reward-UCB ascent gradient
    g_c,   # (B, A) cost-LCB descent gradient (already negated)
    var,   # (B, A) diag Σ entries (>=0)
    lam,       # scalar λ
    eps = 1e-5,
    ):
    v = g_r + lam[:, None] * g_c

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

    v_star = th.where(
        (m_r & m_c).unsqueeze(-1), v_both,
        th.where((m_r & ~m_c).unsqueeze(-1), v_Hr,
        th.where((~m_r & m_c).unsqueeze(-1), v_Hc, v))
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

    v_star = sigma_mgda_cone_two(g_r, g_c, var, lam, eps=eps)     # (B,A)

    denom   = sigmadot(v_star, v_star, var) + eps                 # (B,)
    delta_t = th.as_tensor(delta, dtype=denom.dtype, device=denom.device)
    eta_kl  = delta_t/th.sqrt(denom)                      # (B,)

    s     = sigmadot(grad_cost, v_star, var)                      # (B,)

    # If s > 0 the step increases cost; shrink to keep violation as small as possible
    eta            = th.where(s > 0.0, 0.0, eta_kl)

    # 4) Final mean shift along v*: Δa = η Σ v*
    delta_a = eta.unsqueeze(-1) * (var * v_star)                  # (B,A)
    return delta_a


class DEOffPolicyAlgorithm(BaseAlgorithm):
    actor: th.nn.Module

    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        beta_policy: float = 4.,
        delta: float = 0.4,
        model_type: str = 'baseline',
        explore_mode = False,
        cost_limit = 2.5,
        topk = 6,
        learning_method = 'saclag',
        share_feature: bool = False,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 10000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = DEDictReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            supported_action_spaces=supported_action_spaces,
        )
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.beta_policy = beta_policy
        self.delta = delta
        self.gradient_steps = gradient_steps
        self.action_noise = action_noise
        self.optimize_memory_usage = optimize_memory_usage
        self.replay_buffer: Optional[ReplayBuffer] = None
        self.replay_buffer_class = replay_buffer_class
        self.replay_buffer_kwargs = replay_buffer_kwargs or {}
        self._episode_storage = None

        self.model_type = model_type
        self.explore_mode = explore_mode
        self.share_feature = share_feature

        self.cost_limit = cost_limit
        self.topk = topk
        self.learning_method = learning_method

        self.train_freq = train_freq

        if sde_support:
            self.policy_kwargs["use_sde"] = self.use_sde
        self.use_sde_at_warmup = use_sde_at_warmup

    def _convert_train_freq(self) -> None:
        if not isinstance(self.train_freq, TrainFreq):
            train_freq = self.train_freq

            if not isinstance(train_freq, tuple):
                train_freq = (train_freq, "step")

            try:
                train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))  # type: ignore[assignment]
            except ValueError as e:
                raise ValueError(
                    f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!"
                ) from e

            if not isinstance(train_freq[0], int):
                raise ValueError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")

            self.train_freq = TrainFreq(*train_freq)  # type: ignore[assignment,arg-type]

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.replay_buffer_class = DEDictReplayBuffer
            else:
                self.replay_buffer_class = ReplayBuffer

        if self.replay_buffer is None:
            # Make a local copy as we should not pickle
            # the environment when using HerReplayBuffer
            replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
            if issubclass(self.replay_buffer_class, HerReplayBuffer):
                assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"
                replay_buffer_kwargs["env"] = self.env
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **replay_buffer_kwargs,
            )

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            None,
            self.model_type,
            self.share_feature,
            self.ensemble_size,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

    def save_replay_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        assert self.replay_buffer is not None, "The replay buffer is not defined"
        save_to_pkl(path, self.replay_buffer, self.verbose)

    def load_replay_buffer(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        truncate_last_traj: bool = True,
    ) -> None:
        self.replay_buffer = load_from_pkl(path, self.verbose)
        assert isinstance(self.replay_buffer, ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"

        if not hasattr(self.replay_buffer, "handle_timeout_termination"):  # pragma: no cover
            self.replay_buffer.handle_timeout_termination = False
            self.replay_buffer.timeouts = np.zeros_like(self.replay_buffer.dones)

        if isinstance(self.replay_buffer, HerReplayBuffer):
            assert self.env is not None, "You must pass an environment at load time when using `HerReplayBuffer`"
            self.replay_buffer.set_env(self.env)
            if truncate_last_traj:
                self.replay_buffer.truncate_last_trajectory()

        self.replay_buffer.device = self.device

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:

        replay_buffer = self.replay_buffer

        truncate_last_traj = (
            self.optimize_memory_usage
            and reset_num_timesteps
            and replay_buffer is not None
            and (replay_buffer.full or replay_buffer.pos > 0)
        )

        if truncate_last_traj:
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated, "
                "see https://github.com/DLR-RM/stable-baselines3/issues/46."
                "You should use `reset_num_timesteps=False` or `optimize_memory_usage=False`"
                "to avoid that issue."
            )
            assert replay_buffer is not None  # for mypy
            # Go to the previous index
            pos = (replay_buffer.pos - 1) % replay_buffer.buffer_size
            replay_buffer.dones[pos] = True

        assert self.env is not None, "You must set the environment before calling _setup_learn()"
        # Vectorize action noise if needed
        if (
            self.action_noise is not None
            and self.env.num_envs > 1
            and not isinstance(self.action_noise, VectorizedActionNoise)
        ):
            self.action_noise = VectorizedActionNoise(self.action_noise, self.env.num_envs)

        return super()._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

    def learn(
        self: SelfDEOffPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        lam = 0,
        delta = 4.,
        limit = 1.
    ) -> SelfDEOffPolicyAlgorithm:
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "You must set the environment before calling learn()"
        assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
                lam = lam,
                delta = delta,
                limit = limit
            )

            if not rollout.continue_training:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)
        """
        raise NotImplementedError()

    def _sample_action_explore(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
        lam = 0.,
        delta = 4.,
        limit = 1.
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
            buffer_action = unscaled_action
            action = buffer_action
        else:
            assert self._last_obs is not None, "self._last_obs was not set"

            if not self.explore_mode:
                unscaled_action, action_tensor, obs, _ = self.predict_explore(self._last_obs, deterministic=False)
                buffer_action = unscaled_action
                action = buffer_action
                return action, buffer_action
            
            _, action_tensor, std_tensor, obs, _ = self.predict_explore(self._last_obs, deterministic=False)
            pre_tanh_mu_T = action_tensor.clone().requires_grad_(True)
            
            def Q_cost_LB(x):
                tanh_mu_T = th.tanh(x)                         # → (B, n_a)
                qs = self.critic(obs, tanh_mu_T)             # → (B, ...)
                q_cost = qs[:, 13:, self.ensemble_size//2:]
                Q_LB = q_cost.mean(-1) + self.beta_policy * q_cost.std(-1)
                return -Q_LB.mean(-1)
            
            def Q_reward_UB(x):
                tanh_mu_T = th.tanh(x)                         # → (B, n_a)
                qs = self.critic(obs, tanh_mu_T)             # → (B, ...)
                q_rew = qs[:, 13:, :self.ensemble_size//2]
                Q_UB = q_rew.mean(-1) + self.beta_policy * q_rew.std(-1)
                return Q_UB.mean(-1)


            def Q_cost_mean(x):
                tanh_mu_T = th.tanh(x)                         # → (B, n_a)
                qs = self.critic(obs, tanh_mu_T)             # → (B, ...)
                q_cost = th.mean(qs[..., self.ensemble_size//2:], 1)
                Q_LB = q_cost.mean(-1)
                return -Q_LB


            J_r_ub = th.autograd.functional.jacobian(Q_reward_UB,
                                            pre_tanh_mu_T,
                                            create_graph=False,
                                            vectorize=True)   # → (B_out=B, B_in=B, n_a)
            J_c_lb = th.autograd.functional.jacobian(Q_cost_LB,
                                            pre_tanh_mu_T,
                                            create_graph=False,
                                            vectorize=True)


            J_c_mean = th.autograd.functional.jacobian(Q_cost_mean,
                                            pre_tanh_mu_T,
                                            create_graph=False,
                                            vectorize=True)


            g_r = J_r_ub.diagonal(dim1=0, dim2=1).transpose(0,1)

            g_c = J_c_lb.diagonal(dim1=0, dim2=1).transpose(0,1)

            g_m = J_c_mean.diagonal(dim1=0, dim2=1).transpose(0,1)

            with th.no_grad():
                qs = self.critic(obs, th.tanh(pre_tanh_mu_T))             # → (B, ...)
                q_current_cost = -th.mean(qs[:, :13, self.ensemble_size//2:], (1, 2))

            lam_in = (lam - th.clamp(10. * (self.cost_limit - q_current_cost), max=lam)).detach()

            '''
            grad = g_r - lam_in*g_c

            sigma_T = std_tensor**2

            denom = th.sum(grad**2 * sigma_T, -1) + 1e-5

            mu_C = self.delta * sigma_T * grad / th.sqrt(denom.unsqueeze(-1))
            mu_E = pre_tanh_mu_T + mu_C
            '''
            var = std_tensor**2
            delta_a = cox_explore(g_r, g_c, g_m, q_current_cost, var, lam_in, 0., self.delta)

            mu_E = pre_tanh_mu_T + delta_a
            
            gaussian_noise = th.randn_like(mu_E) * std_tensor
            explore_actions = th.tanh(mu_E + gaussian_noise)
            buffer_action = explore_actions.detach().cpu().numpy().reshape((-1, *self.action_space.shape))
            action = buffer_action
        return action, buffer_action

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_cost_mean", safe_mean([ep_info["cost"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_ret_mean", safe_mean([ep_info["return"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def _on_step(self) -> None:
        pass

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        reward_2d: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_, reward_2d_ = self._last_obs, new_obs, reward, reward_2d 

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,  # type: ignore[arg-type]
            next_obs,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            reward_2d_,
            dones,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
        lam = 1.,
        delta = 4.,
        limit = 1.
    ) -> RolloutReturn:
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action_explore(learning_starts, action_noise, env.num_envs, lam, delta, limit)
            
            # Rescale and perform action
            new_obs, rewards_nd, dones, infos = env.step(actions)
            # rewards_nd *= 3
            rewards = np.sum(rewards_nd, -1)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, rewards_nd, dones, infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def predict_explore(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        self.policy.set_training_mode(False)
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            if not self.explore_mode:
                actions_raw = self.actor(obs_tensor, deterministic)
            else:
                actions_raw, std_raw = self.actor.raw_forward(obs_tensor)

        # Convert to numpy, and reshape to the original action shape
        if not self.explore_mode:
            actions = actions_raw.cpu().numpy().reshape((-1, *self.action_space.shape))  

            # Remove batch dimension if needed
            if not vectorized_env:
                assert isinstance(actions, np.ndarray)
                actions = actions.squeeze(axis=0)

            return actions, actions_raw, obs_tensor, state  # type: ignore[return-value]
        
        actions = actions_raw.cpu().numpy().reshape((-1, *self.action_space.shape)) 

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)

        return actions, actions_raw, std_raw, obs_tensor, state  # type: ignore[return-value]

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        self.policy.set_training_mode(False)
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions_raw = self.actor(obs_tensor, deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions_raw.cpu().numpy().reshape((-1, *self.action_space.shape))  

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)

        return actions, state  # type: ignore[return-value]

    def obs_to_tensor(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]):
        vectorized_env = False
        if isinstance(observation, dict):
            assert isinstance(
                self.observation_space, spaces.Dict
            ), f"The observation provided is a dict but the obs space is {self.observation_space}"
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                if is_image_space(obs_space):
                    obs_ = maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
                # Add batch dimension if needed
                observation[key] = obs_.reshape((-1, *self.observation_space[key].shape))  # type: ignore[misc]

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            vectorized_env = is_vectorized_observation(observation, self.observation_space)
            # Add batch dimension if needed
            observation = observation.reshape((-1, *self.observation_space.shape))  # type: ignore[misc]

        obs_tensor = obs_as_tensor(observation, self.device)
        return obs_tensor, vectorized_env


class DETD3(DEOffPolicyAlgorithm):
    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        ensemble_size,
        tail_reward = 5,
        tail_cost = 5,
        convex = 10.,
        cost_limit = 2.5,
        topk = 6,
        beta_policy: float = 3.,
        delta: float = 3.,
        model_type: str = 'baseline',
        learning_method: str = 'saclag', # {"mo", "cop", "gaussian"}
        explore_mode = False,
        share_feature: bool = False,
        learning_rate: Union[float, Schedule] = 3e-4,
        multiplier_learning_rate = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        warmup: int = 102400,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = NormalActionNoise(0, 0.1),
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            beta_policy,
            delta,
            model_type,
            explore_mode,
            cost_limit,
            topk,
            learning_method,
            share_feature,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        self.ensemble_size = ensemble_size
        self.beta_policy = beta_policy
        self.delta = delta
        self.model_type = model_type
        self.explore_mode = explore_mode

        self.multiplier_learning_rate = multiplier_learning_rate

        self.tail_reward = tail_reward
        self.tail_cost = tail_cost
        self.convex = convex
        self.warmup = warmup

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

        self.target_entropy = float(-2)
        self.log_ent_coef = th.zeros(1, device=self.device).requires_grad_(True)
        self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))

        self.lagrangian_multiplier = th.ones(1, device=self.device).requires_grad_(True)
        self.lagrange_optimizer = th.optim.Adam([self.lagrangian_multiplier], lr=self.multiplier_learning_rate)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer, self.ent_coef_optimizer])
        
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        for gradient_step in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef = th.exp(self.log_ent_coef.detach())

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = self.critic_target(replay_data.next_observations, next_actions)
                target_q_reward, target_q_cost = self.compute_target_q(replay_data, next_q_values, next_log_prob, ent_coef)

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = self.compute_critic_loss(current_q_values, target_q_reward, target_q_cost)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)

            if (gradient_step + 1) % self.policy_delay == 0:
                qs = self.critic(replay_data.observations, actions_pi)
                qs_current = self.critic(replay_data.observations, replay_data.actions)
                actor_loss, cost_estimate = self.compute_actor_loss(qs, qs_current, log_prob, ent_coef)
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                ent_coef_loss = -(th.exp(self.log_ent_coef) * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())

                ent_coefs.append(ent_coef.item())

                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                
                if self._n_updates >= self.warmup:
                    lag_loss = th.exp(self.lagrangian_multiplier) * (self.cost_limit - cost_estimate)
                    self.lagrange_optimizer.zero_grad()
                    lag_loss.backward()
                    self.lagrange_optimizer.step()
                    self.lagrangian_multiplier.data.clamp_(min=-7.)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/multiplier", th.exp(self.lagrangian_multiplier).item())
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfDETD3,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "TD3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDETD3:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
            lam = th.exp(self.lagrangian_multiplier).item(),
            delta = self.delta,
            limit = 1.
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []

    def compute_actor_loss(self, q_action, qs_current, log_prob, ent_coef):
        lam = th.exp(self.lagrangian_multiplier).item()
        if self.learning_method == 'saclag':
            q_reward = th.amin(q_action[:,0,:2], -1)
            q_cost = -th.amin(q_action[:,0,2:], -1)
            if self._n_updates < self.warmup:
                actor_loss = ent_coef * log_prob - q_reward + q_cost
                return actor_loss.mean(), q_cost.mean().detach()

            loss_r = ent_coef * log_prob - q_reward
            loss_c = lam*q_cost
            actor_loss = (loss_r + loss_c).mean()/(1 + lam)
            return actor_loss, q_cost.mean().detach()

        if self.learning_method == 'cal':
            q_reward = th.amin(q_action[:,0,:2], -1)
            q_cost = -q_action[:,0,2:]
            q_current = -qs_current[:,0,2:]

            actor_QC = q_cost.mean(-1) + 0.5 * q_cost.std(-1)
            
            current_QC = q_current.mean(-1) + 0.5 * q_current.std(-1)

            rect = th.clamp(self.convex * (self.cost_limit - current_QC.mean()), max=lam)

            actor_loss = ent_coef * log_prob - q_reward + (lam - rect).detach()*actor_QC

            return actor_loss.mean(), current_QC.mean().detach()
        
        if self.learning_method == 'tqc':
            q_reward = th.mean(q_action[..., :self.ensemble_size//2], (1,2))
            q_cost = -th.mean(q_action[..., self.ensemble_size//2:], (1,2))
            if self._n_updates < self.warmup:
                actor_loss = ent_coef * log_prob - q_reward + q_cost
                return actor_loss.mean(), q_cost.mean().detach()
            q_current = -th.mean(qs_current[..., self.ensemble_size//2:], (1,2))

            rect = th.clamp(self.convex * (self.cost_limit - q_current.mean()), max=lam)

            actor_loss = ent_coef * log_prob - q_reward + (lam - rect).detach()*q_cost

            return actor_loss.mean(), q_current.mean().detach()

        if self.learning_method in ['orac', 'coxq']:
            q_reward = th.mean(q_action[..., :self.ensemble_size//2], (1,2))
            q_cost = -th.mean(q_action[:, :13, self.ensemble_size//2:], (1,2))
            q_current = -th.mean(qs_current[:, :13, self.ensemble_size//2:], (1,2))

            rect = th.clamp(self.convex * (self.cost_limit - q_current.mean()), max=lam)

            actor_loss = ent_coef * log_prob - q_reward + (lam - rect).detach()*q_cost

            return actor_loss.mean(), q_current.mean().detach()

    def compute_target_q(self, replay_data, next_q, next_log_prob, ent_coef):
        if self.learning_method == 'saclag':
            next_q_reward = th.amin(next_q[:,0,:2], -1) - ent_coef * next_log_prob.squeeze()
            target_q_reward = replay_data.rewards_nd[...,1] + (1 - replay_data.dones.squeeze()) * self.gamma * next_q_reward

            next_q_cost = th.amin(next_q[:,0,2:], -1)
            target_q_cost = replay_data.rewards_nd[...,0] + (1 - replay_data.dones.squeeze()) * self.gamma * next_q_cost

        if self.learning_method == 'cal':
            next_q_reward = th.amin(next_q[:,0,:2], -1) - ent_coef * next_log_prob.squeeze()
            target_q_reward = replay_data.rewards_nd[...,1] + (1 - replay_data.dones.squeeze()) * self.gamma * next_q_reward

            next_q_cost = th.mean(next_q[:,0,2:], -1) - 0.5 * th.std(next_q[:,0,2:], -1)
            target_q_cost = replay_data.rewards_nd[...,0] + (1 - replay_data.dones.squeeze()) * self.gamma * next_q_cost


        if self.learning_method in ['orac']:
            next_quantiles_reward, _ = th.sort(next_q[...,:self.ensemble_size//2].reshape(next_q.shape[0], -1), -1)
            # target_quantiles_reward = next_quantiles_reward[:,:-self.tail_reward*self.ensemble_size] - ent_coef*next_log_prob[:,None]
            target_quantiles_reward = next_quantiles_reward - ent_coef*next_log_prob[:,None]
            target_q_reward = replay_data.rewards_nd[...,1:] + (1 - replay_data.dones) * self.gamma * target_quantiles_reward

            next_quantiles_cost, _ = th.sort(next_q[...,self.ensemble_size//2:].reshape(next_q.shape[0], -1), -1)
            target_quantiles_cost = next_quantiles_cost # [:,:-self.tail_cost*self.ensemble_size]
            target_q_cost = replay_data.rewards_nd[...,:1] + (1 - replay_data.dones) * self.gamma * target_quantiles_cost

        if self.learning_method == 'coxq':
            next_quantiles_reward, _ = th.sort(next_q[...,:self.ensemble_size//2].reshape(next_q.shape[0], -1), -1)
            # target_quantiles_reward = next_quantiles_reward[:,:-self.tail_reward*self.ensemble_size//2] - ent_coef*next_log_prob[:,None]
            target_quantiles_reward = next_quantiles_reward - ent_coef*next_log_prob[:,None]
            target_q_reward = replay_data.rewards_nd[...,1:] + (1 - replay_data.dones) * self.gamma * target_quantiles_reward

            next_quantiles_cost, _ = th.sort(next_q[...,self.ensemble_size//2:].reshape(next_q.shape[0], -1), -1)
            target_quantiles_cost = next_quantiles_cost # [:,:-self.tail_cost*self.ensemble_size//2]
            target_q_cost = replay_data.rewards_nd[...,:1] + (1 - replay_data.dones) * self.gamma * target_quantiles_cost.clamp_(-10., 0.0)


        return target_q_reward, target_q_cost

    def compute_critic_loss(self, current_q, target_q_reward, target_q_cost):
        if self.model_type == 'baseline':
            current_q_reward = current_q[:, 0, :2]
            current_q_cost = current_q[:, 0, 2:]
            q_loss = th.mean((current_q_reward - 
                              target_q_reward[:, None])**2) + th.mean((current_q_cost - 
                                                                       target_q_cost[:, None])**2)

        if self.model_type in ['mo', 'cop']:
            lam = th.exp(self.lagrangian_multiplier).item()
            current_q_reward = current_q[:, 0, :2]
            current_q_cost = current_q[:, 0, 2:]
            q_loss = th.mean((current_q_reward - 
                              target_q_reward[:, None])**2) + th.mean((current_q_cost - 
                                                                       target_q_cost[:, None])**2)

        if self.model_type == 'quantile':
            current_q_reward = current_q[..., :self.ensemble_size//2]
            current_q_cost = current_q[..., self.ensemble_size//2:]
            q_loss = self.quantile_huber_loss(current_q_reward, target_q_reward) + self.quantile_huber_loss(current_q_cost, target_q_cost)

        return q_loss

    def quantile_huber_loss(self, current_q, target_q):
        current_quantiles = current_q.transpose(-2,-1) #(B, nc, nq)
        n_quantiles = current_quantiles.shape[-1]

        cum_prob = (th.arange(n_quantiles, dtype=current_quantiles.dtype, device=current_quantiles.device) + 0.5) / n_quantiles
        #print(target_q.shape, current_quantiles.shape)
        pairwise_delta = target_q[:, None, None, :] - current_quantiles[...,None] #(B, nc, nq, nqt)

        abs_pairwise_delta = th.abs(pairwise_delta)
        huber_loss = th.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5)

        indicator = th.where(pairwise_delta < 0, 1.0, 0.0)
        loss = th.abs(cum_prob[...,None] - indicator) * huber_loss
        return th.mean(loss)


