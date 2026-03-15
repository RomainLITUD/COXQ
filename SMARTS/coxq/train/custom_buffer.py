from typing import Dict, Generator, Tuple, Union, NamedTuple, Optional, List, Any

import numpy as np
import torch as th
from gymnasium import spaces
import warnings

#from stable_baselines3.common.type_aliases import DictRolloutBufferSamples

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.buffers import ReplayBuffer

from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
)

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None
    print("CANNOT CHECK MEMORY USE!!!!!!!!!!!!!!!!")

TensorDict = Dict[str, th.Tensor]

class DictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    preference: th.Tensor
    old_log_prob: th.Tensor
    advantages_mean: th.Tensor
    rewards: th.Tensor
    returns: th.Tensor

class EnsembleDictRolloutBuffer(RolloutBuffer):

    observation_space: spaces.Dict
    obs_shape: Dict[str, Tuple[int, ...]]  # type: ignore[assignment]
    observations: Dict[str, np.ndarray]  # type: ignore[assignment]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        ensemble_size: int = 5,
        reward_dim: int= 3,
        cov_dim: int= 2,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.reward_dim = reward_dim
        self.cov_dim = cov_dim

        self.cov_par = int((cov_dim)*(cov_dim+1)/2)

        self.pos_re = np.zeros(n_envs, dtype=int)

        self.ensemble_size = ensemble_size

        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = np.zeros((self.buffer_size, self.n_envs, *obs_input_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs, self.reward_dim, self.ensemble_size), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs, self.reward_dim, self.ensemble_size), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.values = np.zeros((self.buffer_size, self.n_envs, self.reward_dim + self.cov_par+1, self.ensemble_size), dtype=np.float32)

        self.preference = np.zeros((self.buffer_size, self.n_envs, self.cov_dim), dtype=np.float32)

        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.advantages_mean = np.zeros((self.buffer_size, self.n_envs, self.reward_dim, self.ensemble_size), dtype=np.float32)

        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def add(  # type: ignore[override]
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        preference: th.Tensor, 
        log_prob: th.Tensor,
    ) -> None:

        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = np.array(obs[key])
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)

        self.rewards[self.pos] = np.array(reward)

        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy()


        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.preference[self.pos] = preference.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(  # type: ignore[override]
        self,
        batch_size = None,
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        # Prepare the data
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions", "values", "preference", "rewards",
                             "log_probs", "advantages_mean",
                             "returns"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(  
        self,
        batch_inds: np.ndarray,
        env = None,
    ) -> DictRolloutBufferSamples:
        #print(batch_inds)
        #b_indices, e_indices = np.divmod(batch_inds, self.n_envs)
        return DictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds]),
            preference=self.to_torch(self.preference[batch_inds]),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages_mean=self.to_torch(self.advantages_mean[batch_inds]),
            rewards=self.to_torch(self.rewards[batch_inds]),
            returns=self.to_torch(self.returns[batch_inds]),
        )
    
    def compute_returns_and_advantages(self, 
                                 last_values: th.Tensor,  
                                 dones: np.ndarray) -> None:
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values[...,:3,:]
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1][...,:3,:]

            next_non_terminal =np.tile(next_non_terminal.reshape(-1, 1, 1), (1, self.reward_dim, self.ensemble_size))
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step][...,:3,:]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal  * last_gae_lam
            # print(last_gae_lam.shape)
            self.advantages_mean[step] = last_gae_lam

        self.returns = self.advantages_mean + self.values[...,:3,:]

class DEDictReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    next_observations: TensorDict
    dones: th.Tensor
    rewards: th.Tensor
    rewards_nd: th.Tensor

class DEDictReplayBuffer(ReplayBuffer):
    observation_space: spaces.Dict
    obs_shape: Dict[str, Tuple[int, ...]]  # type: ignore[assignment]
    observations: Dict[str, np.ndarray]  # type: ignore[assignment]
    next_observations: Dict[str, np.ndarray]  # type: ignore[assignment]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert not optimize_memory_usage, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.rewards_nd = np.zeros((self.buffer_size, self.n_envs, 2), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage: float = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if not optimize_memory_usage:
                next_obs_nbytes = 0
                for _, obs in self.observations.items():
                    next_obs_nbytes += obs.nbytes
                total_memory_usage += next_obs_nbytes

            print("total memory usage is: ", total_memory_usage/1e9)
            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(  # type: ignore[override]
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        reward_2d: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = np.array(obs[key])

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][self.pos] = np.array(next_obs[key])

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.rewards_nd[self.pos] = np.array(reward_2d)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(  # type: ignore[override]
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
    ) -> DEDictReplayBufferSamples:
        return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DEDictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}, env)
        next_obs_ = self._normalize_obs(
            {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env
        )

        assert isinstance(obs_, dict)
        assert isinstance(next_obs_, dict)
        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return DEDictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
            rewards_nd=self.to_torch(self.rewards_nd[batch_inds, env_indices].reshape(-1, 2), env),
        )
    
    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype