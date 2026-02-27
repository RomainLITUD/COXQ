# Copyright 2025 The Brax Authors.
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

"""Network definitions."""

import dataclasses
import functools
from typing import Any, Callable, Literal, Mapping, Sequence, Tuple
import warnings

from brax.training import types
from brax.training.acme import running_statistics
from brax.training.spectral_norm import SNDense
from flax import linen
from flax import linen as nn
import jax
import jax.numpy as jnp


ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


ACTIVATION = {
    'celu': nn.activation.celu,
    'compact': nn.activation.compact,
    'elu': nn.activation.elu,
    'gelu': nn.activation.gelu,
    'glu': nn.activation.glu,
    'hard_sigmoid': nn.activation.hard_sigmoid,
    'hard_silu': nn.activation.hard_silu,
    'hard_swish': nn.activation.hard_swish,
    'hard_tanh': nn.activation.hard_tanh,
    'leaky_relu': nn.activation.leaky_relu,
    'linear': lambda x: x,
    'log_sigmoid': nn.activation.log_sigmoid,
    'log_softmax': nn.activation.log_softmax,
    'logsumexp': nn.activation.logsumexp,
    'mish': jax.nn.mish,
    'normalize': nn.activation.normalize,
    'one_hot': nn.activation.one_hot,
    'relu': nn.activation.relu,
    'relu6': nn.activation.relu6,
    'selu': nn.activation.selu,
    'sigmoid': nn.activation.sigmoid,
    'silu': nn.activation.silu,
    'soft_sign': nn.activation.soft_sign,
    'softmax': nn.activation.softmax,
    'softplus': nn.activation.softplus,
    'standardize': nn.activation.standardize,
    'swish': nn.activation.swish,
    'tanh': nn.activation.tanh,
}
KERNEL_INITIALIZER = {
    'constant': jax.nn.initializers.constant,
    'delta_orthogonal': jax.nn.initializers.delta_orthogonal,
    'glorot_normal': jax.nn.initializers.glorot_normal,
    'glorot_uniform': jax.nn.initializers.glorot_uniform,
    'he_normal': jax.nn.initializers.he_normal,
    'he_uniform': jax.nn.initializers.he_uniform,
    'kaiming_normal': jax.nn.initializers.kaiming_normal,
    'kaiming_uniform': jax.nn.initializers.kaiming_uniform,
    'lecun_normal': jax.nn.initializers.lecun_normal,
    'lecun_uniform': jax.nn.initializers.lecun_uniform,
    'normal': jax.nn.initializers.normal,
    'ones': jax.nn.initializers.ones,
    'orthogonal': jax.nn.initializers.orthogonal,
    'truncated_normal': jax.nn.initializers.truncated_normal,
    'uniform': jax.nn.initializers.uniform,
    'variance_scaling': jax.nn.initializers.variance_scaling,
    'xavier_normal': jax.nn.initializers.xavier_normal,
    'xavier_uniform': jax.nn.initializers.xavier_uniform,
    'zeros': jax.nn.initializers.zeros,
}


@dataclasses.dataclass
class FeedForwardNetwork:
  init: Callable[..., Any]
  apply: Callable[..., Any]



class MLP(linen.Module):
  """MLP module."""

  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True
  layer_norm: bool = False

  @linen.compact
  def __call__(self, data: jnp.ndarray):
    hidden = data
    for i, hidden_size in enumerate(self.layer_sizes):
      hidden = linen.Dense(
          hidden_size,
          name=f'hidden_{i}',
          kernel_init=self.kernel_init,
          use_bias=self.bias,
      )(hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = self.activation(hidden)
        if self.layer_norm:
          hidden = linen.LayerNorm()(hidden)
    return hidden


def _get_obs_state_size(obs_size: types.ObservationSize, obs_key: str) -> int:
  obs_size = obs_size[obs_key] if isinstance(obs_size, Mapping) else obs_size
  return jax.tree_util.tree_flatten(obs_size)[0][-1]


class Param(linen.Module):
  """Scalar parameter module."""

  init_value: float = 0.0
  size: int = 1

  @linen.compact
  def __call__(self):
    return self.param(
        'value', init_fn=lambda keys: jnp.full((self.size,), self.init_value)
    )


class LogParam(linen.Module):
  """Scalar parameter module with log scale."""

  init_value: float = 1.0
  size: int = 1

  @linen.compact
  def __call__(self):
    log_value = self.param(
        'log_value',
        init_fn=lambda key: jnp.full((self.size,), jnp.log(self.init_value)),
    )
    return jnp.exp(log_value)


class PolicyModuleWithStd(linen.Module):
  """Policy module with learnable mean and standard deviation."""

  param_size: int
  hidden_layer_sizes: Sequence[int]
  activation: ActivationFn
  kernel_init: jax.nn.initializers.Initializer
  layer_norm: bool
  noise_std_type: Literal['scalar', 'log']
  init_noise_std: float
  state_dependent_std: bool = False

  @linen.compact
  def __call__(self, obs):
    if self.noise_std_type not in ['scalar', 'log']:
      raise ValueError(
          f'Unsupported noise std type: {self.noise_std_type}. Must be one of'
          ' "scalar" or "log".'
      )

    outputs = MLP(
        layer_sizes=list(self.hidden_layer_sizes),
        activation=self.activation,
        kernel_init=self.kernel_init,
        layer_norm=self.layer_norm,
        activate_final=True,
    )(obs)

    mean_params = linen.Dense(
        self.param_size,
        kernel_init=self.kernel_init,
    )(outputs)

    if self.state_dependent_std:
      log_std_output = linen.Dense(
          self.param_size, kernel_init=self.kernel_init
      )(outputs)
      if self.noise_std_type == 'log':
        std_params = jnp.exp(log_std_output)
      else:
        std_params = log_std_output
    else:
      if self.noise_std_type == 'scalar':
        std_module = Param(
            self.init_noise_std, size=self.param_size, name='std_param'
        )
      else:
        std_module = LogParam(
            self.init_noise_std, size=self.param_size, name='std_logparam'
        )
      std_params = std_module()

    return mean_params, jnp.broadcast_to(std_params, mean_params.shape)


def make_policy_network(
    param_size: int,
    obs_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False,
    obs_key: str = 'state',
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type: Literal['scalar', 'log'] = 'scalar',
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
) -> FeedForwardNetwork:
  """Creates a policy network."""
  if distribution_type == 'tanh_normal':
    policy_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
    )
  elif distribution_type == 'normal':
    policy_module = PolicyModuleWithStd(
        param_size=param_size,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
        noise_std_type=noise_std_type,
        init_noise_std=init_noise_std,
        state_dependent_std=state_dependent_std,
    )
  else:
    raise ValueError(
        f'Unsupported distribution type: {distribution_type}. Must be one'
        ' of "normal" or "tanh_normal".'
    )

  def apply(processor_params, policy_params, obs):
    if isinstance(obs, Mapping):
      obs = preprocess_observations_fn(
          obs[obs_key], normalizer_select(processor_params, obs_key)
      )
    else:
      obs = preprocess_observations_fn(obs, processor_params)
    return policy_module.apply(policy_params, obs)

  obs_size = _get_obs_state_size(obs_size, obs_key)
  dummy_obs = jnp.zeros((1, obs_size))

  def init(key):
    policy_module_params = policy_module.init(key, dummy_obs)
    return policy_module_params

  return FeedForwardNetwork(init=init, apply=apply)



class QModule(linen.Module):
  hidden_layer_sizes: Sequence[int]
  num_obj: int
  activation: ActivationFn
  mode: str
  random_prior: bool

  @linen.compact
  def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
    hidden = jnp.concatenate([obs, actions], axis=-1)
    if self.mode == 'mo':
      q = MLP(
          layer_sizes=list(self.hidden_layer_sizes) + [1],
          activation=self.activation,
          kernel_init=jax.nn.initializers.lecun_uniform())(
              hidden)

      return q.squeeze(-1)
    
    if self.mode == 'qr':
      q = MLP(
          layer_sizes=list(self.hidden_layer_sizes) + [25],
          activation=self.activation,
          kernel_init=jax.nn.initializers.lecun_uniform())(
              hidden) 
      
      # q_prior = MLP(
      #     layer_sizes=list(self.hidden_layer_sizes) + [1],
      #     activation=self.activation,
      #     kernel_init=jax.nn.initializers.lecun_uniform())(
      #         hidden) 

      return q # + jax.lax.stop_gradient(q_prior)
    
    if self.mode == 'orac':
      q1 = MLP(
          layer_sizes=list(self.hidden_layer_sizes) + [1],
          activation=self.activation,
          kernel_init=jax.nn.initializers.lecun_uniform())(
              hidden)
      
      q2 = MLP(
          layer_sizes=list(self.hidden_layer_sizes) + [25],
          activation=self.activation,
          kernel_init=jax.nn.initializers.lecun_uniform())(
              hidden) 

      return jnp.concatenate([q2, q1], -1)
    


class QEnsemble(nn.Module):
  hidden_layer_sizes: Sequence[int]
  num_obj: int
  activation: ActivationFn
  mode: str
  n_critics: int
  random_prior: bool = False

  @nn.compact
  def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
    #hidden = jnp.concatenate([obs, actions], axis=-1)
    if self.mode != 'orac':
      power_nb = 1 << (self.n_critics*2 - 1).bit_length()
    else:
      power_nb = 1 << (self.n_critics - 1).bit_length()
    CriticVmap = nn.vmap(
        QModule,
        variable_axes={'params': 0},   # Batch the parameters over the first axis
        split_rngs={'params': True},   # Split the random number generators
        in_axes=None,                  # Input x is shared among critics
        out_axes=-1,                    # Output values stacked over the last axis
        axis_size=power_nb,  # Size of the ensemble
    )

    # Instantiate the vectorized Critic module
    critics = CriticVmap(self.hidden_layer_sizes, self.num_obj, self.activation, self.mode, self.random_prior)

    # Apply the critics to the input x
    values = critics(obs, actions)
    if self.mode != 'orac':
      output = jnp.reshape(values, values.shape[:-1] + (2, -1))
    else:
      output = values

    return output[...,:self.n_critics]


def make_q_ensemble(
    obs_size: int,
    action_size: int,
    num_obj: int,
    mode: str,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_critics: int = 2,
    random_prior: bool = False) -> FeedForwardNetwork:
  """Creates a value network."""

  q_module = QEnsemble(hidden_layer_sizes, num_obj, activation, mode, n_critics, random_prior)

  def apply(processor_params, q_params, obs, actions):
    obs = preprocess_observations_fn(obs, processor_params)
    return q_module.apply(q_params, obs, actions)

  dummy_obs = jnp.zeros((1, obs_size))
  dummy_action = jnp.zeros((1, action_size))
  return FeedForwardNetwork(
      init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply)



def normalizer_select(
    processor_params: running_statistics.RunningStatisticsState, obs_key: str
) -> running_statistics.RunningStatisticsState:
  return running_statistics.RunningStatisticsState(
      count=processor_params.count,
      mean=processor_params.mean[obs_key],
      summed_variance=processor_params.summed_variance[obs_key],
      std=processor_params.std[obs_key],
  )


