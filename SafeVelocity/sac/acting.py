# Copyright 2024 The Brax Authors.
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

"""Brax training acting functions."""

import time
from typing import Callable, Sequence, Tuple, Union

from brax import envs
from sac.types import Metrics
from sac.types import Policy
from sac.types import PolicyParams
from sac.types import PRNGKey
from sac.types import Transition
import jax
import jax.numpy as jnp
import numpy as np

State = envs.State
Env = envs.Env

def get_multi_rewards_backup(env_name, nstate):
  if env_name in ['hopper', 'walker2d']:
    multi_reward = jnp.array([nstate.metrics['reward_forward'], 
                              0.01*nstate.metrics['reward']
                              ])
  elif env_name in ['ant']:
    multi_reward = jnp.array([nstate.metrics['velocity_total'], 
                              nstate.metrics['reward']
                              ])
  else:
    multi_reward = jnp.array([nstate.metrics['forward_reward'], 
                              nstate.metrics['reward']
                              ])

  return multi_reward


def get_multi_rewards(env_name, nstate):
  multi_reward = jnp.array([-nstate.metrics['cost'], 
                            nstate.metrics['reward']
                            ]
                          )

  return multi_reward



def actor_step( 
    env: Env,
    env_state: State,
    env_name: str,
    policy: Policy,
    key: PRNGKey,
    prefill: bool = False,
    extra_fields: Sequence[str] = ()
) -> Tuple[State, Transition]:
  """Collect data."""
  key_p, key_prefill = jax.random.split(key, 2)
  actions, policy_extras = policy(env_state.obs, key_p)
  if prefill:
    actions = jax.random.uniform(key_prefill, shape=actions.shape, minval=-1.0, maxval=1.0)

  new_state = env.step(env_state, actions)
  nstate = new_state
  
  multi_reward = get_multi_rewards(env_name, nstate)

  multi_reward = jnp.transpose(multi_reward)
  
  state_extras = {x: nstate.info[x] for x in extra_fields}
  return new_state, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=env_state.obs,
      action=actions,
      reward=nstate.reward,
      multi_reward = multi_reward,
      discount=1 - nstate.done,
      next_observation=nstate.obs,
      extras={
          'state_extras': state_extras,
          'policy_extras': policy_extras,
          #'metrics': info
      })


def generate_unroll(
    env,
    env_state,
    env_name, 
    policy: Policy,
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = ()
) -> Tuple[State, Transition]:
  """Collect trajectories of given unroll_length."""

  @jax.jit
  def f(carry, unused_t):
    state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    nstate, transition = actor_step(
        env, state, env_name, policy, current_key, extra_fields=extra_fields)
    return (nstate, next_key), transition

  (final_state, _), data = jax.lax.scan(
      f, (env_state, key), (), length=unroll_length)
  return final_state, data


def _agg_fn(metric, fn, to_aggregate, to_normalize, episode_lengths):
  if not to_aggregate:
    return metric
  if to_normalize:
    return fn(metric / episode_lengths)
  return fn(metric)


# TODO: Consider moving this to its own file.
class Evaluator:
  """Class to run evaluations."""

  def __init__(self, eval_env: envs.Env, env_name,
               eval_policy_fn: Callable[[PolicyParams], Policy],
               num_eval_envs: int,
               episode_length: int, action_repeat: int, key: PRNGKey):

    self._key = key
    self._eval_walltime = 0.

    eval_env = envs.training.EvalWrapper(eval_env)
    self._eval_state_to_donate = jax.jit(eval_env.reset)(
        jax.random.split(key, num_eval_envs)
    )

    def generate_eval_unroll(policy_params: PolicyParams,
                             key: PRNGKey) -> State:
      reset_keys = jax.random.split(key, num_eval_envs)
      eval_first_state = eval_env.reset(reset_keys)
      return generate_unroll(
          eval_env,
          eval_first_state,
          env_name,
          eval_policy_fn(policy_params),
          key,
          unroll_length=episode_length // action_repeat)[0]

    self._generate_eval_unroll = jax.jit(
        generate_eval_unroll, donate_argnums=(0,), keep_unused=True
    )
    self._steps_per_unroll = episode_length * num_eval_envs

  def run_evaluation(self,
                     policy_params: PolicyParams,
                     training_metrics: Metrics,
                     aggregate_episodes: bool = True) -> Metrics:
    """Run one epoch of evaluation."""
    self._key, unroll_key = jax.random.split(self._key)

    t = time.time()
    eval_state = self._generate_eval_unroll(policy_params, unroll_key)
    self._eval_state_to_donate = eval_state

    eval_metrics = eval_state.info['eval_metrics']
    eval_metrics.active_episodes.block_until_ready()
    epoch_eval_time = time.time() - t
    episode_lengths = np.maximum(eval_metrics.episode_steps, 1.0).astype(float)

    metrics = {}
    for fn in [np.mean, np.std]:
      suffix = '_std' if fn == np.std else ''
      for name, value in eval_metrics.episode_metrics.items():
        metrics[f'eval/episode_{name}{suffix}'] = _agg_fn(
            value,
            fn,
            aggregate_episodes,
            name.endswith('per_step'),
            episode_lengths,
        )
    
    metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
    metrics['eval/std_episode_length'] = np.std(eval_metrics.episode_steps)
    metrics['eval/epoch_eval_time'] = epoch_eval_time
    metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
    self._eval_walltime = self._eval_walltime + epoch_eval_time
    metrics = {
        'eval/walltime': self._eval_walltime,
        **training_metrics,
        **metrics,
    }

    return metrics  # pytype: disable=bad-return-type  # jax-ndarray

