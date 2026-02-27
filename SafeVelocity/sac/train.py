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

"""Soft Actor-Critic training.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

import functools
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union
import gc

from absl import logging
from brax import base
from brax import envs
from sac import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import replay_buffers
from sac import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from sac import checkpoint
from sac import losses as sac_losses
from sac import networks as sac_networks
from sac.types import Params
from sac.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import optax

Metrics = types.Metrics
Transition = types.Transition
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState = Any

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  policy_optimizer_state: optax.OptState
  policy_params: Params
  q_optimizer_state: optax.OptState
  q_params: Params
  target_q_params: Params
  gradient_steps: jnp.ndarray
  env_steps: jnp.ndarray
  cost_fly: jnp.ndarray
  alpha_optimizer_state: optax.OptState
  alpha_params: Params
  multiplier_optimizer_state: optax.OptState
  multiplier: Params
  delta_optimizer_state: optax.OptState
  delta: Params
  normalizer_params: running_statistics.RunningStatisticsState


def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)


def _init_training_state(
    key: PRNGKey, 
    obs_size: int, 
    local_devices_to_use: int,
    num_envs: int,
    sac_network: sac_networks.SACNetworks,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    multiplier_optimizer:optax.GradientTransformation,
    delta_optimizer:optax.GradientTransformation) -> TrainingState:
  """Inits the training state and replicates it over devices."""
  key_policy, key_q = jax.random.split(key)
  log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
  alpha_optimizer_state = alpha_optimizer.init(log_alpha)

  policy_params = sac_network.policy_network.init(key_policy)
  policy_optimizer_state = policy_optimizer.init(policy_params)
  q_params = sac_network.q_network.init(key_q)
  q_optimizer_state = q_optimizer.init(q_params)

  multiplier=jnp.asarray(0., dtype=jnp.float32)
  multiplier_optimizer_state = multiplier_optimizer.init(multiplier)

  delta_=jnp.asarray(1., dtype=jnp.float32)
  delta_optimizer_state = delta_optimizer.init(delta_)

  normalizer_params = running_statistics.init_state(
      specs.Array((obs_size,), jnp.dtype('float32')))
  

  training_state = TrainingState(
      policy_optimizer_state=policy_optimizer_state,
      policy_params=policy_params,
      q_optimizer_state=q_optimizer_state,
      q_params=q_params,
      target_q_params=q_params,
      gradient_steps=jnp.zeros(()),
      env_steps=jnp.zeros(()),
      cost_fly=jnp.zeros(2560//num_envs),
      alpha_optimizer_state=alpha_optimizer_state,
      alpha_params=log_alpha,
      multiplier=multiplier,
      multiplier_optimizer_state = multiplier_optimizer_state,
      delta=delta_,
      delta_optimizer_state = delta_optimizer_state,
      normalizer_params=normalizer_params)
  return jax.device_put_replicated(training_state,
                                   jax.local_devices()[:local_devices_to_use])


def train(seed,
    environment: envs.Env,
        name: str = 'humanoid',
    mode: str = 'baseline',
    num_timesteps: int = 1000000,
    episode_length: int = 1000,
    beta_reward: float = 0.,
    beta_cost: float = 0.,
    num_obj: int =1,
    exploration_strategy = False,
    exploration_method = 'qc',
    delta: float = 6.86,
    method: str = 'redq',
    tail_r: int = 5,
    tail_c: int = 5,
    cost_limit: float = 100.,
    budget: float = 2.5,
    budget_st: float = 2.5,
    topk: int = 5,
    convex_coeff: float = 10., 
    ensemble_size: int = 2,
    action_repeat: int = 1,
    actor_delay: int = 1,
    num_envs: int = 64,
    num_eval_envs: int = 64,
    learning_rate: float = 3e-4,
    q_learning_rate: float = 3e-4,
    multiplier_learning_rate: float = 3e-4,
    discounting: float = 0.99,
    batch_size: int = 256,
    num_evals: int = 51,
    normalize_observations: bool = True,
    max_devices_per_host: Optional[int] = 1,
    tau: float = 0.005,
    network_size = (256, 256),
    min_replay_size: int = 10000,
    max_replay_size: Optional[int] = 1000000,
    grad_updates_per_step: int = 64,
    deterministic_eval: bool = True,
    network_factory: types.NetworkFactory[
        sac_networks.SACNetworks
    ] = sac_networks.make_sac_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    checkpoint_logdir: Optional[str] = None,
    eval_env = None,
):
  xdata, ydata = [], []
  """SAC training."""
  process_id = jax.process_index()
  local_devices_to_use = jax.local_device_count()
  if max_devices_per_host is not None:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  device_count = local_devices_to_use * jax.process_count()
  logging.info(
      'local_device_count: %s; total_device_count: %s',
      local_devices_to_use,
      device_count,
  )

  if min_replay_size >= num_timesteps:
    raise ValueError(
        'No training will happen because min_replay_size >= num_timesteps'
    )

  if max_replay_size is None:
    max_replay_size = num_timesteps

  # The number of environment steps executed for every `actor_step()` call.
  env_steps_per_actor_step = action_repeat * num_envs
  # equals to ceil(min_replay_size / env_steps_per_actor_step)
  num_prefill_actor_steps = -(-min_replay_size // num_envs)
  num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
  assert num_timesteps - num_prefill_env_steps >= 0
  num_evals_after_init = max(num_evals - 1, 1)
  # The number of run_one_sac_epoch calls per run_sac_training.
  # equals to
  # ceil(num_timesteps - num_prefill_env_steps /
  #      (num_evals_after_init * env_steps_per_actor_step))
  num_training_steps_per_epoch = -(
      -(num_timesteps - num_prefill_env_steps)
      // (num_evals_after_init * env_steps_per_actor_step)
  )

  assert num_envs % device_count == 0
  env = environment
  wrap_for_training = envs.training.wrap

  rng = jax.random.PRNGKey(seed)
  rng, key = jax.random.split(rng)
  v_randomization_fn = None
    
  env = wrap_for_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )  # pytype: disable=wrong-keyword-args

  obs_size = env.observation_size
  if isinstance(obs_size, Dict):
    raise NotImplementedError('Dictionary observations not implemented in SAC')
  action_size = env.action_size

  normalize_fn = lambda x, y: x
  if normalize_observations:
    normalize_fn = running_statistics.normalize
  sac_network = network_factory(
      observation_size=obs_size,
      action_size=action_size,
      num_obj = num_obj,
      mode=mode,
      hidden_layer_sizes = network_size,
      ensemble_size = ensemble_size,
      preprocess_observations_fn=normalize_fn)
  
  if exploration_strategy:
    make_exploration_policy = sac_networks.make_exploration_inference_fn(sac_network,
                                                                         delta,
                                                                         beta_reward,
                                                                         beta_cost,
                                                                         topk,
                                                                         budget_st,
                                                                         exploration_method)
  make_policy = sac_networks.make_inference_fn(sac_network)

  alpha_optimizer = optax.adam(learning_rate=3e-4)
  policy_optimizer = optax.adam(learning_rate=learning_rate)
  q_optimizer = optax.adam(learning_rate=q_learning_rate)
  multiplier_optimizer = optax.adam(learning_rate=multiplier_learning_rate)
  delta_optimizer = optax.adam(learning_rate=1e-4)

  dummy_obs = jnp.zeros((obs_size,))
  dummy_action = jnp.zeros((action_size,))
  dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=dummy_obs,
      action=dummy_action,
      reward=0.,
      multi_reward=jnp.zeros(2),
      discount=0.,
      next_observation=dummy_obs,
      extras={
          'state_extras': {
              'truncation': 0.
          },
          'policy_extras': {}
      })
  
  replay_buffer = replay_buffers.UniformSamplingQueue(
      max_replay_size=max_replay_size // device_count,
      dummy_data_sample=dummy_transition,
      sample_batch_size=batch_size * grad_updates_per_step // device_count,
  )

  replay_buffer = replay_buffers.UniformSamplingQueue(
      max_replay_size=max_replay_size // device_count,
      dummy_data_sample=dummy_transition,
      sample_batch_size=batch_size * grad_updates_per_step // device_count)

  alpha_loss, critic_loss, actor_loss, penalty_loss, cost_loss = sac_losses.make_losses(
      sac_network=sac_network,
      discounting=discounting,
      action_size=action_size,
      method=method,
      budget=budget,
      cost_limit = cost_limit,
      beta_cost = beta_cost,
      convex=convex_coeff,
      tail_c=tail_c,
      tail_r=tail_r,
      )
  
  alpha_update = gradients.gradient_update_fn(
      alpha_loss, alpha_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)
  
  critic_update = gradients.gradient_update_fn(
    critic_loss,
    q_optimizer,
    has_aux=False,
    pmap_axis_name=_PMAP_AXIS_NAME)
  
  actor_update = gradients.gradient_update_fn(
        actor_loss, policy_optimizer, has_aux=True,
        pmap_axis_name=_PMAP_AXIS_NAME)
  
  multiplier_update = gradients.gradient_update_fn(
    penalty_loss,
    multiplier_optimizer,
    has_aux=False,
    pmap_axis_name=_PMAP_AXIS_NAME)
  
  delta_update = gradients.gradient_update_fn(
    cost_loss,
    delta_optimizer,
    has_aux=False,
    pmap_axis_name=_PMAP_AXIS_NAME)

  def sgd_step(
      carry: Tuple[TrainingState, PRNGKey], transitions: Transition
  ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
    training_state, key = carry

    key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)

    alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
        training_state.alpha_params,
        training_state.policy_params,
        training_state.normalizer_params,
        transitions,
        key_alpha,
        optimizer_state=training_state.alpha_optimizer_state)
    

    alpha = jnp.exp(training_state.alpha_params)

    critic_loss, q_params, q_optimizer_state = critic_update(
        training_state.q_params,
        training_state.policy_params,
        training_state.normalizer_params,
        training_state.target_q_params,
        alpha, 
        transitions,
        training_state.multiplier,
        key_critic,
        optimizer_state=training_state.q_optimizer_state)

    new_target_q_params = jax.tree_util.tree_map(
        lambda x, y: x * (1 - tau) + y * tau, training_state.target_q_params,
        q_params)
    
    def act_true_fn(_):
        (actor_loss, cost_info), policy_params, policy_optimizer_state = actor_update(
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.q_params,
            training_state.multiplier,
            alpha, 
            transitions,
            key_actor,
            optimizer_state=training_state.policy_optimizer_state)
        
        q_cost = cost_info

        _, multiplier_new, multiplier_optimizer_state = multiplier_update(
                    training_state.multiplier,
                    #jnp.sum(training_state.cost_fly)/2560.*100.,
                    q_cost,
                    optimizer_state=training_state.multiplier_optimizer_state
                )
        
        _, delta_new, delta_optimizer_state = delta_update(
                    training_state.delta,
                    jnp.sum(training_state.cost_fly)/2560.*100.,
                    optimizer_state=training_state.delta_optimizer_state
                )

        delta_new = jnp.clip(delta_new, max=jnp.log(6.))
        
        new_training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=new_target_q_params,
        gradient_steps=training_state.gradient_steps + 1,
        env_steps=training_state.env_steps,
        cost_fly=training_state.cost_fly,
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=alpha_params,
        multiplier = multiplier_new,
        multiplier_optimizer_state = multiplier_optimizer_state,
        delta = delta_new,
        delta_optimizer_state = delta_optimizer_state,
        normalizer_params=training_state.normalizer_params)

        metrics = {
                'critic_loss': critic_loss,
                'alpha': jnp.exp(alpha_params),
            }
        
        return new_training_state, metrics
    
    def act_false_fn(_):
        new_training_state = TrainingState(
        policy_optimizer_state=training_state.policy_optimizer_state,
        policy_params=training_state.policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=new_target_q_params,
        gradient_steps=training_state.gradient_steps + 1,
        env_steps=training_state.env_steps,
        cost_fly=training_state.cost_fly,
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=alpha_params,
        multiplier = training_state.multiplier,
        multiplier_optimizer_state = training_state.multiplier_optimizer_state,
        delta = training_state.delta,
        delta_optimizer_state = training_state.delta_optimizer_state,
        normalizer_params=training_state.normalizer_params)

        metrics = {
                'critic_loss': critic_loss,
                'alpha': jnp.exp(alpha_params),
            }
        
        return new_training_state, metrics
    
    new_training_state, metrics = jax.lax.cond(
        (training_state.gradient_steps + 1) % actor_delay == 0, 
        act_true_fn, 
        act_false_fn, 
        operand=None
    )

    return (new_training_state, key), metrics

  def get_experience(
      normalizer_params: running_statistics.RunningStatisticsState,
      policy_params: Params, q_params,
      env_state: envs.State,
      buffer_state: ReplayBufferState, multiplier, delta_fly, cost_fly,
      key: PRNGKey, prefill = False
  ) -> Tuple[
      running_statistics.RunningStatisticsState,
      envs.State,
      ReplayBufferState,
  ]:
    if exploration_strategy:
        policy = make_exploration_policy((normalizer_params, policy_params), 
                                         (normalizer_params, q_params),
                                         multiplier, delta_fly, cost_fly)
    else:
        policy = make_policy((normalizer_params, policy_params))
    env_state, transitions = acting.actor_step(
        env, env_state, name, policy, key, extra_fields=('truncation',)
    )

    normalizer_params = running_statistics.update(
        normalizer_params,
        transitions.observation,
        pmap_axis_name=_PMAP_AXIS_NAME,
    )

    buffer_state = replay_buffer.insert(buffer_state, transitions)
    return normalizer_params, env_state, buffer_state, -jnp.sum(transitions.multi_reward[:, 0])

  def training_step(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[
      TrainingState,
      envs.State,
      ReplayBufferState,
      Metrics,
  ]:
    experience_key, training_key = jax.random.split(key)
    normalizer_params, env_state, buffer_state, cost_cum = get_experience(
        training_state.normalizer_params, training_state.policy_params, training_state.q_params,
        env_state, buffer_state, training_state.multiplier, training_state.delta, training_state.cost_fly,
        experience_key)
    
    cost_fly_new = jnp.concatenate([training_state.cost_fly[1:], jnp.array([cost_cum])])
    training_state = training_state.replace(
        normalizer_params=normalizer_params,
        env_steps=training_state.env_steps + env_steps_per_actor_step,
        cost_fly = cost_fly_new)
    
    buffer_state, transitions = replay_buffer.sample(buffer_state)
    transitions = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
        transitions,
    )
    (training_state, _), metrics = jax.lax.scan(
        sgd_step, (training_state, training_key), transitions
    )

    metrics['buffer_current_size'] = replay_buffer.size(buffer_state)
    return training_state, env_state, buffer_state, metrics

  def prefill_replay_buffer(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:

    def f(carry, unused):
      del unused
      training_state, env_state, buffer_state, key = carry
      key, new_key = jax.random.split(key)
      new_normalizer_params, env_state, buffer_state, cost_cum = get_experience(
          training_state.normalizer_params, training_state.policy_params, training_state.q_params,
          env_state, buffer_state, training_state.multiplier, training_state.delta, training_state.cost_fly,
          key, prefill=True)
      
      cost_fly_new = jnp.concatenate([training_state.cost_fly[1:], jnp.array([cost_cum])])
      new_training_state = training_state.replace(
          normalizer_params=new_normalizer_params,
          env_steps=training_state.env_steps + env_steps_per_actor_step,
          cost_fly = cost_fly_new)
      return (new_training_state, env_state, buffer_state, new_key), ()

    return jax.lax.scan(
        f, (training_state, env_state, buffer_state, key), (),
        length=num_prefill_actor_steps)[0]

  prefill_replay_buffer = jax.pmap(
      prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME,
      donate_argnums=(0, 1, 2)
  )

  def training_epoch(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:

    def f(carry, unused_t):
      ts, es, bs, k = carry
      k, new_key = jax.random.split(k)
      ts, es, bs, metrics = training_step(ts, es, bs, k)
      return (ts, es, bs, new_key), metrics

    (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
        f,
        (training_state, env_state, buffer_state, key),
        (),
        length=num_training_steps_per_epoch,
    )
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return training_state, env_state, buffer_state, metrics

  training_epoch = jax.pmap(
      training_epoch, axis_name=_PMAP_AXIS_NAME, donate_argnums=(0, 1, 2)
  )

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
    nonlocal training_walltime
    t = time.time()
    (training_state, env_state, buffer_state, metrics) = training_epoch(
        training_state, env_state, buffer_state, key
    )
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (
        env_steps_per_actor_step * num_training_steps_per_epoch
    ) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()},
    }
    return (training_state, env_state, buffer_state, metrics)  # pytype: disable=bad-return-type  # py311-upgrade

  global_key, local_key = jax.random.split(rng)
  local_key = jax.random.fold_in(local_key, process_id)

  # Training state init
  training_state = _init_training_state(
      key=global_key,
      obs_size=obs_size,
      local_devices_to_use=local_devices_to_use,
      num_envs=num_envs,
      sac_network=sac_network,
      alpha_optimizer=alpha_optimizer,
      policy_optimizer=policy_optimizer,
      q_optimizer=q_optimizer,
      multiplier_optimizer=multiplier_optimizer,
      delta_optimizer=delta_optimizer)
  del global_key


  local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

  # Env init
  env_keys = jax.random.split(env_key, num_envs // jax.process_count())
  env_keys = jnp.reshape(
      env_keys, (local_devices_to_use, -1) + env_keys.shape[1:]
  )
  env_state = jax.pmap(env.reset)(env_keys)

  # Replay buffer init
  buffer_state = jax.pmap(replay_buffer.init)(
      jax.random.split(rb_key, local_devices_to_use)
  )

  eval_env = environment

  eval_env = wrap_for_training(
        eval_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=None,
    )  # pytype: disable=wrong-keyword-args

  evaluator = acting.Evaluator(
      eval_env, name,
      functools.partial(make_policy, deterministic=deterministic_eval),
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key)
  
  def get_eval_performance(metrics):
    y2 = metrics['eval/episode_reward']
    y1 = metrics['eval/episode_cost']
    return [y1, y2]

  # Run initial eval
  metrics = {}
  if process_id == 0 and num_evals > 1:
    metrics = evaluator.run_evaluation(
        _unpmap(
            (training_state.normalizer_params, training_state.policy_params)),
        training_metrics={})

    logging.info(metrics)

    xdata.append(0)
    y_ = get_eval_performance(metrics)
    ydata.append(y_)

    progress_fn(0, metrics)

  # Create and initialize the replay buffer.
  t = time.time()
  prefill_key, local_key = jax.random.split(local_key)
  prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
  training_state, env_state, buffer_state, _ = prefill_replay_buffer(
      training_state, env_state, buffer_state, prefill_keys) 

  replay_size = jnp.sum(jax.vmap(
      replay_buffer.size)(buffer_state)) * jax.process_count()
  logging.info('replay size after prefill %s', replay_size)
  assert replay_size >= min_replay_size
  training_walltime = time.time() - t

  current_step = 0
  for _ in range(num_evals_after_init):
    logging.info('step %s', current_step)

    # Optimization
    epoch_key, local_key = jax.random.split(local_key)
    epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
    (training_state, env_state, buffer_state, training_metrics) = (
        training_epoch_with_timing(
            training_state, env_state, buffer_state, epoch_keys
        )
    )
    current_step = int(_unpmap(training_state.env_steps))
    #print(current_step)

    # Eval and logging
    if process_id == 0:
      if checkpoint_logdir:
        # Save current policy.
        params = _unpmap(
            (training_state.normalizer_params, training_state.policy_params))
        ckpt_config = checkpoint.network_config(
            observation_size=obs_size,
            action_size=env.action_size,
            normalize_observations=normalize_observations,
            network_factory=network_factory,
        )
        checkpoint.save(checkpoint_logdir, current_step, params, ckpt_config)


      # Run evals.
      metrics = evaluator.run_evaluation(
          _unpmap(
              (training_state.normalizer_params, training_state.policy_params)),
          training_metrics)

      logging.info(metrics)

      xdata.append(current_step)
      y_ = get_eval_performance(metrics)
      ydata.append(y_)

      progress_fn(current_step, metrics)

  total_steps = current_step
  assert total_steps >= num_timesteps

  params = _unpmap(
      (training_state.normalizer_params, training_state.policy_params))

  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(training_state)
  logging.info('total steps: %s', total_steps)
  pmap.synchronize_hosts()
  gc.collect()
  #pmap.synchronize_hosts()
  #return (make_policy, params, metrics)
  flat = buffer_state.data
  # print(flat.shape)
  flat = flat.reshape(-1, flat.shape[-1])
  reward_offset = obs_size + action_size + 1
  all_rewards = flat[:, reward_offset:reward_offset+2]

  return jax.device_get(xdata), jax.device_get(ydata), jax.device_get(all_rewards), params #, make_policy
