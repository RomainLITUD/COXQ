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

# pylint:disable=g-multiple-import
"""Trains a 2D walker to run in the +x direction."""

from typing import Tuple

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp


class Walker2d(PipelineEnv):



  def __init__(
      self,
      forward_reward_weight: float = 1.0,
      ctrl_cost_weight: float = 1e-3,
      healthy_reward: float = 1.0,
      terminate_when_unhealthy: bool = True,
      healthy_z_range: Tuple[float, float] = (0.8, 2.0),
      healthy_angle_range=(-1.0, 1.0),
      reset_noise_scale=5e-3,
      exclude_current_positions_from_observation=True,
      backend='generalized',
      **kwargs
  ):
    """Creates a Walker environment.

    Args:
      forward_reward_weight: Weight for the forward reward, i.e. velocity in
        x-direction.
      ctrl_cost_weight: Weight for the control cost.
      healthy_reward: Reward for staying healthy, i.e. respecting the posture
        constraints.
      terminate_when_unhealthy: Done bit will be set when unhealthy if true.
      healthy_z_range: Range of the z-position for being healthy.
      healthy_angle_range: Range of joint angles for being healthy.
      reset_noise_scale: Scale of noise to add to reset states.
      exclude_current_positions_from_observation: x-position will be hidden from
        the observations if true.
      backend: str, the physics backend to use
      **kwargs: Arguments that are passed to the base class.
    """
    path = epath.resource_path('brax') / 'envs/assets/walker2d.xml'
    sys = mjcf.load(path)

    n_frames = 4
    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    super().__init__(sys=sys, backend=backend, **kwargs)

    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._healthy_angle_range = healthy_angle_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.init_q + jax.random.uniform(
        rng1, (self.sys.q_size(),), minval=low, maxval=hi
    )
    qvel = jax.random.uniform(
        rng2, (self.sys.qd_size(),), minval=low, maxval=hi
    )

    pipeline_state = self.pipeline_init(qpos, qvel)

    obs = self._get_obs(pipeline_state)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_forward': zero,
        'reward_ctrl': zero,
        'reward_healthy': zero,
        'x_position': zero,
        'x_velocity': zero,
        'reward': zero,
        'cost': zero,
        'pure_reward': zero,
    }
    return State(pipeline_state, obs, reward, done, metrics)

  def step(self, state: State, action: jax.Array) -> State:
    """Runs one timestep of the environment's dynamics."""
    pipeline_state0 = state.pipeline_state
    assert pipeline_state0 is not None
    pipeline_state = self.pipeline_step(pipeline_state0, action)

    x_velocity = (
        pipeline_state.x.pos[0, 0] - pipeline_state0.x.pos[0, 0]
    ) / self.dt
    forward_reward = self._forward_reward_weight * x_velocity

    z, angle = pipeline_state.x.pos[0, 2], pipeline_state.q[2]
    min_z, max_z = self._healthy_z_range
    min_angle, max_angle = self._healthy_angle_range
    is_healthy = (
        (z > min_z) & (z < max_z) * (angle > min_angle) & (angle < max_angle)
    )
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy

    ctrl_cost = jp.sum(jp.square(action))

    obs = self._get_obs(pipeline_state)

    reward = forward_reward + healthy_reward - self._ctrl_cost_weight*ctrl_cost
    pure_reward = forward_reward - self._ctrl_cost_weight*ctrl_cost
    cost = jp.where(x_velocity > 2.3415, 1., 0.)

    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

    state.metrics.update(
        reward_forward=forward_reward,
        reward_ctrl=ctrl_cost,
        reward_healthy=healthy_reward,
        x_position=pipeline_state.x.pos[0, 0],
        x_velocity=x_velocity,
        reward = reward,
        cost = cost,
        pure_reward = pure_reward,
    )
    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )

  def _get_obs(self, pipeline_state: base.State) -> jax.Array:
    """Returns the environment observations."""
    position = pipeline_state.q
    position = position.at[1].set(pipeline_state.x.pos[0, 2])
    velocity = jp.clip(pipeline_state.qd, -10, 10)

    if self._exclude_current_positions_from_observation:
      position = position[1:]

    return jp.concatenate((position, velocity))
