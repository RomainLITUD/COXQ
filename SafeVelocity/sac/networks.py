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

"""SAC networks."""

from typing import Sequence, Tuple

from brax.training import distribution
from sac import nets
from sac import types
from sac.types import PRNGKey
import flax
from flax import linen
import jax.numpy as jnp
import jax
from sac.cox_explore import *
from sac.oac_cost import *

@flax.struct.dataclass
class SACNetworks: 
  policy_network: nets.FeedForwardNetwork
  q_network: nets.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(sac_networks: SACNetworks):
  """Creates params and inference function for the SAC agent."""

  def make_policy(params: types.PolicyParams,
                  deterministic: bool = False) -> types.Policy:

    def policy(observations: types.Observation,
               key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
      logits = sac_networks.policy_network.apply(*params, observations)
      if deterministic:
        return sac_networks.parametric_action_distribution.mode(logits), {}
      return sac_networks.parametric_action_distribution.sample(
          logits, key_sample), {}

    return policy

  return make_policy


def make_exploration_inference_fn(sac_networks: SACNetworks,
                                  delta, 
                                  k_reward,
                                  k_cost,
                                  topk,
                                  budget,
                                  exploration_method): 
  """Creates params and inference function for the SAC agent."""

  def make_policy(policy_params: types.PolicyParams,
                  q_params: types.Params,
                  multiplier, step_length, cost_fly) -> types.Policy:

    def policy(observations: types.Observation,
               key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
      key_sample, key_rand = jax.random.split(key_sample, 2)
      logits = sac_networks.policy_network.apply(*policy_params, observations)

      action, scale = jnp.split(logits, 2, axis=-1)
      sigma = jax.nn.softplus(scale) + 0.001 
      pre_tanh_mu_T = action.copy()

      onpolicy_cost = jnp.sum(cost_fly)/2560.*100.,

      def compute_Q_UB_reward(in_action, index):
        tanh_mu_T = jnp.tanh(pre_tanh_mu_T.at[index].set(in_action))
        qs = sac_networks.q_network.apply(*q_params, observations, tanh_mu_T)
        qs_reward = qs[:, :, 1].mean(1)
        q_reward_ub = qs_reward.mean(-1) + k_reward * qs_reward.std(-1)

        return q_reward_ub[index]
      
      def compute_Q_mean_cost(in_action, index):
        tanh_mu_T = jnp.tanh(pre_tanh_mu_T.at[index].set(in_action))
        qs = sac_networks.q_network.apply(*q_params, observations, tanh_mu_T)
        qs_cost = -qs[:,:,0]
        q_ucb_cost = qs_cost.mean(-1).mean(-1)

        return q_ucb_cost[index]
      
      def compute_UCB_cost(in_action, index):
        tanh_mu_T = jnp.tanh(pre_tanh_mu_T.at[index].set(in_action))
        qs = sac_networks.q_network.apply(*q_params, observations, tanh_mu_T)
        qs_cost = -qs[:,:topk,0]
        q_ucb_cost = (qs_cost.mean(-1) + k_cost * qs_cost.std(-1)).mean(-1)

        return q_ucb_cost[index]
      
      def compute_LCB_cost(in_action, index):
        tanh_mu_T = jnp.tanh(pre_tanh_mu_T.at[index].set(in_action))
        qs = sac_networks.q_network.apply(*q_params, observations, tanh_mu_T)
        qs_cost = -qs[:,:topk,0]#.mean(1)
        # qs_cost = -qs[:,0]
        q_lcb_cost = (qs_cost.mean(-1) - k_cost * qs_cost.std(-1)).mean(-1)

        return q_lcb_cost[index]
     
      def compute_Q_UB_orac(in_action, index):
        tanh_mu_T = jnp.tanh(pre_tanh_mu_T.at[index].set(in_action))
        qs = sac_networks.q_network.apply(*q_params, observations, tanh_mu_T)
        qs_cost = -qs[:, -topk-1:-1]
        qs_reward = qs[:, -1, :2]

        q_cost_lb = qs_cost.mean(-1) - k_cost * qs_cost.std(-1)

        q_reward_ub = qs_reward.mean(-1) + k_reward * jnp.abs(qs_reward[:,0] - qs_reward[:,1]) /2.

        lam = jnp.exp(multiplier)

        rect = jnp.clip(10. * (budget - jnp.mean(qs_cost, (1,2))), max=lam)

        q_opt = q_reward_ub - jax.lax.stop_gradient(lam - rect)*q_cost_lb.mean(-1)
        # q_opt = -q_lb_cost
        return q_opt[index]

      if exploration_method == "orac":
        grad_fn = jax.vmap(lambda action, idx: jax.grad(compute_Q_UB_orac)(action, idx))
        grad = grad_fn(pre_tanh_mu_T, jnp.arange(pre_tanh_mu_T.shape[0]))

      if exploration_method == "coxq":
        tanh_mu_T_all = jnp.tanh(pre_tanh_mu_T)
        Qs = sac_networks.q_network.apply(*q_params, observations, tanh_mu_T_all)
        Q_cost = -Qs[:, :13, 0]
        Q_UB_cost_muT = jnp.mean(Q_cost, (-2, -1))

        lam = jnp.exp(multiplier)

        grad_fn_reward = jax.vmap(lambda action, idx: jax.grad(compute_Q_UB_reward)(action, idx))
        g_reward = grad_fn_reward(pre_tanh_mu_T, jnp.arange(pre_tanh_mu_T.shape[0]))

        grad_fn_cost = jax.vmap(lambda action, idx: jax.grad(compute_LCB_cost)(action, idx))
        g_c = grad_fn_cost(pre_tanh_mu_T, jnp.arange(pre_tanh_mu_T.shape[0]))

        grad_fn_mean = jax.vmap(lambda action, idx: jax.grad(compute_Q_mean_cost)(action, idx))
        g_mean = grad_fn_mean(pre_tanh_mu_T, jnp.arange(pre_tanh_mu_T.shape[0]))

        delta_r = jnp.exp(step_length)

        sigma_T = jnp.pow(sigma, 2)

        rect = jnp.clip(10. * (budget - Q_UB_cost_muT), max=0.)
        lam_in = jax.lax.stop_gradient(lam - rect)

        delta_a = cox_shift(g_reward, g_c, g_mean, Q_UB_cost_muT, sigma_T, lam_in, budget, delta_r)
        
        mu_E = pre_tanh_mu_T + delta_a
        new_logits = jnp.concatenate([mu_E, scale], -1)

        return sac_networks.parametric_action_distribution.sample(
            new_logits, key_sample), {}

      sigma_T = jnp.pow(sigma, 2) 

      denom = jnp.sum(jnp.power(grad, 2) * sigma_T, -1)

      # mu_C = jnp.exp(step_length) * sigma_T * grad / (jnp.sqrt(denom[:,None])+1e-5)
      mu_C = delta * sigma_T * grad / (jnp.sqrt(denom[:,None])+1e-5)
      mu_E = pre_tanh_mu_T + mu_C

      new_logits = jnp.concatenate([mu_E, scale], -1)

      return sac_networks.parametric_action_distribution.sample(
          new_logits, key_sample), {}

    return policy

  return make_policy


def make_sac_networks(
    observation_size: int,
    action_size: int,
    num_obj: int,
    mode: str,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: nets.ActivationFn = linen.relu,
    ensemble_size: int = 15) -> SACNetworks:
  """Make SAC networks."""
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size)
  policy_network = nets.make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=(256, 256),
      activation=activation)
  q_network = nets.make_q_ensemble(
      observation_size,
      action_size,
      num_obj,
      mode,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation, random_prior=False,
      n_critics=ensemble_size)
  return SACNetworks(
      policy_network=policy_network,
      q_network=q_network,
      parametric_action_distribution=parametric_action_distribution)
