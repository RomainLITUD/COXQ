"""Soft Actor-Critic losses.

See: https://arxiv.org/pdf/1812.05905.pdf
"""
from typing import Any

from sac import types
from sac import networks as sac_networks
from sac.types import Params
from sac.types import PRNGKey
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

Transition = types.Transition


def quantile_huber_loss(
    current_quantiles_in, #(B, nq, nc)
    target_quantiles): #(B, nqt)
    current_quantiles = jnp.swapaxes(current_quantiles_in, -1, -2)#(B, nc, nq)
    n_quantiles = current_quantiles.shape[-1]
    cum_prob = (jnp.arange(n_quantiles, dtype=current_quantiles.dtype) + 0.5) / n_quantiles

    pairwise_delta = target_quantiles[...,None, None, :] - current_quantiles[...,None] #(B, nc, nq, nqt)
    abs_pairwise_delta = jnp.abs(pairwise_delta)
    huber_loss = jnp.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5)

    indicator = jnp.where(pairwise_delta < 0, 1.0, 0.0)
    indicator = jax.lax.stop_gradient(indicator)
    loss = jnp.abs(cum_prob[...,None] - indicator) * huber_loss
    # return loss
    return jnp.mean(loss, axis=tuple(range(1, loss.ndim)))


def qr_huber_loss(
    current_quantiles_in: jnp.ndarray,  # (B, nq, nc)
    target_quantiles: jnp.ndarray,      # (B, nq, nc)
) -> jnp.ndarray:

    B, nq, nc = current_quantiles_in.shape

    taus = (jnp.arange(nq, dtype=current_quantiles_in.dtype) + 0.5) / nq

    delta = target_quantiles[:, :, None, :] - current_quantiles_in[:, None, :, :]  # (B, nq, nq, nc)

    abs_delta = jnp.abs(delta)
    huber = jnp.where(
        abs_delta <= 1,
        0.5 * jnp.square(delta),
        (abs_delta - 0.5),
    )

    tau = taus[None, None, :, None]  # (1, 1, nq, 1)
    weight = jnp.abs(tau - (delta < 0))  # (B, nq, nq, nc)

    loss = weight * huber  # (B, nq, nq, nc)

    return loss.mean(axis=(1, 2, 3))


def make_losses(sac_network: sac_networks.SACNetworks, 
                discounting: float, 
                action_size: int, 
                budget: float,
                method: str, 
                cost_limit,
                tail_r,
                tail_c,
                convex,
                beta_cost,
                ):
  """Creates the SAC losses."""
  target_entropy = -1.*action_size
  policy_network = sac_network.policy_network
  q_network = sac_network.q_network
  parametric_action_distribution = sac_network.parametric_action_distribution

  def alpha_loss(log_alpha: jnp.ndarray, policy_params: Params,
                 normalizer_params: Any, transitions: Transition,
                 key: PRNGKey) -> jnp.ndarray:
    """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
    dist_params = policy_network.apply(normalizer_params, policy_params,
                                       transitions.observation)
    
    action = parametric_action_distribution.sample_no_postprocessing(
        dist_params, key)
    log_prob = parametric_action_distribution.log_prob(dist_params, action)
    alpha = jnp.exp(log_alpha)
    alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
    return jnp.mean(alpha_loss)


  def critic_loss(q_params: Params, policy_params: Params,
                  normalizer_params: Any, target_q_params: Params,
                  alpha: jnp.ndarray, transitions: Transition, multiplier,
                  key: PRNGKey) -> jnp.ndarray:
    
    policy_key, sample_key, mask_key = jax.random.split(key, 3)
    q_old_action = q_network.apply(normalizer_params, q_params,
                                   transitions.observation, transitions.action)

    next_dist_params = policy_network.apply(normalizer_params, policy_params,
                                            transitions.next_observation)

    next_action = parametric_action_distribution.sample_no_postprocessing(
        next_dist_params, policy_key)
    next_log_prob = parametric_action_distribution.log_prob(
        next_dist_params, next_action)
    next_action = parametric_action_distribution.postprocess(next_action)

    next_q = q_network.apply(normalizer_params, target_q_params,
                             transitions.next_observation, next_action)

    entropy_term = jnp.stack([jnp.zeros_like(next_log_prob), alpha * next_log_prob], -1)


    if method in ['orac', 'wcsac']:
        q_min_r= next_q[:,-1, :2].min(-1)

        target_q = jax.lax.stop_gradient(transitions.multi_reward[:,1] +
                                transitions.discount * discounting * 
                                (q_min_r - alpha * next_log_prob)) # (B, 2)
        
        obj_loss1 = jnp.mean((q_old_action[:,-1,:2] - target_q[...,None])**2, -1)

        target_quantiles_cost = jax.lax.stop_gradient(transitions.multi_reward[...,:1, None] +
                                (transitions.discount * discounting)[...,None, None] * 
                                next_q[:,:25])
        
        obj_loss2 = qr_huber_loss(q_old_action[:,:25], target_quantiles_cost)
        truncation = transitions.extras['state_extras']['truncation']
        obj_loss1 *= (1 - truncation)
        obj_loss2 *= (1 - truncation)
        obj_loss = obj_loss1.mean() + obj_loss2.mean()
        q_loss = jnp.sum(obj_loss)


    if method == 'saclag_ucb':
        q_min= next_q.min(-1)

        target_q = jax.lax.stop_gradient(transitions.multi_reward +
                                transitions.discount[...,None] * discounting * 
                                (q_min-entropy_term)) # (B, 2)
        
        q_loss = jnp.mean((q_old_action - target_q[...,None])**2, -1)
        truncation = transitions.extras['state_extras']['truncation']
        q_loss *= (1 - truncation)[...,None]


    if method == 'tqc':
        next_quantiles_reward = jnp.sort(next_q[:,:,1].reshape(next_q.shape[:1]+(-1,))) # (B, nq*nc)
        next_quantiles_reward = next_quantiles_reward[..., :-int(tail_r*5)] - alpha * next_log_prob[:,None]
        # next_quantiles_reward = next_quantiles_reward - alpha * next_log_prob[:,None]

        # random_indices = jax.random.randint(sample_key, (2,), 0, 5)
        next_quantiles_cost = jnp.sort(next_q[:,:,0].reshape(next_q.shape[:1]+(-1,))) # (B, nq*nc)
        next_quantiles_cost = next_quantiles_cost[..., :-int(tail_c*5)]

        # next_quantiles_cost = next_q[:,:,0]

        target_quantiles_cost = jax.lax.stop_gradient(transitions.multi_reward[...,:1] +
                                (transitions.discount * discounting)[...,None] * 
                                next_quantiles_cost)
        
        target_quantiles_reward = jax.lax.stop_gradient(transitions.multi_reward[...,1:] +
                                (transitions.discount * discounting)[...,None] * 
                                next_quantiles_reward)
        
        obj_loss1 = quantile_huber_loss(q_old_action[:,:,1], target_quantiles_reward)
        obj_loss2 = quantile_huber_loss(q_old_action[:,:,0], target_quantiles_cost)
        # obj_loss2 = qr_huber_loss(q_old_action[:,:,0], target_quantiles_cost)


        truncation = transitions.extras['state_extras']['truncation']
        obj_loss1 *= (1 - truncation)#[:,None]*weights
        obj_loss2 *= (1 - truncation)#[:,None]*weights
        obj_loss = obj_loss1.mean() + obj_loss2.mean()
  
        q_loss = jnp.sum(obj_loss)

    
    if method == 'cal':
        qr_min= next_q[:,1, :2].min(-1)
        qc= next_q[:,0]
        qc_min = qc.mean(-1) - 0.5*qc.std(-1)
        q_min = jnp.stack([qc_min, qr_min], -1)

        target_q = jax.lax.stop_gradient(transitions.multi_reward +
                                transitions.discount[...,None] * discounting * 
                                (q_min-entropy_term)) # (B, 2)
        
        q_loss = jnp.mean((q_old_action - target_q[...,None])**2, -1)
        truncation = transitions.extras['state_extras']['truncation']
        q_loss *= (1 - truncation)[...,None]


    return q_loss.mean()


  def actor_loss(policy_params: Params, normalizer_params: Any,
                 q_params: Params, multiplier: Params, 
                 alpha: jnp.ndarray, transitions: Transition,
                 key: PRNGKey) -> jnp.ndarray:
    
    dist_params = policy_network.apply(normalizer_params, policy_params,
                                       transitions.observation)
    action = parametric_action_distribution.sample_no_postprocessing(
        dist_params, key)
    log_prob = parametric_action_distribution.log_prob(dist_params, action)
    action = parametric_action_distribution.postprocess(action)

    q_action = q_network.apply(normalizer_params, q_params,
                               transitions.observation, action)
    
    current_q = q_network.apply(normalizer_params, q_params,
                               transitions.observation, transitions.action)
    
    lam = jnp.exp(multiplier)

    if method == 'saclag_ucb':
       actor_loss = alpha * log_prob - q_action[:,1].min(-1) - lam*q_action[:,0].min(-1)
       return jnp.mean(actor_loss), -jax.lax.stop_gradient(q_action[:,0].min(-1)).mean()
       
    
    if method == 'cal':
        q_cost = -q_action[:, 0]
        q_current = -current_q[:, 0]
        q_reward = q_action[:,1,:2].min(-1)

        actor_QC = jnp.mean(q_cost, -1) + 0.5*jnp.std(q_cost, -1)

        current_qc = jnp.mean(q_current, -1) + 0.5*jnp.std(q_current, -1)

        rect = jnp.clip(convex * jnp.mean(cost_limit - current_qc), max=lam)

        actor_loss = alpha * log_prob - q_reward + jax.lax.stop_gradient(lam - rect)*actor_QC


        return jnp.mean(actor_loss), jax.lax.stop_gradient(q_cost).mean()


    if method == 'tqc':
        q_cost = -jnp.mean(q_action[:,:13,0], (1,2))
        q_current = -jnp.mean(current_q[:,:13,0], (1,2))

        q_reward = jnp.mean(q_action[:,:,1], (1,2))
        lam = jnp.exp(multiplier)

        rect = jnp.clip(convex * (cost_limit - q_current), max=lam)

        actor_loss = alpha * log_prob - q_reward + jax.lax.stop_gradient(lam - rect)*q_cost
        # actor_loss = alpha * log_prob - q_reward + jax.lax.stop_gradient(lam)*q_cost

        return jnp.mean(actor_loss), jax.lax.stop_gradient(q_current).mean()


    if method == 'orac':
        q_cost = -jnp.mean(q_action[:,:13], (1,2))
        q_current = -jnp.mean(current_q[:,:13], (1,2))

        q_reward = jnp.min(q_action[:,-1,:2], -1)
        lam = jnp.exp(multiplier)

        rect = jnp.clip(convex * (cost_limit - q_current), max=lam)

        actor_loss = alpha * log_prob - q_reward + jax.lax.stop_gradient(lam - rect)*q_cost

        return jnp.mean(actor_loss), jax.lax.stop_gradient(q_current).mean()
  
    
    if method == 'wcsac':
        q_cost = -jnp.mean(q_action[:,:13], (1,2))

        q_reward = jnp.min(q_action[:,-1,:2], -1)
        lam = jnp.exp(multiplier)

        actor_loss = (alpha * log_prob - q_reward + lam*q_cost) / (1. + lam)

        return jnp.mean(actor_loss), jax.lax.stop_gradient(q_cost).mean()


  def penalty_loss(multiplier, cost_mean):
      lambda_loss = jnp.exp(multiplier)*(cost_limit-cost_mean)
      return lambda_loss.mean()


  def cost_loss(step_length, cost_mean):
      cost_loss = jnp.exp(step_length)*(cost_mean - budget)
      return cost_loss.mean()

  
  return alpha_loss, critic_loss, actor_loss, penalty_loss, cost_loss
