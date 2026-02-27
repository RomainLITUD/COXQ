from typing import Callable, Optional

import jax
from jax import numpy as jnp
import optax
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map


def cagrad_full(g1, g2, lambda_=0.5, steps=50, lr=1e-2):
    def objective(alpha):
        # Compute weighted sum
        w = jnp.array([alpha, 1.0 - alpha])
        g = w[0] * g1 + w[1] * g2

        # Norm term
        norm_term = jnp.sum(g ** 2)

        # Cosine terms
        g_norm = jnp.linalg.norm(g) + 1e-8
        cos1 = jnp.dot(g, g1) / (g_norm * (jnp.linalg.norm(g1) + 1e-8))
        cos2 = jnp.dot(g, g2) / (g_norm * (jnp.linalg.norm(g2) + 1e-8))

        alignment = jnp.minimum(cos1, cos2)

        return norm_term - lambda_ * alignment

    # Initialize alpha = 0.5
    alpha = jnp.array(0.5)

    # Set up optimizer
    opt = optax.adam(lr)
    opt_state = opt.init(alpha)

    @jax.jit
    def step(alpha, opt_state):
        loss, grad = jax.value_and_grad(objective)(alpha)
        updates, opt_state = opt.update(grad, opt_state)
        alpha = optax.apply_updates(alpha, updates)
        alpha = jnp.clip(alpha, 0.0, 1.0)  # enforce simplex constraint
        return alpha, opt_state

    # Run optimization
    for _ in range(steps):
        alpha, opt_state = step(alpha, opt_state)

    # Final weights and gradient
    w = jnp.array([alpha, 1.0 - alpha])
    g_final = w[0] * g1 + w[1] * g2

    return g_final, w


def value_and_jacobian(fun):
    """Returns a function that, when called, returns (fun(...) , jacobian_of_fun)."""
    def wrapped(*args, **kwargs):
        value, aux = fun(*args, **kwargs)
        jac, aux = jax.jacobian(fun, has_aux=True)(*args, **kwargs)
        return (value, aux), jac
    return wrapped


def loss_and_jacobian(loss_fn: Callable[..., float],
                   pmap_axis_name: Optional[str]):
  g = value_and_jacobian(loss_fn)

  def h(*args, **kwargs):
    value, grad = g(*args, **kwargs)
    return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)

  return g if pmap_axis_name is None else h

def loss_and_pgrad(
    loss_fn: Callable[..., float],
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
  g = jax.value_and_grad(loss_fn, has_aux=has_aux)

  def h(*args, **kwargs):
    value, grad = g(*args, **kwargs)
    return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)

  return g if pmap_axis_name is None else h


def surgery(g1, g2, mode):
    g1g1 = jnp.dot(g1, g1)
    g2g2 = jnp.dot(g2, g2)
    g1g2 = jnp.dot(g1, g2)
    if mode == "pcgrad":
       alpha = g1g2 / (g2g2 + 1e-8)
       beta = g1g2 / (g1g1 + 1e-8)

       g = g1 - alpha * g2 + g2 - beta * g1
       weight = jnp.array([1-beta, 1-alpha])
       #weight = 2*weight/(jnp.linalg.norm(weight) + 1e-8)
       #g = weight[0]*g1 + weight[1]*g2

    if mode == "mgda":
       alpha = (g2g2 - g1g2) / (g1g1 + g2g2 - 2 * g1g2 + 1e-8)
       alpha = jnp.clip(alpha, 0, 1)*2

       g = alpha * g1 + (2 - alpha) * g2
       weight = jnp.array([alpha, 2-alpha])

   
    return g, weight


def gradient_manipulation(g1, g2, surgery_mode):
    cos_g1 = jnp.dot(g2, g1)
    #cos_g2 = jnp.dot(g1+g2, g2)

    def aligned_case(_):
        avg = g1 + g2
        return avg, jnp.asarray([1.0, 1.0])
    
    def misaligned_case(_):
        return surgery(g1, g2, surgery_mode)
    
    #non_conflict = ((cos_g1 > 0.) & (cos_g2 > 0.))

    #jax.debug.print("x: {}", cos_g1)
   
    return jax.lax.cond(cos_g1 > 0., aligned_case, misaligned_case, operand=None)

def conflicting_gradient_update_fn(loss_fn: Callable[..., float],
                       optimizer: optax.GradientTransformation,
                       pmap_axis_name: Optional[str], 
                       gradient_surgery_mode: str):

  loss_and_pjacob_fn = loss_and_jacobian(
      loss_fn, pmap_axis_name=pmap_axis_name)

  def f(*args, optimizer_state):
    value, grads = loss_and_pjacob_fn(*args)
    grad1 = jax.tree_map(lambda x: x[0], grads)
    grad2 = jax.tree_map(lambda x: x[1], grads)

    flat_grad1, unravel_fn = ravel_pytree(grad1)
    flat_grad2, _ = ravel_pytree(grad2)

    g, weight = gradient_manipulation(flat_grad1, flat_grad2, gradient_surgery_mode)

    final_grad = unravel_fn(g)

    params_update, optimizer_state = optimizer.update(final_grad, optimizer_state)
    params = optax.apply_updates(args[0], params_update)
    return value, params, optimizer_state, weight

  return f

def famo_gradient_update_fn(loss_fn: Callable[..., float],
                       optimizer: optax.GradientTransformation,
                       pmap_axis_name: Optional[str],
                       has_aux = True):

  loss_and_pjacob_fn = loss_and_jacobian(
      loss_fn, pmap_axis_name=pmap_axis_name)

  def f(*args, optimizer_state):
    (value, aux), grads = loss_and_pjacob_fn(*args)
    grad1 = jax.tree_map(lambda x: x[0], grads)
    grad2 = jax.tree_map(lambda x: x[1], grads)

    flat_grad1, unravel_fn = ravel_pytree(grad1)
    flat_grad2, _ = ravel_pytree(grad2)

    zi = jax.nn.softmax(args[4])
    ct = (zi/(value+1e-8)).sum()

    coeff = zi/(value+1e-8)/ct
    g = coeff[0]*flat_grad1 + coeff[1]*flat_grad2

    final_grad = unravel_fn(g)

    params_update, optimizer_state = optimizer.update(final_grad, optimizer_state)
    params = optax.apply_updates(args[0], params_update)
    return (value, aux), params, optimizer_state

  return f

def gradient_update_fn_decay(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):

  loss_and_pgrad_fn = loss_and_pgrad(
      loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux
  )

  def f(*args, optimizer_state):
    value, grads = loss_and_pgrad_fn(*args)
    params_update, optimizer_state = optimizer.update(grads, optimizer_state, params = args[0])
    params = optax.apply_updates(args[0], params_update)
    return value, params, optimizer_state

  return f