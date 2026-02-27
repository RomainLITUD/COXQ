import jax.numpy as jnp

# Σ-inner product with Σ = diag(var)
def _sigmadot(u, v, var):
    return jnp.sum(u * var * v, axis=-1)

# Cone-constrained Σ-MGDA for two objectives (weights (1, lam))
# Ensures <g_r, v*>_Σ >= 0 and <g_c, v*>_Σ >= 0, and maximizes <g_r + lam g_c, ·>_Σ within that cone.
def sigma_mgda_cone_two(g_r, g_c, var, lam, eps: float = 1e-5):
    v = g_r + lam[:, None] * g_c

    s_rr = _sigmadot(g_r, g_r, var) + eps
    s_cc = _sigmadot(g_c, g_c, var) + eps
    s_rc = _sigmadot(g_r, g_c, var)
    v_r  = _sigmadot(v,   g_r, var)
    v_c  = _sigmadot(v,   g_c, var)

    m_r = v_r < 0.0
    m_c = v_c < 0.0

    # Project to the reward boundary if needed
    v_Hr = v - (v_r / s_rr)[..., None] * g_r
    # Project to the cost boundary if needed
    v_Hc = v - (v_c / s_cc)[..., None] * g_c

    # Project to the intersection of both boundaries if both conflict
    det   = s_rr * s_cc - s_rc * s_rc + eps
    alpha = ( s_cc * v_r - s_rc * v_c) / det
    beta  = (-s_rc * v_r + s_rr * v_c) / det
    v_both = v - alpha[..., None] * g_r - beta[..., None] * g_c

    # Piecewise selection (order-invariant)
    v_star = jnp.where(
        (m_r & m_c)[..., None], v_both,
        jnp.where((m_r & ~m_c)[..., None], v_Hr,
        jnp.where((~m_r & m_c)[..., None], v_Hc, v))
    )
    return v_star

def cox_shift(
    g1_ucb,        # (B,A)  ∇_a(μ1 + β1 σ1) at a=μ  (reward-UCB)
    g2_lcb,        # (B,A)  ∇_a(μ2 - β2 σ2) at a=μ  (cost-LCB)
    grad_cost,     # (B,A)  h = ∇_a Q2^{cap}  (mean or conservative)
    q2_cap_mu,     # (B,)    Q2^{cap}(μ)
    var,           # (B,A)   diag Σ (nonnegative)
    lam: float,    # scalar trade-off weight (1, -λ)
    tau: float,    # scalar cap threshold
    delta: float,  # scalar KL radius
    eps: float = 1e-5,
):
    """
    1) v* = cone–Σ-MGDA(g_r, g_c, lam)
    2) Δa = η* Σ v* with η* chosen to satisfy KL and minimize [h^T Δa - r]_+ along the ray.
    Returns Δa: (B,A)
    """
    # Improvement gradients (reward up, cost down)
    g_r = g1_ucb
    g_c = -g2_lcb

    # 1) conflict-free exploration direction in Σ-metric
    v_aligned = sigma_mgda_cone_two(g_r, g_c, var, lam, eps=eps)      # (B,A)

    v_naive   = g_r + lam[:,None] * g_c
    # print(v_naive.shape)
    slack = tau - q2_cap_mu
    v_star = jnp.where(slack[:, None]>0., v_naive, v_aligned)

    # 2) KL step size along that direction
    denom   = _sigmadot(v_star, v_star, var) + eps                 # (B,)
    eta_kl  = delta/jnp.sqrt(denom)                         # (B,)

    # 3) Cap violation slope along the ray and available slack
    r = tau - q2_cap_mu                                            # (B,)
    s = _sigmadot(grad_cost, v_star, var)                          # (B,)

    # If s > 0 the step increases cost; shrink to keep violation minimal (≤ 0 if possible).
    eta_cap_upper = jnp.maximum(0.0, r / (s + eps))                # (B,) valid only when s>0
    eta_when_s_pos = jnp.minimum(eta_kl, eta_cap_upper)            # (B,)

    # If s <= 0 the step does not increase cost (or reduces it): best is the largest step (η_kl)
    eta = jnp.where(s > 0.0, eta_when_s_pos, eta_kl)               # (B,)

    # 4) Final mean shift along v*: Δa = η Σ v*
    delta_a = eta[:, None] * (var * v_star)                        # (B,A)
    return delta_a
