import jax.numpy as jnp

# Σ-inner product with Σ = diag(var)  (var: (B,A))
def _sigmadot(u, v, var):
    return jnp.sum(u * var * v, axis=-1)

# Cone-constrained Σ-MGDA (two objectives, weights (1, lam))
# Finds v* maximizing <g_r + lam g_c, ·>_Σ s.t. <g_r,·>_Σ>=0 and <g_c,·>_Σ>=0
def _sigma_mgda_cone_weighted_2(g_r, g_c, var, lam, eps=1e-5):
    v = g_r + lam * g_c                              # (B,A)

    s_rr = _sigmadot(g_r, g_r, var) + eps           # (B,)
    s_cc = _sigmadot(g_c, g_c, var) + eps
    s_rc = _sigmadot(g_r, g_c, var)
    v_r  = _sigmadot(v,   g_r, var)
    v_c  = _sigmadot(v,   g_c, var)

    m_r = v_r < 0.0
    m_c = v_c < 0.0

    # Project onto <g_r,·>_Σ = 0
    v_Hr = v - (v_r / s_rr)[..., None] * g_r
    # Project onto <g_c,·>_Σ = 0
    v_Hc = v - (v_c / s_cc)[..., None] * g_c

    # Project onto intersection <g_r,·>=0 and <g_c,·>=0
    det = s_rr * s_cc - s_rc * s_rc + eps
    a = ( s_cc * v_r - s_rc * v_c) / det
    b = (-s_rc * v_r + s_rr * v_c) / det
    v_both = v - a[..., None] * g_r - b[..., None] * g_c

    v_star = jnp.where(
        (m_r & m_c)[..., None], v_both,
        jnp.where((m_r & ~m_c)[..., None], v_Hr,
        jnp.where((~m_r & m_c)[..., None], v_Hc, v))
    )
    return v_star

def oac_cbox(
    g1_ucb,        # (B,A)  ∇_a(μ1 + β1 σ1) @ a=μ   (reward-UCB)
    g2_lcb,        # (B,A)  ∇_a(μ2 - β2 σ2) @ a=μ   (cost-LCB)
    grad_cost,     # (B,A)  ∇_a Q2 used in hard cap (mean or conservative)
    q2_at_mu,      # (B,)    Q2(μ) for the cap (same variant as grad_cost)
    var,           # (B,A)   diag Σ of policy
    lam: float,    # weight for cost (weights = (1, -λ))
    tau: float,    # cost threshold
    delta: float,  # KL radius
    eps: float = 1e-5,
):
    B, A = g1_ucb.shape

    # Over-cap rule: if Q2(μ) > τ, force zero step
    over_cap = q2_at_mu > tau                               # (B,)
    R = jnp.where(over_cap, 0.0, jnp.sqrt(2.0 * delta))     # (B,)

    # Improvement gradients (reward up, cost down)
    g_r = g1_ucb
    g_c = -g2_lcb

    # 1) Conflict-free target direction in Σ-metric
    v_star = _sigma_mgda_cone_weighted_2(g_r, g_c, var, lam)  # (B,A)

    # 2) Whitened variables
    sv = jnp.sqrt(var)                                        # (B,A)
    a  = sv * v_star                                          # (B,A)
    b  = sv * grad_cost                                       # (B,A)
    r  = tau - q2_at_mu                                       # (B,)

    # 3) Unconstrained OAC step on KL sphere
    a_norm = jnp.linalg.norm(a, axis=-1) + eps                # (B,)
    z_u = (R / a_norm)[..., None] * a                         # (B,A)

    # 4) Plane–sphere candidate for cost cap (computed for all; selected by masks)
    b_norm2   = jnp.sum(b * b, axis=-1) + eps                 # (B,)
    z0        = (r / b_norm2)[..., None] * b                  # (B,A)
    z0_norm2  = jnp.sum(z0 * z0, axis=-1)                     # (B,)
    plane_feas= z0_norm2 <= (R * R + 1e-5)                   # (B,)
    rad       = jnp.sqrt(jnp.maximum(0.0, R * R - z0_norm2))  # (B,)

    a_tan     = a - (jnp.sum(a * b, axis=-1) / b_norm2)[..., None] * b
    a_tan_norm= jnp.linalg.norm(a_tan, axis=-1)               # (B,)
    z_plane   = z0 + (rad / (a_tan_norm + eps))[..., None] * a_tan
    z_plane   = jnp.where((a_tan_norm > 1e-5)[..., None], z_plane, z0)

    # 5) Masks and selection (no Python branching)
    cap_ok    = jnp.sum(b * z_u, axis=-1) <= r + 1e-5        # (B,)
    use_plane = (~cap_ok) & plane_feas & (~over_cap)          # (B,)
    z = jnp.where(use_plane[:, None], z_plane,
        jnp.where(cap_ok[:, None], z_u, jnp.zeros_like(z_u)))

    # 6) Map back and enforce zero for over-cap items
    delta_a = sv * z                                           # (B,A)
    delta_a = jnp.where(over_cap[:, None], 0.0, delta_a)
    return delta_a
