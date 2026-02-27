import jax.numpy as jnp

# -- geometry helpers (all batched over B) ------------------------------------

def _solve_eq_on_sphere(a, n, R, eps=1e-5):
    """Max a^T z  s.t. ||z||<=R and n^T z = 0  (great-circle step)."""
    nn = jnp.sum(n*n, axis=-1) + eps
    a_perp = a - (jnp.sum(a*n, axis=-1)/nn)[..., None] * n
    an = jnp.linalg.norm(a_perp, axis=-1) + eps
    return (R/an)[..., None] * a_perp  # (B,A)

def _solve_eq_plus_plane(a, n_eq, n_pl, r_pl, R, eps=1e-5):
    """
    Max a^T z  s.t. ||z||<=R, n_eq^T z = 0, n_pl^T z = r_pl.
    Returns (z, feasible_on_sphere).
    """
    n1, n2 = n_eq, n_pl
    s11 = jnp.sum(n1*n1, -1)
    s22 = jnp.sum(n2*n2, -1)
    s12 = jnp.sum(n1*n2, -1)
    G = jnp.stack([jnp.stack([s11, s12], -1),
                   jnp.stack([s12, s22], -1)], -2)               # (B,2,2)
    I2 = jnp.array([[1.,0.],[0.,1.]], dtype=a.dtype)[None, ...]
    Ginv = jnp.linalg.solve(G + 1e-8*I2, I2)                    # (B,2,2)

    # Min-norm point on both planes
    rvec = jnp.stack([jnp.zeros_like(r_pl), r_pl], -1)[..., None]  # (B,2,1) = [0, r_pl]^T
    w0 = (Ginv @ rvec)[..., 0]                                     # (B,2)
    z0 = w0[...,0,None]*n1 + w0[...,1,None]*n2                     # (B,A)

    z0n2 = jnp.sum(z0*z0, -1)
    feas = z0n2 <= (R*R + 1e-12)
    rad  = jnp.sqrt(jnp.maximum(0.0, R*R - z0n2))

    # Move along the circle (nullspace of [n1,n2]) in the direction of a
    ta = jnp.stack([jnp.sum(n1*a, -1), jnp.sum(n2*a, -1)], -1)[..., None]  # (B,2,1)
    w  = (Ginv @ ta)[..., 0]                                                # (B,2)
    Pa = w[...,0,None]*n1 + w[...,1,None]*n2
    p  = a - Pa
    pn = jnp.linalg.norm(p, -1)

    z  = z0 + (rad/(pn+eps))[..., None]*p
    z  = jnp.where((pn>1e-10)[...,None], z, z0)
    return z, feas

# -- main: reward-locked, cost-exploring OAC with HARD mean-cost cap ----------

def oac_cost_shift(
    g2_lcb,         # (B,A)  ∇_a Q_c^{LCB} at a=μ      (for exploration; we minimize LCB ⇒ step along -g2_lcb)
    h_r_mean,       # (B,A)  ∇_a Q_r^{mean} at a=μ     (reward mean lock)
    h_c_mean,       # (B,A)  ∇_a Q_c^{mean} at a=μ     (cost mean & cap normal)
    q2_mean_mu,     # (B,)    Q_c^{mean}(μ)            (for cap slack)
    var,            # (B,A)  diag Σ  (nonnegative)
    tau: float,     # scalar  mean-cost threshold
    delta: float,   # scalar  KL radius
    eps: float = 1e-5,
):
    """
    Solve:
      maximize   (-g2_lcb)^T Δa
      s.t.       1/2 Δa^T Σ^{-1} Δa <= δ        (KL)
                 h_r_mean^T Δa = 0              (lock reward mean)
                 h_c_mean^T Δa <= 0             (no cost-mean increase)
                 h_c_mean^T Δa <= r_c           (HARD cap), r_c = τ - Q_c^{mean}(μ)

    Returns Δa with shape (B,A). If the hard cap makes the feasible set empty ⇒ returns 0.
    """
    B, A = g2_lcb.shape
    sv   = jnp.sqrt(var)                                             # (B,A)
    R    = jnp.full((B,), jnp.sqrt(2.0 * delta))                     # (B,)

    # Whitened quantities (sphere geometry)
    a   = sv * (-g2_lcb)                                             # (B,A) objective (cost-down)
    u_r = sv * h_r_mean                                              # (B,A) reward equality normal
    u_c = sv * h_c_mean                                              # (B,A) cost normal

    # Hard cap right-hand side in whitened coordinates uses same normal:
    # Combine "no increase" and cap: u_c^T z ≤ min(0, r_c)
    r_c = jnp.asarray(tau) - q2_mean_mu                              # (B,)
    s_cap = jnp.minimum(0.0, r_c)                                    # (B,)

    # Candidate 1: reward equality only (great circle)
    z_eq = _solve_eq_on_sphere(a, u_r, R, eps)                       # (B,A)
    # Check if it satisfies the hard cap u_c^T z ≤ s_cap
    cap_ok = jnp.sum(u_c * z_eq, axis=-1) <= (s_cap + 1e-10)         # (B,)

    # If not, enforce both: u_r^T z = 0 and u_c^T z = s_cap
    z_cap, feas = _solve_eq_plus_plane(a, u_r, u_c, s_cap, R, eps)   # (B,A), (B,)

    # Select: if eq-only obeys cap -> use it; else if plane feasible -> use plane; else -> zero (no feasible step)
    z = jnp.where(cap_ok[:, None], z_eq,
        jnp.where(feas[:, None],   z_cap, jnp.zeros_like(z_eq)))

    # Map back to action space
    delta_a = sv * z                                                 # (B,A)
    return delta_a
