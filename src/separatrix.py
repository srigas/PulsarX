import jax
import jax.numpy as jnp

from flax import nnx

EPS = 1e-12


@nnx.jit
def safe_sep_eval(model_sep, t_vals, R=0.25):
    """Evaluates the separatrix model and clips output to physical bounds.

    Args:
        model_sep: Trained separatrix network.
        t_vals: Array of θ values, shape (N, 1).
        R: Stellar radius (lower bound for r).

    Returns:
        Clipped separatrix radius array of shape (N, 1).
    """
    raw_r = model_sep(t_vals)
    sin_theta = jnp.sin(t_vals)
    r_max = 0.999 / (sin_theta + EPS)
    return jnp.clip(raw_r, R, r_max)


@nnx.jit
def train_step_sep(model, optimizer, t, r_target):
    """Single training step for the 1D separatrix curve.

    Args:
        model: Separatrix network.
        optimizer: Flax NNX optimizer for the separatrix model.
        t: θ input array, shape (N, 1).
        r_target: Target radius values, shape (N, 1).

    Returns:
        Scalar MSE loss.
    """
    def loss_fn(m):
        preds = m(t)
        return jnp.mean((preds - r_target) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def compute_boundary_pressures(model_c, model_o, sep_collocs):
    """Computes magnetic pressure on both sides of the separatrix.

    Args:
        model_c: PINN model for the closed-line region.
        model_o: PINN model for the open-line region.
        sep_collocs: Separatrix collocation points, shape (N, 2).

    Returns:
        Tuple (p_in, p_out), each of shape (N, 1).
    """
    def psi_c_fn(r, t):
        return model_c(jnp.array([[r, t]]))[0, 0]

    psi_c_r_fn = jax.grad(psi_c_fn, argnums=0)
    psi_c_t_fn = jax.grad(psi_c_fn, argnums=1)

    def psi_o_fn(r, t):
        return model_o(jnp.array([[r, t]]))[0][0, 0]

    def i_o_fn(r, t):
        return model_o(jnp.array([[r, t]]))[1][0, 0]

    psi_o_r_fn = jax.grad(psi_o_fn, argnums=0)
    psi_o_t_fn = jax.grad(psi_o_fn, argnums=1)

    def pressure_fn(r, t):
        dr_c, dt_c = psi_c_r_fn(r, t), psi_c_t_fn(r, t)
        dr_o, dt_o = psi_o_r_fn(r, t), psi_o_t_fn(r, t)
        i_val = i_o_fn(r, t)

        lc_mult = 1.0 - (r * jnp.sin(t)) ** 2

        p_in  = lc_mult * (dr_c ** 2 + (1.0 / r ** 2) * dt_c ** 2)
        p_out = lc_mult * (dr_o ** 2 + (1.0 / r ** 2) * dt_o ** 2) + i_val ** 2

        return p_in, p_out

    p_in, p_out = jax.vmap(pressure_fn, in_axes=(0, 0))(sep_collocs[:, 0], sep_collocs[:, 1])
    return p_in.reshape(-1, 1), p_out.reshape(-1, 1)


def adjust_separatrix(model_c, model_o, model_sep, optimizer_sep, t_sep_pool,
                      theta_pc, beta=0.025, R=0.25, taper_pow=12, update_epochs=2000):
    """Updates the separatrix geometry based on the local pressure imbalance.

    Computes the signed pressure ratio on the current separatrix, optionally
    applies a spatial taper near the polar cap, and fits the separatrix network
    to the updated target positions.

    Args:
        model_c: PINN model for the closed-line region.
        model_o: PINN model for the open-line region.
        model_sep: Separatrix network to update.
        optimizer_sep: Flax NNX optimizer for the separatrix model.
        t_sep_pool: θ values along the separatrix, shape (N, 1).
        theta_pc: Polar cap angle (radians).
        beta: Step size for the geometric update.
        R: Stellar radius.
        taper_pow: Exponent for the spatial taper near the polar cap.
            Pass None to disable tapering.
        update_epochs: Number of epochs to re-fit the separatrix network.

    Returns:
        Updated separatrix radius array of shape (N, 1).
    """
    r_s_current = safe_sep_eval(model_sep, t_sep_pool, R)
    sep_collocs = jnp.concatenate([r_s_current, t_sep_pool], axis=1)

    p_in, p_out = compute_boundary_pressures(model_c, model_o, sep_collocs)
    pressure_ratio = (p_in - p_out) / (p_in + p_out + EPS)

    if taper_pow is not None:
        dist_from_eq = jnp.abs(t_sep_pool - jnp.pi / 2.0)
        max_dist = jnp.pi / 2.0 - theta_pc
        spatial_taper = 1.0 - (dist_from_eq / (max_dist + EPS)) ** taper_pow
        local_beta = beta * spatial_taper
    else:
        local_beta = beta

    r_s_new = r_s_current + 2.0 * local_beta * pressure_ratio

    sin_theta = jnp.sin(t_sep_pool)
    max_r_s = 0.999 / (sin_theta + EPS)
    min_r_s = R + EPS
    r_s_new = jnp.clip(r_s_new, min_r_s, max_r_s)

    for epoch in range(update_epochs):
        loss_sep = train_step_sep(model_sep, optimizer_sep, t_sep_pool, r_s_new)

    print(f"  -> Separatrix update complete. Final fit MSE: {loss_sep:.4e}")

    return r_s_new


@nnx.jit(static_argnames=("n_closed", "n_open"))
def generate_split_domains(model_sep, n_closed, n_open, theta_pc, r_max,
                           R=0.25, bias=1.0, seed=42):
    """Generates collocation points for the closed-line and open-line regions.

    Uses a continuous mapping to guarantee exactly n_closed and n_open
    points without rejection sampling. A bias exponent controls how
    points are distributed radially within each zone.

    Args:
        model_sep: Trained separatrix network.
        n_closed: Number of closed-line-region collocation points.
        n_open: Number of open-line-region collocation points.
        theta_pc: Polar cap angle (radians).
        r_max: Outer radial boundary.
        R: Stellar radius.
        bias: Exponent applied to the uniform radial samples; values > 1
            push points toward the inner boundary.
        seed: JAX random seed.

    Returns:
        Tuple (pde_c, pde_o), each an array of shape (n, 2) with columns [r, θ].
    """
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    t_c = jax.random.uniform(k1, (n_closed, 1), minval=theta_pc, maxval=jnp.pi - theta_pc)
    r_bound_c = safe_sep_eval(model_sep, t_c, R)
    u_c = jax.random.uniform(k2, (n_closed, 1))
    r_c = R + (u_c ** bias) * (r_bound_c - R)
    pde_c = jnp.hstack([r_c, t_c])

    t_o = jax.random.uniform(k3, (n_open, 1), minval=0.0, maxval=jnp.pi)
    r_bound_o = safe_sep_eval(model_sep, t_o, R)
    in_polar_cap = (t_o < theta_pc) | (t_o > jnp.pi - theta_pc)
    lower_limit_o = jnp.where(in_polar_cap, R, r_bound_o)
    u_o = jax.random.uniform(k4, (n_open, 1))
    r_o = lower_limit_o + (u_o ** bias) * (r_max - lower_limit_o)
    pde_o = jnp.hstack([r_o, t_o])

    return pde_c, pde_o


def generate_smooth_sep_targets(t_vals, theta_pc, R=0.25, R_target=0.8):
    """Generates a smooth initial separatrix shape that avoids the light cylinder.

    Args:
        t_vals: Array of θ values to evaluate.
        theta_pc: The polar cap boundary angle.
        R: Star radius.
        R_target: Maximum equatorial extent of the separatrix.

    Returns:
        Array of target r values with the same shape as t_vals.
    """
    sin_theta = jnp.sin(t_vals) + EPS
    sin_theta_pc = jnp.sin(theta_pc)

    k = jnp.log(R_target / R) / jnp.log(1.0 / sin_theta_pc)
    r_target = R * (sin_theta / sin_theta_pc) ** k
    r_target = jnp.clip(r_target, R, None)

    return r_target
