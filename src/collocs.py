import jax
import jax.numpy as jnp
import numpy as np


def generate_biased_sep_collocs(n_points, theta_pc, percentage=100, seed=42):
    """Generates separatrix θ collocation values with optional equatorial oversampling.

    Args:
        n_points: Total number of points to generate.
        theta_pc: Polar cap angle (radians); defines the valid θ range.
        percentage: Fraction (0-100) of points drawn uniformly. The remainder
            are drawn from a Gaussian centered at π/2 to oversample near the
            equator. Defaults to 100 (pure uniform).
        seed: JAX random seed.

    Returns:
        Array of θ values, shape (n_points, 1).
    """
    key = jax.random.PRNGKey(seed)

    if percentage >= 100:
        return jax.random.uniform(key, (n_points, 1), minval=theta_pc, maxval=jnp.pi - theta_pc)

    k1, k2 = jax.random.split(key)

    n_uni = int(n_points * (percentage / 100.0))
    n_eq = n_points - n_uni

    t_uni = jax.random.uniform(k1, (n_uni, 1), minval=theta_pc, maxval=jnp.pi - theta_pc)

    # Gaussian std of 0.15 rad ≈ 8.5° — concentrates points near the equator
    t_eq = jax.random.normal(k2, (n_eq, 1)) * 0.15 + (jnp.pi / 2.0)
    t_eq = jnp.clip(t_eq, theta_pc, jnp.pi - theta_pc)

    t_pool = jnp.vstack([t_uni, t_eq])
    t_pool = jax.random.permutation(key, t_pool, axis=0)

    return t_pool


def generate_lc_collocs(n_points, theta_pc, seed=42):
    """Generates collocation points on the light cylinder r = 1/sin(θ).

    Args:
        n_points: Number of points to generate.
        theta_pc: Polar cap angle; poles are excluded to avoid r → ∞.
        seed: JAX random seed.

    Returns:
        Array of shape (n_points, 2) with columns [r, θ].
    """
    key = jax.random.PRNGKey(seed)
    t_vals = jax.random.uniform(key, (n_points, 1), minval=theta_pc, maxval=jnp.pi - theta_pc)
    r_lc = 1.0 / jnp.sin(t_vals)
    return jnp.hstack([r_lc, t_vals])


def generate_eq_collocs(n_points, r_min, r_max, seed=42):
    """Generates collocation points on the equatorial plane θ = π/2.

    Args:
        n_points: Number of points to generate.
        r_min: Minimum radial coordinate.
        r_max: Maximum radial coordinate.
        seed: JAX random seed.

    Returns:
        Array of shape (n_points, 2) with columns [r, π/2].
    """
    key = jax.random.PRNGKey(seed)
    r_eq = jax.random.uniform(key, (n_points, 1), minval=r_min, maxval=r_max)
    t_eq = jnp.full_like(r_eq, jnp.pi / 2.0)
    return jnp.hstack([r_eq, t_eq])


def generate_star_collocs(theta_pc, Psi_max, Psi_S, R, size_c=256, size_o=256):
    """Generates static collocation points and flux targets on the stellar surface.

    The closed-line region spans [θ_pc, π - θ_pc]. The open-line region covers the two
    polar-cap patches [0, θ_pc] and [π - θ_pc, π].

    Args:
        theta_pc: Polar cap angle (radians).
        Psi_max: Maximum flux value (star surface).
        Psi_S: Separatrix flux value.
        R: Stellar radius.
        size_c: Number of closed-line-region star surface points.
        size_o: Number of open-line-region star surface points per polar-cap patch.

    Returns:
        Tuple (star_c_collocs, star_c_targets, star_o_collocs, star_o_targets),
        each a JAX array.
    """
    t_star_c = np.linspace(theta_pc, np.pi - theta_pc, size_c)
    r_star_c = np.full_like(t_star_c, R)
    star_c_collocs = jnp.stack([r_star_c, t_star_c], axis=1)
    star_c_targets = jnp.array(Psi_max * (np.sin(t_star_c) ** 2)).reshape(-1, 1)

    t_star_o_north = np.linspace(0.0, theta_pc, size_o)
    psi_target_o_north = Psi_max * (np.sin(t_star_o_north) ** 2)

    t_star_o_south = np.linspace(np.pi - theta_pc, np.pi, size_o)
    psi_target_o_south = 2.0 * Psi_S - Psi_max * (np.sin(t_star_o_south) ** 2)

    t_star_o = np.concatenate([t_star_o_south, t_star_o_north])
    r_star_o = np.full_like(t_star_o, R)
    star_o_collocs = jnp.stack([r_star_o, t_star_o], axis=1)
    star_o_targets = jnp.array(np.concatenate([psi_target_o_south, psi_target_o_north])).reshape(-1, 1)

    return star_c_collocs, star_c_targets, star_o_collocs, star_o_targets


def generate_rmax_collocs(n_points, r_max, seed=42):
    """Generates collocation points on the outer radial boundary r = r_max.

    Args:
        n_points: Number of points to generate.
        r_max: Outer boundary radius.
        seed: JAX random seed.

    Returns:
        Array of shape (n_points, 2) with columns [r_max, θ].
    """
    key = jax.random.PRNGKey(seed)
    t_rmax = jax.random.uniform(key, (n_points, 1), minval=0.0, maxval=jnp.pi)
    r_rmax = jnp.full_like(t_rmax, r_max)
    return jnp.hstack([r_rmax, t_rmax])
