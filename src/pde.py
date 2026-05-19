import jax
import jax.numpy as jnp

from flax import nnx

from jaxkan.pikan.adaptive import update_rba_weights, apply_rba_weights

EPS = 1e-12

gamma_def, eta_def = 0.999, 0.01


def pde_res_closed(model_c, collocs):
    """Computes the PDE residual for the closed-line region.

    Args:
        model_c: PINN model for the closed-line region.
        collocs: Collocation points of shape (N, 2) with columns [r, θ].

    Returns:
        Tuple (res_raw, res_w), each of shape (N, 1). res_raw is the physical
        residual; res_w is the residual used for RBA weight computation.
    """
    def u_fn(r, t):
        return model_c(jnp.array([[r, t]]))[0, 0]

    u_r_fn, u_t_fn = jax.grad(u_fn, argnums=0), jax.grad(u_fn, argnums=1)
    u_rr_fn, u_tt_fn = jax.grad(u_r_fn, argnums=0), jax.grad(u_t_fn, argnums=1)

    def residual_fn(r, t):
        u_r, u_t = u_r_fn(r, t), u_t_fn(r, t)
        u_rr, u_tt = u_rr_fn(r, t), u_tt_fn(r, t)

        sin_t, cos_t = jnp.sin(t), jnp.cos(t)
        lc_mult = 1.0 - (r * sin_t) ** 2

        standard = u_rr - (cos_t / (r ** 2 * sin_t + EPS)) * u_t + (1.0 / r ** 2) * u_tt
        additional = -2.0 * r * sin_t * ((cos_t / r) * u_t + sin_t * u_r)

        raw_residual = lc_mult * standard + additional
        return raw_residual, raw_residual

    res_raw, res_w = jax.vmap(residual_fn, in_axes=(0, 0))(collocs[:, 0], collocs[:, 1])
    return res_raw.reshape(-1, 1), res_w.reshape(-1, 1)


def pde_res_open(model_o, collocs):
    """Computes the PDE residual and alignment condition for the open-line region.

    Args:
        model_o: PINN model for the open-line region.
        collocs: Collocation points of shape (N, 2) with columns [r, θ].

    Returns:
        Tuple (res_gs_raw, res_gs_w, res_align), each of shape (N, 1).
        res_gs_raw is the physical GS residual; res_gs_w is the weighted version;
        res_align enforces ∇Ψ x ∇I = 0.
    """
    def u_fn(r, t):
        return model_o(jnp.array([[r, t]]))[0][0, 0]

    def i_fn(r, t):
        return model_o(jnp.array([[r, t]]))[1][0, 0]

    u_r_fn, u_t_fn = jax.grad(u_fn, argnums=0), jax.grad(u_fn, argnums=1)
    u_rr_fn, u_tt_fn = jax.grad(u_r_fn, argnums=0), jax.grad(u_t_fn, argnums=1)
    i_r_fn, i_t_fn = jax.grad(i_fn, argnums=0), jax.grad(i_fn, argnums=1)

    def residual_fn(r, t):
        u_r, u_t = u_r_fn(r, t), u_t_fn(r, t)
        u_rr, u_tt = u_rr_fn(r, t), u_tt_fn(r, t)
        i_val = i_fn(r, t)
        i_r, i_t = i_r_fn(r, t), i_t_fn(r, t)

        sin_t, cos_t = jnp.sin(t), jnp.cos(t)
        lc_mult = 1.0 - (r * sin_t) ** 2

        standard = u_rr - (cos_t / (r ** 2 * sin_t + EPS)) * u_t + (1.0 / r ** 2) * u_tt
        additional = -2.0 * r * sin_t * ((cos_t / r) * u_t + sin_t * u_r)

        grad_dot = i_r * u_r + (1.0 / r ** 2) * i_t * u_t
        grad_u_sq = u_r ** 2 + (1.0 / r ** 2) * u_t ** 2
        i_prime = grad_dot / (grad_u_sq + EPS)
        source = i_val * i_prime

        gs_res = lc_mult * standard + additional + source
        align_res = u_r * i_t - u_t * i_r

        return gs_res, gs_res, align_res

    res_gs_raw, res_gs_w, res_align = jax.vmap(residual_fn, in_axes=(0, 0))(collocs[:, 0], collocs[:, 1])
    return res_gs_raw.reshape(-1, 1), res_gs_w.reshape(-1, 1), res_align.reshape(-1, 1)


def lc_res(model_o, collocs):
    """Computes the light-cylinder boundary residual II' - 2B_z = 0.

    Args:
        model_o: PINN model for the open-line region.
        collocs: Collocation points on the light cylinder, shape (N, 2).

    Returns:
        Residual array of shape (N, 1).
    """
    r = collocs[:, 0:1]
    t = collocs[:, 1:2]

    Psi, I = model_o(collocs)

    def get_Psi_scalar(coords):
        return model_o(jnp.expand_dims(coords, axis=0))[0][0, 0]

    def get_I_scalar(coords):
        return model_o(jnp.expand_dims(coords, axis=0))[1][0, 0]

    dPsi_dcoords = jax.vmap(jax.grad(get_Psi_scalar))(collocs)
    dI_dcoords = jax.vmap(jax.grad(get_I_scalar))(collocs)

    dPsi_dr, dPsi_dt = dPsi_dcoords[:, 0:1], dPsi_dcoords[:, 1:2]
    dI_dr, dI_dt = dI_dcoords[:, 0:1], dI_dcoords[:, 1:2]

    Bz = (jnp.cos(t) / (r ** 2 * jnp.sin(t) + EPS)) * dPsi_dt + (1.0 / r) * dPsi_dr

    dot_product = dI_dr * dPsi_dr + (1.0 / r ** 2) * dI_dt * dPsi_dt
    grad_psi_sq = dPsi_dr ** 2 + (1.0 / r ** 2) * dPsi_dt ** 2
    I_prime = dot_product / (grad_psi_sq + EPS)
    II_prime = I * I_prime

    return II_prime - 2.0 * Bz


def rmax_res(model_o, collocs_rmax):
    """Evaluates the Neumann boundary condition ∂Ψ/∂r = 0 at the outer edge.

    Args:
        model_o: PINN model for the open-line region.
        collocs_rmax: Collocation points on the outer boundary, shape (N, 2).

    Returns:
        Radial derivative ∂Ψ/∂r of shape (N, 1).
    """
    def get_Psi_scalar(coords):
        return model_o(jnp.expand_dims(coords, axis=0))[0][0, 0]

    dPsi_dcoords = jax.vmap(jax.grad(get_Psi_scalar))(collocs_rmax)
    return dPsi_dcoords[:, 0:1]


def star_closed_res(model_c, collocs, targets):
    """Star surface residual for the closed-line region.

    Args:
        model_c: PINN model for the closed-line region.
        collocs: Star surface collocation points, shape (N, 2).
        targets: Target Ψ values, shape (N, 1).

    Returns:
        Residual array of shape (N, 1).
    """
    return model_c(collocs) - targets


def star_open_res(model_o, collocs, targets):
    """Star surface residual for the open-line region.

    Args:
        model_o: PINN model for the open-line region.
        collocs: Star surface collocation points, shape (N, 2).
        targets: Target Ψ values, shape (N, 1).

    Returns:
        Residual array of shape (N, 1).
    """
    return model_o(collocs)[0] - targets


def eq_res(model_o, collocs_eq, Psi_S):
    """Equatorial boundary residual Ψ = Ψ_S.

    Args:
        model_o: PINN model for the open-line region.
        collocs_eq: Equatorial collocation points, shape (N, 2).
        Psi_S: Separatrix flux value (scalar).

    Returns:
        Residual array of shape (N, 1).
    """
    return model_o(collocs_eq)[0] - Psi_S


def sep_val_res(model_c, model_o, sep_collocs, Psi_S):
    """Separatrix continuity residual Ψ_c = Ψ_S and Ψ_o = Ψ_S.

    Args:
        model_c: PINN model for the closed-line region.
        model_o: PINN model for the open-line region.
        sep_collocs: Separatrix collocation points, shape (N, 2).
        Psi_S: Separatrix flux value (scalar).

    Returns:
        Tuple (res_c, res_o), each of shape (N, 1).
    """
    return model_c(sep_collocs) - Psi_S, model_o(sep_collocs)[0] - Psi_S


def pde_loss_closed(model_c, l_pde_c, collocs):
    """RBA-weighted PDE loss for the closed-line region.

    Args:
        model_c: PINN model for the closed-line region.
        l_pde_c: Current RBA weight vector, shape (N, 1).
        collocs: Collocation points, shape (N, 2).

    Returns:
        Tuple (loss, loss_metric_true, l_new): scalar weighted loss, scalar
        physical loss, updated RBA weights.
    """
    res_raw, res_w = pde_res_closed(model_c, collocs)

    l_new = update_rba_weights(res_w, l_pde_c, gamma=gamma_def, eta=eta_def)
    loss = jnp.mean(apply_rba_weights(res_w, l_new) ** 2)
    loss_metric_true = jnp.mean(res_raw ** 2)

    return loss, loss_metric_true, l_new


def pde_loss_open(model_o, l_pde_o, l_align, collocs):
    """RBA-weighted PDE and alignment losses for the open-line region.

    Args:
        model_o: PINN model for the open-line region.
        l_pde_o: RBA weights for the GS residual, shape (N, 1).
        l_align: RBA weights for the alignment condition, shape (N, 1).
        collocs: Collocation points, shape (N, 2).

    Returns:
        Tuple (loss_pde, loss_metric_true, loss_align, l_pde_new, l_align_new).
    """
    res_gs_raw, res_gs_w, res_align = pde_res_open(model_o, collocs)

    l_pde_new = update_rba_weights(res_gs_w, l_pde_o, gamma=gamma_def, eta=eta_def)
    loss_pde = jnp.mean(apply_rba_weights(res_gs_w, l_pde_new) ** 2)

    l_align_new = update_rba_weights(res_align, l_align, gamma=gamma_def, eta=eta_def)
    loss_align = jnp.mean(apply_rba_weights(res_align, l_align_new) ** 2)

    loss_metric_true = jnp.mean(res_gs_raw ** 2)

    return loss_pde, loss_metric_true, loss_align, l_pde_new, l_align_new


def sep_loss(model_c, model_o, l_sep_c, l_sep_o, sep_collocs, Psi_S):
    """RBA-weighted separatrix continuity loss for both regions.

    Args:
        model_c: PINN model for the closed-line region.
        model_o: PINN model for the open-line region.
        l_sep_c: RBA weights for closed-side residual, shape (N, 1).
        l_sep_o: RBA weights for open-side residual, shape (N, 1).
        sep_collocs: Separatrix collocation points, shape (N, 2).
        Psi_S: Separatrix flux value (scalar).

    Returns:
        Tuple (loss_c, loss_o, l_sep_c_new, l_sep_o_new).
    """
    res_c, res_o = sep_val_res(model_c, model_o, sep_collocs, Psi_S)

    l_sep_c_new = update_rba_weights(res_c, l_sep_c, gamma=gamma_def, eta=eta_def)
    loss_c = jnp.mean(apply_rba_weights(res_c, l_sep_c_new) ** 2)

    l_sep_o_new = update_rba_weights(res_o, l_sep_o, gamma=gamma_def, eta=eta_def)
    loss_o = jnp.mean(apply_rba_weights(res_o, l_sep_o_new) ** 2)

    return loss_c, loss_o, l_sep_c_new, l_sep_o_new


def star_loss(model_c, model_o, l_star_c, l_star_o, collocs_c, targets_c, collocs_o, targets_o):
    """RBA-weighted star surface loss for both zones.

    Args:
        model_c: PINN model for the closed-line region.
        model_o: PINN model for the open-line region.
        l_star_c: RBA weights for closed-line-region star BC, shape (N, 1).
        l_star_o: RBA weights for open-line-region star BC, shape (N, 1).
        collocs_c: Closed-line-region star surface collocation points, shape (N, 2).
        targets_c: Target Ψ values for the closed-line region, shape (N, 1).
        collocs_o: Open-line-region star surface collocation points, shape (N, 2).
        targets_o: Target Ψ values for the open-line region, shape (N, 1).

    Returns:
        Tuple (loss_total, loss_c, loss_o, l_star_c_new, l_star_o_new).
    """
    res_c = star_closed_res(model_c, collocs_c, targets_c)
    l_star_c_new = update_rba_weights(res_c, l_star_c, gamma=gamma_def, eta=eta_def)
    loss_c = jnp.mean(apply_rba_weights(res_c, l_star_c_new) ** 2)

    res_o = star_open_res(model_o, collocs_o, targets_o)
    l_star_o_new = update_rba_weights(res_o, l_star_o, gamma=gamma_def, eta=eta_def)
    loss_o = jnp.mean(apply_rba_weights(res_o, l_star_o_new) ** 2)

    return loss_c + loss_o, loss_c, loss_o, l_star_c_new, l_star_o_new


def lc_loss(model_o, l_lc, collocs):
    """RBA-weighted light-cylinder boundary loss.

    Args:
        model_o: PINN model for the open-line region.
        l_lc: RBA weights for the LC residual, shape (N, 1).
        collocs: Light-cylinder collocation points, shape (N, 2).

    Returns:
        Tuple (loss, l_new).
    """
    res = lc_res(model_o, collocs)
    l_new = update_rba_weights(res, l_lc, gamma=gamma_def, eta=eta_def)
    loss = jnp.mean(apply_rba_weights(res, l_new) ** 2)
    return loss, l_new


def rmax_loss(model_o, l_rmax, collocs_rmax):
    """RBA-weighted outer boundary loss.

    Args:
        model_o: PINN model for the open-line region.
        l_rmax: RBA weights for the outer BC residual, shape (N, 1).
        collocs_rmax: Outer boundary collocation points, shape (N, 2).

    Returns:
        Tuple (loss, l_new).
    """
    res = rmax_res(model_o, collocs_rmax)
    l_new = update_rba_weights(res, l_rmax, gamma=gamma_def, eta=eta_def)
    loss = jnp.mean(apply_rba_weights(res, l_new) ** 2)
    return loss, l_new


def eq_loss(model_o, l_eq, collocs_eq, Psi_S):
    """RBA-weighted equatorial boundary loss.

    Args:
        model_o: PINN model for the open-line region.
        l_eq: RBA weights for the equatorial residual, shape (N, 1).
        collocs_eq: Equatorial collocation points, shape (N, 2).
        Psi_S: Separatrix flux value (scalar).

    Returns:
        Tuple (loss, l_new).
    """
    res = eq_res(model_o, collocs_eq, Psi_S)
    l_new = update_rba_weights(res, l_eq, gamma=gamma_def, eta=eta_def)
    loss = jnp.mean(apply_rba_weights(res, l_new) ** 2)
    return loss, l_new


@nnx.jit(static_argnames=("compute_grads_sep",))
def train_step_physics(
    model_c, model_o, optimizer,
    pde_c_collocs, pde_o_collocs, sep_collocs,
    star_c_collocs, star_c_targets, star_o_collocs, star_o_targets, Psi_S,
    lc_collocs, eq_collocs, rmax_collocs,
    λ_pde_c, λ_pde_o, λ_align, λ_sep_c, λ_sep_o, λ_star, λ_lc, λ_eq, λ_rmax,
    l_pde_c, l_pde_o, l_align, l_sep_c, l_sep_o, l_star_c, l_star_o, l_lc, l_eq, l_rmax,
    compute_grads_sep=False,
):
    """Single JIT-compiled training step for the full physics loss.

    Computes the total weighted loss over all boundary conditions and PDE
    residuals, updates both PINN models, and optionally returns per-loss
    gradient norms for loss-weight annealing.

    Args:
        model_c: PINN model for the closed-line region.
        model_o: PINN model for the open-line region.
        optimizer: Joint Flax NNX optimizer for (model_c, model_o).
        pde_c_collocs: Closed-line-region PDE collocation points, shape (N_c, 2).
        pde_o_collocs: Open-line-region PDE collocation points, shape (N_o, 2).
        sep_collocs: Separatrix collocation points, shape (N_s, 2).
        star_c_collocs: Closed-line-region star surface points, shape (N_sc, 2).
        star_c_targets: Target Ψ on closed-line-region star surface, shape (N_sc, 1).
        star_o_collocs: Open-line-region star surface points, shape (N_so, 2).
        star_o_targets: Target Ψ on open-line-region star surface, shape (N_so, 1).
        Psi_S: Separatrix flux value (scalar).
        lc_collocs: Light-cylinder collocation points, shape (N_lc, 2).
        eq_collocs: Equatorial collocation points, shape (N_eq, 2).
        rmax_collocs: Outer boundary collocation points, shape (N_r, 2).
        λ_pde_c, λ_pde_o, λ_align, λ_sep_c, λ_sep_o, λ_star, λ_lc, λ_eq, λ_rmax:
            Scalar loss weights for each term.
        l_pde_c, l_pde_o, l_align, l_sep_c, l_sep_o, l_star_c, l_star_o, l_lc, l_eq, l_rmax:
            Current RBA weight vectors for each loss term.
        compute_grads_sep: If True, also compute per-loss gradient tuples for
            loss-weight annealing (default False).

    Returns:
        Tuple (loss, aux, sep_grads):
            loss: Scalar total loss.
            aux: Tuple of individual losses and updated RBA weights.
            sep_grads: Tuple of per-loss gradient pairs, or None.
    """
    def total_loss_fn(model_c, model_o):
        loss_pde_c, loss_pde_c_true, l_pde_c_new = pde_loss_closed(model_c, l_pde_c, pde_c_collocs)
        loss_pde_o, loss_pde_o_true, loss_align, l_pde_o_new, l_align_new = pde_loss_open(model_o, l_pde_o, l_align, pde_o_collocs)
        loss_sep_c, loss_sep_o, l_sep_c_new, l_sep_o_new = sep_loss(model_c, model_o, l_sep_c, l_sep_o, sep_collocs, Psi_S)

        loss_star_total, loss_star_c, loss_star_o, l_star_c_new, l_star_o_new = star_loss(
            model_c, model_o, l_star_c, l_star_o, star_c_collocs, star_c_targets, star_o_collocs, star_o_targets
        )

        loss_lc, l_lc_new = lc_loss(model_o, l_lc, lc_collocs)
        loss_eq, l_eq_new = eq_loss(model_o, l_eq, eq_collocs, Psi_S)
        loss_rmax, l_rmax_new = rmax_loss(model_o, l_rmax, rmax_collocs)

        total = (
            λ_pde_c * loss_pde_c + λ_pde_o * loss_pde_o + λ_align * loss_align +
            λ_sep_c * loss_sep_c + λ_sep_o * loss_sep_o + λ_star * loss_star_total +
            λ_lc * loss_lc + λ_eq * loss_eq + λ_rmax * loss_rmax
        )

        aux = (
            loss_pde_c, loss_pde_o, loss_align, loss_sep_c, loss_sep_o,
            loss_star_c, loss_star_o, loss_lc, loss_eq, loss_rmax,
            l_pde_c_new, l_pde_o_new, l_align_new, l_sep_c_new, l_sep_o_new,
            l_star_c_new, l_star_o_new, l_lc_new, l_eq_new, l_rmax_new,
            loss_pde_c_true, loss_pde_o_true,
        )

        return total, aux

    (loss, aux), grads = nnx.value_and_grad(total_loss_fn, argnums=(0, 1), has_aux=True)(model_c, model_o)

    sep_grads = None

    if compute_grads_sep:
        g_pde_c = nnx.grad(lambda mc, mo: pde_loss_closed(mc, l_pde_c, pde_c_collocs)[0], argnums=(0, 1))(model_c, model_o)
        g_pde_o = nnx.grad(lambda mc, mo: pde_loss_open(mo, l_pde_o, l_align, pde_o_collocs)[0], argnums=(0, 1))(model_c, model_o)
        g_align = nnx.grad(lambda mc, mo: pde_loss_open(mo, l_pde_o, l_align, pde_o_collocs)[2], argnums=(0, 1))(model_c, model_o)
        g_sep_c = nnx.grad(lambda mc, mo: sep_loss(mc, mo, l_sep_c, l_sep_o, sep_collocs, Psi_S)[0], argnums=(0, 1))(model_c, model_o)
        g_sep_o = nnx.grad(lambda mc, mo: sep_loss(mc, mo, l_sep_c, l_sep_o, sep_collocs, Psi_S)[1], argnums=(0, 1))(model_c, model_o)
        g_star  = nnx.grad(lambda mc, mo: star_loss(mc, mo, l_star_c, l_star_o, star_c_collocs, star_c_targets, star_o_collocs, star_o_targets)[0], argnums=(0, 1))(model_c, model_o)
        g_lc    = nnx.grad(lambda mc, mo: lc_loss(mo, l_lc, lc_collocs)[0], argnums=(0, 1))(model_c, model_o)
        g_eq    = nnx.grad(lambda mc, mo: eq_loss(mo, l_eq, eq_collocs, Psi_S)[0], argnums=(0, 1))(model_c, model_o)
        g_rmax  = nnx.grad(lambda mc, mo: rmax_loss(mo, l_rmax, rmax_collocs)[0], argnums=(0, 1))(model_c, model_o)

        sep_grads = (g_pde_c, g_pde_o, g_align, g_sep_c, g_sep_o, g_star, g_lc, g_eq, g_rmax)

    optimizer.update((model_c, model_o), grads)

    return loss, aux, sep_grads
