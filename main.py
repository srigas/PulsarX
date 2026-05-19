#!/usr/bin/env python3
"""
Axisymmetric pulsar magnetosphere solver — single-experiment entry point.

All hyper-parameters are passed as command-line arguments so that run.py
can launch multiple independent experiments without manual editing.

Usage
-----
    python main.py [options]
    python main.py --help
"""

import argparse
import json
import os
import time


def _str_to_bool(v):
    return v.lower() in ("true", "1", "yes")


parser = argparse.ArgumentParser(
    description="Pulsar magnetosphere PINN solver (single experiment).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--seed",             type=int,          default=42)
parser.add_argument("--R",                type=float,        default=0.25,
                    help="Stellar radius (R_*/R_LC)")
parser.add_argument("--multiplier",       type=float,        default=1.176,
                    help="Polar-cap angle multiplier (θ_pc = multiplier·√R)")
parser.add_argument("--double-precision", type=_str_to_bool, default=False,
                    metavar="BOOL")
parser.add_argument("--beta-sep",         type=float,        default=0.025,
                    help="Separatrix update step size β")
parser.add_argument("--tol-sep",          type=float,        default=8e-4,
                    help="Separatrix convergence tolerance")
parser.add_argument("--tol-dp",           type=float,        default=1e-1,
                    help="Pressure-balance convergence tolerance")
parser.add_argument("--alpha",            type=float,        default=1.0)
parser.add_argument("--beta",             type=float,        default=1.0)
parser.add_argument("--cycle-config",     type=str,
                    default='{"0": 10000, "2": 10000}',
                    help="JSON dict mapping cycle index to epoch count")
parser.add_argument("--sym-sep",          type=_str_to_bool, default=True,
                    metavar="BOOL",
                    help="Use SymmetricSeparatrix (True) or SeparatrixModel (False)")
args = parser.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", args.double_precision)

from flax import nnx
import optax

from jaxkan.pikan.adaptive import lr_anneal
from jaxkan.models.utils import count_params, get_adam

from src.models     import ModelOpen, ModelClosed, SymmetricSeparatrix, SeparatrixModel
from src.collocs    import (generate_biased_sep_collocs, generate_lc_collocs,
                             generate_eq_collocs, generate_star_collocs,
                             generate_rmax_collocs)
from src.separatrix import (generate_split_domains, adjust_separatrix,
                             safe_sep_eval, train_step_sep,
                             compute_boundary_pressures,
                             generate_smooth_sep_targets)
from src.pde        import train_step_physics
from src.io         import save_pinn_models
from src.logger     import Logger

seed             = args.seed
R                = args.R
multiplier       = args.multiplier
double_precision = args.double_precision
beta_sep         = args.beta_sep
tol_sep          = args.tol_sep
tol_dp           = args.tol_dp
alpha            = args.alpha
beta             = args.beta
cycle_config     = {int(k): v for k, v in json.loads(args.cycle_config).items()}
sym_sep          = args.sym_sep

results_dir = "results"
models_dir  = "model_ckpt"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir,  exist_ok=True)

prec_str = "dp" if double_precision else "sp"
_cc_str  = "-".join(f"{c}x{e}" for c, e in sorted(cycle_config.items()))
experiment_name = (
    f"R={R}_m={multiplier}_{prec_str}_"
    f"bs={beta_sep}_ts={tol_sep}_td={tol_dp}_"
    f"cc={_cc_str}_sym={'1' if sym_sep else '0'}_a={alpha}_b={beta}_seed={seed}"
)

logger = Logger(active=True)
logger.init(experiment_name)

print("=" * 70)
print("  Axisymmetric Pulsar Magnetosphere PINN Solver")
print("=" * 70)
print(f"  Experiment    : {experiment_name}")
print(f"  Precision     : {'double (64-bit)' if double_precision else 'single (32-bit)'}")
print(f"  R             : {R}")
print(f"  multiplier    : {multiplier}")
print(f"  β_sep         : {beta_sep}")
print(f"  tol_sep       : {tol_sep}  |  tol_dp : {tol_dp}")
print(f"  α             : {alpha}   |  β      : {beta}")
print(f"  cycle config  : {cycle_config}")
print(f"  sym_sep       : {sym_sep}")
print(f"  seed          : {seed}")
print("=" * 70 + "\n")

np.random.seed(seed)

R_max   = 4.0
Psi_max = 1.0

theta_pc = multiplier * np.sqrt(R / 1.0)
mu_pc    = np.cos(theta_pc)
Psi_S    = Psi_max * (np.sin(theta_pc) ** 2)

eval_freq         = 1_000
taper_pow         = 12
sep_update_epochs = 2500
beta_decay        = 1.0
lr_freq           = 2001
grad_mixing       = 0.95
resample_freq     = 5001
max_cycles        = 35

separatrix_params = {"n_hidden": 64, "num_layers": 2, "rff_std": 1.0, "seed": seed}
closed_params     = {"n_hidden": 15, "num_blocks": 1, "D": 5, "sine_D": 8,
                     "alpha": alpha, "beta": beta, "seed": seed}
open_params       = {"Psi_S": Psi_S, "n_hidden": 20, "num_blocks": 1, "D": 5, "sine_D": 8,
                     "alpha": alpha, "beta": beta, "seed": seed}

model_schedule = {
    "learning_rate": 5e-4,
    "schedule_type": "exponential",
    "decay_steps":   2000,
    "decay_rate":    0.90,
    "warmup_steps":  1000,
}

sep_schedule = optax.exponential_decay(
    init_value=5e-3, transition_steps=2000, decay_rate=0.6
)

pde_o_size       = 2 ** 11
pde_c_size       = 2 ** 10
pde_collocs_bias = 1.0
star_c_size      = 2 ** 8
star_o_size      = 2 ** 8
rmax_size        = 2 ** 9
sep_size         = 2 ** 9
sep_percentage   = 100
lc_size          = 2 ** 8
eq_size          = 2 ** 9

star_c_collocs, star_c_targets, star_o_collocs, star_o_targets = generate_star_collocs(
    theta_pc, Psi_max, Psi_S, R, size_c=star_c_size, size_o=star_o_size
)

t_sep_pool = generate_biased_sep_collocs(
    n_points=sep_size, theta_pc=theta_pc, percentage=sep_percentage, seed=seed
)

t_eval = jnp.linspace(theta_pc, jnp.pi - theta_pc, 1000).reshape(-1, 1)

lc_collocs      = generate_lc_collocs(n_points=lc_size, theta_pc=theta_pc, seed=seed)
eq_collocs_pool = generate_eq_collocs(n_points=eq_size, r_min=R, r_max=R_max, seed=seed)
rmax_collocs    = generate_rmax_collocs(n_points=rmax_size, r_max=R_max, seed=seed)

model_sep = (SymmetricSeparatrix if sym_sep else SeparatrixModel)(**separatrix_params)
print(f"Initialized model for the separatrix determination with {count_params(model_sep)} parameters.\n")

opt_sep       = optax.adam(learning_rate=sep_schedule)
optimizer_sep = nnx.Optimizer(model_sep, opt_sep, wrt=nnx.Param)

print("Pre-training Separatrix NN to Vacuum Dipole...")

t_init  = jnp.linspace(theta_pc, jnp.pi - theta_pc, 1000).reshape(-1, 1)
x_t_exp = R / (np.sin(theta_pc) ** 2)

if x_t_exp <= 0.98:
    print(f"  -> Experimental T-point (x = {x_t_exp:.3f}) is safe.")
    print("  -> Initializing with standard Vacuum Dipole.\n")
    r_target = R * (jnp.sin(t_init) ** 2) / (np.sin(theta_pc) ** 2)
else:
    print(f"  -> WARNING: Experimental T-point (x = {x_t_exp:.3f}) crosses the LC!")
    print(f"  -> Deploying the Squashed Dipole Initialization.\n")
    r_target = generate_smooth_sep_targets(t_init, theta_pc, R=R, R_target=0.9)

for epoch in range(sep_update_epochs):
    loss_sep = train_step_sep(model_sep, optimizer_sep, t_init, r_target)
    if epoch % 1000 == 0 or epoch == sep_update_epochs - 1:
        print(f"  Pre-train Epoch {epoch:4d} | MSE: {loss_sep:.4e}")

print("\nSeparatrix Pre-training complete!\n")

model_c = ModelClosed(**closed_params)
print(f"Initialized model for the closed lines region with {count_params(model_c)} parameters.")

model_o = ModelOpen(**open_params)
print(f"Initialized model for the open lines region with {count_params(model_o)} parameters.")

pde_c_collocs, pde_o_collocs = generate_split_domains(
    model_sep, pde_c_size, pde_o_size, theta_pc, R_max, R,
    bias=pde_collocs_bias, seed=seed
)

l_pde_c  = jnp.ones((pde_c_collocs.shape[0],    1))
l_pde_o  = jnp.ones((pde_o_collocs.shape[0],    1))
l_align  = jnp.ones((pde_o_collocs.shape[0],    1))
l_sep_c  = jnp.ones((t_sep_pool.shape[0],       1))
l_sep_o  = jnp.ones((t_sep_pool.shape[0],       1))
l_star_c = jnp.ones((star_c_collocs.shape[0],   1))
l_star_o = jnp.ones((star_o_collocs.shape[0],   1))
l_lc     = jnp.ones((lc_collocs.shape[0],       1))
l_eq_pool = jnp.ones((eq_collocs_pool.shape[0], 1))
l_rmax   = jnp.ones((rmax_collocs.shape[0],     1))

λ_pde_c = λ_pde_o = λ_align = λ_sep_c = λ_sep_o = λ_star = λ_lc = λ_eq = λ_rmax = jnp.array(1.0)

opt_type  = get_adam(**model_schedule)
optimizer = nnx.Optimizer((model_c, model_o), opt_type, wrt=nnx.Param)

times = []
max_p = 1.0

for cycle in range(max_cycles + 1):

    tick = time.time()

    if cycle in cycle_config:
        epochs_per_cycle = cycle_config[cycle]

    if cycle > 0:

        print("\n" + "=" * 60)
        print(f"=== CYCLE {cycle}: GEOMETRY & DOMAIN SETUP ===")
        print("=" * 60)

        r_s_old      = safe_sep_eval(model_sep, t_eval, R)
        current_beta = beta_sep * (beta_decay ** cycle)
        print(f"  -> Readjusting Separatrix... (Beta = {current_beta:.3e})")

        r_s_new = adjust_separatrix(
            model_c, model_o, model_sep, optimizer_sep, t_eval,
            theta_pc, current_beta, R, taper_pow, sep_update_epochs
        )

        r_s_current         = safe_sep_eval(model_sep, t_sep_pool, R)
        updated_sep_collocs = jnp.concatenate([r_s_current, t_sep_pool], axis=1)

        r_eval_new      = safe_sep_eval(model_sep, t_eval, R)
        sep_points_eval = jnp.concatenate([r_eval_new, t_eval], axis=1)

        rel_change = jnp.abs(r_s_new - r_s_old) / r_s_old
        avg_change = float(jnp.mean(rel_change))
        max_change = float(jnp.max(rel_change))
        print(f"  -> Max Relative Separatrix Change: {max_change:.4e}")
        print(f"  -> Avg Relative Separatrix Change: {avg_change:.4e} (Target: < {tol_sep:.1e})")

        pde_c_collocs, pde_o_collocs = generate_split_domains(
            model_sep, pde_c_size, pde_o_size, theta_pc, R_max, R,
            bias=pde_collocs_bias, seed=seed + cycle
        )

        l_pde_c = jnp.ones((pde_c_collocs.shape[0], 1))
        l_pde_o = jnp.ones((pde_o_collocs.shape[0], 1))
        l_align = jnp.ones((pde_o_collocs.shape[0], 1))

    else:
        r_s_current         = safe_sep_eval(model_sep, t_sep_pool, R)
        updated_sep_collocs = jnp.concatenate([r_s_current, t_sep_pool], axis=1)

        r_eval_current  = safe_sep_eval(model_sep, t_eval, R)
        sep_points_eval = jnp.concatenate([r_eval_current, t_eval], axis=1)

        avg_change = 1.0

    r_t_point = float(safe_sep_eval(model_sep, jnp.array([[jnp.pi / 2.0]]), R)[0, 0])
    x_t_point = r_t_point
    print(f"T-point position: x = {x_t_point:.4f} R_LC\n")

    # Points inside the closed-line region get zero RBA weight so they contribute nothing
    # to the equatorial loss.
    mask_eq_o  = (eq_collocs_pool[:, 0] > r_t_point).reshape(-1, 1)
    eq_collocs = eq_collocs_pool
    l_eq       = l_eq_pool * mask_eq_o

    print("=" * 60)
    print(f"=== CYCLE {cycle}: PINN TRAINING ===")
    print("=" * 60)

    for epoch in range(1, epochs_per_cycle + 1):

        do_anneal = (epoch % lr_freq == 0)

        loss, aux, sep_grads = train_step_physics(
            model_c, model_o, optimizer, pde_c_collocs, pde_o_collocs, updated_sep_collocs,
            star_c_collocs, star_c_targets, star_o_collocs, star_o_targets, Psi_S,
            lc_collocs, eq_collocs, rmax_collocs,
            λ_pde_c, λ_pde_o, λ_align, λ_sep_c, λ_sep_o, λ_star, λ_lc, λ_eq, λ_rmax,
            l_pde_c, l_pde_o, l_align, l_sep_c, l_sep_o, l_star_c, l_star_o, l_lc, l_eq, l_rmax,
            compute_grads_sep=do_anneal,
        )

        (loss_pde_c, loss_pde_o, loss_align, loss_sep_c, loss_sep_o,
         loss_star_c, loss_star_o, loss_lc, loss_eq, loss_rmax,
         l_pde_c, l_pde_o, l_align, l_sep_c, l_sep_o,
         l_star_c, l_star_o, l_lc, l_eq, l_rmax,
         loss_pde_c_true, loss_pde_o_true) = aux

        l_eq = l_eq * mask_eq_o

        if do_anneal:
            λ_pde_c, λ_pde_o, λ_align, λ_sep_c, λ_sep_o, λ_star, λ_lc, λ_eq, λ_rmax = lr_anneal(
                sep_grads,
                (λ_pde_c, λ_pde_o, λ_align, λ_sep_c, λ_sep_o, λ_star, λ_lc, λ_eq, λ_rmax),
                grad_mixing,
            )
            print(f"\n  [Epoch {epoch:5d}] Performed LR Annealing. New Weights:")
            print(f"    PDE -> Closed: {λ_pde_c:.3f} | Open: {λ_pde_o:.3f} | Align: {λ_align:.3f}")
            print(f"    BCs -> Sep_C: {λ_sep_c:.3f} | Sep_O: {λ_sep_o:.3f} | Star: {λ_star:.3f} | LC: {λ_lc:.3f} | Eq: {λ_eq:.3f} | Rmax: {λ_rmax:.3f}")

        if epoch % resample_freq == 0:
            current_seed = seed + cycle * epochs_per_cycle + epoch
            pde_c_collocs, pde_o_collocs = generate_split_domains(
                model_sep, pde_c_size, pde_o_size, theta_pc, R_max, R,
                bias=pde_collocs_bias, seed=current_seed
            )
            l_pde_c = jnp.ones((pde_c_collocs.shape[0], 1))
            l_pde_o = jnp.ones((pde_o_collocs.shape[0], 1))
            l_align = jnp.ones((pde_o_collocs.shape[0], 1))
            print(f"\n  [Epoch {epoch:5d}] Performed PDE Collocation Points Resampling.")
            print(f"    New Distribution -> Closed-line region: {pde_c_collocs.shape[0]} | Open-line region: {pde_o_collocs.shape[0]}")

        if epoch % eval_freq == 0 or epoch == epochs_per_cycle:
            p_in, p_out = compute_boundary_pressures(model_c, model_o, sep_points_eval)
            pres_diff   = jnp.abs(p_in - p_out)
            pres_sum    = jnp.array(0.5 * (p_in + p_out + 1e-12))
            avg_p       = float(jnp.mean(pres_diff / pres_sum))
            max_p       = float(jnp.max(pres_diff / pres_sum))

            print(f"\nCycle {cycle} | Epoch {epoch:5d} | Total Loss: {loss:.4e}")
            print(f"  Closed-line region -> PDE: {loss_pde_c:.2e} | Sep: {loss_sep_c:.2e} | Star: {loss_star_c:.2e}")
            print(f"  Open-line region   -> PDE: {loss_pde_o:.2e} | Align: {loss_align:.2e} | Sep: {loss_sep_o:.2e} | Star: {loss_star_o:.2e} | LC: {loss_lc:.2e} | Eq: {loss_eq:.2e} | Rmax: {loss_rmax:.2e}\n")
            print(f" ~> Avg Pressure Difference: {100*avg_p:.2f}% | Max Pressure Difference: {100*max_p:.2f}%")
            print(f" ~> True PDE Closed Loss: {loss_pde_c_true:.2e} | True PDE Open Loss: {loss_pde_o_true:.2e}")
            print("-" * 130)

    if (max_p < tol_dp) and (avg_change < tol_sep):
        r_t_point = float(safe_sep_eval(model_sep, jnp.array([[jnp.pi / 2.0]]), R)[0, 0])
        x_t_point = r_t_point
        mask_eq_o  = (eq_collocs_pool[:, 0] > r_t_point).reshape(-1, 1)
        eq_collocs = eq_collocs_pool
        l_eq       = l_eq_pool * mask_eq_o
        print(f"\nSEPARATRIX CONVERGED AT CYCLE {cycle}! T-point at {x_t_point:.4f}. Freezing geometry.")
        break

    if cycle == max_cycles:
        print(f"\nReached {max_cycles} geometry cycles. Freezing separatrix here and moving to Phase 2.")

    tack = time.time()
    times.append(tack - tick)
    print(f"Cycle {cycle} time elapsed: {times[-1] / 60:.2f} minutes.")

savedir = os.path.join(models_dir, experiment_name)
save_pinn_models(model_c, model_o, model_sep, save_dir=savedir)

logger.close()
print(f"\nExperiment '{experiment_name}' completed successfully.")
