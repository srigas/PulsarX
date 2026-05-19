# =============================================================================
#  config.py  —  Experiment configuration for the pulsar magnetosphere solver
# =============================================================================
#
# Each parameter maps to a list of values. run.py takes the Cartesian product
# of all lists and launches one run per combination.
#
# A single-element list means that parameter is fixed across all runs.
#
# =============================================================================

# Global random seed
SEED = [42]

# Stellar radius (R_*/R_LC)
R = [0.05]

# Polar-cap angle multiplier (θ_pc = MULTIPLIER · √R)
MULTIPLIER = [1.176]

# Floating-point precision
#   True  → 64-bit / double precision
#   False → 32-bit / single precision
DOUBLE_PRECISION = [True]

# Separatrix update step size β
BETA_SEP = [0.025]

# Convergence tolerances as (tol_sep, tol_dp) pairs.
#   tol_sep : max relative separatrix change threshold
#   tol_dp  : max normalised pressure imbalance threshold
TOL = [(15e-4, 1e-1)]

# Training schedule: how many epochs to run per geometry cycle.
# Each entry is a dict mapping cycle_index → epochs_per_cycle.
# The last entry in the dict applies to all subsequent cycles.
CYCLE_CONFIG = [{0: 20_000, 6: 10_000}]

# Separatrix model symmetry
#   True  → SymmetricSeparatrix (input is θ², enforces even symmetry)
#   False → SeparatrixModel (unconstrained; input is θ)
SYM_SEP = [False]

# Model α and β parameters — passed to both closed and open networks
ALPHA = [1.0]
BETA  = [1.0]
