# PulsarX: Axisymmetric Pulsar Magnetosphere PINN Solver

A Physics-Informed Neural Network (PINN) solver for the axisymmetric pulsar magnetosphere problem in spherical coordinates. The solver trains a pair of neural networks, one for the closed-line region and one for the open-line region, to satisfy the pulsar equation, iteratively updating the separatrix geometry until both the pressure balance condition and the geometry convergence criteria are met.

---

## Requirements & Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv env
source env/bin/activate        # Linux / macOS
# env\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

> **GPU note** — `jaxkan[gpu]` (listed in `requirements.txt`) installs the
> GPU-enabled JAX backend. On a CPU-only machine replace it with `jaxkan`.

---

## Project Structure

```
pulsarx/
├── config.py          # ← Edit this to configure your experiments
├── main.py            # Single-experiment runner
├── run.py             # Multi-experiment sweep orchestrator
├── requirements.txt
└── src/
    ├── collocs.py     # Collocation-point generators
    ├── io.py          # Model save / load utilities
    ├── logger.py      # Experiment logger (stdout tee)
    ├── models.py      # Neural network definitions
    ├── pde.py         # PDE residuals and the main training step
    └── separatrix.py  # Separatrix network utilities and domain splitting
```

Output directories are created automatically on the first run:

```
pulsarx/
├── logs/              # Per-experiment logs
│   └── <experiment>/
│       └── log.txt
├── model_ckpt/        # Saved model weights, one sub-folder per experiment
│   └── <experiment>/
│       ├── model_c/
│       ├── model_o/
│       └── model_sep/
```

---

## Configuration (`config.py`)

Open `config.py` and set **lists** of values for the parameters you want to explore. `run.py` forms the **Cartesian product** of all lists and runs one experiment per combination.

```python
SEED             = [42]                    # global random seed
R                = [0.25]                  # stellar radius  (R_*/R_LC)
MULTIPLIER       = [1.176]                 # θ_pc = MULTIPLIER · √R
DOUBLE_PRECISION = [False]                 # True → 64-bit, False → 32-bit
BETA_SEP         = [0.025]                 # separatrix update step size β
TOL              = [(15e-4, 1e-1)]         # (tol_sep, tol_dp) convergence pair
CYCLE_CONFIG     = [{0: 20000, 4: 10000}]  # training schedule per cycle
SYM_SEP          = [False]                 # True → SymmetricSeparatrix
ALPHA            = [1.0]                   # model α  (closed & open networks)
BETA             = [1.0]                   # model β  (closed & open networks)
```

---

## Running Experiments

### Single run

Run one experiment directly with `main.py`, passing parameters explicitly:

```bash
python main.py \
    --seed 42 \
    --R 0.25 \
    --multiplier 1.176 \
    --double-precision false \
    --beta-sep 0.025 \
    --tol-sep 15e-4 \
    --tol-dp 1e-1 \
    --cycle-config '{"0": 20000, "4": 10000}' \
    --sym-sep false \
    --alpha 1.0 \
    --beta 1.0
```

All flags are optional and fall back to the defaults shown above.

```bash
python main.py --help    # show all options with defaults
```

### Sweep over multiple configurations

1. Edit `config.py` to set the desired value lists.
2. Run the orchestrator:

```bash
python run.py
```

`run.py` prints a summary at the end showing which runs succeeded or failed and the total wall-clock time.

Runs are executed **sequentially** (one at a time), which is the safe default for GPU workloads where concurrent processes compete for memory.

---

## Citation

If you find this code useful in your research, please cite the paper that introduced this framework:

```bibtex
@article{pulsarx,
  author        = {Rigas, Spyros and Contopoulos, Ioannis and Alexandridis, Georgios and Nathanail, Antonios},
  title         = {An adaptive framework for the axisymmetric pulsar magnetosphere using physics-informed Kolmogorov-Arnold networks},
  year          = {2026},
  eprint        = {2606.10686},
  archivePrefix = {arXiv},
  primaryClass  = {physics.comp-ph},
  url           = {https://arxiv.org/abs/2606.10686}, 
}
```
