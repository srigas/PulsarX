import os
from pathlib import Path

import flax.nnx as nnx
import orbax.checkpoint as ocp

from .models import ModelClosed, ModelOpen, SymmetricSeparatrix, SeparatrixModel


def save_pinn_models(model_c, model_o, model_sep, save_dir="saved_pinn_models"):
    """Saves all three PINN models to disk using Orbax.

    Each model is stored in its own subdirectory under save_dir.

    Args:
        model_c: PINN model for the closed-line region.
        model_o: PINN model for the open-line region.
        model_sep: Separatrix network.
        save_dir: Root directory for checkpoint storage.
    """
    base_path = Path(save_dir).resolve()
    checkpointer = ocp.StandardCheckpointer()

    _, state_c   = nnx.split(model_c)
    _, state_o   = nnx.split(model_o)
    _, state_sep = nnx.split(model_sep)

    checkpointer.save(base_path / "model_c",   state_c,   force=True)
    checkpointer.save(base_path / "model_o",   state_o,   force=True)
    checkpointer.save(base_path / "model_sep", state_sep, force=True)

    checkpointer.wait_until_finished()

    print(f"All 3 models successfully saved to: {base_path}/")


def load_pinn_models(params_c: dict, params_o: dict, params_sep: dict,
                     load_dir="saved_pinn_models", sym_sep=False, seed=42):
    """Loads all three PINN models from disk.

    Concrete model instances are created first to obtain the graph definition
    and state structure required by Orbax for restoration.

    Args:
        params_c: Constructor keyword arguments for ModelClosed.
        params_o: Constructor keyword arguments for ModelOpen.
        params_sep: Constructor keyword arguments for the separatrix model.
        load_dir: Root directory containing the checkpoint subdirectories.
        sym_sep: If True, load a SymmetricSeparatrix; otherwise SeparatrixModel.
        seed: Fallback random seed injected if not present in the param dicts.

    Returns:
        Tuple (model_c, model_o, model_sep) with weights restored from disk.

    Raises:
        FileNotFoundError: If load_dir does not exist.
    """
    base_path = Path(load_dir).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Could not find directory: {base_path}")

    print(f"Loading models from {base_path}/...")
    checkpointer = ocp.StandardCheckpointer()

    config_c   = {**params_c,   "seed": params_c.get("seed", seed)}
    config_o   = {**params_o,   "seed": params_o.get("seed", seed)}
    config_sep = {**params_sep, "seed": params_sep.get("seed", seed)}

    model_c       = ModelClosed(**config_c)
    model_o       = ModelOpen(**config_o)
    model_sep_tmp = (SymmetricSeparatrix if sym_sep else SeparatrixModel)(**config_sep)

    graphdef_c,   abstract_state_c   = nnx.split(model_c)
    graphdef_o,   abstract_state_o   = nnx.split(model_o)
    graphdef_sep, abstract_state_sep = nnx.split(model_sep_tmp)

    state_c   = checkpointer.restore(base_path / "model_c",   abstract_state_c)
    state_o   = checkpointer.restore(base_path / "model_o",   abstract_state_o)
    state_sep = checkpointer.restore(base_path / "model_sep", abstract_state_sep)

    model_c   = nnx.merge(graphdef_c,   state_c)
    model_o   = nnx.merge(graphdef_o,   state_o)
    model_sep = nnx.merge(graphdef_sep, state_sep)

    print("Models loaded and ready for evaluation!")
    return model_c, model_o, model_sep
