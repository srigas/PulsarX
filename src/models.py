import jax.numpy as jnp

from flax import nnx

from jaxkan.layers.Dense import DenseLayer
from jaxkan.models.utils import RFFEmbedder
from jaxkan.models.RGAKAN import RGAKAN


class ModifiedMLP(nnx.Module):
    """Modified MLP, see https://doi.org/10.1137/20M1318043.

    Args:
        n_in: Input dimension.
        n_out: Output dimension.
        n_hidden: Width of all hidden layers and the RFF embedding.
        num_layers: Number of hidden layers (not counting input/output).
        rff_std: Standard deviation for the RFF embedding.
        seed: Random seed.
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int, num_layers: int,
                 rff_std: float = 1.0, seed: int = 42):

        layer_configs = {'RWF': {"mean": 1.0, "std": 0.1}, 'seed': seed}

        self.FE = RFFEmbedder(std=rff_std, n_in=n_in, embed_dim=n_hidden, seed=seed)
        self.U = DenseLayer(n_in=n_hidden, n_out=n_hidden, activation=nnx.tanh, **layer_configs)
        self.V = DenseLayer(n_in=n_hidden, n_out=n_hidden, activation=nnx.tanh, **layer_configs)
        self.layers = nnx.List([
            DenseLayer(n_in=n_hidden, n_out=n_hidden, activation=nnx.tanh, **layer_configs)
        ])
        for _ in range(num_layers - 1):
            self.layers.append(
                DenseLayer(n_in=n_hidden, n_out=n_hidden, activation=nnx.tanh, **layer_configs)
            )
        self.layers.append(
            DenseLayer(n_in=n_hidden, n_out=n_out, activation=None, **layer_configs)
        )

    def __call__(self, x):
        """Forward pass.

        Args:
            x: Input array of shape (batch, n_in).

        Returns:
            Output array of shape (batch, n_out).
        """
        x = self.FE(x)
        u = self.U(x)
        v = self.V(x)
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx != len(self.layers) - 1:
                x = x * u + (1 - x) * v
        return x


class ModelClosed(nnx.Module):
    """PINN network for the closed-line region.

    Args:
        n_hidden: Hidden layer width.
        num_blocks: Number of RGAKAN blocks.
        D: KAN grid size parameter.
        sine_D: Sine embedding dimension.
        alpha: RGAKAN α parameter.
        beta: RGAKAN β parameter.
        seed: Random seed.
    """

    def __init__(self, n_hidden: int, num_blocks: int, D: int, sine_D: int,
                 alpha: float, beta: float, seed: int = 42):
        self.nn = RGAKAN(
            n_in=2, n_out=1, n_hidden=n_hidden, num_blocks=num_blocks,
            flavor='exact', D=D, init_scheme={'type': 'glorot_fine'},
            alpha=alpha, beta=beta, ref=None, period_axes=None,
            rff_std=None, sine_D=sine_D, seed=seed,
        )

    def __call__(self, x):
        """Forward pass.

        Args:
            x: Input array of shape (batch, 2) with columns [r, θ].

        Returns:
            Ψ predictions of shape (batch, 1).
        """
        theta = x[:, 1:2]
        return (jnp.sin(theta) ** 2) * self.nn(x)


class ModelOpen(nnx.Module):
    """PINN network for the open-line region.

    Args:
        Psi_S: Flux value at the separatrix (boundary condition anchor).
        n_hidden: Hidden layer width.
        num_blocks: Number of RGAKAN blocks.
        D: KAN grid size parameter.
        sine_D: Sine embedding dimension.
        alpha: RGAKAN α parameter.
        beta: RGAKAN β parameter.
        seed: Random seed.
    """

    def __init__(self, Psi_S: float, n_hidden: int, num_blocks: int, D: int,
                 sine_D: int, alpha: float, beta: float, seed: int = 42):
        self.Psi_S = Psi_S
        self.nn = RGAKAN(
            n_in=2, n_out=2, n_hidden=n_hidden, num_blocks=num_blocks,
            flavor='exact', D=D, init_scheme={'type': 'glorot_fine'},
            alpha=alpha, beta=beta, ref=None, period_axes=None,
            rff_std=None, sine_D=sine_D, seed=seed,
        )

    def __call__(self, x):
        """Forward pass.

        Args:
            x: Input array of shape (batch, 2) with columns [r, θ].

        Returns:
            Tuple (Ψ, I), each of shape (batch, 1).
        """
        theta = x[:, 1:2]
        raw_out = self.nn(x)
        raw_psi, raw_I = raw_out[:, 0:1], raw_out[:, 1:2]
        psi_o = self.Psi_S * (1.0 - jnp.cos(theta)) + (jnp.sin(theta) ** 2) * raw_psi
        I_o = (jnp.sin(theta) ** 2) * raw_I
        return psi_o, I_o


class SymmetricSeparatrix(nnx.Module):
    """Separatrix network that enforces equatorial symmetry.

    Args:
        n_hidden: Hidden layer width.
        num_layers: Number of hidden layers.
        rff_std: Standard deviation for the RFF embedding.
        seed: Random seed.
    """

    def __init__(self, n_hidden: int, num_layers: int, rff_std: float = 1.0, seed: int = 42):
        self.nn = ModifiedMLP(n_in=1, n_out=1, n_hidden=n_hidden,
                              num_layers=num_layers, rff_std=rff_std, seed=seed)

    def __call__(self, x):
        """Forward pass.

        Args:
            x: Input array of shape (batch, 1) containing θ values.

        Returns:
            Predicted separatrix radius r of shape (batch, 1).
        """
        return self.nn(x ** 2)


class SeparatrixModel(nnx.Module):
    """Separatrix network without symmetry constraints.

    Args:
        n_hidden: Hidden layer width.
        num_layers: Number of hidden layers.
        rff_std: Standard deviation for the RFF embedding.
        seed: Random seed.
    """

    def __init__(self, n_hidden: int, num_layers: int, rff_std: float = 1.0, seed: int = 42):
        self.nn = ModifiedMLP(n_in=1, n_out=1, n_hidden=n_hidden,
                              num_layers=num_layers, rff_std=rff_std, seed=seed)

    def __call__(self, x):
        """Forward pass.

        Args:
            x: Input array of shape (batch, 1) containing θ values.

        Returns:
            Predicted separatrix radius r of shape (batch, 1).
        """
        return self.nn(x)
