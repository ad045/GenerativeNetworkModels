import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from ot import emd2

from .evaluation_base import EvaluationCriterion


class WassersteinDistance:
    def __init__(self, criterion: EvaluationCriterion, p: float = 1.0):
        self.criterion = criterion
        self.p = p

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self,
        synthetic_matrices: Float[
            torch.Tensor, "num_synthetic_networks num_nodes num_nodes"
        ],
        real_matrices: Float[torch.Tensor, "num_real_networks num_nodes num_nodes"],
    ) -> float:
        num_synthetic_networks = synthetic_matrices.shape[0]
        num_real_networks = real_matrices.shape[0]

        # Compute distance matrix
        distances = self.criterion(synthetic_matrices, real_matrices)
        distances_p = distances**self.p

        a = torch.ones(num_synthetic_networks) / num_synthetic_networks
        b = torch.ones(num_real_networks) / num_real_networks

        wasserstein_p = emd2(a.numpy(), b.numpy(), distances_p.numpy())

        # Take p-th root
        wasserstein = wasserstein_p ** (1 / self.p)

        return float(wasserstein)
