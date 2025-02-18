from jaxtyping import Float, jaxtyped
from typeguard import typechecked
import torch


@jaxtyped(typechecker=typechecked)
def binary_checks(matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]):
    """Check that the matrices are binary, symmetric, and not self-connected.

    Args:
        matrices:
            Adjacency matrices to check of shape.
    """
    # Check that the matrices are binary:
    assert torch.all((matrices == 0) | (matrices == 1)), "Matrices must be binary"
    # Check that the matrices are symmetric:
    assert torch.allclose(
        matrices, matrices.transpose(-1, -2)
    ), "Matrices must be symmetric"
    # Check that the matrices are not self-connected:
    assert torch.all(
        matrices.diagonal(dim1=-2, dim2=-1) == 0
    ), "Matrices must not be self-connected"


@jaxtyped(typechecker=typechecked)
def weighted_checks(matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]):
    """Checks that matrices are non-negative, symmetric, and not self-connected.

    Args:
        matrices:
            Weight matrices to check of shape.
    """
    # Check that the matrices are non-negative:
    assert torch.all(matrices >= 0), "Matrices must be non-negative"
    # Check that the matrices are symmetric:
    assert torch.allclose(
        matrices, matrices.transpose(-1, -2)
    ), "Matrices must be symmetric"
    # Check that the matrices are not self-connected:
    assert torch.all(
        matrices.diagonal(dim1=-2, dim2=-1) == 0
    ), "Matrices must not be self-connected"
