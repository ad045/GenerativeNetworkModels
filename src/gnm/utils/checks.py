r"""Validation functions for network matrices.

This module provides functions to verify that network matrices conform to expected
properties required for generative network modeling, such as symmetry and absence
of self-connections.
"""

from jaxtyping import Float, jaxtyped
from typeguard import typechecked
import torch


@jaxtyped(typechecker=typechecked)
def binary_checks(matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"], device=None):
    r"""Check that matrices satisfy binary network constraints.

    Validates that the provided adjacency matrices conform to the expected properties
    for binary networks:

    1. All values are either 0 or 1 (matrices are binary)
    2. Matrices are symmetric (undirected)
    3. No self-connections (zeros on the diagonal)

    Args:
        matrices:
            Adjacency matrices to check with shape [num_networks, num_nodes, num_nodes]

    Raises:
        AssertionError: If any of the conditions are not met, with a descriptive error message

    Examples:
        >>> import torch
        >>> from gnm.utils import binary_checks
        >>> # Create a valid binary network
        >>> valid_network = torch.zeros((1, 3, 3))
        >>> valid_network[0, 0, 1] = 1
        >>> valid_network[0, 1, 0] = 1
        >>> binary_checks(valid_network)  # No error
        >>>
        >>> # Invalid binary network with non-binary values
        >>> non_binary_network = torch.zeros((1, 3, 3))
        >>> non_binary_network[0, 0, 1] = 0.5
        >>> non_binary_network[0, 1, 0] = 0.5
        >>> binary_checks(non_binary_network)  # Raises AssertionError: "Matrices must be binary"
        >>>
        >>> # Invalid binary network which is not symmetric
        >>> non_symmetric_network = torch.zeros((1, 3, 3))
        >>> non_symmetric_network[0, 0, 1] = 1
        >>> binary_checks(non_symmetric_network)  # Raises AssertionError: "Matrices must be symmetric"

    See Also:
        - [`utils.weighted_checks`][gnm.utils.weighted_checks]: For validating weighted networks
        - [`defaults.get_binary_network`][gnm.defaults.get_binary_network]: For loading pre-validated binary networks
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
def weighted_checks(matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"], device=None):
    r"""Check that matrices satisfy weighted network constraints.

    Validates that the provided weight matrices conform to the expected properties
    for weighted networks:

    1. All values are non-negative
    2. Matrices are symmetric (undirected)
    3. No self-connections (zeros on the diagonal)

    Args:
        matrices:
            Weight matrices to check with shape [num_networks, num_nodes, num_nodes]

    Raises:
        AssertionError: If any of the conditions are not met, with a descriptive error message

    Examples:
        >>> import torch
        >>> from gnm.utils import weighted_checks
        >>> # Create a valid weighted network
        >>> valid_network = torch.zeros((1, 3, 3))
        >>> valid_network[0, 0, 1] = 0.5
        >>> valid_network[0, 1, 0] = 0.5
        >>> weighted_checks(valid_network)  # No error
        >>>
        >>> # Invalid weighted network with negative values
        >>> negative_network = torch.zeros((1, 3, 3))
        >>> negative_network[0, 0, 1] = -0.5
        >>> negative_network[0, 1, 0] = -0.5
        >>> weighted_checks(negative_network)  # Raises AssertionError: "Matrices must be non-negative"
        >>>
        >>> # Invalid weighted network which is self-connected
        >>> self_connected_network = torch.zeros((1, 3, 3))
        >>> self_connected_network[0, 0, 0] = 1
        >>> weighted_checks(self_connected_network)  # Raises AssertionError: "Matrices must not be self-connected"

    See Also:
        - [`utils.binary_checks`][gnm.utils.binary_checks]: For validating binary networks
        - [`defaults.get_weighted_network`][gnm.defaults.get_weighted_network]: For loading pre-validated weighted networks
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
