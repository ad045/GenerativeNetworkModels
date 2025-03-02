r"""Control networks for network evaluation and comparison."""

import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
import numpy as np


@jaxtyped(typechecker=typechecked)
def get_control(
    matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
) -> Float[torch.Tensor, "num_networks num_nodes num_nodes"]:
    """Generate control networks by randomly permuting connections while preserving network properties.

    This function creates randomized versions of the input networks while maintaining:
    - The same number of connections (for binary networks) or weight distribution (for weighted networks)
    - Symmetry (undirected graph structure)
    - No self-connections (zeros on diagonal)

    Args:
        matrices:
            Input adjacency or weight matrices with shape [num_networks, num_nodes, num_nodes]

    Returns:
        Permuted control networks with the same shape as input matrices, preserving key properties

    Examples:
        >>> import torch
        >>> from gnm.utils import get_control
        >>> from gnm.defaults import get_binary_network
        >>> # Get a real network
        >>> real_network = get_binary_network()
        >>> # Generate a control with preserved properties
        >>> control_network = get_control(real_network)
        >>> # Check that control has same number of connections
        >>> real_network.sum() == control_network.sum()
        tensor(True)

    Notes:
        - For binary networks, this is equivalent to randomly rewiring all connections
        - For weighted networks, connection weights are preserved but redistributed
    """
    num_networks, num_nodes, _ = matrices.shape
    control_networks = torch.zeros_like(matrices)

    # Process each network in the batch
    for i in range(num_networks):
        network = matrices[i]

        # Get upper triangular indices (excluding diagonal)
        indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
        upper_values = network[indices[0], indices[1]]

        # Permute the upper triangular values
        perm_idx = torch.randperm(indices.shape[1])
        permuted_indices = (indices[0, perm_idx], indices[1, perm_idx])

        # Create a new network with permuted connections
        control = torch.zeros_like(network)
        control[permuted_indices[0], permuted_indices[1]] = upper_values

        # Ensure symmetry
        control = control + control.T
        control_networks[i] = control

    return control_networks
