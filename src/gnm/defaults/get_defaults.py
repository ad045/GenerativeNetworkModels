import torch
import os
from jaxtyping import Float, jaxtyped
from typing import Optional
from typeguard import typechecked

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def display_available_defaults():
    """prints available defaults that can be loaded in.
    There are available defaults for distance matrices, binary consensus networks,
    and weighted consensus networks.
    """

    print("=== Distance matrices ===")
    for file in os.listdir("distance_matrices"):
        print(file)

    print("=== Binary consensus networks ===")
    for file in os.listdir("binary_consensus_networks"):
        print(file)

    print("=== Weighted consensus networks ===")
    for file in os.listdir("weighted_consensus_networks"):
        print(file)


@jaxtyped(typechecker=typechecked)
def get_distance_matrix(
    name: Optional[str] = None, device: Optional[torch.device] = None
) -> Float[torch.Tensor, "num_nodes num_nodes"]:
    """Loads a default distance matrix.

    Available distance matrices are:

    1. [FILL IN LATER]

    Args:
        name:
            Name of the distance matrix to be loaded in.
            If unspecified, the [FILL IN LATER] distance matrix is loaded in.
        device:
            Device to load the distance matrix on.
            If unspecified, the device is automatically set to "cuda" if available,
            otherwise "cpu".

    Returns:
        The requested distance matrix as a torch tensor.
    """
    if device is None:
        device = DEVICE

    if name is None:
        name = "[FILL IN LATER]"

    return torch.load(f"distance_matrices/{name}.pt", map_location=device)


@jaxtyped(typechecker=typechecked)
def get_binary_consensus_network(
    name: Optional[str], device: Optional[torch.device] = None
) -> Float[torch.Tensor, "num_nodes num_nodes"]:
    """Loads a default binary consensus network.

    Available binary consensus matrices are:

    1. [FILL IN LATER]

    Args:
        name:
            Name of the binary consensus network to be loaded in.
            If unspecified, the [FILL IN LATER] binary consensus network is loaded in.
        device:
            Device to load the binary consensus network on.
            If unspecified, the device is automatically set to "cuda" if available,
            otherwise "cpu".

    Returns:
        The requested binary consensus network as a torch tensor.
    """
    if device is None:
        device = DEVICE

    if name is None:
        name = "[FILL IN LATER]"

    return torch.load(f"binary_consensus_networks/{name}.pt", map_location=device)


@jaxtyped(typechecker=typechecked)
def get_weighted_consensus_network(
    name: Optional[str], device: Optional[torch.device] = None
) -> Float[torch.Tensor, "num_nodes num_nodes"]:
    """Loads a default weighted consensus network.

    Available weighted consensus matrices are:

    1. [FILL IN LATER]

    Args:
        name:
            Name of the weighted consensus network to be loaded in.
            If unspecified, the [FILL IN LATER] weighted consensus network is loaded in.
        device:
            Device to load the weighted consensus network on.
            If unspecified, the device is automatically set to "cuda" if available,
            otherwise "cpu".

    Returns:
        The requested weighted consensus network as a torch tensor.
    """
    if device is None:
        device = DEVICE

    if name is None:
        name = "[FILL IN LATER]"

    return torch.load(f"weighted_consensus_networks/{name}.pt", map_location=device)
