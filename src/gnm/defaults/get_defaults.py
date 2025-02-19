import torch
import os
from jaxtyping import Float, jaxtyped
from typing import Optional
from typeguard import typechecked

from gnm.utils import binary_checks, weighted_checks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BASE_PATH = os.path.dirname(__file__)


def display_available_defaults():
    """prints available defaults that can be loaded in.
    There are available defaults for distance matrices, binary consensus networks,
    and weighted consensus networks.
    """

    print("=== Distance matrices ===")
    distance_matrices_path = os.path.join(BASE_PATH, "distance_matrices")
    for file in os.listdir(distance_matrices_path):
        print(file.split(".")[0])
    print("=== Coordinates ===")
    coordinates_path = os.path.join(BASE_PATH, "coordinates")
    for file in os.listdir(coordinates_path):
        print(file.split(".")[0])
    print("=== Binary networks ===")
    binary_consensus_networks_path = os.path.join(BASE_PATH, "binary_networks")
    for file in os.listdir(binary_consensus_networks_path):
        print(file.split(".")[0])
    print("=== Weighted networks ===")
    weighted_consensus_networks_path = os.path.join(BASE_PATH, "weighted_networks")
    for file in os.listdir(weighted_consensus_networks_path):
        print(file.split(".")[0])


@jaxtyped(typechecker=typechecked)
def get_distance_matrix(
    name: Optional[str] = None, device: Optional[torch.device] = None
) -> Float[torch.Tensor, "num_nodes num_nodes"]:
    """Loads a default distance matrix.

    Available distance matrices are:

    1. AAL_DISTANCES

    Args:
        name:
            Name of the distance matrix to be loaded in.
            If unspecified, the AAL_DISTANCES distance matrix is loaded in.
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
        name = "AAL_DISTANCES"

    distance_matrix = torch.load(
        os.path.join(BASE_PATH, f"distance_matrices/{name.split('.')[0].upper()}.pt"),
        map_location=device,
    )

    weighted_checks(distance_matrix.unsqueeze(0))

    return distance_matrix


@jaxtyped(typechecker=typechecked)
def get_coordinates(
    name: Optional[str] = None, device: Optional[torch.device] = None
) -> Float[torch.Tensor, "num_nodes 3"]:
    """Loads a default set of coordinates.

    Available coordinate sets are:

    1. AAL_COORDINATES

    Args:
        name:
            Name of the coordinates to be loaded in.
            If unspecified, the AAL_COORDINATES coordinates are loaded in.
        device:
            Device to load the coordinates on.
            If unspecified, the device is automatically set to "cuda" if available,
            otherwise "cpu".

    Returns:
        The requested coordinates as a torch tensor.
    """
    if device is None:
        device = DEVICE

    if name is None:
        name = "AAL_COORDINATES"

    return torch.load(
        os.path.join(BASE_PATH, f"coordinates/{name.split('.')[0].upper()}.pt"),
        map_location=device,
    )


@jaxtyped(typechecker=typechecked)
def get_binary_network(
    name: Optional[str] = None, device: Optional[torch.device] = None
) -> Float[torch.Tensor, "dataset_size num_nodes num_nodes"]:
    """Loads a default binary network.

    Available binary matrices are:

    1. CALM_BINARY_CONSENSUS

    Args:
        name:
            Name of the binary network to be loaded in.
            If unspecified, the CALM_BINARY_CONSENSUS binary network is loaded in.
        device:
            Device to load the binary network on.
            If unspecified, the device is automatically set to "cuda" if available,
            otherwise "cpu".

    Returns:
        The requested binary network as a torch tensor.
    """
    if device is None:
        device = DEVICE

    if name is None:
        name = "CALM_BINARY_CONSENSUS"

    binary_networks = torch.load(
        os.path.join(BASE_PATH, f"binary_networks/{name.split('.')[0].upper()}.pt"),
        map_location=device,
    )

    binary_checks(binary_networks)

    return binary_networks


@jaxtyped(typechecker=typechecked)
def get_weighted_network(
    name: Optional[str] = None, device: Optional[torch.device] = None
) -> Float[torch.Tensor, "dataset_size num_nodes num_nodes"]:
    """Loads a default weighted network.

    Available weighted matrices are:

    1. CALM_WEIGHTED_CONSENSUS

    Args:
        name:
            Name of the weighted network to be loaded in.
            If unspecified, the CALM_WEIGHTED_CONSENSUS weighted network is loaded in.
        device:
            Device to load the weighted network on.
            If unspecified, the device is automatically set to "cuda" if available,
            otherwise "cpu".

    Returns:
        The requested weighted network as a torch tensor.
    """
    if device is None:
        device = DEVICE

    if name is None:
        name = "CALM_WEIGHTED_CONSENSUS"

    weighted_networks = torch.load(
        os.path.join(BASE_PATH, f"weighted_networks/{name.split('.')[0].upper()}.pt"),
        map_location=device,
    )

    weighted_checks(weighted_networks)

    return weighted_networks
