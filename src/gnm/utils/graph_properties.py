from jaxtyping import Float, jaxtyped
from typeguard import typechecked
import torch
import networkx as nx
import numpy as np

from .checks import binary_checks, weighted_checks


@jaxtyped(typechecker=typechecked)
def node_strengths(
    adjacency_matrix: Float[torch.Tensor, "*batch num_nodes num_nodes"]
) -> Float[torch.Tensor, "*batch num_nodes"]:
    """Computes the node strengths (or equivalently the nodal degree) for each node in the network.

    Returns:
        Vector of node strengths for each node in the network.
    """
    return adjacency_matrix.sum(dim=-1)


@jaxtyped(typechecker=typechecked)
def binary_clustering_coefficients(
    adjacency_matrix: Float[torch.Tensor, "*batch num_nodes num_nodes"]
) -> Float[torch.Tensor, "*batch num_nodes"]:
    """Computes the clustering coefficients for each node in a binary network.

    The clustering coefficient for a node $i$ is computed as:
    $$
        c(i) = \\frac{ 2t_i }{ k_i (k_i - 1) },
    $$
    where $t_i$ is the number of (unordered) triangles around node $i$, and $k_i$ is the degree of node $i$.

    Returns:
        The clustering coefficients for each node.
    """
    binary_checks(adjacency_matrix)

    degrees = adjacency_matrix.sum(dim=-1)
    number_of_pairs = degrees * (degrees - 1)

    number_of_triangles = torch.diagonal(
        torch.matmul(
            torch.matmul(adjacency_matrix, adjacency_matrix), adjacency_matrix
        ),
        dim1=-2,
        dim2=-1,
    )

    clustering = torch.zeros_like(number_of_triangles)
    mask = number_of_pairs > 0
    clustering[mask] = 2 * number_of_triangles[mask] / number_of_pairs[mask]
    return clustering


@jaxtyped(typechecker=typechecked)
def weighted_clustering_coefficients(
    weight_matrices: Float[torch.Tensor, "*batch num_nodes num_nodes"]
) -> Float[torch.Tensor, "*batch num_nodes"]:
    """Implements the Onnela et al. (2005) definition of weighted clustering, which uses
    the geometric mean of triangle weights. For each node $i$, the clustering coefficient is:

    $$
    c(i) = \\frac{1}{k_i (k_i - 1)} \sum_{jk} (\hat{w}_{ij} \\times \hat{w}_{jk} \\times \hat{w}_{ki})^{1/3},
    $$

    where $k_i$ is the node strength of node $i$, and $\hat{w}_{ij}$ is the weight of the edge between nodes $i$ and $j$,
    *after* normalising by dividing by the maximum weight in the network.

    Args:
        weight_matrices: Batch of weighted adjacency matrices. Shape [*batch, num_nodes, num_nodes]
           Weights should be non-negative.

    Returns:
        Clustering coefficients for each node in each network. Shape [*batch, num_nodes]
    """
    weighted_checks(weight_matrices)

    # Get max weight for normalization (keeping batch dims)
    max_weight = weight_matrices.amax(dim=(-2, -1), keepdim=True)  # [*batch, 1, 1]

    # Normalize weights
    normalised_w = (weight_matrices / max_weight) ** (
        1 / 3
    )  # [*batch, num_nodes, num_nodes]

    # Get node strengths (sum of weights)
    node_strenghts = weight_matrices.sum(dim=-1)  # [*batch, num_nodes]

    # For each node u, compute the geometric mean of triangle weights:
    # (w_uv * w_vw * w_wu) ^ (1/3)
    triangles = torch.diagonal(
        torch.matmul(torch.matmul(normalised_w, normalised_w), normalised_w),
        dim1=-2,
        dim2=-1,
    )  # [*batch, num_nodes]

    # Compute denominator k * (k-1)
    denom = node_strenghts * (node_strenghts - 1)  # [*batch, num_nodes]

    # Handle division by zero - set clustering to 0 where k <= 1
    clustering = torch.zeros_like(triangles)
    mask = denom > 0
    clustering[mask] = triangles[mask] / denom[mask]

    return clustering


@jaxtyped(typechecker=typechecked)
def communicability(
    weight_matrix: Float[torch.Tensor, "*batch num_nodes num_nodes"]
) -> Float[torch.Tensor, "*batch num_nodes num_nodes"]:
    """Communicability optimisation criterion.
    To compute the communicability matrix, we go through the following steps:

    1. Compute the diagonal node strength matrix, $S_{ii} = \sum_j W_{ij}$ (plus a small constant to prevent division by zero).
    2. Compute the normalised weight matrix, $S^{-1/2} W S^{-1/2}$.
    3. Compute the communicability matrix by taking the matrix exponential, $\exp( S^{-1/2} W S^{-1/2} )$.
    """
    # Compute the node strengths, with a small constant addition to prevent division by zero.
    node_strengths = (
        0.5 * (weight_matrix.sum(dim=-1) + weight_matrix.sum(dim=-2)) + 1e-6
    )

    # Create diagonal matrix for each batch element
    batch_shape = weight_matrix.shape[:-2]
    num_nodes = weight_matrix.shape[-1]
    inv_sqrt_node_strengths = torch.zeros(
        *batch_shape, num_nodes, num_nodes, device=weight_matrix.device
    )

    # Set diagonal values for each batch element
    diag_indices = torch.arange(num_nodes)
    inv_sqrt_node_strengths[..., diag_indices, diag_indices] = 1.0 / torch.sqrt(
        node_strengths
    )

    # Compute the normalised weight matrix
    normalised_weight_matrix = torch.matmul(
        torch.matmul(inv_sqrt_node_strengths, weight_matrix), inv_sqrt_node_strengths
    )

    # Compute the communicability matrix
    communicability_matrix = torch.matrix_exp(normalised_weight_matrix)

    return communicability_matrix


def binary_betweenness_centrality(
    matrices: Float[torch.Tensor, "num_matrices num_nodes num_nodes"]
) -> Float[torch.Tensor, "num_matrices num_nodes"]:
    """Compute betweenness centrality for each node in the network.

    Args:
        matrices: Batch of adjacency matrices. Shape [num_matrices, num_nodes, num_nodes]

    Returns:
        torch.Tensor: Array of betweenness centralities for each node in each network.
    """
    graphs = [nx.from_numpy_array(matrix.cpu().numpy()) for matrix in matrices]
    betweenness_values = [
        np.array(list(nx.betweenness_centrality(g).values())) for g in graphs
    ]
    return torch.tensor(np.array(betweenness_values), dtype=matrices.dtype)
