from jaxtyping import Float, jaxtyped
from typeguard import typechecked
import torch


@jaxtyped(typechecker=typechecked)
def node_strenghts(
    adjacency_matrix: Float[torch.Tensor, "*batch num_nodes num_nodes"]
) -> Float[torch.Tensor, "*batch num_nodes"]:
    return adjacency_matrix.sum(dim=-1)


@jaxtyped(typechecker=typechecked)
def binary_clustering_coefficients(
    adjacency_matrix: Float[torch.Tensor, "*batch num_nodes num_nodes"]
) -> Float[torch.Tensor, "*batch num_nodes"]:
    degrees = adjacency_matrix.sum(dim=-1)
    number_of_pairs = degrees * (degrees - 1)

    number_of_triangles = torch.diagonal(
        torch.matmul(
            torch.matmul(adjacency_matrix, adjacency_matrix), adjacency_matrix
        ),
        dim1=-2,
        dim2=-1,
    )

    clustering_coefficients = number_of_triangles / number_of_pairs
    return clustering_coefficients


def weighted_clustering_coefficients(
    weight_matrices: Float[torch.Tensor, "*batch num_nodes num_nodes"]
) -> Float[torch.Tensor, "*batch num_nodes"]:
    """KS statistic comparing weighted clustering coefficient distributions between networks.

    Implements the Onnela et al. (2005) definition of weighted clustering, which uses
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
    # Check that the weight matrix is non-negative
    assert (weight_matrices >= 0).all(), "Weight matrix must be non-negative"
    # Check that the weight matrix is symmetric
    assert (
        weight_matrices == weight_matrices.transpose(-2, -1)
    ).all(), "Weight matrices must be symmetric"

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
