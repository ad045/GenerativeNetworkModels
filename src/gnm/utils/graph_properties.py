r"""Graph theory metrics for analyzing network properties.

This module provides various metrics from graph theory for characterising network
structures in both binary and weighted networks. These metrics include node strengths,
clustering coefficients, communicability, and betweenness centrality.
"""

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
    r"""Compute the node strengths (or nodal degree) for each node in the network.

    For binary networks, this is equivalent to the node degree (number of connections).
    For weighted networks, this represents the sum of all edge weights connected to each node.

    Args:
        adjacency_matrix:
            Adjacency matrix (binary or weighted) with shape [*batch, num_nodes, num_nodes]

    Returns:
        Vector of node strengths for each node in the network with shape [*batch, num_nodes]

    Examples:
        >>> import torch
        >>> from gnm.utils import node_strengths
        >>> # Create a sample binary network
        >>> adj_matrix = torch.zeros(1, 4, 4)
        >>> adj_matrix[0, 0, 1] = 1
        >>> adj_matrix[0, 1, 0] = 1
        >>> adj_matrix[0, 1, 2] = 1
        >>> adj_matrix[0, 2, 1] = 1
        >>> strength = node_strengths(adj_matrix)
        >>> strength
        tensor([[1., 2., 1., 0.]])

    See Also:
        - [`evaluation.DegreeKS`][gnm.evaluation.DegreeKS]: Binary evaluation criterion which compares the distribution of node degrees between two binary networks.
        - [`evaluation.WeightedNodeStrengthKS`][gnm.evaluation.WeightedNodeStrengthKS]: Weighted evaluation criterion which compares the distribution of node strengths between two weighted networks.
        - [`evaluation.DegreeCorrelation`][gnm.evaluation.DegreeCorrelation]: Binary evaluation criterion which compares the correlations between the node degrees between two binary networks.
    """
    return adjacency_matrix.sum(dim=-1)


@jaxtyped(typechecker=typechecked)
def binary_clustering_coefficients(
    adjacency_matrix: Float[torch.Tensor, "*batch num_nodes num_nodes"]
) -> Float[torch.Tensor, "*batch num_nodes"]:
    r"""Compute the clustering coefficients for each node in a binary network.

    The clustering coefficient measures the degree to which nodes in a graph tend to cluster together.
    For a node i, it quantifies how close its neighbors are to being a complete subgraph (clique).

    The clustering coefficient for a node $i$ is computed as:
    $$
        c(i) = \\frac{ 2t_i }{ k_i (k_i - 1) },
    $$
    where $t_i$ is the number of (unordered) triangles around node $i$, and $k_i$ is the degree of node $i$.

    Args:
        adjacency_matrix:
            Binary adjacency matrix with shape [*batch, num_nodes, num_nodes]

    Returns:
        The clustering coefficients for each node with shape [*batch, num_nodes]

    Examples:
        >>> import torch
        >>> from gnm.utils import binary_clustering_coefficients
        >>> # Create a binary network with a triangle
        >>> adj_matrix = torch.zeros(1, 4, 4)
        >>> adj_matrix[0, 0, 1] = 1
        >>> adj_matrix[0, 1, 0] = 1
        >>> adj_matrix[0, 1, 2] = 1
        >>> adj_matrix[0, 2, 1] = 1
        >>> adj_matrix[0, 0, 2] = 1
        >>> adj_matrix[0, 2, 0] = 1
        >>> clustering = binary_clustering_coefficients(adj_matrix)
        >>> clustering
        tensor([[1., 1., 1., 0.]])

    See Also:
        - [`utils.weighted_clustering_coefficients`][gnm.utils.weighted_clustering_coefficients]: For calculating clustering coefficient in weighted networks.
        - [`evaluation.ClusteringKS`][gnm.evaluation.ClusteringKS]: Binary evaluation criterion which compares the distribution of clustering coefficients between two binary networks.
        - [`evaluation.ClusteringCorrelation`][gnm.evaluation.ClusteringCorrelation]: Binary evaluation criterion which compares the correlations between the clustering coefficients between two binary networks.
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

    # removed 2 * to match BCT output
    clustering[mask] = number_of_triangles[mask] / number_of_pairs[mask]
    return clustering


@jaxtyped(typechecker=typechecked)
def weighted_clustering_coefficients(
    weight_matrices: Float[torch.Tensor, "*batch num_nodes num_nodes"]
) -> Float[torch.Tensor, "*batch num_nodes"]:
    r"""Compute weighted clustering coefficients based on Onnela et al. (2005) definition.

    This implementation uses the geometric mean of triangle weights. For each node $i$,
    the clustering coefficient is:

    $$
    c(i) = \frac{1}{k_i (k_i - 1)} \sum_{jk} (\hat{w}_{ij} \times \hat{w}_{jk} \times \hat{w}_{ki})^{1/3},
    $$

    where $k_i$ is the node strength of node $i$, and $\hat{w}_{ij}$ is the weight of the edge between nodes $i$ and $j$,
    *after* normalising by dividing by the maximum weight in the network.

    Args:
        weight_matrices:
            Batch of weighted adjacency matrices with shape [*batch, num_nodes, num_nodes].
            Weights should be non-negative.

    Returns:
        Clustering coefficients for each node in each network with shape [*batch, num_nodes]

    Examples:
        >>> import torch
        >>> from gnm.utils import weighted_clustering_coefficients
        >>> # Create a weighted network with a triangle
        >>> weight_matrix = torch.zeros(1, 4, 4)
        >>> weight_matrix[0, 0, 1] = 0.5
        >>> weight_matrix[0, 1, 0] = 0.5
        >>> weight_matrix[0, 1, 2] = 0.8
        >>> weight_matrix[0, 2, 1] = 0.8
        >>> weight_matrix[0, 0, 2] = 0.6
        >>> weight_matrix[0, 2, 0] = 0.6
        >>> clustering = weighted_clustering_coefficients(weight_matrix)
        >>> clustering.shape
        torch.Size([1, 4])

    See Also:
        - [`utils.binary_clustering_coefficients`][gnm.utils.binary_clustering_coefficients]: For calculating clustering in binary networks.
        - [`evaluation.WeightedClusteringKS`][gnm.evaluation.WeightedClusteringKS]: Weighted evaluation criterion which compares the distribution of (weighted) clustering coefficients between two weighted networks.
    """
    weighted_checks(weight_matrices)

    # each triange to exponent of 1/3 for cube root norm
    normalised_w = torch.pow(weight_matrices, 1/3)

    # Get max weight for normalization (keeping batch dims)
    max_weight = normalised_w.amax(dim=(-2, -1), keepdim=True)  # [*batch, 1, 1]
    normalised_w = normalised_w / max_weight 

    # For each node u, compute the geometric mean of triangle weights:
    # (w_uv * w_vw * w_wu) ^ (1/3)
    triangles = torch.diagonal(
        torch.matmul(torch.matmul(normalised_w, normalised_w), normalised_w),
        dim1=-2,
        dim2=-1,
    ) # [*batch, num_nodes]

    # Get node strengths (sum of weights)
    degree = torch.sum(weight_matrices > 0, dim=-1)  # [*batch, num_nodes]

    # Compute denominator k * (k-1) (k = degree)
    denom = degree * (degree - 1)  # [*batch, num_nodes]

    # Handle division by zero - set clustering to 0 where k <= 1
    clustering = torch.zeros_like(triangles)
    mask = denom > 0
    clustering[mask] = triangles[mask] / denom[mask]

    return clustering


@jaxtyped(typechecker=typechecked)
def communicability(
    weight_matrix: Float[torch.Tensor, "*batch num_nodes num_nodes"]
) -> Float[torch.Tensor, "*batch num_nodes num_nodes"]:
    r"""Compute the communicability matrix for a network.

    Communicability measures the ease of information flow between nodes, taking into
    account all possible paths between them. It's based on the matrix exponential of
    the normalized adjacency matrix.

    To compute the communicability matrix, we go through the following steps:

    1. Compute the diagonal node strength matrix, $S_{ii} = \sum_j W_{ij}$ (plus a small constant to prevent division by zero).
    2. Compute the normalised weight matrix, $S^{-1/2} W S^{-1/2}$.
    3. Compute the communicability matrix by taking the matrix exponential, $\exp( S^{-1/2} W S^{-1/2} )$.

    Args:
        weight_matrix:
            Weighted adjacency matrix with shape [*batch, num_nodes, num_nodes]

    Returns:
        Communicability matrix with shape [*batch, num_nodes, num_nodes]

    Examples:
        >>> import torch
        >>> from gnm.utils import communicability
        >>> # Create a simple weighted network
        >>> weight_matrix = torch.zeros(1, 3, 3)
        >>> weight_matrix[0, 0, 1] = 0.5
        >>> weight_matrix[0, 1, 0] = 0.5
        >>> weight_matrix[0, 1, 2] = 0.8
        >>> weight_matrix[0, 2, 1] = 0.8
        >>> comm_matrix = communicability(weight_matrix)
        >>> comm_matrix.shape
        torch.Size([1, 3, 3])

    See Also:
        - [`weight_criteria.Communicability`][gnm.weight_criteria.Communicability]: weight optimisation criterion which minimises total communicability.
        - [`weight_criteria.NormalisedCommunicability`][gnm.weight_criteria.NormalisedCommunicability]: weight optimisation criterion which minimises total communicability, divided by the maximum communicability.
        - [`weight_criteria.DistanceWeightedCommunicability`][gnm.weight_criteria.DistanceWeightedCommunicability]: weight optimisation criterion which minimises total communicability, weighted by the distance between nodes.
        - [`weight_criteria.NormalisedDistanceWeightedCommunicability`][gnm.weight_criteria.NormalisedDistanceWeightedCommunicability]: weight optimisation criterion which minimises total communicability, weighted by the distance between nodes and divided by the maximum distance-weighted communicability.
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
    r"""Compute betweenness centrality for each node in binary networks.

    Betweenness centrality quantifies the number of times a node acts as a bridge along
    the shortest path between two other nodes. It identifies nodes that control information
    flow in a network.

    This function uses NetworkX for calculation and is intended for binary networks.

    Args:
        matrices:
            Batch of binary adjacency matrices with shape [num_matrices, num_nodes, num_nodes]

    Returns:
        Array of betweenness centralities for each node in each network with shape [num_matrices, num_nodes]

    Examples:
        >>> import torch
        >>> from gnm.utils import binary_betweenness_centrality
        >>> # Create a simple binary network
        >>> adj_matrix = torch.zeros(1, 4, 4)
        >>> adj_matrix[0, 0, 1] = 1
        >>> adj_matrix[0, 1, 0] = 1
        >>> adj_matrix[0, 1, 2] = 1
        >>> adj_matrix[0, 2, 1] = 1
        >>> adj_matrix[0, 2, 3] = 1
        >>> adj_matrix[0, 3, 2] = 1
        >>> betweenness = binary_betweenness_centrality(adj_matrix)
        >>> betweenness.shape
        torch.Size([1, 4])

    Notes:
        This function converts PyTorch tensors to NumPy arrays for NetworkX processing,
        then converts the results back to PyTorch tensors. For large networks or batches,
        this may be computationally expensive.

    See Also:
        - [`evaluation.BetweennessKS`][gnm.evaluation.BetweennessKS]: Binary evaluation criterion which compares the distribution of betweenness centralities between two binary networks.
    """
    graphs = [nx.from_numpy_array(matrix.cpu().numpy()) for matrix in matrices]
    betweenness_values = [
        np.array(list(nx.betweenness_centrality(g).values())) for g in graphs
    ]
    return torch.tensor(np.array(betweenness_values), dtype=matrices.dtype)



@jaxtyped(typechecker=typechecked)
def binary_betweenness_centrality(connectome: Float[torch.Tensor, "*batch num_nodes num_nodes"], device=None):
    if device is None:
        device = connectome.device

    batch_size = connectome.shape[0]
    num_nodes = connectome.shape[-1]  

    # Identity matrix over batches
    single_identity = torch.eye(num_nodes, device=device)
    batch_identity = single_identity.repeat(batch_size, 1, 1)  # I

    num_shortest_paths = connectome.clone().to(device)  # NPd
    num_shortest_paths_length_d = connectome.clone().to(device)  # NSPd
    num_shortest_paths_lengths_any = connectome.clone().to(device)  # NSP
    length_shortest_path = connectome.clone().to(device)  # L

    # Self-connections have a shortest path of 1
    num_shortest_paths_lengths_any[batch_identity.bool()] = 1
    length_shortest_path[batch_identity.bool()] = 1

    max_distance = num_nodes  # Maximum possible path length (in worst case, it's num_nodes - 1)

    for d in range(2, max_distance + 1):
        num_shortest_paths = torch.bmm(num_shortest_paths, connectome)
        
        num_shortest_paths_length_d = torch.where(
            length_shortest_path == 0, num_shortest_paths, torch.zeros_like(num_shortest_paths)
        )

        # Update shortest path counts and lengths
        num_shortest_paths_lengths_any += num_shortest_paths_length_d
        length_shortest_path += d * (num_shortest_paths_length_d != 0)

        # Break if no new shortest paths are found
        if torch.all(num_shortest_paths_length_d == 0):
            break

    # Assign infinite length to disconnected edges
    length_shortest_path = torch.where(length_shortest_path == 0, torch.inf, length_shortest_path)
    length_shortest_path[batch_identity.bool()] = 0

    # Assign 1 to disconnected paths
    num_shortest_paths_lengths_any = torch.where(num_shortest_paths_lengths_any == 0, 1, num_shortest_paths_lengths_any)

    # Initialize dependency matrix
    dependency = torch.zeros((batch_size, num_nodes, num_nodes), device=device)

    # Compute graph diameter
    diameter = d - 1

    for d in range(diameter, 1, -1):
        DPd1 = torch.bmm(
            ((length_shortest_path == d).float() * (1 + dependency) / (num_shortest_paths_lengths_any + 1e-10)),
            connectome.transpose(-1, -2)
        ) * ((length_shortest_path == (d - 1)).float() * num_shortest_paths_lengths_any)

        dependency += DPd1

    return dependency.sum(dim=1)  # Sum over node dependencies