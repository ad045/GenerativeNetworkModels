r"""Control networks for network evaluation and comparison."""

import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
import numpy as np
import networkx as nx
from tqdm import tqdm
from typing import Optional


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


@jaxtyped(typechecker=typechecked)
def generate_random_networks(
    num_nodes: int, 
    density: Optional[Float], 
    seed: int, 
    n: int = 1, 
    weighted: bool = False
) -> Float[torch.Tensor, "n num_nodes num_nodes"]:
    """Create a random graph with the given number of nodes and density.

    Args:
        num_nodes (int): Number of nodes in the graph.
        density (float): Density of the graph (between 0 and 1).
        seed (int): Random seed for reproducibility.
        n (int): Number of graphs to create.
        weighted (bool): If True, create a weighted graph.

    Returns:
        Tensor: Adjacency matrices of shape (n, num_nodes, num_nodes)
    """

    torch.manual_seed(seed)

    graphs = torch.bernoulli(torch.full((n, num_nodes, num_nodes), density)).int()

    # Make symmetric, no self-loops
    graphs = torch.triu(graphs, diagonal=1)
    graphs = graphs + graphs.transpose(1, 2)

    if weighted:
        weights = torch.rand(n, num_nodes, num_nodes)
        weights = torch.triu(weights, diagonal=1)
        weights = weights + weights.transpose(1, 2)
        graphs = graphs * weights

    return graphs

@jaxtyped(typechecker=typechecked)
def simulate_random_graph_clustering(
    num_nodes: int,
    n_iter: int = 100,
    density: Optional[Float] = None,
    weighted: bool = False,
) -> tuple[float, float]:
    r"""
    Simulate random graphs and compute average clustering coefficient and path length.

    This function generates random graphs with a specified density and computes the
    average clustering coefficient and average shortest path length for the generated graphs.
    It supports both binary and weighted graphs.

    Args:
        num_nodes (int): Number of nodes in the graph.
        n_iter (int): Number of random graphs to generate. Default is 100.
        density (float): Density of the graph (between 0 and 1). Must be provided.
        weighted (bool): If True, compute metrics for weighted graphs. Default is False.

    Returns:
        tuple[float, float]: A tuple containing:
            - Average shortest path length across the random graphs.
            - Average clustering coefficient across the random graphs.

    Examples:
        >>> from gnm.utils import simulate_random_graph_clustering
        >>> length_mean, clustering_mean = simulate_random_graph_clustering(
        ...     num_nodes=10, n_iter=50, density=0.2, weighted=False
        ... )
        >>> print(length_mean, clustering_mean)

    Notes:
        - For weighted graphs, edge weights are randomly assigned.
        - The clustering coefficients are normalized to the range [0, 1].

    Raises:
        AssertionError: If `density` is not provided or is not in the range (0, 1].
    """
    assert density is not None and 0 < density <= 1, "Density must be greater than 0 and less than or equal to 1."

    clustering_from_random_graph_list = []
    avg_degree_length_from_random_graph_list = []
    networks = generate_random_networks(num_nodes, density, seed=0, n=n_iter, weighted=weighted)

    for i in tqdm(range(n_iter), desc="Simulating random graphs"):
        # Create a random graph with the same number of nodes and edges
        random_graph = networks[i, :, :]
        random_graph_nx = nx.from_numpy_array(random_graph.cpu().numpy())

        if weighted:
            # Extract weights and set them as edge attributes
            edges = random_graph_nx.edges()
            weights = {(u, v): random_graph[u, v].item() for u, v in edges}
            nx.set_edge_attributes(random_graph_nx, weights, "weight")

            # Apply NX weighted measures
            betweenness = nx.betweenness_centrality(random_graph_nx, weight="weight", normalized=False)
            clustering_from_random_graph_list.append(np.mean(list(betweenness.values())))
            avg_degree_length_from_random_graph_list.append(nx.average_shortest_path_length(random_graph_nx, weight="weight"))
        else:
            # Apply NX binary measures
            betweenness = nx.betweenness_centrality(random_graph_nx, normalized=False)
            clustering_from_random_graph_list.append(np.mean(list(betweenness.values())))
            avg_degree_length_from_random_graph_list.append(nx.average_shortest_path_length(random_graph_nx))

    clust_min = np.min(clustering_from_random_graph_list)
    clust_max = np.max(clustering_from_random_graph_list)
    clustering_from_random_graph_list = np.array(clustering_from_random_graph_list)

    if clust_max != clust_min:
        clustering_from_random_graph_list = (clustering_from_random_graph_list - clust_min) / (clust_max - clust_min)
    else:
        clustering_from_random_graph_list = clustering_from_random_graph_list * 0

    length_random_mean = np.mean(avg_degree_length_from_random_graph_list)
    clustering_random_mean = np.mean(clustering_from_random_graph_list)

    print(f"Average random graph clustering coefficient: {clustering_random_mean}")
    print(f"Average random graph path length: {length_random_mean}")

    return length_random_mean, clustering_random_mean