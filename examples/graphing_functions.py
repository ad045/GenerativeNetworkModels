from jaxtyping import Float, Int, jaxtyped
from typeguard import typechecked
import torch
import numpy as np

@jaxtyped(typechecker=typechecked)
def local_clustering_coefficient(adj_matrix:Float[torch.Tensor, "num_nodes num_nodes"], directed:bool):
    """
    Compute the local clustering coefficient for each node in a graph.
    
    Parameters:
        adj_matrix (numpy.ndarray): Adjacency matrix of the graph (NxN).
        directed (bool): Whether the graph is directed or undirected.
    
    Returns:
        numpy.ndarray: Local clustering coefficient for each node.
    """
    N = adj_matrix.shape[0]
    clustering_coeffs = torch.zeros(N, device=adj_matrix.device)

    for i in range(N):
        neighbors = torch.where(adj_matrix[i] > 0)[0]
        k_i = len(neighbors)

        if k_i < 2:
            clustering_coeffs[i] = 0
            continue

        # Extract subgraph of neighbors
        subgraph = adj_matrix[neighbors][:, neighbors]
        actual_links = torch.sum(subgraph)

        if directed:
            max_links = k_i * (k_i - 1)  # Directed formula
        else:
            actual_links /= 2  # Since undirected counts edges twice
            max_links = (k_i * (k_i - 1)) / 2  # Undirected formula

        clustering_coeffs[i] = actual_links / max_links if max_links > 0 else 0

    return clustering_coeffs