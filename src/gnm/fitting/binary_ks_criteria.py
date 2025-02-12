import torch
import networkx as nx
import numpy as np
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from .evaluation_base import KSCriterion


class DegreeKS(KSCriterion):
    """KS statistic comparing degree distributions between networks."""

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "_"]:
        """Compute degree for each node in the network.

        Args:
            matrix:
                Adjacency matrix of the network

        Returns:
            Vector of node degrees
        """
        return matrix.sum(dim=1)


class ClusteringKS(KSCriterion):
    """KS statistic comparing clustering coefficient distributions between networks."""

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "_"]:
        """Compute clustering coefficient for each node in the network.

        Args:
            matrix:
                Adjacency matrix of the network

        Returns:
            Vector of clustering coefficients
        """
        # Convert to networkx for clustering calculation
        G = nx.from_numpy_array(matrix.detach().cpu().numpy())
        clustering = nx.clustering(G)
        # Convert dict to list preserving node order
        return torch.tensor([clustering[i] for i in range(len(clustering))])


class BetweennessKS(KSCriterion):
    """KS statistic comparing betweenness centrality distributions between networks."""

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "_"]:
        """Compute betweenness centrality for each node in the network.

        Args:
            matrix: Adjacency matrix of the network

        Returns:
            torch.Tensor: Vector of betweenness centralities
        """
        # Convert to networkx for betweenness calculation
        G = nx.from_numpy_array(matrix.detach().cpu().numpy())
        betweenness = nx.betweenness_centrality(G)
        # Convert dict to list preserving node order
        return torch.tensor([betweenness[i] for i in range(len(betweenness))])


class EdgeLengthKS(KSCriterion):
    """KS statistic comparing edge length distributions between networks."""

    def __init__(self, distance_matrix: Float[torch.Tensor, "num_nodes num_nodes"]):
        """Initialize with a distance matrix.

        Args:
            distance_matrix: Matrix of distances between nodes
        """
        self.distance_matrix = distance_matrix

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "_"]:
        """Compute lengths of all edges present in the network.

        Args:
            matrix: Adjacency matrix of the network

        Returns:
            torch.Tensor: Vector of edge lengths
        """
        # Get indices where edges exist (upper triangle only to avoid duplicates)
        upper_tri = torch.triu(matrix, diagonal=1)
        edges = torch.where(upper_tri > 0)
        # Return the distances for these edges as a 1D tensor
        return self.distance_matrix[edges]
