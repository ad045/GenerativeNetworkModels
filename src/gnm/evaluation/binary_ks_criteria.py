r"""Kolmogorov-Smirnov criteria for evaluating binary networks.

This module provides evaluation criteria based on the Kolmogorov-Smirnov (KS) test
to compare distributions of various network properties between synthetic and real
binary networks. These criteria quantify differences in degree distributions, clustering
coefficients, betweenness centrality, and edge length distributions.
"""

import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from .evaluation_base import KSCriterion, BinaryEvaluationCriterion

from gnm.utils import (
    binary_clustering_coefficients,
    binary_betweenness_centrality,
    ks_statistic,
)


class DegreeKS(KSCriterion, BinaryEvaluationCriterion):
    r"""Compare degree distributions between binary networks using KS statistic.

    This criterion measures the dissimilarity between the degree distributions of
    synthetic and real networks using the Kolmogorov-Smirnov test. The degree of
    a node is the number of connections it has to other nodes.

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_binary_network
        >>> from gnm.evaluation import DegreeKS
        >>> from gnm.utils import get_control
        >>> # Load a default binary network
        >>> real_network = get_binary_network()
        >>> # Create a random network
        >>> random_network = get_control(real_network)
        >>> # Calculate degree distribution dissimilarity
        >>> criterion = DegreeKS()
        >>> dissimilarity = criterion(random_network, real_network)

    See Also:
        - [`evaluation.BinaryEvaluationCriterion`][gnm.evaluation.BinaryEvaluationCriterion]: Base class for binary evaluation criteria, from which this class inherits.
        - [`evaluation.KSCriterion`][gnm.evaluation.KSCriterion]: Base class for KS criteria, from which this class inherits.
    """

    def __init__(self):
        KSCriterion.__init__(self)
        BinaryEvaluationCriterion.__init__(self)

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks _"]:
        r"""Extract degree values for each node in the networks.

        Computes the sum of connections for each node, which represents the
        node's degree in an undirected binary network.

        Args:
            matrices:
                Batch of adjacency matrices with shape [num_networks, num_nodes, num_nodes]

        Returns:
            Tensor of node degrees with shape [num_networks, num_nodes]
        """
        return matrices.sum(dim=-1)


class ClusteringKS(KSCriterion, BinaryEvaluationCriterion):
    r"""Compare clustering coefficient distributions between binary networks using KS statistic.

    This criterion measures the dissimilarity between the clustering coefficient
    distributions of synthetic and real networks using the Kolmogorov-Smirnov test.
    The clustering coefficient measures the degree to which nodes in a graph tend
    to cluster together.

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_binary_network
        >>> from gnm.evaluation import ClusteringKS
        >>> from gnm.utils import get_control
        >>> # Load a default binary network
        >>> real_network = get_binary_network()
        >>> # Create a random network
        >>> random_network = get_control(real_network)
        >>> # Calculate clustering coefficient distribution dissimilarity
        >>> criterion = ClusteringKS()
        >>> dissimilarity = criterion(random_network, real_network)

    See Also:
        - [`evaluation.BinaryEvaluationCriterion`][gnm.evaluation.BinaryEvaluationCriterion]: Base class for binary evaluation criteria, from which this class inherits.
        - [`evaluation.KSCriterion`][gnm.evaluation.KSCriterion]: Base class for KS criteria, from which this class inherits.
        - [`utils.binary_clustering_coefficients`][gnm.utils.binary_clustering_coefficients]: Function to compute clustering coefficients for binary networks, which this class uses.
    """

    def __init__(self):
        KSCriterion.__init__(self)
        BinaryEvaluationCriterion.__init__(self)

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks _"]:
        r"""Extract clustering coefficient values for each node in the networks.

        Computes the clustering coefficient for each node, representing the
        fraction of possible triangles through that node that exist.

        Args:
            matrices:
                Batch of adjacency matrices with shape [num_networks, num_nodes, num_nodes]

        Returns:
            Tensor of clustering coefficients with shape [num_networks, num_nodes]
        """
        return binary_clustering_coefficients(matrices)


class BetweennessKS(KSCriterion, BinaryEvaluationCriterion):
    r"""Compare betweenness centrality distributions between binary networks using KS statistic.

    This criterion measures the dissimilarity between the betweenness centrality
    distributions of synthetic and real networks using the Kolmogorov-Smirnov test.
    Betweenness centrality quantifies the number of times a node acts as a bridge
    along the shortest path between two other nodes.

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_binary_network
        >>> from gnm.evaluation import BetweennessKS
        >>> from gnm.utils import get_control
        >>> # Load a default binary network
        >>> real_network = get_binary_network()
        >>> # Create a random network
        >>> random_network = get_control(real_network)
        >>> # Calculate betweenness centrality distribution dissimilarity
        >>> criterion = BetweennessKS()
        >>> dissimilarity = criterion(random_network, real_network)

    See Also:
        - [`evaluation.BinaryEvaluationCriterion`][gnm.evaluation.BinaryEvaluationCriterion]: Base class for binary evaluation criteria, from which this class inherits.
        - [`evaluation.KSCriterion`][gnm.evaluation.KSCriterion]: Base class for KS criteria, from which this class inherits.
        - [`utils.binary_betweenness_centrality`][gnm.utils.binary_betweenness_centrality]: Function used to calculate betweenness centrality.
    """

    def __init__(self):
        KSCriterion.__init__(self)
        BinaryEvaluationCriterion.__init__(self)

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks _"]:
        r"""Extract betweenness centrality values for each node in the networks.

        Computes the betweenness centrality for each node, which measures how often
        a node lies on the shortest path between other nodes.

        Args:
            matrices:
                Batch of adjacency matrices with shape [num_networks, num_nodes, num_nodes]

        Returns:
            Tensor of betweenness centrality values with shape [num_networks, num_nodes]
        """
        return binary_betweenness_centrality(matrices)


class EdgeLengthKS(BinaryEvaluationCriterion):
    r"""Compare edge length distributions between binary networks using KS statistic.

    This criterion measures the dissimilarity between the edge length distributions
    of synthetic and real networks using the Kolmogorov-Smirnov test. Edge length
    is determined by a distance matrix that defines the spatial distance between nodes.

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_binary_network, get_distance_matrix
        >>> from gnm.evaluation import EdgeLengthKS
        >>> from gnm.utils import get_control
        >>> # Load a default binary network and distance matrix
        >>> real_network = get_binary_network()
        >>> distance_matrix = get_distance_matrix()
        >>> # Create a random network
        >>> random_network = get_control(real_network)
        >>> # Calculate edge length distribution dissimilarity
        >>> criterion = EdgeLengthKS(distance_matrix)
        >>> dissimilarity = criterion(random_network, real_network)

    See Also:
        - [`evaluation.BinaryEvaluationCriterion`][gnm.evaluation.BinaryEvaluationCriterion]: Base class for binary evaluation criteria, from which this class inherits.
    """

    def __init__(self, distance_matrix: Float[torch.Tensor, "num_nodes num_nodes"]):
        r"""
        Args:
            distance_matrix:
                Matrix defining the spatial distance between each pair of nodes
                with shape [num_nodes, num_nodes]
        """
        BinaryEvaluationCriterion.__init__(self)
        self.distance_matrix = distance_matrix

    @jaxtyped(typechecker=typechecked)
    def _evaluate(
        self,
        synthetic_matrices: Float[
            torch.Tensor, "num_synthetic_networks num_nodes num_nodes"
        ],
        real_matrices: Float[torch.Tensor, "num_real_networks num_nodes num_nodes"],
    ) -> Float[torch.Tensor, "num_synthetic_networks num_real_networks"]:
        r"""Compute KS statistics between edge length distributions.

        For each pair of synthetic and real networks, extracts the lengths of all edges
        using the provided distance matrix and computes the KS statistic between these
        distributions.

        Args:
            synthetic_matrices:
                Batch of adjacency matrices of synthetic networks with shape
                [num_synthetic_networks, num_nodes, num_nodes]
            real_matrices:
                Batch of adjacency matrices of real networks with shape
                [num_real_networks, num_nodes, num_nodes]

        Returns:
            KS statistics for all pairs of synthetic and real networks with shape
            [num_synthetic_networks, num_real_networks]
        """
        num_synthetic_networks = synthetic_matrices.shape[0]
        num_real_networks = real_matrices.shape[0]
        ks_distances = torch.zeros(
            num_synthetic_networks, num_real_networks, dtype=synthetic_matrices.dtype
        )
        synthetic_edge_lengths = []
        real_edge_lengths = []
        # Iterate through all pairs of synthetic and real networks
        for i in range(num_synthetic_networks):
            synthetic_edge_lengths.append(
                self._get_edge_lengths(synthetic_matrices[i, :, :])
            )
        for j in range(num_real_networks):
            real_edge_lengths.append(self._get_edge_lengths(real_matrices[j, :, :]))

        for i in range(num_synthetic_networks):
            for j in range(num_real_networks):
                ks_distances[i, j] = ks_statistic(
                    synthetic_edge_lengths[i], real_edge_lengths[j]
                )

        return ks_distances

    @jaxtyped(typechecker=typechecked)
    def _get_edge_lengths(
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "1 num_non_zero_edges"]:
        r"""Extract lengths of all edges in a network.

        Identifies all existing edges in the upper triangular part of the adjacency matrix
        (to avoid counting edges twice in an undirected network) and returns their
        corresponding distances from the distance matrix.

        Args:
            adjacency_matrix:
                Binary adjacency matrix with shape [num_nodes, num_nodes]

        Returns:
            Tensor of edge lengths with shape [1, num_edges]
        """
        adj = torch.triu(adjacency_matrix, diagonal=1)
        return self.distance_matrix[adj.bool()].flatten().unsqueeze(0)
