import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from .evaluation_base import CorrelationCriterion, BinaryEvaluationCriterion

from gnm.utils import binary_clustering_coefficients, binary_betweenness_centrality


class DegreeCorrelation(CorrelationCriterion, BinaryEvaluationCriterion):
    """
    Correlation criterion comparing degree distributions between networks.
    """

    def __init__(self, smoothing_matrix: Float[torch.Tensor, "num_nodes num_nodes"]):
        CorrelationCriterion.__init__(self, smoothing_matrix)
        BinaryEvaluationCriterion.__init__(self)

    def __str__(self) -> str:
        return "Binary degree correlation"

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks num_nodes"]:
        """Compute degree for each node in the network.

        Args:
            matrix:
                Adjacency matrix of the network

        Returns:
            Vector of node degrees
        """
        return matrices.sum(dim=-1)


class ClusteringCorrelation(CorrelationCriterion, BinaryEvaluationCriterion):

    def __init__(self, smoothing_matrix: Float[torch.Tensor, "num_nodes num_nodes"]):
        CorrelationCriterion.__init__(self, smoothing_matrix)
        BinaryEvaluationCriterion.__init__(self)

    def __str__(self) -> str:
        return "Clustering correlation"

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks num_nodes"]:
        """Compute clustering coefficient for each node in the network.

        Args:
            matrix:
                Adjacency matrix of the network

        Returns:
            Vector of clustering coefficients
        """
        return binary_clustering_coefficients(matrices)


class BetweennessCorrelation(CorrelationCriterion, BinaryEvaluationCriterion):

    def __init__(self, smoothing_matrix: Float[torch.Tensor, "num_nodes num_nodes"]):
        CorrelationCriterion.__init__(self, smoothing_matrix)
        BinaryEvaluationCriterion.__init__(self)

    def __str__(self) -> str:
        return "Binary betweenness centrality correlation"

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks num_nodes"]:
        """Compute clustering coefficient for each node in the network.

        Args:
            matrix:
                Adjacency matrix of the network

        Returns:
            Vector of clustering coefficients
        """
        return binary_betweenness_centrality(matrices)
