import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from abc import ABC, abstractmethod

from gnm.utils import ks_statistic


class EvaluationCriterion(ABC):
    """Base abstract class for network evaluation criteria.

    This class provides a framework for defining various criteria to evaluate the similarity
    between a synthetic (generated) network and a real (target) network. Each criterion
    computes a dissimilarity measure between the two networks based on specific network
    properties.

    Note:
        Subclasses must implement the `__call__` method to define their specific
        evaluation criterion.
    """

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def __call__(
        self,
        synthetic_matrices: Float[
            torch.Tensor, "num_synthetic_networks num_nodes num_nodes"
        ],
        real_matrices: Float[torch.Tensor, "num_real_networks num_nodes num_nodes"],
    ) -> Float[torch.Tensor, "num_synthetic_networks num_real_networks"]:
        """Compute the dissimilarity between two networks.

        Args:
            synthetic_matrices:
                Batch of adjacency/weight matrices of the synthetic networks
            real_matrices:
                Adjacency/weight matrices of the real networks

        Returns:
            Tensor of dissimilarity values (higher values indicate greater dissimilarity)
        """
        pass


class KSCriterion(ABC, EvaluationCriterion):
    """Base class for Kolmogorov-Smirnov (KS) test based network evaluation.

    This class implements network comparison using the KS test statistic between
    distributions of network properties (e.g., degree distribution, clustering
    coefficients). The KS statistic measures the maximum difference between two
    cumulative distribution functions, providing a measure of how different two
    distributions are.

    Note:
        Subclasses must implement the `_get_graph_statistics` method to define the
        network property to use in the KS test
    """

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self,
        synthetic_matrices: Float[
            torch.Tensor, "num_synthetic_networks num_nodes num_nodes"
        ],
        real_matrices: Float[torch.Tensor, "num_real_networks num_nodes num_nodes"],
    ) -> Float[torch.Tensor, "num_synthetic_networks num_real_networks"]:
        """Compute the KS statistic between network property distributions.

        Args:
            synthetic_matrices:
                Batch of adjacency/weight matrices of the synthetic networks
            real_matrices:
                Adjacency/weight matrices of the real networks

        Returns:
            KS statistics for all pairs of synthetic and real networks
        """
        # Compute network property values for each network
        synthetic_statistics = self._get_graph_statistics(synthetic_matrices)
        real_statistics = self._get_graph_statistics(real_matrices)

        # Compute KS statistics between all pairs of distributions
        return ks_statistic(synthetic_statistics, real_statistics)

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks _"]:
        """Compute network properties for KS comparison.

        Must be implemented by subclasses to define which network property
        to use in the KS test.

        Args:
            matrices: Adjacency/weight matrix of the network

        Returns:
            1D tensor of network property values
        """
        pass


class MaxCriteria(EvaluationCriterion):
    """Combines multiple evaluation criteria by taking their maximum value.

    This class enables the evaluation of networks using multiple criteria
    simultaneously, where the overall dissimilarity is determined by the
    worst-performing (maximum) criterion. This approach ensures that the
    synthetic network must match the real network well across all specified
    properties.
    """

    def __init__(self, criteria: list[EvaluationCriterion]):
        """
        Args:
            criteria:
                List of evaluation criteria to combine
        """
        self.criteria = criteria

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self,
        synthetic_matrices: Float[
            torch.Tensor, "num_synthetic_networks num_nodes num_nodes"
        ],
        real_matrices: Float[torch.Tensor, "num_real_networks num_nodes num_nodes"],
    ) -> Float[torch.Tensor, "num_synthetic_networks num_real_networks"]:
        """Compute maximum dissimilarity across all criteria.

        Args:
            synthetic_matrix:
                Adjacency/weight matrix of the synthetic network
            real_matrix:
                Adjacency/weight matrix of the real network

        Returns:
            Maximum dissimilarity value across all criteria
        """
        return (
            torch.stack(
                [
                    criterion(synthetic_matrices, real_matrices)
                    for criterion in self.criteria
                ]
            )
            .max(dim=0)
            .values
        )


class MeanCriteria(EvaluationCriterion):
    """Combines multiple evaluation criteria by taking their mean value.

    This class enables the evaluation of networks using multiple criteria
    simultaneously, where the overall dissimilarity is determined by the
    average value of all criteria.
    """

    def __init__(self, criteria: list[EvaluationCriterion]):
        """
        Args:
            criteria:
                List of evaluation criteria to combine
        """
        self.criteria = criteria

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self,
        synthetic_matrices: Float[
            torch.Tensor, "num_synthetic_networks num_nodes num_nodes"
        ],
        real_matrices: Float[torch.Tensor, "num_real_networks num_nodes num_nodes"],
    ) -> Float[torch.Tensor, "num_synthetic_networks num_real_networks"]:
        """Compute mean dissimilarity across all criteria.

        Args:
            synthetic_matrix:
                Adjacency/weight matrix of the synthetic network
            real_matrix:
                Adjacency/weight matrix of the real network

        Returns:
            Mean dissimilarity value across all criteria
        """
        return torch.stack(
            [
                criterion(synthetic_matrices, real_matrices)
                for criterion in self.criteria
            ]
        ).mean(dim=0)
