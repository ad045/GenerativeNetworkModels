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
    def __str__(self) -> str:
        """Return a string representation of the criterion."""
        pass

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


class CorrelationCriterion(ABC, EvaluationCriterion):
    """Base class for correlation-based network evaluation criteria.

    This class implements network comparison using correlation coefficients
    between network properties (e.g., degree distribution, clustering
    coefficients). The correlation coefficient measures the linear relationship
    between two variables, providing a measure of how similar two distributions
    are.

    Note:
        Subclasses must implement the `_get_graph_statistics` method to define the
        network property to use in the correlation test
    """

    def __init__(self, smoothing_matrix: Float[torch.Tensor, "num_nodes num_nodes"]):
        """
        Args:
            smoothing_matrix:
                Matrix used to smooth the network property values
        """
        self.smoothing_matrix = smoothing_matrix

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self,
        synthetic_matrices: Float[
            torch.Tensor, "num_synthetic_networks num_nodes num_nodes"
        ],
        real_matrices: Float[torch.Tensor, "num_real_networks num_nodes num_nodes"],
    ) -> Float[torch.Tensor, "num_synthetic_networks num_real_networks"]:
        """Compute the correlation coefficient between network property distributions.

        Args:
            synthetic_matrices:
                Batch of adjacency/weight matrices of the synthetic networks
            real_matrices:
                Adjacency/weight matrices of the real networks

        Returns:
            Correlation coefficients for all pairs of synthetic and real networks
        """
        # Compute network property values for each network
        synthetic_statistics = self._get_graph_statistics(synthetic_matrices)
        real_statistics = self._get_graph_statistics(real_matrices)

        smoothed_synthetic_statistics = torch.matmul(
            self.smoothing_matrix, synthetic_statistics
        )  # Shape [num_synthetic_networks num_nodes]
        smoothed_real_statistics = torch.matmul(
            self.smoothing_matrix, real_statistics
        )  # Shape [num_real_networks num_nodes]

        # Compute correlation coefficients between all pairs of distributions
        # Center the data
        real_centered = smoothed_real_statistics - smoothed_real_statistics.mean(
            dim=1, keepdim=True
        )
        synth_centered = (
            smoothed_synthetic_statistics
            - smoothed_synthetic_statistics.mean(dim=1, keepdim=True)
        )

        # Compute standard deviations
        real_std = torch.sqrt((real_centered**2).sum(dim=1, keepdim=True))
        synth_std = torch.sqrt((synth_centered**2).sum(dim=1, keepdim=True))

        # Normalize the data
        real_normalized = real_centered / real_std
        synth_normalized = synth_centered / synth_std

        # Compute correlation matrix using matrix multiplication
        corr_matrix = torch.mm(synth_normalized, real_normalized.t())

        # Divide by number of nodes to get Pearson R
        corr_matrix = corr_matrix / smoothed_real_statistics.shape[1]

        return corr_matrix

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks num_nodes"]:
        """Compute network properties for correlation comparison.

        Must be implemented by subclasses to define which network property
        to use in the correlation test.

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

    def __str__(self) -> str:
        return f"Maximum({', '.join(str(criterion) for criterion in self.criteria)})"

    def __init__(self, criteria: list[EvaluationCriterion]):
        """
        Args:
            criteria:
                List of evaluation criteria to combine
        """
        self.criteria = criteria
        self.accepts = self.criteria[0].accepts
        assert all(
            criterion.accepts == self.accepts for criterion in self.criteria
        ), "All criteria must accept the same type of network"

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

    def __str__(self) -> str:
        return (
            f"MeanCriteria({', '.join(str(criterion) for criterion in self.criteria)})"
        )

    def __init__(self, criteria: list[EvaluationCriterion]):
        """
        Args:
            criteria:
                List of evaluation criteria to combine
        """
        self.criteria = criteria
        # Check that all the criteria accept the same type of network
        self.accepts = self.criteria[0].accepts
        assert all(
            criterion.accepts == self.accepts for criterion in self.criteria
        ), "All criteria must accept the same type of network"

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


class WeightedSumCriteria(EvaluationCriterion):
    """Combines multiple evaluation criteria by taking their weighted sum.

    This class enables the evaluation of networks using multiple criteria
    """

    def __init__(self, criteria: list[EvaluationCriterion], weights: list[float]):
        """
        Args:
            criteria:
                List of evaluation criteria to combine
            weights:
                List of weights for each criterion
        """
        self.criteria = criteria
        self.weights = weights
        self.accepts = self.criteria[0].accepts
        assert all(
            criterion.accepts == self.accepts for criterion in self.criteria
        ), "All criteria must accept the same type of network"
        assert len(self.criteria) == len(
            self.weights
        ), "Number of criteria must match number of weights"

    def __str__(self) -> str:
        criteria_str = ", ".join(
            f"{str(criterion)} (weight={weight})"
            for criterion, weight in zip(self.criteria, self.weights)
        )
        return f"WeightedSumCriteria({criteria_str})"

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self,
        synthetic_matrices: Float[
            torch.Tensor, "num_synthetic_networks num_nodes num_nodes"
        ],
        real_matrices: Float[torch.Tensor, "num_real_networks num_nodes num_nodes"],
    ) -> Float[torch.Tensor, "num_synthetic_networks num_real_networks"]:
        """Compute weighted sum of the evaluation criteria.

        Args:
            synthetic_matrix:
                Adjacency/weight matrix of the synthetic network
            real_matrix:
                Adjacency/weight matrix of the real network

        Returns:
            Weighted sum of evaluation criteria
        """
        return torch.stack(
            [
                weight * criterion(synthetic_matrices, real_matrices)
                for criterion, weight in zip(self.criteria, self.weights)
            ]
        ).sum(dim=0)
