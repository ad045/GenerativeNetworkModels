import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from abc import ABC, abstractmethod

from gnm.utils import ks_statistic

from gnm.utils import binary_checks, weighted_checks


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
        self._pre_call(synthetic_matrices)
        self._pre_call(real_matrices)
        return self._evaluate(synthetic_matrices, real_matrices)

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def _pre_call(
        self, matrices: Float[torch.Tensor, "num_network num_nodes num_nodes"]
    ):
        """Perform checks on matrices before they are evaluated.

        Args:
            matrices:
                Adjacency/weight matrices of the network
        """
        pass

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def _evaluate(
        self,
        synthetic_matrices: Float[
            torch.Tensor, "num_synthetic_networks num_nodes num_nodes"
        ],
        real_matrices: Float[torch.Tensor, "num_real_networks num_nodes num_nodes"],
    ) -> Float[torch.Tensor, "num_synthetic_networks num_real_networks"]:
        pass


class BinaryEvaluationCriterion(EvaluationCriterion, ABC):
    def __init__(self):
        self.accepts = "binary"

    @jaxtyped(typechecker=typechecked)
    def _pre_call(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ):
        """Perform checks on matrices before they are evaluated.

        Args:
            matrices:
                Binary adjacency matrices
        """
        binary_checks(matrices)


class WeightedEvaluationCriterion(EvaluationCriterion, ABC):
    def __init__(self):
        self.accepts = "weighted"

    @jaxtyped(typechecker=typechecked)
    def _pre_call(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ):
        """Perform checks on matrices before they are evaluated.

        Args:
            matrices:
                Weighted adjacency matrices
        """
        weighted_checks(matrices)


class WeightedEvaluation(EvaluationCriterion, ABC):
    def __init__(self):
        self.accepts = "weighted"

    def _pre_call(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ):
        """Perform checks on matrices before they are evaluated.

        Args:
            matrices:
                Weighted adjacency matrices
        """
        # check that the matrices are non-negative:
        assert torch.all(matrices >= 0), "Matrices must be non-negative"

        # Check that the matrices are symmetric:
        assert torch.allclose(
            matrices, matrices.transpose(-1, -2)
        ), "Matrices must be symmetric"


class KSCriterion(EvaluationCriterion, ABC):
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
    def _evaluate(
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


class CorrelationCriterion(EvaluationCriterion, ABC):
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
    def _evaluate(
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
            synthetic_statistics, self.smoothing_matrix
        )  # Shape [num_synthetic_networks num_nodes]
        smoothed_real_statistics = torch.matmul(
            real_statistics, self.smoothing_matrix
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
        real_std = torch.sqrt((real_centered**2).sum(dim=1, keepdim=True) + 1e-12)
        synth_std = torch.sqrt((synth_centered**2).sum(dim=1, keepdim=True) + 1e-12)

        # Normalize the data
        real_normalised = real_centered / (real_std + 1e-12)
        synth_normalised = synth_centered / (synth_std + 1e-12)

        # Compute correlation matrix using matrix multiplication
        corr_matrix = torch.mm(synth_normalised, real_normalised.t())

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


class CompositeCriterion(EvaluationCriterion, ABC):
    def __init__(self, criteria: list[EvaluationCriterion]):
        """
        Args:
            criteria:
                List of evaluation criteria to combine
        """
        assert len(criteria) > 0, "Must provide at least one criterion"
        self.criteria = criteria
        self.accepts = self.criteria[0].accepts
        assert all(
            criterion.accepts == self.accepts for criterion in self.criteria
        ), "All criteria must accept the same type of network"

    @jaxtyped(typechecker=typechecked)
    def _pre_call(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ):
        self.criteria[0]._pre_call(matrices)
