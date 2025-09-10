from ._device import get_device
r"""Composite evaluation criteria for combining multiple network comparison methods.

This module provides classes for combining multiple evaluation criteria into a single
metric using different aggregation methods (maximum, mean, weighted sum). These composite
criteria allow for simultaneous evaluation of multiple network properties while reducing
the result to a single scalar value for optimization purposes.
"""

import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked

from .evaluation_base import CompositeCriterion, EvaluationCriterion

DEVICE = get_device()


class MaxCriteria(CompositeCriterion):
    r"""Combine multiple evaluation criteria by taking their maximum value.

    This class enables the evaluation of networks using multiple criteria
    simultaneously, where the overall dissimilarity is determined by the
    worst-performing (maximum) criterion. This approach ensures that the
    synthetic network must match the real network well across all specified
    properties.

    Examples:
        >>> import torch
        >>> from gnm.evaluation import DegreeKS, ClusteringKS, MaxCriteria
        >>> from gnm.defaults import get_binary_network
        >>> from gnm.utils import get_control
        >>> # Create individual criteria
        >>> degree_ks = DegreeKS()
        >>> clustering_ks = ClusteringKS()
        >>> # Combine them using MaxCriteria
        >>> max_criterion = MaxCriteria([degree_ks, clustering_ks])
        >>> # Load a default binary network
        >>> real_network = get_binary_network()
        >>> # Create a random network
        >>> random_network = get_control(real_network)
        >>> # Calculate degree distribution dissimilarity
        >>> dissimilarity = max_criterion(random_network, real_network)

    See Also:
        - [`evaluation.CompositeCriterion`][gnm.evaluation.CompositeCriterion]: Base class for combining criteria, from which this class inherits.
        - [`evaluation.MeanCriteria`][gnm.evaluation.MeanCriteria]: For averaging multiple criteria instead of taking the maximum.
    """

    def __str__(self) -> str:
        r"""Get string representation of the composite criterion.

        Returns:
            String in the format "MaxCriteria(criterion1, criterion2, ...)"
        """
        return (
            f"MaxCriteria({', '.join(str(criterion) for criterion in self.criteria)})"
        )

    @jaxtyped(typechecker=typechecked)
    def _evaluate(
        self,
        synthetic_matrices: Float[
            torch.Tensor, "num_synthetic_networks num_nodes num_nodes"
        ],
        real_matrices: Float[torch.Tensor, "num_real_networks num_nodes num_nodes"],
    ) -> Float[torch.Tensor, "num_synthetic_networks num_real_networks"]:
        r"""Compute maximum dissimilarity across all criteria.

        For each pair of synthetic and real networks, evaluates all criteria and returns
        the maximum value, representing the worst-case dissimilarity across all network
        properties being measured.

        Args:
            synthetic_matrices:
                Batch of adjacency/weight matrices of synthetic networks with shape
                [num_synthetic_networks, num_nodes, num_nodes]
            real_matrices:
                Batch of adjacency/weight matrices of real networks with shape
                [num_real_networks, num_nodes, num_nodes]

        Returns:
            Maximum dissimilarity value across all criteria with shape
            [num_synthetic_networks, num_real_networks]
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


class MeanCriteria(CompositeCriterion):
    r"""Combine multiple evaluation criteria by taking their mean value.

    This class enables the evaluation of networks using multiple criteria
    simultaneously, where the overall dissimilarity is determined by the
    average value of all criteria. This approach balances different aspects
    of network similarity without letting a single criterion dominate.


    Examples:
        >>> import torch
        >>> from gnm.evaluation import DegreeKS, ClusteringKS, MeanCriteria
        >>> from gnm.defaults import get_binary_network
        >>> from gnm.utils import get_control
        >>> # Create individual criteria
        >>> degree_ks = DegreeKS()
        >>> clustering_ks = ClusteringKS()
        >>> # Combine them using MeanCriteria
        >>> max_criterion = MeanCriteria([degree_ks, clustering_ks])
        >>> # Load a default binary network
        >>> real_network = get_binary_network()
        >>> # Create a random network
        >>> random_network = get_control(real_network)
        >>> # Calculate degree distribution dissimilarity
        >>> dissimilarity = max_criterion(random_network, real_network)

    See Also:
        - [`evaluation.CompositeCriterion`][gnm.evaluation.CompositeCriterion]: Base class for combining criteria, from which this class inherits.
        - [`evaluation.MaxCriteria`][gnm.evaluation.MaxCriteria]: For maximum-based combination.
        - [`evaluation.WeightedSumCriteria`][gnm.evaluation.WeightedSumCriteria]: For weighted combinations of criteria.
    """

    def __str__(self) -> str:
        r"""Get string representation of the composite criterion.

        Returns:
            String in the format "MeanCriteria(criterion1, criterion2, ...)"
        """
        return (
            f"MeanCriteria({', '.join(str(criterion) for criterion in self.criteria)})"
        )

    @jaxtyped(typechecker=typechecked)
    def _evaluate(
        self,
        synthetic_matrices: Float[
            torch.Tensor, "num_synthetic_networks num_nodes num_nodes"
        ],
        real_matrices: Float[torch.Tensor, "num_real_networks num_nodes num_nodes"],
    ) -> Float[torch.Tensor, "num_synthetic_networks num_real_networks"]:
        r"""Compute mean dissimilarity across all criteria.

        For each pair of synthetic and real networks, evaluates all criteria and returns
        the average value, representing the average dissimilarity across all network
        properties being measured.

        Args:
            synthetic_matrices:
                Batch of adjacency/weight matrices of synthetic networks with shape
                [num_synthetic_networks, num_nodes, num_nodes].
            real_matrices:
                Batch of adjacency/weight matrices of real networks with shape
                [num_real_networks, num_nodes, num_nodes].

        Returns:
            Mean dissimilarity value across all criteria with shape
            [num_synthetic_networks, num_real_networks].
        """
        return torch.stack(
            [
                criterion(synthetic_matrices, real_matrices)
                for criterion in self.criteria
            ]
        ).mean(dim=0)


class WeightedSumCriteria(CompositeCriterion):
    r"""Combine multiple evaluation criteria by taking their weighted sum.

    This class enables the evaluation of networks using multiple criteria with
    different importance weights assigned to each criterion. By adjusting the weights,
    users can emphasize certain network properties over others in the evaluation.

    Examples:
        >>> import torch
        >>> from gnm.evaluation import DegreeKS, ClusteringKS, WeightedSumCriteria
        >>> from gnm.defaults import get_binary_network
        >>> from gnm.utils import get_control
        >>> # Create individual criteria
        >>> degree_ks = DegreeKS()
        >>> clustering_ks = ClusteringKS()
        >>> # Combine them using WeightedSumCriteria (more weight on degree)
        >>> weighted_criterion = WeightedSumCriteria(
        ...     [degree_ks, clustering_ks],
        ...     weights=[0.7, 0.3]
        ... )
        >>> # Calculate dissimilarity between networks
        >>> real_network = get_binary_network()  # Example real network
        >>> random_network = get_control(real_network)  # Example random network
        >>> dissimilarity = weighted_criterion(random_network, real_network)

    See Also:
        - [`evaluation.CompositeCriterion`][gnm.evaluation.CompositeCriterion]: Base class for combining criteria, from which this class inherits.
        - [`evaluation.MaxCriteria`][gnm.evaluation.MaxCriteria]: For maximum-based combination.
        - [`evaluation.MeanCriteria`][gnm.evaluation.MeanCriteria]: For averaging multiple criteria.
    """

    def __init__(self, criteria: list[EvaluationCriterion], weights: list[float], device=None):
        r"""
        Args:
            criteria:
                List of evaluation criteria to combine.
            weights:
                List of weights for each criterion. Should have the same length as criteria.
        """
        self.weights = weights
        super().__init__(criteria)

    def __str__(self) -> str:
        r"""Get string representation of the composite criterion.

        Returns:
            String in the format "WeightedSumCriteria(criterion1 (weight=w1), criterion2 (weight=w2), ...)"
        """
        criteria_str = ", ".join(
            f"{str(criterion)} (weight={weight})"
            for criterion, weight in zip(self.criteria, self.weights)
        )
        return f"WeightedSumCriteria({criteria_str})"

    @jaxtyped(typechecker=typechecked)
    def _evaluate(
        self,
        synthetic_matrices: Float[
            torch.Tensor, "num_synthetic_networks num_nodes num_nodes"
        ],
        real_matrices: Float[torch.Tensor, "num_real_networks num_nodes num_nodes"],
    ) -> Float[torch.Tensor, "num_synthetic_networks num_real_networks"]:
        r"""Compute weighted sum of dissimilarities across all criteria.

        For each pair of synthetic and real networks, evaluates all criteria, multiplies
        each result by its corresponding weight, and returns the sum of these weighted values.

        Args:
            synthetic_matrices:
                Batch of adjacency/weight matrices of synthetic networks with shape
                [num_synthetic_networks, num_nodes, num_nodes]
            real_matrices:
                Batch of adjacency/weight matrices of real networks with shape
                [num_real_networks, num_nodes, num_nodes]

        Returns:
            Weighted sum of dissimilarity values with shape
            [num_synthetic_networks, num_real_networks]
        """
        return torch.stack(
            [
                weight * criterion(synthetic_matrices, real_matrices)
                for criterion, weight in zip(self.criteria, self.weights)
            ]
        ).sum(dim=0)
