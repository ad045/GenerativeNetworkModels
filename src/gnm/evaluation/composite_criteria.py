import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked

from .evaluation_base import CompositeCriterion, EvaluationCriterion


class MaxCriteria(CompositeCriterion):
    """Combines multiple evaluation criteria by taking their maximum value.

    This class enables the evaluation of networks using multiple criteria
    simultaneously, where the overall dissimilarity is determined by the
    worst-performing (maximum) criterion. This approach ensures that the
    synthetic network must match the real network well across all specified
    properties.
    """

    def __str__(self) -> str:
        return f"Maximum({', '.join(str(criterion) for criterion in self.criteria)})"

    @jaxtyped(typechecker=typechecked)
    def _evaluate(
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


class MeanCriteria(CompositeCriterion):
    """Combines multiple evaluation criteria by taking their mean value.

    This class enables the evaluation of networks using multiple criteria
    simultaneously, where the overall dissimilarity is determined by the
    average value of all criteria.
    """

    def __str__(self) -> str:
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


class WeightedSumCriteria(CompositeCriterion):
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
        self.weights = weights
        super().__init__(criteria)

    def __str__(self) -> str:
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
