r"""Weight optimisation criteria for weighted generative network models.

This subpackage provides various criteria for optimising the weights in weighted
generative network models. Each criterion defines an objective function that guides
how weights evolve during network generation.

The module includes:

- Base abstract class for defining custom optimisation criteria
- Communicability-based criteria that optimise based on network communication properties
- Distance-based criteria that incorporate spatial constraints
- Simple weight-based criteria that focus on the weight distribution itself

These criteria can be used to create networks with different properties by guiding
the optimisation process toward different objective functions.
"""

from .optimisation_criteria import (
    OptimisationCriterion,
    Communicability,
    NormalisedCommunicability,
    DistanceWeightedCommunicability,
    NormalisedDistanceWeightedCommunicability,
    Weight,
    NormalisedWeight,
    WeightedDistance,
    NormalisedWeightedDistance,
)

__all__ = [
    "OptimisationCriterion",
    "Communicability",
    "NormalisedCommunicability",
    "DistanceWeightedCommunicability",
    "NormalisedDistanceWeightedCommunicability",
    "Weight",
    "NormalisedWeight",
    "WeightedDistance",
    "NormalisedWeightedDistance",
]
