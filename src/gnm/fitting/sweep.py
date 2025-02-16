from .evaluation_base import EvaluationCriterion
import torch
from typing import List, Iterator, Tuple, Optional
from itertools import product
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from dataclasses import dataclass
from abc import ABC, abstractmethod

from gnm import (
    BinaryGenerativeParameters,
    WeightedGenerativeParameters,
    GenerativeNetworkModel,
)
from gnm.generative_rules import GenerativeRule
from gnm.weight_criteria import OptimisationCriterion


class SweepConfiguration(ABC):
    @abstractmethod
    def __iter__(self):
        pass


@dataclass
class BinarySweepParameters(SweepConfiguration):
    eta: List[float]
    gamma: List[float]
    lambdah: List[float]
    distance_relationship_type: List[str]
    preferential_relationship_type: List[str]
    heterochronicity_relationship_type: List[str]
    generative_rule: List[GenerativeRule]
    prob_offset: List[float] = [1e-6]
    binary_updates_per_iteration: List[int] = [1]
    num_iterations: List[int]

    def __iter__(self) -> Iterator[BinaryGenerativeParameters]:
        """Creates an iterator over all combinations of parameters.
        Each combination is used to create a BinaryGenerativeParameters instance.

        Returns:
            Iterator yielding BinaryGenerativeParameters instances, one for each
            combination of parameters.
        """
        # Get all parameter names and their corresponding lists
        param_names = [
            "eta",
            "gamma",
            "lambdah",
            "distance_relationship_type",
            "preferential_relationship_type",
            "heterochronicity_relationship_type",
            "generative_rule",
            "prob_offset",
            "binary_updates_per_iteration",
        ]
        param_lists = [getattr(self, name) for name in param_names]

        for values in product(*param_lists):
            params = dict(zip(param_names, values))
            yield BinaryGenerativeParameters(**params)


@dataclass
class WeightedSweepParameters:
    alpha: List[float]
    optimisation_criterion: List[OptimisationCriterion]
    optimisation_normalisation: List[bool] = [False]
    weight_lower_bound: List[float] = [0.0]
    weight_upper_bound: List[float] = [float("inf")]
    maximise_criterion: List[bool] = [False]
    weight_updates_per_iteration: List[int] = [1]

    def __iter__(self) -> Iterator[WeightedGenerativeParameters]:
        """Creates an iterator over all combinations of parameters.
        Each combination is used to create a WeightedGenerativeParameters instance.

        Returns:
            Iterator yielding WeightedGenerativeParameters instances, one for each
            combination of parameters.
        """
        # Get all parameter names and their corresponding lists
        param_names = [
            "alpha",
            "optimisation_criterion",
            "optimisation_normalisation",
            "weight_lower_bound",
            "weight_upper_bound",
            "maximise_criterion",
            "weight_updates_per_iteration",
        ]
        param_lists = [getattr(self, name) for name in param_names]

        for values in product(*param_lists):
            params = dict(zip(param_names, values))
            yield WeightedGenerativeParameters(**params)


@dataclass
class JointParameterSweep(SweepConfiguration):
    binary: BinarySweepParameters
    weighted: WeightedSweepParameters

    def __iter__(
        self,
    ) -> Iterator[Tuple[BinaryGenerativeParameters, WeightedGenerativeParameters]]:
        """Creates an iterator over all combinations of binary and weighted parameters.

        Yields pairs of parameter objects representing every possible combination
        of binary and weighted parameters.

        Returns:
            Iterator yielding tuples of (BinaryGenerativeParameters, WeightedGenerativeParameters),
            one for each combination of parameters.
        """
        # Iterate over both parameter sets simultaneously
        for binary_params in self.binary:
            for weighted_params in self.weighted:
                yield (binary_params, weighted_params)


def perform_sweep(
    config: SweepConfiguration,
    evaluation_criterion: EvaluationCriterion,
    real_binary_matrix: Float[torch.Tensor, "num_nodes num_nodes"],
    real_weighted_matrix: Optional[Float[torch.Tensor, "num_nodes num_nodes"]],
):
    # Create some datastructure that stores key-value pairs of parameters and the corresponding runs.

    # Iterate
    for _ in config:
        model = GenerativeNetworkModel()
        added_edges_list, adjacency_snapshots, weight_snapshots = model.run_model()
