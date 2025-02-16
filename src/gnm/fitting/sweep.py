from .evaluation_base import EvaluationCriterion
import torch
from typing import List, Iterator, Optional, Any
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


@dataclass
class BinarySweepParameters:
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
class SweepConfig:
    binary_sweep_parameters: BinarySweepParameters
    num_simulations: int
    seed_adjacency_matrix: Optional[
        List[Float[torch.Tensor, "... num_nodes num_nodes"]]
    ] = None
    distance_matrix: Optional[List[Float[torch.Tensor, "num_nodes num_nodes"]]] = None
    weighted_sweep_parameters: Optional[WeightedSweepParameters] = None
    seed_weight_matrix: Optional[List[Float[torch.Tensor, "... num_nodes num_nodes"]]]
    heterochronous_matrix: Optional[
        List[Float[torch.Tensor, "... num_nodes num_nodes"]]
    ] = None

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Creates an iterator over all combinations of run parameters.
        Yields pairs of parameter objects representing every possible combination
        of parameters

        Returns:
            Iterator yielding dictionaries of parameters
        """
        # Create lists for optional parameters, using empty list if None
        seed_adj_list = (
            self.seed_adjacency_matrix
            if self.seed_adjacency_matrix is not None
            else [None]
        )
        distance_list = (
            self.distance_matrix if self.distance_matrix is not None else [None]
        )
        heterochronous_list = (
            self.heterochronous_matrix
            if self.heterochronous_matrix is not None
            else [None]
        )
        seed_weight_list = (
            self.seed_weight_matrix if self.seed_weight_matrix is not None else [None]
        )

        # Get weighted parameters iterator if it exists, otherwise use single None
        weighted_sweep_parameters = (
            iter(self.weighted_sweep_parameters)
            if self.weighted_sweep_parameters is not None
            else [None]
        )

        # Create product of all parameter combinations
        for params in product(
            iter(self.binary_parameters),
            seed_adj_list,
            distance_list,
            weighted_sweep_parameters,
            seed_weight_list,
            heterochronous_list,
        ):
            # Unpack the values
            (
                binary_params,
                seed_adj,
                distance_matrix,
                weighted_parameters,
                seed_weights,
                heterochronous_matrix,
            ) = params

            run_config = RunConfig(
                binary_parameters=binary_params,
                num_simulations=self.num_simulations,
                seed_adjacency_matrix=seed_adj,
                distance_matrix=distance_matrix,
                weighted_parameters=weighted_parameters,
                seed_weight_matrix=seed_weights,
                heterochronous_matrix=heterochronous_matrix,
            )

            yield run_config


@dataclass
class RunConfig:
    binary_parameters: BinaryGenerativeParameters
    num_simulations: int
    seed_adjacency_matrix: Optional[Float[torch.Tensor, "... num_nodes num_nodes"]]
    distance_matrix: Optional[Float[torch.Tensor, "num_nodes num_nodes"]]
    weighted_parameters: Optional[WeightedGenerativeParameters]
    seed_weight_matrix: Optional[Float[torch.Tensor, "... num_nodes num_nodes"]]
    heterochronous_matrix: Optional[Float[torch.Tensor, "... num_nodes num_nodes"]]


@dataclass
class Results:
    added_edges: List[Float[torch.Tensor, "... num_nodes num_nodes"]]
    adjacency_snapshots: List[Float[torch.Tensor, "... num_nodes num_nodes"]]
    weight_snapshots: List[Float[torch.Tensor, "... num_nodes num_nodes"]]


@jaxtyped(typechecker=typechecked)
def perform_sweep(
    sweep_config: SweepConfig,
    evaluations: List[EvaluationCriterion],
    real_binary_matrices: Optional[
        Float[torch.Tensor, "num_real_binary_networks num_nodes num_nodes"]
    ] = None,
    real_weighted_matrices: Optional[
        Float[torch.Tensor, "num_real_weighted_networks num_nodes num_nodes"]
    ] = None,
    wandb_logging: bool = False,
):
    """Perform a parameter sweep over the specified configuration.

    Args:
        sweep_config:
            Configuration for the parameter sweep
        num_simulations:
            Number of simulations to run for each parameter combination
        evaluation_criterion:
            Evaluation criterion to use for comparing synthetic and real networks
        real_binary_matrices:
            Real binary networks to compare synthetic networks against
        real_weighted_matrices:
            Real weighted networks to compare synthetic networks against
        wandb_logging:
            Whether to log results to Weights & Biases
    """

    for run_config in sweep_config:

        model = GenerativeNetworkModel(
            binary_parmeters=run_config.binary_parameters,
            num_simulations=run_config.num_simulations,
            seed_adjacency_matrix=run_config.seed_adjacency_matrix,
            distance_matrix=run_config.distance_matrix,
            weighted_parameters=run_config.weighted_parameters,
            seed_weight_matrix=run_config.seed_weight_matrix,
        )

        added_edges_list, adjacency_snapshots, weight_snapshots = model.run_model(
            heterochronous_matrix=run_config.heterochronous_matrix
        )

        # Find the evaluation type
        evaluation_type = evaluation_criterion.accepts
        if evaluation_type == "binary":
            if real_binary_matrices is None:
                pass

            real_matrix = real_binary_matrices
            matrix = model.adjacency_matrix

        elif evaluation_type == "weighted":
            if real_weighted_matrices is None:
                pass

            real_matrix = real_weighted_matrices
            matrix = model.weight_matrix
        else:
            raise ValueError(f"Unknown evaluation type: {evaluation_type}")

        # Evaluate the model
        difference_grid = evaluation_criterion(matrix, real_weighted_matrices)
