import torch
from typing import List, Iterator, Optional, Any, Dict, Union
from itertools import product
from jaxtyping import Float, Int, jaxtyped
from typeguard import typechecked
from dataclasses import dataclass, field

import wandb

from gnm import (
    BinaryGenerativeParameters,
    WeightedGenerativeParameters,
    GenerativeNetworkModel,
)
from gnm.generative_rules import GenerativeRule
from gnm.weight_criteria import OptimisationCriterion


@dataclass
class BinarySweepParameters:
    eta: Float[torch.Tensor, "eta_samples"]
    gamma: Float[torch.Tensor, "gamma_samples"]
    lambdah: Float[torch.Tensor, "lambda_samples"]
    distance_relationship_type: List[str]
    preferential_relationship_type: List[str]
    heterochronicity_relationship_type: List[str]
    generative_rule: List[GenerativeRule]
    num_iterations: List[int]
    prob_offset: List[float] = field(default_factory=lambda: [1e-6])
    binary_updates_per_iteration: List[int] = field(default_factory=lambda: [1])

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
            "num_iterations",
            "prob_offset",
            "binary_updates_per_iteration",
        ]
        param_lists = [getattr(self, name) for name in param_names]

        for values in product(*param_lists):
            params = dict(zip(param_names, values))
            yield BinaryGenerativeParameters(**params)


@dataclass
class WeightedSweepParameters:
    alpha: Float[torch.Tensor, "alpha_samples"]
    optimisation_criterion: List[OptimisationCriterion]
    weight_lower_bound: List[float] = field(default_factory=lambda: [0.0])
    weight_upper_bound: List[float] = field(default_factory=lambda: [float("inf")])
    maximise_criterion: List[bool] = field(default_factory=lambda: [False])
    weight_updates_per_iteration: List[int] = field(default_factory=lambda: [1])

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
class RunConfig:
    binary_parameters: BinaryGenerativeParameters
    num_simulations: Optional[int] = None
    seed_adjacency_matrix: Optional[
        Union[
            Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
            Float[torch.Tensor, "num_nodes num_nodes"],
        ]
    ] = None
    distance_matrix: Optional[Float[torch.Tensor, "num_nodes num_nodes"]] = None
    weighted_parameters: Optional[WeightedGenerativeParameters] = None
    seed_weight_matrix: Optional[
        Union[
            Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
            Float[torch.Tensor, "num_nodes num_nodes"],
        ]
    ] = None
    heterochronous_matrix: Optional[
        Union[
            Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
            Float[torch.Tensor, "num_nodes num_nodes"],
        ]
    ] = None


@dataclass
class SweepConfig:
    binary_sweep_parameters: BinarySweepParameters
    num_simulations: Optional[int] = None
    seed_adjacency_matrix: Optional[
        List[
            Union[
                Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
                Float[torch.Tensor, "num_nodes num_nodes"],
            ]
        ]
    ] = None
    distance_matrices: Optional[List[Float[torch.Tensor, "num_nodes num_nodes"]]] = None
    weighted_sweep_parameters: Optional[WeightedSweepParameters] = None
    seed_weight_matrix: Optional[
        List[
            Union[
                Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
                Float[torch.Tensor, "num_nodes num_nodes"],
            ]
        ]
    ] = None
    heterochronous_matrix: Optional[
        List[
            Union[
                Float[
                    torch.Tensor,
                    "num_binary_updates num_simulations num_nodes num_nodes",
                ],
                Float[torch.Tensor, "num_binary_updates num_nodes num_nodes"],
            ]
        ]
    ] = None

    def __iter__(self) -> Iterator[RunConfig]:
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
            self.distance_matrices if self.distance_matrices is not None else [None]
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
            iter(self.binary_sweep_parameters),
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
class Results:
    added_edges: Int[torch.Tensor, "num_binary_updates num_simulations 2"]
    adjacency_snapshots: Float[
        torch.Tensor, "num_binary_updates num_simulations num_nodes num_nodes"
    ]
    weight_snapshots: Optional[
        Float[torch.Tensor, "num_weight_updates num_simulations num_nodes num_nodes"]
    ]
    binary_evaluations: Dict[
        str, Float[torch.Tensor, "num_real_binary_networks num_simulations"]
    ]
    weighted_evaluations: Dict[
        str, Float[torch.Tensor, "num_real_weighted_networks num_simulations"]
    ]


@dataclass
class Experiment:
    run_config: RunConfig
    model: GenerativeNetworkModel
    results: Results
