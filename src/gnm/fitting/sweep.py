from gnm.evaluation import (
    BinaryEvaluationCriterion,
    WeightedEvaluationCriterion,
)
import torch
from typing import List, Iterator, Optional, Any, Dict, Union
from itertools import product
from jaxtyping import Float, jaxtyped
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
from gnm.utils import binary_checks, weighted_checks


@dataclass
class BinarySweepParameters:
    eta: List[float]
    gamma: List[float]
    lambdah: List[float]
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
    num_simulations: int
    seed_adjacency_matrix: Optional[
        Union[
            Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
            Float[torch.Tensor, "num_nodes num_nodes"],
        ]
    ]
    distance_matrix: Optional[Float[torch.Tensor, "num_nodes num_nodes"]]
    weighted_parameters: Optional[WeightedGenerativeParameters]
    seed_weight_matrix: Optional[
        Union[
            Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
            Float[torch.Tensor, "num_nodes num_nodes"],
        ]
    ]
    heterochronous_matrix: Optional[
        Union[
            Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
            Float[torch.Tensor, "num_nodes num_nodes"],
        ]
    ]


@dataclass
class SweepConfig:
    binary_sweep_parameters: BinarySweepParameters
    num_simulations: int
    seed_adjacency_matrix: Optional[
        List[
            Union[
                Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
                Float[torch.Tensor, "num_nodes num_nodes"],
            ]
        ]
    ] = None
    distance_matrix: Optional[List[Float[torch.Tensor, "num_nodes num_nodes"]]] = None
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
class Results:
    added_edges: Float[torch.Tensor, "num_binary_updates num_simulations 2"]
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


@jaxtyped(typechecker=typechecked)
def perform_run(
    run_config: RunConfig,
    binary_evaluations: Optional[List[BinaryEvaluationCriterion]],
    weighted_evaluations: Optional[List[WeightedEvaluationCriterion]],
    real_binary_matrices: Optional[
        Float[torch.Tensor, "num_real_binary_networks num_nodes num_nodes"]
    ] = None,
    real_weighted_matrices: Optional[
        Float[torch.Tensor, "num_real_weighted_networks num_nodes num_nodes"]
    ] = None,
    wandb_logging: bool = False,
) -> Dict[str, Any]:
    """Perform a single run of the generative network model.

    Args:
        run_config:
            Configuration for the run
        binary_evaluations:
            List of binary evaluation criteria to use for comparing synthetic and real networks
        weighted_evaluations:
            List of weighted evaluation criteria to use for comparing synthetic and real networks
        real_binary_matrices:
            Real binary networks to compare synthetic networks against. Defaults to None
        real_weighted_matrices:
            Real weighted networks to compare synthetic networks against. Defaults to None
        wandb_logging:
            Whether or not to use wandb to log the runs. Defaults to False.

    Returns:
        Dictionary with keys "run_config" and "results" containing the run configuration and results
    """
    if real_binary_matrices is not None:
        binary_checks(real_binary_matrices)
    if real_weighted_matrices is not None:
        weighted_checks(real_weighted_matrices)

    model = GenerativeNetworkModel(
        binary_parmeters=run_config.binary_parameters,
        num_simulations=run_config.num_simulations,
        seed_adjacency_matrix=run_config.seed_adjacency_matrix,
        distance_matrix=run_config.distance_matrix,
        weighted_parameters=run_config.weighted_parameters,
        seed_weight_matrix=run_config.seed_weight_matrix,
    )

    added_edges, adjacency_snapshots, weight_snapshots = model.run_model(
        heterochronous_matrix=run_config.heterochronous_matrix
    )

    if binary_evaluations is not None and real_binary_matrices is not None:
        synthetic_adjacency_matrices = model.adjacency_matrix
        binary_evaluations_results = {
            str(evaluation): evaluation(
                synthetic_adjacency_matrices,
                real_binary_matrices,
            )
            for evaluation in binary_evaluations
        }
    else:
        binary_evaluations_results = {}

    if (
        weighted_evaluations is not None
        and real_weighted_matrices is not None
        and run_config.weighted_parameters is not None
    ):
        synthetic_weight_matrices = model.weight_matrix
        weighted_evaluations_results = {
            str(evaluation): evaluation(
                synthetic_weight_matrices,
                real_weighted_matrices,
            )
            for evaluation in weighted_evaluations
        }
    else:
        weighted_evaluations_results = {}

    results = Results(
        added_edges=added_edges,
        adjacency_snapshots=adjacency_snapshots,
        weight_snapshots=weight_snapshots,
        binary_evaluations=binary_evaluations_results,
        weighted_evaluations=weighted_evaluations_results,
    )

    return {"run_config": run_config, "results": results}


@jaxtyped(typechecker=typechecked)
def perform_sweep(
    sweep_config: SweepConfig,
    binary_evaluations: Optional[List[BinaryEvaluationCriterion]],
    weighted_evaluations: Optional[List[WeightedEvaluationCriterion]],
    real_binary_matrices: Optional[
        Float[torch.Tensor, "num_real_binary_networks num_nodes num_nodes"]
    ] = None,
    real_weighted_matrices: Optional[
        Float[torch.Tensor, "num_real_weighted_networks num_nodes num_nodes"]
    ] = None,
    wandb_logging: bool = False,
) -> List[Dict[str, Any]]:
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
    run_results = []

    for run_config in sweep_config:
        run_dict = perform_run(
            run_config=run_config,
            binary_evaluations=binary_evaluations,
            weighted_evaluations=weighted_evaluations,
            real_binary_matrices=real_binary_matrices,
            real_weighted_matrices=real_weighted_matrices,
            wandb_logging=wandb_logging,
        )

        run_results.append(run_dict)

    return run_results
