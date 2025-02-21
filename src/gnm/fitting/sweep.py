from gnm.evaluation import (
    BinaryEvaluationCriterion,
    WeightedEvaluationCriterion,
    CompositeCriterion,
)
import torch
from typing import List, Optional, Any, Union
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
import gc

# import wandb

from .experiment_dataclasses import (
    Experiment,
    EvaluationResults,
    RunHistory,
    RunConfig,
    SweepConfig,
)

from gnm import (
    GenerativeNetworkModel,
)
from gnm.utils import binary_checks, weighted_checks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@jaxtyped(typechecker=typechecked)
def perform_run(
    run_config: RunConfig,
    binary_evaluations: Optional[
        List[Union[BinaryEvaluationCriterion, CompositeCriterion]]
    ] = None,
    weighted_evaluations: Optional[
        List[
            Union[
                WeightedEvaluationCriterion,
                CompositeCriterion,
            ]
        ]
    ] = None,
    real_binary_matrices: Optional[
        Float[torch.Tensor, "num_real_binary_networks num_nodes num_nodes"]
    ] = None,
    real_weighted_matrices: Optional[
        Float[torch.Tensor, "num_real_weighted_networks num_nodes num_nodes"]
    ] = None,
    save_only_evaluations: Optional[bool] = False,
    device: Optional[Union[torch.device,str]] = None,
) -> Experiment:
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

    Returns:
        Dictionary with keys "run_config" and "results" containing the run configuration and results
    """

    model = GenerativeNetworkModel(
        binary_parameters=run_config.binary_parameters,
        num_simulations=run_config.num_simulations,
        seed_adjacency_matrix=run_config.seed_adjacency_matrix,
        distance_matrix=run_config.distance_matrix,
        weighted_parameters=run_config.weighted_parameters,
        seed_weight_matrix=run_config.seed_weight_matrix,
        device=device,
    )

    added_edges, adjacency_snapshots, weight_snapshots = model.run_model(
        heterochronous_matrix=run_config.heterochronous_matrix
    )

    run_history = RunHistory(
        added_edges=added_edges,
        adjacency_snapshots=adjacency_snapshots,
        weight_snapshots=weight_snapshots,
    )

    evaluation_results = perform_evaluations(
        model=model,
        binary_evaluations=binary_evaluations,
        weighted_evaluations=weighted_evaluations,
        real_binary_matrices=real_binary_matrices,
        real_weighted_matrices=real_weighted_matrices,
        device=device,
    )

    if save_only_evaluations:
        experiment = Experiment(run_config=run_config, evaluation_results=evaluation_results)
    else:
        experiment = Experiment(run_config=run_config, model=model, run_history=run_history, evaluation_results=evaluation_results)
    
    gc.collect()
    torch.cuda.empty_cache()

    return experiment


@jaxtyped(typechecker=typechecked)
def perform_sweep(
    sweep_config: SweepConfig,
    binary_evaluations: Optional[
        List[Union[BinaryEvaluationCriterion, CompositeCriterion]]
    ] = None,
    weighted_evaluations: Optional[
        List[
            Union[
                WeightedEvaluationCriterion,
                CompositeCriterion,
            ]
        ]
    ] = None,
    real_binary_matrices: Optional[
        Float[torch.Tensor, "num_real_binary_networks num_nodes num_nodes"]
    ] = None,
    real_weighted_matrices: Optional[
        Float[torch.Tensor, "num_real_weighted_networks num_nodes num_nodes"]
    ] = None,
    save_only_evaluations: Optional[bool] = False,
    device: Optional[Union[torch.device,str]] = None,
) -> List[Experiment]:
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
    """
    run_results = []

    for run_config in sweep_config:
        experiment = perform_run(
            run_config=run_config,
            binary_evaluations=binary_evaluations,
            weighted_evaluations=weighted_evaluations,
            real_binary_matrices=real_binary_matrices,
            real_weighted_matrices=real_weighted_matrices,
            save_only_evaluations=save_only_evaluations,
            device=device,
        )

        run_results.append(experiment)

        gc.collect()
        torch.cuda.empty_cache()

    return run_results


@jaxtyped(typechecker=typechecked)
def perform_evaluations(
    model: GenerativeNetworkModel,
    binary_evaluations: Optional[
        List[Union[BinaryEvaluationCriterion, CompositeCriterion]]
    ] = None,
    weighted_evaluations: Optional[
        List[
            Union[
                WeightedEvaluationCriterion,
                CompositeCriterion,
            ]
        ]
    ] = None,
    real_binary_matrices: Optional[
        Float[torch.Tensor, "num_real_binary_networks num_nodes num_nodes"]
    ] = None,
    real_weighted_matrices: Optional[
        Float[torch.Tensor, "num_real_weighted_networks num_nodes num_nodes"]
    ] = None,
    device: Optional[Union[torch.device,str]] = None,
) -> EvaluationResults:
    
    if binary_evaluations is not None:
        for evaluation in binary_evaluations:
            assert (
                evaluation.accepts == "binary"
            ), f"Binary evaluations must accept binary matrices. Evaluation {evaluation} accepts {evaluation.accepts}."

    if weighted_evaluations is not None:
        for evaluation in weighted_evaluations:
            assert (
                evaluation.accepts == "weighted"
            ), f"Weighted evaluations must accept weighted matrices. Evaluation {evaluation} accepts {evaluation.accepts}."

    if real_binary_matrices is not None:
        try:
            binary_checks(real_binary_matrices)
        except AssertionError as e:
            raise AssertionError(f"real_binary_matrices are not valid. {e}")
    if real_weighted_matrices is not None:
        try:
            weighted_checks(real_weighted_matrices)
        except AssertionError as e:
            raise AssertionError(f"real_weighted_matrices are not valid. {e}")

    # Move the experiment onto the desired device. 
    if device is not None:
        model.to_device(device)
        if isinstance(device, str):
            device = torch.device(device)

        real_binary_matrices = real_binary_matrices.to(device)
        real_weighted_matrices = real_weighted_matrices.to(device)

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
        and model.weight_matrix is not None
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

    return EvaluationResults(
        binary_evaluations=binary_evaluations_results,
        weighted_evaluations=weighted_evaluations_results,
    )

