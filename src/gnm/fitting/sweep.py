r"""Functions for running generative network model simulations and parameter sweeps.

This module provides functions for executing generative network models with specific
parameters and performing systematic parameter sweeps. It includes functions for running
individual model simulations, evaluating generated networks against real networks, and
exploring large parameter spaces efficiently.

The main components are:

- perform_run: Executes a single generative network model run with specific parameters
- perform_sweep: Runs multiple model simulations across a parameter space
- perform_evaluations: Evaluates generated networks against real networks using various criteria
"""

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
from tqdm import tqdm

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
        List[
            Union[BinaryEvaluationCriterion, CompositeCriterion]
            ]
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
    save_model: bool = True,
    save_run_history: bool = True,
    device: Optional[Union[torch.device, str]] = None,
) -> Experiment:
    r"""Perform a single run of the generative network model.

    This function executes a generative network model with the specified configuration,
    creates synthetic networks, and evaluates them against real networks using provided
    evaluation criteria. It returns an Experiment object containing the results.

    Args:
        run_config:
            Configuration for the run, specifying parameters and input matrices.
        binary_evaluations:
            List of criteria for evaluating binary network properties.
            Defaults to None (no binary evaluation).
        weighted_evaluations:
            List of criteria for evaluating weighted network properties.
            Defaults to None (no weighted evaluation).
        real_binary_matrices:
            Real binary networks to compare synthetic networks against.
            Required if binary_evaluations is provided.
        real_weighted_matrices:
            Real weighted networks to compare synthetic networks against.
            Required if weighted_evaluations is provided.
        save_model:
            If True, saves the model in the experiment. Set this argument to False to save on memory.
            Defaults to True.
        save_run_history:
            If True, saves the adjacency and weight snapshots in the run history.
            Set this argument to False to save on memory. Defaults to True.
        device:
            Device to run the model on. If unspecified, uses CUDA if available, else CPU.

    Returns:
        An Experiment object containing the run configuration, evaluation results,
        and optionally the model and run history.

    Examples:
        >>> from gnm import BinaryGenerativeParameters
        >>> from gnm.generative_rules import MatchingIndex
        >>> from gnm.fitting import RunConfig, perform_run
        >>> from gnm.evaluation import ClusteringKS
        >>> from gnm.defaults import get_binary_network, get_distance_matrix
        >>> # Create run configuration
        >>> binary_params = BinaryGenerativeParameters(
        ...     eta=-2.0,
        ...     gamma=0.3,
        ...     lambdah=0.0,
        ...     distance_relationship_type="powerlaw",
        ...     preferential_relationship_type="powerlaw",
        ...     heterochronicity_relationship_type="powerlaw",
        ...     generative_rule=MatchingIndex(),
        ...     num_iterations=100,
        ... )
        >>> config = RunConfig(
        ...     binary_parameters=binary_params,
        ...     num_simulations=5,
        ...     distance_matrix=get_distance_matrix(),
        ... )
        >>> # Define evaluation
        >>> binary_evals = [ClusteringKS()]
        >>> real_networks = get_binary_network()
        >>> # Run the model
        >>> experiment = perform_run(
        ...     run_config=config,
        ...     binary_evaluations=binary_evals,
        ...     real_binary_matrices=real_networks,
        ... )

    See Also:
        - [`fitting.RunConfig`][gnm.fitting.RunConfig]: Configuration for a single run
        - [`fitting.Experiment`][gnm.fitting.Experiment]: Result container for experiments
        - [`fitting.perform_sweep`][gnm.fitting.perform_sweep]: Function for running multiple parameter combinations
        - [`GenerativeNetworkModel`][gnm.GenerativeNetworkModel]: The network model being executed
    """

    model = GenerativeNetworkModel(
        binary_parameters=run_config.binary_parameters,
        num_simulations=run_config.num_simulations,
        seed_adjacency_matrix=run_config.seed_adjacency_matrix,
        distance_matrix=run_config.distance_matrix,
        weighted_parameters=run_config.weighted_parameters,
        seed_weight_matrix=run_config.seed_weight_matrix,
        device=device,
        verbose=False
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

    experiment = Experiment(
        run_config=run_config,
        model=model if save_model else None,
        run_history=run_history if save_run_history else None,
        evaluation_results=evaluation_results,
    )

    experiment.to_device("cpu")

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
    save_model: bool = True,
    save_run_history: bool = True,
    device: Optional[Union[torch.device, str]] = None,
    verbose: Optional[bool] = False
) -> List[Experiment]:
    r"""Perform a parameter sweep over multiple model configurations.

    This function systematically explores a parameter space by running the generative
    network model with different parameter combinations. It generates and evaluates
    synthetic networks for each configuration, returning a list of experiments.

    Args:
        sweep_config:
            Configuration for the parameter sweep, defining the parameter space to explore.

        binary_evaluations:
            List of criteria for evaluating binary network properties.
            Defaults to None (no binary evaluation).

        weighted_evaluations:
            List of criteria for evaluating weighted network properties.
            Defaults to None (no weighted evaluation).

        real_binary_matrices:
            Real binary networks to compare synthetic networks against.
            Required if binary_evaluations is provided.

        real_weighted_matrices:
            Real weighted networks to compare synthetic networks against.
            Required if weighted_evaluations is provided.

        save_model:
            If True, saves the model in the experiment. Set this argument to False to save on memory.
            Defaults to True.

        save_run_history:
            If True, saves the adjacency and weight snapshots in the run history.
            Set this argument to False to save on memory. Defaults to True.

        device:
            Device to run the models on. If None, uses CUDA if available, else CPU.

        verbose:
            If True, displays a progress bar for the sweep. Defaults to False.

    Returns:
        A list of Experiment objects, one for each parameter combination in the sweep.

    Examples:
        >>> import torch
        >>> from gnm.generative_rules import MatchingIndex
        >>> from gnm.fitting import BinarySweepParameters, SweepConfig, perform_sweep
        >>> from gnm.evaluation import ClusteringKS
        >>> from gnm.defaults import get_binary_network, get_distance_matrix
        >>> # Define parameter space
        >>> binary_sweep = BinarySweepParameters(
        ...     eta=torch.tensor([-3.0, -2.0, -1.0]),
        ...     gamma=torch.tensor([0.2, 0.3]),
        ...     lambdah=torch.tensor([0.0]),
        ...     distance_relationship_type=["powerlaw"],
        ...     preferential_relationship_type=["powerlaw"],
        ...     heterochronicity_relationship_type=["powerlaw"],
        ...     generative_rule=[MatchingIndex()],
        ...     num_iterations=[100],
        ... )
        >>> # Create sweep configuration
        >>> sweep_config = SweepConfig(
        ...     binary_sweep_parameters=binary_sweep,
        ...     num_simulations=50,
        ...     distance_matrices=[get_distance_matrix()],
        ... )
        >>> # Define evaluation
        >>> binary_evals = [ClusteringKS()]
        >>> real_networks = get_binary_network()
        >>> # Run the sweep
        >>> experiments = perform_sweep(
        ...     sweep_config=sweep_config,
        ...     binary_evaluations=binary_evals,
        ...     real_binary_matrices=real_networks,
        ...     save_only_evaluations=True,  # Save memory during sweep
        ... )
        >>> len(experiments)
        6

    See Also:
        - [`fitting.SweepConfig`][gnm.fitting.SweepConfig]: Configuration for parameter sweeps
        - [`fitting.perform_run`][gnm.fitting.perform_run]: Function for running a single configuration
        - [`fitting.optimise_evaluation`][gnm.fitting.optimise_evaluation]: Function for finding optimal parameters
    """
    run_results = []

    for run_config in tqdm(sweep_config, desc='Configuration Iterations', total=len(list(sweep_config)), disable=not verbose):
        experiment = perform_run(
            run_config=run_config,
            binary_evaluations=binary_evaluations,
            weighted_evaluations=weighted_evaluations,
            real_binary_matrices=real_binary_matrices,
            real_weighted_matrices=real_weighted_matrices,
            save_model=save_model,
            save_run_history=save_run_history,
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
    device: Optional[Union[torch.device, str]] = None,
) -> EvaluationResults:
    r"""Evaluate synthetic networks against real networks using various criteria.

    This function compares networks generated by a model against real networks using
    the specified evaluation criteria. It performs both binary and weighted evaluations
    if the corresponding parameters are provided.

    Args:
        model:
            The generative network model containing the synthetic networks to evaluate.

        binary_evaluations:
            List of criteria for evaluating binary network properties.
            Defaults to None (no binary evaluation).
        weighted_evaluations:
            List of criteria for evaluating weighted network properties.
            Defaults to None (no weighted evaluation).
        real_binary_matrices:
            Real binary networks to compare synthetic networks against.
            Required if binary_evaluations is provided.
        real_weighted_matrices:
            Real weighted networks to compare synthetic networks against.
            Required if weighted_evaluations is provided.
        device:
            Device to perform the evaluations on. If None, uses CUDA if available, else CPU.

    Returns:
        An EvaluationResults object containing the results of all evaluations.

    Examples:
        >>> from gnm import GenerativeNetworkModel, BinaryGenerativeParameters
        >>> from gnm.generative_rules import MatchingIndex
        >>> from gnm.fitting import perform_evaluations
        >>> from gnm.evaluation import ClusteringKS, DegreeKS
        >>> from gnm.defaults import get_binary_network, get_distance_matrix
        >>> # Create and run a model
        >>> binary_params = BinaryGenerativeParameters(
        ...     eta=-2.0,
        ...     gamma=0.3,
        ...     lambdah=0.0,
        ...     distance_relationship_type="powerlaw",
        ...     preferential_relationship_type="powerlaw",
        ...     heterochronicity_relationship_type="powerlaw",
        ...     generative_rule=MatchingIndex(),
        ...     num_iterations=100,
        ... )
        >>> model = GenerativeNetworkModel(
        ...     binary_parameters=binary_params,
        ...     num_simulations=15,
        ...     distance_matrix=get_distance_matrix(),
        ... )
        >>> _, _, _ = model.run_model()
        >>> # Define evaluations
        >>> binary_evals = [ClusteringKS(), DegreeKS()]
        >>> real_networks = get_binary_network()
        >>> # Perform evaluations
        >>> eval_results = perform_evaluations(
        ...     model=model,
        ...     binary_evaluations=binary_evals,
        ...     real_binary_matrices=real_networks,
        ... )
        >>> # Access results
        >>> clustering_scores = eval_results.binary_evaluations["ClusteringKS"]
        >>> degree_scores = eval_results.binary_evaluations["DegreeKS"]

    See Also:
        - [`evaluation.BinaryEvaluationCriterion`][gnm.evaluation.BinaryEvaluationCriterion]: Criteria for binary networks
        - [`evaluation.WeightedEvaluationCriterion`][gnm.evaluation.WeightedEvaluationCriterion]: Criteria for weighted networks
        - [`fitting.EvaluationResults`][gnm.fitting.EvaluationResults]: Container for evaluation results
        - [`GenerativeNetworkModel`][gnm.GenerativeNetworkModel]: The model being evaluated
    """

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
