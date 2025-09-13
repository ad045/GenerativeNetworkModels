r"""Analysis tools for evaluating and optimising generative network model experiments.

This module provides classes and functions for analyzing the results of generative network
model experiments. It includes aggregation methods for summarizing evaluation scores across
multiple simulations and functions for finding optimal parameter combinations based on
evaluation criteria.

The main components are:

- Aggregator classes for different ways to combine scores (mean, max, min, quantile)
- The optimise_evaluation function to find best-performing experiments across a set of results
"""

import torch
from typing import List, Union, Tuple
from jaxtyping import Float, jaxtyped
from typeguard import typechecked

from .experiment_dataclasses import (
    Experiment,
)

from gnm.evaluation import (
    BinaryEvaluationCriterion,
    WeightedEvaluationCriterion,
    CompositeCriterion,
)
from abc import ABC, abstractmethod


class Aggregator(ABC):
    r"""Abstract base class for aggregating evaluation scores across simulations.

    Aggregators reduce a matrix of evaluation scores from multiple simulations into
    a single score for each real network. Different aggregation methods (mean, max, min,
    quantile) provide different perspectives on model performance.

    All aggregators transform a scores tensor with shape [num_synthetic_networks, num_real_networks]
    into a reduced tensor with shape [num_real_networks], applying their specific aggregation
    method along the first dimension, *i.e.*, across the synthetic networks.

    See Also:
        - [`fitting.MeanAggregator`][gnm.fitting.MeanAggregator]: Computes the mean across simulations
        - [`fitting.MaxAggregator`][gnm.fitting.MaxAggregator]: Takes the maximum score across simulations
        - [`fitting.MinAggregator`][gnm.fitting.MinAggregator]: Takes the minimum score across simulations
        - [`fitting.QuantileAggregator`][gnm.fitting.QuantileAggregator]: Computes a quantile of the scores
    """

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, scores: Float[torch.Tensor, "num_synthetic_networks num_real_networks"]
    ) -> Float[torch.Tensor, "num_real_networks"]:
        pass


class MeanAggregator(Aggregator):
    r"""Aggregates scores by taking the mean across simulations.

    This aggregator computes the average score across all synthetic networks for each
    real network. It provides a measure of central tendency in model performance.

    Examples:
        >>> import torch
        >>> from gnm.fitting import MeanAggregator
        >>> # Create some example scores
        >>> scores = torch.tensor([
        ...     [0.1, 0.2, 0.3],  # Scores for synthetic network 1
        ...     [0.2, 0.3, 0.4],  # Scores for synthetic network 2
        ...     [0.3, 0.4, 0.5],  # Scores for synthetic network 3
        ... ])
        >>> # Aggregate using the mean
        >>> aggregator = MeanAggregator()
        >>> mean_scores = aggregator(scores)
        >>> mean_scores
        tensor([0.2000, 0.3000, 0.4000])

    See Also:
        - [`fitting.Aggregator`][gnm.fitting.Aggregator]: The abstract base class for aggregators, from which this class inherits
        - [`fitting.optimise_evaluation`][gnm.fitting.optimise_evaluation]: Uses aggregators to find optimal experiments
    """

    def __call__(
        self, scores: Float[torch.Tensor, "num_synthetic_networks num_real_networks"]
    ) -> Float[torch.Tensor, "num_real_networks"]:
        return torch.mean(scores, dim=0)


class MaxAggregator(Aggregator):
    r"""Aggregates scores by taking the maximum across simulations.

    This aggregator selects the maximum score across all synthetic networks for each
    real network. It provides a measure of the worst-case performance when the score
    represents dissimilarity (higher is worse).

    Examples:
        >>> import torch
        >>> from gnm.fitting import MaxAggregator
        >>> # Create some example scores
        >>> scores = torch.tensor([
        ...     [0.1, 0.2, 0.3],  # Scores for synthetic network 1
        ...     [0.2, 0.3, 0.4],  # Scores for synthetic network 2
        ...     [0.3, 0.4, 0.5],  # Scores for synthetic network 3
        ... ])
        >>> # Aggregate using the maximum
        >>> aggregator = MaxAggregator()
        >>> max_scores = aggregator(scores)
        >>> max_scores
        tensor([0.3000, 0.4000, 0.5000])

    See Also:
        - [`fitting.Aggregator`][gnm.fitting.Aggregator]: The abstract base class for aggregators
        - [`fitting.optimise_evaluation`][gnm.fitting.optimise_evaluation]: Uses aggregators to find optimal experiments
    """

    def __call__(
        self, scores: Float[torch.Tensor, "num_synthetic_networks num_real_networks"]
    ) -> Float[torch.Tensor, "num_real_networks"]:
        return torch.max(scores, dim=0)


class MinAggregator(Aggregator):
    r"""Aggregates scores by taking the minimum across simulations.

    This aggregator selects the minimum score across all synthetic networks for each
    real network. It provides a measure of the best-case performance when the score
    represents dissimilarity (lower is better).

    Examples:
        >>> import torch
        >>> from gnm.fitting import MinAggregator
        >>> # Create some example scores
        >>> scores = torch.tensor([
        ...     [0.1, 0.2, 0.3],  # Scores for synthetic network 1
        ...     [0.2, 0.3, 0.4],  # Scores for synthetic network 2
        ...     [0.3, 0.4, 0.5],  # Scores for synthetic network 3
        ... ])
        >>> # Aggregate using the minimum
        >>> aggregator = MinAggregator()
        >>> min_scores = aggregator(scores)
        >>> min_scores
        tensor([0.1000, 0.2000, 0.3000])

    See Also:
        - [`fitting.Aggregator`][gnm.fitting.Aggregator]: The abstract base class for aggregators
        - [`fitting.optimise_evaluation`][gnm.fitting.optimise_evaluation]: Uses aggregators to find optimal experiments
    """

    def __call__(
        self, scores: Float[torch.Tensor, "num_synthetic_networks num_real_networks"]
    ) -> Float[torch.Tensor, "num_real_networks"]:
        return torch.min(scores, dim=0)


class QuantileAggregator(Aggregator):
    r"""Aggregates scores by computing a specific quantile across simulations.

    This aggregator calculates a specified quantile (*e.g.*, median, 75th percentile)
    across all synthetic networks for each real network. It provides a flexible way
    to characterize the distribution of scores beyond simple mean, min, or max.
    Defaults to the median, which is a more robust measure of central tendency than the mean.

    Examples:
        >>> import torch
        >>> from gnm.fitting import QuantileAggregator
        >>> # Create some example scores
        >>> scores = torch.tensor([
        ...     [0.1, 0.2, 0.3],  # Scores for synthetic network 1
        ...     [0.2, 0.3, 0.4],  # Scores for synthetic network 2
        ...     [0.3, 0.4, 0.5],  # Scores for synthetic network 3
        ... ])
        >>> # Aggregate using the median (0.5 quantile)
        >>> aggregator = QuantileAggregator(quantile=0.5)
        >>> median_scores = aggregator(scores)
        >>> median_scores
        tensor([0.2000, 0.3000, 0.4000])
        >>> # Aggregate using the 75th percentile
        >>> aggregator = QuantileAggregator(quantile=0.75)
        >>> q75_scores = aggregator(scores)
        >>> q75_scores
        tensor([0.2500, 0.3500, 0.4500])

    See Also:
        - [`fitting.Aggregator`][gnm.fitting.Aggregator]: The abstract base class for aggregators
        - [`fitting.optimise_evaluation`][gnm.fitting.optimise_evaluation]: Uses aggregators to find optimal experiments
    """

    def __init__(self, quantile: float = 0.5):
        r"""
        Args:
            quantile:
                The quantile to compute. Must be in the range [0, 1]. Default is 0.5 for the median.
        """
        self.quantile = quantile

    def __call__(
        self, scores: Float[torch.Tensor, "num_synthetic_networks num_real_networks"]
    ) -> Float[torch.Tensor, "num_real_networks"]:
        return torch.quantile(scores, self.quantile, dim=0)


@jaxtyped(typechecker=typechecked)
def optimise_evaluation(
    experiments: List[Experiment],
    criterion: Union[
        BinaryEvaluationCriterion, WeightedEvaluationCriterion, CompositeCriterion, str
    ],
    maximise_criterion: bool = False,
    aggregation: Aggregator = MeanAggregator(),
) -> Tuple[List[Experiment], Float[torch.Tensor, "num_real_networks"]]:
    r"""Find the optimal experiments based on evaluation criteria.

    This function searches through a list of experiments to find the ones that best
    satisfy a given criterion for each real network. It can either minimise or maximise
    the criterion value, depending on the desired optimisation direction.

    The function handles both binary and weighted evaluation criteria, and can work
    with criteria specified either by name (string) or object instance.

    Args:
        experiments:
            A list of experiments to search through for the optimal ones.
        criterion:
            The criterion to optimise. Can either be specified by name (string) or
            by passing in the criterion object directly.
        maximise_criterion:
            Whether to maximise the criterion. If True, the experiment with the highest
            criterion value is considered optimal. If False (default), the experiment
            with the lowest criterion value is considered optimal.
        aggregation:
            The method to aggregate evaluation scores across synthetic networks.
            Default is the MeanAggregator, which averages the evaluation values
            across all synthetic networks for each real network.

    Returns:
        optimal_experiments: A list of experiments, one for each real network, where
                           each experiment is the one that best satisfies the criterion
                           for that particular real network.
        current_best: The evaluation values of the optimal experiments for each real network.

    Examples:
        >>> from gnm.fitting import perform_sweep, perform_evaluations, optimise_evaluation
        >>> from gnm.evaluation import ClusteringKS
        >>> # Run a parameter sweep and get experiments.
        >>> experiments = perform_sweep(...)
        >>> # Find the experiments that best match clustering coefficients
        >>> criterion = ClusteringKS()
        >>> best_experiments, best_scores = optimise_evaluation(
        ...     experiments=experiments,
        ...     criterion=criterion,
        ...     maximise_criterion=False,
        ... )
        >>> # For the first real network, show the optimal parameters
        >>> best_exp = best_experiments[0]
        >>> print(f"Best eta: {best_exp.run_config.binary_parameters.eta}")
        >>> print(f"Best gamma: {best_exp.run_config.binary_parameters.gamma}")
        >>> print(f"Best score: {best_scores[0]}")

    See Also:
        - [`fitting.Aggregator`][gnm.fitting.Aggregator]: Base class for score aggregation methods
        - [`fitting.perform_evaluations`][gnm.fitting.perform_evaluations]: Function to evaluate networks against criteria
        - [`evaluation.BinaryEvaluationCriterion`][gnm.evaluation.BinaryEvaluationCriterion]: Criteria for binary networks
        - [`evaluation.WeightedEvaluationCriterion`][gnm.evaluation.WeightedEvaluationCriterion]: Criteria for weighted networks
    """
    assert len(experiments) > 0, "No experiments provided."

    available_criteria = set(
        experiments[0].evaluation_results.binary_evaluations.keys()
    ).union(experiments[0].evaluation_results.weighted_evaluations.keys())

    if isinstance(criterion, str):
        criterion_name = criterion

        if (
            criterion_name
            in experiments[0].evaluation_results.binary_evaluations.keys()
        ):
            criterion_type = "binary"
        elif (
            criterion_name
            in experiments[0].evaluation_results.weighted_evaluations.keys()
        ):
            criterion_type = "weighted"
        else:
            raise ValueError(
                f"Criterion not found in experiments. Available criteria are {available_criteria}. You may wish to call 'fitting.perform_evaluations' with the desired criterion before this function."
            )
    else:
        criterion_name = str(criterion)
        criterion_type = criterion.accepts

        if criterion_type == "binary":
            assert (
                criterion_name
                in experiments[0].evaluation_results.binary_evaluations.keys()
            ), f"Criterion not found in experiments. Available criteria are {available_criteria}. You may wish to call 'fitting.perform_evaluations' with the desired criterion before this function."
        elif criterion_type == "weighted":
            assert (
                criterion_name
                in experiments[0].evaluation_results.weighted_evaluations.keys()
            ), f"Criterion not found in experiments. Available criteria are {available_criteria}. You may wish to call 'fitting.perform_evaluations' with the desired criterion before this function."
        else:
            raise ValueError(f"Do not recognise criterion type {criterion_type}.")

    num_real_networks = (
        (experiments[0].evaluation_results.binary_evaluations[criterion_name].shape[-1])
        if criterion_type == "binary"
        else experiments[0]
        .evaluation_results.weighted_evaluations[criterion_name]
        .shape[-1]
    )

    optimal_experiments = [experiments[0]] * num_real_networks
    current_best = aggregation(
        experiments[0].evaluation_results.binary_evaluations[criterion_name]
        if criterion_type == "binary"
        else experiments[0].evaluation_results.weighted_evaluations[criterion_name]
    )

    for experiment in experiments[1:]:
        current_evaluation = aggregation(
            experiment.evaluation_results.binary_evaluations[criterion_name]
            if criterion_type == "binary"
            else experiment.evaluation_results.weighted_evaluations[criterion_name]
        )  # This has shape [num_real_networks]

        if maximise_criterion:
            for idx in range(num_real_networks):
                if current_evaluation[idx] > current_best[idx]:
                    optimal_experiments[idx] = experiment
                    current_best[idx] = current_evaluation[idx]
        else:
            for idx in range(num_real_networks):
                if current_evaluation[idx] < current_best[idx]:
                    optimal_experiments[idx] = experiment
                    current_best[idx] = current_evaluation[idx]

    return optimal_experiments, current_best
