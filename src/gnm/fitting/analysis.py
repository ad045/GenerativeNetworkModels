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
    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, scores: Float[torch.Tensor, "num_synthetic_networks num_real_networks"]
    ) -> Float[torch.Tensor, "num_real_networks"]:
        pass


class MeanAggregator(Aggregator):
    def __call__(
        self, scores: Float[torch.Tensor, "num_synthetic_networks num_real_networks"]
    ) -> Float[torch.Tensor, "num_real_networks"]:
        return torch.mean(scores, dim=0)


class MaxAggregator(Aggregator):
    def __call__(
        self, scores: Float[torch.Tensor, "num_synthetic_networks num_real_networks"]
    ) -> Float[torch.Tensor, "num_real_networks"]:
        return torch.max(scores, dim=0)


class MinAggregator(Aggregator):
    def __call__(
        self, scores: Float[torch.Tensor, "num_synthetic_networks num_real_networks"]
    ) -> Float[torch.Tensor, "num_real_networks"]:
        return torch.min(scores, dim=0)


class QuantileAggregator(Aggregator):
    def __init__(self, quantile: float):
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
    r"""Finds the experiment within a set of experiments which minimises the criterion given.

    Args:
        experiments:
            A list of experiments, from which we wish to find the optimal experiment.
        criterion:
            The criterion to optimise. Can either be specified by name or by passing in the criterion object directly.
        maximise_criterion:
            Whether to maximise the criterion. If True, the experiment which maximises the criterion is found. Default is False.
        aggregation:
            The method to aggregate the evaluation_results of the experiment across synthetic networks. Default is the MeanAggregator,
            which averages the evaluation value across all synthetic experiments for each real network.

    Returns:
        optimal_experiments: A list of each experiment which is optimal for each real network.
        current_best: The evaluation value of the optimal experiment for each real network.
    """
    assert len(experiments) > 0, "No experiments provided."

    available_criteria = set(experiments[0].evaluation_results.binary_evaluations.keys()).union(
        experiments[0].evaluation_results.weighted_evaluations.keys()
    )

    if isinstance(criterion, str):
        criterion_name = criterion

        if criterion_name in experiments[0].evaluation_results.binary_evaluations.keys():
            criterion_type = "binary"
        elif criterion_name in experiments[0].evaluation_results.weighted_evaluations.keys():
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
                criterion_name in experiments[0].evaluation_results.binary_evaluations.keys()
            ), f"Criterion not found in experiments. Available criteria are {available_criteria}. You may wish to call 'fitting.perform_evaluations' with the desired criterion before this function."
        elif criterion_type == "weighted":
            assert (
                criterion_name in experiments[0].evaluation_results.weighted_evaluations.keys()
            ), f"Criterion not found in experiments. Available criteria are {available_criteria}. You may wish to call 'fitting.perform_evaluations' with the desired criterion before this function."
        else:
            raise ValueError(f"Do not recognise criterion type {criterion_type}.")

    num_real_networks = (
        (experiments[0].evaluation_results.binary_evaluations[criterion_name].shape[-1])
        if criterion_type == "binary"
        else experiments[0].evaluation_results.weighted_evaluations[criterion_name].shape[-1]
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
