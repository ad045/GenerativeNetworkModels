r"""Parameter fitting and analysis for generative network models.

This subpackage provides tools for systematically exploring parameter spaces,
running experiments with generative network models, and analyzing the results.
The module implements functionality for fitting model parameters to observed networks
and evaluating how well generated networks match real-world data.

The module includes:

- Data structures for defining parameter sweeps and storing experiment results
- Functions for running model simulations and parameter explorations
- Analysis tools for finding optimal parameter combinations
- Aggregation methods for summarising results across simulations

These tools enable users to:

1. Define parameter spaces to explore systematically
2. Run generative models with different parameter combinations
3. Evaluate network similarity using various criteria
4. Identify parameter values that produce the most realistic networks
"""

from .sweep import (
    perform_run,
    perform_sweep,
    perform_evaluations,
)
from .experiment_dataclasses import (
    BinarySweepParameters,
    WeightedSweepParameters,
    Experiment,
    EvaluationResults,
    RunHistory,
    RunConfig,
    SweepConfig,
)
from .analysis import (
    Aggregator,
    MeanAggregator,
    MaxAggregator,
    MinAggregator,
    QuantileAggregator,
    optimise_evaluation,
)

from .experiment_saving import (
    ExperimentEvaluation
)

__all__ = [
    "BinarySweepParameters",
    "WeightedSweepParameters",
    "SweepConfig",
    "RunConfig",
    "Experiment",
    "EvaluationResults",
    "RunHistory",
    "perform_run",
    "perform_sweep",
    "perform_evaluations",
    "Aggregator",
    "MeanAggregator",
    "MaxAggregator",
    "MinAggregator",
    "QuantileAggregator",
    "optimise_evaluation",
    "ExperimentEvaluation"
]
