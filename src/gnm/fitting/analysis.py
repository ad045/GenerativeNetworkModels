import torch
from typing import List, Optional, Any, Union
from jaxtyping import Float, jaxtyped
from typeguard import typechecked

from .experiment_dataclasses import (
    Experiment,
    Results,
    RunConfig,
    SweepConfig,
)

from gnm.evaluation import (
    BinaryEvaluationCriterion,
    WeightedEvaluationCriterion,
    CompositeCriterion,
)


def filter_experiments(
    experiments: List[Experiment],
    conditions: List[dict[str, Any]],
) -> List[Experiment]:
    pass
