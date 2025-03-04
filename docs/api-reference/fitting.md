# gnm.fitting

::: gnm.fitting
    options:
      members: false
      show_root_heading: true
      show_root_full_path: false
    
## Dataclasses

::: gnm.fitting.BinarySweepParameters
    options:
        members: []

::: gnm.fitting.WeightedSweepParameters
    options:
        members: []

::: gnm.fitting.SweepConfig
    options:
        members: []

::: gnm.fitting.Experiment
    options:
        members: 
            - to_device

::: gnm.fitting.RunConfig
    options:
        members: []

::: gnm.fitting.RunHistory
    options:
        members: []

::: gnm.fitting.EvaluationResults
    options:
        members: []

## Performing sweeps and evaluations

::: gnm.fitting.perform_run

::: gnm.fitting.perform_sweep

::: gnm.fitting.perform_evaluations

::: gnm.fitting.optimise_evaluation

## Aggregating evaluations

::: gnm.fitting.Aggregator
    options:
        members:
            - __call__

::: gnm.fitting.MeanAggregator
    options:
        members: []

::: gnm.fitting.MaxAggregator
    options:
        members: []

::: gnm.fitting.MinAggregator
    options:
        members: []

::: gnm.fitting.QuantileAggregator
    options:
        members: 
            - __init__