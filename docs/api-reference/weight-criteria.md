# Weight Criteria

::: gnm.weight_criteria
    options:
      members: false
      show_root_heading: true
      show_root_full_path: false

::: gnm.weight_criteria.OptimisationCriterion
    options:
        members:
            - __call__

## Basic Weight Criteria

::: gnm.weight_criteria.Weight
    options:
        members: 
            - __init__

::: gnm.weight_criteria.NormalisedWeight
    options:
        members:
            - __init__

## Distance-Based Criteria

::: gnm.weight_criteria.WeightedDistance
    options:
        members:
            - __init__

::: gnm.weight_criteria.NormalisedWeightedDistance
    options:
        members:
            - __init__

## Communicability-Based Criteria

::: gnm.weight_criteria.Communicability
    options:
        members:
            - __init__

::: gnm.weight_criteria.NormalisedCommunicability
    options:
        members:
            - __init__

## Distance-Weighted Communicability Criteria

::: gnm.weight_criteria.DistanceWeightedCommunicability
    options:
        members:
            - __init__

::: gnm.weight_criteria.NormalisedDistanceWeightedCommunicability
    options:
        members:
            - __init__



