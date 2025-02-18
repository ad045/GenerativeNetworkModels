# gnm.evaluation

## Evaluation 

::: gnm.evaluation.EvaluationCriterion
    options:
        members:
            - __call__

::: gnm.evaluation.KSCriterion
    options:
        members:
            - __call__

::: gnm.evaluation.CorrelationCriterion
    options:
        members:
            - __call__ 

::: gnm.evaluation.MaxCriteria
    options:
        members:
            - __call__

::: gnm.evaluation.MeanCriteria
    options:
        members: []

::: gnm.evaluation.WeightedSumCriteria
    options:
        members:
            - __init__
            - __call__

### Binary KS Criteria

::: gnm.evaluation.BetweennessKS
    options:
        members: []

::: gnm.evaluation.ClusteringKS
    options:
        members: []

::: gnm.evaluation.DegreeKS
    options:
        members: []

::: gnm.evaluation.EdgeLengthKS
    options:
        members:
            - __init__

### Weighted KS Criteria

::: gnm.evaluation.WeightedNodeStrengthKS
    options:
        members:
            - __init__

::: gnm.evaluation.WeightedBetweennessKS
    options:
        members:
            - __init__

::: gnm.evaluation.WeightedClusteringKS
    options:
        members: []

### Binary Correlation Criteria

::: gnm.evaluation.DegreeCorrelation
    options:
        members: []
    
::: gnm.evaluation.ClusteringCorrelation
    options:
        members: []
    
::: gnm.evaluation.BetweennessCorrelation
    options:
        members: []