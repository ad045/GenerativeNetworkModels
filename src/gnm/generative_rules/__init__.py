r"""Generative rules for determining connection affinities in network models.

This subpackage provides various generative rules that determine how nodes
in a network form connections based on different topological properties.
These rules produce affinity matrices that represent the likelihood of
connections forming between pairs of nodes.

The rules fall into several categories:
- Homophily-based rules (MatchingIndex, Neighbours): Nodes connect based on shared neighborhoods
- Clustering-based rules: Nodes connect based on clustering coefficient relationships
- Degree-based rules: Nodes connect based on node degree relationships
- Geometric rule: A baseline rule where all connections have equal affinity

Each rule produces an affinity matrix that can be used within a generative network model
to determine connection probabilities along with other factors such as distance.
"""

from .generative_rules import (
    GenerativeRule,
)
from .generative_rules import (
    MatchingIndex,
    Neighbours,
    Geometric,
)
from .generative_rules import (
    ClusteringRule,
    ClusteringAverage,
    ClusteringMax,
    ClusteringMin,
    ClusteringDifference,
    ClusteringProduct,
)
from .generative_rules import (
    DegreeRule,
    DegreeAverage,
    DegreeMax,
    DegreeMin,
    DegreeDifference,
    DegreeProduct,
)


__all__ = [
    "GenerativeRule",
    "MatchingIndex",
    "Neighbours",
    "Geometric",
    "ClusteringRule",
    "ClusteringAverage",
    "ClusteringMax",
    "ClusteringMin",
    "ClusteringDifference",
    "ClusteringProduct",
    "DegreeRule",
    "DegreeAverage",
    "DegreeMax",
    "DegreeMin",
    "DegreeDifference",
    "DegreeProduct",
]
