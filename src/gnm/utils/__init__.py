r"""Utility functions for working with generative network models.

This subpackage provides various utility functions that support the core generative
network modeling functionality. It includes:

- **Statistical measures**: Functions for statistical comparisons between networks
- **Graph properties**: Various network metrics and measures for analyzing graph structure
- **Data validation**: Functions to verify the validity of network data structures
- **Control networks**: Functions for generating control networks with preserved properties

These utilities handle both binary and weighted networks and are optimised
for use with PyTorch tensors.
"""

from .statistics import ks_statistic
from .graph_properties import (
    node_strengths,
    binary_clustering_coefficients,
    weighted_clustering_coefficients,
    communicability,
    binary_betweenness_centrality,
)
from .checks import binary_checks, weighted_checks
from .control import get_control

__all__ = [
    "ks_statistic",
    "node_strengths",
    "binary_clustering_coefficients",
    "weighted_clustering_coefficients",
    "communicability",
    "binary_betweenness_centrality",
    "binary_checks",
    "weighted_checks",
    "get_control",
]
