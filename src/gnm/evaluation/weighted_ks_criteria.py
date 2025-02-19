import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
import networkx as nx
from typing import Optional

from .evaluation_base import KSCriterion, WeightedEvaluationCriterion

from gnm.utils import node_strengths, weighted_clustering_coefficients


class WeightedNodeStrengthKS(KSCriterion, WeightedEvaluationCriterion):
    """KS statistic comparing node strength distributions between networks.

    Node strength is the weighted equivalent of node degree - it is the sum of the
    weights of all edges connected to a node.
    """

    def __str__(self) -> str:
        return "Weighted node strength KS"

    def __init__(self, normalise: Optional[bool] = True):
        """
        Args:
            normalise:
                If True, normalise the weights of the network by the maximum weight in the network. Defaults to True.
        """
        KSCriterion.__init__(self)
        WeightedEvaluationCriterion.__init__(self)
        self.normalise = normalise

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks num_nodes"]:
        """Compute strength for each node in the network.

        Args:
            matrix: Weight matrix of the network

        Returns:
            torch.Tensor: Vector of node strengths
        """
        if self.normalise:
            return node_strengths(matrices / matrices.amax(dim=-1, keepdim=True))
        else:
            return node_strengths(matrices)


class WeightedBetweennessKS(KSCriterion, WeightedEvaluationCriterion):
    """KS statistic comparing weighted betweenness centrality distributions between networks.

    We compute betweenness centrality using Brandes algorithm. This computes betweenness
    centrality for a node $u$ in a weighted network is:

    $$
    c_B(u) = \sum_{v,w} \\frac{\sigma(v,w|u)}{\sigma(v,w)},
    $$

    where $\sigma(v,w)$ is the number of shortest paths from $v$ to $w$, and
    $\sigma(v,w|u)$ is the number of those that pass through $u$.
    For weighted networks, path lengths are computed using the edge weights as distance.
    """

    def __str__(self) -> str:
        return "Weighted betweenness centrality KS"

    def __init__(self, normalise: Optional[bool] = True):
        """
        Args:
            normalise: If True, normalise the weights of the network by the maximum weight in the network. Defaults to True.
        """
        KSCriterion.__init__(self)
        WeightedEvaluationCriterion.__init__(self)
        self.normalise = normalise

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks num_nodes"]:
        """Compute weighted betweenness centrality for each node in the network.

        Args:
            matrices: Weight matrices of the network

        Returns:
            torch.Tensor: array of weighted betweenness centralities
        """
        if self.normalise:
            to_graph = matrices / matrices.amax(dim=-1, keepdim=True)
        else:
            to_graph = matrices

        betweenness_values = []
        for network_idx in range(to_graph.shape[0]):
            # Convert to networkx for betweenness calculation
            G = nx.from_numpy_array(to_graph[network_idx].detach().cpu().numpy())
            betweenness = nx.betweenness_centrality(G, weight="weight")
            # Convert dict to list preserving node order
            betweenness_values.append(
                torch.tensor([betweenness[i] for i in range(len(betweenness))])
            )
        return torch.stack(betweenness_values)


class WeightedClusteringKS(KSCriterion, WeightedEvaluationCriterion):
    """KS statistic comparing weighted clustering coefficient distributions between networks.

    Implements the Onnela et al. (2005) definition of weighted clustering, which uses
    the geometric mean of triangle weights. For each node $u$, the clustering coefficient is:

    $$
    c(u) = \\frac{1}{k_u(k_u-1)} \sum_{v,w} (\hat{w}_{uv} \\times \hat{w}_{uw} \\times \hat{w}_{vw})^{1/3},
    $$

    where $k_u$ is the node strength of node $u$, and $\hat{w}_{uv}$ is the weight of the edge between nodes $u$ and $v$,
    after normalising by dividing by the maximum weight in the network.
    """

    def __init__(self):
        KSCriterion.__init__(self)
        WeightedEvaluationCriterion.__init__(self)

    def __str__(self) -> str:
        return "Weighted clustering coefficient KS"

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrices: Float[torch.Tensor, "num_networks num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_networks num_nodes"]:
        """Compute weighted clustering coefficient for each node.

        Args:
            matrix: Weight matrix of the network

        Returns:
            torch.Tensor: Vector of weighted clustering coefficients
        """
        with torch.no_grad():
            return weighted_clustering_coefficients(matrices)
