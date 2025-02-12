import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
import networkx as nx
from typing import Optional

from .evaluation_base import KSCriterion


class WeightedNodeStrengthKS(KSCriterion):
    """KS statistic comparing node strength distributions between networks.

    Node strength is the weighted equivalent of node degree - it is the sum of the
    weights of all edges connected to a node.
    """

    def __init__(self, normalise: Optional[bool] = True):
        """
        Args:
            normalise:
                If True, normalise the weights of the network by the maximum weight in the network. Defaults to True.
        """
        super().__init__()
        self.normalise = normalise

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes"]:
        """Compute strength for each node in the network.

        Args:
            matrix: Weight matrix of the network

        Returns:
            torch.Tensor: Vector of node strengths
        """
        if self.normalise:
            return matrix.sum(dim=1) / matrix.max()
        else:
            return matrix.sum(dim=1)


class WeightedBetweennessKS(KSCriterion):
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

    def __init__(self, normalise: Optional[bool] = True):
        """
        Args:
            normalise: If True, normalise the weights of the network by the maximum weight in the network. Defaults to True.
        """
        super().__init__()
        self.normalise = normalise

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes"]:
        """Compute weighted betweenness centrality for each node in the network.

        Args:
            matrix: Weight matrix of the network

        Returns:
            torch.Tensor: Vector of weighted betweenness centralities
        """
        if self.normalise:
            to_graph = matrix / matrix.max()
        else:
            to_graph = matrix

        # Convert to networkx for weighted betweenness calculation
        G = nx.from_numpy_array(to_graph)
        betweenness = nx.betweenness_centrality(G, weight="weight")
        # Convert dict to list preserving node order
        return torch.tensor([betweenness[i] for i in range(len(betweenness))])


class WeightedClusteringKS(KSCriterion):
    """KS statistic comparing weighted clustering coefficient distributions between networks.

    Implements the Onnela et al. (2005) definition of weighted clustering, which uses
    the geometric mean of triangle weights. For each node $u$, the clustering coefficient is:

    $$
    c_u = \\frac{1}{k_u(k_u-1)} \sum_{v,w} (\hat{w}_{uv} \\times \hat{w}_{uw} \\times \hat{w}_{vw})^(1/3),
    $$
    where $k_u$ is the node strength of node $u$, and $\hat{w}_{uv}$ is the weight of the edge between nodes $u$ and $v$,
    after normalising by dividing by the maximum weight in the network.
    """

    @jaxtyped(typechecker=typechecked)
    def _get_graph_statistics(
        self, matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes"]:
        """Compute weighted clustering coefficient for each node.

        Args:
            matrix: Weight matrix of the network

        Returns:
            torch.Tensor: Vector of weighted clustering coefficients
        """
        # Normalize weights by maximum weight
        normalized_matrix = matrix / matrix.max()

        # Convert to networkx graph
        G = nx.from_numpy_array(normalized_matrix.detach().cpu().numpy())

        # Compute clustering coefficients using Onnela's method
        clustering = nx.clustering(G, weight="weight")

        # Convert dict to list preserving node order
        return torch.tensor([clustering[i] for i in range(len(clustering))])
