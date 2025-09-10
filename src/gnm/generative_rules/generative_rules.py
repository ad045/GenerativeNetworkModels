r"""Implementation of generative rules for network formation.

This module contains classes that define different rules for calculating connection
affinities between nodes in a network. Each rule produces an affinity matrix that
can be used in generative network models to determine how likely nodes are to form
connections with each other based on their topological properties.
"""

from jaxtyping import Float, jaxtyped
from typeguard import typechecked
import torch
from abc import ABC, abstractmethod

from gnm.utils import binary_checks, weighted_checks, binary_clustering_coefficients


class GenerativeRule(ABC):
    r"""Abstract base class for generative rules.

    Generative rules compute affinity factors between nodes in a network, which
    determine how likely nodes are to connect to each other. These affinity factors
    can be based on various network properties such as shared neighbors, clustering
    coefficients, or node degrees.

    This base class provides common functionality for input validation and output
    processing, ensuring that all rule implementations follow consistent behavior.

    Concrete classes inheriting from this abstract base must implement the `_rule' method.

    See Also:
        - [`model.GenerativeNetworkModel`][gnm.model.GenerativeNetworkModel]: Uses generative rules to build network models
    """

    def __str__(self) -> str:
        return self.__class__.__name__

    @jaxtyped(typechecker=typechecked)
    def input_checks(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    , device=None):
        r"""Validate input adjacency matrices.

        Performs checks to ensure that the input adjacency matrices are binary,
        symmetric, and have no self-connections.

        Args:
            adjacency_matrix:
                Batch of adjacency matrices to validate with shape
                [..., num_nodes, num_nodes]

        Raises:
            AssertionError: If any check fails
        """
        binary_checks(adjacency_matrix)

        # Check that the adjacency matrices have no self-connections
        batch_shape = adjacency_matrix.shape[:-2]
        num_nodes = adjacency_matrix.shape[-1]
        diagonal = torch.diagonal(adjacency_matrix, dim1=-2, dim2=-1)
        assert torch.allclose(
            diagonal,
            torch.zeros(*batch_shape, num_nodes, device=adjacency_matrix.device),
        ), "Adjacency matrices should not have self-connections."

    @jaxtyped(typechecker=typechecked)
    def output_processing(
        self, affinity_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    , device=None):
        r"""Process the output affinity matrix.

        Ensures that the affinity matrix has no self-connections and passes
        weighted matrix validation (non-negative, symmetric).

        Args:
            affinity_matrix:
                Batch of affinity matrices to process with shape
                [..., num_nodes, num_nodes]

        Returns:
            Processed affinity matrices with same shape

        Raises:
            AssertionError: If any check fails
        """
        # Remove all self-connections from the affinity matrices
        diagonal_indices = torch.arange(affinity_matrix.shape[-1])
        affinity_matrix[..., diagonal_indices, diagonal_indices] = 0

        weighted_checks(affinity_matrix)

        return affinity_matrix

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        r"""Apply the generative rule to compute affinity matrices.

        This method handles the complete process of validating inputs,
        applying the specific rule implementation, and processing outputs.

        Args:
            adjacency_matrix:
                Batch of adjacency matrices with shape [..., num_nodes, num_nodes]

        Returns:
            Affinity matrices with the same shape
        """
        self.input_checks(adjacency_matrix)
        affinity_matrix = self._rule(adjacency_matrix)
        affinity_matrix = self.output_processing(affinity_matrix)
        return affinity_matrix

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        r"""Apply the specific generative rule.

        This abstract method must be implemented by subclasses to define
        the specific rule for computing affinities between nodes.

        Args:
            adjacency_matrix:
                Batch of adjacency matrices with shape [..., num_nodes, num_nodes]

        Returns:
            Affinity matrices with the same shape
        """
        pass


class MatchingIndex(GenerativeRule):
    r"""Compute affinity based on shared neighborhoods between nodes.

    The matching index measures the similarity between the neighborhoods of two nodes,
    which is a form of homophily. Nodes with many common neighbors will have a high
    matching index, making them more likely to connect.

    Two calculation methods are supported:
    1. "mean" - Normalizes by the average size of the two neighborhoods
    2. "union" - Normalizes by the size of the union of the two neighborhoods

    Let $N(u)$ be the neighborhood of node $u$.

    When the divisor is set to 'mean', the matching index is computed as:
    $$
        K(u,v) = \frac{ | N(u) \cap N(v) | }{ ( |N(u) - \{v\}| + |N(v) - \{u\}| ) /2 }.
    $$

    When the divisor is set to 'union', the matching index is computed as:
    $$
        K(u,v) = \frac{ | N(u) \cap N(v) | }{  | N(u) \cup N(v) - \{u,v\}| }.
    $$

    When $N(u) - \{v\}$ and $N(v) - \{u\}$ are both empty, the matching index is zero.

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_binary_network
        >>> from gnm.generative_rules import MatchingIndex
        >>> # Load a default binary network
        >>> network = get_binary_network()
        >>> # Apply matching index with mean divisor
        >>> rule = MatchingIndex(divisor="mean")
        >>> affinity_matrix = rule(network)
        >>> # Apply matching index with union divisor
        >>> rule_union = MatchingIndex(divisor="union")
        >>> affinity_matrix_union = rule_union(network)

    See Also:
        - [`generative_rules.GenerativeRule`][gnm.generative_rules.GenerativeRule]: The base class for generative rules, from which this class inherits.
        - [`generative_rules.Neighbours`][gnm.generative_rules.Neighbours]: A simpler neighborhood-based rule
    """

    def __init__(self, divisor: str = "mean", device=None):
        r"""
        Args:
            divisor:
                Which division mode to use: 'union' or 'mean'. The 'union' mode
                normalises by the size of the union of neighborhoods, while 'mean'
                normalises by the average size of the two neighborhoods. Defaults to "mean".

        Raises:
            AssertionError: If divisor is not one of "mean" or "union".
        """
        self.divisor = divisor
        assert self.divisor in [
            "mean",
            "union",
        ], f"Divisor must be one of 'mean' or 'union'. Recieved {self.divisor}."

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        r"""Compute the matching index between all node pairs.

        Calculates the size of the neighborhood intersection and divides by
        the appropriate denominator based on the selected divisor method.

        Args:
            adjacency_matrix:
                Batch of adjacency matrices with shape [..., num_nodes, num_nodes]

        Returns:
            Matching index matrices with the same shape

        Raises:
            ValueError: If the divisor is not supported
        """
        if self.divisor == "mean":
            denominator = self._mean_divisor(adjacency_matrix)
        elif self.divisor == "union":
            denominator = self._union_divisor(adjacency_matrix)
        else:
            raise ValueError(
                f"Divisor must be one of 'mean' or 'union'. Divisor {self.divisor} not supported."
            )

        intersection = torch.matmul(
            adjacency_matrix.transpose(-2, -1), adjacency_matrix
        )

        matching_indices = intersection / denominator
        return matching_indices

    @jaxtyped(typechecker=typechecked)
    def _mean_divisor(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        r"""Compute denominators using the mean method.

        Calculates the average size of the two neighborhoods (excluding the nodes themselves)
        for all node pairs in the network.

        Args:
            adjacency_matrix:
                Batch of adjacency matrices with shape [..., num_nodes, num_nodes]

        Returns:
            Denominator matrix with the same shape
        """
        node_degrees = adjacency_matrix.sum(dim=-1)
        denominator = (
            node_degrees.unsqueeze(-2)
            + node_degrees.unsqueeze(-1)
            - adjacency_matrix
            - adjacency_matrix.transpose(-2, -1)
        ) / 2
        denominator[denominator == 0] = 1
        return denominator

    @jaxtyped(typechecker=typechecked)
    def _union_divisor(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        r"""Compute denominators using the union method.

        Calculates the size of the union of the two neighborhoods (excluding the nodes themselves)
        for all node pairs in the network.

        Args:
            adjacency_matrix:
                Batch of adjacency matrices with shape [..., num_nodes, num_nodes]

        Returns:
            Denominator matrix with the same shape
        """
        denominator = (
            torch.max(
                adjacency_matrix.unsqueeze(-2), adjacency_matrix.unsqueeze(-3)
            ).sum(dim=-1)
            - adjacency_matrix
            - adjacency_matrix.transpose(-2, -1)
        )
        denominator[denominator == 0] = 1
        return denominator


class Neighbours(GenerativeRule):
    r"""Compute affinity based on the number of shared neighbors.

    This rule is a simpler version of the matching index rule. Instead of normalising
    by neighborhood sizes, it normalises by the total number of nodes in the network.

    The affinity factor is computed as:
    $$
        K(u,v) = | N(u) \cap N(v) | / |V|,
    $$
    where $|V|$ is the number of nodes in the graph.

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_binary_network
        >>> from gnm.generative_rules import Neighbours
        >>> # Load a default binary network
        >>> network = get_binary_network()
        >>> # Apply neighbors rule
        >>> rule = Neighbours()
        >>> affinity_matrix = rule(network)

    See Also:
        - [`generative_rules.GenerativeRule`][gnm.generative_rules.GenerativeRule]: The base class for generative rules, from which this class inherits.
        - [`generative_rules.MatchingIndex`][gnm.generative_rules.MatchingIndex]: A more complex homophily rule.
    """

    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        r"""Compute the number of shared neighbors, normalized by network size.

        Args:
            adjacency_matrix:
                Batch of adjacency matrices with shape [..., num_nodes, num_nodes]

        Returns:
            Affinity matrices with the same shape
        """
        num_nodes = adjacency_matrix.shape[-1]
        return torch.matmul(adjacency_matrix, adjacency_matrix) / num_nodes


class ClusteringRule(GenerativeRule, ABC):
    r"""Base class for clustering coefficient-based generative rules.

    Clustering-based rules use the clustering coefficient of nodes to determine
    connection affinities. The clustering coefficient of each node measures how
    connected the neighbourhood of that node is.

    The clustering coefficient is computed as:
    $$
        c_u = \frac{t_u}{k_u(k_u - 1)},
    $$
    where $k_u$ is the degree of node $u$, and $t_u$ is the number of (directed)
    triangles around node $u$, computed as:
    $$
        t_u = \sum_{v,w} A_{uv}A_{vw}A_{wu}.
    $$

    Classes that inherit from this base class use the clustering coefficients to form
    various affinity factors based on different relationships between node pairs.

    See Also:
        - [`generative_rules.GenerativeRule`][gnm.generative_rules.GenerativeRule]: The base class for generative rules, from which this class inherits.
        - [`utils.binary_clustering_coefficients`][gnm.utils.binary_clustering_coefficients]: Function to compute clustering coefficients for binary networks.
    """

    @jaxtyped(typechecker=typechecked)
    def _clustering_coefficients(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes 1"]:
        r"""Calculate clustering coefficients for all nodes in the networks.

        Computes the clustering coefficient for each node, representing the
        fraction of possible triangles through that node that actually exist.

        Args:
            adjacency_matrix:
                Batch of adjacency matrices with shape [..., num_nodes, num_nodes]

        Returns:
            Tensor of clustering coefficients with shape [..., num_nodes, 1]
        """
        clustering_coefficients = binary_clustering_coefficients(adjacency_matrix)
        return clustering_coefficients.unsqueeze(-1)


class ClusteringAverage(ClusteringRule):
    r"""Compute affinity based on the average clustering coefficient of node pairs.

    This rule creates affinities based on the average of the clustering coefficients
    of the two nodes being considered for connection. Higher average clustering
    tends to create more locally clustered networks.

    The affinity factor is computed as the average of the clustering coefficients of the two nodes:
    $$
        K(u,v) = (c_u + c_v) / 2.
    $$

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_binary_network
        >>> from gnm.generative_rules import ClusteringAverage
        >>> # Load a default binary network
        >>> network = get_binary_network()
        >>> # Apply clustering average rule
        >>> rule = ClusteringAverage()
        >>> affinity_matrix = rule(network)

    See Also:
        - [`generative_rules.ClusteringRule`][gnm.generative_rules.ClusteringRule]: The base class for clustering coefficient-based rules.
        - [`generative_rules.DegreeAverage`][gnm.generative_rules.DegreeAverage]: Similar generative rule which uses degrees instead of clustering coefficients.
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        r"""Compute the average clustering coefficient between node pairs.

        Args:
            adjacency_matrix:
                Batch of adjacency matrices with shape [..., num_nodes, num_nodes]

        Returns:
            Affinity matrices with the same shape, where each entry is the average
            of the clustering coefficients of the corresponding nodes
        """
        clustering_coefficients = self._clustering_coefficients(adjacency_matrix)
        clustering_avg = (
            clustering_coefficients + clustering_coefficients.transpose(-2, -1)
        ) / 2
        return clustering_avg


class ClusteringDifference(ClusteringRule):
    r"""Compute affinity based on the difference in clustering coefficients between nodes.

    This rule creates affinities based on how different the clustering coefficients
    of two nodes are. It can be used to connect nodes with either similar (when using
    negative exponents in the generative model) or dissimilar (with positive exponents)
    clustering properties.

    The affinity factor is computed as the absolute difference of the clustering coefficients:
    $$
        K(u,v) = |c_u - c_v|.
    $$

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_binary_network
        >>> from gnm.generative_rules import ClusteringDifference
        >>> # Load a default binary network
        >>> network = get_binary_network()
        >>> # Apply clustering difference rule
        >>> rule = ClusteringDifference()
        >>> affinity_matrix = rule(network)

    See Also:
        - [`generative_rules.ClusteringRule`][gnm.generative_rules.ClusteringRule]: The base class for clustering coefficient-based rules.
        - [`generative_rules.DegreeDifference`][gnm.generative_rules.DegreeDifference]: Similar generative rule which uses degrees instead of clustering coefficients.
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        r"""Compute the absolute difference in clustering coefficients between node pairs.

        Args:
            adjacency_matrix:
                Batch of adjacency matrices with shape [..., num_nodes, num_nodes]

        Returns:
            Affinity matrices with the same shape, where each entry is the absolute
            difference between the clustering coefficients of the corresponding nodes
        """
        clustering_coefficients = self._clustering_coefficients(adjacency_matrix)
        clustering_diff = torch.abs(
            clustering_coefficients - clustering_coefficients.transpose(-2, -1)
        )
        return clustering_diff


class ClusteringMax(ClusteringRule):
    r"""Compute affinity based on the maximum clustering coefficient between nodes.

    This rule creates affinities based on the maximum of the clustering coefficients
    of the two nodes being considered for connection. It tends to favor connections
    where at least one node has high clustering.

    The affinity factor is computed as the maximum of the clustering coefficients:
    $$
        K(u,v) = \max(c_u, c_v).
    $$

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_binary_network
        >>> from gnm.generative_rules import ClusteringMax
        >>> # Load a default binary network
        >>> network = get_binary_network()
        >>> # Apply clustering max rule
        >>> rule = ClusteringMax()
        >>> affinity_matrix = rule(network)

    See Also:
        - [`generative_rules.ClusteringRule`][gnm.generative_rules.ClusteringRule]: The base class for clustering coefficient-based rules.
        - [`generative_rules.DegreeMax`][gnm.generative_rules.DegreeMax]: Similar generative rule which uses degrees instead of clustering coefficients.
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        r"""Compute the maximum clustering coefficient between node pairs.

        Args:
            adjacency_matrix:
                Batch of adjacency matrices with shape [..., num_nodes, num_nodes]

        Returns:
            Affinity matrices with the same shape, where each entry is the maximum
            of the clustering coefficients of the corresponding nodes
        """
        clustering_coefficients = self._clustering_coefficients(adjacency_matrix)
        clustering_max = torch.maximum(
            clustering_coefficients, clustering_coefficients.transpose(-2, -1)
        )
        return clustering_max


class ClusteringMin(ClusteringRule):
    r"""Compute affinity based on the minimum clustering coefficient between nodes.

    This rule creates affinities based on the minimum of the clustering coefficients
    of the two nodes being considered for connection. It tends to favor connections
    where both nodes have reasonably high clustering.

    The affinity factor is computed as the minimum of the clustering coefficients:
    $$
        K(u,v) = \min(c_u, c_v).
    $$

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_binary_network
        >>> from gnm.generative_rules import ClusteringMin
        >>> # Load a default binary network
        >>> network = get_binary_network()
        >>> # Apply clustering min rule
        >>> rule = ClusteringMin()
        >>> affinity_matrix = rule(network)

    See Also:
        - [`generative_rules.ClusteringRule`][gnm.generative_rules.ClusteringRule]: The base class for clustering coefficient-based rules.
        - [`generative_rules.DegreeMin`][gnm.generative_rules.DegreeMin]: Similar generative rule which uses degrees instead of clustering coefficients.
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        r"""Compute the minimum clustering coefficient between node pairs.

        Args:
            adjacency_matrix:
                Batch of adjacency matrices with shape [..., num_nodes, num_nodes]

        Returns:
            Affinity matrices with the same shape, where each entry is the minimum
            of the clustering coefficients of the corresponding nodes
        """
        clustering_coefficients = self._clustering_coefficients(adjacency_matrix)
        clustering_min = torch.minimum(
            clustering_coefficients, clustering_coefficients.transpose(-2, -1)
        )
        return clustering_min


class ClusteringProduct(ClusteringRule):
    r"""Compute affinity based on the product of clustering coefficients between nodes.

    This rule creates affinities based on the product of the clustering coefficients
    of the two nodes being considered for connection. It strongly favors connections
    between nodes that both have high clustering.

    The affinity factor is computed as the product of the clustering coefficients:
    $$
        K(u,v) = c_u \times c_v.
    $$

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_binary_network
        >>> from gnm.generative_rules import ClusteringProduct
        >>> # Load a default binary network
        >>> network = get_binary_network()[0]  # Get first network from batch
        >>> # Apply clustering product rule
        >>> rule = ClusteringProduct()
        >>> affinity_matrix = rule(network.unsqueeze(0))

    See Also:
        - [`generative_rules.ClusteringAverage`][gnm.generative_rules.ClusteringAverage]: Uses average of clustering coefficients
        - [`generative_rules.DegreeProduct`][gnm.generative_rules.DegreeProduct]: Similar principle applied to degrees
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        r"""Compute the product of clustering coefficients between node pairs.

        Args:
            adjacency_matrix:
                Batch of adjacency matrices with shape [..., num_nodes, num_nodes]

        Returns:
            Affinity matrices with the same shape, where each entry is the product
            of the clustering coefficients of the corresponding nodes
        """
        clustering_coefficients = self._clustering_coefficients(adjacency_matrix)
        clustering_product = (
            clustering_coefficients * clustering_coefficients.transpose(-2, -1)
        )
        return clustering_product


class DegreeRule(GenerativeRule, ABC):
    r"""Base class for degree-based generative rules.

    Degree-based rules use the degree (number of connections) of nodes to determine
    connection affinities. These rules can create various network structures by
    favoring connections between nodes with specific degree relationships.

    The (normalised) degree of a node $u$ is computed as:
    $$
        k_u = \frac{1}{|V|} \sum_{v} A_{uv}.
    $$

    The division by $|V|$ (number of nodes) ensures that the degree is between 0 and 1.

    Classes which inherit from this base class use the normalised degrees to form
    the affinity factor using different relationships between node pairs.

    See Also:
        - [`generative_rules.GenerativeRule`][gnm.generative_rules.GenerativeRule]: The base class for generative rules, from which this class inherits.
    """

    @jaxtyped(typechecker=typechecked)
    def _degrees(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes 1"]:
        r"""Calculate normalised degrees for all nodes in the networks.

        Computes the degree for each node, normalised by the number of nodes
        in the network to ensure values between 0 and 1.

        Args:
            adjacency_matrix:
                Batch of adjacency matrices with shape [..., num_nodes, num_nodes]

        Returns:
            Tensor of normalised degrees with shape [..., num_nodes, 1]
        """
        num_nodes = adjacency_matrix.shape[-1]
        return adjacency_matrix.sum(dim=-1, keepdim=True) / num_nodes


class DegreeAverage(DegreeRule):
    r"""Compute affinity based on the average degree of node pairs.

    This rule creates affinities based on the average of the degrees of the two
    nodes being considered for connection. It tends to create networks with more
    uniform degree distributions.

    The affinity factor is computed as the average of the normalised degrees:
    $$
        K(u,v) = (k_u + k_v) / 2.
    $$

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_binary_network
        >>> from gnm.generative_rules import DegreeAverage
        >>> # Load a default binary network
        >>> network = get_binary_network()
        >>> # Apply degree average rule
        >>> rule = DegreeAverage()
        >>> affinity_matrix = rule(network)

    See Also:
        - [`generative_rules.DegreeRule`][gnm.generative_rules.DegreeRule]: The base class for degree-based rules, from which this class inherits.
        - [`generative_rules.ClusteringAverage`][gnm.generative_rules.ClusteringAverage]: Similar generative rule which uses clustering coefficients instead of degrees.
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        r"""Compute the average degree between node pairs.

        Args:
            adjacency_matrix:
                Batch of adjacency matrices with shape [..., num_nodes, num_nodes]

        Returns:
            Affinity matrices with the same shape, where each entry is the average
            of the normalised degrees of the corresponding nodes
        """
        degrees = self._degrees(adjacency_matrix)
        return (degrees + degrees.transpose(-2, -1)) / 2


class DegreeDifference(DegreeRule):
    r"""Compute affinity based on the difference in degrees between nodes.

    This rule creates affinities based on how different the degrees of two nodes are.
    It can be used to connect nodes with either similar (when using negative exponents
    in the generative model) or dissimilar (with positive exponents) connectivity patterns.

    The affinity factor is computed as the absolute difference of the normalised degrees:
    $$
        K(u,v) = |k_u - k_v|.
    $$

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_binary_network
        >>> from gnm.generative_rules import DegreeDifference
        >>> # Load a default binary network
        >>> network = get_binary_network()
        >>> # Apply degree difference rule
        >>> rule = DegreeDifference()
        >>> affinity_matrix = rule(network)

    See Also:
        - [`generative_rules.DegreeRule`][gnm.generative_rules.DegreeRule]: The base class for degree-based rules, from which this class inherits.
        - [`generative_rules.ClusteringDifference`][gnm.generative_rules.ClusteringDifference]: Similar generative rule which uses clustering coefficients instead of degrees.
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        r"""Compute the absolute difference in degrees between node pairs.

        Args:
            adjacency_matrix:
                Batch of adjacency matrices with shape [..., num_nodes, num_nodes]

        Returns:
            Affinity matrices with the same shape, where each entry is the absolute
            difference between the normalised degrees of the corresponding nodes
        """
        degrees = self._degrees(adjacency_matrix)
        return torch.abs(degrees - degrees.transpose(-2, -1))


class DegreeMax(DegreeRule):
    r"""Compute affinity based on the maximum degree between nodes.

    This rule creates affinities based on the maximum of the degrees of the two
    nodes being considered for connection. It tends to favor connections where
    at least one node has high degree.

    The affinity factor is computed as the maximum of the normalised degrees:
    $$
        K(u,v) = \max(k_u,k_v).
    $$

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_binary_network
        >>> from gnm.generative_rules import DegreeMax
        >>> # Load a default binary network
        >>> network = get_binary_network()
        >>> # Apply degree max rule
        >>> rule = DegreeMax()
        >>> affinity_matrix = rule(network)

    See Also:
        - [`generative_rules.DegreeRule`][gnm.generative_rules.DegreeRule]: The base class for degree-based rules, from which this class inherits.
        - [`generative_rules.ClusteringMax`][gnm.generative_rules.ClusteringMax]: Similar generative rule which uses clustering coefficients instead of degrees.
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        r"""Compute the maximum degree between node pairs.

        Args:
            adjacency_matrix:
                Batch of adjacency matrices with shape [..., num_nodes, num_nodes]

        Returns:
            Affinity matrices with the same shape, where each entry is the maximum
            of the normalised degrees of the corresponding nodes
        """
        degrees = self._degrees(adjacency_matrix)
        return torch.maximum(degrees, degrees.transpose(-2, -1))


class DegreeMin(DegreeRule):
    r"""Compute affinity based on the minimum degree between nodes.

    This rule creates affinities based on the minimum of the degrees of the two
    nodes being considered for connection. It tends to favor connections where
    both nodes have reasonably high degree.

    The affinity factor is computed as the minimum of the normalised degrees:
    $$
        K(u,v) = \min(k_u,k_v).
    $$

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_binary_network
        >>> from gnm.generative_rules import DegreeMin
        >>> # Load a default binary network
        >>> network = get_binary_network()
        >>> # Apply degree min rule
        >>> rule = DegreeMin()
        >>> affinity_matrix = rule(network)

    See Also:
        - [`generative_rules.DegreeRule`][gnm.generative_rules.DegreeRule]: The base class for degree-based rules, from which this class inherits.
        - [`generative_rules.ClusteringMin`][gnm.generative_rules.ClusteringMin]: Similar generative rule which uses clustering coefficients instead of degrees.
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        r"""Compute the minimum degree between node pairs.

        Args:
            adjacency_matrix:
                Batch of adjacency matrices with shape [..., num_nodes, num_nodes]

        Returns:
            Affinity matrices with the same shape, where each entry is the minimum
            of the normalised degrees of the corresponding nodes
        """
        degrees = self._degrees(adjacency_matrix)
        return torch.minimum(degrees, degrees.transpose(-2, -1))


class DegreeProduct(DegreeRule):
    r"""Compute affinity based on the product of degrees between nodes.

    This rule creates affinities based on the product of the degrees of the two
    nodes being considered for connection. It strongly favors connections between
    nodes that both have high degree, potentially leading to rich-club structures.

    The affinity factor is computed as the product of the normalised degrees:
    $$
        K(u,v) = k_u \times k_v.
    $$

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_binary_network
        >>> from gnm.generative_rules import DegreeProduct
        >>> # Load a default binary network
        >>> network = get_binary_network()
        >>> # Apply degree product rule
        >>> rule = DegreeProduct()
        >>> affinity_matrix = rule(network)

    See Also:
        - [`generative_rules.DegreeRule`][gnm.generative_rules.DegreeRule]: The base class for degree-based rules, from which this class inherits.
        - [`generative_rules.ClusteringProduct`][gnm.generative_rules.ClusteringProduct]: Similar principle applied to clustering coefficients
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        r"""Compute the product of degrees between node pairs.

        Args:
            adjacency_matrix:
                Batch of adjacency matrices with shape [..., num_nodes, num_nodes]

        Returns:
            Affinity matrices with the same shape, where each entry is the product
            of the normalised degrees of the corresponding nodes
        """
        degrees = self._degrees(adjacency_matrix)
        return degrees * degrees.transpose(-2, -1)


class Geometric(GenerativeRule):
    r"""A baseline generative rule that assigns equal affinity to all node pairs.

    This rule creates a constant affinity matrix, meaning all potential connections
    have equal probability from the perspective of this rule. When used in a generative
    model, other factors like distance will determine connection formation.

    The affinity factor is constant:
    $$
        K(u,v) = 1.
    $$

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_binary_network
        >>> from gnm.generative_rules import Geometric
        >>> # Load a default binary network
        >>> network = get_binary_network()
        >>> # Apply geometric rule
        >>> rule = Geometric()
        >>> affinity_matrix = rule(network)
        >>> # All non-diagonal elements should be 1
        >>> torch.all(affinity_matrix[0][~torch.eye(network.shape[0], dtype=bool)] == 1)
        tensor(True)

    See Also:
        - [`generative_rules.GenerativeRule`][gnm.generative_rules.GenerativeRule]: The base class for generative rules, from which this class inherits.
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        r"""Create a constant affinity matrix with all entries set to 1.

        Args:
            adjacency_matrix:
                Batch of adjacency matrices with shape [..., num_nodes, num_nodes]

        Returns:
            Affinity matrices with the same shape, filled with ones
        """
        return torch.ones_like(adjacency_matrix)
