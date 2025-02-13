from jaxtyping import Float, jaxtyped
from typeguard import typechecked
import torch
from abc import ABC, abstractmethod


class GenerativeRule(ABC):
    """
    Base class for generative rules. Generative rules are used to compute the affinity factor between nodes in a graph.
    """

    @jaxtyped(typechecker=typechecked)
    def input_checks(
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ):
        # Check that the adjacency matrix is binary
        assert torch.allclose(
            adjacency_matrix,
            torch.where(adjacency_matrix != 0, torch.tensor(1), torch.tensor(0)),
        ), "Adjacency matrix should be binary."

        # Check that the adjacency matrix is symmetric
        assert torch.allclose(
            adjacency_matrix, adjacency_matrix.T
        ), "Adjacency matrix should be symmetric."

        # Check that the adjacency matrix has no self-connections
        assert torch.allclose(
            torch.diag(adjacency_matrix), torch.zeros_like(torch.diag(adjacency_matrix))
        ), "Adjacency matrix should not have self-connections."

    @jaxtyped(typechecker=typechecked)
    def output_processing(
        self, affinity_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ):
        # Check that the affinity matrix is symmetric
        assert torch.allclose(
            affinity_matrix, affinity_matrix.T
        ), "Affinity matrix should be symmetric."

        # Check that the affinity matrix is non-negative
        assert torch.all(
            affinity_matrix >= 0
        ), "Affinity matrix should be non-negative."

        # Remove all self-connections from the affinity matrix
        affinity_matrix.fill_diagonal_(0)

        return affinity_matrix

    # pipeline for applying rule to adjacency matrix
    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        self.input_checks(adjacency_matrix)
        affinity_matrix = self._rule(adjacency_matrix)
        affinity_matrix = self.output_processing(affinity_matrix)
        return affinity_matrix

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        pass


class MatchingIndex(GenerativeRule):
    """
    Matching Index generative rule for computing affinity factor.
    Let $N(u)$ be the neighbourhood of node $u$.

    When the divisor is set to 'mean', the matching index is computed as:
    $$
        K(u,v) = \\frac{ | N(u) \cap N(v) | }{ ( |N(u) - \{v\}| + |N(v) - \{u\}| ) /2 }
    $$
    when the divisor is set of 'union', the matching index is computed as:
    $$
        K(u,v) = \\frac{ | N(u) \cap N(v) | }{  | N(u) \cup N(v) - \{u,v\}| )
    $$
    When $N(u) - \{v\}$ and $N(v) - \{u\}$ are both empty the matching index is zero.
    """

    def __init__(self, divisor: str = "mean"):
        """
        Args:
            divisor:
                Which division mode to use (e.g., 'union' or 'mean'). Defaults to "mean".

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
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        if self.divisor == "mean":
            denominator = self._mean_divisor(adjacency_matrix)
        elif self.divisor == "union":
            denominator = self._union_divisor(adjacency_matrix)
        else:
            raise ValueError(
                f"Divisor must be one of 'mean' or 'union'. Divisor {self.divisor} not supported."
            )
        intersection = adjacency_matrix.T @ adjacency_matrix

        # apply normalization, get matching index, remove self-connections
        matching_indices = intersection / denominator
        return matching_indices

    @jaxtyped(typechecker=typechecked)
    def _mean_divisor(
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        node_degrees = adjacency_matrix.sum(dim=0)
        denominator = (
            node_degrees.unsqueeze(0)
            + node_degrees.unsqueeze(1)
            - adjacency_matrix
            - adjacency_matrix.T
        ) / 2
        denominator[denominator == 0] = 1
        return denominator

    @jaxtyped(typechecker=typechecked)
    def _union_divisor(
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        denominator = (
            torch.max(adjacency_matrix.unsqueeze(1), adjacency_matrix.unsqueeze(2)).sum(
                dim=0
            )
            - adjacency_matrix
            - adjacency_matrix.T
        )
        denominator[denominator == 0] = 1
        return denominator


class Neighbours(GenerativeRule):
    """
    Neighbours generative rule for computing affinity factor.
    $N(u)$ denotes the neighbourhood of node $u$.
    The affinity factor is computed as
    $$
        K(u,v) = | N(u) \cap N(v) | / |V|,
    $$
    where $|V|$ is the number of nodes in the graph.
    """

    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        num_nodes = adjacency_matrix.shape[0]
        return (adjacency_matrix @ adjacency_matrix) / num_nodes


class ClusteringRule(GenerativeRule, ABC):
    """Base class for clustering rules.
    The clustering coefficient is computed as:
    $$
        c_u = \\frac{t_u}{k_u(k_u - 1)},
    $$
    where $k_u$ is the degree of node $u$, and $t_u$ is the number of (directed) triangles around node $u$, computed as:
    $$
        t_u = \sum_{v,w} A_{uv}A_{vw}A_{wu}.
    $$
    Classes which inherit from this base class use the clustering coefficients to form the affinity factor.
    """

    @jaxtyped
    def _clustering_coefficients(
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes 1"]:
        degrees = adjacency_matrix.sum(dim=1)
        # Compute the number of (ordered) pairs of neighbors for each node
        number_of_pairs = degrees * (degrees - 1)
        # Compute the number of (directed) triangles around each node
        number_of_triangles = torch.diag(
            adjacency_matrix @ adjacency_matrix @ adjacency_matrix
        )
        clustering_coefficients = number_of_triangles / number_of_pairs
        clustering_coefficients.unsqueeze_(1)
        return clustering_coefficients


class ClusteringAverage(ClusteringRule):
    """
    Clustering Average generative rule for computing affinity factor.
    The affinity factor is computed as the average of the clustering coefficients of the two nodes:
    $$
        K(u,v) = (c_u + c_v) / 2.
    $$
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        clustering_coefficents = self._clustering_coefficients(
            adjacency_matrix
        )  # This has shape (num_nodes, 1)
        clustering_avg = (clustering_coefficents + clustering_coefficents.T) / 2
        return clustering_avg


class ClusteringDifference(ClusteringRule):
    """
    Clustering Difference generative rule for computing affinity factor.
    The affinity factor is computed as the absolute difference of the clustering coefficients of the two nodes:
    $$
        K(u,v) = |c_u - c_v|.
    $$
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        clustering_coefficents = self._clustering_coefficients(adjacency_matrix)
        clustering_diff = torch.abs(clustering_coefficents - clustering_coefficents.T)
        return clustering_diff


class ClusteringMax(ClusteringRule):
    """
    Clustering Max generative rule for computing affinity factor.
    The affinity factor is computed as the maximum of the clustering coefficients of the two nodes:
    $$
        K(u,v) = \\max(c_u, c_v).
    $$
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        clustering_coefficents = self._clustering_coefficients(adjacency_matrix)
        clustering_max = torch.max(clustering_coefficents, clustering_coefficents.T)
        return clustering_max


class ClusteringMin(ClusteringRule):
    """
    Clustering Min generative rule for computing affinity factor.
    The affinity factor is computed as the minimum of the clustering coefficients of the two nodes:
    $$
        K(u,v) = \\min(c_u, c_v).
    $$
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        clustering_coefficents = self._clustering_coefficients(adjacency_matrix)
        clustering_min = torch.min(clustering_coefficents, clustering_coefficents.T)
        return clustering_min


class ClusteringProduct(ClusteringRule):
    """
    Clustering Product generative rule for computing affinity factor.
    The affinity factor is computed as the product of the clustering coefficients of the two nodes:
    $$
        K(u,v) = c_u \\times c_v.
    $$
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        clustering_coefficents = self._clustering_coefficients(adjacency_matrix)
        clustering_product = clustering_coefficents * clustering_coefficents.T
        return clustering_product


class DegreeRule(GenerativeRule, ABC):
    """
    Base class for degree-based generative rules.
    Classes which inherit from this base class use the (normalised) degrees of the nodes to form the affinity factor.
    The (normalised) degree of a node $u$ is computed as:
    $$
        k_u = \\frac{1}{|V|} \\sum_{v} A_{uv}.
    $$
    The division by $|V|$ ensures that the degree is between 0 and 1.
    """

    @jaxtyped(typechecker=typechecked)
    def _degrees(
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes 1"]:
        num_nodes = adjacency_matrix.shape[0]
        return adjacency_matrix.sum(dim=1, keepdim=True) / num_nodes


class DegreeAverage(DegreeRule):
    """
    Degree Average generative rule for computing affinity factor.
    The affinity factor is computed as the average of the degrees of the two nodes:
    $$
        K(u,v) = (k_u + k_v) / 2.
    $$
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tenosr, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        degrees = self._degrees(adjacency_matrix)
        return (degrees + degrees.T) / 2


class DegreeDifference(DegreeRule):
    """
    Degree Difference generative rule for computing affinity factor.
    The affinity factor is computed as the absolute difference of the degrees of the two nodes:
    $$
        K(u,v) = |k_u - k_v|.
    $$
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        degrees = self._degrees(adjacency_matrix)
        return torch.abs(degrees - degrees.T)


class DegreeMax(DegreeRule):
    """
    Degree Max generative rule for computing affinity factor.
    The affinity factor is computed as the maximum of the degrees of the two nodes:
    $$
        K(u,v) = \\max(k_u,k_v).
    $$
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        degrees = self._degrees(adjacency_matrix)
        return torch.max(degrees, degrees.T)


class DegreeMin(DegreeRule):
    """
    Degree Min generative rule for computing affinity factor.
    The affinity factor is computed as the minimum of the degrees of the two nodes:
    $$
        K(u,v) = \\min(k_u,k_v).
    $$
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        degrees = self._degrees(adjacency_matrix)
        return torch.min(degrees, degrees.T)


class DegreeProduct(DegreeRule):
    """
    Degree Product generative rule for computing affinity factor.
    The affinity factor is computed as the product of the degrees of the two nodes:
    $$
        K(u,v) = k_u \\times k_v.
    $$
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        degrees = self._degrees(adjacency_matrix)
        return degrees * degrees.T


class Geometric(GenerativeRule):
    """
    Geometric generative rule for computing affinity factor.
    The affinity factor is constant:
    $$
        K(u,v) = 1.
    $$
    """

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        return torch.ones_like(adjacency_matrix)
