from jaxtyping import Float, jaxtyped
from typeguard import typechecked
import torch
from abc import ABC, abstractmethod

from gnm.utils import binary_checks, weighted_checks


class GenerativeRule(ABC):
    """
    Base class for generative rules. Generative rules are used to compute the affinity factor between nodes in a graph.
    """

    @abstractmethod
    def __str__(self) -> str:
        pass

    @jaxtyped(typechecker=typechecked)
    def input_checks(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ):
        binary_checks(adjacency_matrix)

        # Check that the adjacency matrices have no self-connections
        batch_shape = adjacency_matrix.shape[:-2]
        num_nodes = adjacency_matrix.shape[-1]
        diagonal = torch.diagonal(adjacency_matrix, dim1=-2, dim2=-1)
        assert torch.allclose(
            diagonal, torch.zeros(*batch_shape, num_nodes)
        ), "Adjacency matrices should not have self-connections."

    @jaxtyped(typechecker=typechecked)
    def output_processing(
        self, affinity_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ):
        # Remove all self-connections from the affinity matrices
        diagonal_indices = torch.arange(affinity_matrix.shape[-1])
        affinity_matrix[..., diagonal_indices, diagonal_indices] = 0

        weighted_checks(affinity_matrix)

        return affinity_matrix

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        self.input_checks(adjacency_matrix)
        affinity_matrix = self._rule(adjacency_matrix)
        affinity_matrix = self.output_processing(affinity_matrix)
        return affinity_matrix

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        pass


class MatchingIndex(GenerativeRule):
    """
    Matching Index generative rule for computing affinity factor.
    Let $N(u)$ be the neighbourhood of node $u$.

    When the divisor is set to 'mean', the matching index is computed as:
    $$
        K(u,v) = \\frac{ | N(u) \cap N(v) | }{ ( |N(u) - \{v\}| + |N(v) - \{u\}| ) /2 }.
    $$
    When the divisor is set of 'union', the matching index is computed as:
    $$
        K(u,v) = \\frac{ | N(u) \cap N(v) | }{  | N(u) \cup N(v) - \{u,v\}| }.
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

    def __str__(self) -> str:
        return "Matching index"

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
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
    """
    Neighbours generative rule for computing affinity factor.
    $N(u)$ denotes the neighbourhood of node $u$.
    The affinity factor is computed as
    $$
        K(u,v) = | N(u) \cap N(v) | / |V|,
    $$
    where $|V|$ is the number of nodes in the graph.
    """

    def __str__(self) -> str:
        return "Neighbours"

    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        num_nodes = adjacency_matrix.shape[-1]
        return torch.matmul(adjacency_matrix, adjacency_matrix) / num_nodes


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

    @jaxtyped(typechecker=typechecked)
    def _clustering_coefficients(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes 1"]:
        degrees = adjacency_matrix.sum(dim=-1)
        number_of_pairs = degrees * (degrees - 1)

        number_of_triangles = torch.diagonal(
            torch.matmul(
                torch.matmul(adjacency_matrix, adjacency_matrix), adjacency_matrix
            ),
            dim1=-2,
            dim2=-1,
        )

        # Avoid division by zero or minus one
        number_of_pairs[number_of_pairs == 0] = 1
        clustering_coefficients = number_of_triangles / number_of_pairs
        return clustering_coefficients.unsqueeze(-1)


class ClusteringAverage(ClusteringRule):
    """
    Clustering Average generative rule for computing affinity factor.
    The affinity factor is computed as the average of the clustering coefficients of the two nodes:
    $$
        K(u,v) = (c_u + c_v) / 2.
    $$
    """

    def __str__(self) -> str:
        return "Clustering average"

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        clustering_coefficients = self._clustering_coefficients(adjacency_matrix)
        clustering_avg = (
            clustering_coefficients + clustering_coefficients.transpose(-2, -1)
        ) / 2
        return clustering_avg


class ClusteringDifference(ClusteringRule):
    """
    Clustering Difference generative rule for computing affinity factor.
    The affinity factor is computed as the absolute difference of the clustering coefficients of the two nodes:
    $$
        K(u,v) = |c_u - c_v|.
    $$
    """

    def __str__(self) -> str:
        return "Clustering difference"

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        clustering_coefficients = self._clustering_coefficients(adjacency_matrix)
        clustering_diff = torch.abs(
            clustering_coefficients - clustering_coefficients.transpose(-2, -1)
        )
        return clustering_diff


class ClusteringMax(ClusteringRule):
    """
    Clustering Max generative rule for computing affinity factor.
    The affinity factor is computed as the maximum of the clustering coefficients of the two nodes:
    $$
        K(u,v) = \\max(c_u, c_v).
    $$
    """

    def __str__(self) -> str:
        return "Clustering max"

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        clustering_coefficients = self._clustering_coefficients(adjacency_matrix)
        clustering_max = torch.maximum(
            clustering_coefficients, clustering_coefficients.transpose(-2, -1)
        )
        return clustering_max


class ClusteringMin(ClusteringRule):
    """
    Clustering Min generative rule for computing affinity factor.
    The affinity factor is computed as the minimum of the clustering coefficients of the two nodes:
    $$
        K(u,v) = \\min(c_u, c_v).
    $$
    """

    def __str__(self) -> str:
        return "Clustering minimum"

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        clustering_coefficients = self._clustering_coefficients(adjacency_matrix)
        clustering_min = torch.minimum(
            clustering_coefficients, clustering_coefficients.transpose(-2, -1)
        )
        return clustering_min


class ClusteringProduct(ClusteringRule):
    """
    Clustering Product generative rule for computing affinity factor.
    The affinity factor is computed as the product of the clustering coefficients of the two nodes:
    $$
        K(u,v) = c_u \\times c_v.
    $$
    """

    def __str__(self) -> str:
        return "Clustering product"

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        clustering_coefficients = self._clustering_coefficients(adjacency_matrix)
        clustering_product = (
            clustering_coefficients * clustering_coefficients.transpose(-2, -1)
        )
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
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes 1"]:
        num_nodes = adjacency_matrix.shape[-1]
        return adjacency_matrix.sum(dim=-1, keepdim=True) / num_nodes


class DegreeAverage(DegreeRule):
    """
    Degree Average generative rule for computing affinity factor.
    The affinity factor is computed as the average of the degrees of the two nodes:
    $$
        K(u,v) = (k_u + k_v) / 2.
    $$
    """

    def __str__(self) -> str:
        return "Degree average"

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        degrees = self._degrees(adjacency_matrix)
        return (degrees + degrees.transpose(-2, -1)) / 2


class DegreeDifference(DegreeRule):
    """
    Degree Difference generative rule for computing affinity factor.
    The affinity factor is computed as the absolute difference of the degrees of the two nodes:
    $$
        K(u,v) = |k_u - k_v|.
    $$
    """

    def __str__(self) -> str:
        return "Degree difference"

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        degrees = self._degrees(adjacency_matrix)
        return torch.abs(degrees - degrees.transpose(-2, -1))


class DegreeMax(DegreeRule):
    """
    Degree Max generative rule for computing affinity factor.
    The affinity factor is computed as the maximum of the degrees of the two nodes:
    $$
        K(u,v) = \\max(k_u,k_v).
    $$
    """

    def __str__(self) -> str:
        return "Degree maximum"

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        degrees = self._degrees(adjacency_matrix)
        return torch.maximum(degrees, degrees.transpose(-2, -1))


class DegreeMin(DegreeRule):
    """
    Degree Min generative rule for computing affinity factor.
    The affinity factor is computed as the minimum of the degrees of the two nodes:
    $$
        K(u,v) = \\min(k_u,k_v).
    $$
    """

    def __str__(self) -> str:
        return "Degree minimum"

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        degrees = self._degrees(adjacency_matrix)
        return torch.minimum(degrees, degrees.transpose(-2, -1))


class DegreeProduct(DegreeRule):
    """
    Degree Product generative rule for computing affinity factor.
    The affinity factor is computed as the product of the degrees of the two nodes:
    $$
        K(u,v) = k_u \\times k_v.
    $$
    """

    def __str__(self) -> str:
        return "Degree product"

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        degrees = self._degrees(adjacency_matrix)
        return degrees * degrees.transpose(-2, -1)


class Geometric(GenerativeRule):
    """
    Geometric generative rule for computing affinity factor.
    The affinity factor is constant:
    $$
        K(u,v) = 1.
    $$
    """

    def __str__(self) -> str:
        return "Geometric"

    @jaxtyped(typechecker=typechecked)
    def _rule(
        self, adjacency_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
        return torch.ones_like(adjacency_matrix)
