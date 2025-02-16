import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from abc import ABC, abstractmethod


class OptimisationCriterion(ABC):
    """Base abstract class for optimisation criteria used in weighted generative networks.

    This class provides a framework for defining various optimisation objectives, $L(W)$ used
    to evolve weights in the weighted generative network model.
    """

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "..."]:
        """Compute the final criterion $L(W)$ for optimisation of the network weights."""
        pass


@jaxtyped(typechecker=typechecked)
def compute_communicability(
    weight_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
) -> Float[torch.Tensor, "... num_nodes num_nodes"]:
    """Communicability optimisation criterion.
    To compute the communicability matrix, we go through the following steps:

    1. Compute the diagonal node strength matrix, $S_{ii} = \sum_j W_{ij}$ (plus a small constant to prevent division by zero).
    2. Compute the normalised weight matrix, $S^{-1/2} W S^{-1/2}$.
    3. Compute the communicability matrix by taking the matrix exponential, $\exp( S^{-1/2} W S^{-1/2} )$.
    """
    # Compute the node strengths, with a small constant addition to prevent division by zero.
    node_strengths = (
        0.5 * (weight_matrix.sum(dim=-1) + weight_matrix.sum(dim=-2)) + 1e-6
    )

    # Create diagonal matrix for each batch element
    batch_shape = weight_matrix.shape[:-2]
    num_nodes = weight_matrix.shape[-1]
    inv_sqrt_node_strengths = torch.zeros(
        *batch_shape, num_nodes, num_nodes, device=weight_matrix.device
    )

    # Set diagonal values for each batch element
    diag_indices = torch.arange(num_nodes)
    inv_sqrt_node_strengths[..., diag_indices, diag_indices] = 1.0 / torch.sqrt(
        node_strengths
    )

    # Compute the normalised weight matrix
    normalised_weight_matrix = torch.matmul(
        torch.matmul(inv_sqrt_node_strengths, weight_matrix), inv_sqrt_node_strengths
    )

    # Compute the communicability matrix
    communicability_matrix = torch.matrix_exp(normalised_weight_matrix)
    return communicability_matrix


class Communicability(OptimisationCriterion):
    """Communicability optimisation criterion.
    To compute this optimisation criterion, we go through the following steps:

    1. Compute the diagonal node strength matrix, $S_{ii} = \sum_j W_{ij}$ (plus a small constant to prevent division by zero).
    2. Compute the normalised weight matrix, $S^{-1/2} W S^{-1/2}$.
    3. Compute the communicability matrix by taking the matrix exponential, $\exp( S^{-1/2} W S^{-1/2} )$.
    4. Raise each element of this product to the power of $\omega$, $\exp( S^{-1/2} W S^{-1/2} )_{ij}^\omega$.
    5. Sum over the elements of the communicability matrix rasied to the power of $\omega$ to get the loss.

    The loss $L(W)$ is then given by:
    $$
    L(W) = \sum_{ij} \\left( \exp( S^{-1/2} W S^{-1/2} )_{ij} \\right)^\omega
    $$

    See Also:

    Note:
        This class is a callable.

    Examples:
        >>> criterion = Communicability(normalisation=False, omega=1.0)
        >>> weight_matrix = torch.rand(10, 10)
        >>> loss = criterion(weight_matrix)
    """

    def __init__(self, omega: float = 1.0):
        """
        Args:
            omega:
                The power to which to raise each element of the communicability matrix before performing the sum. Defaults to 1.0.
        """
        self.omega = omega

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "..."]:
        communicability_matrix = compute_communicability(weight_matrix)
        tilted_communicability = torch.pow(communicability_matrix, self.omega)
        return torch.sum(tilted_communicability, dim=(-2, -1))


class NormalisedCommunicability(OptimisationCriterion):
    """Communicability optimisation criterion.
    To compute this optimisation criterion, we go through the following steps:

    1. Compute the diagonal node strength matrix, $S_{ii} = \sum_j W_{ij}$ (plus a small constant to prevent division by zero).
    2. Compute the normalised weight matrix, $S^{-1/2} W S^{-1/2}$.
    3. Compute the communicability matrix by taking the matrix exponential, $\exp( S^{-1/2} W S^{-1/2} )$.
    4. Raise each element of this product to the power of $\omega$, $\exp( S^{-1/2} W S^{-1/2} )_{ij}^\omega$.
    5. Normalise by dividing by the maximum element.
    6. Sum over the elements of the normalised communicability matrix rasied to the power of $\omega$ to get the loss.

    The loss is then given by:
    $$
    L(W) = \\frac{ \sum_{ij} \exp( S^{-1/2} W S^{-1/2} )_{ij}^\omega }{ \max_{ij} \exp( S^{-1/2} W S^{-1/2} )_{ij}^\omega }
    $$

    See Also:

    Note:
        This class is a callable.

    Examples:
        >>> criterion = Communicability(normalisation=True, omega=1.0)
        >>> weight_matrix = torch.rand(10, 10)
        >>> loss = criterion(weight_matrix)
    """

    def __init__(self, omega: float = 1.0):
        """
        Args:
            omega:
                The power to which to raise each element of the communicability matrix before performing the sum. Defaults to 1.0.
        """
        self.omega = omega

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "..."]:
        communicability_matrix = compute_communicability(weight_matrix)
        tilted_communicability = torch.pow(communicability_matrix, self.omega)
        max_tilted_communicability = torch.amax(
            tilted_communicability, dim=(-2, -1), keepdim=True
        )
        normalised_tilted_communicability = (
            tilted_communicability / max_tilted_communicability
        )
        return torch.sum(normalised_tilted_communicability, dim=(-2, -1))


class DistanceWeightedCommunicability(OptimisationCriterion):
    """Distance Weighted Communicability optimisation criterion.
    To compute this optimisation criterion, we go through the following steps:

    1. Compute the diagonal node strength matrix, $S_{ii} = \sum_j W_{ij}$ (plus a small constant to prevent division by zero).
    2. Compute the normalised weight matrix, $S^{-1/2} W S^{-1/2}$.
    3. Compute the communicability matrix by taking the matrix exponential, $\exp( S^{-1/2} W S^{-1/2} )$.
    4. Take the element-wise product of the communicability matrix and the distance matrix, $\exp( S^{-1/2} W S^{-1/2} ) \odot D$
    5. Raise each element of this product to the power of $\omega$, $(\exp( S^{-1/2} W S^{-1/2} ) \odot D)_{ij}^\omega$.
    6. Sum over the elements of the distance-weighted communicability matrix rasied to the power of $\omega$ to get the loss.

    The loss is then given by:
    $$
    L(W) = \sum_{ij} \\left( \exp( S^{-1/2} W S^{-1/2} )_{ij} D_{ij} \\right)^\omega
    $$

    See Also:

    Notes:
        This class is a callable.
    """

    def __init__(
        self,
        distance_matrix: Float[torch.Tensor, "num_nodes num_nodes"],
        omega: float = 1.0,
    ):
        """
        Args:
            distance_matrix:
                The distance matrix of the network.
            omega:
                The power to which to raise each element of the distance weighted communicability before performing
                the sum. Defaults to 1.0."""
        self.distance_matrix = distance_matrix
        self.omega = omega

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "..."]:
        """
        Computes the distance-weighted communicability of a network with a given weight matrix.

        Args:
            weight_matrix:
                The weight matrix of the network.

        Returns:
            distance_weighted_communicability:
                The distance-weighted communicability matrix of the network
        """
        communicability_matrix = compute_communicability(weight_matrix)
        # Expand distance matrix to match batch dimensions
        expanded_distance = self.distance_matrix.expand_as(communicability_matrix)
        distance_weighted_communicability = torch.pow(
            communicability_matrix * expanded_distance, self.omega
        )
        return torch.sum(distance_weighted_communicability, dim=(-2, -1))


class NormalisedDistanceWeightedCommunicability(OptimisationCriterion):
    """Normalised Distance Weighted Communicability optimisation criterion.
    To compute this optimisation criterion, we go through the following steps:

    1. Compute the diagonal node strength matrix, $S_{ii} = \sum_j W_{ij}$ (plus a small constant to prevent division by zero).
    2. Compute the normalised weight matrix, $S^{-1/2} W S^{-1/2}$.
    3. Compute the communicability matrix by taking the matrix exponential, $\exp( S^{-1/2} W S^{-1/2} )$.
    4. Take the element-wise product of the communicability matrix and the distance matrix, $\exp( S^{-1/2} W S^{-1/2} ) \odot D$
    5. Raise each element of this product to the power of $\omega$, $(\exp( S^{-1/2} W S^{-1/2} ) \odot D)_{ij}^\omega$.
    6. Normalise by dividing by the maximum element.
    7. Sum over the elements of the distance-weighted communicability matrix rasied to the power of $\omega$ to get the loss.

    The loss is then given by:
    $$
    L(W) = \\frac{ \sum_{ij} \\left( \exp( S^{-1/2} W S^{-1/2} )_{ij} D_{ij} \\right)^\omega }{ \max_{ij} \\left( \exp( S^{-1/2} W S^{-1/2} )_{ij} D_{ij} \\right)^\omega }
    $$

    See Also:

    Notes:
        This class is a callable.
    """

    def __init__(
        self,
        distance_matrix: Float[torch.Tensor, "num_nodes num_nodes"],
        omega: float = 1.0,
    ):
        """
        Args:
            distance_matrix:
                The distance matrix of the network.
            omega:
                The power to which to raise each element of the distance weighted communicability before performing
                the sum. Defaults to 1.0."""
        self.distance_matrix = distance_matrix
        self.omega = omega

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "..."]:
        """
        Computes the distance-weighted communicability of a network with a given weight matrix.

        Args:
            weight_matrix:
                The weight matrix of the network.

        Returns:
            distance_weighted_communicability:
                The distance-weighted communicability matrix of the network
        """
        communicability_matrix = compute_communicability(weight_matrix)
        # Expand distance matrix to match batch dimensions
        expanded_distance = self.distance_matrix.expand_as(communicability_matrix)
        distance_weighted_communicability = torch.pow(
            communicability_matrix * expanded_distance, self.omega
        )
        max_distance_weighted_communicability = torch.amax(
            distance_weighted_communicability, dim=(-2, -1), keepdim=True
        )
        normalised_distance_weighted_communicability = (
            distance_weighted_communicability / max_distance_weighted_communicability
        )
        return torch.sum(normalised_distance_weighted_communicability, dim=(-2, -1))


class WeightedDistance(OptimisationCriterion):
    """
    Weighted Distance optimisation criterion.
    To compute the optimisation criterion, we go through the following steps:

    1. Take the element-wise product of the distance matrix and the weight matrix, $D \odot W$.
    2. Raise each element of this product to the power of $\omega$, $(D \odot W)_{ij}^\omega$.
    3. Sum over the elements of the weighted distance matrix rasied to the power of $\omega$ to get the loss.

    The loss is then given by:
    $$
    L(W) = \sum_{ij} \\left( D_{ij} W_{ij} \\right)^\omega
    $$
    """

    def __init__(
        self,
        distance_matrix: Float[torch.Tensor, "num_nodes num_nodes"],
        omega: float = 1.0,
    ):
        self.distance_matrix = distance_matrix
        self.omega = omega

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "..."]:
        expanded_distance = self.distance_matrix.expand_as(weight_matrix)
        weighted_distance = torch.pow(expanded_distance * weight_matrix, self.omega)
        return torch.sum(weighted_distance, dim=(-2, -1))


class NormalisedWeightedDistance(OptimisationCriterion):
    """
    Normalised Weighted Distance optimisation criterion.
    To compute the optimisation criterion, we go through the following steps:

    1. Take the element-wise product of the distance matrix and the weight matrix, $D \odot W$.
    2. Raise each element of this product to the power of $\omega$, $(D \odot W)_{ij}^\omega$.
    3. Normalise by dividing by the maximum element.
    4. Sum over the elements of the weighted distance matrix rasied to the power of $\omega$ to get the loss.

    The loss is then given by:
    $$
    L(W) = \\frac{ \sum_{ij} \\left( D_{ij} W_{ij} \\right)^\omega }{ \max_{ij} \\left( D_{ij} W_{ij} \\right)^\omega }
    $$
    """

    def __init__(
        self,
        distance_matrix: Float[torch.Tensor, "num_nodes num_nodes"],
        omega: float = 1.0,
    ):
        self.distance_matrix = distance_matrix
        self.omega = omega

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "..."]:
        expanded_distance = self.distance_matrix.expand_as(weight_matrix)
        weighted_distance = torch.pow(expanded_distance * weight_matrix, self.omega)
        max_weighted_distance = torch.max(weighted_distance, dim=(-2, -1), keepdim=True)
        normalised_weighted_distance = weighted_distance / max_weighted_distance
        return torch.sum(normalised_weighted_distance, dim=(-2, -1))


class Weight(OptimisationCriterion):
    """
    Weight optimisation criterion.
    To compute the optimisation criterion, we sum over the elements of
    the weight matrix raised to the power of $\omega$.

    The loss is then given by:
    $$
    L(W) = \sum_{ij} W_{ij}^\omega
    $$
    """

    def __init__(self, omega: float = 1.0):
        self.omega = omega

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "..."]:
        return torch.sum(torch.pow(weight_matrix, self.omega), dim=(-2, -1))


class NormalisedWeight(OptimisationCriterion):
    """
    Normalised Weight optimisation criterion.
    To compute the optimisation criterion, we normalise the weight matrix
    by dividing by the maximum element, and then sum over the elements of
    the normalised weight matrix raised to the power of $\omega$.

    The loss is then given by:
    $$
    L(W) = \\frac{ \sum_{ij} W^\omega_{ij} }{ \max_{ij} W^\omega_{ij} }
    $$
    """

    def __init__(self, omega: float = 1.0):
        self.omega = omega

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "... num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "..."]:
        max_weight = torch.max(weight_matrix, dim=(-2, -1), keepdim=True)
        normalised_weight = weight_matrix / max_weight
        return torch.sum(normalised_weight, dim=(-2, -1))
