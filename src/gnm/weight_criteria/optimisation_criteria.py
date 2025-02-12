import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from abc import ABC, abstractmethod


class OptimisationCriterion(ABC):
    """Base abstract class for optimisation criteria used in weighted generative networks.

    This class provides a framework for defining various optimisation objectives, $L(W)$ used
    to evolve weights in the weighted generative network model.

    Attributes:
        normalisation (bool):
            Whether to normalise the criterion values. When True, returns raw summed values.
            When False, normalises by the maximum value before summing.

    Note:
        Subclasses must implement the `_unnormalised_call` method to define their specific
        optimisation criterion.
    """

    def __init__(self, normalisation: bool = False):
        self.normalisation = normalisation

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def _unnormalised_call(
        self, weight_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        """Computes the unnormalised criterion matrix for a given weight matrix.

        This abstract method must be implemented by all optimisation criteria. It defines
        how the criterion matrix value $C_{ij}$ is computed from the network's weight matrix, $W_{ij}$.

        Args:
            weight_matrix:
                The current weight matrix $W_{ij}$ of the network. Must be a square matrix
                of shape (num_nodes, num_nodes).

        Returns:
            The unnormalized criterion values $C_{ij}$ as a matrix of the same shape
            as the input weight matrix.
        """
        pass

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, ""]:
        """Compute the final criterion $L(W)$ for optimisation of the network weights.

        Processes the weight matrix through these steps:

        1. Computes unnormalized criterion values through `_unnormalised_call`, $C_{ij} \gets f(W_{ij})$
        3. If normalise=True, normalise the criterion values by their maximum value:
           $$
           C_{ij} \gets \\frac{C_{ij}}{\max_{kl} C_{kl}}
           $$
        4. Returns the sum $L = \sum_{ij} C_{ij}$

        Args:
            weight_matrix: The current weight matrix $W_{ij}$ of the network.

        Returns:
            A scalar tensor containing the criterion value for optimisation, $L(W)$

        Examples:
            >>> criterion = MyCriterion(normalisation=False)
            >>> weight_matrix = torch.rand(10, 10)
            >>> loss = criterion(weight_matrix)
        """

        symmetrised_matrix = 0.5 * (weight_matrix + weight_matrix.T)

        if self.normalisation == True:
            return torch.sum(self._unnormalised_call(symmetrised_matrix))
        else:
            unnormalised_value = self._unnormalised_call(symmetrised_matrix)
            normalisation_term = unnormalised_value.max()
            return torch.sum(unnormalised_value / normalisation_term)


class Communicability(OptimisationCriterion):
    """Communicability optimisation criterion.
    To compute the criterion matrix, we go through the following steps:

    1. Compute the diagonal node strength matrix, $S_{ii} = \sum_j W_{ij}$ (plus a small constant to prevent division by zero).
    2. Compute the normalised weight matrix, $S^{-1/2} W S^{-1/2}$.
    3. Compute the communicability matrix by taking the matrix exponential, $\exp( S^{-1/2} W S^{-1/2} )$.
    5. Raise each element of this product to the power of $\omega$, $\exp( S^{-1/2} W S^{-1/2} )_{ij}^\omega$.

    The loss is then given by summing over the elements of the communicability matrix,
    raised to the power of $\omega$:
    $$
    L(W) = \sum_{ij} \\left( \exp( S^{-1/2} W S^{-1/2} )_{ij} \\right)^\omega
    $$
    if normalisation is True, we first divide by the maximum summand before performing the sum.

    See Also:

    Note:
        This class is a callable.

    Examples:
        >>> criterion = Communicability(normalisation=False, omega=1.0)
        >>> weight_matrix = torch.rand(10, 10)
        >>> loss = criterion(weight_matrix)
    """

    def __init__(self, normalisation: bool = False, omega: float = 1.0):
        """
        Args:
            normalisation:
                Determines whether to divide by the maximum summand before summing to obtain the loss.
            omega (float, optional): _description_. Defaults to 1.0.
        """
        super().__init__(normalisation=normalisation)
        self.omega = omega

    def _unnormalised_call(
        self, weight_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ):
        # Compute the node strengths, with a small constant addition to prevent division by zero.
        node_strengths = (
            0.5 * (weight_matrix.sum(dim=0) + weight_matrix.sum(dim=1)) + 1e-6
        )
        # Compute the inverse square root of the node strengths
        inv_sqrt_node_strenghts = torch.diag(1 / torch.sqrt(node_strengths))
        # Compute the normalised weight matrix
        normalised_weight_matrix = (
            inv_sqrt_node_strenghts @ weight_matrix @ inv_sqrt_node_strenghts
        )

        # Compute the communicability matrix
        communicability_matrix = torch.matrix_exp(normalised_weight_matrix)
        # Compute the distance-weighted communicability to the power of omega
        tilted_communicability = torch.pow(communicability_matrix, self.omega)

        return tilted_communicability


class DistanceWeightedCommunicability(OptimisationCriterion):
    """Distance Weighted Communicability optimisation criterion.
    To compute the criterion matrix, we go through the following steps:

    1. Compute the diagonal node strength matrix, $S_{ii} = \sum_j W_{ij}$ (plus a small constant to prevent division by zero).
    2. Compute the normalised weight matrix, $S^{-1/2} W S^{-1/2}$.
    3. Compute the communicability matrix by taking the matrix exponential, $\exp( S^{-1/2} W S^{-1/2} )$.
    4. Take the element-wise product of the communicability matrix and the distance matrix, $\exp( S^{-1/2} W S^{-1/2} ) \odot D$
    5. Raise each element of this product to the power of $\omega$, $(\exp( S^{-1/2} W S^{-1/2} ) \odot D)_{ij}^\omega$.

    The loss is then given by summing over the resulting matrix:
    $$
    L(W) = \sum_{ij} \\left( \exp( S^{-1/2} W S^{-1/2} )_{ij} D_{ij} \\right)^\omega
    $$
    if normalisation is True, we first divide by the maximum summand before performing the sum.

    See Also:

    Notes:
        This class is a callable.
    """

    def __init__(
        self,
        distance_matrix: Float[torch.Tensor, "num_nodes num_nodes"],
        normalisation: bool = False,
        omega: float = 1.0,
    ):
        """
        Args:
            distance_matrix:
                The distance matrix of the network.
            normalisation:
                Determines whether to divide by the maximum summand before summing to obtain the loss.
            omega:
                The power to which to raise each element of the distance weighted communicability before performing
                the sum. Defaults to 1.0."""
        super().__init__(normalisation=normalisation)
        self.distance_matrix = distance_matrix
        self.omega = omega

    def _unnormalised_call(
        self, weight_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        """
        Computes the distance-weighted communicability of a network with a given weight matrix.

        Args:
            weight_matrix:
                The weight matrix of the network.

        Returns:
            distance_weighted_communicability:
                The distance-weighted communicability matrix of the network
        """
        # Compute the node strengths, with a small constant addition to prevent division by zero.
        node_strengths = (
            0.5 * (weight_matrix.sum(dim=0) + weight_matrix.sum(dim=1)) + 1e-6
        )
        # Compute the inverse square root of the node strengths
        inv_sqrt_node_strenghts = torch.diag(1 / torch.sqrt(node_strengths))
        # Compute the normalised weight matrix
        normalised_weight_matrix = (
            inv_sqrt_node_strenghts @ weight_matrix @ inv_sqrt_node_strenghts
        )

        # Compute the communicability matrix
        communicability_matrix = torch.matrix_exp(normalised_weight_matrix)
        # Compute the distance-weighted communicability to the power of omega
        distance_weighted_communicability = torch.pow(
            communicability_matrix * self.distance_matrix, self.omega
        )

        # Return the cumulative distance-weighted communicability
        return distance_weighted_communicability


class WeightedDistance(OptimisationCriterion):
    def __init__(
        self,
        distance_matrix: Float[torch.Tensor, "num_nodes num_nodes"],
        normalisation: bool = False,
        omega: float = 1.0,
    ):
        super().__init__(normalisation=normalisation)
        self.distance_matrix = distance_matrix
        self.omega = omega

    def _unnormalised_call(
        self, weight_matrix: Float[torch.Tensor, "num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_nodes num_nodes"]:
        weighted_distance = torch.pow(self.distance_matrix * weight_matrix, self.omega)
        return weighted_distance
