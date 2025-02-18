import torch
from jaxtyping import Float, jaxtyped
from typing import Union
from typeguard import typechecked
from abc import ABC, abstractmethod

from gnm.utils import communicability, weighted_checks


class OptimisationCriterion(ABC):
    """Base abstract class for optimisation criteria used in weighted generative networks.

    This class provides a framework for defining various optimisation objectives, $L(W)$ used
    to evolve weights in the weighted generative network model.
    """

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "num_simulations num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_simulations"]:
        """Compute the final criterion $L(W)$ for optimisation of the network weights."""
        pass


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

    def __str__(self) -> str:
        return "Communicability"

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "num_simulations num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_simulations"]:
        communicability_matrix = communicability(weight_matrix)
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

    def __str__(self) -> str:
        return "Normalised communicability"

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "num_simulations num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_simulations"]:
        communicability_matrix = communicability(weight_matrix)
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
        distance_matrix: Union[
            Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
            Float[torch.Tensor, "num_nodes num_nodes"],
        ],
        omega: float = 1.0,
    ):
        """
        Args:
            distance_matrix:
                The distance matrix of the network.
            omega:
                The power to which to raise each element of the distance weighted communicability before performing
                the sum. Defaults to 1.0."""
        if len(distance_matrix.shape) == 2:
            self.distance_matrix = distance_matrix.unsqueeze(0)
        else:
            self.distance_matrix = distance_matrix

        weighted_checks(self.distance_matrix)
        self.omega = omega

    def __str__(self) -> str:
        return "Distance weighted communicability"

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "num_simulations num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_simulations"]:
        """
        Computes the distance-weighted communicability of a network with a given weight matrix.

        Args:
            weight_matrix:
                The weight matrix of the network.

        Returns:
            distance_weighted_communicability:
                The distance-weighted communicability matrix of the network
        """
        communicability_matrix = communicability(weight_matrix)
        distance_weighted_communicability = torch.pow(
            communicability_matrix * self.distance_matrix, self.omega
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
        distance_matrix: Union[
            Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
            Float[torch.Tensor, "num_nodes num_nodes"],
        ],
        omega: float = 1.0,
    ):
        """
        Args:
            distance_matrix:
                The distance matrix of the network.
            omega:
                The power to which to raise each element of the distance weighted communicability before performing
                the sum. Defaults to 1.0."""
        if len(distance_matrix.shape) == 2:
            self.distance_matrix = distance_matrix.unsqueeze(0)
        else:
            self.distance_matrix = distance_matrix

        weighted_checks(self.distance_matrix)
        self.omega = omega

    def __str__(self) -> str:
        return "Normalised distance weighted communicability"

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "num_simulations num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_simulations"]:
        """
        Computes the distance-weighted communicability of a network with a given weight matrix.

        Args:
            weight_matrix:
                The weight matrix of the network.

        Returns:
            distance_weighted_communicability:
                The distance-weighted communicability matrix of the network
        """
        communicability_matrix = communicability(weight_matrix)
        # Expand distance matrix to match batch dimensions
        distance_weighted_communicability = torch.pow(
            communicability_matrix * self.distance_matrix, self.omega
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
        distance_matrix: Union[
            Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
            Float[torch.Tensor, "num_nodes num_nodes"],
        ],
        omega: float = 1.0,
    ):
        if len(distance_matrix.shape) == 2:
            self.distance_matrix = distance_matrix.unsqueeze(0)
        else:
            self.distance_matrix = distance_matrix

        weighted_checks(self.distance_matrix)
        self.omega = omega

    def __str__(self) -> str:
        return "Weighted distance"

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "num_simulations num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_simulations"]:
        weighted_distance = torch.pow(self.distance_matrix * weight_matrix, self.omega)
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
        distance_matrix: Union[
            Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
            Float[torch.Tensor, "num_nodes num_nodes"],
        ],
        omega: float = 1.0,
    ):
        if len(distance_matrix.shape) == 2:
            self.distance_matrix = distance_matrix.unsqueeze(0)
        else:
            self.distance_matrix = distance_matrix

        weighted_checks(self.distance_matrix)
        self.omega = omega

    def __str__(self) -> str:
        return "Normalised weighted distance"

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "num_simulations num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_simulations"]:
        weighted_distance = torch.pow(self.distance_matrix * weight_matrix, self.omega)
        max_weighted_distance = torch.amax(
            weighted_distance, dim=(-2, -1), keepdim=True
        )
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

    def __str__(self) -> str:
        return "Weight"

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "num_simulations num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_simulations"]:
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

    def __str__(self) -> str:
        return "Normalised weight"

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "num_simulations num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_simulations"]:
        max_weight = torch.amax(weight_matrix, dim=(-2, -1), keepdim=True)
        normalised_weight = weight_matrix / max_weight
        return torch.sum(normalised_weight, dim=(-2, -1))
