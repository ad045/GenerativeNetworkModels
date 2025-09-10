import torch
from jaxtyping import Float, jaxtyped
from typing import Union, List
from typeguard import typechecked
from abc import ABC, abstractmethod

from gnm.utils import communicability, weighted_checks


class OptimisationCriterion(ABC):
    r"""Abstract base class for optimisation criteria used in weighted generative networks.

    This class provides a framework for defining various optimisation objectives that guide
    the evolution of weights in weighted generative network models. Each criterion implements
    an objective function L(W) that the network attempts to optimise during generation.

    Subclasses must implement a `__call__` method to compute the objective function value for a given weight matrix

    Examples:
        >>> # Define a custom optimisation criterion
        >>> class CustomCriterion(OptimisationCriterion):
        ...     def __call__(self, weight_matrix):
        ...         # Calculate some objective function
        ...         return torch.sum(weight_matrix)

    See Also:
        - [`model.WeightedGenerativeParameters`][gnm.model.WeightedGenerativeParameters]: Uses optimisation criteria to guide weight evolution
        - [`generative_rules.GenerativeRule`][gnm.generative_rules.GenerativeRule]: Base class for rules for the generation of binary networks
    """

    def __str__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "num_simulations num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_simulations"]:
        """Compute the final criterion $L(W)$ for optimisation of the network weights."""
        pass

    @jaxtyped(typechecker=typechecked)
    def __rmul__(self, coefficient: float) -> "ScaledCriterion":
        return ScaledCriterion(self, coefficient)

    @jaxtyped(typechecker=typechecked)
    def __mul__(self, coefficient: float) -> "ScaledCriterion":
        return ScaledCriterion(self, coefficient)

    @jaxtyped(typechecker=typechecked)
    def __add__(
        self, criterion: "OptimisationCriterion"
    ) -> "LinearCombinationCriterion":
        return LinearCombinationCriterion(
            weight_criteria=[self, criterion], coefficients=[1.0, 1.0]
        )

    @jaxtyped(typechecker=typechecked)
    def __sub__(
        self, criterion: "OptimisationCriterion"
    ) -> "LinearCombinationCriterion":
        return LinearCombinationCriterion(
            weight_criteria=[self, criterion], coefficients=[1.0, -1.0]
        )

    def __eq__(self, criterion: "OptimisationCriterion") -> bool:
        return str(self) == str(criterion)


class ScaledCriterion(OptimisationCriterion):
    r"""Scaled optimisation criterion.

    This class scales an existing optimisation criterion by a given coefficient.

    Examples:
        >>> import torch
        >>> from gnm.weight_criteria import Communicability, ScaledCriterion
        >>> # Define the scaled criterion directly
        >>> scaled_communicability_direct = ScaledCriterion(weight_criterion=Communicability(), coefficient=2.4)
        >>> # Define the scaled criterion indirectly
        >>> scaled_communicability_indirect = 2.4 * Communicability()
        >>> # Check that the two definitions are equivalent
        >>> scaled_communicability_direct == scaled_communicability_indirect
        True
        >>> # Verify that scaling works as intended
        >>> communiability = Communicability()
        >>> from gnm.defaults import get_weighted_network
        >>> weight_matrix = get_weighted_network()
        >>> ( scaled_communicability_direct(weight_matrix) == 2.4 * communicability(weight_matrix) ).item()
        True

    See Also:
        - [`weight_criteria.OptimisationCriterion`][gnm.weight_criteria.OptimisationCriterion]: Base class for defining custom optimisation criteria, from which this class inherits.
        - [`weight_criteria.LinearCombinationCriterion`][gnm.weight_criteria.LinearCombinationCriterion]: Linear combination of multiple optimisation criteria.
    """

    @jaxtyped(typechecker=typechecked)
    def __init__(self, weight_criterion: OptimisationCriterion, coefficient: float):
        r"""
        Args:
            weight_criterion:
                The optimisation criterion to scale.
            coefficient:
                The coefficient by which to scale the criterion.
        """

        self.criterion = weight_criterion
        self.coefficient = coefficient

    def __str__(self) -> str:
        return (
            f"{str(self.criterion)} (coefficient={self.coefficient})"
            if self.coefficient != 1.0
            else str(self.criterion)
        )

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "num_simulations num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_simulations"]:
        return self.coefficient * self.criterion(weight_matrix)


class LinearCombinationCriterion(OptimisationCriterion):
    r"""Linear combination optimisation criterion.

    This class allows for the combination of multiple optimisation criteria into a single criterion via a weighted sum.

    Examples:
        >>> from gnm.weight_criteria import Communicability, Weight, LinearCombinationCriterion
        >>> # Define a linear combination criterion directly
        >>> lc_direct = LinearCombinationCriterion(weight_criteria=[Communicability(), Weight()], coefficients=[1.0,-0.5])
        >>> # Define a linear combination cirterion indirectly
        >>> lc_indirect = Communicability() - 0.5 * Weight()
        >>> # Check that these are equivalent
        >>> lc_direct == lc_indirect
        True
        >>> str(lc_direct)
        'LinearCombinationCriterion(Communicability, Weight (coefficient=-0.5))'

    See Also:
        - [`weight_criteria.OptimisationCriterion`][gnm.weight_criteria.OptimisationCriterion]: Base class for defining custom optimisation criteria, from which this class inherits.
        - [`weight_criteria.ScaledCriterion`][gnm.weight_criteria.ScaledCriterion]: Scaled version of a single optimisation criterion.
    """

    @jaxtyped(typechecker=typechecked)
    def __init__(
        self, weight_criteria: List[OptimisationCriterion], coefficients: List[float]
    ):
        r"""
        Args:
            weight_criteria:
                List of optimisation criteria to combine.
            coefficients:
                List of coefficients to apply to each criterion.
        """
        assert len(weight_criteria) == len(
            coefficients
        ), f"List of weight criteria and coefficients must be the same length. Got {len(weight_criteria)} and {len(coefficients)} respectively."

        self.weight_criteria = []
        self.coefficients = []

        for ii in range(len(weight_criteria)):
            if isinstance(weight_criteria[ii], LinearCombinationCriterion):
                self.weight_criteria.extend(weight_criteria[ii].weight_criteria)
                self.coefficients.extend(
                    [
                        coefficients[ii] * coeff
                        for coeff in weight_criteria[ii].coefficients
                    ]
                )
            else:
                self.weight_criteria.append(weight_criteria[ii])
                self.coefficients.append(coefficients[ii])

        for ii in range(len(self.weight_criteria)):
            if isinstance(self.weight_criteria[ii], ScaledCriterion):
                self.coefficients[ii] = (
                    self.coefficients[ii] * self.weight_criteria[ii].coefficient
                )
                self.weight_criteria[ii] = self.weight_criteria[ii].criterion

    def __str__(self) -> str:
        criteria_str = ", ".join(
            (
                f"{str(criterion)} (coefficient={coefficient})"
                if coefficient != 1.0
                else str(criterion)
            )
            for criterion, coefficient in zip(self.weight_criteria, self.coefficients)
        )
        return f"LinearCombinationCriterion({criteria_str})"

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "num_simulations num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_simulations"]:
        return torch.stack(
            [
                coefficient * criterion(weight_matrix)
                for criterion, coefficient in zip(
                    self.weight_criteria, self.coefficients
                )
            ]
        ).sum(dim=0)


class Communicability(OptimisationCriterion):
    r"""Communicability optimisation criterion.

    This criterion optimises network weights based on the total communicability of the network.
    Communicability measures the ease with which information can flow between nodes through
    all possible paths.

    To compute this optimisation criterion, we follow these steps:

    1. Compute the diagonal node strength matrix, $S_{ii} = \sum_j W_{ij}$ (plus a small constant to prevent division by zero)
    2. Compute the normalised weight matrix, $S^{-1/2} W S^{-1/2}$
    3. Compute the communicability matrix by taking the matrix exponential, $\exp(S^{-1/2} W S^{-1/2})$
    4. Raise each element of this product to the power of $\omega$, $\exp(S^{-1/2} W S^{-1/2})_{ij}^\omega$
    5. Sum over the elements of the communicability matrix raised to the power of $\omega$ to get the loss

    The loss $L(W)$ is given by:
    $$
    L(W) = \sum_{ij} \\left( \exp( S^{-1/2} W S^{-1/2} )_{ij} \\right)^\omega
    $$

    Examples:
        >>> from gnm.weight_criteria import Communicability
        >>> from gnm.defaults import get_weighted_network
        >>> # Create a communicability criterion with default parameters
        >>> criterion = Communicability(omega=1.0)
        >>> # Apply to a network
        >>> weight_matrix = get_weighted_network()
        >>> loss = criterion(weight_matrix)
        >>> loss.shape
        torch.Size([1])

    See Also:
        - [`utils.communicability`][gnm.utils.communicability]: Function used to calculate network communicability.
        - [`weight_criteria.OptimisationCriterion`][gnm.weight_criteria.OptimisationCriterion]: Base class for defining custom optimisation criteria, from which this class inherits.
        - [`weight_criteria.NormalisedCommunicability`][gnm.weight_criteria.NormalisedCommunicability]: Normalised version of this criterion.
        - [`weight_criteria.DistanceWeightedCommunicability`][gnm.weight_criteria.DistanceWeightedCommunicability]: Distance-weighted version of this criterion.
    """

    def __init__(self, omega: float = 1.0):
        r"""
        Args:
            omega:
                The power to which to raise each element of the communicability matrix before performing the sum. Defaults to 1.0.
        """
        self.omega = omega

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "num_simulations num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_simulations"]:
        communicability_matrix = communicability(weight_matrix)
        tilted_communicability = torch.pow(communicability_matrix, self.omega)
        return torch.sum(tilted_communicability, dim=(-2, -1))


class NormalisedCommunicability(OptimisationCriterion):
    r"""Normalised communicability optimisation criterion.

    This criterion optimises network weights based on the normalised total communicability
    of the network. The normalisation makes this criterion less sensitive to absolute weight
    magnitudes by dividing by the maximum communicability value.

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

    Examples:
        >>> from gnm.weight_criteria import NormalisedCommunicability
        >>> from gnm.defaults import get_weighted_network
        >>> # Create a normalied communicability criterion with default parameters
        >>> criterion = NormalisedCommunicability()
        >>> # Apply to a network
        >>> weight_matrix = get_weighted_network()
        >>> loss = criterion(weight_matrix)
        >>> loss.shape
        torch.Size([1])

    See Also:
        - [`utils.communicability`][gnm.utils.communicability]: Function used to calculate network communicability.
        - [`weight_criteria.OptimisationCriterion`][gnm.weight_criteria.OptimisationCriterion]: Base class for defining custom optimisation criteria, from which this class inherits.
        - [`weight_criteria.Communicability`][gnm.weight_criteria.Communicability]: Non-normalised version of this criterion.
        - [`weight_criteria.NormalisedDistanceWeightedCommunicability`][gnm.weight_criteria.NormalisedDistanceWeightedCommunicability]: Distance-weighted version of this criterion.

    """

    def __init__(self, omega: float = 1.0):
        r"""
        Args:
            omega:
                The power to which to raise each element of the communicability matrix before performing the sum and normalising. Defaults to 1.0.
        """
        self.omega = omega

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
    r"""Distance-weighted communicability optimisation criterion.

    This criterion optimises network weights based on the communicability weighted by
    the distances between nodes. This adds a spatial constraint to the optimisation,
    where communicability between distant nodes contributes more to the objective function.

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

    Examples:
        >>> from gnm.weight_criteria import DistanceWeightedCommunicability
        >>> from gnm.defaults import get_weighted_network, get_distance_matrix
        >>> # Create a distance-weighted communicability criterion with default parameters
        >>> distance_matrix = get_distance_matrix()
        >>> criterion = DistanceWeightedCommunicability(distance_matrix, omega=1.0)
        >>> # Apply to a network
        >>> weight_matrix = get_weighted_network()
        >>> loss = criterion(weight_matrix)
        >>> loss.shape
        torch.Size([1])

    See Also:
        - [`utils.communicability`][gnm.utils.communicability]: Function used to calculate network communicability.
        - [`weight_criteria.OptimisationCriterion`][gnm.weight_criteria.OptimisationCriterion]: Base class for defining custom optimisation criteria, from which this class inherits.
        - [`weight_criteria.NormalisedDistanceWeightedCommunicability`][gnm.weight_criteria.NormalisedDistanceWeightedCommunicability]: Normalised version of this criterion.
        - [`weight_criteria.Communicability`][gnm.weight_criteria.Communicability]: Non-distance-weighted version of this criterion.
    """

    def __init__(
        self,
        distance_matrix: Union[
            Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
            Float[torch.Tensor, "num_nodes num_nodes"],
        ],
        omega: float = 1.0,
    ):
        r"""
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
    r"""Normalised distance-weighted communicability optimisation criterion.

    This criterion optimises network weights based on the normalised communicability
    weighted by the distances between nodes. The normalisation makes this criterion
    less sensitive to absolute magnitudes by dividing by the maximum value.

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

    Examples:
        >>> from gnm.weight_criteria import NormalisedDistanceWeightedCommunicability
        >>> from gnm.defaults import get_weighted_network, get_distance_matrix
        >>> # Create a normalised distance-weighted communicability criterion with default parameters
        >>> distance_matrix = get_distance_matrix()
        >>> criterion = NormalisedDistanceWeightedCommunicability(distance_matrix, omega=1.0)
        >>> # Apply to a network
        >>> weight_matrix = get_weighted_network()
        >>> loss = criterion(weight_matrix)
        >>> loss.shape
        torch.Size([1])

    See Also:
        - [`utils.communicability`][gnm.utils.communicability]: Function used to calculate network communicability.
        - [`weight_criteria.OptimisationCriterion`][gnm.weight_criteria.OptimisationCriterion]: Base class for defining custom optimisation criteria, from which this class inherits.
        - [`weight_criteria.DistanceWeightedCommunicability`][gnm.weight_criteria.DistanceWeightedCommunicability]: Non-normalised version of this criterion.
        - [`weight_criteria.NormalisedCommunicability`][gnm.weight_criteria.NormalisedCommunicability]: Non-distance-weighted version of this criterion.
    """

    def __init__(
        self,
        distance_matrix: Union[
            Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
            Float[torch.Tensor, "num_nodes num_nodes"],
        ],
        omega: float = 1.0,
    ):
        r"""
        Args:
            distance_matrix:
                The distance matrix of the network.
            omega:
                The power to which to raise each element of the distance weighted communicability before performing
                the sum and normalising. Defaults to 1.0."""
        if len(distance_matrix.shape) == 2:
            self.distance_matrix = distance_matrix.unsqueeze(0)
        else:
            self.distance_matrix = distance_matrix

        weighted_checks(self.distance_matrix)
        self.omega = omega

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "num_simulations num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_simulations"]:
        r"""
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
    r"""Weighted distance optimisation criterion.

    This criterion optimises network weights based on the product of weights and distances.
    It penalises strong connections between distant nodes, encouraging a more
    spatially efficient network structure.

    To compute the optimisation criterion, we go through the following steps:

    1. Take the element-wise product of the distance matrix and the weight matrix, $D \odot W$.
    2. Raise each element of this product to the power of $\omega$, $(D \odot W)_{ij}^\omega$.
    3. Sum over the elements of the weighted distance matrix rasied to the power of $\omega$ to get the loss.

    The loss is then given by:
    $$
    L(W) = \sum_{ij} \\left( D_{ij} W_{ij} \\right)^\omega
    $$

    Examples:
        >>> from gnm.weight_criteria import WeightedDistance
        >>> from gnm.defaults import get_weighted_network, get_distance_matrix
        >>> # Create a weighted distance criterion with default parameters
        >>> distance_matrix = get_distance_matrix()
        >>> criterion = WeightedDistance(distance_matrix, omega=1.0)
        >>> # Apply to a network
        >>> weight_matrix = get_weighted_network()
        >>> loss = criterion(weight_matrix)
        >>> loss.shape
        torch.Size([1])

    See Also:
        - [`weight_criteria.OptimisationCriterion`][gnm.weight_criteria.OptimisationCriterion]: Base class for defining custom optimisation criteria, from which this class inherits.
        - [`weight_criteria.NormalisedWeightedDistance`][gnm.weight_criteria.NormalisedWeightedDistance]: Normalised version of this criterion.
        - [`weight_criteria.Weight`][gnm.weight_criteria.Weight]: Non-distance-weighted version of this criterion.
    """

    def __init__(
        self,
        distance_matrix: Union[
            Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
            Float[torch.Tensor, "num_nodes num_nodes"],
        ],
        omega: float = 1.0,
    ):
        r"""
        Args:
            distance_matrix:
                The distance matrix of the network.
            omega:
                The power to which to raise each element of the weighted distance before performing
                the sum. Defaults to 1.0.
        """

        if len(distance_matrix.shape) == 2:
            self.distance_matrix = distance_matrix.unsqueeze(0)
        else:
            self.distance_matrix = distance_matrix

        weighted_checks(self.distance_matrix)
        self.omega = omega

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "num_simulations num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_simulations"]:
        weighted_distance = torch.pow(self.distance_matrix * weight_matrix, self.omega)
        return torch.sum(weighted_distance, dim=(-2, -1))


class NormalisedWeightedDistance(OptimisationCriterion):
    r"""Normalised weighted distance optimisation criterion.

    This criterion optimises network weights based on the normalised product of weights and distances.
    The normalisation makes this criterion less sensitive to absolute magnitudes by dividing
    by the maximum value.

    To compute the optimisation criterion, we go through the following steps:

    1. Take the element-wise product of the distance matrix and the weight matrix, $D \odot W$.
    2. Raise each element of this product to the power of $\omega$, $(D \odot W)_{ij}^\omega$.
    3. Normalise by dividing by the maximum element.
    4. Sum over the elements of the weighted distance matrix rasied to the power of $\omega$ to get the loss.

    The loss is then given by:
    $$
    L(W) = \\frac{ \sum_{ij} \\left( D_{ij} W_{ij} \\right)^\omega }{ \max_{ij} \\left( D_{ij} W_{ij} \\right)^\omega }
    $$

    Examples:
        >>> from gnm.weight_criteria import NormalisedWeightedDistance
        >>> from gnm.defaults import get_weighted_network, get_distance_matrix
        >>> # Create a normalised weighted distance criterion with default parameters
        >>> distance_matrix = get_distance_matrix()
        >>> criterion = NormalisedWeightedDistance(distance_matrix, omega=1.0)
        >>> # Apply to a network
        >>> weight_matrix = get_weighted_network()
        >>> loss = criterion(weight_matrix)
        >>> loss.shape
        torch.Size([1])

    See Also:
        - [`weight_criteria.OptimisationCriterion`][gnm.weight_criteria.OptimisationCriterion]: Base class for defining custom optimisation criteria, from which this class inherits.
        - [`weight_criteria.WeightedDistance`][gnm.weight_criteria.WeightedDistance]: Non-normalised version of this criterion.
        - [`weight_criteria.NormalisedWeight`][gnm.weight_criteria.NormalisedWeight]: Non-distance-weighted version of this criterion.
    """

    def __init__(
        self,
        distance_matrix: Union[
            Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
            Float[torch.Tensor, "num_nodes num_nodes"],
        ],
        omega: float = 1.0,
    ):
        r"""
        Args:
            distance_matrix:
                The distance matrix of the network.
            omega:
                The power to which to raise each element of the weighted distance before performing
                the sum and normalising. Defaults to 1.0.
        """
        if len(distance_matrix.shape) == 2:
            self.distance_matrix = distance_matrix.unsqueeze(0)
        else:
            self.distance_matrix = distance_matrix

        weighted_checks(self.distance_matrix)
        self.omega = omega

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
    r"""Weight optimisation criterion.

    This criterion simply optimises based on the sum of all weights raised to a power.
    It provides a basic measure of the overall magnitude of weights in the network.

    To compute the optimisation criterion, we sum over the elements of
    the weight matrix raised to the power of $\omega$.

    The loss is then given by:
    $$
    L(W) = \sum_{ij} W_{ij}^\omega
    $$

    Examples:
        >>> from gnm.weight_criteria import Weight
        >>> from gnm.defaults import get_weighted_network
        >>> # Create a weight criterion with default parameters
        >>> criterion = Weight(omega=1.0)
        >>> # Apply to a network
        >>> weight_matrix = get_weighted_network()
        >>> loss = criterion(weight_matrix)
        >>> loss.shape
        torch.Size([1])

    See Also:
        - [`weight_criteria.OptimisationCriterion`][gnm.weight_criteria.OptimisationCriterion]: Base class for defining custom optimisation criteria, from which this class inherits.
        - [`weight_criteria.NormalisedWeight`][gnm.weight_criteria.NormalisedWeight]: Normalised version of this criterion.
        - [`weight_criteria.WeightedDistance`][gnm.weight_criteria.WeightedDistance]: Distance-weighted version of this criterion.
    """

    def __init__(self, omega: float = 1.0):
        r"""
        Args:
            omega:
                The power to which to raise each element of the weight matrix before performing the sum. Defaults to 1.0.
        """
        self.omega = omega

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "num_simulations num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_simulations"]:
        return torch.sum(torch.pow(weight_matrix, self.omega), dim=(-2, -1))


class NormalisedWeight(OptimisationCriterion):
    r"""Normalised weight optimisation criterion.

    This criterion optimises based on the sum of all normalised weights.
    The normalisation makes this criterion invariant to the absolute scale
    of weights by dividing by the maximum weight.


    To compute the optimisation criterion, we normalise the weight matrix
    by dividing by the maximum element, and then sum over the elements of
    the normalised weight matrix raised to the power of $\omega$.

    The loss is then given by:
    $$
    L(W) = \\frac{ \sum_{ij} W^\omega_{ij} }{ \max_{ij} W^\omega_{ij} }
    $$

    Examples:
        >>> from gnm.weight_criteria import NormalisedWeight
        >>> from gnm.defaults import get_weighted_network
        >>> # Create a normalised weight criterion with default parameters
        >>> criterion = NormalisedWeight(omega=1.0)
        >>> # Apply to a network
        >>> weight_matrix = get_weighted_network()
        >>> loss = criterion(weight_matrix)
        >>> loss.shape
        torch.Size([1])

    See Also:
        - [`weight_criteria.OptimisationCriterion`][gnm.weight_criteria.OptimisationCriterion]: Base class for defining custom optimisation criteria, from which this class inherits.
        - [`weight_criteria.Weight`][gnm.weight_criteria.Weight]: Non-normalised version of this criterion.
        - [`weight_criteria.NormalisedWeightedDistance`][gnm.weight_criteria.NormalisedWeightedDistance]: Distance-weighted version of this criterion.
    """

    def __init__(self, omega: float = 1.0):
        r"""
        Args:
            omega:
                The power to which to raise each element of the weight matrix before performing the sum and normalising. Defaults to 1.0.
        """
        self.omega = omega

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self, weight_matrix: Float[torch.Tensor, "num_simulations num_nodes num_nodes"]
    ) -> Float[torch.Tensor, "num_simulations"]:
        max_weight = torch.amax(weight_matrix, dim=(-2, -1), keepdim=True)
        normalised_weight = weight_matrix / max_weight
        return torch.sum(normalised_weight, dim=(-2, -1))