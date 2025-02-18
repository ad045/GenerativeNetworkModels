from jaxtyping import Float, jaxtyped
from typing import Optional, Tuple, Union, Any
from typeguard import typechecked

from .weight_criteria import OptimisationCriterion
from .generative_rules import GenerativeRule

import torch
import torch.optim as optim

from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class BinaryGenerativeParameters:
    """Parameters controlling the binary generative network model's evolution.

    This dataclass encapsulates the parameters that determine how a binary generative
    network model grows and forms connections. The parameters control three main aspects
    of network generation:

    1. The influence of physical distance, $\eta$
    2. The influence of topological similarity, $\gamma$
    3. The influence of developmental factors, $\lambda$

    Each influence can be modeled using either a power law or exponential relationship,
    as specified by the relationship type parameters. The total wiring probability is
    proportional to the product of a distance factor $d_{ij}$, a preferential wiring
    factor $k_{ij}$, and a developmental factor $h_{ij}$:
    $$
        P_{ij} \propto d_{ij} \\times k_{ij} \\times h_{ij}
    $$

    Attributes:
        eta (float):
            Parameter ($\eta$) controlling the influence of Euclidean distances $D_{ij}$
            on wiring probability. More negative values indicate lower wiring probabilities
            between nodes that are futher away.

            - For power law: $d_{ij} = D_{ij}^{\eta}$
            - For exponential: $d_{ij} = \exp(\eta D_{ij})$

        gamma (float):
            Parameter ($\gamma$) controlling the influence of the preferential wiring rule $K_{ij}$
            on wiring probability. Larger values indicate stronger preference creating
            connections between nodes that have high preferential value.

            - For power law: $k_{ij} = K_{ij}^{\gamma}$
            - For exponential: $k_{ij} = \exp(\gamma K_{ij})$

        lambdah (float):
            Parameter ($\lambda$) controlling the influence of heterochronicity $H_{ij}$ on wiring
            probability. Larger values indicate stronger temporal dependence in development.

            - For power law: $h_{ij} = H_{ij}^{\lambda}$
            - For exponential: $h_{ij} = \exp(\lambda H_{ij})$

        distance_relationship_type (str):
            The relationship between distance $D_{ij}$ and distance factor $d_{ij}$.
            Must be one of ['powerlaw', 'exponential'].

        preference_relationship_type (str):
            The relationship between the generative rule output $K_{ij}$ and preferential wiring factor $k_{ij}$.
            Must be one of ['powerlaw', 'exponential'].

        prob_offset (float, optional):
            Small constant added to unnormalized probabilities to prevent division by zero.
            Defaults to 1e-6.

        generative_rule (GenerativeRule):
            The generative rule that transforms the adjacency matrix to a matching index matrix.
            This computes the preferential wiring rule $K_{ij}$ from the adjacency matrix $A_{ij}$.

    Examples:
        >>> binary_parameters = BinaryGenerativeParameters(
        ...     eta=1.0,
        ...     gamma=0.5,
        ...     lambdah=2.0,
        ...     distance_relationship_type='powerlaw',
        ...     preferential_relationship_type='exponential',
        ...     heterochronicity_relationship_type='powerlaw',
        ...     generative_rule=MatchingIndex(divisor='mean')
        ... )

    See Also:
        - GenerativeRule: A base class for generative rules that transform an adjacency matrix $A_{ij}$ into a preferential wiring matrix $K_{ij}$
    """

    eta: float
    gamma: float
    lambdah: float
    distance_relationship_type: str
    preferential_relationship_type: str
    heterochronicity_relationship_type: str
    generative_rule: GenerativeRule
    num_iterations: int
    prob_offset: float = 1e-6
    binary_updates_per_iteration: int = 1

    def __post_init__(self):
        # Perform checks on the distance and matching index relationship type.
        if self.distance_relationship_type not in ["powerlaw", "exponential"]:
            raise NotImplementedError(
                f"Distance relationship type '{self.distance_relationship_type}' is not supported for the binary generative network model."
            )
        if self.preferential_relationship_type not in ["powerlaw", "exponential"]:
            raise NotImplementedError(
                f"Matching relationship type '{self.preferential_relationship_type}' is not supported for the binary generative network model."
            )
        if self.heterochronicity_relationship_type not in ["powerlaw", "exponential"]:
            raise NotImplementedError(
                f"Matching relationship type '{self.heterochronicity_relationship_type}' is not supported for the binary generative network model."
            )

    def __dict__(self) -> dict[str, Any]:
        return {
            "eta": self.eta,
            "gamma": self.gamma,
            "lambdah": self.lambdah,
            "distance_relationship_type": self.distance_relationship_type,
            "preferential_relationship_type": self.preferential_relationship_type,
            "heterochronicity_relationship_type": self.heterochronicity_relationship_type,
            "generative_rule": str(self.generative_rule),
            "prob_offset": self.prob_offset,
            "binary_updates_per_iteration": self.binary_updates_per_iteration,
            "num_iterations": self.num_iterations,
        }


@dataclass
class WeightedGenerativeParameters:
    """Parameters controlling the weighted generative network model's evolution.

    This dataclass encapsulates the parameters that determine how weights evolve in a
    weighted generative network model. While the binary parameters control network
    topology, these parameters control the optimisation of connection weights through
    gradient descent. The optimisation process minimises (or maximises) an objective
    function.

    At each step, the weights are updated according to:
    $$
    W_{ij} \gets W_{ij} - \\alpha \\frac{\partial L}{\partial W_{ij}},
    $$
    where $L$ is the optimisation criterion and $\\alpha$ is the learning rate.
    Note that only those weights present in the binary network adjacency matrix $A_{ij}$
    are updated.
    Additionally, symmetry is enforced so that we always have $W_{ij} = W_{ji}$.

    Attributes:
        alpha (float):
            Learning rate ($\\alpha$) for gradient descent optimisation of weights.
            Controls how much weights change in response to gradients:
            larger values mean bigger steps but potential instability,
            smaller values mean more stable but slower optimisation.

        optimisation_criterion (OptimisationCriterion):
            The objective function $L(W)$ to optimise. This determines what
            properties the final weight configuration will exhibit.
            See OptimisationCriterion class for available options like
            distance-weighted communicability or weighted distance.

        weight_lower_bound (float, optional):
            Minimum allowed value for any weight ($W_{\\rm lower}$). All weights
            will be clipped to stay above this value. Must be non-negative.
            Defaults to 0.0.

        weight_upper_bound (float, optional):
            Maximum allowed value for any weight ($W_{\\rm upper}$). All weights
            will be clipped to stay below this value. Must be greater
            than weight_lower_bound. Defaults to infinity.

        maximise_criterion (bool, optional):
            Whether to maximise rather than minimise the optimisation criterion.
            When True, gradients are flipped to ascend rather than descend.
            Defaults to False.

    Examples:
        >>> communicability_optimisation_criterion = Communicability(normalisation=False, omega=1.0)
        >>> weighted_parameters = WeightedGenerativeParameters(
        ...     alpha=0.01,
        ...     optimisation_criterion=communicability_optimisation_criterion,
        ...     weight_lower_bound=0.0,
        ...     weight_upper_bound=1.0,
        ...     maximise_criterion=False
        ... )

    See Also:
        - OptimisationCriterion: Base class for optimisation objectives
        - DistanceWeightedCommunicability: optimisation criterion based on network communication
        - GenerativeNetworkModel.weighted_update: Method that uses these parameters
    """

    alpha: float
    optimisation_criterion: OptimisationCriterion
    weight_lower_bound: float = 0.0
    weight_upper_bound: float = float("inf")
    maximise_criterion: bool = False
    weighted_updates_per_iteration: int = 1

    def __dict__(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "optimisation_criterion": str(self.optimisation_criterion),
            "weight_lower_bound": self.weight_lower_bound,
            "weight_upper_bound": self.weight_upper_bound,
            "maximise_criterion": self.maximise_criterion,
            "weighted_updates_per_iteration": self.weighted_updates_per_iteration,
        }


class GenerativeNetworkModel:
    """A class implementing both binary and weighted Generative Network Models (GNM).

    This class provides a unified framework for growing networks using both binary and weighted
    generative processes. The model works in two phases:

    1. Binary Growth Phase:
       The network's topology is determined by iteratively adding edges to an adjacency matrix
       $A_{ij}$ based on three factors (a) Physical distance between nodes, (b) Topological similarity
       (through the generative rule), (c) Developmental timing (heterochronicity).
       For more details, see (REF BinaryGenerativeParameters and binary_update method).

    2. Weight Optimisation Phase (Optional):
       If weighted parameters are provided, the model also optimizes edge weights $W_{ij}$
       through gradient descent on a loss, $L(W)$.
       For more details, see (REF WeightedGenerativeParameters and weighted_update method).

    Attributes:
        num_simulations (int):
            Number of simulations to run in parallel.
        seed_adjacency_matrix (torch.Tensor):
            Initial binary adjacency matrix (num_nodes, num_nodes).
        adjacency_matrix (torch.Tensor):
            Current state of the network's adjacency matrix.
        distance_matrix (torch.Tensor):
            Matrix of (Euclidean) distances between nodes.
        num_nodes (int):
            Number of nodes in the network.
        binary_parameters (BinaryGenerativeParameters):
            Parameters controlling binary network growth.
        distance_factor (torch.Tensor):
            Precomputed distance influence on edge formation.
        seed_weight_matrix (torch.Tensor, optional):
            Initial weight matrix if using weighted GNM.
        weight_matrix (torch.Tensor, optional):
            Current state of the weight matrix.
        weighted_parameters (WeightedGenerativeParameters, optional):
            Parameters controlling weight optimisation.
        optimiser (torch.optim.Optimizer, optional):
            Optimiser for weight updates.

    Examples:
        >>> binary_parameters = BinaryGenerativeParameters(
        ...     eta=1.0,
        ...     gamma=-0.5,
        ...     lambdah=1.0,
        ...     distance_relationship_type='powerlaw',
        ...     preferential_relationship_type='exponential',
        ...     heterochronicity_relationship_type='powerlaw',
        ...     generative_rule=MatchingIndex(divisor='mean')
        ... )
        >>> seed_adjacency_matrix = torch.zeros((10, 10))
        >>> model = GenerativeNetworkModel(
        ...     binary_parameters=binary_parameters,
        ...     seed_adjacency_matrix=seed_adjacency_matrix
        ... )

    See Also:
        - BinaryGenerativeParameters: Parameters controlling binary network growth
        - WeightedGenerativeParameters: Parameters controlling weight optimisation
    """

    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        binary_parameters: BinaryGenerativeParameters,
        num_simulations: int,
        seed_adjacency_matrix: Union[
            Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
            Float[torch.Tensor, "num_nodes num_nodes"],
        ],
        distance_matrix: Optional[Float[torch.Tensor, "num_nodes num_nodes"]] = None,
        weighted_parameters: Optional[WeightedGenerativeParameters] = None,
        seed_weight_matrix: Optional[
            Union[
                Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
                Float[torch.Tensor, "num_nodes num_nodes"],
            ]
        ] = None,
    ):
        """The initialisation process for the Generative Network Model:

        1. Validates input matrices (symmetry, binary values, etc.).
        2. Stores the binary parameters and optionally the weighted parameters.
        3. Precomputes a distance factor matrix based on distance_relationship_type.
        4. If weighted parameters are provided, prepares the weight matrix and optimiser.

        Args:
            binary_parameters:
                Parameters controlling network growth.
            num_simulations:
                Number of simulations to run in parallel
            seed_adjacency_matrix:
                Initial network structure(s). Must be a binary symmetric matrix.
            distance_matrix:
                Physical distances between nodes. Must be symmetric and non-negative. If not provided,
                all distances are set to 1.
            weighted_parameters:
                Parameters controlling weight optimisation. If None, only binary growth is performed.
            seed_weight_matrix:
                Initial weight matrix for weighted networks(s). If None but weighted parameters
                are provided, a matrix matching the adjacency support is used.

        Raises:
            ValueError: If input matrices don't meet requirements (binary, symmetric, etc.) or
                        if weight matrix doesn't match adjacency support.
        """
        # -----------------
        # Handle batch dimensions
        # Check if seed matrix has batch dimension
        if len(seed_adjacency_matrix.shape) > 2:
            if seed_adjacency_matrix.shape[0] != num_simulations:
                raise ValueError(
                    f"Seed adjacency matrix batch size ({seed_adjacency_matrix.shape[0]}) "
                    f"does not match number of simulations ({num_simulations})"
                )
        else:
            # Add batch dimension if not present
            seed_adjacency_matrix = seed_adjacency_matrix.unsqueeze(0).expand(
                num_simulations, -1, -1
            )

        # If seed weight matrix is provided, handle its batch dimension
        if seed_weight_matrix is not None:
            if len(seed_weight_matrix.shape) > 2:
                if seed_weight_matrix.shape[0] != num_simulations:
                    raise ValueError(
                        f"Seed weight matrix batch size ({seed_weight_matrix.shape[0]}) "
                        f"does not match number of simulations ({num_simulations})"
                    )
            else:
                seed_weight_matrix = seed_weight_matrix.unsqueeze(0).expand(
                    num_simulations, -1, -1
                )

        # -----------------
        # Validate adjacency
        if not torch.all((seed_adjacency_matrix == 0) | (seed_adjacency_matrix == 1)):
            raise ValueError("seed_adjacency_matrix must be binary (only 0s and 1s).")

        # Check symmetry across last two dimensions
        if not torch.allclose(
            seed_adjacency_matrix, seed_adjacency_matrix.transpose(-2, -1)
        ):
            raise ValueError("seed_adjacency_matrix must be symmetric.")

        # Check and remove self-connections
        if torch.any(torch.diagonal(seed_adjacency_matrix, dim1=-2, dim2=-1) != 0):
            print("Removing self-connections from adjacency matrix.")
            seed_adjacency_matrix = seed_adjacency_matrix.clone()
            diagonal_indices = torch.arange(seed_adjacency_matrix.shape[-1])
            seed_adjacency_matrix[..., diagonal_indices, diagonal_indices] = 0

        # -----------------
        # Validate distance
        if distance_matrix is None:
            distance_matrix = torch.ones(
                (seed_adjacency_matrix.shape[-2], seed_adjacency_matrix.shape[-1]),
                dtype=seed_adjacency_matrix.dtype,
                device=seed_adjacency_matrix.device,
            )

        if not torch.allclose(distance_matrix, distance_matrix.T):
            raise ValueError("distance_matrix must be symmetric.")
        if torch.any(distance_matrix < 0):
            raise ValueError("distance_matrix must be non-negative.")

        self.seed_adjacency_matrix = seed_adjacency_matrix
        self.adjacency_matrix = seed_adjacency_matrix.clone()
        self.distance_matrix = distance_matrix
        self.num_nodes = seed_adjacency_matrix.shape[-1]
        self.num_simulations = num_simulations

        # -----------------
        # Store binary parameters
        self.binary_parameters = binary_parameters

        # Precompute distance factor - expand to match batch dimension
        if self.binary_parameters.distance_relationship_type == "powerlaw":
            self.distance_factor = (
                distance_matrix.pow(self.binary_parameters.eta)
                .unsqueeze(0)
                .expand(num_simulations, -1, -1)
            )
        elif self.binary_parameters.distance_relationship_type == "exponential":
            self.distance_factor = (
                torch.exp(self.binary_parameters.eta * distance_matrix)
                .unsqueeze(0)
                .expand(num_simulations, -1, -1)
            )
        else:
            raise ValueError(
                f"Unsupported distance relationship: {self.binary_parameters.distance_relationship_type}"
            )

        # -----------------
        # Weighted parameters
        self.weighted_parameters = weighted_parameters

        if self.weighted_parameters is not None:
            self.weighted_initialisation(weighted_parameters, seed_weight_matrix)
        else:
            self.weighted_parameters = None
            self.seed_weight_matrix = None
            self.weight_matrix = None
            self.optimiser = None

    @jaxtyped(typechecker=typechecked)
    def weighted_initialisation(
        self,
        weighted_parameters: WeightedGenerativeParameters,
        seed_weight_matrix: Optional[
            Union[
                Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
                Float[torch.Tensor, "num_nodes num_nodes"],
            ]
        ] = None,
    ):
        """Initialise the weight matrix and optimiser for the weighted GNM.
        If weighted parameters are not passed in during initialisation, this method
        must be called before any weighted updates can be performed.

        Args:
            weighted_parameters:
                Parameters controlling weight optimisation.
            seed_weight_matrix:
                A seed weight matrix to initialise $W_{ij}$.
                If this is not provided, then the weight matrix is initialised to the
                current adjacency matrix, $W_{ij} \gest A_{ij}$.
                If provided, the matrix must be symmetric, non-negative, and have support
                only where the adjacency matrix is non-zero.
                Defaults to None.

        Raises:
            ValueError: If the seed_weight_matrix is not symmetric, non-negative, or has
                        support where the adjacency matrix is zero.

        See Also:
            - WeightedGenerativeParameters: Parameters controlling weight optimisation
            - weighted_update: Method that uses these parameters
            - __init__: Initialisation method that calls this function if weighted_parameters are provided.
        """
        self.weighted_parameters = weighted_parameters

        # If user didn't provide seed_weight_matrix, initialise from adjacency.
        if seed_weight_matrix is None:
            print("No weight matrix provided. Initialising from adjacency matrix.")
            seed_weight_matrix = self.adjacency_matrix.clone()
        else:
            # Handle batch dimension if needed
            if len(seed_weight_matrix.shape) == 2:
                seed_weight_matrix = seed_weight_matrix.unsqueeze(0).expand(
                    self.num_simulations, -1, -1
                )
            elif seed_weight_matrix.shape[0] != self.num_simulations:
                raise ValueError(
                    f"Seed weight matrix batch size ({seed_weight_matrix.shape[0]}) "
                    f"does not match number of simulations ({self.num_simulations})"
                )

        # Validate user-provided weight matrix.
        # Check symmetry along last two dimensions
        if not torch.allclose(seed_weight_matrix, seed_weight_matrix.transpose(-2, -1)):
            raise ValueError("seed_weight_matrix must be symmetric.")

        if torch.any(seed_weight_matrix < 0):
            raise ValueError("seed_weight_matrix must be non-negative.")

        # Check support against adjacency matrix using broadcasting
        if torch.any((self.adjacency_matrix == 0) & (seed_weight_matrix != 0)):
            raise ValueError(
                "seed_weight_matrix must have support only where adjacency is non-zero."
            )

        # If the checks pass, store the weight matrix and initialise the optimiser.
        self.seed_weight_matrix = seed_weight_matrix

        # Create a copy for the actual weight matrix that will be optimised.
        self.weight_matrix = self.seed_weight_matrix.clone().requires_grad_(True)

        # Initialise optimiser.
        self.optimiser = optim.SGD(
            [self.weight_matrix],
            lr=self.weighted_parameters.alpha,
            maximize=self.weighted_parameters.maximise_criterion,
        )

    @jaxtyped(typechecker=typechecked)
    def binary_update(
        self,
        heterochronous_matrix: Optional[
            Union[
                Float[torch.Tensor, "num_simulations num_nodes num_nodes"],
                Float[torch.Tensor, "num_nodes num_nodes"],
            ]
        ] = None,
    ) -> Tuple[
        Float[torch.Tensor, "num_simulations 2"],  # added edges for each simulation
        Float[
            torch.Tensor, "num_simulations num_nodes num_nodes"
        ],  # updated adjacency matrices
    ]:
        """
        Performs one update step of the adjacency matrix for the binary GNM.
        To perform an update, the model calculates the unnormalised wiring probabilities for each edge
        not currently present within the adjacency matrix (i.e., all notes with $A_{ij} = 0$).
        The wiring probability $(i,j)$ based on a distance factor $d_{ij}$, a preferential wiring
        factor $k_{ij}$, and a developmental factor $h_{ij}$.
        The unnormalised probability is proportional to the product of these factors:
        $$
        P_{ij} = d_{ij} \\times k_{ij} \\times h_{ij}
        $$
        These probabilities are then postprocessed by:

        1. Set the probability for all existing connections to be zero, $P_{ij} \gets P_{ij} \\times (1 - A_{ij})$
        2. Set the probability of self-connections to be zero, $P_{ii} \gets 0$
        3. Add on a small offset to prevent division by zero, $P_{ij} \gets P_{ij} + \\epsilon$
        4. Normalise the probabilities to sum to one, $P_{ij} \gets P_{ij} / \sum_{kl} P_{kl}$

        An edge $(a,b)$ is then sampled from the normalised probabilities.
        This edge is added to the adjacency matrix, $A_{ab} \gets 1, A_{ba} \gets 1$.
        If the model is weighted, the edge is also added to the weight matrix, $W_{ab} \gets 1, W_{ba} \gets 1$.

        Args:
            heterochronous_matrix:
                The heterochronous development matrix $H_{ij}$ for this time step. Can be provided
                for each simulation in the batch or as a single matrix to be used across all simulations.
                Defaults to None.

        Returns:
            added_edges: The edges that were added to each adjacency matrix in the batch
            adjacency_matrices: (A copy of) the updated adjacency matrices after the binary update

        See Also:
            - BinaryGenerativeParameters: Parameters controlling binary network growth
            - GenerativeRule: Base class for generative rules that transform an adjacency matrix $A_{ij}$ into a preferential wiring matrix $K_{ij}$
        """
        # Handle heterochronous matrix
        if heterochronous_matrix is None:
            heterochronous_matrix = torch.ones(
                (self.num_nodes, self.num_nodes),
                dtype=self.seed_adjacency_matrix.dtype,
                device=self.seed_adjacency_matrix.device,
            )

        # If heterochronous matrix isn't batched, expand it
        if len(heterochronous_matrix.shape) == 2:
            heterochronous_matrix = heterochronous_matrix.unsqueeze(0).expand(
                self.num_simulations, -1, -1
            )
        elif heterochronous_matrix.shape[0] != self.num_simulations:
            raise ValueError(
                f"Heterochronous matrix batch size ({heterochronous_matrix.shape[0]}) "
                f"does not match number of simulations ({self.num_simulations})"
            )

        # Implement generative rule - already handles batch dimensions
        matching_index_matrix = self.binary_parameters.generative_rule(
            self.adjacency_matrix
        )

        # Add prob_offset to prevent zero to the power of negative number
        matching_index_matrix[
            matching_index_matrix == 0
        ] += self.binary_parameters.prob_offset

        # Calculate factors - broadcasting handles batch dimensions
        if self.binary_parameters.preferential_relationship_type == "powerlaw":
            matching_factor = matching_index_matrix.pow(self.binary_parameters.gamma)
            heterochronous_factor = torch.exp(
                self.binary_parameters.lambdah * heterochronous_matrix
            )
        elif self.binary_parameters.preferential_relationship_type == "exponential":
            matching_factor = torch.exp(
                self.binary_parameters.gamma * matching_index_matrix
            )
            heterochronous_factor = torch.exp(
                self.binary_parameters.lambdah * heterochronous_matrix
            )

        # Calculate unnormalised wiring probabilities for each edge
        unnormalised_wiring_probabilities = (
            heterochronous_factor * self.distance_factor * matching_factor
        )

        # Add prob_offset to prevent division by zero
        unnormalised_wiring_probabilities += self.binary_parameters.prob_offset

        # Set probability for existing connections to zero
        unnormalised_wiring_probabilities = unnormalised_wiring_probabilities * (
            1 - self.adjacency_matrix
        )

        # Set diagonal to zero to prevent self-connections
        diagonal_indices = torch.arange(
            self.num_nodes, device=self.adjacency_matrix.device
        )
        unnormalised_wiring_probabilities[..., diagonal_indices, diagonal_indices] = 0

        # Normalize the wiring probabilities for each simulation
        wiring_probability = (
            unnormalised_wiring_probabilities
            / unnormalised_wiring_probabilities.sum(dim=(-2, -1), keepdim=True)
        )

        # Sample edges for each simulation in the batch
        flattened_probs = wiring_probability.view(self.num_simulations, -1)
        edge_indices = torch.multinomial(flattened_probs, num_samples=1).squeeze(-1)

        # Convert to node pairs
        first_nodes = edge_indices // self.num_nodes
        second_nodes = edge_indices % self.num_nodes
        added_edges = torch.stack([first_nodes, second_nodes], dim=-1)

        # Add edges to adjacency matrices
        batch_indices = torch.arange(
            self.num_simulations, device=self.adjacency_matrix.device
        )
        self.adjacency_matrix[batch_indices, first_nodes, second_nodes] = 1
        self.adjacency_matrix[batch_indices, second_nodes, first_nodes] = 1

        # Add edges to weight matrices if they exist
        if hasattr(self, "weight_matrix"):
            self.weight_matrix.data[batch_indices, first_nodes, second_nodes] = 1
            self.weight_matrix.data[batch_indices, second_nodes, first_nodes] = 1

        # Return the added edges and copies of updated adjacency matrices
        return added_edges, self.adjacency_matrix.clone()

    @jaxtyped(typechecker=typechecked)
    def weighted_update(
        self,
    ) -> Float[torch.Tensor, "num_simulations num_nodes num_nodes"]:
        """
        Performs one update step of the weight matrix $W_{ij}$ for the weighted GNM. The weights are updated
        using gradient descent on the specified optimisation criterion, with the learning rate $\\alpha$:
        $$
        W_{ij} \gets W_{ij} - \\alpha \\frac{\partial L}{\partial W_{ij}}
        $$
        Following the update step, the following postprocessing steps are performed:
        1. Symmetry: The weight matrix is made symmetric by averaging with its transpose, $W \gets (1/2)(W + W^T)$.
        2. Clipping: The weights are clipped to the specified bounds $W_{\\rm lower} \leq W_{ij} \leq W_{\\rm upper}$.
        3. Consistency with binary adjacency: All weights where the adjacency matrix is zero are set to zero, so that if $A_{ij} = 0$ then $W_{ij} = 0$.

        Raises:
            AttributeError: If the model does not have a weight matrix, optimisation criterion, or optimiser.

        Returns:
            weight_matrix: (A detached copy of) the updated weight matrix, $W_{ij}$
        """
        # Check that the model has a weight matrix, optimisation criterion, and optimiser
        if not hasattr(self, "weight_matrix"):
            raise AttributeError(
                "Model does not have a weight matrix. Cannot perform weighted updates. Please call the weighted_initialisation method first."
            )
        if not hasattr(self, "optimisation_criterion"):
            raise AttributeError(
                "Model does not have an optimisation criterion. Cannot perform weighted updates. Please call the weighted_initialisation method first."
            )
        if not hasattr(self, "optimiser"):
            raise AttributeError(
                "Model does not have an optimiser. Cannot perform weighted updates. Please call the weighted_initialisation method first."
            )

        # Perform the optimisation step on the weights.
        # Compute the loss - criterion outputs shape (num_simulations,)
        loss = self.optimisation_criterion(self.weight_matrix)
        # Sum over the batch to get a scalar loss
        total_loss = torch.sum(loss)

        # Compute the gradients
        self.optimiser.zero_grad()
        total_loss.backward()

        # Update the weights
        self.optimiser.step()

        # Ensure the weight matrix is symmetric
        self.weight_matrix.data = 0.5 * (
            self.weight_matrix.data + self.weight_matrix.data.transpose(-2, -1)
        )

        # Clip the weights to the specified bounds
        self.weight_matrix.data = torch.clamp(
            self.weight_matrix.data,
            self.weighted_parameters.weight_lower_bound,
            self.weighted_parameters.weight_upper_bound,
        )

        # Zero out all weights where the adjacency matrix is zero
        self.weight_matrix.data = self.weight_matrix.data * self.adjacency_matrix

        # Return the updated weight matrix
        return self.weight_matrix.detach().clone().cpu()

    @jaxtyped(typechecker=typechecked)
    def run_model(
        self,
        heterochronous_matrix: Union[
            Float[
                torch.Tensor, "num_binary_updates num_simulations num_nodes num_nodes"
            ],
            Float[torch.Tensor, "num_binary_updates num_nodes num_nodes"],
        ] = None,
    ) -> Tuple[
        Float[
            torch.Tensor, "num_binary_updates num_simulations 2"
        ],  # added edges for each update
        Float[
            torch.Tensor,
            "num_binary_updates num_simulations num_nodes num_nodes",
        ],  # Adjacency snapshots
        Optional[
            Float[
                torch.Tensor,
                "num_weight_updates num_simulations num_nodes num_nodes",
            ]  # Weight snapshots
        ],
    ]:
        """Trains the network for a specified number of iterations.
        At each iteration, a number of binary updates and weighted updates are performed.

        Args:
            heterochronous_matrix:
                The heterochronous development probability matrix, $H_{ij}(t)$, for each binary update step $t$.
                Can be provided for each simulation in the batch or as a single matrix sequence to be used
                across all simulations. Defaults to None.

        Returns:
            added_edges: The edges $(a,b)$ that were added to the adjacency matrices $A_{ij}$ at each iteration.
            adjacency_snapshots: The adjacency matrices $A_{ij}$ at each binary update step.
            weight_snapshots: The weight matrices $W_{ij}$ at each iteration of the weighted updates.
        """
        num_iterations = self.binary_parameters.num_iterations
        added_edges_list = []
        binary_updates_per_iteration = (
            self.binary_parameters.binary_updates_per_iteration
        )
        total_binary_updates = num_iterations * binary_updates_per_iteration

        # Initialize snapshots with steps and batch dimensions at start
        adjacency_snapshots = torch.zeros(
            (
                total_binary_updates,
                self.num_simulations,
                self.num_nodes,
                self.num_nodes,
            ),
            device=self.adjacency_matrix.device,
            dtype=self.adjacency_matrix.dtype,
        )

        if self.weighted_parameters is not None:
            weighted_updates_per_iteration = (
                self.weighted_parameters.weighted_updates_per_iteration
            )
            total_weighted_updates = num_iterations * weighted_updates_per_iteration
            weight_snapshots = torch.zeros(
                (
                    total_weighted_updates,
                    self.num_simulations,
                    self.num_nodes,
                    self.num_nodes,
                ),
                device=self.adjacency_matrix.device,
                dtype=self.adjacency_matrix.dtype,
            )
        else:
            weighted_updates_per_iteration = 0
            weight_snapshots = None

        # Handle heterochronous matrix
        if heterochronous_matrix is None:
            heterochronous_matrix = torch.ones(
                (
                    total_binary_updates,
                    self.num_nodes,
                    self.num_nodes,
                ),
                dtype=self.adjacency_matrix.dtype,
                device=self.adjacency_matrix.device,
            )

        # Expand heterochronous matrix if it doesn't have batch dimension
        if len(heterochronous_matrix.shape) == 3:
            heterochronous_matrix = heterochronous_matrix.unsqueeze(1).expand(
                -1, self.num_simulations, -1, -1
            )
        elif heterochronous_matrix.shape[1] != self.num_simulations:
            raise ValueError(
                f"Heterochronous matrix batch size ({heterochronous_matrix.shape[1]}) "
                f"does not match number of simulations ({self.num_simulations})"
            )

        for ii in tqdm(range(num_iterations)):
            for jj in range(binary_updates_per_iteration):
                update_idx = ii * binary_updates_per_iteration + jj
                added_edges, adjacency_matrix = self.binary_update(
                    heterochronous_matrix[update_idx]
                )
                adjacency_snapshots[update_idx] = adjacency_matrix
                added_edges_list.append(added_edges)

            for jj in range(weighted_updates_per_iteration):
                update_idx = ii * weighted_updates_per_iteration + jj
                weight_matrix = self.weighted_update()
                weight_snapshots[update_idx] = weight_matrix

        if weighted_updates_per_iteration == 0:
            return added_edges_list, adjacency_snapshots, None

        return added_edges_list, adjacency_snapshots, weight_snapshots
