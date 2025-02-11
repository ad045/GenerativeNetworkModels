from jaxtyping import Float, jaxtyped
from typing import Optional, Tuple, List
from typeguard import typechecked

from optimisation_criteria import OptimisationCriterion

import torch
import torch.optim as optim

from generative_rules import *
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
    prob_offset: float = 1e-6
    generative_rule: GenerativeRule

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
    where L is the optimisation criterion and $\\alpha$ is the learning rate.
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

        optimisation_normalisation (bool):
            Whether to normalise the optimisation criterion before computing
            gradients. Normalisation can help prevent numerical instability
            by keeping values in a reasonable range.

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
        >>> weighted_parameters = WeightedGenerativeParameters(
        ...     alpha=0.01,  # Small learning rate for stable optimisation
        ...     optimisation_criterion=DistanceWeightedCommunicability(
        ...         normalisation=True,
        ...         distance_matrix=D,
        ...         omega=1.0
        ...     ),
        ...     optimisation_normalisation=True,
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
    optimisation_normalisation: bool
    weight_lower_bound: float = 0.0
    weight_upper_bound: float = float("inf")
    maximise_criterion: bool = False


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

    See Also:
        - BinaryGenerativeParameters: Parameters controlling binary network growth
        - WeightedGenerativeParameters: Parameters controlling weight optimisation
    """

    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        binary_parameters: BinaryGenerativeParameters,
        seed_adjacency_matrix: Float[torch.Tensor, "num_nodes num_nodes"],
        distance_matrix: Optional[Float[torch.Tensor, "num_nodes num_nodes"]] = None,
        weighted_parameters: Optional[WeightedGenerativeParameters] = None,
        seed_weight_matrix: Optional[Float[torch.Tensor, "num_nodes num_nodes"]] = None,
    ):
        """Initialise a new Generative Network Model using the specified parameters.

        The initialisation process:

        1. Validates input matrices (symmetry, binary values, etc.).
        2. Stores the binary parameters and optionally the weighted parameters.
        3. Precomputes a distance factor matrix based on distance_relationship_type.
        4. If weighted parameters are provided, prepares the weight matrix and optimiser.

        Args:
            binary_parameters:
                Parameters controlling network growth.
            seed_adjacency_matrix:
                Initial network structure. Must be a binary symmetric matrix.
            distance_matrix:
                Physical distances between nodes. Must be symmetric and non-negative. If not provided,
                all distances are set to 1.
            weighted_parameters:
                Parameters controlling weight optimisation. If None, only binary growth is performed.
            seed_weight_matrix:
                Initial weight matrix for weighted networks. If None but weighted parameters
                are provided, a matrix matching the adjacency support is used.

        Raises:
            ValueError: If input matrices don't meet requirements (binary, symmetric, etc.) or
                        if weight matrix doesn't match adjacency support.
        """
        # -----------------
        # Validate adjacency
        if not torch.all((seed_adjacency_matrix == 0) | (seed_adjacency_matrix == 1)):
            raise ValueError("seed_adjacency_matrix must be binary (only 0s and 1s).")
        if not torch.allclose(seed_adjacency_matrix, seed_adjacency_matrix.T):
            raise ValueError("seed_adjacency_matrix must be symmetric.")
        if torch.any(torch.diag(seed_adjacency_matrix) != 0):
            print("Removing self-connections from adjacency matrix.")
            seed_adjacency_matrix = seed_adjacency_matrix.clone()
            seed_adjacency_matrix.fill_diagonal_(0)

        # -----------------
        # Validate distance
        if not torch.allclose(distance_matrix, distance_matrix.T):
            raise ValueError("distance_matrix must be symmetric.")
        if torch.any(distance_matrix < 0):
            raise ValueError("distance_matrix must be non-negative.")

        self.seed_adjacency_matrix = seed_adjacency_matrix
        self.adjacency_matrix = seed_adjacency_matrix.clone()
        if distance_matrix is None:
            distance_matrix = torch.ones(
                (seed_adjacency_matrix.shape[0], seed_adjacency_matrix.shape[1]),
                dtype=seed_adjacency_matrix.dtype,
            )
        else:
            self.distance_matrix = distance_matrix
        self.num_nodes = seed_adjacency_matrix.shape[0]

        # -----------------
        # Store binary parameters
        self.binary_parameters = binary_parameters

        # Precompute distance factor
        if self.binary_parameters.distance_relationship_type == "powerlaw":
            self.distance_factor = distance_matrix.pow(self.binary_parameters.eta)
        elif self.binary_parameters.distance_relationship_type == "exponential":
            self.distance_factor = torch.exp(
                self.binary_parameters.eta * distance_matrix
            )
        else:
            raise ValueError(
                f"Unsupported distance relationship: {self.binary_parameters.distance_relationship_type}"
            )

        # -----------------
        # Weighted parameters
        if self.weighted_parameters is not None:
            self.weighted_initialisation(weighted_parameters, seed_weight_matrix)
        else:
            self.weighted_parameters = None
            self.seed_weight_matrix = None
            self.weight_matrix = None
            self.optimiser = None

    @typechecked
    def weighted_initialisation(
        self,
        weighted_parameters: WeightedGenerativeParameters,
        seed_weight_matrix: Optional[Float[torch.Tensor, "num_nodes num_nodes"]] = None,
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

        # Validate user-provided weight matrix.
        if not torch.allclose(seed_weight_matrix, seed_weight_matrix.T):
            raise ValueError("seed_weight_matrix must be symmetric.")
        if torch.any(seed_weight_matrix < 0):
            raise ValueError("seed_weight_matrix must be non-negative.")
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
            Float[torch.Tensor, "{self.num_nodes} {self.num_nodes}"]
        ] = None,
    ) -> Tuple[Tuple[int, int], Float[torch.Tensor, "num_nodes num_nodes"]]:
        """
        Performs one update step of the adjacency matrix for the binary GNM.
        To perform an update, the model calculates the unnormalised wiring probabilities for each edge
        not  currently present within the adjacency matrix (i.e., all notes with $A_{ij} = 0$).
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
                The heterochronous development matrix $H_{ij}$ for this time step. Defaults to None.

        Returns:
            added_edges: The edge $(a,b)$ that was added to the adjacency matrix, $A_{ab} \gets 1, A_{ba} \gets 1$.
            adjacency_matrix: (A copy of) the updated adjacency matrix after the binary update, $A_{ij}$.

        See Also:
            - BinaryGenerativeParameters: Parameters controlling binary network growth
            - GenerativeRule: Base class for generative rules that transform an adjacency matrix $A_{ij}$ into a preferential wiring matrix $K_{ij}$
        """

        if heterochronous_matrix is None:
            heterochronous_matrix = torch.ones(
                (self.num_nodes, self.num_nodes), dtype=self.seed_adjacency_matrix.dtype
            )

        # implement generative rule
        matching_index_matrix = self.generative_rule(
            self.adjacency_matrix
        )  # matching_index(self.adjacency_matrix)

        # Add on the prob_offset term to prevent zero to the power of negative number
        matching_index_matrix[matching_index_matrix == 0] += self.prob_offset

        if self.matching_relationship_type == "powerlaw":
            matching_factor = matching_index_matrix.pow(self.gamma)
            heterochronous_factor = torch.exp(self.lambdah * heterochronous_matrix)
            # heterochronous_factor = heterochronous_matrix.pow(self.lambdah)
        elif self.matching_relationship_type == "exponential":
            matching_factor = torch.exp(self.gamma * matching_index_matrix)
            heterochronous_factor = torch.exp(self.lambdah * heterochronous_matrix)

        # Calculate the unnormalised wiring probabilities for each edge.
        unnormalised_wiring_probabilities = (
            heterochronous_factor * self.distance_factor * matching_factor
        )
        # Add on the prob_offset term to prevent division by zero
        unnormalised_wiring_probabilities += self.prob_offset
        # Set the probability for all existing connections to be zero
        unnormalised_wiring_probabilities = unnormalised_wiring_probabilities * (
            1 - self.adjacency_matrix
        )
        # Set the diagonal to zero to prevent self-connections
        unnormalised_wiring_probabilities.fill_diagonal_(0)
        # Normalize the wiring probability
        wiring_probability = (
            unnormalised_wiring_probabilities / unnormalised_wiring_probabilities.sum()
        )
        # Sample an edge to add
        edge_idx = torch.multinomial(wiring_probability.view(-1), num_samples=1).item()
        first_node = edge_idx // self.num_nodes
        second_node = edge_idx % self.num_nodes
        # Add the edge to the adjacency matrix
        self.adjacency_matrix[first_node, second_node] = 1
        self.adjacency_matrix[second_node, first_node] = 1
        # Record the added edge
        added_edges = (first_node, second_node)
        # Add the edge to the weight matrix if it exists
        if hasattr(self, "weight_matrix"):
            self.weight_matrix.data[first_node, second_node] = 1
            self.weight_matrix.data[second_node, first_node] = 1

        # Return the added edge and (a copy of) the updated adjacency matrix
        return added_edges, self.adjacency_matrix.clone()

    @jaxtyped(typechecker=typechecked)
    def weighted_update(
        self,
    ) -> Float[torch.Tensor, "{self.num_nodes} {self.num_nodes}"]:
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
        # Compute the loss
        loss = self.optimisation_criterion(self.weight_matrix)
        # Compute the gradients
        self.optimiser.zero_grad()
        loss.backward()
        # Update the weights
        self.optimiser.step()
        # Ensure the weight matrix is symmetric
        self.weight_matrix.data = 0.5 * (
            self.weight_matrix.data + self.weight_matrix.data.T
        )
        # Clip the weights to the specified bounds
        self.weight_matrix.data = torch.clamp(
            self.weight_matrix.data, self.weight_lower_bound, self.weight_upper_bound
        )
        # Zero out all weights where the adjacency matrix is zero
        self.weight_matrix.data = self.weight_matrix.data * self.adjacency_matrix

        # Return the updated weight matrix
        return self.weight_matrix.detach().clone().cpu()

    @jaxtyped(typechecker=typechecked)
    def train_loop(
        self,
        num_iterations: int,
        binary_updates_per_iteration: int = 1,
        weighted_updates_per_iteration: int = 1,
        heterochronous_matrix: Optional[
            Float[
                torch.Tensor,
                "{self.num_nodes} {self.num_nodes} {num_iterations*binary_updates_per_iteration}",
            ]
        ] = None,
    ) -> Tuple[
        List[Tuple[int, int]],
        Float[
            torch.Tensor,
            "{self.num_nodes} {self.num_nodes} {num_iterations*binary_updates_per_iteration}",
        ],
        Optional[
            Float[
                torch.Tensor,
                "{self.num_nodes} {self.num_nodes} {num_iterations*weighted_updates_per_iteration}",
            ]
        ],
    ]:
        """Trains the network for a specified number of iterations.
        At each iteration, a number of binary updates and weighted updates are performed.

        Args:
            num_iterations:
                The number of iterations to update the network for.
            binary_updates_per_iteration:
                The number of binary updates to perform at each iteration. Defaults to 1.
            weighted_updates_per_iteration:
                The number of weighted updates to perform at each iteration. Defaults to 1.
            heterochronous_matrix:
                The heterochronous development probability matrix, $H_{ij}(t)$, for each binary update step $t$. Defaults to None.

        Returns:
            added_edges: The edges $(a,b)$ that were added to the adjacency matrix $A_{ij}$ at each iteration.
            adjacency_snapshots: The adjacency matrices $A_{ij}$ at each binary update step.
            weight_snapshots: The weight matrices $W_{ij}$ at each iteration of the weighted updates.
        """

        added_edges_list = []
        adjacency_snapshots = torch.zeros(
            (
                self.num_nodes,
                self.num_nodes,
                num_iterations * binary_updates_per_iteration,
            )
        )

        if weighted_updates_per_iteration != 0:
            assert hasattr(
                self, "weight_matrix"
            ), "Weighted updates per iteration was specified, but no alpha value was initialised."
            weight_snapshots = torch.zeros(
                (
                    self.num_nodes,
                    self.num_nodes,
                    num_iterations * weighted_updates_per_iteration,
                )
            )

        if heterochronous_matrix is None:
            heterochronous_matrix = torch.ones(
                (
                    self.num_nodes,
                    self.num_nodes,
                    num_iterations * binary_updates_per_iteration,
                ),
                dtype=self.seed_adjacency_matrix.dtype,
            )

        for ii in tqdm(range(num_iterations)):
            for jj in range(binary_updates_per_iteration):
                added_edges, adjacency_matrix = self.binary_update(
                    heterochronous_matrix[:, :, ii * binary_updates_per_iteration + jj]
                )
                adjacency_snapshots[:, :, ii] = adjacency_matrix
                added_edges_list.append(added_edges)

            for jj in range(weighted_updates_per_iteration):
                weight_matrix = self.weighted_update()
                weight_snapshots[:, :, ii * weighted_updates_per_iteration + jj] = (
                    weight_matrix
                )

        if weighted_updates_per_iteration == 0:
            return added_edges_list, adjacency_snapshots, None

        return added_edges_list, adjacency_snapshots, weight_snapshots
