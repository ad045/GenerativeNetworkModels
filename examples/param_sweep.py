import wandb
import torch
import networkx as nx
import numpy as np
from scipy.stats import ks_2samp
from GNM import GenerativeNetworkModel
import scipy.io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from jaxtyping import Float, Int, jaxtyped
from typeguard import typechecked
from typing import Tuple, Literal

@dataclass
@jaxtyped(typechecker=typechecked)
class ParameterSweepValues:
    """
    The parameter bounds for exploration e.g. search
    from eta = -1 to eta = 1
    """

    method: Literal["bayes", "grid"]
    eta: Tuple[float, float] # minimum maximum
    gamma: Tuple[float, float]
    alpha: Tuple[float, float]
    lambdah: float
    omega: float
    weight_bounds: Tuple[float, float]
    distance_relationship_type: Literal["powerlaw", "exponential"]
    matching_relationship_type: Literal["powerlaw", "exponential"]
    optimisation_criterion: Literal["distance_weighted_communicability", "weighted_distance"]
    maximise_criterion: Tuple[bool, bool]
    num_iterations: Tuple[float, float]
    binary_updates_per_iteration: int
    weighted_updates_per_iteration: Tuple[float, float]

@dataclass
@jaxtyped(typechecker=typechecked)
class OptimizedParameters():
    """
    The actual parameters being optimized and their values in a given
    iteration/GNM loop.
    """
    eta: float = 0
    gamma: float = 0
    alpha: float = 0
    lambdah: float = 0
    omega: float = 0
    num_iterations: float = 0
    weighted_updates_per_iteration: float = 0

class ParamSweep(ABC):
    @jaxtyped(typechecker=typechecked)
    def __init__(self, 
                 adjacency_matrix:Float[torch.Tensor, 'num_nodes num_nodes'], 
                 distance_matrix:Float[torch.Tensor, 'num_nodes num_nodes'],
                 param_set:ParameterSweepValues, use_wandb:bool):
        self.matrix = adjacency_matrix
        self.use_wandb = use_wandb
        self.param_set = param_set
        self.distance_matrix = distance_matrix

        self.optimized_parameters = OptimizedParameters()

        if use_wandb:
            self.sweep_configuration = {
                "name": "parameter_sweep",
                "method": param_set.method,
                "metric": {"goal": "minimize", "name": "max_KS_statistic"},
                "parameters": {
                    "eta": {"min": param_set.eta[0], "max": param_set.eta[1]},
                    "gamma": {"min": param_set.gamma[0], "max": param_set.gamma[1]},
                    "lambdah": {"value": param_set.lambdah}, # {"min": 0.1, "max": 0.9},
                    "distance_relationship_type": {"value": param_set.distance_relationship_type}, # {"values": ["powerlaw", "exponential"]},
                    "matching_relationship_type": {"value": param_set.matching_relationship_type}, #{"values": ["powerlaw", "exponential"]},
                    "alpha": {"min": param_set.alpha[0], "max": param_set.alpha[1]},
                    "optimisation_criterion": {"value": param_set.optimisation_criterion}, #{"values": ["distance_weighted_communicability", "weighted_distance"]},
                    "omega": {"value": param_set.omega}, #{"min": 0.1, "max": 0.9},
                    "optimisation_normalisation": {"values": [True, False]},
                    "weight_lower_bound": {"value": 0.0},
                    "weight_upper_bound": {"value": None},
                    "maximise_criterion": {"values": [param_set.maximise_criterion[0], param_set.maximise_criterion[1]]},
                    "num_iterations": {"min": param_set.num_iterations[0], "max": param_set.num_iterations[1]},
                    "binary_updates_per_iteration": {"value": param_set.binary_updates_per_iteration},
                    "weighted_updates_per_iteration": {"min": param_set.weighted_updates_per_iteration[0], "max": param_set.weighted_updates_per_iteration[1]},
                },
            }

        self.num_nodes = self.matrix.shape[0]

    @abstractmethod
    def get_vals(self):
        """
        Gets the values required using wandb, grid search etc
        """
        pass
    
    @abstractmethod
    def log(self, info_to_log):
        """
        Method of logging, e.g. if wandb then send ks stat to wandb,
        else log in csv or similar
        """
        pass

    def run_parameter_set(self, eta, gamma, lambdah, alpha, num_iterations, weighted_updates_per_iteration):
        param_set = self.param_set
        seed_adjacency_matrix = torch.zeros(self.num_nodes, self.num_nodes)
        seed_weight_matrix = seed_adjacency_matrix.clone()

        # gnm = GenerativeNetworkModel(
        #     seed_adjacency_matrix=seed_adjacency_matrix,
        #     distance_matrix=self.distance_matrix,
        #     eta=eta,
        #     gamma=gamma,
        #     lambdah=lambdah,
        #     distance_relationship_type=param_set.distance_relationship_type,
        #     matching_relationship_type=param_set.matching_relationship_type,
        #     seed_weight_matrix=seed_weight_matrix,
        #     alpha=alpha,
        #     optimisation_criterion=param_set.optimisation_criterion,
        #     optimisation_criterion_kwargs=param_set.optimisation_criterion_kwargs,
        #     optimisation_normalisation=param_set.optimisation_normalisation,
        #     weight_lower_bound=param_set.weight_bounds[0],
        #     weight_upper_bound=param_set.weight_bounds[1],
        #     maximise_criterion=param_set.maximise_criterion,
        #     optimisation_criterion_kwargs = {"omega":1}
        # )

        gnm = GenerativeNetworkModel(seed_adjacency_matrix = seed_adjacency_matrix,
                distance_matrix = self.distance_matrix,
                eta = eta,
                gamma = gamma,
                lambdah = lambdah,
                distance_relationship_type=param_set.distance_relationship_type,
                matching_relationship_type=param_set.matching_relationship_type,
                alpha = 0,
                optimisation_normalisation=True,
                #generative_rule=generative_rule,
                optimisation_criterion_kwargs = {"omega":1},
                optimisation_criterion = None
        )


        gnm.train_loop(num_iterations=num_iterations, 
                       binary_updates_per_iteration=param_set.binary_updates_per_iteration, 
                       weighted_updates_per_iteration=weighted_updates_per_iteration)
        
        Greal = nx.from_numpy_array(self.matrix.cpu().detach().numpy())

        # Compute metrics
        real_weighted_degree_list = list(dict(Greal.degree(weight='weight')).values())
        real_clustering_coefficients_list = list(nx.clustering(Greal, weight='weight').values())
        real_betweenness_centrality_list = list(nx.betweenness_centrality(Greal, weight='weight').values())
        
        # Extract distances for connected nodes
        real_connected_indices = np.triu(self.matrix, k=1) > 0
        real_connected_distances = self.distance_matrix.numpy()[real_connected_indices]
        
        # take the weighted adjacency matrix (i.e., final synthetic network)
        wfinal_numpy = gnm.adjacency_matrix.numpy()
        
        # Create the graph
        G = nx.from_numpy_array(wfinal_numpy)
        
        # Compute metrics
        weighted_degree_list = list(dict(G.degree(weight='weight')).values())
        clustering_coefficients_list = list(nx.clustering(G, weight='weight').values())
        betweenness_centrality_list = list(nx.betweenness_centrality(G, weight='weight').values())
        
        # Extract distances for connected nodes
        connected_indices = np.triu(wfinal_numpy, k=1) > 0
        connected_distances = self.distance_matrix.numpy()[connected_indices]
        
        # Compute KS statistics
        degree_KS_statistic = ks_2samp(real_weighted_degree_list, weighted_degree_list).statistic
        clustering_KS_statistic = ks_2samp(real_clustering_coefficients_list, clustering_coefficients_list).statistic
        betweenness_KS_statistic = ks_2samp(real_betweenness_centrality_list, betweenness_centrality_list).statistic
        edge_length_KS_statistic = ks_2samp(real_connected_distances, connected_distances).statistic

        return degree_KS_statistic, clustering_KS_statistic, betweenness_KS_statistic, edge_length_KS_statistic

    def __call__(self):
        iterations = 0
        next_values = self.get_vals() 
        while next_values is not None and iterations < 100000:
            print(iterations)
            eta, gamma, lambdah, alpha, num_iterations, weighted_updates_per_iteration = self.get_vals() 
            info_to_log = self.run_parameter_set(eta, gamma, lambdah, alpha, num_iterations, weighted_updates_per_iteration)
            self.log(info_to_log)
            iterations += 1

    # # TODO: make more modular later - should be more than KS Stat optimization criterion
    # @abstractmethod
    # def optimization_criterion(self):
    #     pass

class BayesWandb(ParamSweep):
    def __init__(self):
        super().__init__()
        wandb.login()

    def get_vals(self):
        eta = wandb.config.eta
        gamma = wandb.config.gamma
        lambdah = wandb.config.lambdah
        alpha = wandb.config.alpha
        omega = wandb.config.omega
        num_iterations = wandb.config.num_iterations
        weighted_updates_per_iteration = wandb.config.weighted_updates_per_iteration
        return eta, gamma, lambdah, alpha, num_iterations, weighted_updates_per_iteration
    
    # TODO: implement kwargs properly
    def log(self, **kwargs):
        wandb.log({
                "degree_KS_statistic": kwargs.get('degree_KS_statistic'),
                "clustering_KS_statistic": kwargs.get('clustering_KS_statistic'),
                "betweenness_KS_statistic": kwargs.get('betweenness_KS_statistic'),
                "edge_length_KS_statistic": kwargs.get('edge_length_KS_statistic'),
            })

            # Compute and log the maximum KS statistic
        max_KS_statistic = max(kwargs.get('degree_KS_statistic'), 
                                   kwargs.get('clustering_KS_statistic'), 
                                   kwargs.get('betweenness_KS_statistic'),
                                    kwargs.get('edge_length_KS_statistic'))
        
            # Log the model statistics along with hyperparameters
        wandb.log({
                "eta": self.optimized_parameters.eta,
                "gamma": self.optimized_parameters.gamma,
                "lambdah": self.optimized_parameters.lambdah,
                "alpha": self.optimized_parameters.alpha,

                "optimisation_normalisation": self.param_set.optimisation_normalisation,
                "maximise_criterion": self.param_setmaximise_criterion,
                "degree_KS_statistic": kwargs.get('degree_KS_statistic'),
                "clustering_KS_statistic": kwargs.get('clustering_KS_statistic'),
                "betweenness_KS_statistic": kwargs.get('betweenness_KS_statistic'),
                "edge_length_KS_statistic": kwargs.get('edge_length_KS_statistic'),
                "max_KS_statistic": max_KS_statistic,
                "num_iterations": self.optimized_parameters.num_iterations,
                "weighted_updates_per_iteration": self.optimized_parameters.weighted_updates_per_iteration
            })


class GridSearch(ParamSweep):
    def __init__(self, 
                 adjacency_matrix:Float[torch.Tensor, 'num_nodes num_nodes'], 
                 distance_matrix:Float[torch.Tensor, 'num_nodes num_nodes'],
                 param_set:ParameterSweepValues, use_wandb:bool,
                 grid_step_space = 10):
        super().__init__(adjacency_matrix, distance_matrix, param_set, use_wandb)
        self.current_grid_step = 0

        alpha_min = self.param_set.alpha[0]
        alpha_max = self.param_set.alpha[1]

        gamma_min = self.param_set.gamma[0]
        gamma_max = self.param_set.gamma[1]

        eta_min = self.param_set.eta[0]
        eta_max = self.param_set.eta[1]

        # set up all parameters for grid search
        grid_space = []

        for a in np.linspace(alpha_min, alpha_max, grid_step_space):
            for e in np.linspace(eta_min, eta_max, grid_step_space):
                for gam in np.linspace(gamma_min, gamma_max, 10):  # Fixed step 10
                        grid_space.append({'alpha': a, 'eta': e, 'gamma':gam})

        self.grid_space = grid_space
        print('begin')


    def log(self, info_to_log):
        print('NA')

    def get_vals(self):
        p_set = self.param_set
        current_iteration_idx = self.current_grid_step

        if current_iteration_idx >= len(self.grid_space):
            return None

        current_values = self.grid_space[current_iteration_idx]
        self.current_grid_step += 1
        
        return (
            current_values['eta'],
            current_values['gamma'],
            self.param_set.lambdah,
            current_values['alpha'],
            self.param_set.num_iterations,
            self.param_set.weighted_updates_per_iteration
        )
