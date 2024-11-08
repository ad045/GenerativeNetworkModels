import wandb
import torch
from GNM import GenerativeNetworkModel

wandb.login()

# Define the sweep configuration
sweep_configuration = {
    "name": "parameter_sweep",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "max_KS_statistic"},
    "parameters": {
        "eta": {"min": 0.1, "max": 0.9},
        "gamma": {"min": 0.1, "max": 0.9},
        "lambdah": {"min": 0.1, "max": 0.9},
        "distance_relationship_type": {"values": ["powerlaw", "exponential"]},
        "matching_relationship_type": {"values": ["powerlaw", "exponential"]},
        "alpha": {"min": 0.1, "max": 0.9},
        "optimisation_criterion": {"values": ["distance_weighted_communicability", "weighted_distance"]},
        "omega": {"min": 0.1, "max": 0.9},
        "optimisation_normalisation": {"values": [True, False]},
        "weight_lower_bound": {"value": 0.0},
        "weight_upper_bound": {"value": None},
        "maximise_criterion": {"values": [True, False]},
        "num_iterations": {"min": 100, "max": 1000},
        "binary_updates_per_iteration": {"value": 1},
        "weighted_updates_per_iteration": {"min": 1, "max": 10},
    },
}


def main():
    wandb.init(project="GNM")
    
    # Unpack the hyperparameters from the sweep config.
    eta = wandb.config.eta
    gamma = wandb.config.gamma
    lambdah = wandb.config.lambdah
    distance_relationship_type = wandb.config.distance_relationship_type
    matching_relationship_type = wandb.config.matching_relationship_type
    alpha = wandb.config.alpha
    optimisation_criterion = wandb.config.optimisation_criterion
    omega = wandb.config.omega
    optimisation_normalisation = wandb.config.optimisation_normalisation
    weight_lower_bound = wandb.config.weight_lower_bound
    weight_upper_bound = wandb.config.weight_upper_bound
    maximise_criterion = wandb.config.maximise_criterion
    num_iterations = wandb.config.num_iterations
    binary_updates_per_iteration = wandb.config.binary_updates_per_iteration
    weighted_updates_per_iteration = wandb.config.weighted_updates_per_iteration

    if optimisation_criterion == "distance_weighted_communicability":
        optimisation_criterion_kwargs = {"omega": omega}
    else:
        optimisation_criterion_kwargs = {}

    # Define a seed adjacency matrix and distance matrix
    # This will be where you want to feed in data
    num_nodes = 100
    num_seed_edges = 250

    seed_adjacency_matrix = torch.zeros(num_nodes, num_nodes)
    seed_edge_indices = torch.randint(0, num_nodes, (num_seed_edges, 2))
    seed_adjacency_matrix[seed_edge_indices[:, 0], seed_edge_indices[:, 1]] = 1
    seed_adjacency_matrix[seed_edge_indices[:, 1], seed_edge_indices[:, 0]] = 1
    seed_adjacency_matrix.fill_diagonal_(0)

    distance_matrix = torch.ones((num_nodes, num_nodes))
    seed_weight_matrix = seed_adjacency_matrix.clone()

    # Set up the generative model
    gnm = GenerativeNetworkModel(
        seed_adjacency_matrix=seed_adjacency_matrix,
        distance_matrix=distance_matrix,
        eta=eta,
        gamma=gamma,
        lambdah=lambdah,
        distance_relationship_type=distance_relationship_type,
        matching_relationship_type=matching_relationship_type,
        seed_weight_matrix=seed_weight_matrix,
        alpha=alpha,
        optimisation_criterion=optimisation_criterion,
        optimisation_criterion_kwargs=optimisation_criterion_kwargs,
        optimisation_normalisation=optimisation_normalisation,
        weight_lower_bound=weight_lower_bound,
        weight_upper_bound=weight_upper_bound,
        maximise_criterion=maximise_criterion,
    )

    # Train the model
    gnm.train_loop(num_iterations=num_iterations, binary_updates_per_iteration=binary_updates_per_iteration, weighted_updates_per_iteration=weighted_updates_per_iteration)

    # Compute model statistics
    # THIS WILL BE WHERE YOU COMPUTE THE STATISTICS FOR EACH MODEL
    degree_KS_statistic = 0
    clustering_KS_statistic = 0
    betweenness_KS_statistic = 0
    edge_length_KS_statistic = 0

    # Log the model statistics
    wandb.log({
        "degree_KS_statistic": degree_KS_statistic,
        "clustering_KS_statistic": clustering_KS_statistic,
        "betweenness_KS_statistic": betweenness_KS_statistic,
        "edge_length_KS_statistic": edge_length_KS_statistic,
    })

    # Compute and log the maximum KS statistic
    max_KS_statistic = max(degree_KS_statistic, clustering_KS_statistic, betweenness_KS_statistic, edge_length_KS_statistic)
    wandb.log({"max_KS_statistic": max_KS_statistic})



sweep_id = wandb.sweep(sweep_configuration, project="GNM")
wandb.agent(sweep_id, function=main, count=10)