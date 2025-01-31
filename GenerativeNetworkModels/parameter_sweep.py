import wandb
import torch
import networkx as nx
import numpy as np
from scipy.stats import ks_2samp
from GNM import GenerativeNetworkModel
import scipy.io

wandb.login()

# Define the sweep configuration
sweep_configuration = {
    "name": "parameter_sweep",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "max_KS_statistic"},
    "parameters": {
        "eta": {"min": -5, "max": 5},
        "gamma": {"min": -5, "max": 5},
        "lambdah": {"value": 0}, # {"min": 0.1, "max": 0.9},
        "distance_relationship_type": {"value": "exponential"}, # {"values": ["powerlaw", "exponential"]},
        "matching_relationship_type": {"value": "exponential"}, #{"values": ["powerlaw", "exponential"]},
        "alpha": {"min": 0, "max": 1},
        "optimisation_criterion": {"value": "distance_weighted_communicability"}, #{"values": ["distance_weighted_communicability", "weighted_distance"]},
        "omega": {"value": 1}, #{"min": 0.1, "max": 0.9},
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
    # Load the provided .mat file to explore its contents
    mat_file_path = r'../Data/Consensus/consensus_wgm_h.mat'

    # Load the .mat file
    mat_contents = scipy.io.loadmat(mat_file_path)
    # Extract the first element from 'k_atlases' to explore its structure
    k_atlas_element = mat_contents['k_atlases'][0]

    # Import consensus network for aal (unused in gnm for now)
    atlas = k_atlas_element[0]
    consensus_matrix = atlas['consensus'][0, 0]

    # Import the distance matrix
    distance_matrix = torch.tensor(atlas['euclidean'][0, 0])

    # Import the coordinates
    ##coord = atlas['coordinates'][0, 0]

    # set number of nodes and edges
    num_nodes = len(consensus_matrix)
    #num_seed_edges = np.sum(consensus_matrix>0)/2

    # start off with an empty seed
    seed_adjacency_matrix = torch.zeros(num_nodes, num_nodes)

    #seed_edge_indices = torch.randint(0, num_nodes, (num_seed_edges, 2))
    #seed_adjacency_matrix[seed_edge_indices[:, 0], seed_edge_indices[:, 1]] = 1
    #seed_adjacency_matrix[seed_edge_indices[:, 1], seed_edge_indices[:, 0]] = 1
    #seed_adjacency_matrix.fill_diagonal_(0)

    #distance_matrix = torch.ones((num_nodes, num_nodes))
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
    
    # Computereal and model statistics
    
    # Create the graph
    Greal = nx.from_numpy_array(consensus_matrix)
    # Compute metrics
    real_weighted_degree_list = list(dict(Greal.degree(weight='weight')).values())
    real_clustering_coefficients_list = list(nx.clustering(Greal, weight='weight').values())
    real_betweenness_centrality_list = list(nx.betweenness_centrality(Greal, weight='weight').values())
    
    # Extract distances for connected nodes
    real_connected_indices = np.triu(consensus_matrix, k=1) > 0
    real_connected_distances = distance_matrix.numpy()[real_connected_indices]
    
    
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
    connected_distances = distance_matrix.numpy()[connected_indices]
    
    # Compute KS statistics
    degree_KS_statistic = ks_2samp(real_weighted_degree_list, weighted_degree_list).statistic
    clustering_KS_statistic = ks_2samp(real_clustering_coefficients_list, clustering_coefficients_list).statistic
    betweenness_KS_statistic = ks_2samp(real_betweenness_centrality_list, betweenness_centrality_list).statistic
    edge_length_KS_statistic = ks_2samp(real_connected_distances, connected_distances).statistic

    # Log the model statistics
    wandb.log({
        "degree_KS_statistic": degree_KS_statistic,
        "clustering_KS_statistic": clustering_KS_statistic,
        "betweenness_KS_statistic": betweenness_KS_statistic,
        "edge_length_KS_statistic": edge_length_KS_statistic,
    })

    # Compute and log the maximum KS statistic
    max_KS_statistic = max(degree_KS_statistic, clustering_KS_statistic, betweenness_KS_statistic, edge_length_KS_statistic)
    # Log the model statistics along with hyperparameters
    wandb.log({
        "eta": eta,
        "gamma": gamma,
        "lambdah": lambdah,
        "alpha": alpha,
        "optimisation_normalisation": optimisation_normalisation,
        "maximise_criterion": maximise_criterion,
        "degree_KS_statistic": degree_KS_statistic,
        "clustering_KS_statistic": clustering_KS_statistic,
        "betweenness_KS_statistic": betweenness_KS_statistic,
        "edge_length_KS_statistic": edge_length_KS_statistic,
        "max_KS_statistic": max_KS_statistic,
        "num_iterations": num_iterations,
        "weighted_updates_per_iteration": weighted_updates_per_iteration
    })




sweep_id = wandb.sweep(sweep_configuration, project="GNM")
wandb.agent(sweep_id, function=main, count=10)