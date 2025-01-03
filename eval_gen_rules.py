"""
Author: Will Mills
Purpose: check to see if there is difference between original GNM code
with modular rules
"""


import importlib
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
import GNM
from GNM import GenerativeNetworkModel
import rules
from GNM_original import OriginalGenerativeNetworkModel
import bct
from scipy.stats import ks_2samp

def run_model(model_type:str):    
    num_nodes = 100
    num_seed_edges = 250

    seed_adjacency_matrix = torch.zeros(num_nodes, num_nodes)
    # Randomly select seed edges
    seed_edge_indices = torch.randint(0, num_nodes, (num_seed_edges, 2))
    seed_adjacency_matrix[seed_edge_indices[:, 0], seed_edge_indices[:, 1]] = 1
    seed_adjacency_matrix[seed_edge_indices[:, 1], seed_edge_indices[:, 0]] = 1
    # Remove diagonals
    seed_adjacency_matrix.fill_diagonal_(0)

    # Set the distance matrix to all ones
    distance_matrix = torch.ones(num_nodes, num_nodes)
    distance_matrix.fill_diagonal_(0)

    # Set other parameters
    eta = 1
    gamma = 1
    distance_relationship_type = "exponential"
    matching_relationship_type = "exponential"

    if model_type == 'og':
        gmn = OriginalGenerativeNetworkModel(seed_adjacency_matrix = seed_adjacency_matrix,
                        distance_matrix = distance_matrix,
                        eta = eta,
                        gamma = gamma,
                        lambdah = 1,
                        distance_relationship_type = distance_relationship_type,
                        matching_relationship_type = matching_relationship_type,
                        alpha = 0.01, 
                        optimisation_criterion_kwargs = {"omega":1},
                        optimisation_normalisation=False
        )
    else:
        gmn = GenerativeNetworkModel(seed_adjacency_matrix = seed_adjacency_matrix,
                        distance_matrix = distance_matrix,
                        eta = eta,
                        gamma = gamma,
                        lambdah = 1,
                        distance_relationship_type = distance_relationship_type,
                        matching_relationship_type = matching_relationship_type,
                        alpha = 0.01, 
                        optimisation_criterion_kwargs = {"omega":1},
                        optimisation_normalisation=False
        )

    added_edges_list, adjacency_snapshot, weight_snapshots = gmn.train_loop(num_iterations = 1000, binary_updates_per_iteration=1, weighted_updates_per_iteration=1)
    
    matrix = gmn.adjacency_matrix.numpy()
    return matrix

def eval_gnm():
    
    new_mat = run_model('matching_index')
    og_mat = run_model('og')

    og_density = bct.density_und(og_mat)
    new_density = bct.density_und(new_mat)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(og_mat)
    axes[0].set_title(f"Original adjacency matrix\nDensity {og_density}")

    axes[1].imshow(new_mat)
    axes[1].set_title(f"Matching index adjacency matrix\nDensity {new_density}")

    centrality_og = bct.betweenness_bin(og_mat)
    centrality_new = bct.betweenness_bin(new_mat)
    ks_centrality, p = ks_2samp(centrality_og, centrality_new)

    plt.xlabel(f'Centrality ks: {ks_centrality:.3} | p-value: {p:.3}')
    plt.show()

eval_gnm()