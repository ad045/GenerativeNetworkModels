import importlib
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
import GNM
import generative_rules
importlib.reload(GNM)
from GNM import GenerativeNetworkModel
import scipy.io
from nilearn import plotting
import plotly
from param_sweep import *

import generative_rules as generative_rules
importlib.reload(generative_rules)


num_nodes = 100

test_connectome = torch.rand(num_nodes, num_nodes)
distance_matrix = torch.ones((num_nodes, num_nodes))

# start off with an empty seed
seed_adjacency_matrix = torch.zeros(num_nodes, num_nodes)


# Set other parameters
eta = -3.0
gamma = 0.5
lambdah = 0
distance_relationship_type = "exponential"
matching_relationship_type = "exponential"
generative_rule = generative_rules.MatchingIndex(mode='in', divisor='mean')

# gnm = GenerativeNetworkModel(seed_adjacency_matrix = seed_adjacency_matrix,
#                 distance_matrix = distance_matrix,
#                 eta = eta,
#                 gamma = gamma,
#                 lambdah = lambdah,
#                 distance_relationship_type = distance_relationship_type,
#                 matching_relationship_type = matching_relationship_type,
#                 alpha = 0,
#                 optimisation_normalisation=True,
#                 generative_rule=generative_rule,
#                 optimisation_criterion_kwargs = {"omega":1},
#                 optimisation_criterion = None
# )

# added_edges_list, adjacency_snapshot, weight_snapshots = gnm.train_loop(num_iterations = 500, 
#                                                                         binary_updates_per_iteration=1, 
#                                                                         weighted_updates_per_iteration=1) 

param_sweep_vals = ParameterSweepValues(
    method = 'grid',
    eta= (-3, 3),
    gamma = (-3, 3),
    alpha = (-3, 3),
    lambdah = 3,
    omega = 3,
    weight_bounds = (-10, 10),
    distance_relationship_type = 'exponential',
    matching_relationship_type = 'exponential',
    optimisation_criterion = 'weighted_distance',
    maximise_criterion = True,
    num_iterations = 100,
    binary_updates_per_iteration = 2,
    weighted_updates_per_iteration= 1
)

parameter_sweep = GridSearch(test_connectome, 
                             distance_matrix,
                             param_sweep_vals,
                             use_wandb=False,
                             grid_step_space=10)

parameter_sweep()


# plt.figure(figsize=(5, 5))
# plt.imshow(gnm.adjacency_matrix.numpy())
# plt.title("Final adjacency matrix")
# plt.show()

# plt.figure(figsize=(5, 5))
# plt.imshow(gnm.weight_matrix.detach().numpy())
# plt.title("Final weight matrix")
# plt.show()

