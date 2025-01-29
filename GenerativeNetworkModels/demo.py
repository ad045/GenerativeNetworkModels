# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 18:32:05 2024

@author: fp02
"""

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

import generative_rules as generative_rules
importlib.reload(generative_rules)

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
coord = atlas['coordinates'][0, 0]

# set number of nodes and edges
num_nodes = len(consensus_matrix)
num_seed_edges = np.sum(consensus_matrix>0)/2

# start off with an empty seed
seed_adjacency_matrix = torch.zeros(num_nodes, num_nodes)


# Set other parameters
eta = -3.0 #problem with eta = 9
gamma = 0.5
lambdah = 0
distance_relationship_type = "exponential"
matching_relationship_type = "exponential"


# decide seed type in a separate script
# none
# minimal consensus
# heterochronous
## direction: back-front vs inside-outside
## long-range: True/False
## cumulative: True/False

generative_rule = generative_rules.MatchingIndex(mode='in', divisor='mean')
gnm = GenerativeNetworkModel(seed_adjacency_matrix = seed_adjacency_matrix,
                distance_matrix = distance_matrix,
                eta = eta,
                gamma = gamma,
                lambdah = lambdah,
                distance_relationship_type = distance_relationship_type,
                matching_relationship_type = matching_relationship_type,
                alpha = 0, 
                optimisation_criterion_kwargs = {"omega":1},
                optimisation_normalisation=True,
                generative_rule=generative_rule
)


added_edges_list, adjacency_snapshot, weight_snapshots = gnm.train_loop(num_iterations = 500, binary_updates_per_iteration=1, weighted_updates_per_iteration=1) 


plt.figure(figsize=(5, 5))
plt.imshow(gnm.adjacency_matrix.numpy())
plt.title("Final adjacency matrix")
plt.show()


plt.figure(figsize=(5, 5))
plt.imshow(gnm.weight_matrix.detach().numpy())
plt.title("Final weight matrix")
plt.show()

