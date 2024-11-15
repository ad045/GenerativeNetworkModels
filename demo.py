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
importlib.reload(GNM)
from GNM import GenerativeNetworkModel
import scipy.io
from nilearn import plotting
import plotly

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


def generate_heterochronous_matrix(coord, setting="posterior", sigma=1.0, num_nodes=100, mseed=0, cumulative=False, local=True):
    """
    Generates a heterochronous matrix based on node coordinates.

    Parameters:
    - coord: numpy array of shape (num_nodes, 3) representing the coordinates of brain regions (nodes).
    - setting: String, determines the reference point for heterochrony. Options are "posterior", "anterior", "internal", "external".
    - sigma: Float, standard deviation for the Gaussian spread.
    - num_nodes: Integer, total number of time steps.
    - mseed: Integer, number of initial seeds.
    - cumulative: Boolean, whether to use cumulative maximum of probabilities.
    - local: Boolean, if True use outer product method, if False use max-based method.

    Returns:
    - heterochronous_matrix: torch.Tensor of shape (num_nodes, num_nodes - mseed)
    """
    # Step 1: Find the reference node based on the setting
    if setting == "posterior" or setting == "anterior":
        # Find the most posterior node (node with the smallest y-coordinate)
        _, posterior_node = np.min(coord[:, 1]), np.argmin(coord[:, 1])
        reference_coord = coord[posterior_node, :]
    elif setting == "internal" or setting == "external":
        # Use a fixed internal coordinate as the reference
        reference_coord = np.array([0, -12, 5])
    else:
        raise ValueError("Unsupported setting: {}. Use 'posterior', 'anterior', 'internal', or 'external'.".format(setting))

    # Step 2: Compute Euclidean distance from reference node to all nodes
    distances = np.sqrt(np.sum((coord - reference_coord) ** 2, axis=1))
    max_distance = np.max(distances)  # Maximum distance for setting Gaussian means

    # Step 3: Calculate means for Gaussian function at each time step
    means = np.linspace(0, max_distance, num_nodes - mseed)  # Linearly spaced means from 0 to max_distance
    heterochronous_matrix = np.zeros((len(distances), num_nodes - mseed))  # Initialize the matrix to store probabilities

    # Gaussian function for probability calculation
    P = lambda d, mu: (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((d - mu) ** 2) / (2 * sigma ** 2))

    # Step 4: Calculate probabilities at each time step
    for t in range(num_nodes - mseed):
        mu = means[t]  # Current mean at time step t
        if setting == "anterior" or setting == "external":
            heterochronous_matrix[:, -1 - t] = P(distances, mu)  # For "outside-in" or "front-back" propagation
        else:
            heterochronous_matrix[:, t] = P(distances, mu)  # For "posterior" or "internal" propagation

    # Step 5: Apply cumulative maximum if requested
    if cumulative:
        heterochronous_matrix = np.maximum.accumulate(heterochronous_matrix, axis=1)

    # Step 6: Convert to matrix form based on local parameter
    heterochronous_matrices = []
    for t in range(num_nodes - mseed):
        Ht = heterochronous_matrix[:, t]
        H_rescaled = (Ht - np.min(Ht)) / (np.max(Ht) - np.min(Ht))  # Normalize H to [0, 1]
        if local:
            # Local method: outer product
            Hmat = np.outer(H_rescaled, H_rescaled)
        else:
            # Non-local method: max-based
            N = len(H_rescaled)
            rows, cols = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
            Hmat = np.maximum(H_rescaled[rows], H_rescaled[cols])
        heterochronous_matrices.append(Hmat)

    # Convert list of matrices to torch tensor
    heterochronous_matrices_tensor = torch.tensor(np.stack(heterochronous_matrices, axis=-1), dtype=torch.float32)

    return heterochronous_matrices_tensor

# connection length - phase where long range connections are encouraged


# Set other parameters
eta = -3.0 #problem with eta = 9
gamma = 0.5
lambdah = 0
distance_relationship_type = "exponential"
matching_relationship_type = "exponential"

# Calculate the standard deviation of the coordinates
sigma = np.std(coord)

# Generate the heterochronous matrices tensor with 20 steps for individual plots
heterochronous_matrices_tensor = generate_heterochronous_matrix(coord, setting="posterior", sigma=sigma, num_nodes=8, mseed=0, cumulative=False, local=True)

# Plot the evolution of the network over time using nilearn's plot_markers
n_steps = heterochronous_matrices_tensor.shape[-1]
for t in range(n_steps):
    Ht = heterochronous_matrices_tensor[:, :, t].numpy()
    node_values = Ht.diagonal()  # Use the diagonal elements to represent nodal values for coloring
    plotting.plot_markers(
        node_values,
        node_coords=coord,
        node_cmap='viridis',
        #title=f'Gaussian Drift (Step {t + 1}/{n_steps})',
        node_size=50,
        display_mode='x',
        output_file=None  # Display directly in the Spyder plot pane
    )
    plt.pause(0.05)  # Pause to visualize each step

plt.show()

# Create a figure with 8 subplots arranged in one row and 8 columns
fig, axes = plt.subplots(1, 8, figsize=(32, 4))

# Plot the evolution of the network over time using nilearn's plot_markers in 8 subplots
n_steps = heterochronous_matrices_tensor.shape[-1]
for t in range(n_steps):
    ax = axes[t]
    Ht = heterochronous_matrices_tensor[:, :, t].numpy()
    node_values = Ht.diagonal()  # Use the diagonal elements to represent nodal values for coloring
    display = plotting.plot_markers(
        node_values,
        node_coords=coord,
        node_cmap='viridis',
        node_size=50,
        display_mode='x',
        axes=ax,
        output_file=None,
        colorbar=(t == n_steps - 1)  # Show colorbar only for the last subplot
    )
    display._colorbar = None # Remove colorbar for all but the last plot

#plt.tight_layout()
plt.show()

# decide seed type in a separate script
# none
# minimal consensus
# heterochronous
## direction: back-front vs inside-outside
## long-range: True/False
## cumulative: True/False

gnm = GenerativeNetworkModel(seed_adjacency_matrix = seed_adjacency_matrix,
                distance_matrix = distance_matrix,
                eta = eta,
                gamma = gamma,
                lambdah = lambdah,
                distance_relationship_type = distance_relationship_type,
                matching_relationship_type = matching_relationship_type,
                alpha = 0, 
                optimisation_criterion_kwargs = {"omega":1},
                optimisation_normalisation=True
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


W = gnm.weight_matrix.detach().numpy()
W.max()

