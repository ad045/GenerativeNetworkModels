import bct
import numpy as np
from gnm.utils import graph_properties as gnm_metrics
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from gnm.utils.statistics import ks_statistic
import os
import matplotlib.pyplot as plt

# TODO: jaxtyping stuff

# comparison of connectomes using different methods
def compare_exact(connectome_1: np.array, connectome_2: np.array, metric_used: str):
    assert np.allclose(connectome_1, connectome_2, atol=1e-2), \
        f"From Metric {metric_used}, Exact Adj. Matrices don't match!"

def compare_cosine(connectome_1: np.array, connectome_2: np.array, metric_used: str):
    cosine_sim = cosine_similarity(connectome_1.reshape(1, -1), connectome_2.reshape(1, -1))
    assert cosine_sim >= 0.9, f"From Metric {metric_used}, Cosine Similarity is {cosine_sim}, Failed!"

def compare_ks(connectome_1:np.array, connectome_2:np.array, metric_used:str):
    connectome_1 = torch.Tensor(connectome_1).unsqueeze(0)
    connectome_2 = torch.Tensor(connectome_2).unsqueeze(0)
    ks = ks_statistic(connectome_1, connectome_2)
    assert ks < 0.1, f'From Metric {metric_used}, KS Statistic Failed, KS={np.round(ks, 3)}'

# metrics to compare connectomes against
def compare_node_strength(connectome:np.array):
    connectome_torch = torch.Tensor(connectome)
    gnm_node_strength = gnm_metrics.node_strengths(connectome_torch).cpu().numpy()
    bct_node_strengths = bct.strengths_und(connectome)
    compare_exact(gnm_node_strength, bct_node_strengths, 'Node Strength')

def compare_binary_clustering_coefficients(connectome:np.array):
    connectome_tensor = torch.Tensor(connectome).unsqueeze(0)
    gnm_clust = gnm_metrics.binary_clustering_coefficients(connectome_tensor)
    gnm_clust = gnm_clust.cpu().numpy()
    gnm_clust = gnm_clust.reshape(-1)
    bct_clust = bct.clustering_coef_bu(connectome)
    compare_exact(gnm_clust, bct_clust, 'Binary Clustering Coefficient')

def compare_weighted_clustering_coefficients(connectome:np.array):
    connectome_tensor = torch.Tensor(connectome).unsqueeze(0)
    gnm_clust = gnm_metrics.weighted_clustering_coefficients(connectome_tensor)
    gnm_clust = gnm_clust.cpu().numpy()
    gnm_clust = gnm_clust.reshape(-1)
    bct_clust = bct.clustering_coef_wu(connectome)

    compare_exact(gnm_clust, bct_clust, 'Weighted Clustering Coefficient')

def compare_binary_betweenness_centrality(connectome:np.array):
    connectome_tensor = torch.Tensor(connectome).unsqueeze(0)
    gnm_bc = gnm_metrics.binary_betweenness_centrality(connectome_tensor)
    gnm_bc = gnm_bc.cpu().numpy()
    gnm_bc = gnm_bc.reshape(-1)
    bct_bc = bct.betweenness_bin(connectome)
    compare_exact(gnm_bc, bct_bc, 'Binary Betweeness Centrality')

scaler = MinMaxScaler((0, 1))
weighted_connectome = np.load('./tests/mean_connectome.npy')
weighted_connectome = scaler.fit_transform(weighted_connectome)
weighted_connectome = np.maximum(weighted_connectome, weighted_connectome.T)
np.fill_diagonal(weighted_connectome, 0) # no self-connections

binary_connectome = np.where(weighted_connectome > 0.4, 1, 0)
binary_connectome = np.maximum(binary_connectome, binary_connectome.T) # symmetry
np.fill_diagonal(binary_connectome, 0) # no self-connections

compare_binary_clustering_coefficients(binary_connectome)
compare_weighted_clustering_coefficients(weighted_connectome)
compare_node_strength(weighted_connectome)