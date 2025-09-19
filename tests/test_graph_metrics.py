import bct
import numpy as np
from gnm.utils import graph_properties as gnm_metrics
from gnm.defaults import get_weighted_network
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from gnm.utils.statistics import ks_statistic
import os
import matplotlib.pyplot as plt
import networkx as nx
from jaxtyping import Float, Int
import torch
import numpy as np
from typing import Literal
import networkx as nx
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from typing import Optional, Union


@jaxtyped(typechecker=typechecked)
def compare_exact(
    connectome_1: Float[np.ndarray, "n_nodes n_nodes"],
    connectome_2: Float[np.ndarray, "n_nodes n_nodes"],
    metric_used: str
) -> None:
    assert np.allclose(connectome_1, connectome_2, atol=1e-2), \
        f"From Metric {metric_used}, Exact Adj. Matrices don't match!"
    
@jaxtyped(typechecker=typechecked)
def compare_exact_metric(
    connectome_1: Float[np.ndarray, "n_nodes"],
    connectome_2: Float[np.ndarray, "n_nodes"],
    metric_used: str,
    verbose: bool = True,
) -> None:
    
    if verbose:    
        print(connectome_1)
        print('-' * 50)
        print(connectome_2)

    assert np.allclose(connectome_1, connectome_2, atol=1e-2), \
        f"From Metric {metric_used}, Exact Adj. Matrices don't match!"


@jaxtyped(typechecker=typechecked)
def compare_cosine(
    connectome_1: Float[np.ndarray, "n_nodes n_nodes"],
    connectome_2: Float[np.ndarray, "n_nodes n_nodes"],
    metric_used: str
) -> None:
    cosine_sim = cosine_similarity(connectome_1.reshape(1, -1), connectome_2.reshape(1, -1))
    assert cosine_sim >= 0.9, f"From Metric {metric_used}, Cosine Similarity is {cosine_sim}, Failed!"


@jaxtyped(typechecker=typechecked)
def compare_ks(
    connectome_1: Float[np.ndarray, "n_nodes n_nodes"],
    connectome_2: Float[np.ndarray, "n_nodes n_nodes"],
    metric_used: str
) -> None:
    connectome_1 = torch.tensor(connectome_1).unsqueeze(0)
    connectome_2 = torch.tensor(connectome_2).unsqueeze(0)
    ks = ks_statistic(connectome_1, connectome_2)
    assert ks < 0.1, f'From Metric {metric_used}, KS Statistic Failed, KS={np.round(ks, 3)}'

# ---- Metric-specific comparisons ----

@jaxtyped(typechecker=typechecked)
def compare_node_strength(
    connectome: Float[np.ndarray, "n_nodes n_nodes"]
) -> None:
    connectome_torch = torch.tensor(connectome)
    gnm_node_strength = gnm_metrics.node_strengths(connectome_torch).cpu().numpy()
    bct_node_strengths = bct.strengths_und(connectome)
    compare_exact(gnm_node_strength, bct_node_strengths, 'Node Strength')


@jaxtyped(typechecker=typechecked)
def compare_binary_clustering_coefficients(
    connectome: Float[np.ndarray, "n_nodes n_nodes"]
) -> None:
    connectome_tensor = torch.tensor(connectome).unsqueeze(0)
    gnm_clust = gnm_metrics.binary_clustering_coefficients(connectome_tensor)
    gnm_clust = gnm_clust.cpu().numpy().reshape(-1)
    bct_clust = bct.clustering_coef_bu(connectome)
    compare_exact(gnm_clust, bct_clust, 'Binary Clustering Coefficient')


@jaxtyped(typechecker=typechecked)
def compare_weighted_clustering_coefficients(
    connectome: Float[np.ndarray, "n_nodes n_nodes"]
) -> None:
    connectome_tensor = torch.tensor(connectome).unsqueeze(0)
    gnm_clust = gnm_metrics.weighted_clustering_coefficients(connectome_tensor)
    gnm_clust = gnm_clust.cpu().numpy().reshape(-1)
    bct_clust = bct.clustering_coef_wu(connectome)

    compare_exact(gnm_clust, bct_clust, 'Weighted Clustering Coefficient')


@jaxtyped(typechecker=typechecked)
def compare_binary_betweenness_centrality(
    connectome: Float[np.ndarray, "n_nodes n_nodes"]
) -> None:
    connectome_tensor = torch.tensor(connectome).unsqueeze(0)
    gnm_bc = gnm_metrics.binary_betweenness_centrality(connectome_tensor).squeeze(0)
    gnm_bc = gnm_bc.cpu().numpy()
    bct_bc = bct.betweenness_bin(connectome)

    print(gnm_bc)
    print('-' * 50)
    print(bct_bc)

    compare_exact(gnm_bc, bct_bc, 'Binary Betweeness Centrality')

@jaxtyped(typechecker=typechecked)
def compare_weighted_betweenness_centrality(
    connectome: Float[np.ndarray, "n_nodes n_nodes"]
) -> None:
    connectome_tensor = torch.tensor(connectome).unsqueeze(0)
    gnm_bc = gnm_metrics.weighted_betweenness_centrality(connectome_tensor).squeeze(0)
    gnm_bc = gnm_bc.cpu().numpy()
    bct_bc = bct.betweenness_wei(connectome)
    
    scaler = MinMaxScaler((0, 1))
    bct_bc = scaler.fit_transform(bct_bc.reshape(-1, 1)).reshape(-1) / 2

    compare_exact_metric(gnm_bc, bct_bc, 'Weighted Betweeness Centrality', verbose=True)


@jaxtyped(typechecker=typechecked)
def compare_characteristic_path_length(
    connectome: Float[torch.Tensor, "n_nodes n_nodes"]
) -> None:
    connectome = torch.tensor(connectome).unsqueeze(0)
    network_nx = nx.from_numpy_array(network_nx)
    nx_charpath = nx.average_shortest_path_length(network_nx)
    gnm_charpath = gnm_metrics.binary_characteristic_path_length(connectome).item()

    assert np.isclose(gnm_charpath, nx_charpath, atol=1e-2), \
        f"Characteristic Path Length Failed, GNM={gnm_charpath}, NetworkX={nx_charpath}"


# weighted_connectome = np.load('./tests/mean_connectome.npy')
weighted_connectome = get_weighted_network().squeeze(0).cpu().numpy()
# weighted_connectome = scaler.fit_transform(weighted_connectome)
weighted_connectome = np.maximum(weighted_connectome, weighted_connectome.T)
np.fill_diagonal(weighted_connectome, 0) # no self-connections

binary_connectome = np.where(weighted_connectome > 0.4, 1, 0)
binary_connectome = np.maximum(binary_connectome, binary_connectome.T) # symmetry
np.fill_diagonal(binary_connectome, 0) # no self-connections

# compare_binary_clustering_coefficients(binary_connectome)
# compare_weighted_clustering_coefficients(weighted_connectome)
# compare_node_strength(weighted_connectome)
# compare_binary_betweenness_centrality(binary_connectome)

compare_weighted_betweenness_centrality(weighted_connectome)