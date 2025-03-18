import bct
import numpy as np
from gnm import *
from gnm import defaults, utils, evaluation, fitting, generative_rules, weight_criteria
from gnm.generative_rules import MatchingIndex
import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import gc
# import cProfile

import numpy as np

def generate_adj_matrix(N, density, directed=False, self_loops=False):
    """Generate an NxN binary adjacency matrix with a given density of connections."""
    # Create an empty matrix
    adj_matrix = np.zeros((N, N), dtype=int)
    num_ones = int(N * N * density)
    
    # Select random indices for ones
    indices = np.random.choice(N * N, num_ones, replace=False)
    adj_matrix.flat[indices] = 1  # Assign ones at selected indices
    
    # Ensure no self-loops
    if not self_loops:
        np.fill_diagonal(adj_matrix, 0)

    # Ensure symmetry for undirected graphs
    if not directed:
        adj_matrix = np.triu(adj_matrix)  
        adj_matrix += adj_matrix.T   

    return adj_matrix


def simulate_bct(num_simulations, 
                 eta, 
                 gamma, 
                 connectome,
                 distance_matrix):
    
    num_connections = int(connectome.sum().item() / 2)
    start_time = time.perf_counter()
    for _ in tqdm(range(num_simulations), desc='BCT Simulations', disable=True):
        
        generative_model = bct.generative_model(connectome, 
                                                distance_matrix, 
                                                num_connections, 
                                                eta, 
                                                gamma, 
                                                'matching',
                                                'powerlaw',
                                                num_connections)
        
    end_time = time.perf_counter()
    return end_time - start_time


def simulate_gnm(num_simulations, eta, gamma, connectome, batch_size):
    start_time = time.perf_counter()
    num_connections = int( (connectome.sum().item() / 2  ) / 10)

    binary_parameters = BinaryGenerativeParameters(
        eta=eta,
        gamma=gamma,
        lambdah=0,
        distance_relationship_type='exponential',
        preferential_relationship_type='powerlaw',
        heterochronicity_relationship_type='powerlaw',
        generative_rule=(MatchingIndex()),
        num_iterations=num_connections,
        binary_updates_per_iteration=1,
    )

    batches = [batch_size] * (num_simulations // batch_size) + [num_simulations % batch_size]
    if batches[-1] == 0:
        batches = batches[:-1]

    for batch_size in batches:
        model = GenerativeNetworkModel(
            binary_parameters=binary_parameters,
            num_simulations=batch_size,
            distance_matrix=distance_matrix_torch,
            verbose=False
        )

    model.run_model()
    gc.collect()
    torch.cuda.empty_cache()
    end_time = time.perf_counter()

    return end_time - start_time


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE} for GNM simulations')

time_gnm = []
time_bct = []
df = {'connectome_size': [], 'density':[], 'time_bct': []} # 'time_gnm': []
connectome_size_range = range(250, 300, 10)
density = np.linspace(0.1, 0.9, 9)

for d in density:
    for connectome_size in tqdm(connectome_size_range, desc='Connectome Size', leave=False):
        connectome_size = int(connectome_size)

        tqdm.write(f'connectome_size: {connectome_size}, density: {d}')

        # setup simulation connectome
        coords = np.random.rand(connectome_size, 3)  # Change to (N, 3) for 3D
        dist_matrix_np = squareform(pdist(coords, metric='euclidean'))
        binary_consensus_network_np = np.ones((connectome_size, connectome_size))

        adj_matrix = generate_adj_matrix(connectome_size, d, directed=False, self_loops=False)

        # Remove self-connections
        np.fill_diagonal(binary_consensus_network_np, 0)
        distance_matrix_torch = torch.Tensor(dist_matrix_np).to(DEVICE)
        binary_consensus_network_torch = torch.Tensor(binary_consensus_network_np).unsqueeze(0).to(DEVICE)

        # set params 
        eta = -0.1
        gamma = 0.1
        num_simulations = 100
        batch_size = 16
        
        #gnm_time = simulate_gnm(num_simulations, eta, gamma, binary_consensus_network_torch, batch_size)
        bct_time = simulate_bct(num_simulations, [eta], [gamma], binary_consensus_network_np, dist_matrix_np)

        df['connectome_size'].append(connectome_size)
        df['density'].append(d)
        #df['time_gnm'].append(bct_time)
        df['time_bct'].append(bct_time)

        # save as you go
        df_pd = pd.DataFrame(df)
        df_pd.to_csv('results_bct_2.csv', index=False)

