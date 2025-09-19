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

def generate_adj_matrix(N, density):
    """Generate an NxN binary adjacency matrix with a given density of connections."""

    adj_matrix = np.zeros((N, N), dtype=int)
    num_ones = density
    
    # Select random indices for ones
    indices = np.random.choice(N * N, num_ones, replace=False)
    adj_matrix.flat[indices] = 1 

    # remove self loops
    np.fill_diagonal(adj_matrix, 0)

    # symmetrical
    adj_matrix = np.triu(adj_matrix)  
    adj_matrix += adj_matrix.T   

    return adj_matrix


def simulate_bct(num_simulations, 
                 eta, 
                 gamma, 
                 connectome,
                 num_connections,
                 distance_matrix):
    
    total_time = 0
    for _ in tqdm(range(num_simulations), desc='BCT Simulations', disable=True):
        while True:
            try:
                start_time = time.perf_counter()
                generative_model = bct.generative_model(connectome, 
                                                        distance_matrix, 
                                                        num_connections, 
                                                        eta, 
                                                        gamma, 
                                                        'matching',
                                                        'powerlaw',
                                                        num_connections)
                end_time = time.perf_counter()
                total_time += end_time - start_time
                break
            except:
                print('Error in BCT simulation, retrying...')
        
    return total_time

def simulate_gnm(num_simulations, eta, gamma, num_connections, distance_matrix, batch_size):
    start_time = time.perf_counter()

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

    #print(batches)
    for i in batches:
        model = GenerativeNetworkModel(
            binary_parameters=binary_parameters,
            num_simulations=i,
            distance_matrix=distance_matrix.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
            verbose=False
        )

    model.run_model()
    gc.collect()
    torch.cuda.empty_cache()
    end_time = time.perf_counter()

    return end_time - start_time


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE} for GNM simulations')

time_gnm = []
time_bct = []
df = {'connectome_size': [], 'num_connections':[], 'time_gnm': []} # , 'time_bct': [] 

connectome_size_range = list(reversed(range(100, 1600, 100)))
num_connections_range = list(reversed(range(100, 3100, 100))) 

for num_connections in tqdm(num_connections_range, leave=False):
    for connectome_size in connectome_size_range:
        connectome_size = int(connectome_size)

        tqdm.write(f'connectome_size: {connectome_size}, N connections: {num_connections}')

        # setup simulation connectome
        coords = np.random.rand(connectome_size, 3)  # Change to (N, 3) for 3D
        dist_matrix_np = squareform(pdist(coords, metric='euclidean'))
        dist_matrix_torch = torch.Tensor(dist_matrix_np).to(DEVICE)

        # seed network
        adj_matrix = generate_adj_matrix(connectome_size, num_connections)

        # set params 
        eta = -0.1
        gamma = 0.1
        num_simulations = 10
        batch_size = 10

        gc.collect()
        torch.cuda.empty_cache()
        
        gnm_time = simulate_gnm(num_simulations, eta, gamma, num_connections, dist_matrix_torch, batch_size)
        #bct_time = simulate_bct(num_simulations, [eta], [gamma], adj_matrix, num_connections, dist_matrix_np)

        df['connectome_size'].append(connectome_size)
        df['num_connections'].append(num_connections)
        
        df['time_gnm'].append(gnm_time)
        #df['time_bct'].append(bct_time)

        # save as you go
        df_pd = pd.DataFrame(df)
        df_pd.to_csv('gnm_gpu_results.csv', index=False)

