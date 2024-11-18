import torch

from typing import Tuple, List
from jaxtyping import Float, Int, jaxtyped
from typeguard import typechecked
import numpy as np

@jaxtyped(typechecker=typechecked)
def matching_index(
    adjacency_matrix:Float[torch.Tensor, "num_nodes num_nodes"],
    mode:str = "in",
    divisor:str = "union"
) -> Float[torch.Tensor, "num_nodes num_nodes"]:
    """
    Computes the normalised matching index for a given adjacency matrix.
    We assume that adjacency_matrix(i,j) = 1 if there is a connection from j to i. 

    Let N(i) be the neighbourhood of node i.

    When the divisor is set to 'mean', the matching index is computed as:
        MI(i,j) = ( |( N(i) intesect N(j) ) - {i,j}| )/ ( (|N(i) - {i,j}| + |N(j) - {i,j}|)/2 )
    when the divisor is set of 'union', the matching index is computed as:
        MI(i,j) = ( |( N(i) intesect N(j) ) - {i,j}| )/ ( |(N(i) union N(j)) - {i,j}| )

    Note that we have adopted the convension that when N(i) - {i,j} and N(j) - {i,j} are both empty (meaning the divisor and the numerator of the expression are both zero) we set the matching index equal to zero. 
    
    Parameters:
    - adjacency_matrix: torch Tensor of shape [num_nodes, num_nodes] which specifies the adjacency matrix of the network.
    - mode: The type of matching index to compute (e.g., 'all', 'in', 'out'). 
    - divisor: Which division mode to use (e.g., 'union' or 'mean').

    Returns:
    - matching_index: [num_nodes, num_nodes] The matrix of matching indices.
    """
    assert mode in ['all', 'in', 'out'], f"Mode '{mode}' is not supported. Must be one of 'all', 'in', or 'out'."

    num_nodes = adjacency_matrix.shape[0]
    matching_indices = torch.zeros((num_nodes, num_nodes), dtype=adjacency_matrix.dtype)

    if mode == 'out':    
        # Create a modified adjacency matrix where self-connections are removed
        modified_adjacency_matrix = adjacency_matrix.clone() 
        modified_adjacency_matrix.fill_diagonal_(0) 

        # Compute the total connections between nodes, excluding connections between the pair of nodes
        if divisor == "mean":            
            # Compute the (outward) node strengths
            node_strengths = modified_adjacency_matrix.sum(dim=0)
            # Compute the divisor term
            denominator = ( node_strengths.unsqueeze(0) + node_strengths.unsqueeze(1) - modified_adjacency_matrix - modified_adjacency_matrix.T )/2
            # Set the denominator to be 1 whenever it is zero to avoid division by zero
            denominator[denominator == 0] = 1

            # Compute the matching index
            matching_indices = (modified_adjacency_matrix.T @ modified_adjacency_matrix) / denominator
            
        elif divisor == "union":
            # In this case, compute the divisor as the size of the union of the neighbourhoods. 
            denominator =  torch.max( modified_adjacency_matrix.unsqueeze(1), modified_adjacency_matrix.unsqueeze(2) ).sum(dim=0) - modified_adjacency_matrix - modified_adjacency_matrix.T
            # Set the denominator to be 1 whenever it is zero to avoid division by zero
            denominator[denominator == 0] = 1

            # Compute the matching index
            matching_indices = (modified_adjacency_matrix.T @ modified_adjacency_matrix) / denominator
            
        elif divisor == "matlab":
            # Implement the MATLAB matching index computation
            for i in range(num_nodes):
                c1 = modified_adjacency_matrix[i, :].clone()
                c1[i] = 0  # Exclude self-connection
                for j in range(num_nodes):
                    if i == j:
                        matching_indices[i, j] = 0
                        continue
                    c2 = modified_adjacency_matrix[j, :].clone()
                    c2[j] = 0  # Exclude self-connection

                    # Exclude direct connections between i and j
                    c1_excl = c1.clone()
                    c2_excl = c2.clone()
                    c1_excl[j] = 0
                    c2_excl[i] = 0

                    # Compute 'use' mask where either c1_excl or c2_excl has a connection
                    use = (c1_excl > 0) | (c2_excl > 0)

                    # Sum of degrees over 'use' mask
                    ncon = (c1_excl[use] > 0).sum() + (c2_excl[use] > 0).sum()

                    # Number of common neighbors over 'use' mask
                    n_common = ((c1_excl[use] > 0) & (c2_excl[use] > 0)).sum()

                    if ncon == 0:
                        matching_indices[i, j] = 0.00001
                    else:
                        matching_indices[i, j] = 2 * n_common / ncon + 0.00001
        elif divisor == "stuart":
            # Implement the 'stuart' matching index computation
            # Convert to NumPy arrays for convenience
            A = modified_adjacency_matrix.cpu().numpy()
            n = num_nodes

            # Compute the number of common neighbors
            nei = (A @ A) * (~np.eye(n, dtype=bool))

            # Compute degrees
            deg = A.sum(axis=1)

            # Compute sum of degrees for each pair
            degsum = (deg.reshape(-1, 1) + deg.reshape(1, -1)) * (~np.eye(n, dtype=bool))

            # Compute numerator and denominator
            numerator = nei * 2
            denominator = ((degsum <= 2) & (nei != 1)).astype(float) + (degsum - (A * 2))

            # Avoid division by zero
            denominator[denominator == 0] = 0.00001

            # Compute the matching index
            m = numerator / denominator

            # Convert back to torch.Tensor
            matching_indices = torch.from_numpy(m).float()
        else:
            raise ValueError("Divisor must be set to either 'mean', 'union', or 'matlab'!")

        
    elif mode == 'in':
        # In the case that we want the inward matching indices, we simply call the function with the matrix transposed. 
        return matching_index(adjacency_matrix.T, mode='out', divisor=divisor)
    elif mode == 'all':
        raise NotImplementedError("Mode 'all' is not implemented yet!")

    return matching_indices