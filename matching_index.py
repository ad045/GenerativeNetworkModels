import torch

from typing import Tuple, List
from jaxtyping import Float, Int, jaxtyped
from typeguard import typechecked

@jaxtyped(typechecker=typechecked)
def matching_index(
    adjacency_matrix:Float[torch.Tensor, "num_nodes num_nodes"],
    mode:str = "all",
    divisor:str = "mean"
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

    if mode == 'in':    
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
            matching_indices.fill_diagonal_(0)
            
        elif divisor == "union":
            # In this case, compute the divisor as the size of the union of the neighbourhoods. 
            denominator =  torch.max( modified_adjacency_matrix.unsqueeze(1), modified_adjacency_matrix.unsqueeze(2) ).sum(dim=0) - modified_adjacency_matrix - modified_adjacency_matrix.T
            # Set the denominator to be 1 whenever it is zero to avoid division by zero
            denominator[denominator == 0] = 1

            # Compute the matching index
            matching_indices = (modified_adjacency_matrix.T @ modified_adjacency_matrix) / denominator
            matching_indices.fill_diagonal_(0)
            
        else: 
            raise ValueError("Divisor must be set to either 'mean' or 'union'!")

    elif mode == 'out':
        # In the case that we want the inward matching indices, we simply call the function with the matrix transposed. 
        return matching_index(adjacency_matrix.T, mode='in', divisor=divisor)
    elif mode == 'all':
        if divisor == "mean":
            # Create a modified adjacency matrix where self-connections are removed
            modified_adjacency_matrix = adjacency_matrix.clone()
            modified_adjacency_matrix.fill_diagonal_(0)
            
            # Combine incoming and outgoing connections
            combined_adjacency_matrix = modified_adjacency_matrix + modified_adjacency_matrix.T
            combined_adjacency_matrix.fill_diagonal_(0)
            
            # Compute the node strengths (degrees)
            node_strengths = combined_adjacency_matrix.sum(dim=1)
            
            # Compute the denominator term
            denominator = (node_strengths.unsqueeze(0) + node_strengths.unsqueeze(1) - 2 * combined_adjacency_matrix) / 2
            denominator[denominator == 0] = 1  # Avoid division by zero
            
            # Compute the matching index
            matching_indices = (combined_adjacency_matrix @ combined_adjacency_matrix) / denominator
            matching_indices.fill_diagonal_(0)
            
        elif divisor == "union":
            # Create a modified adjacency matrix where self-connections are removed
            modified_adjacency_matrix = adjacency_matrix.clone()
            modified_adjacency_matrix.fill_diagonal_(0)
            
            # Combine incoming and outgoing connections
            combined_adjacency_matrix = modified_adjacency_matrix + modified_adjacency_matrix.T
            combined_adjacency_matrix.fill_diagonal_(0)
            
            # In this case, compute the divisor as the size of the union of the neighbourhoods. 
            denominator =  torch.max( combined_adjacency_matrix.unsqueeze(1), combined_adjacency_matrix.unsqueeze(2) ).sum(dim=0) - combined_adjacency_matrix - combined_adjacency_matrix.T
            # Set the denominator to be 1 whenever it is zero to avoid division by zero
            denominator[denominator == 0] = 1

            # Compute the matching index
            matching_indices = (combined_adjacency_matrix.T @ combined_adjacency_matrix) / denominator
            matching_indices.fill_diagonal_(0)
            
        else: 
            raise ValueError("Divisor must be set to either 'mean' or 'union'!")
        
    return matching_indices