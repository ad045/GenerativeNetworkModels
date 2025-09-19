import torch
import numpy as np


def np_to_tensor(np_adj_matrix: np.array, device=None) -> torch.Tensor:
    """
    Converts a NumPy adjacency matrix to a PyTorch tensor with batch dimensionality if it didn't exist before.

    Args:
        np_adj_matrix (np.array): The input NumPy adjacency matrix. Can be either 2D or 3D.
        device (torch.device, optional): The device on which to create the tensor. Defaults to 'cuda:0' if available, otherwise 'cpu'.

    Returns:
        torch.Tensor: The converted Pydef np_to_tensor(np_adj_matrix: np.array, device=None) -> torch.Tensor:
    """

    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    is_batched = len(np_adj_matrix.shape) == 3
    
    if is_batched:
        np_adj_matrix = np.expand_dims(np_adj_matrix, 0)
    
    torch_adj_matrix = torch.Tensor(np_adj_matrix, device=device)
    return torch_adj_matrix