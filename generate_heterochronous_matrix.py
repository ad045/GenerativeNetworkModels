import numpy as np
import torch

# Function to generate heterochronous matrices
def generate_heterochronous_matrix(
    coord,
    reference_coord=[0,0,0],
    sigma=1.0,
    num_nodes=100,
    mseed=0,
    cumulative=False,
    local=True
):
    """
    Generate heterochronous matrices based on a dynamic starting node.
    
    Parameters:
    - coord (array): Coordinates of nodes.
    - starting_node_index (int): Index of the node to use as the starting point.
    - sigma (float): Standard deviation for the Gaussian.
    - num_nodes (int): Number of time steps/nodes.
    - mseed (int): Number of seed nodes to exclude from computation.
    - cumulative (bool): Whether to apply cumulative maximum.
    - local (bool): Whether to generate matrices locally or globally.
    
    Returns:
    - torch.tensor: Heterochronous matrices tensor.
    """
    
    # Compute Euclidean distances from the reference node to all nodes
    distances = np.sqrt(np.sum((coord - reference_coord) ** 2, axis=1))
    max_distance = np.max(distances)  # Maximum distance for setting Gaussian means

    # Calculate means for Gaussian function at each time step
    means = np.linspace(0, max_distance, num_nodes - mseed)
    heterochronous_matrix = np.zeros((len(distances), num_nodes - mseed))
    P = lambda d, mu: (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((d - mu) ** 2) / (2 * sigma ** 2))

    # Calculate probabilities at each time step
    for t in range(num_nodes - mseed):
        mu = means[t]
        heterochronous_matrix[:, t] = P(distances, mu)
        #heterochronous_matrix[:, -1 - t] = P(distances, mu)

    # Apply cumulative maximum if requested
    if cumulative:
        heterochronous_matrix = np.maximum.accumulate(heterochronous_matrix, axis=1)

    # Convert to matrix form based on local parameter
    heterochronous_matrices = []
    for t in range(num_nodes - mseed):
        Ht = heterochronous_matrix[:, t]
        H_rescaled = (Ht - np.min(Ht)) / (np.max(Ht) - np.min(Ht))
        if local:
            Hmat = np.outer(H_rescaled, H_rescaled)
        else:
            N = len(H_rescaled)
            rows, cols = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
            Hmat = np.maximum(H_rescaled[rows], H_rescaled[cols])
        heterochronous_matrices.append(Hmat)

    heterochronous_matrices_tensor = torch.tensor(np.stack(heterochronous_matrices, axis=-1), dtype=torch.float32)
    return heterochronous_matrices_tensor
