import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked


@jaxtyped(typechecker=typechecked)
def ks_statistic(
    samples_1: Float[torch.Tensor, "batch_1 num_samples_1"],
    samples_2: Float[torch.Tensor, "batch_2 num_samples_2"],
) -> Float[torch.Tensor, "batch_1 batch_2"]:
    """
    Compute KS statistics between all pairs of distributions in two batches.

    Args:
        samples_1: First batch of samples
        samples_2: Second batch of samples

    Returns:
        KS statistics for all pairs
    """
    # Sort samples for CDF computation
    sorted_1, _ = torch.sort(samples_1, dim=1)  # [batch_1, n_samples_1]
    sorted_2, _ = torch.sort(samples_2, dim=1)  # [batch_2, n_samples_2]

    # Get all unique values that could be CDF evaluation points
    # Combine all samples and get unique sorted values
    all_values = torch.unique(
        torch.cat([sorted_1.reshape(-1), sorted_2.reshape(-1)])
    )  # [n_unique]

    # Compute CDFs for all distributions at these points
    # For each batch, count fraction of samples less than each value
    cdf_1 = (
        (sorted_1.unsqueeze(-1) <= all_values.unsqueeze(0).unsqueeze(0))
        .float()
        .mean(dim=1)
    )

    cdf_2 = (
        (sorted_2.unsqueeze(-1) <= all_values.unsqueeze(0).unsqueeze(0))
        .float()
        .mean(dim=1)
    )

    # Compute absolute differences between all pairs of CDFs
    # Use broadcasting to compute differences between all pairs in the batches
    differences = torch.abs(
        cdf_1.unsqueeze(1) - cdf_2.unsqueeze(0)
    )  # [batch_1, batch_2, n_unique]

    # Get maximum difference for each pair
    ks_statistics = torch.max(differences, dim=2).values  # [batch_1, batch_2]

    return ks_statistics
