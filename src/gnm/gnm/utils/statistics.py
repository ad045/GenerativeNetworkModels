r"""Statistical functions for comparing distributions in generative network models.

This module provides statistical measures for comparing distributions, which are
particularly useful for evaluating the similarity between observed and generated
networks based on their property distributions.
"""

import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked


@jaxtyped(typechecker=typechecked)
def ks_statistic(
    samples_1: Float[torch.Tensor, "batch_1 num_samples_1"],
    samples_2: Float[torch.Tensor, "batch_2 num_samples_2"],
) -> Float[torch.Tensor, "batch_1 batch_2"]:
    r"""Compute Kolmogorov-Smirnov statistics between all pairs of distributions in two batches.

    The Kolmogorov-Smirnov (KS) statistic measures the maximum absolute difference
    between two cumulative distribution functions. This function efficiently computes
    KS statistics for all pairs of distributions between two batches of samples, which
    is useful for comparing multiple generated networks with observed networks.

    Args:
        samples_1:
            First batch of samples with shape [batch_1, num_samples_1]
        samples_2:
            Second batch of samples with shape [batch_2, num_samples_2]

    Returns:
        KS statistics for all pairs with shape [batch_1, batch_2]

    Examples:
        >>> import torch
        >>> from gnm.utils import ks_statistic
        >>> # Create two batches of samples
        >>> samples_1 = torch.randn(3, 100)  # 3 distributions, 100 samples each
        >>> samples_2 = torch.randn(2, 150)  # 2 distributions, 150 samples each
        >>> ks_stats = ks_statistic(samples_1, samples_2)
        >>> ks_stats.shape
        torch.Size([3, 2])
        >>> # Each entry ks_stats[i,j] is the KS statistic between
        >>> # the i-th distribution from batch 1 and j-th distribution from batch 2

    See Also:
        - [`evaluation.KSCriterion`][gnm.evaluation.KSCriterion]: Uses KS statistics to compute discrepancy between networks measure distributions
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
