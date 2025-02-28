from scipy.stats import ks_2samp
from gnm.utils.statistics import ks_statistic
import numpy as np
import torch
import math

def compare_ks_results(sample_1, sample_2):

    gnm_sample_1 = torch.Tensor(sample_1).unsqueeze(0)
    gnm_sample_2 = torch.Tensor(sample_2).unsqueeze(0)
    gnm_ks = ks_statistic(gnm_sample_1, gnm_sample_2).cpu().numpy()[0][0]
    scipy_ks, _ = ks_2samp(sample_1, sample_2)

    assert math.isclose(gnm_ks, scipy_ks, rel_tol=1e-6), f'KS Statistic Comparison Failed. GNM: {gnm_ks} | Scipy: {scipy_ks}'


connectome_1 = np.random.normal(loc=5, scale=2, size=400)
connectome_2 = np.random.normal(loc=5 + (np.random.random() / 2), scale=2, size=400)
connectome_3 = np.random.normal(loc=15, scale=2, size=400)

compare_ks_results(connectome_1, connectome_2)
compare_ks_results(connectome_1, connectome_3)
