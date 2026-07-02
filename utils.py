import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def batch_ssim(a, b):
    if hasattr(b, "numpy"):
        b = b.numpy()
    scores = [
        structural_similarity(a[i, ..., 0], b[i, ..., 0], data_range=1)
        for i in range(len(a))
    ]
    return float(np.mean(scores))


def batch_ssim_full(a, b):
    """Returns (mean_ssim, list_of_similarity_maps)."""
    if hasattr(b, "numpy"):
        b = b.numpy()
    scores, maps = [], []
    for i in range(len(a)):
        score, smap = structural_similarity(
            a[i, ..., 0], b[i, ..., 0], data_range=1, full=True
        )
        scores.append(score)
        maps.append(smap)
    return float(np.mean(scores)), maps


def batch_psnr(a, b):
    if hasattr(b, "numpy"):
        b = b.numpy()
    scores = [
        peak_signal_noise_ratio(a[i, ..., 0], b[i, ..., 0], data_range=1)
        for i in range(len(a))
    ]
    return float(np.mean(scores))
