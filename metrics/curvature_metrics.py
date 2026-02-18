import numpy as np


def mean_curvature_stats(H: np.ndarray) -> dict:
    """
    Compute basic statistics for mean curvature.

    Parameters
    ----------
    H : (n_vertices,) array of mean curvature values

    Returns
    -------
    stats : dict
    """
    stats = {
        "min": float(np.min(H)),
        "max": float(np.max(H)),
        "mean": float(np.mean(H)),
    }
    return stats
