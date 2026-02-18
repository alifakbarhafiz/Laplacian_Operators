import numpy as np


def mesh_volume(V: np.ndarray, F: np.ndarray) -> float:
    """
    Compute volume of a closed mesh using the divergence theorem:
        V = (1/6) * sum over faces of (v0 . (v1 x v2))

    Parameters
    ----------
    V : (n_vertices, 3) array of vertex positions
    F : (n_faces, 3) array of triangle indices

    Returns
    -------
    volume : float
    """
    volume = 0.0
    for f in F:
        v0, v1, v2 = V[f[0]], V[f[1]], V[f[2]]
        volume += np.dot(v0, np.cross(v1, v2))
    return abs(volume) / 6.0
