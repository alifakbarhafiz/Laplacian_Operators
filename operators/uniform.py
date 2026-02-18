import numpy as np
from scipy.sparse import diags

from geometry.adjacency import build_adjacency_matrix


def build_uniform_laplacian(F: np.ndarray, n_vertices: int):
    """
    Construct the uniform (combinatorial) Laplacian:
        L = D - A

    Parameters
    ----------
    F : (n_faces, 3) int array
        Triangle indices
    n_vertices : int

    Returns
    -------
    L : sparse matrix (n_vertices, n_vertices)
    """
    # Adjacency matrix
    A = build_adjacency_matrix(F, n_vertices)

    # Degree vector
    degrees = np.array(A.sum(axis=1)).flatten()

    # Degree matrix
    D = diags(degrees)

    # Laplacian
    L = D - A

    return L


def apply_laplacian(L, V: np.ndarray):
    """
    Apply Laplacian to vertex positions.

    Returns
    -------
    LV : (n_vertices, 3)
    """
    return L @ V


def laplacian_smoothing(V: np.ndarray, L, step_size=0.01, iterations=10):
    """
    Perform explicit Laplacian smoothing:
        V_new = V - step_size * L V
    """
    V_smooth = V.copy()

    for _ in range(iterations):
        V_smooth = V_smooth - step_size * (L @ V_smooth)

    return V_smooth
