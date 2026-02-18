import numpy as np
from scipy.sparse import diags, identity
from geometry.adjacency import build_adjacency_matrix


def build_graph_laplacian(F: np.ndarray, n_vertices: int, normalized=True):
    """
    Build a graph Laplacian for a mesh.

    Parameters
    ----------
    F : (n_faces, 3) int array
        Triangle indices
    n_vertices : int
    normalized : bool
        Whether to compute symmetric normalized Laplacian

    Returns
    -------
    L : sparse matrix (n_vertices, n_vertices)
    """
    # Adjacency matrix
    A = build_adjacency_matrix(F, n_vertices)

    # Degree vector
    degrees = np.array(A.sum(axis=1)).flatten()

    if normalized:
        # Avoid division by zero
        degrees[degrees == 0] = 1
        D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
        L = identity(n_vertices) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        # Combinatorial Laplacian
        D = diags(degrees)
        L = D - A

    return L


def laplacian_smoothing(V: np.ndarray, L, step_size=0.01, iterations=10):
    """
    Perform explicit Laplacian smoothing using a graph Laplacian.

    Parameters
    ----------
    V : (n_vertices, 3) array
        Vertex positions
    L : sparse matrix
        Laplacian operator
    step_size : float
        Smoothing step size
    iterations : int
        Number of smoothing iterations

    Returns
    -------
    V_smooth : (n_vertices, 3)
        Smoothed vertex positions
    """
    V_smooth = V.copy()
    for _ in range(iterations):
        V_smooth = V_smooth - step_size * (L @ V_smooth)
    return V_smooth
