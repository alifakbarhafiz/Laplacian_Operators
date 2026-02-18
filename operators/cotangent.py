import numpy as np
from scipy.sparse import coo_matrix, diags
from geometry.adjacency import build_vertex_adjacency_list
from geometry.areas import compute_vertex_areas


# Clamp cotangent to avoid explosion for near-degenerate triangles
COTAN_CLAMP = 1e3


def cotangent(a, b):
    """Compute cotangent of angle between vectors a and b. Clamped for stability."""
    cos_angle = np.dot(a, b)
    sin_angle = np.linalg.norm(np.cross(a, b))
    cot = cos_angle / (sin_angle + 1e-16)
    return np.clip(cot, -COTAN_CLAMP, COTAN_CLAMP)


def build_cotangent_laplacian(V: np.ndarray, F: np.ndarray, normalized=True):
    """
    Construct the cotangent Laplacian.

    Parameters
    ----------
    V : (n_vertices, 3) float array
    F : (n_faces, 3) int array
    normalized : bool
        If True, divide by vertex areas (mass lumping)

    Returns
    -------
    L : sparse matrix (n_vertices, n_vertices)
    """
    n_vertices = len(V)

    I, J, W = [], [], []

    for face in F:
        i, j, k = face
        vi, vj, vk = V[i], V[j], V[k]

        # Edge vectors
        e0 = vj - vk
        e1 = vk - vi
        e2 = vi - vj

        # Cotangents of angles at vertices i, j, k
        cot_i = cotangent(vj - vi, vk - vi)
        cot_j = cotangent(vi - vj, vk - vj)
        cot_k = cotangent(vi - vk, vj - vk)

        # Off-diagonal weights (symmetric)
        I.extend([i, j, j, k, k, i])
        J.extend([j, i, k, j, i, k])
        W.extend([cot_k, cot_k, cot_i, cot_i, cot_j, cot_j])

    W = np.array(W) * 0.5  # cotangent Laplacian factor
    L = coo_matrix((W, (I, J)), shape=(n_vertices, n_vertices))

    # Diagonal
    diag = np.array(L.sum(axis=1)).flatten()
    L = -L
    L.setdiag(diag)

    if normalized:
        vertex_areas = compute_vertex_areas(V, F)
        # Avoid division by zero or huge weights from tiny/degenerate areas
        area_floor = max(1e-12, np.median(vertex_areas) * 1e-6)
        vertex_areas = np.maximum(vertex_areas, area_floor)
        L = diags(1.0 / vertex_areas) @ L

    return L


def apply_laplacian(L, V):
    """Apply Laplacian to vertex positions."""
    return L @ V


def laplacian_smoothing(V: np.ndarray, L, step_size=0.01, iterations=10):
    """Explicit Laplacian smoothing: V_new = V - step_size * L V"""
    V_smooth = V.copy()
    for _ in range(iterations):
        V_smooth -= step_size * (L @ V_smooth)
    return V_smooth
