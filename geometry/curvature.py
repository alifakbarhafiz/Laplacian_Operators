import numpy as np
from geometry.areas import compute_vertex_areas

# Clamp cotangent to avoid explosion for near-degenerate triangles
_COTAN_CLAMP = 1e3


def _cotangent(a, b):
    """Compute cotangent of angle between vectors a and b. Clamped for stability."""
    cos_angle = np.dot(a, b)
    sin_angle = np.linalg.norm(np.cross(a, b))
    cot = cos_angle / (sin_angle + 1e-16)
    return np.clip(cot, -_COTAN_CLAMP, _COTAN_CLAMP)


# Public alias for backward compatibility
cotangent = _cotangent


def compute_mean_curvature(V: np.ndarray, F: np.ndarray):
    """
    Compute discrete mean curvature per vertex using cotangent formula.

    Parameters
    ----------
    V : (n_vertices, 3)
    F : (n_faces, 3)

    Returns
    -------
    H : (n_vertices,) mean curvature magnitude per vertex
    H_vec : (n_vertices, 3) mean curvature vectors
    """
    n_vertices = len(V)
    H_vec = np.zeros_like(V)
    vertex_areas = compute_vertex_areas(V, F)

    # Iterate over faces for cotangent weights
    for face in F:
        i, j, k = face
        vi, vj, vk = V[i], V[j], V[k]

        # Edges
        e0 = vj - vi
        e1 = vk - vi
        e2 = vk - vj

        # Cotangents opposite each vertex
        cot_alpha = _cotangent(e0, e1)  # at i
        cot_beta = _cotangent(-e0, e2)  # at j
        cot_gamma = _cotangent(-e1, -e2)  # at k

        # Accumulate Laplacian-like contribution
        H_vec[i] += (cot_beta + cot_gamma) * (vi - vj) / 2
        H_vec[j] += (cot_gamma + cot_alpha) * (vj - vk) / 2
        H_vec[k] += (cot_alpha + cot_beta) * (vk - vi) / 2

    # Normalize by vertex areas (guard against zero for isolated vertices)
    vertex_areas = np.maximum(vertex_areas, 1e-12)
    H_vec /= vertex_areas[:, np.newaxis]
    H = 0.5 * np.linalg.norm(H_vec, axis=1)

    return H, H_vec
