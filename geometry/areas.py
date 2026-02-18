import numpy as np


def compute_triangle_areas(V: np.ndarray, F: np.ndarray):
    """
    Compute area of each triangle.

    Parameters
    ----------
    V : (n_vertices, 3) float array
        Vertex positions
    F : (n_faces, 3) int array
        Triangle indices

    Returns
    -------
    areas : (n_faces,) array
        Area of each triangle
    """
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    cross_prod = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross_prod, axis=1)

    return areas


def compute_vertex_areas(V: np.ndarray, F: np.ndarray):
    """
    Compute per-vertex area using barycentric (1/3) area distribution.

    Each triangle area is equally distributed to its 3 vertices.

    Returns
    -------
    vertex_areas : (n_vertices,) array
    """
    n_vertices = V.shape[0]
    triangle_areas = compute_triangle_areas(V, F)

    vertex_areas = np.zeros(n_vertices)

    for i in range(3):
        np.add.at(vertex_areas, F[:, i], triangle_areas / 3.0)

    return vertex_areas
