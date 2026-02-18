import numpy as np
from geometry.boundaries import find_boundary_vertices, find_boundary_edges


def boundary_stats(F: np.ndarray) -> dict:
    """
    Compute boundary metrics for a mesh.

    Parameters
    ----------
    F : (n_faces, 3) array of triangle indices

    Returns
    -------
    stats : dict
        - n_boundary_edges
        - n_boundary_vertices
    """
    edges = find_boundary_edges(F)
    vertices = find_boundary_vertices(F)
    stats = {"n_boundary_edges": len(edges), "n_boundary_vertices": len(vertices)}
    return stats
