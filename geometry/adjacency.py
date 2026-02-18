import numpy as np
from scipy.sparse import coo_matrix


def validate_face_indices(F: np.ndarray, n_vertices: int) -> None:
    """
    Raise ValueError if any face index is out of range [0, n_vertices - 1].
    Helps catch bad meshes early with a clear error.
    """
    if F.size == 0:
        return
    min_idx = int(np.min(F))
    max_idx = int(np.max(F))
    if min_idx < 0 or max_idx >= n_vertices:
        raise ValueError(
            f"Face indices must be in [0, n_vertices-1]. "
            f"Got min={min_idx}, max={max_idx}, n_vertices={n_vertices}."
        )


def build_vertex_adjacency_list(F: np.ndarray, n_vertices: int, validate: bool = True):
    """
    Build vertex adjacency as a list of sets.

    Parameters
    ----------
    F : (n_faces, 3) int array
        Triangle indices
    n_vertices : int
        Number of vertices
    validate : bool
        If True (default), check that all face indices are in [0, n_vertices-1].

    Returns
    -------
    adjacency : list of sets
        adjacency[i] = set of vertex indices adjacent to i
    """
    if validate:
        validate_face_indices(F, n_vertices)
    adjacency = [set() for _ in range(n_vertices)]

    for tri in F:
        i, j, k = tri

        adjacency[i].update([j, k])
        adjacency[j].update([i, k])
        adjacency[k].update([i, j])

    return adjacency


def build_adjacency_matrix(F: np.ndarray, n_vertices: int, validate: bool = True):
    """
    Build symmetric vertex adjacency matrix (unweighted).

    Parameters
    ----------
    F : (n_faces, 3) int array
        Triangle indices
    n_vertices : int
        Number of vertices
    validate : bool
        If True (default), check that all face indices are in [0, n_vertices-1].

    Returns
    -------
    A : (n_vertices, n_vertices) sparse matrix
        A[i, j] = 1 if edge exists
    """
    if validate:
        validate_face_indices(F, n_vertices)
    rows = []
    cols = []

    for tri in F:
        i, j, k = tri

        edges = [(i, j), (j, i), (i, k), (k, i), (j, k), (k, j)]

        for r, c in edges:
            rows.append(r)
            cols.append(c)

    data = np.ones(len(rows))
    A = coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))

    # Remove duplicates by converting to CSR and back
    A = A.tocsr()
    A.data[:] = 1.0

    return A


def compute_vertex_degrees(adjacency_list):
    """
    Compute degree of each vertex.

    Returns
    -------
    degrees : (n_vertices,) array
    """
    return np.array([len(neigh) for neigh in adjacency_list])
