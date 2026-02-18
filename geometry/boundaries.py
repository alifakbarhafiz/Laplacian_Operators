import numpy as np
from collections import defaultdict


def extract_edges(F: np.ndarray):
    """
    Extract all undirected edges from triangle faces.

    Returns
    -------
    edges : (n_edges, 2) array
        Each row is a sorted edge (i, j) with i < j
    """
    edges = []

    for tri in F:
        i, j, k = tri

        edges.append(tuple(sorted((i, j))))
        edges.append(tuple(sorted((j, k))))
        edges.append(tuple(sorted((k, i))))

    return np.array(edges)


def count_edge_occurrences(edges: np.ndarray):
    """
    Count how many times each edge appears.
    """
    edge_count = defaultdict(int)

    for edge in edges:
        edge_count[tuple(edge)] += 1

    return edge_count


def find_boundary_edges(F: np.ndarray):
    """
    Return list of boundary edges.
    """
    edges = extract_edges(F)
    edge_count = count_edge_occurrences(edges)

    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    return boundary_edges


def find_boundary_vertices(F: np.ndarray):
    """
    Return set of boundary vertices.
    """
    boundary_edges = find_boundary_edges(F)

    boundary_vertices = set()
    for i, j in boundary_edges:
        boundary_vertices.add(i)
        boundary_vertices.add(j)

    return boundary_vertices
