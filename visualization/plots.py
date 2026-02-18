# visualization/plots.py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_vertex_values(
    V: np.ndarray,
    values: np.ndarray,
    title="Vertex Values",
    cmap="viridis",
    save_path=None,
):
    """
    Plot per-vertex scalar values on a 3D mesh.

    Parameters
    ----------
    V : (n_vertices, 3) array
        Vertex coordinates
    values : (n_vertices,) array
        Scalar values to color vertices
    title : str
        Plot title
    cmap : str
        Colormap
    save_path : str or Path, optional
        If set, save figure to this path and close (no interactive show).
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(V[:, 0], V[:, 1], V[:, 2], c=values, cmap=cmap)
    plt.colorbar(sc, ax=ax, shrink=0.5, label="Value")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_histogram(
    values: np.ndarray,
    bins=30,
    title="Histogram",
    xlabel="Value",
    ylabel="Frequency",
    save_path=None,
):
    """
    Plot a histogram of values.

    Parameters
    ----------
    values : array
        Data to plot
    bins : int
        Number of bins
    save_path : str or Path, optional
        If set, save figure to this path and close (no interactive show).
    """
    fig = plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins, color="skyblue", edgecolor="k")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
