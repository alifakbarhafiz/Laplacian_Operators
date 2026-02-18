import numpy as np
import polyscope as ps

from visualization.camera import frame_scene_mesh


def save_heatmap_screenshot(
    V: np.ndarray,
    F: np.ndarray,
    values: np.ndarray,
    screenshot_path: str,
    mesh_name: str = "mesh",
    quantity_name: str = "mean_curvature",
    normalize: bool = True,
    clip_percentile: float = 99.0,
):
    """
    Open a clean Polyscope scene, show the mesh with a vertex heatmap,
    frame the camera, save a screenshot, and clear. Safe to call per-mesh.
    """
    ps.init()
    ps.remove_all_structures()
    add_vertex_heatmap(
        mesh_name=mesh_name,
        V=V,
        F=F,
        values=values,
        quantity_name=quantity_name,
        normalize=normalize,
        clip_percentile=clip_percentile,
    )
    frame_scene_mesh(V, margin_ratio=0.2, distance_scale=1.7)
    ps.screenshot(str(screenshot_path))
    ps.remove_all_structures()


def add_vertex_heatmap(
    mesh_name: str,
    V: np.ndarray,
    F: np.ndarray,
    values: np.ndarray,
    quantity_name: str = "scalar_field",
    normalize: bool = True,
    clip_percentile: float = 99.0,
):
    """
    Add a vertex-based heatmap to a Polyscope mesh.

    Parameters
    ----------
    mesh_name : str
        Name of the mesh in Polyscope
    V : (n_vertices, 3)
        Vertex positions
    F : (n_faces, 3)
        Faces
    values : (n_vertices,)
        Scalar values defined on vertices
    quantity_name : str
        Name of scalar quantity
    normalize : bool
        Whether to normalize values to [0,1]
    clip_percentile : float
        Percentile clipping to remove extreme outliers
    """

    if normalize:
        # Clip extreme outliers (important for curvature)
        lower = np.percentile(values, 100 - clip_percentile)
        upper = np.percentile(values, clip_percentile)
        values = np.clip(values, lower, upper)

        # Normalize to [0, 1]
        min_val = values.min()
        max_val = values.max()
        if max_val > min_val:
            values = (values - min_val) / (max_val - min_val)

    ps_mesh = ps.register_surface_mesh(mesh_name, V, F)
    ps_mesh.add_scalar_quantity(
        quantity_name, values, defined_on="vertices", enabled=True
    )
