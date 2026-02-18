"""
Shared camera framing so all visualizations show the scene properly.
Uses explicit look_at(eye, target) instead of relying on Polyscope's home view.
"""
import numpy as np
import polyscope as ps


def _bbox_from_vertices(V: np.ndarray, margin_ratio: float = 0.15) -> tuple:
    """Compute (low, high) bounding box from vertex array with proportional margin."""
    V = np.asarray(V, dtype=np.float64)
    low = np.min(V, axis=0)
    high = np.max(V, axis=0)
    size = np.max(high - low)
    margin = max(size * margin_ratio, 1e-6)
    return low - margin, high + margin


def frame_scene_bbox(
    low: np.ndarray,
    high: np.ndarray,
    distance_scale: float = 2.0,
    look_from_negative_z: bool = False,
):
    """
    Set the camera to frame the given bounding box. Call after structures are registered.
    Uses look_at(eye, target) so the whole box is in view.
    If look_from_negative_z is True, camera is placed at -Z (e.g. to show mesh "front").
    """
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    center = (low + high) / 2.0
    size = np.max(high - low)
    distance = max(size * distance_scale, 0.1)
    sign = -1.0 if look_from_negative_z else 1.0
    eye = center + np.array([0.0, 0.0, sign * distance], dtype=np.float64)
    ps.look_at(eye, center)


def frame_scene_mesh(V: np.ndarray, margin_ratio: float = 0.15, distance_scale: float = 2.0):
    """
    Set the camera to frame a mesh from its vertices. Call after the mesh is registered.
    """
    low, high = _bbox_from_vertices(V, margin_ratio=margin_ratio)
    frame_scene_bbox(low, high, distance_scale=distance_scale)
