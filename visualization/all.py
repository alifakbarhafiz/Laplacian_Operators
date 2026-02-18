import polyscope as ps
import numpy as np
import imageio
from geometry.curvature import compute_mean_curvature
from operators.uniform import build_uniform_laplacian, laplacian_smoothing
from pathlib import Path

from visualization.camera import frame_scene_mesh


def create_smooth_curvature_gif(
    V: np.ndarray,
    F: np.ndarray,
    gif_path: str = "mesh_smooth_curvature.gif",
    n_frames: int = 36,
    step_size: float = 0.01,
    smooth_iterations: int = 50,
    rotation_axis: np.ndarray = np.array([0, 1, 0]),
):
    """Create a GIF of a Laplacian-smoothed mesh colored by mean curvature, rotating 360°."""

    # Step 0: Smooth the mesh
    L = build_uniform_laplacian(F, len(V))
    V_smooth = laplacian_smoothing(
        V, L, step_size=step_size, iterations=smooth_iterations
    )

    # Center mesh at origin so rotation is in place and camera can frame consistently
    center = V_smooth.mean(axis=0)
    V_smooth = V_smooth - center

    # Step 1: Mean curvature
    curvature, _ = compute_mean_curvature(V_smooth, F)
    c_norm = (curvature - curvature.min()) / (
        curvature.max() - curvature.min() + 1e-12
    )
    # Polyscope expects Nx3 RGB; use grayscale from scalar
    curvature_colors = np.column_stack([c_norm, c_norm, c_norm])

    # Step 2: Initialize Polyscope and register mesh
    ps.init()
    mesh_ps = ps.register_surface_mesh("smoothed_mesh", V_smooth.copy(), F)
    mesh_ps.add_color_quantity(
        "mean_curvature", curvature_colors, defined_on="vertices", enabled=True
    )
    frame_scene_mesh(V_smooth, margin_ratio=0.2, distance_scale=1.7)

    # Step 3: Rotation setup (guard against zero axis)
    axis_norm = np.linalg.norm(rotation_axis)
    if axis_norm < 1e-10:
        rotation_axis = np.array([0.0, 1.0, 0.0])
    else:
        rotation_axis = rotation_axis / axis_norm
    angle_step = 2 * np.pi / n_frames
    images = []

    for i in range(n_frames):
        angle = i * angle_step

        # Rodrigues rotation
        K = np.array(
            [
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0],
            ]
        )
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        V_rot = V_smooth @ R.T

        mesh_ps.update_vertex_positions(V_rot)
        frame = ps.screenshot_to_buffer()
        if frame is not None:
            # Copy so we don't keep a reference to a reused buffer (avoids black frames)
            images.append(np.asarray(frame).copy())

    # Step 4: Save GIF (screenshot_to_buffer returns RGBA; imageio accepts it)
    if images:
        imageio.mimsave(gif_path, images, duration=0.05)
        print(f"GIF saved to {gif_path}")
    else:
        print(f"[WARN] No frames captured; skipping GIF {gif_path}")


def create_smoothing_transition_gif(
    V: np.ndarray,
    F: np.ndarray,
    gif_path: str = "smoothing_transition.gif",
    n_frames: int = 72,
    step_size: float = 0.01,
    smooth_iterations: int = 50,
    rotation_axis: np.ndarray = np.array([0, 1, 0]),
):
    """
    Create a single GIF that shows the full process: the mesh morphs from
    original (start) to fully smoothed (end) while rotating. So you see
    the transformation from beginning to end over time.
    """
    # Compute fully smoothed mesh
    L = build_uniform_laplacian(F, len(V))
    V_smooth = laplacian_smoothing(
        V, L, step_size=step_size, iterations=smooth_iterations
    )
    # Center both for consistent framing and in-place rotation
    center_orig = V.mean(axis=0)
    V_orig_c = np.asarray(V, dtype=np.float64) - center_orig
    V_smooth_c = np.asarray(V_smooth, dtype=np.float64) - V_smooth.mean(axis=0)

    ps.init()
    ps.remove_all_structures()
    mesh_ps = ps.register_surface_mesh("mesh", V_orig_c.copy(), F)
    mesh_ps.set_color((0.2, 1.0, 0.6))
    V_union = np.vstack([V_orig_c, V_smooth_c])
    frame_scene_mesh(V_union, margin_ratio=0.2, distance_scale=1.7)

    axis = np.asarray(rotation_axis, dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        axis = np.array([0.0, 1.0, 0.0])
    else:
        axis = axis / axis_norm

    images = []
    for i in range(n_frames):
        # t: 0 -> 1 over the GIF (morph from original to smooth)
        t = i / max(n_frames - 1, 1)
        # One full rotation over the GIF
        angle = i * (2 * np.pi / n_frames)
        V_blend = (1 - t) * V_orig_c + t * V_smooth_c
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        V_rot = V_blend @ R.T
        mesh_ps.update_vertex_positions(V_rot)
        frame = ps.screenshot_to_buffer()
        if frame is not None:
            images.append(np.asarray(frame).copy())

    ps.remove_all_structures()
    if images:
        imageio.mimsave(gif_path, images, duration=0.05)
        print(f"GIF saved to {gif_path}")
    else:
        print(f"[WARN] No frames captured; skipping GIF {gif_path}")


def _center_and_normalize(V: np.ndarray, box_size: float = 0.5) -> np.ndarray:
    """Center and scale so mesh fits in a box of side box_size. Avoids huge coords (e.g. cotangent blow-up)."""
    V = np.asarray(V, dtype=np.float64)
    V_c = V - V.mean(axis=0)
    r = np.abs(V_c).max()
    if r < 1e-12:
        return V_c
    scale = (box_size / 2.0) / r
    return V_c * scale


def create_morph_transition_gif(
    V_orig: np.ndarray,
    V_smooth: np.ndarray,
    F: np.ndarray,
    gif_path: str,
    n_frames: int = 72,
    rotation_axis: np.ndarray = np.array([0, 1, 0]),
):
    """
    Morph from V_orig to V_smooth while rotating. Both meshes are normalized to the same
    scale so the view stays valid even when V_smooth has huge values (e.g. cotangent).
    """
    V_orig_c = _center_and_normalize(V_orig, box_size=0.5)
    V_smooth_c = _center_and_normalize(V_smooth, box_size=0.5)

    ps.init()
    ps.remove_all_structures()
    mesh_ps = ps.register_surface_mesh("mesh", V_orig_c.copy(), F)
    mesh_ps.set_color((0.2, 1.0, 0.6))
    V_union = np.vstack([V_orig_c, V_smooth_c])
    frame_scene_mesh(V_union, margin_ratio=0.2, distance_scale=1.7)

    axis = np.asarray(rotation_axis, dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        axis = np.array([0.0, 1.0, 0.0])
    else:
        axis = axis / axis_norm

    images = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        angle = i * (2 * np.pi / n_frames)
        V_blend = (1 - t) * V_orig_c + t * V_smooth_c
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        V_rot = V_blend @ R.T
        mesh_ps.update_vertex_positions(V_rot)
        frame = ps.screenshot_to_buffer()
        if frame is not None:
            images.append(np.asarray(frame).copy())

    ps.remove_all_structures()
    if images:
        imageio.mimsave(gif_path, images, duration=0.05)
        print(f"GIF saved to {gif_path}")
    else:
        print(f"[WARN] No frames captured; skipping GIF {gif_path}")
