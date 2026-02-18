import polyscope as ps
import numpy as np
import imageio
from pathlib import Path

from visualization.camera import frame_scene_mesh


def create_original_rotation_gif(
    V: np.ndarray,
    F: np.ndarray,
    gif_path: str = "original_rotation.gif",
    n_frames: int = 36,
    axis: np.ndarray = np.array([0, 1, 0]),
):
    """
    Create a GIF of the mesh rotating 360° (no smoothing). Uses a consistent
    camera: mesh is centered, then camera is set to home view so the object
    is well-framed. Clears the scene so this can be called per-mesh.
    """
    # Center mesh so rotation is in place and camera framing is consistent
    center = V.mean(axis=0)
    V_centered = np.asarray(V, dtype=np.float64) - center

    ps.init()
    ps.remove_all_structures()

    mesh_ps = ps.register_surface_mesh("mesh", V_centered.copy(), F)
    frame_scene_mesh(V_centered, margin_ratio=0.2, distance_scale=1.7)

    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        axis = np.array([0.0, 1.0, 0.0])
    else:
        axis = np.asarray(axis, dtype=np.float64) / axis_norm
    angle_step = 2 * np.pi / n_frames
    images = []

    for i in range(n_frames):
        angle = i * angle_step
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        V_rot = V_centered @ R.T
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


def create_mesh_rotation_gif(
    mesh_name: str,
    V: np.ndarray,
    F: np.ndarray,
    gif_path: str = "mesh_rotation.gif",
    n_frames: int = 36,
    axis: np.ndarray = np.array([0, 1, 0]),
):
    ps.init()

    images = []

    axis = axis / np.linalg.norm(axis)  # normalize
    angle_step = 2 * np.pi / n_frames

    for i in range(n_frames):
        angle = i * angle_step
        # Rotation matrix using Rodrigues' formula
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        V_rot = V @ R.T  # rotate vertices

        ps_mesh = ps.register_surface_mesh(mesh_name, V_rot, F)
        frame = ps.screenshot_to_buffer()
        if frame is not None:
            images.append(frame)
        ps.remove_all_structures()  # clear for next frame

    if images:
        imageio.mimsave(gif_path, images, duration=0.05)
        print(f"GIF saved to {gif_path}")
    else:
        print(f"[WARN] No frames captured; skipping GIF {gif_path}")
