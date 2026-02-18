"""
Experiment: side-by-side smoothing evolution GIF.

Shows 4 meshes (Original | Uniform | Graph | Cotangent) and animates how
the three operators smooth the mesh over time from start to finish.
"""
import sys
from pathlib import Path
import numpy as np
import polyscope as ps
import trimesh
import imageio

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from operators.uniform import build_uniform_laplacian
from operators.graph import build_graph_laplacian
from operators.cotangent import build_cotangent_laplacian
from visualization.camera import frame_scene_bbox

# ----------------------------
# Config
# ----------------------------
MESH_DIR = ROOT_DIR / "data" / "meshes"
FIGURE_DIR = ROOT_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

STEP_SIZE = 0.01
STEP_SIZE_COTANGENT = 1e-4
SMOOTH_ITERATIONS = 50
GIF_FRAMES = 40
SPACING = 0.75
BOX_SIZE = 0.5

SIDEBYSIDE_COLORS = {
    "Original": (0.35, 0.75, 1.0),
    "Uniform": (0.2, 1.0, 0.45),
    "Graph": (1.0, 0.55, 0.2),
    "Cotangent": (0.75, 0.25, 1.0),
}


def load_mesh(mesh_path: Path):
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    mesh = trimesh.load(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(mesh)}")
    V = np.array(mesh.vertices, dtype=np.float64)
    F = np.array(mesh.faces, dtype=np.int64)
    if F.ndim > 2:
        F = F.squeeze()
    if F.shape[1] != 3:
        raise ValueError(f"Faces must be (n,3), got {F.shape}")
    return V, F


def _center_and_scale_to_fit(V: np.ndarray, box_size: float = 0.5) -> np.ndarray:
    V = np.asarray(V, dtype=np.float64)
    center = V.mean(axis=0)
    V_c = V - center
    r = np.abs(V_c).max()
    if r < 1e-12:
        return V_c
    scale = (box_size / 2.0) / r
    return V_c * scale


def _smoothing_trajectory(V, L, step_size, max_iter):
    """Return list of vertex arrays: [V at iter 0, 1, ..., max_iter]."""
    traj = [V.copy()]
    V_cur = V.copy()
    for _ in range(max_iter):
        V_cur = V_cur - step_size * (L @ V_cur)
        if np.isfinite(V_cur).all() and np.abs(V_cur).max() < 1e8:
            traj.append(V_cur.copy())
        else:
            traj.append(traj[-1].copy())
    return traj


def run_evolution_gif(mesh_file: Path, gif_path: Path):
    V, F = load_mesh(mesh_file)
    mesh_name = mesh_file.stem
    n_frames = GIF_FRAMES
    max_iter = SMOOTH_ITERATIONS

    # Build Laplacians and trajectories (original = static)
    L_uniform = build_uniform_laplacian(F, len(V))
    L_graph = build_graph_laplacian(F, len(V))
    L_cotan = build_cotangent_laplacian(V, F)

    traj_uniform = _smoothing_trajectory(V, L_uniform, STEP_SIZE, max_iter)
    traj_graph = _smoothing_trajectory(V, L_graph, STEP_SIZE, max_iter)
    traj_cotangent = _smoothing_trajectory(V, L_cotan, STEP_SIZE_COTANGENT, max_iter)

    layout_labels = ["Original", "Uniform", "Graph", "Cotangent"]
    n = len(layout_labels)
    offsets = np.linspace(-SPACING * (n - 1) / 2, SPACING * (n - 1) / 2, n)
    half = BOX_SIZE / 2.0
    x_min = offsets[0] - half
    x_max = offsets[-1] + half
    margin = 0.25
    low = np.array([x_min - margin, -half - margin, -half - margin], dtype=np.float64)
    high = np.array([x_max + margin, half + margin, half + margin], dtype=np.float64)
    look_from_negative_z = "armadillo" in mesh_name.lower()

    ps.init()
    ps.remove_all_structures()

    # Pre-place four mesh structures; we'll update positions each frame
    meshes_ps = []
    for i, label in enumerate(layout_labels):
        # Initial positions (frame 0)
        if label == "Original":
            V_placed = _center_and_scale_to_fit(V, box_size=BOX_SIZE)
        else:
            idx = 0
            if label == "Uniform":
                V_m = traj_uniform[idx]
            elif label == "Graph":
                V_m = traj_graph[idx]
            else:
                V_m = traj_cotangent[idx]
            V_placed = _center_and_scale_to_fit(V_m, box_size=BOX_SIZE)
        V_placed[:, 0] += offsets[i]
        name = label.lower().replace(" ", "_")
        color = SIDEBYSIDE_COLORS.get(label, (0.5, 0.5, 0.5))
        m = ps.register_surface_mesh(name, V_placed, F, color=color)
        meshes_ps.append((label, m))

    frame_scene_bbox(
        low, high, distance_scale=0.65, look_from_negative_z=look_from_negative_z
    )

    images = []
    for f in range(n_frames):
        iter_idx = round(f * max_iter / max(n_frames - 1, 1))
        iter_idx = min(iter_idx, max_iter)

        for i, (label, m_ps) in enumerate(meshes_ps):
            if label == "Original":
                V_m = V
            elif label == "Uniform":
                V_m = traj_uniform[iter_idx]
            elif label == "Graph":
                V_m = traj_graph[iter_idx]
            else:
                V_m = traj_cotangent[iter_idx]
            V_placed = _center_and_scale_to_fit(V_m, box_size=BOX_SIZE)
            V_placed[:, 0] += offsets[i]
            m_ps.update_vertex_positions(V_placed)

        frame = ps.screenshot_to_buffer()
        if frame is not None:
            images.append(np.asarray(frame).copy())

    ps.remove_all_structures()

    if images:
        imageio.mimsave(str(gif_path), images, duration=0.08)
        print(f"[INFO] Saved smooth evolution GIF: {gif_path}")
    else:
        print(f"[WARN] No frames captured for {gif_path}")


if __name__ == "__main__":
    mesh_files = list(MESH_DIR.glob("*.obj"))
    if not mesh_files:
        print("[WARN] No mesh files in data/meshes/")
        sys.exit(1)

    for mesh_file in mesh_files:
        try:
            gif_path = FIGURE_DIR / f"{mesh_file.stem}_smooth_evolution_sidebyside.gif"
            run_evolution_gif(mesh_file, gif_path)
        except Exception as e:
            print(f"[ERROR] {mesh_file.name}: {e}")
