import sys
from pathlib import Path
import numpy as np
import polyscope as ps
import trimesh
from tqdm import tqdm

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Import modules
from geometry.curvature import compute_mean_curvature
from geometry.boundaries import find_boundary_vertices, find_boundary_edges
from metrics.volume import mesh_volume
from metrics.curvature_metrics import mean_curvature_stats
from metrics.boundary_metrics import boundary_stats
from operators.uniform import build_uniform_laplacian, laplacian_smoothing
from operators.graph import build_graph_laplacian
from operators.cotangent import build_cotangent_laplacian
from visualization.all import (
    create_smooth_curvature_gif,
    create_morph_transition_gif,
)
from visualization.gifs import create_original_rotation_gif
from visualization.heatmaps import save_heatmap_screenshot
from visualization.plots import plot_histogram, plot_vertex_values
from visualization.camera import frame_scene_bbox

# ----------------------------
# Configuration
# ----------------------------
MESH_DIR = ROOT_DIR / "data" / "meshes"
FIGURE_DIR = ROOT_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

STEP_SIZE = 0.01
STEP_SIZE_COTANGENT = 1e-4  # smaller step: cotangent L can have large weights
SMOOTH_ITERATIONS = 50
GIF_FRAMES = 36


# ----------------------------
# Utility functions
# ----------------------------
def load_mesh(mesh_path: Path):
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    mesh = trimesh.load(mesh_path, process=False)

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a Trimesh object, got {type(mesh)}")

    # Convert to numpy arrays with correct types
    V = np.array(mesh.vertices, dtype=np.float64)
    F = np.array(mesh.faces, dtype=np.int64)

    # Some OBJ files have faces with shape (n,3,1), squeeze it
    if F.ndim > 2:
        F = F.squeeze()

    if F.shape[1] != 3:
        raise ValueError(f"Faces array must have shape (n_faces,3), got {F.shape}")

    F = F.astype(np.int64)

    return V, F


def apply_all_laplacians(V, F):
    """Return a dict of smoothed meshes using all Laplacians"""
    smoothed_meshes = {}

    # Uniform
    L_uniform = build_uniform_laplacian(F, len(V))
    smoothed_meshes["uniform"] = laplacian_smoothing(
        V, L_uniform, STEP_SIZE, SMOOTH_ITERATIONS
    )

    # Graph
    L_graph = build_graph_laplacian(F, len(V))
    smoothed_meshes["graph"] = laplacian_smoothing(
        V, L_graph, STEP_SIZE, SMOOTH_ITERATIONS
    )

    # Cotangent (smaller step for stability with large cotan weights)
    L_cotan = build_cotangent_laplacian(V, F)
    smoothed_meshes["cotangent"] = laplacian_smoothing(
        V, L_cotan, STEP_SIZE_COTANGENT, SMOOTH_ITERATIONS
    )

    return smoothed_meshes


def _scalar_to_fluorescent_rgb(s: np.ndarray) -> np.ndarray:
    """Map scalar in [0,1] to fluorescent-style RGB (cyan -> yellow)."""
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    r = s
    g = np.ones_like(s)
    b = 1.0 - s
    return np.column_stack([r, g, b])


def _center_and_scale_to_fit(V: np.ndarray, box_size: float = 0.5) -> np.ndarray:
    """Center mesh at origin and scale so it fits in a box of side box_size."""
    V = np.asarray(V, dtype=np.float64)
    center = V.mean(axis=0)
    V_c = V - center
    r = np.abs(V_c).max()
    if r < 1e-12:
        return V_c
    scale = (box_size / 2.0) / r
    return V_c * scale


# Distinct colors per operator for side-by-side comparison (Original + 3 Laplacians)
SIDEBYSIDE_COLORS = {
    "Original": (0.35, 0.75, 1.0),   # light blue / cyan
    "Uniform": (0.2, 1.0, 0.45),      # green
    "Graph": (1.0, 0.55, 0.2),        # orange
    "Cotangent": (0.75, 0.25, 1.0),   # purple
}


def visualize_side_by_side(V, F, smoothed_meshes, mesh_name, screenshot_path=None):
    """
    Compare Original + 3 Laplacian operators side by side: Original, Uniform-smoothed,
    Graph-smoothed, Cotangent-smoothed. Each has a distinct color. Four meshes when all valid.
    """
    ps.init()
    ps.remove_all_structures()

    # Always show 4: Original + uniform, graph, cotangent. Use V for any missing or blown-up mesh.
    def _ok(V_m):
        if V_m is None:
            return False
        if not np.isfinite(V_m).all():
            return False
        if np.abs(V_m).max() > 1e8:
            return False
        return True

    u = smoothed_meshes.get("uniform")
    g = smoothed_meshes.get("graph")
    c = smoothed_meshes.get("cotangent")
    uniform_V = u if _ok(u) else None
    graph_V = g if _ok(g) else None
    cotangent_V = c if _ok(c) else None
    layout = [
        ("Original", V),
        ("Uniform", uniform_V if uniform_V is not None else V),
        ("Graph", graph_V if graph_V is not None else V),
        ("Cotangent", cotangent_V if cotangent_V is not None else V),
    ]

    spacing = 0.5
    n = len(layout)
    offsets = np.linspace(-spacing * (n - 1) / 2, spacing * (n - 1) / 2, n)

    H_orig, _ = compute_mean_curvature(V, F)
    H_norm = (H_orig - H_orig.min()) / (H_orig.max() - H_orig.min() + 1e-12)
    H_colors = _scalar_to_fluorescent_rgb(H_norm)

    box_size = 0.5
    for i, (label, V_mesh) in enumerate(layout):
        V_placed = _center_and_scale_to_fit(V_mesh, box_size=box_size)
        V_placed[:, 0] += offsets[i]
        name = label.lower().replace(" ", "_")
        mesh_color = SIDEBYSIDE_COLORS.get(label, (0.5, 0.5, 0.5))
        ps_mesh = ps.register_surface_mesh(name, V_placed, F, color=mesh_color)
        ps_mesh.add_color_quantity(
            "mean_curvature", H_colors, defined_on="vertices", enabled=False
        )

    # Zoom in so all 4 meshes are clearly visible and fill the frame
    half = box_size / 2.0
    x_min = offsets[0] - half
    x_max = offsets[-1] + half
    margin = 0.25
    low = np.array([x_min - margin, -half - margin, -half - margin], dtype=np.float64)
    high = np.array([x_max + margin, half + margin, half + margin], dtype=np.float64)
    look_from_negative_z = "armadillo" in mesh_name.lower()
    frame_scene_bbox(
        low, high, distance_scale=0.65, look_from_negative_z=look_from_negative_z
    )
    if screenshot_path is None:
        screenshot_path = FIGURE_DIR / f"{mesh_name}_comparison_sidebyside.png"
    ps.screenshot(str(screenshot_path))
    print(f"[INFO] Saved side-by-side comparison: {screenshot_path}")
    ps.remove_all_structures()


def visualize_comparison(V, F, smoothed_meshes, mesh_name, screenshot_path=None):
    """Visualize original + smoothed meshes overlaid with curvature coloring. Scene is cleared at start."""
    ps.init()
    ps.remove_all_structures()

    print(f"[DEBUG] Original V dtype: {V.dtype}, sample:\n{V[:5]}")
    print(f"[DEBUG] F dtype: {F.dtype}, sample:\n{F[:5]}")

    mesh_color = (0.2, 1.0, 0.6)
    ps.register_surface_mesh("original", V, F, color=mesh_color)

    H, _ = compute_mean_curvature(V, F)
    H_normalized = (H - H.min()) / (H.max() - H.min() + 1e-12)
    H_colors = _scalar_to_fluorescent_rgb(H_normalized)

    for lap_name, V_smooth in smoothed_meshes.items():
        print(
            f"[DEBUG] {lap_name} V_smooth dtype: {V_smooth.dtype}, sample:\n{V_smooth[:5]}"
        )
        ps_mesh = ps.register_surface_mesh(
            f"{lap_name}_smoothed", V_smooth, F, color=mesh_color
        )
        ps_mesh.add_color_quantity(
            "mean_curvature", H_colors, defined_on="vertices", enabled=True
        )

    if screenshot_path is None:
        screenshot_path = FIGURE_DIR / f"{mesh_name}_comparison.png"
    ps.screenshot(str(screenshot_path))
    print(f"[INFO] Saved comparison screenshot: {screenshot_path}")

    ps.remove_all_structures()


# ----------------------------
# Main processing
# ----------------------------
def process_mesh(mesh_file: Path):
    print(f"\n[START] Processing mesh: {mesh_file.name}")

    V, F = load_mesh(mesh_file)
    print(f"V shape: {V.shape}, F shape: {F.shape}, F dtype: {F.dtype}")
    print(f"F sample (first 5 faces):\n{F[:5]}")  # DEBUG: check first faces

    # Metrics
    vol = mesh_volume(V, F)

    # DEBUG: wrap curvature computation in try-except
    try:
        print("[DEBUG] Calling compute_mean_curvature...")
        H, H_vec = compute_mean_curvature(V, F)
        print("[DEBUG] compute_mean_curvature finished successfully")
    except Exception as e:
        print("[ERROR] Failed in compute_mean_curvature")
        # Try to find problematic face
        for idx, face in enumerate(F):
            try:
                i, j, k = face
                _ = V[i], V[j], V[k]
            except Exception as e_inner:
                print(f"[DEBUG] Problematic face index {idx}: {face}, error: {e_inner}")
                break
        raise e

    H_stats = mean_curvature_stats(H)
    b_stats = boundary_stats(F)

    print(f"[INFO] Volume: {vol:.4f}")
    print(
        f"[INFO] Mean curvature stats: min={H_stats['min']:.3f}, max={H_stats['max']:.3f}, mean={H_stats['mean']:.3f}"
    )
    print(
        f"[INFO] Boundary edges: {b_stats['n_boundary_edges']}, vertices: {b_stats['n_boundary_vertices']}"
    )

    # Smoothing
    smoothed_meshes = apply_all_laplacians(V, F)

    # Drop any smoothed meshes with inf/nan (e.g. cotangent blow-up on bad meshes)
    valid_meshes = {
        name: V_smooth
        for name, V_smooth in smoothed_meshes.items()
        if np.isfinite(V_smooth).all()
    }
    if len(valid_meshes) < len(smoothed_meshes):
        skipped = set(smoothed_meshes.keys()) - set(valid_meshes.keys())
        print(f"[WARN] Skipping visualization for {skipped}: inf/nan in smoothed positions")

    # -------------------------------------------------------------------------
    # Visualization story (each step clears the scene so the next mesh starts clean)
    # -------------------------------------------------------------------------
    stem = mesh_file.stem

    # 01 – Meet the mesh: rotation of the original (before any smoothing)
    create_original_rotation_gif(
        V, F,
        gif_path=str(FIGURE_DIR / f"{stem}_01_original_rotation.gif"),
        n_frames=GIF_FRAMES,
    )
    print(f"[INFO] Saved 01 original rotation GIF")

    # 02 – Curvature heatmap on the original mesh (where is it curved?)
    save_heatmap_screenshot(
        V, F, H,
        screenshot_path=str(FIGURE_DIR / f"{stem}_02_curvature_heatmap.png"),
        mesh_name="original",
        quantity_name="mean_curvature",
    )
    print(f"[INFO] Saved 02 curvature heatmap")

    # 03 – Compare the 3 Laplacians side by side: Original | Uniform | Graph | Cotangent
    visualize_side_by_side(
        V, F, valid_meshes, stem,
        screenshot_path=str(FIGURE_DIR / f"{stem}_03_comparison_sidebyside.png"),
    )
    print(f"[INFO] Saved 03 side-by-side comparison (Original vs 3 operators)")

    # 04 – After smoothing: rotation of smoothed mesh colored by curvature
    create_smooth_curvature_gif(
        V, F,
        gif_path=str(FIGURE_DIR / f"{stem}_04_smooth_curvature.gif"),
        n_frames=GIF_FRAMES,
        step_size=STEP_SIZE,
        smooth_iterations=SMOOTH_ITERATIONS,
    )
    print(f"[INFO] Saved 04 smooth curvature GIF")

    # 05 – Transition GIF per operator: original → smoothed while rotating (skip if invalid or blown-up)
    for op_name in ("uniform", "graph", "cotangent"):
        V_smooth_op = valid_meshes.get(op_name)
        if V_smooth_op is None or not np.isfinite(V_smooth_op).all():
            continue
        if np.abs(V_smooth_op).max() > 1e8:
            continue
        create_morph_transition_gif(
            V, V_smooth_op, F,
            gif_path=str(FIGURE_DIR / f"{stem}_05_transition_{op_name}.gif"),
            n_frames=72,
        )
        print(f"[INFO] Saved 05 transition GIF ({op_name})")

    # 06 – Data: histogram of mean curvature (distribution)
    plot_histogram(
        H,
        bins=40,
        title=f"Mean curvature – {stem}",
        xlabel="Mean curvature",
        ylabel="Frequency",
        save_path=str(FIGURE_DIR / f"{stem}_06_curvature_histogram.png"),
    )
    print(f"[INFO] Saved 06 curvature histogram")

    # 07 – Data: 3D scatter of vertices colored by curvature
    plot_vertex_values(
        V, H,
        title=f"Mean curvature on mesh – {stem}",
        cmap="viridis",
        save_path=str(FIGURE_DIR / f"{stem}_07_curvature_3d.png"),
    )
    print(f"[INFO] Saved 07 curvature 3D plot")

    print(f"[DONE] Finished processing {mesh_file.name}\n")


# ----------------------------
# Run everything
# ----------------------------
if __name__ == "__main__":
    mesh_files = list(MESH_DIR.glob("*.obj"))
    if not mesh_files:
        print("[WARN] No mesh files found in data/meshes/")
        sys.exit(1)

    # Wrap the loop with tqdm for a progress bar
    for mesh_file in tqdm(mesh_files, desc="Processing meshes", unit="mesh"):
        try:
            process_mesh(mesh_file)
        except Exception as e:
            print(f"[ERROR] Error processing {mesh_file.name}: {e}")
