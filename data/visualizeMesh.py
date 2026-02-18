import polyscope as ps
import trimesh
import numpy as np
from pathlib import Path

# Setup paths
MESH_DIR = Path("data/meshes")
OUTPUT_DIR = Path("data/visualization")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Global Settings (Set once before the loop) ---
ps.init()
ps.set_screenshot_extension(".png")
ps.set_ground_plane_mode("shadow_only")  # Clean floor with shadows
# Set background to solid white for a "clean" look
ps.set_background_color([1.0, 1.0, 1.0])

# 1. Loop through all .obj files in the directory
for mesh_path in MESH_DIR.glob("*.obj"):
    print(f"Processing: {mesh_path.name}")

    # 2. Load the mesh
    mesh = trimesh.load(mesh_path, process=False)

    # 3. Normalize for Uniformity
    # Ensures a small sphere and giant armadillo are the same size in frame
    mesh.vertices -= mesh.vertices.mean(axis=0)  # Center at origin [0,0,0]
    # Scale such that the furthest vertex is at distance 1.0
    mesh.vertices /= np.max(np.abs(mesh.vertices))

    # 4. Register to Polyscope
    ps_mesh = ps.register_surface_mesh(mesh_path.stem, mesh.vertices, mesh.faces)

    # --- Uniform Material Settings ---
    ps_mesh.set_color([0.2, 0.5, 0.8])  # Soft "Engineering Blue"
    ps_mesh.set_smooth_shade(True)
    ps_mesh.set_edge_width(1.0)  # Wireframe overlay for detail

    # --- Fixed Camera View ---
    # Since meshes are normalized to a 1.0 unit sphere, this view is always perfect
    ps.look_at([1.8, 1.2, 1.8], [0, 0, 0])

    # 5. Save the screenshot
    screenshot_name = OUTPUT_DIR / f"{mesh_path.stem}_uniform.png"
    ps.screenshot(str(screenshot_name))

    # 6. Clear for next mesh
    ps_mesh.remove()

print(f"Done! Check your uniform screenshots in: {OUTPUT_DIR.absolute()}")
