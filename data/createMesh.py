"""
Geometry Generation Utilities for Laplacian Operator Analysis.

This module provides functions to generate standard and edge-case meshes
to test the robustness of discrete Laplacian operators.
"""

import trimesh
import numpy as np
from pathlib import Path

# Constants for file management
OUTPUT_DIR = Path("data/meshes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Base Case: Regular Unit Sphere
sphere = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
sphere.export(OUTPUT_DIR / "sphere_regular.obj")

# 2. Noisy Case: Randomly Remeshed Sphere
V = sphere.vertices.copy()
noise = 0.05 * np.random.randn(*V.shape)
V_noisy = V + noise

irregular = trimesh.Trimesh(vertices=V_noisy, faces=sphere.faces)
irregular.export(OUTPUT_DIR / "sphere_irregular.obj")

# 3. Boundary Case: Flat Disk
n = 30
x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
z = np.zeros_like(x)

V = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

faces = []
for i in range(n - 1):
    for j in range(n - 1):
        idx = i * n + j
        faces.append([idx, idx + 1, idx + n])
        faces.append([idx + 1, idx + n + 1, idx + n])

plane = trimesh.Trimesh(vertices=V, faces=np.array(faces))
plane.export(OUTPUT_DIR / "plane_boundary.obj")

# 4. Stress Case: Skinny-Triangle Mesh
V_bad = plane.vertices.copy()
V_bad[:, 0] *= 10.0
V_bad[:, 1] *= 0.05

skinny = trimesh.Trimesh(vertices=V_bad, faces=plane.faces)
skinny.export(OUTPUT_DIR / "skinny_triangles.obj")
