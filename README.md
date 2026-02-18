# Laplacian Operators on Triangle Meshes

A Python project that implements **discrete Laplacians** (uniform, graph, and cotangent) on triangle meshes, with smoothing experiments, mean curvature computation, and visualization (Polyscope, matplotlib, GIFs).

## Requirements

- **Python 3.8+**
- See `requirements.txt` for pinned dependencies (NumPy, SciPy, trimesh, Polyscope, matplotlib, imageio, tqdm, etc.)

## Setup

1. **Clone or download** this repository and go into the project folder:

   ```bash
   cd LaplacianOperator_Debug
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   ```

   - **Windows:** `venv\Scripts\activate`
   - **macOS/Linux:** `source venv/bin/activate`

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Reproducing results

All commands below are meant to be run from the **project root** (`LaplacianOperator_Debug`), with the virtual environment activated.

### 1. Generate meshes (optional)

If `data/meshes/` is empty or you want to regenerate the built-in test meshes:

```bash
python data/createMesh.py
```

This creates in `data/meshes/`:

- `sphere_regular.obj` — regular icosphere  
- `sphere_irregular.obj` — noisy sphere  
- `plane_boundary.obj` — flat disk with boundary  
- `skinny_triangles.obj` — skinny-triangle stress case  

You can add your own `.obj` files to `data/meshes/` (e.g. `armadillo.obj`, `stanford-bunny.obj`); `run_all.py` will pick them up.

### Where the meshes come from

- **Generated meshes** (`sphere_regular`, `sphere_irregular`, `plane_boundary`, `skinny_triangles`) — Created by `data/createMesh.py` using [trimesh](https://trimsh.org/) (e.g. `trimesh.creation.icosphere`). No external download; run `python data/createMesh.py` to create them.
- **Stanford Bunny** — From the [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/) (Stanford University). See [Bunny](https://graphics.stanford.edu/data/3Dscanrep/#bunny); often used as a reduced `.obj` from that page or from [The Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/).
- **Armadillo** — Also from the [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/). See [Armadillo](https://graphics.stanford.edu/data/3Dscanrep/#armadillo).

If you use Stanford meshes, download the desired resolution (e.g. `.ply` or `.obj`), convert to `.obj` if needed, and place them in `data/meshes/` (e.g. `stanford-bunny.obj`, `armadillo.obj`). The repo does not ship these files; add them locally to reproduce results that use them.

### 2. Run the full pipeline

Process all meshes in `data/meshes/`, compare the three Laplacians, and write figures and GIFs to `figures/`:

```bash
python experiments/run_all.py
```

This will:

- Load each `.obj` in `data/meshes/`
- Compute volume, mean curvature stats, and boundary stats
- Smooth with **uniform**, **graph**, and **cotangent** Laplacians
- Save comparison screenshots and smoothing/rotation GIFs in `figures/`

**Note:** A Polyscope window may open and close per mesh; that’s expected. Ensure `data/meshes/` contains at least one `.obj` (either from `createMesh.py` or your own).

### 3. Run a single-mesh smoothing experiment

Smooth one mesh with the **uniform** Laplacian and show original vs smoothed in Polyscope:

```bash
python experiments/smooth_mesh.py --mesh data/meshes/sphere_regular.obj
```

Options:

- `--step` — smoothing step size (default: `0.01`)
- `--iters` — number of iterations (default: `20`)

Example:

```bash
python experiments/smooth_mesh.py --mesh data/meshes/plane_boundary.obj --step 0.01 --iters 50
```

### 4. Visualize meshes (screenshots)

To take normalized screenshots of all meshes in `data/meshes/` (no smoothing):

```bash
python data/visualizeMesh.py
```

Outputs go to `data/visualization/` (created if missing).

## Project layout

```
LaplacianOperator_Debug/
├── data/
│   ├── createMesh.py      # Generate test meshes
│   ├── visualizeMesh.py   # Screenshot meshes
│   └── meshes/            # Input .obj files
├── geometry/              # Adjacency, areas, boundaries, curvature
├── operators/             # Uniform, graph, cotangent Laplacians
├── metrics/               # Volume, curvature stats, boundary stats
├── visualization/        # Plots, heatmaps, GIFs, Polyscope helpers
├── experiments/
│   ├── smooth_mesh.py     # Single-mesh smoothing (CLI)
│   └── run_all.py        # Full pipeline: all meshes, all Laplacians
├── figures/               # Output screenshots and GIFs (created by run_all)
├── requirements.txt
└── README.md
```

## Troubleshooting

- **“Mesh not found”** — Run from `LaplacianOperator_Debug` and ensure `data/meshes/` contains the requested `.obj` (or run `python data/createMesh.py` first).
- **Polyscope window closes immediately** — Normal when running `run_all.py` in batch; figures are still saved to `figures/`.
- **Import errors** — Scripts assume they are run from the project root so that `sys.path` includes `LaplacianOperator_Debug`; don’t run them from a different directory without adjusting `PYTHONPATH` or `sys.path`.
