import argparse
import numpy as np
import trimesh
import polyscope as ps

from operators.uniform import build_uniform_laplacian, laplacian_smoothing


def load_mesh(path):
    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(mesh)}")
    V = np.array(mesh.vertices, dtype=np.float64)
    F = np.array(mesh.faces, dtype=np.int64)
    if F.ndim > 2:
        F = F.squeeze()
    return V, F


def smooth_with_operator(V, F, operator_builder, step_size=0.01, iterations=20):
    """
    Generic smoothing pipeline.
    operator_builder(F, n_vertices) must return a Laplacian matrix L.
    Use build_uniform_laplacian or build_graph_laplacian; build_cotangent_laplacian
    has signature (V, F) and is not supported here.
    """
    L = operator_builder(F, len(V))
    V_smooth = laplacian_smoothing(V, L, step_size, iterations)
    return V_smooth


def visualize(V_original, V_smooth, F):
    ps.init()

    ps.register_surface_mesh("original", V_original, F)
    ps.register_surface_mesh("smoothed", V_smooth, F)

    ps.show()


def main():
    parser = argparse.ArgumentParser(description="Mesh smoothing experiment")
    parser.add_argument("--mesh", type=str, required=True, help="Path to mesh file")
    parser.add_argument("--step", type=float, default=0.01, help="Smoothing step size")
    parser.add_argument(
        "--iters", type=int, default=20, help="Number of smoothing iterations"
    )

    args = parser.parse_args()

    V, F = load_mesh(args.mesh)

    V_smooth = smooth_with_operator(
        V,
        F,
        operator_builder=build_uniform_laplacian,
        step_size=args.step,
        iterations=args.iters,
    )

    visualize(V, V_smooth, F)


if __name__ == "__main__":
    main()
