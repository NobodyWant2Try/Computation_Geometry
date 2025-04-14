import trimesh
import LaplacianSmoothing
import BilateralNormalFilter
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='Bilateral', type=str)
    parser.add_argument('--stepsize', default=1e-5, type=float)
    parser.add_argument('--iteration', default=50, type=int)
    parser.add_argument('--mode', default='uniform', type=str)
    parser.add_argument('--cell', default='Barycentric', type=str)
    args = parser.parse_args()

    mesh = trimesh.load('./smoothing.obj')
    if args.method == 'Bilateral':
        obj = BilateralNormalFilter.bilateral_normal_filter(mesh)
        obj.update_faces()
        result = obj.update_vertex()
        result.export('result.obj')
    elif args.method == 'Laplacian':
        obj = LaplacianSmoothing.LaplacianSmoothing(h=args.stepsize, iteration=args.iteration, mesh=mesh, cell=args.cell, mode=args.mode)
        result = obj.smooth()
        result.export('result.obj')
    else:
        print("error!")