import torch
import utils
import Model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--obj_index", type=int)
    args = parser.parse_args()

    model = Model.MLPmodel(in_dim=16, fourier=True)
    model_path = './checkpoints/fourier_model_for_obj{}.pth'.format(args.obj_index)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data, _ = utils.load_point_data(args.data_path)
    output_path = './results/fourier_obj{}_for_test.ply'.format(args.obj_index)
    utils.test(model_path, data, device, 512, output_path, fourier=1, obj_index=args.obj_index)