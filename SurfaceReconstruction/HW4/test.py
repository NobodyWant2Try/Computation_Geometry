import torch
import utils
import Model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--obj_index", type=int)
    args = parser.parse_args()

    model = Model.MLPmodel()
    model_path = './checkpoints/model_for_obj{}'.format(args.obj_index)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data, _ = utils.load_point_data(args.data_path)
    output_path = './results/obj{}_for_test.ply'.format(args.obj_index)
    utils.test(model_path, data, device, 512, output_path)