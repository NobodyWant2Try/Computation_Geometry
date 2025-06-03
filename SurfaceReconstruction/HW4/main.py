import torch
import argparse
import utils
import Model
import train
import time

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--obj_index", type=int)
    args = parser.parse_args()

    model = Model.MLPmodel()
    data, local_sigmas = utils.load_point_data(args.data_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './checkpoints/model_for_obj{}'.format(args.obj_index)
    train.train(model, data, epoches=8000, lr=0.001, local_sigmas=local_sigmas, batch_size=16384, device=device, file_path=model_path)
    
    output_path = './results/obj{}.ply'.format(args.obj_index)
    utils.test(model_path, data, device, 512, output_path)
    t = time.time() - start
    with open("record.txt", "a") as f:
        f.write("object {} running time: {}s\n".format(args.obj_index, t))