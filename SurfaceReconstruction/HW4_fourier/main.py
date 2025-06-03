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
    parser.add_argument("--sample_data_path", type=str)
    parser.add_argument("--fourier", type=int, default=1)
    parser.add_argument("--epoches", type=int, default=10000)
    args = parser.parse_args()

    if args.fourier == 1:
        model = Model.MLPmodel(in_dim=16, fourier=True)
    else:
        fourier = None
        model = Model.MLPmodel(in_dim=3, skip_layer=4, fourier=False)

    data, local_sigmas = utils.load_point_data(args.data_path)
    sample_data = utils.load_sample_data(args.sample_data_path)
    # print(sample_data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.fourier == 1:
        model_path = './checkpoints/fourier_model_for_obj{}.pth'.format(args.obj_index)
    else:
        model_path = './checkpoints/model_for_obj{}.pth'.format(args.obj_index)


    train.train(model, data, sample_data, epoches=args.epoches, lr=0.005, local_sigmas=local_sigmas, batch_size=24000, device=device, file_path=model_path, fourier=args.fourier)
    
    if args.fourier == 1:
        output_path = './results/fourier_obj{}.ply'.format(args.obj_index)
    else:
        output_path = './results/obj{}.ply'.format(args.obj_index)
    utils.test(model_path, data, device, 512, output_path, args.fourier, args.obj_index)

    t = time.time() - start
    with open("record.txt", "a") as f:
        f.write("object {} running time: {}s\n".format(args.obj_index, t))