#!/bin/bash

python main.py --data_path=data/1a04e3eab45ca15dd86060f189eb133/pointcloud.npz --sample_data_path=data/1a04e3eab45ca15dd86060f189eb133/sdf.npz --obj_index=1 --epoches=80000 --fourier=1

python main.py --data_path=data/1a6ad7a24bb89733f412783097373bdc/pointcloud.npz --sample_data_path=data/1a6ad7a24bb89733f412783097373bdc/sdf.npz --obj_index=2 --epoches=100000 --fourier=1

python main.py --data_path=data/1a9b552befd6306cc8f2d5fe7449af61/pointcloud.npz --sample_data_path=data/1a9b552befd6306cc8f2d5fe7449af61/sdf.npz --obj_index=3 --epoches=100000 --fourier=1

python main.py --data_path=data/1a32f10b20170883663e90eaf6b4ca52/pointcloud.npz --sample_data_path=data/1a32f10b20170883663e90eaf6b4ca52/sdf.npz --obj_index=4 --epoches=100000 --fourier=1

python main.py --data_path=data/1a54a2319e87bd4071d03b466c72ce41/pointcloud.npz --sample_data_path=data/1a54a2319e87bd4071d03b466c72ce41/sdf.npz --obj_index=5 --epoches=100000 --fourier=1

