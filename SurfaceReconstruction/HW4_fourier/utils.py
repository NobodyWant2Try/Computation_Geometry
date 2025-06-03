import torch
from scipy.spatial import cKDTree
import numpy as np
import trimesh
from skimage import measure
from tqdm import tqdm
import Model

def load_point_data(file_path):
    dataset = np.load(file=file_path)
    data = np.hstack((dataset['points'], dataset['normals']))
    data = torch.from_numpy(data).float()
    # 为采集表面附近的扰动点做准备
    sigmas = [] 
    kdtree = cKDTree(data) # 用于查询最近邻点，用于计算高斯核
    for point in np.split(data, 100, axis=0):
        info = kdtree.query(point, 50 + 1)
        sigmas.append(info[0][:,-1]) # [0] 只取distance，不取索引。[:,-1] 只取每个点的第51个点的距离（因为包含了自己），以此作为高斯核
    
    local_sigma = np.concatenate(sigmas)
    local_sigma = torch.from_numpy(local_sigma).float().cuda()

    return data, local_sigma

def load_sample_data(file_path):
    dataset = np.load(file=file_path)
    sdf = dataset['sdf']
    sdf = sdf[:, None]
    data = np.hstack((dataset['points'], dataset['grad'], sdf))
    data = torch.from_numpy(data).float()
    return data

def sample(inputs, local_sigma, global_sigma):
    # 采集表面附近的点，同时采集其他位置的点，用于约束SDF的梯度范数为1
    samlpe_size, dim = inputs.shape
    local_sample = inputs + (torch.randn_like(inputs) * local_sigma.unsqueeze(-1)) # 生成标准正态分布噪声，并用local_sigma进行缩放，保证随机扰动与点所处位置是合理的
    global_sample = (torch.rand(samlpe_size // 4, dim, device=inputs.device) * (global_sigma * 2)) - global_sigma # 生成元素大小在[-global_sigma, global_sigma]范围内的张量，8是经验参数，控制全局采样点数不太多，提高训练效率
    sample_pts = torch.cat([local_sample, global_sample], dim=0) # 在样本数维度拼接
    return sample_pts

def gradient(inputs, outputs):
    points_weight = torch.ones_like(outputs, requires_grad=False, device=outputs.device) # outputs不是标量，要手动设置每个权重为1
    points_grad = torch.autograd.grad(outputs=outputs, inputs=inputs, grad_outputs=points_weight, create_graph=True, retain_graph=True)[0][:, -3:]
    return points_grad

def compute_sdf(model, resolution, bounds, device):
    # 划分体素格点，计算出每一个格点sdf
    sdf = []
    _min, _max = bounds
    x = np.linspace(_min.cpu(), _max.cpu(), resolution)
    y = x
    z = x
    xx, yy, zz = np.meshgrid(x, y, z)
    grid = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    grid = torch.tensor(grid).float().to(device)
    with torch.no_grad():
        model.eval()
        for _, pts in enumerate(tqdm(torch.split(grid, 10000, dim=0))):
            sdf.append(model(pts).detach().cpu().numpy())
        sdf = np.concatenate(sdf, axis=0).astype(float)
        sdf = np.reshape(sdf, (resolution, resolution, resolution))
    return sdf

def to_mesh(sdf, file_path, obj_index):
    vertices, faces, normals, values = measure.marching_cubes(sdf, level=0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=False)
    if obj_index == 1:
        mesh.export(file_path) 
        return
    components = mesh.split(only_watertight=False)
    main_component = max(components, key=lambda c: c.area)  # 或 c.volume
    smooth = laplacian_smoothing(main_component)
    smooth.export(file_path)
    return 

def test(model_path, data, device, resolution, file_path, fourier, obj_index):
    checkpoint = torch.load(model_path)
    
    if fourier == 1:
        model = Model.MLPmodel(in_dim=16, fourier=True).to(device)
        model.load_state_dict(checkpoint)
    else:
        model = Model.MLPmodel(in_dim=3, skip_layer=4, fourier=False).to(device)
        model.load_state_dict(checkpoint)
    
    points = data.to(device)
    bounds = (torch.min(points[:, :3]), torch.max(points[:, :3]))
    sdf = compute_sdf(model, resolution, bounds, device)
    to_mesh(sdf, file_path, obj_index)
    return 

def normalize(pts):
    center = pts.mean(dim=0)
    pts = pts - center
    scale = pts.abs().max()
    pts = pts / scale
    return pts


def laplacian_smoothing(mesh, iterations=10, lambda_=0.3):
    V = mesh.vertices.copy()
    F = mesh.faces
    adj = mesh.vertex_neighbors
    for _ in range(iterations):
        new_V = V.copy()
        for i in range(len(V)):
            neighbors = adj[i]
            if len(neighbors) == 0:
                continue
            avg = np.mean(V[neighbors], axis=0)
            new_V[i] = V[i] + lambda_ * (avg - V[i])
        V = new_V
    smoothed_mesh = trimesh.Trimesh(vertices=V, faces=F)
    return smoothed_mesh