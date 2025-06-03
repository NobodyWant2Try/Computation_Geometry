import torch
import numpy as np
from tqdm import tqdm
import utils


def train(model, points, epoches, lr, local_sigmas, batch_size, device, file_path):
    # 参数
    points = points.to(device)
    points.requires_grad_()
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    global_sigma = 1.8
    normal_lam = 1.0
    grad_lam = 0.1
    min_loss = float('inf')
    # 训练
    for epoch in tqdm(range(epoches)):
        model.train()
        indices = torch.tensor(np.random.choice(points.shape[0], batch_size, replace=False))
        q = points[indices, :3]
        N_q = points[indices, 3:]
        local_sigma = local_sigmas[indices]
        q_sample = utils.sample(q, local_sigma, global_sigma) # 采集不在表面上的点
        y = model(q)
        y_sample = model(q_sample)
        q_grad = utils.gradient(q, y)
        q_sample_grad = utils.gradient(q_sample, y_sample)
        y_loss = torch.mean(torch.abs(y))
        normal_loss = torch.mean(torch.norm(q_grad - N_q, dim=1))
        grad_loss = torch.mean(torch.pow(torch.norm(q_sample_grad, dim=1) - 1, 2))

        loss = y_loss + normal_lam * normal_loss + grad_lam * grad_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #评估
        if epoch % 100 == 0:
            print(f"epoch{epoch}, loss = {loss.item()}, y_loss = {y_loss.item()}, normal_loss = {normal_loss.item()}, grad_loss = {grad_loss.item()}")
            if loss.item() < min_loss:
                min_loss = loss.item()
                torch.save(model.state_dict(), file_path)