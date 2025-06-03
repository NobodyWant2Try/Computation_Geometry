import torch
import numpy as np
from tqdm import tqdm
import utils


def train(model, points, sample_points, epoches, lr, local_sigmas, batch_size, device, file_path, fourier):
    # 参数
    points = points.to(device)
    points.requires_grad_()
    sample_points.to(device)
    sample_points = sample_points.requires_grad_()
    model = model.to(device)

    if fourier == 0:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)
    global_sigma = 1.8
    normal_lam = 1.0
    grad_lam = 0.1
    min_loss = float('inf')
    
    # if fourier: 
        # grad_lam = 0.2

    # 训练
    for epoch in tqdm(range(epoches)):
        
        model.train()
        indices = torch.tensor(np.random.choice(points.shape[0], batch_size, replace=False))
        q = points[indices, :3]
        # q = (q - q.mean()) / torch.max(torch.abs(q))
        N_q = points[indices, 3:]
        if fourier == 0:   
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
        else:
            index = torch.randint(0, sample_points.shape[0], (batch_size // 4,))
            sample = sample_points[index].to(device)
            q_sample = sample[:, :3]
            y_sample_GT = sample[:, 6].unsqueeze(-1)
            q_sample_grad_GT = sample[:, 3:6]
            y = model(q)
            y_sample = model(q_sample)
            q_grad = utils.gradient(q, y)
            q_sample_grad = utils.gradient(q_sample, y_sample)
            sdf_loss = torch.mean(torch.sum(y ** 2) + torch.sum((y_sample - y_sample_GT) ** 2))
            grad_loss = torch.mean(torch.norm(q_grad - N_q, dim=1)) + torch.mean(torch.norm(q_sample_grad - q_sample_grad_GT, dim=1))
            grad_normal_loss = torch.mean(torch.pow(torch.norm(q_sample_grad, dim=1) - 1, 2))
            loss = sdf_loss + 0.1 * grad_loss + grad_normal_loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        #评估
        if (epoch + 1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                if fourier == 0:
                    print(f"epoch{epoch + 1}, loss = {loss.item()}, y_loss = {y_loss.item()}, normal_loss = {normal_loss.item()}, grad_loss = {grad_loss.item()}")
                else:
                    print(f"epoch{epoch + 1}, loss = {loss.item()}, sdf_loss = {sdf_loss.item()}, grad_loss = {grad_loss.item()}, grad_normal_loss = {grad_normal_loss.item()}")
                if loss.item() < min_loss:
                    min_loss = loss.item()
                    torch.save(model.state_dict(), file_path)
