import torch
import torch.nn as nn
import numpy as np

class MLPmodel(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=512, out_dim=1, skip_layer=5, beta=100, fourier=False):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = [self.in_dim]
        self.out_dim = out_dim
        self.skip = skip_layer
        self.activation = nn.Softplus(beta=beta)
        # if fourier:
        #     self.activation = nn.Softplus(beta=beta)
        # else:
        #     self.activation = nn.Softplus(beta=10*beta)
        if fourier:
            self.fourier = Fourier_feature()
        else:
            self.fourier = None
    
        if fourier:
            self.n = 10
        else:
            self.n = 8
        n = self.n
        for i in range(n):
            if i != skip_layer:
                self.hidden_dims.append(hidden_dim)
            else:
                self.hidden_dims.append(hidden_dim - in_dim)
        self.hidden_dims.append(self.out_dim)

        for i in range(0, n + 1):
            if i - 1 != self.skip:
                layer = nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1])
            else:
                layer = nn.Linear(self.hidden_dims[i] + in_dim, self.hidden_dims[i + 1])
            
            # geometric initialize
            if i == n - 1:
                # 输出层特殊处理
                if fourier:
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
                else:
                    nn.init.normal_(layer.weight, mean=np.sqrt(np.pi)/np.sqrt(512), std=0.00001)
                    nn.init.constant_(layer.bias, -1.0)
            else:
                if fourier:
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    nn.init.zeros_(layer.bias)
                else:    
                    nn.init.normal_(layer.weight, mean=0.0, std=np.sqrt(2)/np.sqrt(self.hidden_dims[i + 1]))
                    nn.init.constant_(layer.bias, 0.0)
            setattr(self, "layer{}".format(i), layer)
    
    def forward(self, x):
        input = x
        n = self.n
        if self.fourier is not None:
            input = self.fourier(input)
            x = input
        for i in range(0, n + 1):
            layer = getattr(self, "layer{}".format(i))
            if i == self.skip + 1:
                x = torch.cat([x, input], -1)/np.sqrt(2) # 两个相互独立的方差为一的正态分布，和是方差为2的正态分布
            x = layer(x)
            if i < n:
                x = self.activation(x)
        return x



class Fourier_feature(nn.Module):
    def __init__(self, input_dim=3, output_dim=16, sigma=5):
        super().__init__()
        self.output_dim = output_dim
        fourier_weight = torch.randn((output_dim // 2, input_dim)) * sigma
        self.register_buffer("fourier_weight", fourier_weight)

    def forward(self, x):
        # x:(N, 3)
        x = 2 * torch.pi * x @ self.fourier_weight.T
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
