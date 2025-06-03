import torch
import torch.nn as nn
import numpy as np

class MLPmodel(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=512, out_dim=1, skip_layer=4, beta=100):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = [self.in_dim]
        self.out_dim = out_dim
        self.skip = skip_layer
        self.activation = nn.Softplus(beta=beta)
    
        for i in range(8):
            if i != skip_layer:
                self.hidden_dims.append(hidden_dim)
            else:
                self.hidden_dims.append(hidden_dim - in_dim)
        self.hidden_dims.append(self.out_dim)

        for i in range(0, 9):
            if i - 1 != self.skip:
                layer = nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1])
            else:
                layer = nn.Linear(self.hidden_dims[i] + in_dim, self.hidden_dims[i + 1])
            
            # geometric initialize
            if i == 8:
                # 输出层特殊处理
                nn.init.normal_(layer.weight, mean=np.sqrt(np.pi)/np.sqrt(512), std=0.00001)
                nn.init.constant_(layer.bias, -1.0)
            else:
                nn.init.normal_(layer.weight, mean=0.0, std=np.sqrt(2)/np.sqrt(self.hidden_dims[i + 1]))
                nn.init.constant_(layer.bias, 0.0)
            setattr(self, "layer{}".format(i), layer)
    
    def forward(self, x):
        input = x
        for i in range(0, 9):
            layer = getattr(self, "layer{}".format(i))
            if i == self.skip + 1:
                x = torch.cat([x, input], -1)/np.sqrt(2) # 两个相互独立的方差为一的正态分布，和是方差为2的正态分布
            x = layer(x)
            if i < 8:
                x = self.activation(x)
        return x


