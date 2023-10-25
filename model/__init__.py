import torch
import torch.nn as nn
class TransformLayer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float)[None, :, None, None].cuda()
        std = torch.as_tensor(std, dtype=torch.float)[None, :, None, None].cuda()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, x):
        return x.sub(self.mean).div(self.std)