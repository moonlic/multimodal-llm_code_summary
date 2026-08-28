import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """x: [B, n, d_model]"""
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


if __name__ == "__main__":
    x = torch.rand(2, 10, 512)
    rms_norm = RMSNorm(d_model=512)
    output = rms_norm(x)
    print(output.shape)                            # torch.Size([2, 10, 512])
    print(output.pow(2).mean(dim=-1)[0, :3])       # 接近 1
