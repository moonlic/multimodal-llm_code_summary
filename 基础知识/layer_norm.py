import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        """x: [B, n, d_model]"""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x + self.bias


if __name__ == "__main__":
    x = torch.rand(2, 10, 512)
    layer_norm = LayerNorm(d_model=512)
    output = layer_norm(x)
    print(output.shape)                  # torch.Size([2, 10, 512])
    print(output.mean(dim=-1)[0, :3])   # 接近 0
    print(output.var(dim=-1, unbiased=False)[0, :3])  # 接近 1
