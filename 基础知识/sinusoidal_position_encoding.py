import math

import torch
import torch.nn as nn


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"

        position = torch.arange(max_len).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        position_encoding = torch.zeros(max_len, d_model)
        position_encoding[:, 0::2] = torch.sin(position * frequency)
        position_encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("position_encoding", position_encoding)

    def forward(self, x):
        """x: [B, n, d_model]"""
        n = x.size(1)
        position_encoding = self.position_encoding[:n].to(dtype=x.dtype)
        return x + position_encoding.unsqueeze(0)


if __name__ == "__main__":
    x = torch.zeros(2, 10, 512)
    position_encoding = SinusoidalPositionEncoding(d_model=512)
    output = position_encoding(x)
    print(output.shape)   # torch.Size([2, 10, 512])
    print(output[0, :2, :6])
