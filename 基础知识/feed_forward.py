import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """x: [B, n, d_model]"""
        return self.W_2(F.relu(self.W_1(x)))


if __name__ == "__main__":
    x = torch.rand(2, 10, 512)
    ffn = FeedForward(d_model=512, d_ff=2048)
    output = ffn(x)
    print(output.shape)  # torch.Size([2, 10, 512])
