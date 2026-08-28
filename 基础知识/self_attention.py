import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        """
        x: [B, n, d_model]
        mask: 可广播到 [B, n, n]，1 表示保留，0 表示屏蔽
        """
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        attention_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, float("-inf"))

        attention_weight = F.softmax(attention_score, dim=-1)
        output = torch.matmul(attention_weight, V)
        return self.W_O(output)


if __name__ == "__main__":
    x = torch.rand(2, 5, 64)
    mask = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    attention = SelfAttention(d_model=64)
    output = attention(x, mask)
    print(output.shape)  # torch.Size([2, 5, 64])
