import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [B, h, n, d_k]
    K: [B, h, m, d_k]
    V: [B, h, m, d_v]
    mask: 可广播到 [B, h, n, m]，1 表示保留，0 表示屏蔽
    """
    d_k = Q.size(-1)
    attention_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        attention_score = attention_score.masked_fill(mask == 0, float("-inf"))

    attention_weight = F.softmax(attention_score, dim=-1)
    output = torch.matmul(attention_weight, V)
    return output, attention_weight


if __name__ == "__main__":
    Q = torch.rand(2, 4, 3, 8)
    K = torch.rand(2, 4, 3, 8)
    V = torch.rand(2, 4, 3, 8)
    mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))

    output, attention_weight = scaled_dot_product_attention(Q, K, V, mask)
    print(output.shape)             # torch.Size([2, 4, 3, 8])
    print(attention_weight.shape)   # torch.Size([2, 4, 3, 3])
