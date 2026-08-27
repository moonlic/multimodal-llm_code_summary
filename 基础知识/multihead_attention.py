import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 定义线性投影层
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, Q, K, V, mask=None):
        """
        Q:[B,n,d_model]
        K:[B,m,d_model]
        V:[B,m,d_model]

        """
        b, n, _ = Q.size()
        m = K.size(1)
        # 线性变换
        Q = (self.W_Q(Q).view(b, n, self.num_heads, self.d_k).transpose(1, 2))  # [B,n,h,d_k]
        K = (self.W_K(K).view(b, m, self.num_heads, self.d_k).transpose(1, 2))  # [B,m,h,d_k]
        V = (self.W_V(V).view(b, m, self.num_heads, self.d_k).transpose(1, 2))  # [B,m,h,d_k]

        # 计算注意力得分
        scores = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.d_k)  # [B,n,h,m]
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)  # [B,n,h,d_k]
        out = out.transpose(1, 2).contiguous().view(b, n, self.d_model)  # [B,n,d_model]
        out = self.W_O(out)
        return out


if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    mha = MultiHeadAttention(d_model, num_heads)
    Q = torch.rand(10, 20, d_model)
    K = torch.rand(10, 30, d_model)
    V = torch.rand(10, 30, d_model)
    out = mha(Q, K, V)
    print(out.size())
    # torch.Size([10, 20, 512])
