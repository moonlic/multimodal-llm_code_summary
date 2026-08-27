import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__()

        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_groups = num_heads // num_kv_heads

        # Q 保留 num_heads 个头
        self.q_proj = nn.Linear(hidden_size, hidden_size)

        # K、V 只有 num_kv_heads 个头
        kv_size = num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(hidden_size, kv_size)
        self.v_proj = nn.Linear(hidden_size, kv_size)

        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, causal=True):
        """
        x: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape

        # q: [B, num_heads, L, head_dim]
        q = self.q_proj(x)
        q = q.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # k, v: [B, num_kv_heads, L, head_dim]
        k = self.k_proj(x)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 每个 KV 头分配给 num_groups 个 Query 头
        # [B, num_kv_heads, L, D]
        # -> [B, num_heads, L, D]
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        # [B, num_heads, L, L]
        attention_score = q @ k.transpose(-2, -1)
        attention_score = attention_score / math.sqrt(self.head_dim)

        if causal:
            mask = torch.triu(torch.ones(seq_len,seq_len,dtype=torch.bool,device=x.device,),diagonal=1,)

            attention_score = attention_score.masked_fill(mask,float("-inf"))

        attention_weight = F.softmax(attention_score, dim=-1)

        # [B, num_heads, L, head_dim]
        output = attention_weight @ v

        # [B, L, hidden_size]
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, -1)

        return self.out_proj(output)


if __name__ == "__main__":
    x = torch.randn(2, 10, 512)

    gqa = GroupedQueryAttention(
        hidden_size=512,
        num_heads=8,
        num_kv_heads=2,
    )

    output = gqa(x)

    print("input :", x.shape)
    print("output:", output.shape)