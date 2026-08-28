import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionKVCache(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache=None, use_cache=False):
        """x: [B, n, d_model], cache: (K, V), K/V: [B, h, past_len, d_k]"""
        B, n, _ = x.shape
        Q = self.W_Q(x).view(B, n, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, n, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, n, self.num_heads, self.d_k).transpose(1, 2)

        past_len = 0
        if cache is not None:
            past_K, past_V = cache
            past_len = past_K.size(2)
            K, V = torch.cat([past_K, K], dim=2), torch.cat([past_V, V], dim=2)

        attention_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        query_position = past_len + torch.arange(n, device=x.device)
        key_position = torch.arange(K.size(2), device=x.device)
        mask = key_position.unsqueeze(0) <= query_position.unsqueeze(1)
        attention_score = attention_score.masked_fill(~mask, float("-inf"))

        attention_weight = F.softmax(attention_score, dim=-1)
        output = torch.matmul(attention_weight, V)
        output = output.transpose(1, 2).contiguous().view(B, n, self.d_model)
        new_cache = (K, V) if use_cache else None
        return self.W_O(output), new_cache


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 6, 64)
    attention = MultiHeadAttentionKVCache(d_model=64, num_heads=4).eval()

    full_output, _ = attention(x)
    cache, step_outputs = None, []
    for i in range(x.size(1)):
        step_output, cache = attention(x[:, i:i + 1], cache, use_cache=True)
        step_outputs.append(step_output)

    cached_output = torch.cat(step_outputs, dim=1)
    torch.testing.assert_close(full_output, cached_output, atol=1e-5, rtol=1e-5)
    print(cached_output.shape, cache[0].shape)
