import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.W_2(F.gelu(self.W_1(x)))


class SparseMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x):
        """x: [B, L, D]"""
        B, L, D = x.shape
        x_flat = x.view(-1, D)
        router_probs = F.softmax(self.router(x_flat), dim=-1)                  # [B*L, E]
        topk_weight, topk_index = router_probs.topk(self.top_k, dim=-1)       # [B*L, K]
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            token_index, slot_index = torch.where(topk_index == expert_id)
            if token_index.numel() == 0:
                continue
            expert_output = expert(x_flat[token_index])
            output.index_add_(0, token_index, topk_weight[token_index, slot_index, None] * expert_output)

        top1_fraction = F.one_hot(topk_index[:, 0], self.num_experts).float().mean(dim=0)
        mean_router_prob = router_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(top1_fraction * mean_router_prob)
        return output.view(B, L, D), aux_loss, topk_index.view(B, L, self.top_k)


if __name__ == "__main__":
    x = torch.randn(2, 8, 64)
    moe = SparseMoE(d_model=64, d_ff=128, num_experts=4, top_k=2)
    output, aux_loss, selected_experts = moe(x)
    print(output.shape, aux_loss, selected_experts.shape)
