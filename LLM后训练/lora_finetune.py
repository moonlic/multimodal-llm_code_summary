import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_output = self.base(x)
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base_output + self.scaling * lora_output


def train_lora(model, x, targets, steps=50, lr=1e-2):
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    for _ in range(steps):
        logits = model(x)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.detach()


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(128, 16)
    targets = (x[:, 0] + x[:, 1] > 0).long()
    model = LoRALinear(in_features=16, out_features=2, rank=4, alpha=8)
    loss = train_lora(model, x, targets)

    assert model.base.weight.grad is None
    print(loss)
    print("trainable:", [name for name, parameter in model.named_parameters() if parameter.requires_grad])
