import torch
import torch.nn.functional as F


def cross_entropy_loss(logits, targets, mask=None, label_smoothing=0.0, ignore_index=-100):
    """
    logits: [B, L, V]
    targets: [B, L]
    mask: [B, L]，1 表示计算 loss
    """
    valid_mask = targets != ignore_index
    safe_targets = targets.masked_fill(~valid_mask, 0)

    log_probs = F.log_softmax(logits, dim=-1)
    nll_loss = -log_probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -log_probs.mean(dim=-1)
    token_loss = (1 - label_smoothing) * nll_loss + label_smoothing * smooth_loss

    loss_mask = valid_mask if mask is None else valid_mask & mask.bool()
    return (token_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1)


if __name__ == "__main__":
    logits = torch.randn(2, 4, 10)
    targets = torch.randint(0, 10, (2, 4))
    targets[0, -1] = -100

    loss = cross_entropy_loss(logits, targets, label_smoothing=0.1)
    reference = F.cross_entropy(logits.view(-1, 10), targets.view(-1), ignore_index=-100, label_smoothing=0.1)
    torch.testing.assert_close(loss, reference)
    print(loss)
