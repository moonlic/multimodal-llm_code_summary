import torch
import torch.nn.functional as F


def masked_mean(x, mask=None):
    if mask is None:
        return x.mean()
    return (x * mask).sum() / mask.sum().clamp_min(1)


def kl_divergence(p_logits, q_logits, mask=None, temperature=1.0, reverse=False):
    """
    p_logits, q_logits: [B, L, V]
    reverse=False: KL(P || Q)
    reverse=True:  KL(Q || P)
    """
    log_p = F.log_softmax(p_logits / temperature, dim=-1)
    log_q = F.log_softmax(q_logits / temperature, dim=-1)
    p, q = log_p.exp(), log_q.exp()

    token_loss = (q * (log_q - log_p)).sum(dim=-1) if reverse else (p * (log_p - log_q)).sum(dim=-1)
    return masked_mean(token_loss, mask)


if __name__ == "__main__":
    p_logits = torch.randn(2, 4, 10)
    q_logits = torch.randn(2, 4, 10)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    forward_kl = kl_divergence(p_logits, q_logits, mask)
    reverse_kl = kl_divergence(p_logits, q_logits, mask, reverse=True)
    zero_kl = kl_divergence(p_logits, p_logits, mask)
    print(forward_kl, reverse_kl, zero_kl)
