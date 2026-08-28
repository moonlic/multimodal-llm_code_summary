import torch


def grpo_loss(new_log_probs, old_log_probs, reference_log_probs, rewards, mask=None,
              clip_eps=0.2, beta=0.04, eps=1e-6):
    """
    new/old/reference_log_probs: [B, G, L]
    rewards: [B, G]，每个 prompt 有 G 个 rollout
    mask: [B, G, L]
    """
    group_mean = rewards.mean(dim=1, keepdim=True)
    group_std = rewards.std(dim=1, keepdim=True, unbiased=False)
    advantages = ((rewards - group_mean) / (group_std + eps)).detach()         # [B, G]

    log_ratio = new_log_probs - old_log_probs.detach()
    ratio = torch.exp(log_ratio)
    advantages_token = advantages.unsqueeze(-1)
    surrogate_1 = ratio * advantages_token
    surrogate_2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages_token

    log_ref_ratio = reference_log_probs.detach() - new_log_probs
    kl = torch.exp(log_ref_ratio) - log_ref_ratio - 1
    token_loss = -(torch.minimum(surrogate_1, surrogate_2) - beta * kl)

    if mask is None:
        loss = token_loss.mean()
    else:
        loss = (token_loss * mask).sum() / mask.sum().clamp_min(1)
    return loss, advantages, kl.mean().detach()


if __name__ == "__main__":
    torch.manual_seed(0)
    old_log_probs = torch.randn(2, 4, 6)
    new_log_probs = (old_log_probs + 0.05 * torch.randn_like(old_log_probs)).requires_grad_()
    reference_log_probs = old_log_probs + 0.1 * torch.randn_like(old_log_probs)
    rewards = torch.tensor([[1.0, 0.0, 2.0, 1.5], [0.2, 0.7, 0.1, 1.0]])
    mask = torch.ones_like(new_log_probs)

    loss, advantages, mean_kl = grpo_loss(new_log_probs, old_log_probs, reference_log_probs, rewards, mask)
    loss.backward()
    print(loss, advantages, mean_kl)
