import torch
import torch.nn.functional as F


def dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps,
             reference_rejected_logps, beta=0.1):
    """
    四个输入均为 [B]，表示 response token log-probability 之和。
    """
    policy_log_ratio = policy_chosen_logps - policy_rejected_logps
    reference_log_ratio = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_log_ratio - reference_log_ratio)
    loss = -F.logsigmoid(logits).mean()
    chosen_reward = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_reward = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return loss, chosen_reward, rejected_reward


if __name__ == "__main__":
    policy_chosen = torch.tensor([-2.0, -1.5], requires_grad=True)
    policy_rejected = torch.tensor([-3.0, -2.5], requires_grad=True)
    reference_chosen = torch.tensor([-2.4, -2.0])
    reference_rejected = torch.tensor([-2.8, -2.3])

    loss, chosen_reward, rejected_reward = dpo_loss(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected
    )
    loss.backward()
    print(loss, chosen_reward, rejected_reward)
