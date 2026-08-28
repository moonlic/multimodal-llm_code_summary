import torch
import torch.nn.functional as F


def opd_loss(student_logits, teacher_logits, mask=None, temperature=1.0, reverse_kl=True):
    """
    student_logits, teacher_logits: [B, L, V]
    这些 prefix 应来自学生自己的 on-policy rollout。
    """
    teacher_logits = teacher_logits.detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_probs, teacher_probs = student_log_probs.exp(), teacher_log_probs.exp()

    if reverse_kl:
        token_loss = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    else:
        token_loss = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)

    token_loss = token_loss * temperature ** 2
    if mask is None:
        return token_loss.mean()
    return (token_loss * mask).sum() / mask.sum().clamp_min(1)


if __name__ == "__main__":
    student_logits = torch.randn(2, 8, 32, requires_grad=True)
    teacher_logits = torch.randn(2, 8, 32)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0]])

    loss = opd_loss(student_logits, teacher_logits, mask, reverse_kl=True)
    loss.backward()
    print(loss)
