import torch


def trajectory_mean(value, mask=None):
    if mask is None:
        return value.mean()
    length = mask.sum(dim=-1)
    per_trajectory = (value * mask).sum(dim=-1) / length.clamp_min(1)
    valid = length > 0
    return (per_trajectory * valid).sum() / valid.sum().clamp_min(1)


def x_opd_loss(current_text_logps, current_speech_logps, old_text_logps,
               old_speech_logps, teacher_text_logps, mask=None, lambda_im=0.5):
    """
    所有 logps: [B, M, L]，M 是每个 prompt 的 rollout 数。
    teacher_text_logps 始终条件于与 speech 语义对齐的 text prompt。
    返回负的 policy objective，用于常规梯度下降优化器。
    """
    old_text_logps = old_text_logps.detach()
    old_speech_logps = old_speech_logps.detach()
    teacher_text_logps = teacher_text_logps.detach()

    in_modal_advantage = (teacher_text_logps - old_text_logps).detach()
    cross_modal_advantage = (teacher_text_logps - old_speech_logps).detach()
    text_ratio = torch.exp(current_text_logps - old_text_logps)
    speech_ratio = torch.exp(current_speech_logps - old_speech_logps)

    in_modal_objective = trajectory_mean(text_ratio * in_modal_advantage, mask)
    cross_modal_objective = trajectory_mean(speech_ratio * cross_modal_advantage, mask)
    loss = -(lambda_im * in_modal_objective + (1 - lambda_im) * cross_modal_objective)
    return loss, in_modal_objective.detach(), cross_modal_objective.detach()


if __name__ == "__main__":
    torch.manual_seed(0)
    old_text_logps = torch.randn(2, 4, 6)
    old_speech_logps = torch.randn(2, 4, 6)
    teacher_text_logps = torch.randn(2, 4, 6)
    current_text_logps = old_text_logps.clone().requires_grad_()
    current_speech_logps = old_speech_logps.clone().requires_grad_()
    mask = torch.ones(2, 4, 6)

    loss, in_modal_objective, cross_modal_objective = x_opd_loss(
        current_text_logps, current_speech_logps, old_text_logps,
        old_speech_logps, teacher_text_logps, mask, lambda_im=0.5
    )
    loss.backward()
    print(loss, in_modal_objective, cross_modal_objective)
