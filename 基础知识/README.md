# 基础积木热身

本页内容用于快速热身，不代表面试复习的主优先级。面试主线见根目录 `INTERVIEW_PLAN.md`。

## 已完成热身

- [x] `scaled_dot_product_attention.py`：缩放点积注意力与 mask
- [x] `self_attention.py`：单头 Self-Attention
- [x] `feed_forward.py`：Transformer 中的两层 FFN
- [x] `layer_norm.py`：手写 LayerNorm
- [x] `rms_norm.py`：手写 RMSNorm
- [x] `sinusoidal_position_encoding.py`：正弦位置编码

## 代码风格

1. 尽量一行表达一个完整操作，不对简单函数调用频繁换行。
2. 关键张量保留 shape 注释。
3. mask 中 `1/True` 表示可以关注，`0/False` 表示屏蔽。
4. 每个文件都可以直接运行。
