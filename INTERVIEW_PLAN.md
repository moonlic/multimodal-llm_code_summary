# 面试手撕路线

目标不是堆积完整训练框架，而是对每个常考点保留一份可以在面试中解释和手写的最小实现。

## 第一批：面试主干

### 损失函数

- [x] `基础知识/cross_entropy.py`：从 `log_softmax` 手写交叉熵，支持 mask 和 label smoothing
- [x] `基础知识/kl_divergence.py`：forward KL、reverse KL 和 temperature
- [x] `LLM后训练/dpo.py`：chosen/rejected 偏好损失
- [x] `LLM后训练/grpo.py`：group advantage、ratio clipping 和 KL penalty
- [x] `LLM后训练/opd.py`：学生 rollout 上的 on-policy distillation
- [x] `LLM后训练/x_opd.py`：in-modal/cross-modal 双优势目标

### 结构与训练

- [x] `基础知识/moe.py`：Top-k Router、Experts 和负载均衡损失
- [x] `基础知识/multihead_attention_kv_cache.py`：Prefill、Decode 与 KV cache
- [x] `LLM后训练/lora_finetune.py`：LoRA 线性层与只更新低秩参数的训练

## 第二批：常见延伸

- [ ] RoPE 与长度外推
- [ ] SwiGLU 与完整 Transformer Block
- [ ] QLoRA 与 4-bit 量化要点
- [ ] SFT loss mask 与数据 collator
- [ ] Reward Model 的 pairwise ranking loss
- [ ] Speculative Decoding 最小实现

## 代码约定

1. 每个文件独立可运行，文件末尾保留 smoke test。
2. 代码优先表达面试核心公式，不引入分布式框架或业务训练配置。
3. 简单操作保持紧凑，只在张量变形或公式分段时换行。
4. 关键张量写明 shape，mask 中 `1/True` 表示有效 token。
