# Multimodal LLM Code Summary

面向多模态大模型学习与求职复习的代码、原理笔记和最小可运行实现。

本仓库只整理公开知识、开源论文与独立编写的教学代码。代码示例将优先使用合成数据或公开数据集，并给出必要的张量形状与最小测试。

## Contents

### 01. 基础知识

- [ ] Transformer：Self-Attention、FFN、残差连接与归一化
- [ ] 位置编码：RoPE、长度外推与多模态位置编码
- [ ] 注意力变体：MHA、MQA、GQA
- [ ] MoE：Router、Top-k Expert 与负载均衡
- [ ] 训练基础：交叉熵、KL 散度、梯度累积与混合精度
- [ ] 推理基础：Prefill、Decode、KV Cache 与吞吐/延迟指标

### 02. LLM 后训练

- [ ] SFT：数据格式、Loss Mask 与冷启动
- [ ] LoRA / QLoRA：低秩适配原理与最小训练示例
- [ ] DPO：偏好优化目标与实现
- [ ] GRPO / GSPO：Group Advantage、Token/Sequence Importance Sampling
- [ ] Reward Design：格式、正确性与过程奖励
- [ ] X-OPD：跨模态在线策略蒸馏的目标函数与教学实现
- [ ] Rollout 分析、Checkpoint 选择与训练稳定性

### 03. 数据生产与质量

- [ ] 原始数据清洗、去重、格式统一与来源追踪
- [ ] 指令改写、难例构造与分布控制
- [ ] Rubric 设计与 LLM-as-Judge
- [ ] 多 Judge 复核、冲突仲裁与一致性分析
- [ ] Bad Case 聚类、错误归因与数据回流
- [ ] 训练集、验证集与固定评测集的隔离

### 04. VLM

- [ ] Vision Encoder、Projector 与视觉 Token 注入
- [ ] 图像/视频预处理、动态分辨率与 Token 预算
- [ ] 多模态 SFT 与 Loss Mask
- [ ] 视觉指令遵循、空间关系与结构化规划
- [ ] 多模态 Reward 与评测协议
- [ ] VLM Serving 与多模态输入链路

### 05. 推理加速

- [ ] Continuous Batching、Paged Attention 与 Prefix Cache
- [ ] Tensor / Pipeline / Expert Parallel
- [ ] Quantization 与服务性能评测
- [ ] Speculative Decoding 基础
- [ ] EAGLE 类 Draft Model
- [ ] 接受长度、Target Query、吞吐与端点质量的联合评测

### 06. RAG

- [ ] 文档解析、切分与元数据设计
- [ ] Parent-Child Chunking：父块检索与子块定位
- [ ] BM25 稀疏检索
- [ ] Dense Embedding 向量检索
- [ ] BM25 + Dense 混合召回与 RRF
- [ ] Cross-Encoder Reranker
- [ ] 面向检索/生成的 LoRA 训练
- [ ] Recall、MRR、NDCG、Faithfulness 与端到端评测

## Papers and Research

### MEPG: Multi-Expert Planning and Generation

第一作者工作，研究复杂组合文生图中的 LLM 规划、结构化布局与多专家生成。

- Paper: [MEPG: Multi-Expert Planning and Generation for Compositionally-Rich Image Generation](https://arxiv.org/abs/2509.04126)
- Repository notes and a public-safe minimal reproduction will be added progressively.

### DCR

关于投机解码中选择性验证的匿名在审研究。

> 为遵守双盲审阅与信息披露要求，当前仅保留高层主题说明，不公开作者、单位、完整标题、投稿信息、实验数字、论文 PDF 或实现细节。相关内容将在匿名与披露限制解除后更新。

## Repository Principles

1. **Public knowledge only**：仅使用论文、官方文档、开源项目与独立推导。
2. **Minimal runnable code**：每个实现尽量包含输入输出、张量形状和 smoke test。
3. **Evidence over claims**：区分原理推导、复现实验、测量结果与理论推断。
4. **No business content**：不上传业务数据、内部代码、内部文档、内部指标或环境路径。
5. **Incremental maintenance**：只将已经理解、实现或验证的条目标记为完成。

## Status

Repository initialized. Topics and minimal implementations will be added incrementally.
