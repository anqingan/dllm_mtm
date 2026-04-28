# CHANGES.md — dllm 项目改动记录

## 2026-04-25: 核心采样器新增 6 种解掩码策略

### 概述

在 `dllm/core/samplers/` 中的 MDLMSampler 和 BD3LMSampler 两个核心采样器中，新增了 6 种解掩码（unmasking）策略，通过 `remasking` 配置参数选择。原有的 `"low_confidence"` 和 `"random"` 策略行为不变，完全向后兼容。

### 新增策略

| # | 名称 | `remasking` 值 | 描述 |
|---|------|----------------|------|
| 1 | Left-to-Right | `"left_to_right"` | 按位置索引从小到大选取 k 个被掩码位置解掩码。k=1 时等同自回归生成 |
| 2 | Greedy Confidence | `"greedy_confidence"` | 选择预测置信度最高的 k 个位置（与 `low_confidence` 行为一致的显式别名） |
| 3 | Probability Margin | `"probability_margin"` | 选择 Top-1 与 Top-2 概率差最大的 k 个位置，追求"真正的确定性" |
| 4 | Confidence Threshold | `"confidence_threshold"` | 自适应策略：解掩码所有置信度 >= 阈值的位置；无满足时回退到最自信的 1 个位置 |
| 5 | KLASS | `"klass"` | 同时要求置信度 >= 阈值 **且** KL 散度 <= 稳定性阈值；带回退机制 |
| 6 | Oracle | `"oracle"` | 穷举块内所有排列找最优解掩码顺序（仅限小规模，超出自动回退到 greedy） |

### 修改的文件

#### 1. `dllm/core/samplers/utils.py`
**新增函数：**
- `compute_confidence_scores(logits, x0, remasking)` — 统一的置信度计算入口，支持所有 8 种策略（含原有的 low_confidence 和 random）
- `select_transfer_positions(confidence, mask_index, num_transfer_tokens, remasking, ...)` — 统一的位置选择逻辑，按策略分发到不同的选择算法
- `compute_kl_divergence(prev_probs, curr_probs)` — 计算 KL(prev || curr) 逐位置的 KL 散度，供 KLASS 策略使用
- `oracle_block_enumerate(x, block_start, block_len, model, ...)` — Oracle 策略的块内穷举实现，枚举所有排列并评估 NLL

**原有函数不变：**
- `get_num_transfer_tokens()` — 未修改
- `add_gumbel_noise()` — 未修改

#### 2. `dllm/core/samplers/mdlm.py`
**`MDLMSamplerConfig` 新增字段：**
- `threshold: float | None = None` — Confidence Threshold 和 KLASS 的置信度阈值
- `kl_threshold: float | None = None` — KLASS 的 KL 散度上限 ν
- `oracle_max_positions: int = 5` — Oracle 策略的最大枚举位置数

**`sample()` 方法改动：**
- 替换内联的置信度计算（原 lines 196-207）→ 调用 `compute_confidence_scores()`
- 替换内联的位置选择循环（原 lines 219-228）→ 调用 `select_transfer_positions()`
- 新增 KLASS 状态变量 `prev_probs`，每个新块重置为 None
- 新增 Oracle 特殊分支：当 B=1 且块内掩码数 <= oracle_max_positions 时调用 `oracle_block_enumerate()`
- 从 kwargs 提取新参数：`threshold`, `kl_threshold`, `oracle_max_positions`

**`infill()` 方法改动：**
- 同 sample() 的改造模式：使用 `compute_confidence_scores()` + `select_transfer_positions()`
- 新增 KLASS 状态管理

#### 3. `dllm/core/samplers/bd3lm.py`
**`BD3LMSamplerConfig` 新增字段：** 同 MDLMSamplerConfig

**`_diffusion_step_block()` 函数改动：**
- 签名扩展：新增 `threshold`, `kl_threshold`, `prev_block_probs` 参数
- 返回类型从 `torch.Tensor` 改为 `tuple[torch.Tensor, torch.Tensor | None]`（返回更新后的块和当前概率分布）
- 内部使用 `compute_confidence_scores()` 和 `select_transfer_positions()` 替换原有硬编码逻辑

**`sample()` 方法改动：**
- 传递新参数给 `_diffusion_step_block()`
- 新增 `prev_block_probs` 状态管理（KLASS 用，每个新块重置）
- 从 kwargs 提取新参数

#### 4. `dllm/core/samplers/__init__.py`
- 新增导出：`compute_confidence_scores`, `compute_kl_divergence`, `oracle_block_enumerate`, `select_transfer_positions`

### 使用示例

```python
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig

# Left-to-Right (自回归式)
config = MDLMSamplerConfig(remasking="left_to_right")

# Probability Margin
config = MDLMSamplerConfig(remasking="probability_margin")

# Confidence Threshold (自适应并行)
config = MDLMSamplerConfig(remasking="confidence_threshold", threshold=0.95)

# KLASS (置信度 + 稳定性双重约束)
config = MDLMSamplerConfig(
    remasking="klass",
    threshold=0.95,   # μ: 置信度阈值
    kl_threshold=0.1, # ν: KL 散度上限
)

# Oracle (性能上限评估，仅限小块)
config = MDLMSamplerConfig(
    remasking="oracle",
    oracle_max_positions=5,
    block_size=5,  # 建议 block_size <= oracle_max_positions
)

# BD3LM 同样支持所有新策略
from dllm.core.samplers import BD3LMSampler, BD3LMSamplerConfig
config = BD3LMSamplerConfig(remasking="klass", threshold=0.9, kl_threshold=0.05)
```

### 向后兼容性

- 所有新配置字段有默认值（`None` 或 `5`），不影响现有代码
- 原有的 `"low_confidence"` 和 `"random"` 策略行为完全不变
- `greedy_confidence` 是 `low_confidence` 的别名，两者产生相同结果

### 性能注意事项

1. **KLASS 内存开销**：需要存储上一步的完整概率分布 `[B, L, V]`。对于大词表模型（如 128K vocab），内存占用较高。建议在长序列场景下注意 GPU 显存。
2. **Oracle 计算开销**：枚举 n! 种排列，每种需要 n 次前向传播。默认 `oracle_max_positions=5`（120 种排列 × 5 次前向 = 600 次）。仅适用于研究和性能上限评估，不建议用于生产推理。
3. **Confidence Threshold 自适应性**：该策略的解掩码数量不固定，可能导致实际步数与预设步数不一致。这是设计预期的行为。
