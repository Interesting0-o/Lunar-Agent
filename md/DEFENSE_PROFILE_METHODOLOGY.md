# Defense Profile 方法论与独立性审计

> 如何为 State Engine 增加新的防御维度——从理论验证到代码集成。
>
> **整合说明**：本文档合并了 DEFENSE_PROFILE_METHODOLOGY.md（扩展流程）+ DEFENSE_PROFILE_INDEPENDENCE_AUDIT.md（独立性审计）。方法论提供标准化流程，审计提供当前实现的独立性验证和对修复效果的评估。

---

## 第一部分：扩展方法论

### 背景

State Engine 的防御剖面 (`state_engine/_defenses.py`) 当前为 **二维 (2,7)** 结构：

| 维度 | 名称 | 作用位置 | 心理学对应 |
|------|------|---------|-----------|
| `profiles[0,:]` | **Deactivation** (去激活) | outer 削减 | Bowlby 去激活策略；高回避的典型防御 |
| `profiles[1,:]` | **Hyperactivation** (过度激活) | inner 放大 | Bowlby 过度激活策略；高焦虑的典型防御 |

每个维度是一个 7 维向量，对应 7 种心理刺激类型 (`ST_ABANDONMENT` … `ST_EMOTIONAL_WEIGHT`)，值域 [0,1]。

二维设计基于：
- **Bowlby (1980)**: 依恋防御二分法——去激活 (deactivating) vs 过度激活 (hyperactivating)
- **Brennan, Clark & Shaver (1998)**: 依恋焦虑和依恋回避的因子分析正交性
- **Richardson, Beath & Boag (2023)**: 依恋焦虑→过度激活防御，依恋回避→去激活防御，判别效度成立
- **Richardson et al. (2025)**: ADQ-50 量表 10 因子结构清晰分为过度激活/去激活两类

### 扩展策略：三步骤

```
Step 1: 概念定义 → Step 2: 参数扫描实验 → Step 3: 独立性检验 → 判定
```

#### Step 1 — 概念定义

新维度必须回答三个问题：

**1a. 心理学构念是什么？**

在学术文献中找到对应概念，说明它为什么是独立构念而非已有维度的别名。需要引用至少一篇实证文献。

检查清单：
- [ ] 是否对应 Bowlby、Vaillant、Richardson 等框架中的某个独立防御类别？
- [ ] 是否有经过因子分析验证的测量工具（如 DSQ、ADQ、DMRS）？
- [ ] 与 deactivation/hyperactivation 的理论关系是正交、相关但独立、还是从属？

**1b. 在 Lunar 流水线中作用在哪？**

防御剖面在 `apply_defenses` 中的数学操作决定了一个维度的唯一性：

```python
# 当前两维度的数学操作
inner[s]  = stimuli[s] × (1 + hyperactivation[s] × gain)
outer[s]  = inner[s] × (1 − deactivation[s] × cut)
```

新维度必须有**不同于现有维度的数学操作**。允许的操作：

| 位置 | 操作 | 示例 |
|------|------|------|
| 刺激进入前 | 过滤 (乘法衰减 stimuli) | boundary: `stimuli'[s] = stimuli[s] × (1 − boundary[s])` |
| inner 形成时 | 放大/缩小 (乘到 inner) | hyperactivation (已有) |
| inner→outer 时 | 削减/泄漏 (乘到 outer) | deactivation (已有) |
| outer 形成后 | 偏置 (加性偏移) | `outer'[s] = outer[s] + bias[s]` — 特殊场景 |
| 跨维度重映射 | 混合矩阵 | `stimuli → 不同的 inner 维度` — 复杂，需充分理由 |

**反例**: 如果一个"新"维度的操作是 `outer[s] *= (1 − new_profile[s])`，那它和 deactivation 做的是同一件事——应合并而非新增。

**1c. 由哪些参数驱动？**

写出剖面计算公式，明确：
- **人格特质** (10 维 Traits): 哪些特质的偏离如何影响该剖面
- **关系状态** (6 维 Relationship): 哪些关系维度如何调制
- **内部状态** (8 维 Internal): 哪些内部情绪如何推动
- **基线值**: 每种刺激类型的默认值

每个系数的符号和量级需要有心理学理由。

#### Step 2 — 参数扫描实验

生成 N ≥ 1000 组随机参数组合，计算三个剖面（deact、hyper、新维度），用于后续统计检验。

```python
import numpy as np
from state_engine._defenses import compute_defense_profiles
from state import (DEFAULT_TRAITS, DEFAULT_INTERNAL, DEFAULT_RELATIONSHIP,
                   ST_SIZE, T_SIZE, I_SIZE, R_SIZE)

n_samples = 2000
np.random.seed(42)

param_samples = []
for _ in range(n_samples):
    traits = np.clip(np.random.normal(0.5, 0.25, T_SIZE), 0.05, 0.95)
    relationship = np.clip(np.random.normal(0.4, 0.25, R_SIZE), 0.02, 0.98)
    internal = np.clip(np.random.normal(0.4, 0.25, I_SIZE), 0.05, 0.95)
    param_samples.append((traits, relationship, internal))

deact_all = np.zeros((n_samples, ST_SIZE))
hyper_all = np.zeros((n_samples, ST_SIZE))
candidate_all = np.zeros((n_samples, ST_SIZE))

for i, (t, r, n) in enumerate(param_samples):
    p = compute_defense_profiles(t, r, n)
    deact_all[i] = p[0]
    hyper_all[i] = p[1]
    candidate_all[i] = my_candidate_profile(t, r, n)
```

#### Step 3 — 独立性检验

**3a. 逐维度 Pearson 相关系数**

```python
for s in range(ST_SIZE):
    r_dh = np.corrcoef(deact_all[:, s], hyper_all[:, s])[0, 1]
    r_dc = np.corrcoef(deact_all[:, s], candidate_all[:, s])[0, 1]
    r_hc = np.corrcoef(hyper_all[:, s], candidate_all[:, s])[0, 1]
```

**3b. 多元回归**

```python
for s in range(ST_SIZE):
    X = np.column_stack([deact_all[:, s], hyper_all[:, s], np.ones(n_samples)])
    coeffs = np.linalg.lstsq(X, candidate_all[:, s], rcond=None)[0]
    predicted = X @ coeffs
    ss_res = np.sum((candidate_all[:, s] - predicted) ** 2)
    ss_tot = np.sum((candidate_all[:, s] - candidate_all[:, s].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    independent_var_pct = max(0, 1 - r2) * 100
```

**这是最关键的检验**: R² 越接近 1，说明候选维度完全可被已有维度线性表示，不值得独立。

### 判定标准

| 条件 | 结论 |
|------|------|
| max(\|r_dc\|, \|r_hc\|) < 0.3 **且** 全局独立方差 > 25% | ✅ **独立维度成立** → 扩展到 (3,7) |
| max(\|r_dc\|, \|r_hc\|) < 0.5 **且** 全局独立方差 > 15% | ⚠️ **边界情况** → 需要进一步分析 |
| max(\|r_dc\|, \|r_hc\|) ≥ 0.5 **或** 全局独立方差 < 15% | ❌ **不成立** → 合并进已有维度 |

### 案例研究：Boundary (边界感)

**心理学构念：** 自我-他人心理边界的渗透性。在 Bowlby 框架中，边界感不是独立的防御类型，而是去激活/过度激活在不同刺激类型上的差异化表现。

**作用位置：** 刺激输入过滤 —— `stimuli'[s] = stimuli[s] × (1 − boundary[s] × 0.5)`

**Step 3 结果：**

| 刺激维度 | deact×hyper | deact×bound | hyper×bound | 独立方差% |
|---------|-------------|-------------|-------------|----------|
| abandonment | +0.005 | +0.409 | **-0.457** | **62.2%** |
| validation | -0.208 | +0.156 | **-0.906** | 17.8% |
| closeness | -0.327 | +0.373 | **-0.857** | 25.6% |
| conflict | -0.205 | **+0.549** | -0.512 | 53.2% |
| dependency | -0.330 | **+0.502** | -0.623 | 51.4% |
| teasing | -0.240 | +0.496 | -0.636 | 47.1% |
| emotional_weight | -0.220 | +0.274 | **-0.705** | 48.8% |
| **全局均值** | **-0.218** | **+0.394** | **-0.671** | 43.7% |

**判定：** ❌ 不作为独立第 3 维。`|r_hb| = 0.671` 与 hyperactivation 高度负相关，全局 R²=0.592 不满足独立性标准。Boundary 的核心语义已编码在 deactivation 的逐维度基线中。

### 当前 (2,7) 剖面参数速查

**Deactivation (去激活)**

公式：`deact[s] = σ(5.0 × (raw[s] - 0.35))`, `raw = baseline + trait_dev + global_mod + rel_mod + internal_push`

| 刺激维度 | 基线 | 主导特质 (+) | 主导特质 (−) | 关系调制 | 内部推动 |
|---------|------|-------------|-------------|---------|---------|
| abandonment | 0.30 | PRIDE(0.45), JEALOUSY(0.18) | — | TRUST, SAFETY (−) | STRESS, INSECURITY (+) |
| validation | 0.25 | PRIDE(0.40) | — | 同上 | 同上 |
| closeness | 0.15 | PRIDE(0.18), AVOIDANCE(0.15) | — | 同上 | 同上 |
| conflict | 0.20 | PRIDE(0.28), ANGER(0.22) | — | 同上 | 同上 |
| teasing | 0.20 | PRIDE(0.32) | — | 同上 | 同上 |
| dependency | 0.28 | PRIDE(0.38), AVOIDANCE(0.20) | — | 同上 | 同上 |
| emotional_weight | 0.25 | PRIDE(0.28) | — | 同上 | 同上 |

全局调制：`−STABILITY×0.22`, `−OPENNESS×0.12`, `+AVOIDANCE×0.18`

**Hyperactivation (过度激活)**

公式：`hyper[s] = σ(5.0 × (raw[s] - 0.38))`, `raw = baseline + trait_dev + global_mod + rel_mod + internal_push`

| 刺激维度 | 基线 | 主导特质 (+) | 主导特质 (−) | 关系调制 | 内部推动 |
|---------|------|-------------|-------------|---------|---------|
| abandonment | 0.45 | ATTACH_ANXIETY(0.55), JEALOUSY(0.30) | — | AFFECTION, ROMANTIC_TENSION (+) | INSECURITY, LONGING (+) |
| closeness | 0.30 | ATTACH_ANXIETY(0.50) | — | 同上 | 同上 |
| dependency | 0.35 | ATTACH_ANXIETY(0.40) | — | 同上 | 同上 |
| validation | 0.15 | ATTACH_ANXIETY(0.20) | — | 同上 | 同上 |
| conflict | 0.15 | ATTACH_ANXIETY(0.30) | — | 同上 | 同上 |
| teasing | 0.10 | JEALOUSY(0.20) | — | 同上 | 同上 |
| emotional_weight | 0.20 | ATTACH_ANXIETY(0.30) | — | 同上 | 同上 |

全局调制：`+SENSITIVITY×0.08`, `−AVOIDANCE×0.30`

---

## 第二部分：独立性审计

> 对 `compute_defense_profiles` 的逐维权重设计进行 PCA + 方差分解审计，验证修复效果并揭示根因。

### 审计方法

```python
n = 100,000  # 随机状态样本
# 配置 A: 随机 traits (uniform[-1,1])  — 原始方法
# 配置 B: 固定 DEFAULT_TRAITS          — 系统实际运行时的场景
traits = np.tile(DEFAULT_TRAITS, (n, 1))
internal = rng.uniform(-1, 1, (n, 8))
relationship = rng.uniform(-1, 1, (n, 6))
# 对每组样本调用 compute_defense_profiles(traits[i], relationship[i], internal[i])
# PCA: 中心化 → SVD → 方差分解 + 独立方差（1 - R²）
```

### 核心发现：固定 traits 后独立方差接近零

#### 审计 A vs B 对比

| 指标 | 修复前（报告） | 随机 traits（审计 A） | 固定 traits（审计 B） |
|------|:------------:|:------------------:|:------------------:|
| **去激活 PC1** | ~99% | 83.5% | **90.4%** |
| **过度激活 PC1** | ~99% | 92.4% | **95.2%** |
| **去激活有效秩** | ~1.0 | 1.93 | **1.40** |
| **过度激活有效秩** | ~1.0 | 1.41 | **1.26** |
| **去激活独立方差均值** | 0.0% | 8.4% | **~0.0%** |
| **过度激活独立方差均值** | 0.0% | 7.9% | **~0.2%** |

**结论：使用随机 traits 会高估修复效果。** 随机 trait 制造了 7 维间因不同 trait 组合而产生的"伪独立性"，而系统运行中 traits 是静态的。

#### 根源：信息瓶颈

固定 traits 后，每个剖面 7 维的**所有动态变化**来自仅 2-4 个标量输入源：

| 剖面 | 输入源 | 独立自由度 | 输出维 | 瓶颈 |
|------|-------|:--------:|:-----:|:----:|
| 去激活 | `trust_bond`（1 个 rel）+ `stress` / `insecurity`（2 个 int） | **3** | 7 | 3→7 |
| 过度激活 | `affection` / `intimacy`（2 个 rel）+ `insecurity` / `longing`（2 个 int） | **4** | 7 | 4→7 |

验证实验——仅变化单一信号源：

| 实验 | PC1 | 有效秩 | max\|r\| | 独立方差 |
|:----|:---:|:-----:|:-------:|:-------:|
| 仅 relationship 变化 | **100.0%** | **1.00** | **1.0000** | 0.0% |
| 仅 internal 变化 | 90.6% | 1.37 | 0.9995 | 0.0% |
| 全随机（固定 traits） | 90.4% | 1.40 | 0.9829 | 0.0% |

**"仅 rel"时 PC1=100%**：因为去激活的关系态调制仅由 `trust_bond` 一个标量驱动。7 个输出是同一标量乘以不同系数的线性组合——完美共线。

#### Pre-sigmoid 动态范围

sigmoid 前的原始线性值显示各维动态范围差异巨大：

| 刺激维度 | pre-sigmoid 均值 | 动态范围 | 退化程度 |
|----------|:--------------:|:--------:|:--------:|
| `abandonment` | 0.4036 | **0.479** | ✅ 正常 |
| `conflict` | 0.2423 | 0.344 | ✅ |
| `emotional_weight` | 0.2922 | 0.336 | ✅ |
| `dependency` | 0.2770 | 0.281 | ✅ |
| `validation` | 0.3101 | 0.248 | 🟡 偏窄 |
| `closeness` | 0.1320 | 0.060 | 🔴 退化 |
| `teasing` | 0.2480 | **0.015** | 🔴 **接近常量** |

TEASING range=0.015 的原因：它对 stress 和 insecurity 的权重均为 0.0，trust_bond 权重仅 -0.03。固定 traits 后没有任何有效输入驱动它。

### 调制强度 vs 独立性（放大实验）

| 调制强度 | 去激活 PC1 | 去激活独立方差 | 过度激活 PC1 | 过度激活独立方差 |
|:-------:|:---------:|:-------------:|:-----------:|:---------------:|
| ×1（当前） | 90.4% | 0.0% | 95.2% | 0.2% |
| ×3 | 89.3% | 0.8% | 91.0% | 1.9% |
| ×5 | 87.8% | 3.0% | 89.1% | 4.8% |
| ×10 | **81.5%** | **15.8%** | 86.7% | **10.8%** |

即使调制强度放大 10 倍，独立方差也只到 ~16%。且放大后 sigmoid 两端饱和，profile 值趋向 {0, 1} 二值化。**放大调制系数不是可行修复路径。**

### 与 REL_INPUT_INFLUENCE_B 的同构性

防御剖面的独立性问题和关系态的维度冗余问题是**同一个模式**：

```
关系态:
  7 维刺激 → B 矩阵密集映射 → 6 维关系态同步运动

防御剖面:
  2-3 个状态变量 → 逐维权重数组 → 7 维剖面同步运动
```

两者的根因相同：**输入信号维度远少于输出维度**。无论权重数组设计得多精细，只要输入源个数 < 输出维数，输出的有效自由度就受限于输入自由度。

### 可选的修复路径

#### 路径 A：增加输入源维度（短期）⭐ 推荐

当前去激活调制用了 3 个输入（trust_bond、stress、insecurity），但关系态有 6 维、内部态有 8 维：

| 新增输入 | 应用剖面 | 心理学依据 |
|---------|--------|-----------|
| `R_INTIMACY` | 去激活 | 暧昧气氛下防御会变化 |
| `R_AFFECTION` | 去激活 | 好感降低伪装需求（安全感） |
| `I_LONELINESS` | 去激活 | 孤独时更渴望联结→降低防御 |
| `I_ENERGY` | 去激活 | 精力充沛时更敢于面对冲突 |
| `I_MENTAL_FATIGUE` | 过度激活 | 疲劳时对关系信号的过度反应 |

#### 路径 B：Trait 演化（中期）

一旦 traits 开始更新，每个剖面就有了 10 个额外的自由度来源。但防御剖面对 traits 的依赖本身是线性且简单（每维仅 1-2 个 trait），真正的改善需要**耦合系数在 trait 空间也具备维度特异性**。

#### 路径 C：语义合并输出维度（远期）

考虑 3 类聚合：威胁类（abandonment + conflict + emotional_weight）、亲近类（closeness + dependency）、社交类（validation + teasing），每个聚合类共享一个防御值。**注意这需要改变 `apply_defenses` 的逐元素乘法语义。**

### 各维度独立方差明细

**固定 traits + 随机 internal/rel：**

| 刺激维度 | 去激活独立方差 | 过度激活独立方差 |
|----------|:------------:|:---------------:|
| `abandonment` | 0.0% | 0.7% |
| `validation` | 0.0% | 0.0% |
| `closeness` | 0.0% | 0.2% |
| `conflict` | 0.0% | 0.0% |
| `dependency` | 0.0% | 0.5% |
| `teasing` | 0.1% | 0.1% |
| `emotional_weight` | 0.0% | 0.0% |

**审计代码：** `tools/audit_defenses.py`。运行：`uv run python tools/audit_defenses.py`

---

## 参考文献

1. **Bowlby, J.** (1980). *Attachment and Loss, Vol. 3: Loss, Sadness and Depression*. — 去激活/过度激活二分法
2. **Brennan, K. A., Clark, C. L., & Shaver, P. R.** (1998). Self-report measurement of adult attachment. — 依恋焦虑/回避的因子分析正交性
3. **Richardson, E., Beath, A., & Boag, S.** (2023). Default defenses: the character defenses of attachment-anxiety and attachment-avoidance. *Current Psychology, 42*, 28755–28770.
4. **Richardson, E., Beath, A., & Boag, S.** (2025). The Development of the Attachment Defenses Questionnaire (ADQ-50). *Journal of Personality Assessment, 107*(1), 58–72.
5. **Girme, Y. U., et al.** (2021). Attachment Anxiety and the Curvilinear Effects of Expressive Suppression. *Journal of Personality and Social Psychology*.
6. **Vaillant, G. E.** (1986). *Ego Mechanisms of Defense: A Guide for Clinicians and Researchers*.
7. **Di Giuseppe, M., & Tanzilli, A.** (2025). Defenses and Attachment in Clinical Practice. *Journal of Personality Assessment, 107*(1), 140–141.
