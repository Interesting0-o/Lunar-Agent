# Defense Profile 扩展方法论

> 如何为 State Engine 增加新的防御维度——从理论验证到代码集成。

---

## 背景

State Engine 的防御剖面 (`state_engine/_defenses.py`) 当前为 **二维 (2,7)** 结构:

| 维度 | 名称 | 作用位置 | 心理学对应 |
|------|------|---------|-----------|
| `profiles[0,:]` | **Deactivation** (去激活) | outer 削减 | Bowlby 去激活策略；高回避的典型防御 |
| `profiles[1,:]` | **Hyperactivation** (过度激活) | inner 放大 | Bowlby 过度激活策略；高焦虑的典型防御 |

每个维度是一个 7 维向量，对应 7 种心理刺激类型 (`ST_ABANDONMENT` … `ST_EMOTIONAL_WEIGHT`)，值域 [0,1]。

二维设计基于:
- **Bowlby (1980)**: 依恋防御二分法 —— 去激活 (deactivating) vs 过度激活 (hyperactivating)
- **Brennan, Clark & Shaver (1998)**: 依恋焦虑和依恋回避的因子分析正交性
- **Richardson, Beath & Boag (2023)**: 依恋焦虑→过度激活防御，依恋回避→去激活防御，判别效度成立
- **Richardson et al. (2025)**: ADQ-50 量表 10 因子结构清晰分为过度激活/去激活两类

---

## 扩展策略: 三步骤

```
Step 1: 概念定义 → Step 2: 参数扫描实验 → Step 3: 独立性检验 → 判定
```

### Step 1 — 概念定义

新维度必须回答三个问题:

#### 1a. 心理学构念是什么？

在学术文献中找到对应概念，说明它为什么是独立构念而非已有维度的别名。需要引用至少一篇实证文献。

检查清单:
- [ ] 是否对应 Bowlby、Vaillant、Richardson 等框架中的某个独立防御类别？
- [ ] 是否有经过因子分析验证的测量工具（如 DSQ、ADQ、DMRS）？
- [ ] 与 deactivation/hyperactivation 的理论关系是正交、相关但独立、还是从属？

#### 1b. 在 Lunar 流水线中作用在哪？

防御剖面在 `apply_defenses` 中的数学操作决定了一个维度的唯一性:

```python
# 当前两维度的数学操作
inner[s]  = stimuli[s] × (1 + hyperactivation[s] × gain)
outer[s]  = inner[s] × (1 − deactivation[s] × cut)
```

新维度必须有**不同于现有维度的数学操作**。允许的操作:

| 位置 | 操作 | 示例 |
|------|------|------|
| 刺激进入前 | 过滤 (乘法衰减 stimuli) | boundary: `stimuli'[s] = stimuli[s] × (1 − boundary[s])` |
| inner 形成时 | 放大/缩小 (乘到 inner) | hyperactivation (已有) |
| inner→outer 时 | 削减/泄漏 (乘到 outer) | deactivation (已有) |
| outer 形成后 | 偏置 (加性偏移) | `outer'[s] = outer[s] + bias[s]` — 特殊场景 |
| 跨维度重映射 | 混合矩阵 | `stimuli → 不同的 inner 维度` — 复杂，需充分理由 |

**反例**: 如果一个"新"维度的操作是 `outer[s] *= (1 − new_profile[s])`，那它和 deactivation 做的是同一件事——应合并而非新增。

#### 1c. 由哪些参数驱动？

写出剖面计算公式，明确:
- **人格特质** (10 维 Traits): 哪些特质的偏离如何影响该剖面
- **关系状态** (6 维 Relationship): 哪些关系维度如何调制
- **内部状态** (8 维 Internal): 哪些内部情绪如何推动
- **基线值**: 每种刺激类型的默认值

每个系数的符号和量级需要有心理学理由。

---

### Step 2 — 参数扫描实验

生成 N ≥ 1000 组随机参数组合，计算三个剖面（deact、hyper、新维度），用于后续统计检验。

#### 实验代码模板

```python
import numpy as np
from state_engine._defenses import compute_defense_profiles
from state import (DEFAULT_TRAITS, DEFAULT_INTERNAL, DEFAULT_RELATIONSHIP,
                   ST_SIZE, T_SIZE, I_SIZE, R_SIZE)

# 生成参数组合
n_samples = 2000
np.random.seed(42)

param_samples = []
for _ in range(n_samples):
    traits = np.clip(np.random.normal(0.5, 0.25, T_SIZE), 0.05, 0.95)
    relationship = np.clip(np.random.normal(0.4, 0.25, R_SIZE), 0.02, 0.98)
    internal = np.clip(np.random.normal(0.4, 0.25, I_SIZE), 0.05, 0.95)
    param_samples.append((traits, relationship, internal))

# 计算三个剖面
deact_all = np.zeros((n_samples, ST_SIZE))
hyper_all = np.zeros((n_samples, ST_SIZE))
candidate_all = np.zeros((n_samples, ST_SIZE))

for i, (t, r, n) in enumerate(param_samples):
    p = compute_defense_profiles(t, r, n)
    deact_all[i] = p[0]
    hyper_all[i] = p[1]
    candidate_all[i] = my_candidate_profile(t, r, n)

# → 进入 Step 3 独立性检验
```

---

### Step 3 — 独立性检验

#### 3a. 逐维度 Pearson 相关系数

```python
for s in range(ST_SIZE):
    r_dh = np.corrcoef(deact_all[:, s], hyper_all[:, s])[0, 1]
    r_dc = np.corrcoef(deact_all[:, s], candidate_all[:, s])[0, 1]
    r_hc = np.corrcoef(hyper_all[:, s], candidate_all[:, s])[0, 1]
```

#### 3b. 多元回归: 候选维度能否被已有维度线性解释？

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

**这是最关键的检验**: R² 越接近 1，说明候选维度完全可被已有维度线性表示，不值得独立。独立方差百分比越高，越有新增价值。

---

### 判定标准

| 条件 | 结论 |
|------|------|
| max(\|r_dc\|, \|r_hc\|) < 0.3 **且** 全局独立方差 > 25% | ✅ **独立维度成立** → 扩展到 (3,7) |
| max(\|r_dc\|, \|r_hc\|) < 0.5 **且** 全局独立方差 > 15% | ⚠️ **边界情况** → 需要进一步分析逐刺激维度的独立方差分布 |
| max(\|r_dc\|, \|r_hc\|) ≥ 0.5 **或** 全局独立方差 < 15% | ❌ **不成立** → 合并进已有维度，或编码进特定维度的基线 |

---

## 案例研究: Boundary (边界感)

### Step 1 — 定义

**心理学构念**: 自我-他人心理边界的渗透性。在 Bowlby 框架中，边界感不是独立的防御类型，而是**去激活/过度激活在不同刺激类型上的差异化表现**——对 positive stimuli (被认可、亲密) 的 boundary 主要由依恋焦虑驱动（与 hyperactivation 高度负相关），对 threat stimuli (被抛弃、冲突) 的 boundary 主要由依恋回避驱动（与 deactivation 中等正相关）。

**作用位置**: 刺激输入过滤 —— `stimuli'[s] = stimuli[s] × (1 − boundary[s] × 0.5)`

### Step 2 — 实验 (N=2000)

### Step 3 — 结果

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

### 判定

- `|r_hb| = 0.671` — 与 hyperactivation 高度负相关（尤其在 validation/closeness 维度达 -0.91）
- 全局 R² = 0.592 — 59.2% 的方差可被 deact+hyper 解释
- 尽管 abandonment/conflict 维度有 >50% 独立方差，但**全局不满足独立性标准**

**结论**: ❌ 不作为独立第 3 维。Boundary 的核心语义已编码在 deactivation 的逐维度基线中（abandonment 基线 0.30，closeness 基线 0.15——正是 boundary 概念的体现）。如需增强 boundary 效应，调整 deact/hyper 的逐维度系数即可。

---

## 附录: 当前 (2,7) 剖面参数速查

### Deactivation (去激活)

公式: `deact[s] = σ(raw[s] - 0.48)`, `raw = baseline + trait_dev + global_mod + rel_mod + internal_push`

| 刺激维度 | 基线 | 主导特质 (+) | 主导特质 (−) | 关系调制 | 内部推动 |
|---------|------|-------------|-------------|---------|---------|
| abandonment | 0.30 | PRIDE(0.45), JEALOUSY(0.18) | — | TRUST, SAFETY (−) | STRESS, INSECURITY (+) |
| validation | 0.25 | PRIDE(0.40) | — | 同上 | 同上 |
| closeness | 0.15 | PRIDE(0.18), AVOIDANCE(0.15) | — | 同上 | 同上 |
| conflict | 0.20 | PRIDE(0.28), ANGER(0.22) | — | 同上 | 同上 |
| teasing | 0.20 | PRIDE(0.32) | — | 同上 | 同上 |
| dependency | 0.28 | PRIDE(0.38), AVOIDANCE(0.20) | — | 同上 | 同上 |
| emotional_weight | 0.25 | PRIDE(0.28) | — | 同上 | 同上 |

全局调制: `−STABILITY×0.22`, `−OPENNESS×0.12`, `+AVOIDANCE×0.18`

### Hyperactivation (过度激活)

公式: `hyper[s] = σ(raw[s] - 0.50)`, `raw = baseline + trait_dev + global_mod + rel_mod + internal_push`

| 刺激维度 | 基线 | 主导特质 (+) | 主导特质 (−) | 关系调制 | 内部推动 |
|---------|------|-------------|-------------|---------|---------|
| abandonment | 0.45 | ATTACH_ANXIETY(0.55), JEALOUSY(0.30) | — | AFFECTION, ROMANTIC_TENSION (+) | INSECURITY, LONGING (+) |
| closeness | 0.30 | ATTACH_ANXIETY(0.50) | — | 同上 | 同上 |
| dependency | 0.35 | ATTACH_ANXIETY(0.40) | — | 同上 | 同上 |
| validation | 0.15 | ATTACH_ANXIETY(0.20) | — | 同上 | 同上 |
| conflict | 0.15 | ATTACH_ANXIETY(0.30) | — | 同上 | 同上 |
| teasing | 0.10 | JEALOUSY(0.20) | — | 同上 | 同上 |
| emotional_weight | 0.20 | ATTACH_ANXIETY(0.30) | — | 同上 | 同上 |

全局调制: `+SENSITIVITY×0.08`, `−AVOIDANCE×0.30`

---

## 参考文献

1. **Bowlby, J.** (1980). *Attachment and Loss, Vol. 3: Loss, Sadness and Depression*. — 去激活/过度激活二分法
2. **Brennan, K. A., Clark, C. L., & Shaver, P. R.** (1998). Self-report measurement of adult attachment. — 依恋焦虑/回避的因子分析正交性
3. **Richardson, E., Beath, A., & Boag, S.** (2023). Default defenses: the character defenses of attachment-anxiety and attachment-avoidance. *Current Psychology, 42*, 28755–28770. — 焦虑→过度激活，回避→去激活，判别效度
4. **Richardson, E., Beath, A., & Boag, S.** (2025). The Development of the Attachment Defenses Questionnaire (ADQ-50). *Journal of Personality Assessment, 107*(1), 58–72. — 10 因子防御结构
5. **Girme, Y. U., et al.** (2021). Attachment Anxiety and the Curvilinear Effects of Expressive Suppression. *Journal of Personality and Social Psychology*. — 依恋焦虑×表达抑制的交互效应
6. **Vaillant, G. E.** (1986). *Ego Mechanisms of Defense: A Guide for Clinicians and Researchers*. — 防御机制四级层次
7. **Di Giuseppe, M., & Tanzilli, A.** (2025). Defenses and Attachment in Clinical Practice. *Journal of Personality Assessment, 107*(1), 140–141. — 依恋特定防御 vs 一般防御机制的区分
8. **Nitta, T., et al.** (1999). Modeling Human Mind. *IEEE SMC'99*. — 最早的计算防御机制模型
