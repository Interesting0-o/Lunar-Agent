# Lunar State Engine —— 计算心理学详细文档

> 本文档描述 Lunar 状态引擎的完整数学模型与计算流程。
> 对应代码：`state_engine/` 包。最后更新：2026-06-15。

---

## 目录

1. [状态向量定义](#1-状态向量定义)
2. [管线概览](#2-管线概览)
3. [① 防御剖面](#3-防御剖面)
4. [② 残差动力学](#4-残差动力学)
5. [③ 表面投影](#5-表面投影)
6. [矩阵定义](#6-矩阵定义)
7. [初始化](#7-初始化)
8. [数值稳定化](#8-数值稳定化)

---

## 1. 状态向量定义

所有向量均为 numpy `ndarray`，值域 [0, 1]。维度索引对应 `state.py` 中的命名常量。

### 1.1 Internal State（内部状态，8 维）

角色真正感受到的情绪，持久化存储。

| 索引 | 常量 | 含义 | 默认基线 |
|------|------|------|---------|
| 0 | `I_ENERGY` | 心理能量 | 0.70 |
| 1 | `I_STRESS` | 压力/焦虑 | 0.20 |
| 2 | `I_LONELINESS` | 孤独感 | 0.30 |
| 3 | `I_INSECURITY` | 不安全感 | 0.25 |
| 4 | `I_IRRITATION` | 烦躁/恼怒 | 0.10 |
| 5 | `I_LONGING` | 思念/渴望 | 0.40 |
| 6 | `I_SOCIAL_BATTERY` | 社交电量 | 0.60 |
| 7 | `I_MENTAL_FATIGUE` | 精神疲劳 | 0.15 |

### 1.2 Relationship State（关系状态，6 维）

角色对用户的关系感知，持久化存储。

| 索引 | 常量 | 含义 | 默认基线 |
|------|------|------|---------|
| 0 | `R_AFFECTION` | 好感度 | 0.33 |
| 1 | `R_TRUST` | 信任 | 0.345 |
| 2 | `R_FAMILIARITY` | 熟悉度 | 0.215 |
| 3 | `R_DEPENDENCY` | 依赖 | 0.20 |
| 4 | `R_EMOTIONAL_SAFETY` | 情感安全 | 0.286 |
| 5 | `R_ROMANTIC_TENSION` | 浪漫张力 | 0.202 |

### 1.3 Surface State（表面状态，7 维）

角色外显的表达特征，**每轮动态投影，不持久化**。

| 索引 | 常量 | 含义 |
|------|------|------|
| 0 | `S_EXPRESSIVENESS` | 情绪外露程度（0=内敛, 1=奔放） |
| 1 | `S_WARMTH` | 语气温度（0=冰冷, 1=温暖） |
| 2 | `S_SHARPNESS` | 攻击性/尖锐感（0=温和, 1=尖锐） |
| 3 | `S_SOFTNESS` | 柔和/脆弱感（0=强硬, 1=柔软） |
| 4 | `S_ENTHUSIASM` | 热情/兴致（0=冷淡, 1=高涨） |
| 5 | `S_RESTRAINT` | 克制/拘谨（0=自由, 1=约束） |
| 6 | `S_VULNERABILITY` | 示弱/柔软度（0=全副武装, 1=暴露脆弱） |

### 1.4 Traits（人格特质，10 维）

长期稳定的性格参数，不随对话更新（暂不支持特质演化）。

| 索引 | 常量 | 含义 | 默认值 |
|------|------|------|--------|
| 0 | `T_SENSITIVITY` | 敏感度 | 0.65 |
| 1 | `T_PRIDE` | 自尊/骄傲 | 0.70 |
| 2 | `T_EMOTIONAL_OPENNESS` | 情感开放性 | 0.30 |
| 3 | `T_EMOTIONAL_STABILITY` | 情绪稳定性 | 0.30 |
| 4 | `T_OPTIMISM` | 乐观倾向 | 0.25 |
| 5 | `T_ANXIETY_PRONENESS` | 焦虑易感性 | 0.65 |
| 6 | `T_ANGER_REACTIVITY` | 易怒性 | 0.30 |
| 7 | `T_JEALOUSY_SENSITIVITY` | 嫉妒敏感性 | 0.45 |
| 8 | `T_ATTACHMENT_ANXIETY` | 依恋焦虑 | 0.70 |
| 9 | `T_ATTACHMENT_AVOIDANCE` | 依恋回避 | 0.20 |

### 1.5 Stimulus Vector（心理刺激，7 维）

Perception 节点从用户输入中提取的心理刺激强度。**不持久化，消费后清理**。

| 索引 | 常量 | 含义 |
|------|------|------|
| 0 | `ST_ABANDONMENT` | 被抛弃/冷落感 |
| 1 | `ST_VALIDATION` | 被认可/肯定感 |
| 2 | `ST_CLOSENESS` | 亲密靠近 |
| 3 | `ST_CONFLICT` | 冲突/对抗 |
| 4 | `ST_DEPENDENCY` | 依赖/需要 |
| 5 | `ST_TEASING` | 逗弄/玩笑 |
| 6 | `ST_EMOTIONAL_WEIGHT` | 情感重量（话题的严肃性） |

---

## 2. 管线概览

```
原始刺激 s (7维)
    ↓
① Defense Profiles (compute + apply)
    → profiles (3×7) 防御剖面矩阵
    → inner_stimuli (7维) — 角色心理实际接收的强度
    → outer_stimuli (7维) — 角色外显表达的强度（经防御过滤）
    ↓
② Residual Dynamics (update_internal + update_relationship)
    → 新 internal_state (8维)
    → 新 relationship_state (6维)
    （含内建稳态恢复，不再需要独立 decay 步骤）
    ↓
③ Surface Projection
    → surface_state (7维) — 从 internal + relationship + outer 动态投影
```

主入口：`update_all(current_internal, current_relationship, traits, stimuli)`。

---

## 3. ① 防御剖面

文件：`state_engine/_defenses.py`

### 3.1 设计原理

每个防御机制不是全局标量，而是一个 **7 维敏感度剖面**——对不同类型的心理刺激有不同的激活程度。这反映了临床心理学中的事实：防御机制是选择性的。例如，高 Pride 的人压抑"被抛弃的恐惧"远甚于压抑"亲密的渴望"。

当前定义 3 个防御剖面，存储为 (3, 7) 矩阵：

```
profiles[0, :] = suppression   — 压抑各类刺激外显的程度
profiles[1, :] = vulnerability — 允许各类刺激泄露到表面的程度
profiles[2, :] = attachment    — 放大各类刺激内部感受的程度
```

### 3.2 剖面计算

```
profile[d, s] = σ(trait_baseline[d, s] × rel_mod[d] + internal_push[d])
```

其中 `σ` 为 sigmoid 激活函数，`trait_baseline[d, s]` 由人格特质偏差 `t_dev = traits − 0.5` 加权得到。

#### Suppression 剖面（压抑）

压抑控制的是"哪些感受不表现出来"。高 Pride → 高压抑（尤其对暴露脆弱的刺激），高 Stability → 低压抑（真淡定，不需压抑）。

```
supp[s] = base[s] + Σ_k t_dev[k] × W_supp[k, s]

关键权重（非零项）:
  ST_ABANDONMENT ← pride(+0.50), jealousy(+0.20), stability(−0.20)
  ST_VALIDATION   ← pride(+0.40)
  ST_DEPENDENCY   ← pride(+0.40), avoidance(+0.20)
  ST_CLOSENESS    ← pride(+0.20)
  ST_CONFLICT     ← pride(+0.30), anger(+0.25)
  ST_TEASING      ← pride(+0.35)
  ST_EMOTIONAL_WEIGHT ← pride(+0.30)

关系调制: supp *= (1 − trust×0.25 − safety×0.18)
内部推动: supp += stress×0.10 + insecurity×0.08

最终: profile[0, s] = σ(supp[s] − 0.50)
```

#### Vulnerability 剖面（脆弱/示弱）

脆弱度控制的是"哪些感受忍不住流露"。高 Sensitivity + 高 Openness → 更愿意示弱，高 Pride → 压制示弱。

```
vuln[s] = base[s] + Σ_k t_dev[k] × W_vuln[k, s]

关键权重:
  ST_VALIDATION   ← sensitivity(+0.50), openness(+0.30)
  ST_CLOSENESS    ← sensitivity(+0.40), openness(+0.30)
  ST_DEPENDENCY   ← sensitivity(+0.40)
  ST_ABANDONMENT  ← sensitivity(+0.30)
  全维度          ← pride(−0.25)  [高自尊全局压制]

关系调制: vuln *= (1 + safety×0.20 + familiarity×0.12)
内部推动: vuln += loneliness×0.12 + longing×0.10

最终: profile[1, s] = σ(vuln[s] − 0.45)
```

#### Attachment 剖面（依恋敏感）

依恋敏感控制的是"哪些感受被内心放大"。高 Attachment Anxiety → 放大关系威胁和亲近信号。

```
att[s] = base[s] + Σ_k t_dev[k] × W_att[k, s]

关键权重:
  ST_ABANDONMENT ← anxiety(+0.55), jealousy(+0.30), avoidance(−0.30)
  ST_CLOSENESS    ← anxiety(+0.50)
  ST_DEPENDENCY   ← anxiety(+0.40)
  ST_CONFLICT     ← anxiety(+0.30)
  ST_TEASING      ← jealousy(+0.20)
  ST_EMOTIONAL_WEIGHT ← anxiety(+0.30)

关系调制: att *= (1 + affection×0.18 + romantic_tension×0.10)
内部推动: att += insecurity×0.12 + longing×0.08

最终: profile[2, s] = σ(att[s] − 0.50)
```

### 3.3 防御应用

将剖面应用到原始刺激，生成 inner/outer 两套刺激向量。

```
inner[s] = stimuli[s] × (1 + attachment[s] × 0.50)
outer[s] = inner[s] × (1 − suppression[s] × 0.70) × (0.30 + vulnerability[s] × 0.70)
```

**逐维度效果分析**：

| 刺激类型 | suppression 高 → | vulnerability 高 → | attachment 高 → |
|---------|-----------------|-------------------|----------------|
| abandonment | outer 被大幅压制 | 略微泄漏 | inner 被放大（内心更恐慌） |
| validation | outer 被压制（不表现高兴） | 可以流露 | 基本不变 |
| closeness | 压制较少 | 更容易流露 | inner 被放大（既渴望又怕） |
| conflict | 中等压制 | 基本不泄漏 | inner 被放大 |
| dependency | 高压制（不表现需要） | 可以流露 | inner 被放大（更依赖） |

---

## 4. ② 残差动力学

文件：`state_engine/_dynamics.py`

### 4.1 核心公式

```
h_t = h_{t-1} + Δt · (α · Δ_coupling + β · Δ_stimulus + γ · Δ_homeostatic)
```

三项分别由不同的速率参数调制。门控控制的是**变化有多快**，不是**能变到哪**。

#### α — 跨维度耦合速率

控制内部维度之间的相互影响速度。

```
α = openness×0.30 + (1−stability)×0.15 + trust×0.12
α ∈ [0.02, 0.35]
```

高 Openness → 情绪维度之间互相影响快（一种情绪容易引发另一种）
高 Stability → 耦合慢（各维度较独立，不轻易联动）
高 Trust → 耦合快（对关系更开放）

#### β — 刺激接受速率

控制外部刺激影响内部状态的速度，由防御剖面决定。

```
β = 0.10 + att_mean×0.12 + vuln_mean×0.08 − supp_mean×0.10
β ∈ [0.01, 0.35]
```

Attachment 高 → 接受快（对刺激更敏感）
Suppression 高 → 接受慢（防御厚，刺激进得慢）
Vulnerability 高 → 接受快（愿意被影响）

#### γ — 稳态恢复速率

控制状态回归人格 setpoint 的速度。

```
γ = 0.08 + stability×0.10 + optimism×0.06 − anxiety×0.06 − supp_mean×0.08
γ ∈ [0.01, 0.25]
```

高 Stability → 恢复快（情绪稳定的人消气快）
高 Optimism → 恢复快（乐观的人想得开）
高 Anxiety → 恢复慢（焦虑的人放不下）
高 Suppression → 恢复慢（压抑的人消化不掉，情绪滞留）

### 4.2 三项 Δ

#### Δ_coupling（跨维度耦合）

使用**差值形式**（ADF 模型的核心设计）：

```
Δ_coupling = A_norm · h_{t-1} − h_{t-1}
```

当 h = A·h 时此项为 0（系统处于不动点）。差值形式自动保证：
- h 偏离不动点时，耦合项将其拉回
- 所有维度相等时耦合项为零
- 不会出现不受控的正反馈

#### Δ_stimulus（刺激输入）

```
Δ_stimulus = inner_stimuli · B
```

其中 `inner_stimuli` 是 (7,) 向量，`B` 是 (7, 8) 输入影响矩阵，结果 (8,) 向量。

每个刺激维度对每个内部状态维度有不同的影响权重（详见 §6.2）。

#### Δ_homeostatic（稳态恢复）

```
Δ_homeostatic = setpoint(traits) − h_{t-1}
```

`setpoint(traits)` 由人格决定（§4.3）。此项永远是恢复力——当 `h` 高于 setpoint 时拉下来，低于时拉上去。

### 4.3 Setpoint 计算

不同人格有不同的"正常"情绪水平。setpoint 是系统在无外部刺激时的长期收敛目标。

```
sp = DEFAULT_INTERNAL + Δ_sp(traits)

主要调幅:
  sp[ENERGY]     += optimism_dev×0.15 − anxiety_dev×0.08
  sp[STRESS]     += anxiety_dev×0.20 + anger_dev×0.05
  sp[LONELINESS] += attachment_dev×0.10 − optimism_dev×0.05
  sp[INSECURITY] += attachment_dev×0.20 + anxiety_dev×0.10
  sp[IRRITATION] += anger_dev×0.15 − stability_dev×0.10
  sp[LONGING]    += attachment_dev×0.15
  sp[BATTERY]    += stability_dev×0.05
  sp[FATIGUE]    −= stability_dev×0.08 + anxiety_dev×0.05

所有偏离值 t_dev = trait − 0.5
```

### 4.4 关系动力学

与内部状态**同构**（同为残差更新），但所有速率参数缩小 5-10 倍：

```
α_rel = openness×0.04 + trust×0.03 + safety×0.02
α_rel ∈ [0.005, 0.06]

β_rel = 0.02 + attachment_anxiety×0.015
β_rel ∈ [0.002, 0.06]

γ_rel = 0.005 + stability×0.005
γ_rel ∈ [0.001, 0.02]
```

关系 setpoint 由依恋风格决定：

```
sp[TRUST]    −= avoidance_dev×0.15
sp[AFFECTION] −= avoidance_dev×0.10
sp[DEPENDENCY] −= avoidance_dev×0.15 + attachment_anxiety_dev×0.10
sp[SAFETY]   −= avoidance_dev×0.12
sp[TENSION]  += attachment_anxiety_dev×0.05
```

### 4.5 残差连接的意义

```
旧方法: h_t = f_gate × h_{t-1} + ...    (f_gate < 1 → h_0 在 200 轮后变为 0)
新方法: h_t = h_{t-1} + Δt × Δh          (h_0 以 1.0 权重保留)
```

残差形式保证：
- 长程对话中第 1 轮建立的状态信息永不丢失
- 变化是**增量的**——每轮只改变一小部分
- 系统收敛到稳态而非漂移

---

## 5. ③ 表面投影

文件：`state_engine/_surface.py`

### 5.1 公式

表面表达 = 内部状态基线 + 外表情刺激影响 + 特质连续修饰

```
s[EXPRESSIVENESS] = 0.30 + energy×0.40 − fatigue×0.30
s[WARMTH]         = 0.30 + affection×0.40 − stress×0.20
                     + outer[VALIDATION]×0.30 + outer[DEPENDENCY]×0.10
s[SHARPNESS]      = 0.10 + irritation×0.50 + stress×0.20
                     + outer[CONFLICT]×0.25 + outer[TEASING]×0.10
s[SOFTNESS]       = 0.20 + (1−stress)×0.30 + safety×0.20
                     + outer[CLOSENESS]×0.20
s[ENTHUSIASM]     = 0.30 + energy×0.50 − fatigue×0.30
s[RESTRAINT]      = 0.20 + insecurity×0.30 + pride×0.20
                     + outer[EMOTIONAL_WEIGHT]×0.20
s[VULNERABILITY]  = 0.10 + loneliness×0.30 + longing×0.20 − pride×0.20
                     + outer[ABANDONMENT]×0.15
```

### 5.2 特质连续修饰

使用 sigmoid 软阈值替代硬分支 `if trait > 0.6`：

```
pride_active = σ((pride − 0.5) / 0.15)
sharpness     += pride_active × pride × 0.10
vulnerability −= pride_active × pride × 0.15

openness_active = σ((openness − 0.5) / 0.15)
expressiveness += openness_active × openness × 0.10
restraint      −= openness_active × openness × 0.10

optimism_active = σ((optimism − 0.5) / 0.15)
enthusiasm     += optimism_active × optimism × 0.10
```

---

## 6. 矩阵定义

文件：`state_engine/_matrices.py`

### 6.1 内部状态耦合矩阵 A（8×8）

`A[i, j]` = 维度 j 对维度 i 的耦合强度。非零元素：

```
STRESS → IRRITATION:        +0.15
STRESS → MENTAL_FATIGUE:    +0.10
STRESS → LONELINESS:        +0.08
LONELINESS → INSECURITY:    +0.12
LONELINESS → LONGING:       +0.15
SOCIAL_BATTERY → FATIGUE:   −0.10   (社交电量耗尽→疲劳)
SOCIAL_BATTERY → IRRITATION: −0.08
ENERGY → STRESS:            −0.05   (精力充沛→减压)
ENERGY → LONELINESS:        −0.05   (精力充沛→减孤独)
INSECURITY → STRESS:        +0.10

对角自保持: A[i, i] = 0.85
```

谱半径 ρ(A) = 0.9486（稳定）。

### 6.2 输入影响矩阵 B（7×8）

`B[s, i]` = 刺激 s 对内部状态 i 的影响权重。

```
ST_ABANDONMENT → INSECURITY:     +0.25
               → LONELINESS:     +0.18
               → STRESS:         +0.12
               → LONGING:        +0.18
               → ENERGY:         −0.12

ST_VALIDATION  → INSECURITY:     −0.20
               → ENERGY:         +0.15
               → LONELINESS:     −0.15

ST_CLOSENESS   → LONELINESS:     −0.25
               → LONGING:        −0.10
               → SOCIAL_BATTERY: −0.10
               → ENERGY:         +0.08

ST_CONFLICT    → STRESS:         +0.35
               → IRRITATION:     +0.30
               → ENERGY:         −0.20
               → FATIGUE:        +0.25
               → SOCIAL_BATTERY: −0.25

ST_DEPENDENCY  → SOCIAL_BATTERY: −0.08
               → LONELINESS:     −0.15
               → ENERGY:         +0.05

ST_TEASING     → SOCIAL_BATTERY: −0.08
               → IRRITATION:     +0.05
               → ENERGY:         +0.05

ST_EMOTIONAL_WEIGHT → STRESS:    +0.20
                    → FATIGUE:   +0.15
```

### 6.3 关系状态耦合矩阵 A_rel（6×6）

```
AFFECTION → TRUST:      +0.08
AFFECTION → FAMILIARITY: +0.05
TRUST → SAFETY:          +0.10
TRUST → DEPENDENCY:      +0.05
FAMILIARITY → SAFETY:    +0.08
AFFECTION → SAFETY:      +0.05
TRUST → SAFETY:          +0.05
AFFECTION → TENSION:     +0.03
TENSION → DEPENDENCY:    +0.05

对角自保持: A_rel[i, i] = 0.90
```

原始谱半径 ρ(A_rel) = 1.0063 > 1（不稳定）。经谱归一化缩放到 ρ = 0.95。

### 6.4 关系输入影响矩阵 B_rel（7×6）

```
ST_ABANDONMENT → TRUST:    −0.08
               → SAFETY:   −0.10
               → TENSION:  +0.06
               → DEPENDENCY: +0.08

ST_VALIDATION  → AFFECTION: +0.12
               → TRUST:     +0.10

ST_CLOSENESS   → AFFECTION: +0.10
               → FAMILIARITY: +0.12
               → SAFETY:    +0.08
               → TENSION:   +0.06

ST_CONFLICT    → TRUST:     −0.18
               → SAFETY:    −0.20
               → AFFECTION: −0.08
               → TENSION:   −0.08

ST_DEPENDENCY  → DEPENDENCY: +0.18
               → FAMILIARITY: +0.06
               → SAFETY:    +0.05

ST_TEASING     → FAMILIARITY: +0.08
               → TENSION:    +0.08
```

### 6.5 谱归一化

所有耦合矩阵在构建时自动检查谱半径。若 ρ ≥ 0.95，整体缩放到 0.95：

```
A_norm = A × 0.95 / ρ(A)   if ρ(A) ≥ 0.95
```

这保证了 LTI 子系统的稳定性——无外部输入时，状态不会自发漂移。

---

## 7. 初始化

文件：`state_engine/_pipeline.py` → `initialize_all()`

首次运行时（`current_internal is None`）：

```
internal     = compute_setpoint(traits)      # 人格决定的情绪基线
relationship = compute_rel_setpoint(traits)  # 人格决定的关系基线
surface      = project_surface(internal, relationship, traits, zeros(7))
```

不再使用固定的 `DEFAULT_INTERNAL` / `DEFAULT_RELATIONSHIP` 作为初始值——初始化即反映人格基线。

---

## 8. 数值稳定化

文件：`state_engine/_utils.py`

### 8.1 soft_clamp

软饱和裁剪，区间外用 tanh 平滑压回，保留"超出量"的梯度信息：

```
upper_delta = x − high
upper_output = high − transition × tanh(upper_delta / transition)
```

transition = 0.1 时：
- x = 1.00 → 1.0000（不变）
- x = 1.10 → 0.9999（几乎到 1）
- x = 2.00 → 1.0000（饱和）

### 8.2 sigmoid

数值稳定实现，避免 ±∞ 时产生 NaN：

```
pos: result = 1 / (1 + exp(−x))
neg: result = exp(x) / (1 + exp(x))
```

### 8.3 运行时不变量

- **谱半径验证**：`validate_matrices()` 检查 A/A_rel ρ < 1
- **门控范围**：所有 profile 值经 `soft_clamp(profiles, 0, 1)`
- **状态边界**：所有状态更新后经 `soft_clamp(result, 0, 1)`
- **速率边界**：α/β/γ 均有 `soft_clamp` 强制上下界

---

## 附录：与旧架构的对比

| | 旧架构 (v0.1) | 新架构 (v0.2) |
|---|-------------|-------------|
| 管线步骤 | 5 步 8 函数 | 3 步 3 核心函数 |
| 门控形式 | 3 个全局标量 | 3×7 逐维度敏感度剖面矩阵 |
| 状态更新 | `f*h + i*(A·h+B·s) + g*bias` | `h + α*(A·h−h) + β*B·s + γ*(setpoint−h)` |
| 门控语义 | 控制状态值比例 | 控制变化速率 |
| 长程稳定性 | 旧状态被 fⁿ 稀释 | 残差连接，旧状态 1.0 保留 |
| 稳态恢复 | 独立 decay 层，向固定基线回归 | 动力学内建，向人格 setpoint 回归 |
| 阈值 | `if x > 0.3` 硬分支 | sigmoid 软阈值 |
| 矩阵稳定性 | A_rel ρ=1.0063（不稳定） | A_rel ρ=0.95（谱归一化） |
