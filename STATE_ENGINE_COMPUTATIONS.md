# Lunar State Engine —— 计算心理学完整算式文档

> 本文档展开 Lunar 状态引擎中每一个数学公式的完整计算过程——从原始刺激到最终表面状态。
> 所有公式与 `state_engine/` 源码精确对应。最后更新：2026-06-15。

---

## 目录

1. [状态向量定义](#1-状态向量定义)
2. [管线概览](#2-管线概览)
3. [① 防御剖面](#3-防御剖面)
4. [② 残差动力学](#4-残差动力学)
5. [③ 表面投影](#5-表面投影)
6. [矩阵定义](#6-矩阵定义)
7. [完整数值示例](#7-完整数值示例)
8. [数值稳定化](#8-数值稳定化)

---

## 1. 状态向量定义

所有向量均为 numpy `ndarray`，值域 [0, 1]，存储在 `state.py`。

### 1.1 Internal State（内部状态，8 维）— 持久化

角色真正感受到的情绪。

| 索引 | 常量 | 含义 | 默认值 |
|------|------|------|--------|
| 0 | `I_ENERGY` | 心理能量 | 0.70 |
| 1 | `I_STRESS` | 压力/焦虑 | 0.20 |
| 2 | `I_LONELINESS` | 孤独感 | 0.30 |
| 3 | `I_INSECURITY` | 不安全感 | 0.25 |
| 4 | `I_IRRITATION` | 烦躁/恼怒 | 0.10 |
| 5 | `I_LONGING` | 思念/渴望 | 0.40 |
| 6 | `I_SOCIAL_BATTERY` | 社交电量 | 0.60 |
| 7 | `I_MENTAL_FATIGUE` | 精神疲劳 | 0.15 |

记法：$h = [h_0, h_1, \dots, h_7]$ 分别对应 energy, stress, loneliness, insecurity, irritation, longing, social_battery, mental_fatigue。

### 1.2 Relationship State（关系状态，6 维）— 持久化

角色对用户的关系感知。

| 索引 | 常量 | 含义 | 默认值 |
|------|------|------|--------|
| 0 | `R_AFFECTION` | 好感度 | 0.33 |
| 1 | `R_TRUST` | 信任 | 0.345 |
| 2 | `R_FAMILIARITY` | 熟悉度 | 0.215 |
| 3 | `R_DEPENDENCY` | 依赖 | 0.20 |
| 4 | `R_EMOTIONAL_SAFETY` | 情感安全 | 0.286 |
| 5 | `R_ROMANTIC_TENSION` | 浪漫张力 | 0.202 |

记法：$r = [r_0, r_1, \dots, r_5]$ 对应 affection, trust, familiarity, dependency, safety, tension。

### 1.3 Surface State（表面状态，7 维）— 每轮动态投影

角色外显的表达特征。

| 索引 | 常量 | 含义 |
|------|------|------|
| 0 | `S_EXPRESSIVENESS` | 情绪外露程度 |
| 1 | `S_WARMTH` | 语气温度 |
| 2 | `S_SHARPNESS` | 攻击性/尖锐感 |
| 3 | `S_SOFTNESS` | 柔和/脆弱感 |
| 4 | `S_ENTHUSIASM` | 热情/兴致 |
| 5 | `S_RESTRAINT` | 克制/拘谨 |
| 6 | `S_VULNERABILITY` | 示弱/柔软度 |

记法：$y = [y_0, y_1, \dots, y_6]$。

### 1.4 Traits（人格特质，10 维）— 稳定参数

长期稳定的性格参数。

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

记法：$p = [p_0, p_1, \dots, p_9]$，偏离中性值 $\Delta p_k = p_k - 0.5$。

**默认角色的 $\Delta p$**（用于后续数值示例）：

$$\Delta p = [+0.15,\; +0.20,\; -0.20,\; -0.20,\; -0.25,\; +0.15,\; -0.20,\; -0.05,\; +0.20,\; -0.30]$$

### 1.5 Stimulus Vector（心理刺激，7 维）— 不持久化

Perception 节点从用户输入中提取，消费后清理。

| 索引 | 常量 | 含义 |
|------|------|------|
| 0 | `ST_ABANDONMENT` | 被抛弃/冷落感 |
| 1 | `ST_VALIDATION` | 被认可/肯定感 |
| 2 | `ST_CLOSENESS` | 亲密靠近 |
| 3 | `ST_CONFLICT` | 冲突/对抗 |
| 4 | `ST_DEPENDENCY` | 依赖/需要 |
| 5 | `ST_TEASING` | 逗弄/玩笑 |
| 6 | `ST_EMOTIONAL_WEIGHT` | 情感重量 |

记法：$s = [s_0, s_1, \dots, s_6]$。

---

## 2. 管线概览

```
原始刺激 s (7维)
    ↓
① Defense Profiles (compute_defense_profiles + apply_defenses)
    → profiles (3×7) 防御剖面矩阵
    → s_inner (7维) — 角色心理实际接收的强度
    → s_outer (7维) — 角色外显表达的强度（经防御过滤）
    ↓
② Residual Dynamics (update_internal_state + update_relationship_state)
    → h_t (8维) — 新内部状态（含内建稳态恢复）
    → r_t (6维) — 新关系状态
    ↓
③ Surface Projection (project_surface)
    → y_t (7维) — 从 h_t + r_t + s_outer 动态投影
```

主入口签名：`update_all(h_{t-1} | None, r_{t-1} | None, p, s) → dict`

---

## 3. ① 防御剖面

### 3.1 设计原理

每个防御机制不是全局标量，而是 **7 维敏感度剖面向量**——对不同类型的心理刺激有不同的激活水平。

$$
\displaylines{
\text{profile}^{(0)} \in [0,1]^7 = \text{suppression} \\
\text{profile}^{(1)} \in [0,1]^7 = \text{vulnerability} \\
\text{profile}^{(2)} \in [0,1]^7 = \text{attachment}
}
$$

对于每个防御 $d$ 和刺激类型 $s$：$\text{profile}[d, s]$ 越高，该防御对该刺激越活跃。

---

### 3.2 Suppression 剖面（压抑）

压抑控制"哪些感受不表现出来"。

#### 公式（逐维度展开）

定义 $q_k = \Delta p_k$（特质偏离中性值），$\sigma(x) = 1/(1+e^{-x})$。

$$
\begin{aligned}
\text{raw\_supp}_0 &= 0.35 + q_{\text{pride}} \cdot 0.50 + q_{\text{jealousy}} \cdot 0.20 \quad &\text{(abandonment)} \\[2pt]
\text{raw\_supp}_1 &= 0.25 + q_{\text{pride}} \cdot 0.40 \quad &\text{(validation)} \\[2pt]
\text{raw\_supp}_2 &= 0.15 + q_{\text{pride}} \cdot 0.20 \quad &\text{(closeness)} \\[2pt]
\text{raw\_supp}_3 &= 0.20 + q_{\text{pride}} \cdot 0.30 + q_{\text{anger}} \cdot 0.25 \quad &\text{(conflict)} \\[2pt]
\text{raw\_supp}_4 &= 0.30 + q_{\text{pride}} \cdot 0.40 + q_{\text{avoidance}} \cdot 0.20 \quad &\text{(dependency)} \\[2pt]
\text{raw\_supp}_5 &= 0.20 + q_{\text{pride}} \cdot 0.35 \quad &\text{(teasing)} \\[2pt]
\text{raw\_supp}_6 &= 0.25 + q_{\text{pride}} \cdot 0.30 \quad &\text{(emotional\_weight)}
\end{aligned}
$$

**全局修正**：

$$
\text{raw\_supp}_i \;-\!\!=\! q_{\text{stability}} \cdot 0.20 \quad \forall i \in [0,6]
$$

**关系调制**（信任和安全感松动压抑）：

$$
m_{\text{rel}} = 1.0 - r_{\text{trust}} \cdot 0.25 - r_{\text{safety}} \cdot 0.18
$$

$$
\text{raw\_supp}_i \;*\}\!\!=\! m_{\text{rel}} \quad \forall i
$$

**内部急性推动**（压力/不安 → 更压抑）：

$$
\text{raw\_supp}_i \;+\!\!=\! h_{\text{stress}} \cdot 0.10 + h_{\text{insecurity}} \cdot 0.08 \quad \forall i
$$

**最终激活**：

$$
\text{profile}[0, i] = \sigma(\text{raw\_supp}_i - 0.50)
$$

#### 用默认角色数值代入

默认角色：$q_{\text{pride}}=+0.20$, $q_{\text{jealousy}}=-0.05$, $q_{\text{anger}}=-0.20$, $q_{\text{stability}}=-0.20$, $q_{\text{avoidance}}=-0.30$, 初始 $h_{\text{stress}}=0.22$, $h_{\text{insecurity}}=0.27$, $r_{\text{trust}}=0.345$, $r_{\text{safety}}=0.286$.

$$
\begin{aligned}
\text{raw\_supp}_0 &= 0.35 + 0.20 \times 0.50 + (-0.05) \times 0.20 &= 0.440 \\
\text{raw\_supp}_1 &= 0.25 + 0.20 \times 0.40 &= 0.330 \\
\text{raw\_supp}_2 &= 0.15 + 0.20 \times 0.20 &= 0.190 \\
\text{raw\_supp}_3 &= 0.20 + 0.20 \times 0.30 + (-0.20) \times 0.25 &= 0.210 \\
\text{raw\_supp}_4 &= 0.30 + 0.20 \times 0.40 + (-0.30) \times 0.20 &= 0.320 \\
\text{raw\_supp}_5 &= 0.20 + 0.20 \times 0.35 &= 0.270 \\
\text{raw\_supp}_6 &= 0.25 + 0.20 \times 0.30 &= 0.310
\end{aligned}
$$

应用全局 stability 修正（$q_{\text{stability}}=-0.20$）：

$$\text{raw\_supp}_i = \text{raw\_supp}_i - (-0.20 \times 0.20) = \text{raw\_supp}_i + 0.04$$

$$
\begin{aligned}
\text{raw\_supp} &= [0.480,\; 0.370,\; 0.230,\; 0.250,\; 0.360,\; 0.310,\; 0.350]
\end{aligned}
$$

关系调制：

$$m_{\text{rel}} = 1.0 - 0.345 \times 0.25 - 0.286 \times 0.18 = 1.0 - 0.0863 - 0.0515 = 0.862$$

$$\text{raw\_supp} = [0.414,\; 0.319,\; 0.198,\; 0.216,\; 0.310,\; 0.267,\; 0.302]$$

内部推动（$h_{\text{stress}}=0.22$, $h_{\text{insecurity}}=0.27$）：

$$\text{raw\_supp}_i \;+\!\!=\; 0.22 \times 0.10 + 0.27 \times 0.08 = 0.022 + 0.022 = 0.044$$

$$\text{raw\_supp} = [0.458,\; 0.363,\; 0.242,\; 0.260,\; 0.354,\; 0.311,\; 0.346]$$

**最终** $\text{profile}[0] = \sigma(\text{raw\_supp} - 0.50)$：

$$\text{profile}[0] = [0.489,\; 0.466,\; 0.436,\; 0.440,\; 0.464,\; 0.453,\; 0.462]$$

---

### 3.3 Vulnerability 剖面（脆弱/示弱）

脆弱度控制"哪些感受忍不住流露"。

#### 公式（逐维度展开）

$$
\begin{aligned}
\text{raw\_vuln}_0 &= 0.15 + q_{\text{sensitivity}} \cdot 0.30 \quad &\text{(abandonment)} \\[2pt]
\text{raw\_vuln}_1 &= 0.30 + q_{\text{sensitivity}} \cdot 0.50 + q_{\text{openness}} \cdot 0.30 \quad &\text{(validation)} \\[2pt]
\text{raw\_vuln}_2 &= 0.40 + q_{\text{sensitivity}} \cdot 0.40 + q_{\text{openness}} \cdot 0.30 \quad &\text{(closeness)} \\[2pt]
\text{raw\_vuln}_3 &= 0.10 \quad &\text{(conflict)} \\[2pt]
\text{raw\_vuln}_4 &= 0.30 + q_{\text{sensitivity}} \cdot 0.40 \quad &\text{(dependency)} \\[2pt]
\text{raw\_vuln}_5 &= 0.15 + q_{\text{sensitivity}} \cdot 0.10 \quad &\text{(teasing)} \\[2pt]
\text{raw\_vuln}_6 &= 0.25 + q_{\text{sensitivity}} \cdot 0.30 \quad &\text{(emotional\_weight)}
\end{aligned}
$$

**全局 pride 压制**：

$$\text{raw\_vuln}_i \;-\!\!=\; q_{\text{pride}} \cdot 0.25 \quad \forall i$$

**关系调制**（情感安全和熟悉鼓励示弱）：

$$m_{\text{rel}} = 1.0 + r_{\text{safety}} \cdot 0.20 + r_{\text{familiarity}} \cdot 0.12$$

$$\text{raw\_vuln}_i \;*\}\!\!=\; m_{\text{rel}} \quad \forall i$$

**内部急性推动**（孤独/渴望 → 更想示弱）：

$$\text{raw\_vuln}_i \;+\!\!=\; h_{\text{loneliness}} \cdot 0.12 + h_{\text{longing}} \cdot 0.10 \quad \forall i$$

**最终激活**：

$$\text{profile}[1, i] = \sigma(\text{raw\_vuln}_i - 0.45)$$

#### 用默认角色数值代入

$q_{\text{sensitivity}}=+0.15$, $q_{\text{openness}}=-0.20$, $q_{\text{pride}}=+0.20$, $r_{\text{safety}}=0.286$, $r_{\text{familiarity}}=0.215$, $h_{\text{loneliness}}=0.302$, $h_{\text{longing}}=0.408$.

$$
\begin{aligned}
\text{raw\_vuln}_0 &= 0.15 + 0.15 \times 0.30 = 0.195 \\
\text{raw\_vuln}_1 &= 0.30 + 0.15 \times 0.50 + (-0.20) \times 0.30 = 0.315 \\
\text{raw\_vuln}_2 &= 0.40 + 0.15 \times 0.40 + (-0.20) \times 0.30 = 0.400 \\
\text{raw\_vuln}_3 &= 0.100 \\
\text{raw\_vuln}_4 &= 0.30 + 0.15 \times 0.40 = 0.360 \\
\text{raw\_vuln}_5 &= 0.15 + 0.15 \times 0.10 = 0.165 \\
\text{raw\_vuln}_6 &= 0.25 + 0.15 \times 0.30 = 0.295
\end{aligned}
$$

Pride 压制：$\text{raw\_vuln}_i \;-\!\!=\; 0.20 \times 0.25 = 0.05$

$$\text{raw\_vuln} = [0.145,\; 0.265,\; 0.350,\; 0.050,\; 0.310,\; 0.115,\; 0.245]$$

关系调制：$m_{\text{rel}} = 1.0 + 0.286 \times 0.20 + 0.215 \times 0.12 = 1.083$

$$\text{raw\_vuln} = [0.157,\; 0.287,\; 0.379,\; 0.054,\; 0.336,\; 0.125,\; 0.265]$$

内部推动：$0.302 \times 0.12 + 0.408 \times 0.10 = 0.077$

$$\text{raw\_vuln} = [0.234,\; 0.364,\; 0.456,\; 0.131,\; 0.413,\; 0.202,\; 0.342]$$

**最终** $\text{profile}[1] = \sigma(\text{raw\_vuln} - 0.45)$：

$$\text{profile}[1] = [0.446,\; 0.479,\; 0.502,\; 0.421,\; 0.491,\; 0.439,\; 0.473]$$

---

### 3.4 Attachment 剖面（依恋敏感）

依恋敏感控制"哪些感受被内心放大"。

#### 公式（逐维度展开）

$$
\begin{aligned}
\text{raw\_att}_0 &= 0.45 + q_{\text{att\_anx}} \cdot 0.55 + q_{\text{jealousy}} \cdot 0.30 \quad &\text{(abandonment)} \\[2pt]
\text{raw\_att}_1 &= 0.15 + q_{\text{att\_anx}} \cdot 0.20 \quad &\text{(validation)} \\[2pt]
\text{raw\_att}_2 &= 0.30 + q_{\text{att\_anx}} \cdot 0.50 \quad &\text{(closeness)} \\[2pt]
\text{raw\_att}_3 &= 0.15 + q_{\text{att\_anx}} \cdot 0.30 \quad &\text{(conflict)} \\[2pt]
\text{raw\_att}_4 &= 0.35 + q_{\text{att\_anx}} \cdot 0.40 \quad &\text{(dependency)} \\[2pt]
\text{raw\_att}_5 &= 0.10 + q_{\text{jealousy}} \cdot 0.20 \quad &\text{(teasing)} \\[2pt]
\text{raw\_att}_6 &= 0.20 + q_{\text{att\_anx}} \cdot 0.30 \quad &\text{(emotional\_weight)}
\end{aligned}
$$

**全局 avoidance 降低**：

$$\text{raw\_att}_i \;-\!\!=\; q_{\text{avoidance}} \cdot 0.30 \quad \forall i$$

**关系调制**（好感和浪漫张力放大依恋）：

$$m_{\text{rel}} = 1.0 + r_{\text{affection}} \cdot 0.18 + r_{\text{tension}} \cdot 0.10$$

$$\text{raw\_att}_i \;*\}\!\!=\; m_{\text{rel}} \quad \forall i$$

**内部急性推动**（不安全/渴望 → 依恋系统激活）：

$$\text{raw\_att}_i \;+\!\!=\; h_{\text{insecurity}} \cdot 0.12 + h_{\text{longing}} \cdot 0.08 \quad \forall i$$

**最终激活**：

$$\text{profile}[2, i] = \sigma(\text{raw\_att}_i - 0.50)$$

#### 用默认角色数值代入

$q_{\text{att\_anx}}=+0.20$, $q_{\text{jealousy}}=-0.05$, $q_{\text{avoidance}}=-0.30$, $r_{\text{affection}}=0.330$, $r_{\text{tension}}=0.202$.

$$
\begin{aligned}
\text{raw\_att}_0 &= 0.45 + 0.20 \times 0.55 + (-0.05) \times 0.30 = 0.545 \\
\text{raw\_att}_1 &= 0.15 + 0.20 \times 0.20 = 0.190 \\
\text{raw\_att}_2 &= 0.30 + 0.20 \times 0.50 = 0.400 \\
\text{raw\_att}_3 &= 0.15 + 0.20 \times 0.30 = 0.210 \\
\text{raw\_att}_4 &= 0.35 + 0.20 \times 0.40 = 0.430 \\
\text{raw\_att}_5 &= 0.10 + (-0.05) \times 0.20 = 0.090 \\
\text{raw\_att}_6 &= 0.20 + 0.20 \times 0.30 = 0.260
\end{aligned}
$$

Avoidance 降低：$\text{raw\_att}_i \;-\!\!=\; (-0.30) \times 0.30 = -0.09$（即 +0.09）

$$\text{raw\_att} = [0.635,\; 0.280,\; 0.490,\; 0.300,\; 0.520,\; 0.180,\; 0.350]$$

关系调制：$m_{\text{rel}} = 1.0 + 0.330 \times 0.18 + 0.202 \times 0.10 = 1.080$

$$\text{raw\_att} = [0.686,\; 0.303,\; 0.529,\; 0.324,\; 0.561,\; 0.194,\; 0.378]$$

内部推动：$0.270 \times 0.12 + 0.408 \times 0.08 = 0.065$

$$\text{raw\_att} = [0.751,\; 0.368,\; 0.594,\; 0.389,\; 0.626,\; 0.259,\; 0.443]$$

**最终** $\text{profile}[2] = \sigma(\text{raw\_att} - 0.50)$：

$$\text{profile}[2] = [0.562,\; 0.467,\; 0.523,\; 0.472,\; 0.531,\; 0.440,\; 0.486]$$

---

### 3.5 防御应用

将三个剖面应用到原始刺激，生成 inner/outer 两套刺激。

#### 公式

对每个刺激维度 $i \in [0, 6]$：

$$
\begin{aligned}
s^{\text{inner}}_i &= s_i \times \big(1.0 + \text{profile}[2, i] \times 0.50\big) \\[4pt]
s^{\text{outer}}_i &= s^{\text{inner}}_i \times \big(1.0 - \text{profile}[0, i] \times 0.70\big) \times \big(0.30 + \text{profile}[1, i] \times 0.70\big)
\end{aligned}
$$

**三层含义**：
- $\text{profile}[2]$（attachment）放大 inner——角色内心实际感受到的比原始刺激更强
- $\text{profile}[0]$（suppression）衰减 outer——压抑"说出来的"少于"感受到的"
- $\text{profile}[1]$（vulnerability）泄露 outer——部分 bypass 压抑，"忍不住流露"

#### 用数值代入（假设 $s = [0.3, 0.5, 0.4, 0.1, 0.2, 0.6, 0.3]$）

使用上面算出的 profiles：

$$\text{profile}[0] = [0.489, 0.466, 0.436, 0.440, 0.464, 0.453, 0.462]$$
$$\text{profile}[1] = [0.446, 0.479, 0.502, 0.421, 0.491, 0.439, 0.473]$$
$$\text{profile}[2] = [0.562, 0.467, 0.523, 0.472, 0.531, 0.440, 0.486]$$

**Inner 计算**：$s^{\text{inner}}_i = s_i \times (1 + \text{att}_i \times 0.50)$

$$
\begin{aligned}
s^{\text{inner}}_0 &= 0.3 \times (1 + 0.562 \times 0.50) = 0.3 \times 1.281 = 0.384 \\
s^{\text{inner}}_1 &= 0.5 \times (1 + 0.467 \times 0.50) = 0.5 \times 1.234 = 0.617 \\
s^{\text{inner}}_2 &= 0.4 \times (1 + 0.523 \times 0.50) = 0.4 \times 1.262 = 0.505 \\
s^{\text{inner}}_3 &= 0.1 \times (1 + 0.472 \times 0.50) = 0.1 \times 1.236 = 0.124 \\
s^{\text{inner}}_4 &= 0.2 \times (1 + 0.531 \times 0.50) = 0.2 \times 1.265 = 0.253 \\
s^{\text{inner}}_5 &= 0.6 \times (1 + 0.440 \times 0.50) = 0.6 \times 1.220 = 0.732 \\
s^{\text{inner}}_6 &= 0.3 \times (1 + 0.486 \times 0.50) = 0.3 \times 1.243 = 0.373
\end{aligned}
$$

$$s^{\text{inner}} = [0.384,\; 0.617,\; 0.505,\; 0.124,\; 0.253,\; 0.732,\; 0.373]$$

**Outer 计算**：$s^{\text{outer}}_i = s^{\text{inner}}_i \times (1 - \text{supp}_i \times 0.70) \times (0.30 + \text{vuln}_i \times 0.70)$

以 dimension 0（abandonment）为例：

$$
\begin{aligned}
\text{supp\_factor} &= 1 - 0.489 \times 0.70 = 0.658 \\
\text{vuln\_factor} &= 0.30 + 0.446 \times 0.70 = 0.612 \\
s^{\text{outer}}_0 &= 0.384 \times 0.658 \times 0.612 = 0.155
\end{aligned}
$$

完整结果：

$$
\begin{aligned}
s^{\text{outer}} &= [0.155,\; 0.279,\; 0.238,\; 0.047,\; 0.113,\; 0.308,\; 0.163] \\[4pt]
\text{outer/inner ratio} &= [0.404,\; 0.452,\; 0.471,\; 0.380,\; 0.446,\; 0.421,\; 0.437]
\end{aligned}
$$

outer/inner < 1 对所有维度成立——防御成功地压抑了外显表达。

---

## 4. ② 残差动力学

### 4.1 核心公式

内部状态更新：

$$
h_t = \text{clamp}\big(h_{t-1} + \Delta t \cdot (\alpha \cdot \Delta_{\text{coupling}} + \beta \cdot \Delta_{\text{stimulus}} + \gamma \cdot \Delta_{\text{homeostatic}}),\; 0,\; 1\big)
$$

其中 $\Delta t = 1.0$（每轮一步），三项速率参数和三项 $\Delta$ 分别如下。

---

### 4.2 速率参数计算

#### α — 跨维度耦合速率

$$
\alpha = \text{clamp}\big(p_{\text{openness}} \times 0.30 + (1 - p_{\text{stability}}) \times 0.15 + r_{\text{trust}} \times 0.12,\; 0.02,\; 0.35\big)
$$

**心理学含义**：Openness 高→情绪联动快，Stability 高→各维度独立（耦合慢），Trust 高→更开放。

默认角色：$\alpha = 0.30 \times 0.30 + 0.70 \times 0.15 + 0.345 \times 0.12 = 0.09 + 0.105 + 0.041 = 0.236$

#### β — 刺激接受速率

$$
\beta = \text{clamp}\big(0.10 + \overline{\text{att}} \times 0.12 + \overline{\text{vuln}} \times 0.08 - \overline{\text{supp}} \times 0.10,\; 0.01,\; 0.35\big)
$$

其中 $\overline{\text{att}}$, $\overline{\text{vuln}}$, $\overline{\text{supp}}$ 是对应 7 维 profile 的均值。

**心理学含义**：Attachment 高→接受快，Suppression 高→接受慢（防御厚），Vulnerability 高→接受快。

默认角色 profiles：$\overline{\text{att}}=0.497$, $\overline{\text{vuln}}=0.464$, $\overline{\text{supp}}=0.459$

$$\beta = 0.10 + 0.497 \times 0.12 + 0.464 \times 0.08 - 0.459 \times 0.10 = 0.10 + 0.060 + 0.037 - 0.046 = 0.151$$

#### γ — 稳态恢复速率

$$
\gamma = \text{clamp}\big(0.08 + p_{\text{stability}} \times 0.10 + p_{\text{optimism}} \times 0.06 - p_{\text{anxiety}} \times 0.06 - \overline{\text{supp}} \times 0.08,\; 0.01,\; 0.25\big)
$$

**心理学含义**：Stability 高→恢复快（情绪稳定消气快），Anxiety 高→恢复慢（焦虑放不下），Suppression 高→恢复慢（压抑消化不掉）。

默认角色：$\gamma = 0.08 + 0.30 \times 0.10 + 0.25 \times 0.06 - 0.65 \times 0.06 - 0.459 \times 0.08$

$$= 0.08 + 0.03 + 0.015 - 0.039 - 0.037 = 0.049$$

---

### 4.3 三项 Δ

#### Δ_coupling（跨维度耦合，差值形式）

$$
\Delta_{\text{coupling}} = A \cdot h_{t-1} - h_{t-1}
$$

用矩阵 A 的元素展开（$h_j$ 是 $h_{t-1}[j]$）：

$$
\Delta_{\text{coupling}}[i] = \sum_{j=0}^{7} A[i,j] \cdot h_j - h_i
$$

$A[i,i] = 0.85$，所以 $\Delta_{\text{coupling}}[i] = (0.85 - 1) \cdot h_i + \sum_{j \neq i} A[i,j] \cdot h_j = -0.15 \cdot h_i + \sum_{j \neq i} A[i,j] \cdot h_j$

**完整的每维展开**（见 §6.1 矩阵定义）：

$$
\begin{aligned}
\Delta_{\text{coupling}}[0] &= -0.15 h_0 \quad &\text{(energy)} \\[2pt]
\Delta_{\text{coupling}}[1] &= -0.15 h_1 + 0.10 h_3 - 0.05 h_0 \quad &\text{(stress)} \\[2pt]
\Delta_{\text{coupling}}[2] &= -0.15 h_2 + 0.08 h_1 - 0.05 h_0 \quad &\text{(loneliness)} \\[2pt]
\Delta_{\text{coupling}}[3] &= -0.15 h_3 + 0.12 h_2 \quad &\text{(insecurity)} \\[2pt]
\Delta_{\text{coupling}}[4] &= -0.15 h_4 + 0.15 h_1 - 0.08 h_6 \quad &\text{(irritation)} \\[2pt]
\Delta_{\text{coupling}}[5] &= -0.15 h_5 + 0.15 h_2 \quad &\text{(longing)} \\[2pt]
\Delta_{\text{coupling}}[6] &= -0.15 h_6 \quad &\text{(social\_battery)} \\[2pt]
\Delta_{\text{coupling}}[7] &= -0.15 h_7 + 0.10 h_1 - 0.10 h_6 \quad &\text{(mental\_fatigue)}
\end{aligned}
$$

#### Δ_stimulus（刺激输入）

$$
\Delta_{\text{stimulus}} = s^{\text{inner}} \cdot B
$$

展开为每个内部状态维度由哪些刺激贡献（见 §6.2 完整矩阵）：

$$
\begin{aligned}
\Delta_{\text{stimulus}}[0] &= s^{\text{inner}}_0 \cdot (-0.12) + s^{\text{inner}}_1 \cdot 0.15 + s^{\text{inner}}_2 \cdot 0.08 + s^{\text{inner}}_3 \cdot (-0.20) + s^{\text{inner}}_4 \cdot 0.05 + s^{\text{inner}}_5 \cdot 0.05 \\[2pt]
\Delta_{\text{stimulus}}[1] &= s^{\text{inner}}_0 \cdot 0.12 + s^{\text{inner}}_3 \cdot 0.35 + s^{\text{inner}}_6 \cdot 0.20 \\[2pt]
&\;\;\vdots \\[2pt]
\Delta_{\text{stimulus}}[7] &= s^{\text{inner}}_3 \cdot 0.25 + s^{\text{inner}}_6 \cdot 0.15
\end{aligned}
$$

#### Δ_homeostatic（稳态恢复）

$$
\Delta_{\text{homeostatic}}[i] = \text{setpoint}_i - h_i
$$

其中 $\text{setpoint} = \text{compute\_setpoint}(p)$ 由人格决定（见 §4.4）。

---

### 4.4 Setpoint 计算

Setpoint 是人格决定的情绪基线——系统在无外部刺激时的长期收敛目标。

$$
\text{sp}[i] = \text{clamp}\big(\text{DEFAULT}[i] + \Delta\text{sp}[i],\; 0.05,\; 0.95\big)
$$

其中 DEFAULT = [0.70, 0.20, 0.30, 0.25, 0.10, 0.40, 0.60, 0.15]（energy, stress, loneliness, insecurity, irritation, longing, battery, fatigue）。

**调幅**（$q_k = p_k - 0.5$）：

$$
\begin{aligned}
\Delta\text{sp}[0] &= q_{\text{optimism}} \times 0.15 - q_{\text{anxiety}} \times 0.08 \\[2pt]
\Delta\text{sp}[1] &= q_{\text{anxiety}} \times 0.20 + q_{\text{anger}} \times 0.05 \\[2pt]
\Delta\text{sp}[2] &= q_{\text{att\_anx}} \times 0.10 - q_{\text{optimism}} \times 0.05 \\[2pt]
\Delta\text{sp}[3] &= q_{\text{att\_anx}} \times 0.20 + q_{\text{anxiety}} \times 0.10 \\[2pt]
\Delta\text{sp}[4] &= q_{\text{anger}} \times 0.15 - q_{\text{stability}} \times 0.10 \\[2pt]
\Delta\text{sp}[5] &= q_{\text{att\_anx}} \times 0.15 \\[2pt]
\Delta\text{sp}[6] &= q_{\text{stability}} \times 0.05 \\[2pt]
\Delta\text{sp}[7] &= -(q_{\text{stability}} \times 0.08 + q_{\text{anxiety}} \times 0.05)
\end{aligned}
$$

**默认角色代入**（$q_{\text{optimism}}=-0.25$, $q_{\text{anxiety}}=+0.15$, $q_{\text{anger}}=-0.20$, $q_{\text{att\_anx}}=+0.20$, $q_{\text{stability}}=-0.20$）：

$$
\begin{aligned}
\Delta\text{sp}[0] &= (-0.25) \times 0.15 - 0.15 \times 0.08 = -0.0375 - 0.012 = -0.050 \\
\Delta\text{sp}[1] &= 0.15 \times 0.20 + (-0.20) \times 0.05 = 0.030 - 0.010 = +0.020 \\
\Delta\text{sp}[2] &= 0.20 \times 0.10 - (-0.25) \times 0.05 = 0.020 + 0.0125 = +0.033 \\
\Delta\text{sp}[3] &= 0.20 \times 0.20 + 0.15 \times 0.10 = 0.040 + 0.015 = +0.055 \\
\Delta\text{sp}[4] &= (-0.20) \times 0.15 - (-0.20) \times 0.10 = -0.030 + 0.020 = -0.010 \\
\Delta\text{sp}[5] &= 0.20 \times 0.15 = +0.030 \\
\Delta\text{sp}[6] &= (-0.20) \times 0.05 = -0.010 \\
\Delta\text{sp}[7] &= -((-0.20) \times 0.08 + 0.15 \times 0.05) = -(-0.016 + 0.0075) = +0.009
\end{aligned}
$$

**最终 setpoint**：

$$h^{\text{sp}} = [0.650,\; 0.220,\; 0.333,\; 0.305,\; 0.090,\; 0.430,\; 0.590,\; 0.159]$$

对比原始 DEFAULT [0.70, 0.20, 0.30, 0.25, 0.10, 0.40, 0.60, 0.15]：
- energy 更低（悲观），stress 更高（焦虑），insecurity 更高（依恋焦虑），longing 更高（同上）——符合人格画像。

---

### 4.5 残差更新（组装）

将三项 $\Delta$ 加权求和：

$$
\Delta h[i] = \alpha \cdot \Delta_{\text{coupling}}[i] + \beta \cdot \Delta_{\text{stimulus}}[i] + \gamma \cdot \Delta_{\text{homeostatic}}[i]
$$

$$
h_t[i] = \text{clamp}\big(h_{t-1}[i] + \Delta h[i],\; 0,\; 1\big)
$$

**单维度完整公式示例**（stress，即 $h[1]$）：

用默认角色数值和 $s^{\text{inner}}$ 代入：

$$
\begin{aligned}
\Delta_{\text{coupling}}[1] &= -0.15 \times 0.22 + 0.10 \times 0.27 - 0.05 \times 0.70 \\
&= -0.033 + 0.027 - 0.035 = -0.041 \\[4pt]
\Delta_{\text{stimulus}}[1] &= 0.384 \times 0.12 + 0.124 \times 0.35 + 0.373 \times 0.20 \\
&= 0.046 + 0.043 + 0.075 = 0.164 \\[4pt]
\Delta_{\text{homeostatic}}[1] &= 0.220 - 0.22 = 0.000 \quad (\text{stress 正好在 setpoint}) \\[4pt]
\Delta h[1] &= 0.236 \times (-0.041) + 0.151 \times 0.164 + 0.049 \times 0.000 \\
&= -0.010 + 0.025 + 0.000 = +0.015 \\[4pt]
h_t[1] &= 0.22 + 0.015 = 0.235
\end{aligned}
$$

---

### 4.6 关系动力学

与内部状态**同构**（同为残差更新），但速率参数缩小 5-10 倍：

$$
r_t[i] = \text{clamp}\big(r_{t-1}[i] + \alpha_{\text{rel}} \cdot \Delta_{\text{coupling}}^{\text{rel}}[i] + \beta_{\text{rel}} \cdot \Delta_{\text{stimulus}}^{\text{rel}}[i] + \gamma_{\text{rel}} \cdot \Delta_{\text{homeostatic}}^{\text{rel}}[i],\; 0,\; 1\big)
$$

**速率参数**：

$$
\begin{aligned}
\alpha_{\text{rel}} &= \text{clamp}(p_{\text{openness}} \times 0.04 + r_{\text{trust}} \times 0.03 + r_{\text{safety}} \times 0.02,\; 0.005,\; 0.06) \\[2pt]
\beta_{\text{rel}} &= \text{clamp}(0.02 + p_{\text{att\_anx}} \times 0.015,\; 0.002,\; 0.06) \\[2pt]
\gamma_{\text{rel}} &= \text{clamp}(0.005 + p_{\text{stability}} \times 0.005,\; 0.001,\; 0.02)
\end{aligned}
$$

默认角色：$\alpha_{\text{rel}}=0.030$, $\beta_{\text{rel}}=0.031$, $\gamma_{\text{rel}}=0.007$。

**关系 setpoint**：

$$
\begin{aligned}
\text{sp}_{\text{rel}}[0] &= 0.33 - (-0.30) \times 0.10 = 0.360 \quad &\text{(affection)} \\[2pt]
\text{sp}_{\text{rel}}[1] &= 0.345 - (-0.30) \times 0.15 = 0.390 \quad &\text{(trust)} \\[2pt]
\text{sp}_{\text{rel}}[2] &= 0.215 - (-0.30) \times 0.05 = 0.230 \quad &\text{(familiarity)} \\[2pt]
\text{sp}_{\text{rel}}[3] &= 0.20 - (-0.30) \times 0.15 + 0.20 \times 0.10 = 0.265 \quad &\text{(dependency)} \\[2pt]
\text{sp}_{\text{rel}}[4] &= 0.286 - (-0.30) \times 0.12 = 0.322 \quad &\text{(safety)} \\[2pt]
\text{sp}_{\text{rel}}[5] &= 0.202 + 0.20 \times 0.05 = 0.212 \quad &\text{(tension)}
\end{aligned}
$$

---

### 4.7 残差连接的意义

旧架构：$h_t = f_g \cdot h_{t-1} + \dots$，其中 $f_g \approx 0.7$。200 轮后 $h_0$ 的贡献 = $0.7^{200} \approx 1.6 \times 10^{-31}$，第 1 轮的状态完全遗忘。

新架构：$h_t = h_{t-1} + \Delta h$。$h_0$ 的信息以权重 1.0 传递到每一轮。$\Delta h$ 是增量——200 轮中每轮只改变一小部分，但 $h_0$ 从不被主动遗忘。

---

## 5. ③ 表面投影

### 5.1 内部状态基线（7 维 → 7 维）

$$
\begin{aligned}
y_0 &= 0.30 + h_{\text{energy}} \times 0.40 - h_{\text{fatigue}} \times 0.30 \\[2pt]
y_1 &= 0.30 + r_{\text{affection}} \times 0.40 - h_{\text{stress}} \times 0.20 \\[2pt]
y_2 &= 0.10 + h_{\text{irritation}} \times 0.50 + h_{\text{stress}} \times 0.20 \\[2pt]
y_3 &= 0.20 + (1 - h_{\text{stress}}) \times 0.30 + r_{\text{safety}} \times 0.20 \\[2pt]
y_4 &= 0.30 + h_{\text{energy}} \times 0.50 - h_{\text{fatigue}} \times 0.30 \\[2pt]
y_5 &= 0.20 + h_{\text{insecurity}} \times 0.30 + p_{\text{pride}} \times 0.20 \\[2pt]
y_6 &= 0.10 + h_{\text{loneliness}} \times 0.30 + h_{\text{longing}} \times 0.20 - p_{\text{pride}} \times 0.20
\end{aligned}
$$

### 5.2 外表情刺激叠加

outer_stimuli（被压抑后的版本）对表面的直接影响：

$$
\begin{aligned}
y_1 &\;+\!\!=\; s^{\text{outer}}_{\text{validation}} \times 0.30 + s^{\text{outer}}_{\text{dependency}} \times 0.10 \\[2pt]
y_2 &\;+\!\!=\; s^{\text{outer}}_{\text{conflict}} \times 0.25 + s^{\text{outer}}_{\text{teasing}} \times 0.10 \\[2pt]
y_3 &\;+\!\!=\; s^{\text{outer}}_{\text{closeness}} \times 0.20 \\[2pt]
y_5 &\;+\!\!=\; s^{\text{outer}}_{\text{emotional\_weight}} \times 0.20 \\[2pt]
y_6 &\;+\!\!=\; s^{\text{outer}}_{\text{abandonment}} \times 0.15
\end{aligned}
$$

### 5.3 特质连续修饰

使用 sigmoid 软阈值（$\sigma$）：

$$
\begin{aligned}
y_2 &\;+\!\!=\; \sigma\left(\frac{p_{\text{pride}} - 0.50}{0.15}\right) \times p_{\text{pride}} \times 0.10 \\[2pt]
y_6 &\;-\!\!=\; \sigma\left(\frac{p_{\text{pride}} - 0.50}{0.15}\right) \times p_{\text{pride}} \times 0.15 \\[2pt]
y_0 &\;+\!\!=\; \sigma\left(\frac{p_{\text{openness}} - 0.50}{0.15}\right) \times p_{\text{openness}} \times 0.10 \\[2pt]
y_5 &\;-\!\!=\; \sigma\left(\frac{p_{\text{openness}} - 0.50}{0.15}\right) \times p_{\text{openness}} \times 0.10 \\[2pt]
y_4 &\;+\!\!=\; \sigma\left(\frac{p_{\text{optimism}} - 0.50}{0.15}\right) \times p_{\text{optimism}} \times 0.10
\end{aligned}
$$

**软阈值效果**（以 pride = 0.70 为例）：

旧方式 `if pride > 0.6`：pride 0.59 和 0.61 产生完全不同的输出（断崖）。

新方式 $\sigma((0.70 - 0.50)/0.15) = \sigma(1.33) = 0.791$——pride 越接近 1.0 修饰越强，pride 接近 0.3 时修饰接近 0。无缝过渡。

---

## 6. 矩阵定义

所有矩阵在 `_matrices.py` 中构建，构建后经谱归一化。

### 6.1 内部状态耦合矩阵 A（8×8）

$$
A = \begin{pmatrix}
0.85 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
-0.05 & 0.85 & 0 & 0.10 & 0 & 0 & 0 & 0 \\
-0.05 & 0.08 & 0.85 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0.12 & 0.85 & 0 & 0 & 0 & 0 \\
0 & 0.15 & 0 & 0 & 0.85 & 0 & -0.08 & 0 \\
0 & 0 & 0.15 & 0 & 0 & 0.85 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0.85 & 0 \\
0 & 0.10 & 0 & 0 & 0 & 0 & -0.10 & 0.85
\end{pmatrix}
\qquad
\begin{array}{l}
\text{行=目标维度，列=源维度} \\
\text{对角线 = 自保持 0.85} \\
\rho(A) = 0.9486
\end{array}
$$

**A 的行解释**（每行：哪些维度影响本维度）：

| 行（目标） | 从哪些维度来 | 权重 |
|-----------|------------|------|
| energy(0) | — | 仅有自保持 |
| stress(1) | insecurity→stress(+0.10), energy→stress(−0.05) | 不安→压力 / 精力→减压 |
| loneliness(2) | stress→loneliness(+0.08), energy→loneliness(−0.05) | 压力→孤独 / 精力→减孤独 |
| insecurity(3) | loneliness→insecurity(+0.12) | 孤独→不安 |
| irritation(4) | stress→irritation(+0.15), battery→irritation(−0.08) | 压力→烦躁 / 电量耗→烦躁 |
| longing(5) | loneliness→longing(+0.15) | 孤独→思念 |
| battery(6) | — | 仅有自保持 |
| fatigue(7) | stress→fatigue(+0.10), battery→fatigue(−0.10) | 压力→疲劳 / 电量低→疲劳 |

### 6.2 输入影响矩阵 B（7×8）

$B[s, i]$ = 刺激 s（行）对内部状态 i（列）的影响。正=增强，负=减弱。

$$
B = \begin{pmatrix}
-0.12 & 0.12 & 0.18 & 0.25 & 0 & 0.18 & 0 & 0 \\[2pt]
0.15 & 0 & -0.15 & -0.20 & 0 & 0 & 0 & 0 \\[2pt]
0.08 & 0 & -0.25 & 0 & 0 & -0.10 & -0.10 & 0 \\[2pt]
-0.20 & 0.35 & 0 & 0 & 0.30 & 0 & -0.25 & 0.25 \\[2pt]
0.05 & 0 & -0.15 & 0 & 0 & 0 & -0.08 & 0 \\[2pt]
0.05 & 0 & 0 & 0 & 0.05 & 0 & -0.08 & 0 \\[2pt]
0 & 0.20 & 0 & 0 & 0 & 0 & 0 & 0.15
\end{pmatrix}
$$

**B 的列解释**（每列：哪些刺激影响本内部维度）：

| 列（内部维度） | 从哪些刺激来 | 权重 |
|-------------|------------|------|
| energy(0) | abandon(−), valid(+), close(+), confl(−), depend(+), teasing(+) | |
| stress(1) | abandon(+), conflict(++), emotional(++) | 冲突+情感重量是最大压力源 |
| loneliness(2) | abandon(+), valid(−), close(−−), depend(−) | 亲密最减孤独 |
| insecurity(3) | abandon(++), valid(−−) | 抛弃增不安，认可减不安 |
| irritation(4) | conflict(++) | 冲突直接致烦躁 |
| longing(5) | abandon(+), close(−) | 抛弃增渴望，亲密减渴望 |
| battery(6) | close(−), confl(−), depend(−), teasing(−) | 所有社交都耗电 |
| fatigue(7) | conflict(++) | 冲突最耗神 |

### 6.3 关系状态耦合矩阵 A_rel（6×6）

$$
A_{\text{rel}} = \begin{pmatrix}
0.90 & 0 & 0 & 0 & 0.05 & 0.03 \\[2pt]
0.08 & 0.90 & 0 & 0 & 0.05 & 0 \\[2pt]
0.05 & 0 & 0.90 & 0 & 0 & 0 \\[2pt]
0 & 0.05 & 0 & 0.90 & 0 & 0 \\[2pt]
0 & 0.10 & 0.08 & 0 & 0.90 & 0 \\[2pt]
0 & 0 & 0 & 0.05 & 0 & 0.90
\end{pmatrix}
\qquad
\rho_{\text{original}} = 1.0063 \xrightarrow{\text{norm}} \rho = 0.9500
$$

关键回路（缩放前）：
- affection → trust(0.08) → safety(0.10) → affection(0.05)：**正反馈环**
- affection → familiarity(0.05) → safety(0.08) → affection(0.05)：**第二个正反馈环**

这两个回路联合导致原始 $\rho > 1$（关系自发漂移到饱和）。谱归一化后该问题已解决。

### 6.4 关系输入影响矩阵 B_rel（7×6）

$$
B_{\text{rel}} = \begin{pmatrix}
0 & -0.08 & 0 & 0.08 & -0.10 & 0.06 \\[2pt]
0.12 & 0.10 & 0 & 0 & 0 & 0 \\[2pt]
0.10 & 0 & 0.12 & 0 & 0.08 & 0.06 \\[2pt]
-0.08 & -0.18 & 0 & 0 & -0.20 & -0.08 \\[2pt]
0 & 0 & 0.06 & 0.18 & 0.05 & 0 \\[2pt]
0 & 0 & 0.08 & 0 & 0 & 0.08 \\[2pt]
0 & 0 & 0 & 0 & 0 & 0
\end{pmatrix}
$$

| 刺激 → 关系 | 主效应 |
|------------|--------|
| abandon → | trust↓, safety↓, tension↑, dependency↑ |
| validation → | affection↑, trust↑ |
| closeness → | affection↑, familiarity↑, safety↑, tension↑ |
| conflict → | trust↓↓, safety↓↓, affection↓ |
| dependency → | dependency↑↑, familiarity↑ |
| teasing → | familiarity↑, tension↑ |

### 6.5 谱归一化

所有耦合矩阵在模块加载时自动检查：

```python
eigenvalues = np.linalg.eigvals(matrix)
rho = max(abs(ev) for ev in eigenvalues)
if rho >= 0.95:
    matrix *= 0.95 / rho
```

**运行时不变量**：`validate_matrices()` 可在 pipeline 每轮调用，验证 $\rho < 1$。

---

## 7. 完整数值示例

使用默认角色（月下誓约）和示例刺激，从初始状态完整走一轮计算。

### 7.1 初始条件

$$\begin{aligned}
p &= [0.65, 0.70, 0.30, 0.30, 0.25, 0.65, 0.30, 0.45, 0.70, 0.20] \quad &\text{(traits)} \\
h_0 &= [0.700, 0.220, 0.300, 0.250, 0.100, 0.400, 0.600, 0.150] \quad &\text{(internal, 初始=setpoint)} \\
r_0 &= [0.330, 0.345, 0.215, 0.200, 0.286, 0.202] \quad &\text{(relationship, 初始=setpoint)} \\
s &= [0.300, 0.500, 0.400, 0.100, 0.200, 0.600, 0.300] \quad &\text{(stimuli — 中等 teasing + validation + closeness)}
\end{aligned}$$

### 7.2 Step 1：防御剖面

已在上文 §3 中完整计算，结果：

$$\begin{aligned}
\text{profile}[0] &= [0.489, 0.466, 0.436, 0.440, 0.464, 0.453, 0.462] \\[2pt]
\text{profile}[1] &= [0.446, 0.479, 0.502, 0.421, 0.491, 0.439, 0.473] \\[2pt]
\text{profile}[2] &= [0.562, 0.467, 0.523, 0.472, 0.531, 0.440, 0.486]
\end{aligned}$$

$$\begin{aligned}
s^{\text{inner}} &= [0.384, 0.617, 0.505, 0.124, 0.253, 0.732, 0.373] \\[2pt]
s^{\text{outer}} &= [0.155, 0.279, 0.238, 0.047, 0.113, 0.308, 0.163]
\end{aligned}$$

### 7.3 Step 2：速率参数

$$\alpha = 0.236,\quad \beta = 0.151,\quad \gamma = 0.049$$

### 7.4 Step 2：完整 $\Delta_h$ 计算（8 维逐维展开）

#### Energy（$h[0]$）

$$
\begin{aligned}
\Delta_{\text{coupling}}[0] &= -0.15 \times 0.70 = -0.1050 \\
\Delta_{\text{stimulus}}[0] &= -0.384 \times 0.12 + 0.617 \times 0.15 + 0.505 \times 0.08 + 0.124 \times (-0.20) + 0.253 \times 0.05 + 0.732 \times 0.05 \\
&= -0.0461 + 0.0925 + 0.0404 - 0.0247 + 0.0127 + 0.0366 = 0.1114 \\
\Delta_{\text{homeostatic}}[0] &= 0.650 - 0.700 = -0.0500 \\[4pt]
\Delta h[0] &= 0.236 \times (-0.1050) + 0.151 \times 0.1114 + 0.049 \times (-0.0500) \\
&= -0.0248 + 0.0168 - 0.0025 = -0.0105 \\[4pt]
h_1[0] &= 0.700 - 0.0105 \approx 0.690
\end{aligned}
$$

#### Stress（$h[1]$）

$$
\begin{aligned}
\Delta_{\text{coupling}}[1] &= -0.15 \times 0.22 + 0.10 \times 0.25 - 0.05 \times 0.70 \\
&= -0.0330 + 0.0250 - 0.0350 = -0.0430 \\
\Delta_{\text{stimulus}}[1] &= 0.384 \times 0.12 + 0.124 \times 0.35 + 0.373 \times 0.20 \\
&= 0.0461 + 0.0433 + 0.0746 = 0.1640 \\
\Delta_{\text{homeostatic}}[1] &= 0.220 - 0.220 = 0.0000 \\[4pt]
\Delta h[1] &= 0.236 \times (-0.0430) + 0.151 \times 0.1640 + 0 = -0.0102 + 0.0248 = 0.0146 \\[4pt]
h_1[1] &= 0.220 + 0.0146 \approx 0.235
\end{aligned}
$$

#### Loneliness（$h[2]$）

$$
\begin{aligned}
\Delta_{\text{coupling}}[2] &= -0.15 \times 0.30 + 0.08 \times 0.22 - 0.05 \times 0.70 \\
&= -0.0450 + 0.0176 - 0.0350 = -0.0624 \\
\Delta_{\text{stimulus}}[2] &= 0.384 \times 0.18 + 0.617 \times (-0.15) + 0.505 \times (-0.25) + 0.253 \times (-0.15) \\
&= 0.0692 - 0.0925 - 0.1262 - 0.0380 = -0.1875 \\
\Delta_{\text{homeostatic}}[2] &= 0.333 - 0.300 = 0.0330 \\[4pt]
\Delta h[2] &= 0.236 \times (-0.0624) + 0.151 \times (-0.1875) + 0.049 \times 0.0330 \\
&= -0.0147 - 0.0283 + 0.0016 = -0.0414 \\[4pt]
h_1[2] &= 0.300 - 0.0414 \approx 0.259
\end{aligned}
$$

#### Insecurity（$h[3]$）

$$
\begin{aligned}
\Delta_{\text{coupling}}[3] &= -0.15 \times 0.25 + 0.12 \times 0.30 = -0.0375 + 0.0360 = -0.0015 \\
\Delta_{\text{stimulus}}[3] &= 0.384 \times 0.25 + 0.617 \times (-0.20) = 0.0960 - 0.1233 = -0.0273 \\
\Delta_{\text{homeostatic}}[3] &= 0.305 - 0.250 = 0.0550 \\[4pt]
\Delta h[3] &= 0.236 \times (-0.0015) + 0.151 \times (-0.0273) + 0.049 \times 0.0550 \\
&= -0.0004 - 0.0041 + 0.0027 = -0.0018 \\[4pt]
h_1[3] &= 0.250 - 0.0018 \approx 0.248
\end{aligned}
$$

#### Irritation（$h[4]$）

$$
\begin{aligned}
\Delta_{\text{coupling}}[4] &= -0.15 \times 0.10 + 0.15 \times 0.22 - 0.08 \times 0.60 \\
&= -0.0150 + 0.0330 - 0.0480 = -0.0300 \\
\Delta_{\text{stimulus}}[4] &= 0.124 \times 0.30 + 0.732 \times 0.05 \\
&= 0.0371 + 0.0366 = 0.0737 \\
\Delta_{\text{homeostatic}}[4] &= 0.090 - 0.100 = -0.0100 \\[4pt]
\Delta h[4] &= 0.236 \times (-0.0300) + 0.151 \times 0.0737 + 0.049 \times (-0.0100) \\
&= -0.0071 + 0.0111 - 0.0005 = 0.0035 \\[4pt]
h_1[4] &= 0.100 + 0.0035 \approx 0.104
\end{aligned}
$$

#### Longing（$h[5]$）

$$
\begin{aligned}
\Delta_{\text{coupling}}[5] &= -0.15 \times 0.40 + 0.15 \times 0.30 = -0.0600 + 0.0450 = -0.0150 \\
\Delta_{\text{stimulus}}[5] &= 0.384 \times 0.18 + 0.505 \times (-0.10) \\
&= 0.0692 - 0.0505 = 0.0187 \\
\Delta_{\text{homeostatic}}[5] &= 0.430 - 0.400 = 0.0300 \\[4pt]
\Delta h[5] &= 0.236 \times (-0.0150) + 0.151 \times 0.0187 + 0.049 \times 0.0300 \\
&= -0.0035 + 0.0028 + 0.0015 = 0.0008 \\[4pt]
h_1[5] &= 0.400 + 0.0008 \approx 0.401
\end{aligned}
$$

#### Social Battery（$h[6]$）

$$
\begin{aligned}
\Delta_{\text{coupling}}[6] &= -0.15 \times 0.60 = -0.0900 \\
\Delta_{\text{stimulus}}[6] &= 0.505 \times (-0.10) + 0.124 \times (-0.25) + 0.253 \times (-0.08) + 0.732 \times (-0.08) \\
&= -0.0505 - 0.0309 - 0.0202 - 0.0586 = -0.1602 \\
\Delta_{\text{homeostatic}}[6] &= 0.590 - 0.600 = -0.0100 \\[4pt]
\Delta h[6] &= 0.236 \times (-0.0900) + 0.151 \times (-0.1602) + 0.049 \times (-0.0100) \\
&= -0.0212 - 0.0242 - 0.0005 = -0.0459 \\[4pt]
h_1[6] &= 0.600 - 0.0459 \approx 0.554
\end{aligned}
$$

#### Mental Fatigue（$h[7]$）

$$
\begin{aligned}
\Delta_{\text{coupling}}[7] &= -0.15 \times 0.15 + 0.10 \times 0.22 - 0.10 \times 0.60 \\
&= -0.0225 + 0.0220 - 0.0600 = -0.0605 \\
\Delta_{\text{stimulus}}[7] &= 0.124 \times 0.25 + 0.373 \times 0.15 \\
&= 0.0309 + 0.0560 = 0.0869 \\
\Delta_{\text{homeostatic}}[7] &= 0.159 - 0.150 = 0.0090 \\[4pt]
\Delta h[7] &= 0.236 \times (-0.0605) + 0.151 \times 0.0869 + 0.049 \times 0.0090 \\
&= -0.0143 + 0.0131 + 0.0004 = -0.0008 \\[4pt]
h_1[7] &= 0.150 - 0.0008 \approx 0.149
\end{aligned}
$$

### 7.5 本轮最终结果

$$\begin{aligned}
h_1 &= [0.690,\; 0.235,\; 0.259,\; 0.248,\; 0.104,\; 0.401,\; 0.554,\; 0.149] \\[2pt]
r_1 &\approx r_0 \quad (\text{关系变化远小于内部，$\beta_{\text{rel}} \ll \beta$}) \\[2pt]
y_1 &\approx [0.568,\; 0.485,\; 0.293,\; 0.537,\; 0.630,\; 0.404,\; 0.081]
\end{aligned}$$

**解释**：
- energy 微降（社交耗电），stress 微升（冲突+情感重量），loneliness 降（closeness+validation 中和），insecurity 基本不变（abandonment 被 validation 抵消），irritation 微升（teasing）
- surface 上 warmth 适中（affection 高 + outer validation 贡献），sharpness 低（irritation 低 + 无冲突），vulnerability 极低（高 pride 压制）

---

## 8. 数值稳定化

文件：`state_engine/_utils.py`

### 8.1 soft_clamp

软饱和裁剪。区间 $[low, high]$ 内直接通过，区间外用 tanh 平滑压回：

$$
\text{output}(x) = \begin{cases}
high - \tau \cdot \tanh\left(\dfrac{x - high}{\tau}\right) & x > high \\[6pt]
low + \tau \cdot \tanh\left(\dfrac{low - x}{\tau}\right) & x < low \\[4pt]
x & \text{otherwise}
\end{cases}
$$

$\tau = 0.1$ 时：$x=1.10 \to 0.9999$（几乎到 1），$x=2.00 \to 1.0000$（饱和）。

### 8.2 sigmoid

数值稳定实现，分正负两段避免 `exp(-x)` 在 $x \ll 0$ 时溢出：

$$
\sigma(x) = \begin{cases}
\dfrac{1}{1 + e^{-x}} & x \geq 0 \\[8pt]
\dfrac{e^x}{1 + e^x} & x < 0
\end{cases}
$$

### 8.3 运行时不变量

| 不变量 | 实现 |
|--------|------|
| 谱半径 < 1 | `validate_matrices()` 检查 A, A_rel |
| 状态 ∈ [0, 1] | 所有更新后经 soft_clamp |
| 速率有界 | α ∈ [0.02, 0.35], β ∈ [0.01, 0.35], γ ∈ [0.01, 0.25] |
| Profiles ∈ [0, 1] | sigmoid 激活 + soft_clamp |
| Δt 显式 | `dt = 1.0`（为 wall-clock 时间预留） |
