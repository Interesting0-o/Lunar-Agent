# Lunar — 带有计算心理学状态机的 AI 角色扮演引擎

基于 LangGraph 构建的 AI 伴侣系统，以《崩坏3》角色「月下誓约·予爱以心」为原型。

**核心理念**：不是用 prompt 咒语驱动角色人格，而是用可计算的多层心理状态机来模拟角色的内在情感变化。

## 架构

```
用户输入 → Perception Node → State Engine → LLM Node → 回复
              │                    │
        社交信号解析          多层状态更新
        (9维 + 4维)         (Surface / Internal
                              Hidden / Relationship)
```

### 四层心理状态模型

| 层 | 字段 | 含义 |
|---|------|------|
| **Surface** | expressiveness, warmth, sharpness... | 用户能看到的情绪表现 |
| **Internal** | energy, stress, loneliness, longing... | 角色真正感受到的情绪 |
| **Hidden** | suppressed_sadness, hidden_affection | 被压抑、不表达的情感 |
| **Relationship** | affection, trust, romantic_tension... | 角色对用户的关系评估 |

**关键机制**：内部状态 ≠ 表面表达。高 pride 角色会压抑好感，高 attachment 角色会放大被抛弃的恐惧——这就是"口是心非"的计算实现。

### 数据流

```
用户输入
  ↓
Perception Node → SocialSignals (affection/attention/intimacy/...) ×9
                → InteractionImpact (emotional_weight/memorability/...) ×4
  ↓
State Engine    → 基于权重规则表 + 特质修饰器更新 4 层状态
                → 检查隐藏情感是否突破阈值 → 触发剧情事件
  ↓
LLM Node        → 注入角色 system prompt，生成回复
```

---

## 状态引擎数学公式

### 状态向量一览

| 向量 | 维度 | 索引常量 | 含义 |
|------|------|----------|------|
| $x$ | 9 | `SS_*` | 社交信号（从用户输入提取） |
| $i$ | 4 | `II_*` | 互动影响（从用户输入提取） |
| $s$ | 7 | `ST_*` | 心理刺激（构造层输出） |
| $g$ | 4 | `G_*` | 门控值 |
| $h_{\text{int}}$ | 8 | `I_*` | 内部心理状态 |
| $h_{\text{rel}}$ | 6 | `R_*` | 关系状态 |
| $h_{\text{hid}}$ | 3 | `H_*` | 隐藏（压抑）状态 |
| $y$ | 7 | `S_*` | 表面表达（动态投影，不存储） |
| $p$ | 10 | `T_*` | 人格特质（稳定参数） |

### 完整流水线

$$
\begin{aligned}
s &= W_{\text{sig2stim}} \; x                     &&\text{① 刺激构造} \\
s &= s \odot \bigl(1 + \Delta p \cdot M_{\text{trait}}\bigr)   &&\text{② 特质调制} \\
s &= s \odot \bigl(1 + h_{\text{rel}} \cdot M_{\text{rel}}\bigr) &&\text{③ 关系调制} \\
g &= \text{gate\_fn}(p, h_{\text{hid}})           &&\text{④ 门控计算} \\
s_g &= \text{gate\_apply}(s, g)                   &&\text{④ 门控应用} \\
h_{\text{int}}' &= A\, h_{\text{int}} + B\, s_g + c(p)      &&\text{⑤ 内部动力系统} \\
h_{\text{rel}}' &= A_{\text{rel}}\, h_{\text{rel}} + B_{\text{rel}}\, s_g + \Delta_{\text{impact}} &&\text{⑤b 关系动力系统} \\
h_{\text{int}}' &= b_{\text{int}} + (h_{\text{int}}' - b_{\text{int}}) \odot d_{\text{int}} &&\text{⑥ 衰减} \\
h_{\text{rel}}' &= b_{\text{rel}} + (h_{\text{rel}}' - b_{\text{rel}}) \odot d_{\text{rel}} \\
\Delta h_{\text{hid}} &= f\bigl(\Delta h_{\text{int}}, s_g, g\bigr) &&\text{⑦ 隐藏积累} \\
h_{\text{hid}}' &= b_{\text{hid}} + (h_{\text{hid}}' - b_{\text{hid}}) \odot d_{\text{hid}} \\
\text{events} &= \text{threshold}(h_{\text{hid}}') &&\text{⑧ 事件触发} \\
y &= \text{project}(h_{\text{int}}', h_{\text{rel}}', h_{\text{hid}}', p, g) &&\text{⑨ 表面投影}
\end{aligned}
$$

---

### ① 刺激构造层

将社交信号映射到心理意义空间：

$$
s_j = \sum_{k} x_k \cdot W_{\text{sig2stim}}[k, j]
$$

其中 $W_{\text{sig2stim}} \in \mathbb{R}^{9 \times 7}$，$s_{\text{emotional\_weight}}$ 直接由 $i_{\text{II\_EMOTIONAL\_WEIGHT}}$ 赋值。

**权重矩阵 $W_{\text{sig2stim}}$ 的非零元素：**

$$
\begin{aligned}
s_{\text{abandonment}} &= 0.7\, x_{\text{rejection}} + 1.2\, x_{\text{abandonment}} \\
s_{\text{validation}}  &= 0.8\, x_{\text{approval}} + 0.3\, x_{\text{affection}} \\
s_{\text{closeness}}   &= 0.9\, x_{\text{intimacy}} + 0.2\, x_{\text{attention}} \\
s_{\text{conflict}}    &= 1.0\, x_{\text{conflict}} + 0.3\, x_{\text{rejection}} \\
s_{\text{dependency}}  &= 0.8\, x_{\text{dependency}} \\
s_{\text{teasing}}     &= 0.7\, x_{\text{teasing}}
\end{aligned}
$$

---

### ② 特质调制层

人格特质放大或衰减心理刺激：

$$
s_j' = s_j \times \Bigl(1 + \sum_{k} (p_k - 0.5) \cdot M_{\text{trait}}[k, j]\Bigr)
$$

其中 $M_{\text{trait}} \in \mathbb{R}^{10 \times 7}$。

**调制矩阵 $M_{\text{trait}}$ 的非零元素：**

$$
\begin{aligned}
M_{\text{trait}}[\text{attachment\_anxiety}, \,\text{abandonment}] &= +0.5 \\
M_{\text{trait}}[\text{attachment\_anxiety}, \,\text{closeness}]    &= +0.3 \\
M_{\text{trait}}[\text{jealousy\_sensitivity}, \,\text{abandonment}] &= +0.4 \\
M_{\text{trait}}[\text{jealousy\_sensitivity}, \,\text{teasing}]    &= +0.2 \\
M_{\text{trait}}[\text{anger\_reactivity}, \,\text{conflict}]      &= +0.5 \\
M_{\text{trait}}[\text{anger\_reactivity}, \,\text{abandonment}]   &= +0.2 \\
M_{\text{trait}}[\text{pride}, \,\text{validation}]                &= -0.2 \\
M_{\text{trait}}[\text{pride}, \,\text{teasing}]                   &= +0.3 \\
M_{\text{trait}}[\text{emotional\_stability}, \,\text{conflict}]   &= -0.3 \\
M_{\text{trait}}[\text{emotional\_stability}, \,\text{abandonment}] &= -0.2
\end{aligned}
$$

系数为正 $\Rightarrow$ 该特质越高刺激越强；为负 $\Rightarrow$ 该特质越高刺激越弱。

---

### ③ 关系调制层

关系状态改变心理刺激的"含义"：

$$
s_j' = s_j \times \Bigl(1 + \sum_{k} h_{\text{rel}}[k] \cdot M_{\text{rel}}[k, j]\Bigr)
$$

其中 $M_{\text{rel}} \in \mathbb{R}^{6 \times 7}$。

**调制矩阵 $M_{\text{rel}}$ 的非零元素：**

$$
\begin{aligned}
M_{\text{rel}}[\text{emotional\_safety}, \,\text{abandonment}] &= -0.5 \\
M_{\text{rel}}[\text{affection}, \,\text{validation}]          &= +0.3 \\
M_{\text{rel}}[\text{emotional\_safety}, \,\text{closeness}]   &= +0.2 \\
M_{\text{rel}}[\text{trust}, \,\text{conflict}]                &= -0.3 \\
M_{\text{rel}}[\text{dependency}, \,\text{dependency}]         &= +0.3
\end{aligned}
$$

---

### ④ 门控层

#### 门控计算

四个门控值由特质和隐藏状态计算：

$$
\begin{aligned}
g_{\text{suppression}} &= \operatorname{clip}\bigl(0.4\,p_{\text{pride}} + 0.3\,(1 - p_{\text{openness}}) + 0.3\,(1 - p_{\text{stability}}) + 0.2\,h_{\text{suppressed\_sadness}} + 0.3\,h_{\text{suppressed\_anger}} + 0.1\,h_{\text{hidden\_affection}},\; 0,\; 1\bigr) \\
g_{\text{vulnerability}} &= \operatorname{clip}\bigl(0.5\,(1 - p_{\text{pride}}) + 0.3\,p_{\text{openness}} + 0.2\,p_{\text{sensitivity}},\; 0,\; 1\bigr) \\
g_{\text{attachment}} &= \operatorname{clip}\bigl(0.6\,p_{\text{attachment\_anxiety}} + 0.4\,(1 - p_{\text{attachment\_avoidance}}),\; 0,\; 1\bigr) \\
g_{\text{leakage}} &= \operatorname{clip}\bigl(1.2 \cdot \frac{1}{3}(h_{\text{suppressed\_sadness}} + h_{\text{suppressed\_anger}} + h_{\text{hidden\_affection}}) - 0.2,\; 0,\; 1\bigr)
\end{aligned}
$$

#### 门控应用

$$
\begin{aligned}
s_g &= s \times (1 - 0.6\,g_{\text{suppression}}) \\
s_g[\text{abandonment}] &= s_g[\text{abandonment}] \times g_{\text{attachment}} \\
s_g[\text{closeness}]   &= s_g[\text{closeness}] \times g_{\text{attachment}} \\
s_g[\text{validation}]  &= s_g[\text{validation}] \times (0.5 + 0.5\,g_{\text{vulnerability}}) \\
s_g[\text{closeness}]   &= s_g[\text{closeness}] \times (0.5 + 0.5\,g_{\text{vulnerability}})
\end{aligned}
$$

---

### ⑤ 内部动力系统

#### ⑤a 内部状态

$$
h_{\text{int}}' = \operatorname{clip}\bigl(A\, h_{\text{int}} + B\, s_g + c(p),\; 0,\; 1\bigr)
$$

**状态耦合矩阵 $A \in \mathbb{R}^{8 \times 8}$：**

$$
\begin{aligned}
A[\text{irritation},\,\text{stress}]              &= +0.15 \\
A[\text{mental\_fatigue},\,\text{stress}]          &= +0.10 \\
A[\text{loneliness},\,\text{stress}]               &= +0.08 \\
A[\text{insecurity},\,\text{loneliness}]           &= +0.12 \\
A[\text{longing},\,\text{loneliness}]              &= +0.15 \\
A[\text{mental\_fatigue},\,\text{social\_battery}] &= -0.10 \\
A[\text{irritation},\,\text{social\_battery}]      &= -0.08 \\
A[\text{stress},\,\text{energy}]                   &= -0.05 \\
A[\text{loneliness},\,\text{energy}]               &= -0.05 \\
A[\text{stress},\,\text{insecurity}]               &= +0.10 \\
A[i,i] &= 0.85 \quad \forall i \quad \text{(自保持 / 惯性)}
\end{aligned}
$$

**输入影响矩阵 $B \in \mathbb{R}^{7 \times 8}$：**

$$
\begin{aligned}
B[\text{abandonment},\,\text{insecurity}]        &= +0.30 \\
B[\text{abandonment},\,\text{loneliness}]         &= +0.20 \\
B[\text{abandonment},\,\text{stress}]             &= +0.15 \\
B[\text{abandonment},\,\text{longing}]            &= +0.20 \\
B[\text{abandonment},\,\text{energy}]             &= -0.15 \\
B[\text{validation},\,\text{insecurity}]          &= -0.20 \\
B[\text{validation},\,\text{energy}]              &= +0.15 \\
B[\text{validation},\,\text{loneliness}]           &= -0.15 \\
B[\text{closeness},\,\text{loneliness}]            &= -0.25 \\
B[\text{closeness},\,\text{longing}]              &= -0.10 \\
B[\text{closeness},\,\text{social\_battery}]       &= -0.10 \\
B[\text{closeness},\,\text{energy}]               &= +0.08 \\
B[\text{conflict},\,\text{stress}]                &= +0.35 \\
B[\text{conflict},\,\text{irritation}]            &= +0.30 \\
B[\text{conflict},\,\text{energy}]                &= -0.20 \\
B[\text{conflict},\,\text{mental\_fatigue}]       &= +0.25 \\
B[\text{conflict},\,\text{social\_battery}]        &= -0.25 \\
B[\text{dependency},\,\text{social\_battery}]      &= -0.08 \\
B[\text{dependency},\,\text{loneliness}]           &= -0.15 \\
B[\text{dependency},\,\text{energy}]              &= +0.05 \\
B[\text{teasing},\,\text{social\_battery}]         &= -0.08 \\
B[\text{teasing},\,\text{irritation}]             &= +0.05 \\
B[\text{teasing},\,\text{energy}]                 &= +0.05 \\
B[\text{emotional\_weight},\,\text{stress}]        &= +0.20 \\
B[\text{emotional\_weight},\,\text{mental\_fatigue}] &= +0.15
\end{aligned}
$$

**人格偏置向量 $c \in \mathbb{R}^{8}$：**

$$
c = [0.01,\; 0,\; -0.005,\; 0,\; -0.01,\; 0,\; 0,\; 0]
$$

随特质动态调整：

$$
\begin{aligned}
c[\text{energy}]      &+= (p_{\text{optimism}} - 0.5) \times 0.02 \\
c[\text{stress}]      &-= (p_{\text{optimism}} - 0.5) \times 0.01 \\
c[\text{stress}]      &+= (p_{\text{anxiety}} - 0.5) \times 0.02 \\
c[\text{insecurity}]  &+= (p_{\text{anxiety}} - 0.5) \times 0.01
\end{aligned}
$$

#### ⑤b 关系状态

$$
h_{\text{rel}}' = \operatorname{clip}\bigl(A_{\text{rel}}\, h_{\text{rel}} + B_{\text{rel}}\, s_g + \Delta_{\text{impact}},\; 0,\; 1\bigr)
$$

**关系状态耦合矩阵 $A_{\text{rel}} \in \mathbb{R}^{6 \times 6}$：**

$$
\begin{aligned}
A_{\text{rel}}[\text{trust},\,\text{affection}]                &= +0.08 \\
A_{\text{rel}}[\text{familiarity},\,\text{affection}]          &= +0.05 \\
A_{\text{rel}}[\text{emotional\_safety},\,\text{trust}]        &= +0.10 \\
A_{\text{rel}}[\text{dependency},\,\text{trust}]               &= +0.05 \\
A_{\text{rel}}[\text{emotional\_safety},\,\text{familiarity}]  &= +0.08 \\
A_{\text{rel}}[\text{affection},\,\text{emotional\_safety}]    &= +0.05 \\
A_{\text{rel}}[\text{trust},\,\text{emotional\_safety}]        &= +0.05 \\
A_{\text{rel}}[\text{affection},\,\text{romantic\_tension}]    &= +0.03 \\
A_{\text{rel}}[\text{romantic\_tension},\,\text{dependency}]   &= +0.05 \\
A_{\text{rel}}[i,i] &= 0.90 \quad \forall i
\end{aligned}
$$

**关系输入影响矩阵 $B_{\text{rel}} \in \mathbb{R}^{7 \times 6}$：**

$$
\begin{aligned}
B_{\text{rel}}[\text{abandonment},\,\text{trust}]              &= -0.12 \\
B_{\text{rel}}[\text{abandonment},\,\text{emotional\_safety}]   &= -0.15 \\
B_{\text{rel}}[\text{abandonment},\,\text{romantic\_tension}]  &= +0.08 \\
B_{\text{rel}}[\text{abandonment},\,\text{dependency}]         &= +0.10 \\
B_{\text{rel}}[\text{validation},\,\text{affection}]           &= +0.12 \\
B_{\text{rel}}[\text{validation},\,\text{trust}]               &= +0.10 \\
B_{\text{rel}}[\text{closeness},\,\text{affection}]            &= +0.10 \\
B_{\text{rel}}[\text{closeness},\,\text{familiarity}]          &= +0.12 \\
B_{\text{rel}}[\text{closeness},\,\text{emotional\_safety}]    &= +0.08 \\
B_{\text{rel}}[\text{closeness},\,\text{romantic\_tension}]    &= +0.06 \\
B_{\text{rel}}[\text{conflict},\,\text{trust}]                 &= -0.18 \\
B_{\text{rel}}[\text{conflict},\,\text{emotional\_safety}]     &= -0.20 \\
B_{\text{rel}}[\text{conflict},\,\text{affection}]             &= -0.08 \\
B_{\text{rel}}[\text{conflict},\,\text{romantic\_tension}]     &= -0.08 \\
B_{\text{rel}}[\text{dependency},\,\text{dependency}]          &= +0.18 \\
B_{\text{rel}}[\text{dependency},\,\text{familiarity}]         &= +0.06 \\
B_{\text{rel}}[\text{dependency},\,\text{emotional\_safety}]   &= +0.05 \\
B_{\text{rel}}[\text{teasing},\,\text{familiarity}]            &= +0.08 \\
B_{\text{rel}}[\text{teasing},\,\text{romantic\_tension}]      &= +0.08
\end{aligned}
$$

**Impact 直接效应 $\Delta_{\text{impact}}$：**

$$
\begin{aligned}
\text{if } i_{\text{closeness}} > 0 &: \quad h_{\text{rel}}'[\text{familiarity}] += 0.15\,i_{\text{closeness}},\; h_{\text{rel}}'[\text{emotional\_safety}] += 0.12\,i_{\text{closeness}} \\
\text{if } i_{\text{closeness}} < 0 &: \quad h_{\text{rel}}'[\text{emotional\_safety}] += 0.15\,i_{\text{closeness}},\; h_{\text{rel}}'[\text{trust}] += 0.10\,i_{\text{closeness}} \\
\text{if } i_{\text{trust}} \neq 0 &: \quad h_{\text{rel}}'[\text{trust}] += 0.15\,i_{\text{trust}}
\end{aligned}
$$

---

### ⑥ 衰减层

各维度不同衰减速率，向基线回归而非归零：

$$
h' = b + (h - b) \odot d
$$

**内部状态衰减向量 $d_{\text{int}} \in \mathbb{R}^{8}$：**

$$
d_{\text{int}} = [0.98,\; 0.92,\; 0.95,\; 0.95,\; 0.85,\; 0.97,\; 0.93,\; 0.90]
$$

值越大衰减越慢：irritation (0.85) 消退最快，energy (0.98) 恢复最慢。

**关系状态衰减向量 $d_{\text{rel}} \in \mathbb{R}^{6}$：**

$$
d_{\text{rel}} = [0.995,\; 0.990,\; 0.985,\; 0.980,\; 0.990,\; 0.970]
$$

关系衰减极慢：affection (0.995) 几乎不衰减，tension (0.970) 相对快。

**隐藏状态衰减向量 $d_{\text{hid}} \in \mathbb{R}^{3}$：**

$$
d_{\text{hid}} = [0.93,\; 0.90,\; 0.95]
$$

---

### ⑦ 隐藏积累层

从内部状态变化推导"原始情绪"，再乘以压抑门进入隐藏层：

$$
\begin{aligned}
\Delta h_{\text{int}} &= h_{\text{int}}' - h_{\text{int}} \\
r_{\text{sadness}} &= \max(0, \Delta h_{\text{int}}[\text{loneliness}]) + 0.5 \cdot \max(0, \Delta h_{\text{int}}[\text{stress}]) \\
r_{\text{anger}} &= \max(0, \Delta h_{\text{int}}[\text{irritation}]) \\
r_{\text{affection}} &= 0.15\,(s_g[\text{closeness}] + s_g[\text{validation}]) \\
\Delta h_{\text{hid}}[\text{suppressed\_sadness}] &= 0.3 \cdot r_{\text{sadness}} \cdot g_{\text{suppression}} \\
\Delta h_{\text{hid}}[\text{suppressed\_anger}]   &= 0.3 \cdot r_{\text{anger}} \cdot g_{\text{suppression}} \\
\Delta h_{\text{hid}}[\text{hidden\_affection}]   &= r_{\text{affection}} \cdot g_{\text{suppression}}
\end{aligned}
$$

高自尊额外压抑好感：

$$
\text{if } p_{\text{pride}} > 0.6: \quad
\Delta h_{\text{hid}}[\text{hidden\_affection}] +\!= 0.3 \cdot r_{\text{affection}} \cdot 2\,(p_{\text{pride}} - 0.6)
$$

---

### ⑧ 事件触发层

$$
\begin{aligned}
h_{\text{hid}}[\text{hidden\_affection}] > 0.85 &\Rightarrow \text{"AFFECTION\_BREAKTHROUGH"} \\
h_{\text{hid}}[\text{suppressed\_sadness}] > 0.85 &\Rightarrow \text{"SADNESS\_BREAKTHROUGH"} \\
h_{\text{hid}}[\text{suppressed\_anger}] > 0.80 &\Rightarrow \text{"ANGER\_BREAKTHROUGH"} \\
h_{\text{hid}}[\text{suppressed\_sadness}] > 0.6 \land p_{\text{attachment\_anxiety}} > 0.6
\land h_{\text{hid}}[\text{hidden\_affection}] > 0.5 &\Rightarrow \text{"CLINGY\_BREAKTHROUGH"}
\end{aligned}
$$

---

### ⑨ 表面投影层

表面表达由内部状态动态投影生成，不存储：

$$
\begin{aligned}
y_{\text{expressiveness}} &= 0.3 + 0.4\,h_{\text{int}}[\text{energy}] - 0.3\,h_{\text{int}}[\text{mental\_fatigue}] \\
y_{\text{warmth}} &= 0.3 + 0.4\,h_{\text{rel}}[\text{affection}] - 0.2\,h_{\text{int}}[\text{stress}] \\
y_{\text{sharpness}} &= 0.1 + 0.5\,h_{\text{int}}[\text{irritation}] + 0.2\,h_{\text{int}}[\text{stress}] \\
y_{\text{softness}} &= 0.2 + 0.3\,(1 - h_{\text{int}}[\text{stress}]) + 0.2\,h_{\text{rel}}[\text{emotional\_safety}] \\
y_{\text{enthusiasm}} &= 0.3 + 0.5\,h_{\text{int}}[\text{energy}] - 0.3\,h_{\text{int}}[\text{mental\_fatigue}] \\
y_{\text{restraint}} &= 0.2 + 0.3\,h_{\text{int}}[\text{insecurity}] + 0.2\,p_{\text{pride}} \\
y_{\text{vulnerability}} &= 0.1 + 0.3\,h_{\text{int}}[\text{loneliness}] + 0.2\,h_{\text{int}}[\text{longing}] - 0.2\,p_{\text{pride}}
\end{aligned}
$$

随后经隐藏泄漏效应和特质修饰调整（见 `project_surface()` 代码）。

## 技术栈

| 组件 | 技术 |
|------|------|
| 图编排 | LangGraph |
| 模型调用 | LangChain + Ollama |
| 本地模型 | qwen2.5:7b |
| 状态持久化 | SQLite (LangGraph Checkpointer) |
| Python | 3.13 |

## 项目结构

```
Lunar/
├── agent.py              # LangGraph 图定义与编译
├── nodes.py              # 四个图节点（inject_system / perception / state_engine / llm）
├── perception.py         # 感知层：社交信号提取 + 验证 + 重试
├── state_engine.py       # 状态引擎：权重规则表 + 特质修饰器 + 压抑/突破机制
├── state.py              # 状态类型定义（TypedDict）
├── model.py              # LLM 模型初始化
├── config.py             # 运行时配置（重试策略等）
├── character_prompt.py   # 角色人设 SYSTEM_PROMPT
├── perception_prompt.py  # 感知层提示词
├── main.py               # 入口
└── db/                   # SQLite 持久化
```

## 快速开始

```bash
# 1. 安装依赖
pip install langgraph langchain langchain-ollama

# 2. 拉取模型（确保 Ollama 已运行）
ollama pull qwen2.5:7b

# 3. 运行
python agent.py
```

## 已知局限 & 改进路线

| 优先级 | 问题 | 方案 |
|--------|------|------|
| P0 | SurfaceState 已注入但方法原始 | 见下方「State Formatter 重设计划」 |
| P0 | 突破事件无实际效果 | 事件触发时修改 prompt 模板或切换回复策略 |
| P1 | 硬编码权重无法学习 | 权重表外部化为 JSON 参数文件，支持用户反馈微调 |
| P1 | 上下文窗口仅 4 条 | 扩展至 10~20 + 接入 Chroma 向量检索 |
| P1 | Checkpointer 未集成 | `graph.compile(checkpointer=saver)` |
| P2 | 无用户心理模型 | 增加 UserProfile 状态层 |

### State Formatter 重设计划（当前 P0）

#### 核心认知

`State → LLM` 是整个系统最难的部分之一——**数值状态 ≠ 自然语言行为**。

现有 `state_formatter.py` 使用 5 级阈值 `_desc()` + 平铺列举的方式，本质上是**把连续人格系统重新离散化**。这会瞬间破坏 State Engine 连续动力学的核心优势。

相反，formatter 不应是 `if-else` 规则引擎，而应是 **Semantic Projection Layer（语义投影层）**——将高维状态向量连续地投影到 LLM 擅长理解的"行为倾向语义空间"。

#### 三层人格系统

```
Layer 1 — State Engine:      角色现在真实处于什么心理状态（连续向量）
Layer 2 — Formatter [★]:     心理 → 行为语义投影（"导演笔记"）
Layer 3 — LLM:               行为 → 自然语言生成
```

Formatter 不应输出"情绪标签"（如 sadness=0.7），而应输出**行为倾向描述**（如"角色会刻意让语气显得平静，但偶尔会流露失落感"）。LLM 擅长模仿行为倾向，不擅长解析高维数值空间。

#### 当前实现的问题

当前 `_desc()` 的 5 级阈值把连续值拉到 5 个离散桶里，就像：

```python
# ❌ 当前做法（离散化）
if value < 0.15: return "极低"
elif value < 0.35: return "偏低"   # 0.34 和 0.36 差一个等级，但实际只差 0.02
```

即使升级到 100 级，这仍然是离散映射——应该让每个维度按原始值连续贡献描述。

#### 重设计方向：权重式连续语义生成

不写条件分支树，而是让每个维度按自身强度**连续贡献**心理描述片段，按权重（即该维度的当前值）排序，取 TOP-K 拼接：

```python
# ✅ 推荐方向（连续贡献）
descriptions = []

if affection > 0.3:   # 不是阈值，而是激活下限
    descriptions.append((affection, "角色明显在意你"))

if pride > 0.4:
    descriptions.append((pride, "会下意识维持自尊，不会轻易示弱"))

if stress > 0.4:
    descriptions.append((stress, "情绪上有一定紧绷感，不是完全放松的状态"))

# 按权重排序取 TOP-K，拼接为自然段
```

这样得到：
- **连续性**——affection 0.61 和 0.72 输出不同强度的同一条描述，而不是"0.55 以下是好感，以上是喜欢"
- **组合性**——任意维度的组合自然表达，无需为每种组合写分支
- **可解释性**——每条描述可追溯到对应的状态维度
- **可扩展性**——加新维度只需加一条 `append`

#### 输出格式：导演笔记（非情绪标签）

Formatter 输出应为 LLM 的"导演笔记"，而非状态 JSON。推荐分层但不强求：

```
Layer 1 — 核心心理态势（1~2句概括）
  "角色目前对用户存在明显依赖倾向，但表达上仍然偏克制。"

Layer 2 — 语气倾向（当前回应的风格提示）
  "温度偏高，有轻微防御感，不会直接暴露脆弱。"

Layer 3 — 表达限制（说与不说的边界）
  "避免直接表达「我需要你」，避免明确示弱。"

Layer 4 — 微行为建议（可选，引导 LLM）
  "可以轻微转移话题，或用调侃掩饰在意。"
```

输出中的每一个信号都应该是**多层状态共同投影的结果**，而非单个维度的翻译：
- 好感度 → 单独看决定"是否在意"
- 好感度 + 自尊心 → 共同决定"在意但说不说"
- 好感度 + 自尊心 + 不安全感 → 共同决定"在意但用防御还是退缩表达"

#### 实施路线

**Step 1（近期）— 连续性改造**
- 删除 `_desc()` 5 级阈值系统
- 改为权重式连续描述生成
- 每维度设置激活下限（而非阈值分支），按值连续贡献权重
- TOP-K 排序拼接，取 5~8 条
- 加入多样性约束：正负向维度至少都有覆盖
- 输出不再显示原始数值

**Step 2（验证后）— 语义结构化**
- 引入中间层 `Semantic Traits`（如 `warm_behavior`, `defensive_behavior` 等），
  将原始状态向量投影到这个更贴近自然语言的语义空间
- 输出格式正式分层（核心态势 / 语气倾向 / 表达限制 / 微行为建议）
- 考虑用 Mini-LLM 或简单线性层替代手工映射函数
- 评估 LLM 对不同格式输出的响应稳定性

## License

MIT
