# Lunar — 带有计算心理学状态机的 AI 角色扮演引擎

基于 LangGraph 构建的 AI 伴侣系统，以《崩坏3》角色「月下誓约·予爱以心」为原型。

**核心理念**：不是用 prompt 咒语驱动角色人格，而是用可计算的多层心理状态机来模拟角色的内在情感变化。

## 架构

```
用户输入 → Perception Node → State Engine → State Formatter → LLM Node → 回复
              │                    │
        心理刺激提取          多层状态更新
        (7维 Stimulus)       (Internal / Relationship
                              → Surface 投影)
```

### 三层心理状态模型

| 层 | 字段 | 含义 |
|---|------|------|
| **Internal** | energy, stress, loneliness, longing... | 角色真正感受到的情绪 |
| **Relationship** | affection, trust, romantic_tension... | 角色对用户的关系评估 |
| **Surface** | expressiveness, warmth, sharpness... | 用户能看到的情绪表现（动态投影，不存储） |

**关键机制**：内部状态 ≠ 表面表达。高 pride 角色会压抑好感，高 attachment 角色会放大被抛弃的恐惧——这就是"口是心非"的计算实现。

### 数据流

```
用户输入
  ↓
Perception Node → StimulusVector (abandonment/validation/closeness/...) ×7
                → LLM 一步到位输出心理刺激强度，不再分两步（社交信号→线性构造）
  ↓
State Engine    → 6 层纯函数 pipeline：特质调制→关系调制→门控→动力系统→衰减→表面投影
  ↓
State Formatter → 数值状态 → 中文"导演笔记"
  ↓
LLM Node        → 注入角色 system prompt + 状态描述，生成回复
```

---

## 状态引擎数学公式

### 状态向量一览

| 向量 | 维度 | 索引常量 | 含义 |
|------|------|----------|------|
| $s$ | 7 | `ST_*` | 心理刺激（perception 直接输出） |
| $g$ | 4 | `G_*` | 门控值 |
| $h_{\text{int}}$ | 8 | `I_*` | 内部心理状态 |
| $h_{\text{rel}}$ | 6 | `R_*` | 关系状态 |
| $y$ | 7 | `S_*` | 表面表达（动态投影，不存储） |
| $p$ | 10 | `T_*` | 人格特质（稳定参数） |

### 完整流水线

$$
\begin{aligned}
s &= s \odot \bigl(1 + \Delta p \cdot M_{\text{trait}}\bigr)   &&\text{① 特质调制} \\
s &= s \odot \bigl(1 + h_{\text{rel}} \cdot M_{\text{rel}}\bigr) &&\text{② 关系调制} \\
g &= \text{gate\_fn}(p, h_{\text{rel}}, h_{\text{int}})           &&\text{③ 门控计算} \\
s_g &= \text{gate\_apply}(s, g)                   &&\text{③ 门控应用} \\
h_{\text{int}}' &= A\, h_{\text{int}} + B\, s_g + c(p)      &&\text{④ 内部动力系统} \\
h_{\text{rel}}' &= A_{\text{rel}}\, h_{\text{rel}} + B_{\text{rel}}\, s_g &&\text{④b 关系动力系统} \\
h_{\text{int}}' &= b_{\text{int}} + (h_{\text{int}}' - b_{\text{int}}) \odot d_{\text{int}} &&\text{⑤ 衰减} \\
h_{\text{rel}}' &= b_{\text{rel}} + (h_{\text{rel}}' - b_{\text{rel}}) \odot d_{\text{rel}} \\
y &= \text{project}(h_{\text{int}}', h_{\text{rel}}', p) &&\text{⑥ 表面投影}
\end{aligned}
$$

---

### ① 特质调制层

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

### ② 关系调制层

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

### ③ 门控层

Gate 是角色"潜意识"的核心防线，由三层输入共同决定：

$$
g = \operatorname{clip}\bigl(\underbrace{g_{\text{trait}}}_{\text{人格基线}} \times \underbrace{m_{\text{rel}}}_{\text{关系调制}} \;+\; \underbrace{\Delta_{\text{int}}}_{\text{内部推动}},\; 0,\; 1\bigr)
$$

- **trait 基线**：人格决定的稳定防御底色
- **rel 调制**：关系状态对基线的调制（信任→松动防御，好感→放大依恋）
- **internal 推动**：当前心理状态的急性 push（压力→更压抑，孤独→更渴望表达）

#### 门控计算

$$
\begin{aligned}
g_{\text{suppression}} &= \operatorname{clip}\bigl(
    \underbrace{(0.4\,p_{\text{pride}} + 0.3\,(1-p_{\text{openness}}) + 0.3\,(1-p_{\text{stability}}))}_{\text{trait 基线}}
    \times \underbrace{(1 - 0.20\,h_{\text{rel}}[\text{trust}] - 0.15\,h_{\text{rel}}[\text{emotional\_safety}])}_{\text{关系调制}}
    \;+\; \underbrace{0.10\,h_{\text{int}}[\text{stress}] + 0.08\,h_{\text{int}}[\text{insecurity}]}_{\text{内部推动}},\; 0,\; 1\bigr) \\[12pt]
g_{\text{vulnerability}} &= \operatorname{clip}\bigl(
    \underbrace{(0.5\,(1-p_{\text{pride}}) + 0.3\,p_{\text{openness}} + 0.2\,p_{\text{sensitivity}})}_{\text{trait 基线}}
    \times \underbrace{(1 + 0.15\,h_{\text{rel}}[\text{emotional\_safety}] + 0.10\,h_{\text{rel}}[\text{familiarity}])}_{\text{关系调制}}
    \;+\; \underbrace{0.12\,h_{\text{int}}[\text{loneliness}] + 0.10\,h_{\text{int}}[\text{longing}]}_{\text{内部推动}},\; 0,\; 1\bigr) \\[12pt]
g_{\text{attachment}} &= \operatorname{clip}\bigl(
    \underbrace{(0.6\,p_{\text{attachment\_anxiety}} + 0.4\,(1-p_{\text{attachment\_avoidance}}))}_{\text{trait 基线}}
    \times \underbrace{(1 + 0.12\,h_{\text{rel}}[\text{affection}] + 0.08\,h_{\text{rel}}[\text{romantic\_tension}])}_{\text{关系调制}}
    \;+\; \underbrace{0.10\,h_{\text{int}}[\text{insecurity}] + 0.08\,h_{\text{int}}[\text{longing}]}_{\text{内部推动}},\; 0,\; 1\bigr) \\[12pt]
g_{\text{leakage}} &= 0 \quad \text{(HiddenState 已移除)}
\end{aligned}
$$

#### 动态效果示例

以压抑门为例——随着关系深入，防御逐渐松动：

| 阶段 | $p_{\text{pride}}$ | $h_{\text{rel}}[\text{trust}]$ | $h_{\text{rel}}[\text{safety}]$ | trait 基线 | rel 调制 | 最终 $g_{\text{supp}}$ |
|------|-------------------|-------------------------------|--------------------------------|-----------|---------|----------------------|
| 初识 | 0.7 | 0.20 | 0.15 | 0.74 | ×0.94 | 0.69 |
| 熟悉 | 0.7 | 0.50 | 0.45 | 0.74 | ×0.83 | 0.62 |
| 亲密 | 0.7 | 0.80 | 0.75 | 0.74 | ×0.73 | 0.54 |

角色仍然受自尊心驱使而防御（trait 基线不变），但不再是铁板一块——信任和安全感积累后，防线自然松动。

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

### ④ 内部动力系统

#### ④a 内部状态

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

#### ④b 关系状态

$$
h_{\text{rel}}' = \operatorname{clip}\bigl(A_{\text{rel}}\, h_{\text{rel}} + B_{\text{rel}}\, s_g,\; 0,\; 1\bigr)
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

---

### ⑤ 衰减层

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

---

### ⑥ 表面投影层

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

随后经特质修饰调整（见 `project_surface()` 代码）。

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
├── nodes.py              # 五个图节点（inject_system / perception / state_engine / state_formatter / llm）
├── perception.py         # 感知层：心理刺激提取 + 验证 + 重试
├── state_engine.py       # 状态引擎：6 层纯函数 pipeline（无线性构造层）
├── state_formatter.py    # 状态格式化：数值状态 → 中文"导演笔记"
├── state.py              # 状态类型定义（TypedDict + 索引常量）
├── default_state.py      # 默认基线值（traits / internal / relationship）
├── model.py              # LLM 模型初始化
├── config.py             # 运行时配置（重试策略等）
├── character_prompt.py   # 角色人设 SYSTEM_PROMPT
├── perception_prompt.py  # 感知层提示词（直接输出 7 维心理刺激）
├── main.py               # FastAPI 入口（stub）
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

### 优先级总览

| 优先级 | 问题 | 涉及层 | 方案概要 |
|--------|------|--------|---------|
| P0 | SurfaceState 注入方法原始 | state_formatter.py | 见下方「State Formatter 重设计划」 |
| P0 | 门控恒定不变 | ④ Gate Control | ✅ **已修复** — `compute_gates()` 现在接收 traits + relationship + current_internal，门控随关系深入动态变化 |
| P0 | 衰减系数静态 | ⑥ Decay | 改为动态 decay，由 traits/relationship/gated_stimuli 每轮计算 |
| P0 | **门控与衰减深层矛盾** | **④ Gate + ⑥ Decay** | **统一 defense 参数同步驱动门控与衰减，消除"压抑但忘得快"的矛盾组合** |
| P1 | 感知层缺少角色状态上下文 | ① Perception | 将角色当前状态摘要注入感知模型的上下文 |
| P1 | ~~刺激构造层在当前实现中冗余~~ | ~~① Stimulus Construction~~ | ✅ **已移除** — perception 现在直接输出 7 维 StimulusVector |
| P1 | 表面投影无时间惯性 | ⑥ Surface Projection | 引入表面状态惯性项，使表达变化有滞后 |
| P1 | 硬编码权重无法学习 | 全局 | 权重外部化为 JSON 参数文件 |
| P1 | 上下文窗口仅 4 条 | perception.py | 扩展至 10~20 + 接入 Chroma 向量检索 |
| P2 | 无用户心理模型 | 全局 | 增加 UserProfile 状态层 |
| P2 | **三个状态缺少完整消息视野** | **全局** | **见下方「RNN 视野限制」** |

---

### P0: 门控恒定不变 ✅ 已修复

#### 问题

~~`compute_gates()` 仅以 `traits` 为输入，traits 在对话中几乎不变化，导致每轮门控值完全一致。~~ 已通过引入 relationship 和 internal 解决。

#### 已实施方案

`compute_gates()` 现在接收三个输入：

```python
def compute_gates(
    traits: np.ndarray,
    relationship: np.ndarray,
    current_internal: np.ndarray,
) -> np.ndarray:
```

三层输入的职责：
- **traits**：人格基线（稳定，决定防御的"底色"）
- **relationship**：关系调制（信任/安全感 → 松动压抑，好感 → 放大依恋）
- **current_internal**：内部推动（压力/不安 → 更压抑，孤独/渴望 → 更愿示弱）

公式模式：`gate = clip(trait_baseline × rel_modulation + internal_push, 0, 1)`

效果示例（压抑门，高自尊角色）：
```
初识: R_Trust=0.20 → g_supp=0.69
熟悉: R_Trust=0.50 → g_supp=0.62
亲密: R_Trust=0.80 → g_supp=0.54
```

角色仍然受自尊心驱使而防御（trait 基线不变），但不再是铁板一块——信任和安全感积累后，防线自然松动。

---

### P0: 衰减系数静态

#### 问题

当前的 decay 系数是固定常数：

```python
INTERNAL_DECAY = [0.98, 0.92, 0.95, 0.95, 0.85, 0.97, 0.93, 0.90]
```

不依赖人格、不依赖事件、不依赖语境。这意味着：
- 一次冲突让 stress=0.8，第二天掉到 0.75——即使角色很记仇（高 Anger_Reactivity + 高 Pride），消气速度也和一碰就忘的角色一样
- 「是否已经被安抚」「是否还在等道歉」「这段关系的信任有多深」——这些信息完全不参与衰减

#### 方案

改为每轮动态计算衰减系数：

```python
def compute_dynamic_decay(
    base_decay: np.ndarray,
    traits: np.ndarray,
    relationship: np.ndarray,
    gated_stimuli: np.ndarray,
) -> np.ndarray:
```

核心逻辑（以 stress 为例）：
```python
# 冲突后，记仇角色衰减极慢（对方没道歉就不原谅）
if gated_stimuli[ST_CONFLICT] > 0.3:
    decay[I_STRESS] -= traits[T_PRIDE] * 0.08
    decay[I_STRESS] -= traits[T_ANGER_REACTIVITY] * 0.06

# 收到道歉（validation 高、conflict 低）→ 加速消解
if gated_stimuli[ST_VALIDATION] > 0.4 and gated_stimuli[ST_CONFLICT] < 0.2:
    decay[I_STRESS] += 0.05

# 高信任 → 更易释怀
decay[I_STRESS] += relationship[R_TRUST] * 0.03
```

**时间线对比（记仇角色，未道歉）：**
```
固定 decay:   stress=0.80 → 0.75 → 0.71 → 0.63（5 轮后基本没事）
动态 decay:   stress=0.80 → 0.78 → 0.77 → 0.74（持续压着，等道歉后才会松动）
```

扩展到其他维度：
- **irritation**：abandonment 刺激→依恋焦虑高的人烦躁持续更久
- **loneliness**：独处时（无 closeness 刺激）依恋焦虑高→孤独衰减更慢
- **关系维度**：冲突期间信任和安全感衰减暂停；和解后恢复正常

---

### P0: 门控与衰减的深层矛盾

#### 问题

门控和衰减在各自的 pipeline 位置中处理不同对象：

```
④ apply_gates(stimuli, gates)    → 处理输入：决定"多少能进来"
⑤ h_t = A·h_{t-1} + B·gated + c  → 状态更新
⑥ apply_decay(state, decay)      → 处理存量：决定"旧的留多久"
```

两者在代码中完全独立，但心理上它们控制的是同一个防御机制的两面：

| | 门控（输入侧） | 衰减（保持侧） |
|--|-------------|-------------|
| 心理隐喻 | "要不要在意" | "能不能放下" |
| 控制对象 | 刺激能否进入 | 已有情绪能留多久 |

**当前实现会产生心理上自相矛盾的组合：**

```
高压抑(门控高) + 标准衰减 = 进得少但忘得和普通人一样快
```

心理上，一个高压抑的角色应该：
- 很少让外界刺激穿透防御 → 刺激进得少 ✅（门控做到了）
- 但进来的那些不会轻易消化 → 应该久久放不下 ❌（衰减没做到，和普通人一样快）

反之亦然——低防御角色应该：
- 感受强烈（全进） ✅（门控低，刺激通过）
- 但恢复快（放得下） ❌（衰减固定，没有比其他人更快）

**衰减和门控共享同一个"防御系数"，但当前代码把它们当成两个独立常数处理。**

#### 修复方向

门控和衰减应该由同一个心理防御参数驱动：

```python
def compute_defense(traits, relationship) -> float:
    """计算统一的心理防御值。"""
    base = traits[T_PRIDE] * 0.4 + (1 - traits[T_EMOTIONAL_OPENNESS]) * 0.3
    base *= (1.0 - relationship[R_TRUST] * 0.25)      # 关系松动防御
    return np.clip(base, 0.0, 1.0)

# 门控由 defense 决定
gates[G_SUPPRESSION] = defense * 0.8                    # 高防御→高压抑

# 衰减由 defense 反向决定
decay[I_STRESS] = base_decay - defense * 0.08            # 高防御→衰减慢
decay[I_IRRITATION] = base_decay - defense * 0.10
```

这样形成一条一致的人格外推：

| defense | 门控效果 | 衰减效果 | 心理画像 |
|---------|---------|---------|---------|
| 0.8 | 刺激砍 60% | stress decay 0.84（慢） | 压抑且记仇，积累型 |
| 0.5 | 刺激砍 30% | stress decay 0.88（中） | 适度防御，正常消解 |
| 0.2 | 刺激砍 12% | stress decay 0.92（快） | 开放且恢复快，体验型 |

随着关系深化，defense 同步下降，门控和衰减同步松动——**一个参数控制两扇门**。

---

### P1: 感知层缺少角色状态上下文

#### 问题

`perception_node` 调用感知模型时的输入只有固定 `PERCEPTION_SYSTEM_PROMPT` + 最近 4 条对话消息：

```python
context = extract_recent_context(state["messages"], cfg["context_window"])
call_perception_with_retry(context, cfg)
```

角色当前的心理状态（loneliness=0.9 vs 0.2）和关系状态（affection=0.8 vs 0.3）**完全不参与感知过程**。这意味着：
- 同样一句"我先睡了"，角色感到孤独时会解读为更大的 rejection/abandonment，但现在感知模型看不到这个差异
- 同样一句"你在干嘛"，好感度高时被解读为 affection/attention，好感度低时可能是客套——感知模型无法区分

#### 方案

将角色当前状态压缩为摘要，拼入感知上下文：

```python
def perception_node(state: State) -> dict:
    context = extract_recent_context(state["messages"], cfg["context_window"])
    # 将角色当前状态摘要加入感知上下文
    state_summary = format_state_perception_context(
        internal=state.get("internal_state"),
        relationship=state.get("relationship_state"),
        traits=state.get("traits"),
    )
    context.insert(0, SystemMessage(
        content=f"[角色当前心理语境]\n{state_summary}\n---"
    ))
    result = call_perception_with_retry(context, cfg)
```

`state_summary` 示例（3~5 句话，不暴露精确数值）：
```
你当前对用户的感情：好感度较高（约0.7），信任感中等偏低（约0.4）
你的内心状态：有一定孤独感（约0.6），情绪上轻微疲惫（约0.4）
核心人格：高敏感、高自尊、依恋焦虑偏高
```

---

### P1: 刺激构造层在当前实现中冗余 ✅ 已解决

#### 问题

~~当前刺激构造是纯线性层：`stimuli = signals @ SIGNAL_TO_STIMULUS`~~ 已移除。

#### 已实施方案

- 删除 `state_engine.py` 中的 `construct_stimuli()`、`_build_signal_to_stimulus()` 和 `SIGNAL_TO_STIMULUS` 矩阵
- 删除 `SocialSignals`（9 维）和 `InteractionImpact`（4 维）类型定义
- `perception_prompt.py` 重写：LLM 直接输出 7 维 `StimulusVector`
- `perception.py` 验证逻辑更新为检查 `user_stimuli` 字段
- `state.py` 中 `State.user_stimuli` 替代 `user_signals` + `user_interaction_impact`
- `nodes.py` 中 `perception_node` / `state_engine_node` 适配新接口
- `update_relationship_dynamics()` 移除 `impact` 参数（trust/closeness 直接影响已由 B_rel 矩阵覆盖）

---

### P1: 表面投影无时间惯性

#### 问题

`project_surface()` 每轮独立计算，不引用上一轮的表面状态：

```python
s[S_WARMTH] = 0.3 + R_AFFECTION × 0.4 - I_STRESS × 0.2
# 没有 S_{t-1} 项
```

结果：角色内部已经消气了（irritation=0.2），表面立刻变得温和（sharpness=0.3）。但现实中，人的神情是有惯性的——即使心里已经不生气了，表情可能还冷着。

#### 方案

引入表面状态惯性：

```python
def project_surface(
    internal, relationship, traits,
    previous_surface: Optional[np.ndarray] = None,
) -> np.ndarray:
    s = _compute_raw_surface(internal, relationship, traits)

    if previous_surface is not None:
        # 表面惯性：70% 保留上一轮表达，30% 反映当前状态
        s = 0.7 * previous_surface + 0.3 * s

    # 特质修饰（同上）
    ...
    return np.clip(s, 0.0, 1.0)
```

这样，即使 irritation 从 0.7 骤降到 0.2，sharpness 在下一轮也是：
```
sharpness_t = 0.7 × 0.65 + 0.3 × 0.25 = 0.53
```
而不是直接跳到 0.25——面部表情的"余怒"被保留。

注意：这要求 `surface_state` 在 State TypedDict 中被持久化并在下一轮传递。

---

### 现存的其他改进项

#### State Formatter 重设计划（P0）

保持下方原有方案。核心问题仍是：5 级阈值 `_desc()` 将连续状态引擎的输出重新离散化。目标是用连续加权语义投影替代。

#### 权重参数外部化（P1）

所有矩阵系数（W_sig2stim、M_trait、M_rel、A、B 等）硬编码在 `state_engine.py` 的 `_build_*()` 函数中。应外部化为 JSON/YAML 配置文件，支持热加载。

#### 上下文窗口扩展（P1）

同原有方案。扩展至 10~20 条消息 + Chroma 向量检索。

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

---

### P2: 三个状态缺少完整消息视野（RNN 瓶颈）

#### 问题

如果把状态引擎看作一个 RNN，每轮的更新是：

```
Perception: 只看 messages[-4:]              → signals[9]
State Engine: h_t = A·h_{t-1} + B·input     → internal[8], rel[6]
Surface: 每轮重算，无时序                      → surface[7]
```

三个状态的信息来源：

| 状态 | 存储 | 信息来源 | 能否看到全部消息 |
|------|------|---------|--------------|
| internal_state | 8 维持久化 | 本轮 signals + 上一轮自身 | ❌ 压缩在 8 维中 |
| relationship_state | 6 维持久化 | 本轮 signals + impact + 上一轮自身 | ❌ 压缩在 6 维中 |
| surface_state | 每轮重算不存储 | 本轮 internal + rel + traits | ❌ 零记忆 |

**两个层次的视野缺失：**

**① Perception 层的窗口截断：**
```python
extract_recent_context(messages, context_window=4)
```
感知模型只看最后 4 条消息。第 5 条前的关键语境信息完全丢失。即使 internal_state 里留下了痕迹，那也是经衰减后的压缩残量——无法替代原文语境。

**② 状态压缩的信息丢失：**
内部状态只有 8 维、关系状态只有 6 维。所有历史消息必须被压缩到这些维度中。维度有半衰期：
- irritation 半衰期 4 轮 → 冲突的愤怒 4 轮后剩一半
- longing 半衰期 23 轮 → 思念保留最久
- trust 半衰期 69 轮 → 信任伤害记住最久

**30 轮前用户说了一句很伤人的话 → irritation 已经完全归零了（4 轮半衰期），trust 虽然还有痕迹（69 轮半衰期）但只剩下"信任低"这个事实，原话内容、语气、上下文都丢了。**

这与 RNN 的长期依赖问题是完全一样的——hidden state 作为有损压缩渠道，长程信息不可避免地被稀释。

#### 方案

**A）增大 perception 上下文窗口（短期）**
- 4 → 12~20，配合 Chroma 向量检索提取相关历史
- 简单有效，但解决不了"文本状态必须压缩进低维空间"的问题

**B）增加状态维度（中期）**
- 加入 8~16 维的"记忆痕迹"状态，专门存储与用户有关的长期记忆特征
- 独立于情绪状态，不受 irritation 等快速维度的衰减影响
- 如：`memory_salience`（这段对话对角色有多重要）、`memory_valence`（正面/负面记忆倾向）

**C）接入 Chroma 向量检索（中期）**
- 已声明依赖但未连接。让 LLM node 可以主动检索相关历史消息
- 与状态引擎解耦——LLM 看到完整历史，状态引擎只看压缩状态
- 参考 RAG 架构

**D）增加长期记忆叙事状态（长期）**
- 在 State 中增加 `memory_traces: np.ndarray`（16~32 维），由检测到"重要事件"时刷新
- 关键事件（情感重量 > 0.7、信任冲击 > 0.3 等）在此留下持久痕迹
- 衰减极慢（decay > 0.999），由新事件覆写而非衰减消失

**推荐路径：** D（增加记忆痕迹向量）→ A（扩大窗口）→ C（Chroma 检索）。D 保持状态引擎的架构完整性，A 和 C 是辅助性增强。

---

### HiddenState 已移除（已执行）

隐藏状态层（hidden_state、突破事件、事件触发）在 commit `d0f726a` 中移除。

### SocialSignals / InteractionImpact / Stimulus Construction 已移除（已执行）

社交信号层（9 维）、互动影响层（4 维）和刺激构造层（线性 W_sig2stim 矩阵）
已移除。perception 节点现在直接输出 7 维 StimulusVector。
详情见上方「P1: 刺激构造层在当前实现中冗余 ✅ 已解决」。

---

## License

MIT
