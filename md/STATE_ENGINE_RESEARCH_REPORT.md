# Lunar 状态引擎独立研究诊断报告

> **范围**：基于对 `state_engine.py`、`state.py`、`nodes.py`、`perception.py`、`state_formatter.py`、`agent.py`、`default_state.py` 的代码阅读，结合计算心理学、情感计算、LLM Agent 记忆架构等领域的最新文献与开源实践，独立诊断当前状态引擎的不足，并提出研究方向。
>
> **时间**：2026-06-15
>
> **说明**：本报告刻意不重复 README.md / TODO_LUNAR_STATE_ENGINE.md 中已列出的问题清单，而是从代码实现本身、跨学科理论与工程生态出发，形成一份独立的诊断与索引。

---

## 一、执行摘要

Lunar 项目试图用一个**低维连续动力系统**替代 prompt 工程来驱动角色人格，这在 AI 角色引擎领域是一个极具潜力的方向。当前实现已经完成了概念验证级别的 pipeline：

```
感知(7 维刺激) → 门控 → LSTM 式内部状态 → 动态衰减 → 表面投影 → LLM 注入
```

然而，从数学稳定性、心理学覆盖面、记忆/时间/动机/用户建模、工程可维护性四个维度看，当前引擎仍处在**"可演示但不可长期运行"**的阶段。核心问题包括：

1. **数学稳定性缺陷**：三门控 LSTM 无约束求和、decay 系数允许 >1、关系矩阵存在正反馈环，长期运行必然饱和或坍缩。
2. **情绪模型过窄**：7 维刺激与 8 维内部状态无法覆盖基本情绪、评价维度、行动倾向等关键心理构造。
3. **缺乏时间与记忆**：以消息为 tick，无 wall-clock 时间；无 episodic/semantic 记忆，角色没有"过去"。
4. **无目标与动机系统**：角色只有刺激-反应，没有目标层级、需求、意图、行动倾向。
5. **无用户模型**：只有"对用户的关系感知"，没有 Theory of Mind、用户人格、用户目标。
6. **工程债务**：全硬编码权重、无测试、序列化靠 hack、state formatter 离散化连续状态。

本报告后半部分系统梳理了相关学术理论（OCC/CPM、Whole Trait Theory、Latent State-Trait、SDT、依恋 IWM、McAdams 叙事身份）与开源实践（FAtiMA、Mem0、LangGraph Memory Store、Nomi.ai、Convai LTM 等），并给出了一条从"修复数学基础"到"构建完整心理主体"的演进路线。

---

## 二、当前引擎架构速览

| 层级 | 维度 | 当前实现 | 问题缩影 |
|------|------|---------|---------|
| **Stimulus** | 7 | abandonment/validation/closeness/conflict/dependency/teasing/emotional_weight | 缺少基本情绪、评价维度、期待/失望/内疚等 |
| **Gate** | 4 (3 用) | suppression/vulnerability/attachment/leakage(未用) | 无统一 defense 参数，三门独立 |
| **Internal State** | 8 | energy/stress/loneliness/insecurity/irritation/longing/social_battery/mental_fatigue | 无 valence/arousal、无情绪类别、无认知状态 |
| **Relationship State** | 6 | affection/trust/familiarity/dependency/emotional_safety/romantic_tension | 无用户侧模型、无双向 IWM |
| **Surface State** | 7 | expressiveness/warmth/sharpness/softness/enthusiasm/restraint/vulnerability | 不持久化、无时间惯性 |
| **Traits** | 10 | sensitivity/pride/openness/stability/optimism/anxiety/anger/jealousy/attachment_anxiety/attachment_avoidance | 静态、无演化、缺少 Big Five 维度 |

核心动力学方程（简化）：

```
h_t = f⊙h_{t-1} + i⊙(A·h_{t-1} + B·e_t) + g⊙bias
h_t' = baseline + (h_t - baseline) ⊙ decay
rel_t = A_rel·rel_{t-1} + B_rel·e_t
surface = project(internal, rel, traits, outer_stimuli)
```

其中 `f, i, g` 为三个独立 sigmoid 门，`decay` 经动态调制后可 >1。

---

## 三、独立诊断：代码层面的结构性不足

> 本节基于对代码的直接阅读与动力系统分析，不依赖 README 中的既有问题清单。

### 3.1 致命级：数学稳定性问题

#### 3.1.1 三门控 LSTM 无守恒约束

代码中 `update_internal_dynamics` 使用：

```python
new_state = (
    f_gate * current          # 遗忘
    + i_gate * raw_dynamics   # 接受
    + g_gate * bias           # 自生
)
```

三个门分别由独立 sigmoid 生成，取值均在 (0,1)，因此三门之和 ∈ (0,3)。标准 LSTM 的关键设计是 **cell state 与 output 分离**，并通过耦合门控（如 `i = 1 - f`）保证状态有界。当前实现没有 cell state，也没有门控守恒约束，导致：

- 若 `f+i+g < 1`：状态快速坍缩；
- 若 `f+i+g > 1` 且 bias/raw_dynamics 为正：状态饱和到 1；
- 同一角色在不同输入下可能表现出完全相反的动力学行为，不可解释。

**学术依据**：Hochreiter & Schmidhuber (1997) 的 LSTM 通过遗忘门与输入门耦合保证状态稳定性；Gers et al. (2000) 的 peephole/coupled forget-input gates 进一步强化这一点。

#### 3.1.2 "动态衰减"实为发散算子

```python
state[t] = baseline + (state[t-1] - baseline) * decay
```

`compute_dynamic_decay` 将内部 decay clamp 到 `[0.70, 1.05]`，关系 decay clamp 到 `[0.95, 1.005]`。当 `decay > 1` 时，系统不是向基线回归，而是**指数放大偏离基线的量**。这与心理学中的 homeostatic recovery 完全相反。

后果：只要某轮触发"被抛弃 + 高依恋焦虑"等共振条件，decay > 1 就会使对应状态永久饱和，角色从"依恋焦虑"变成"永远崩溃"。

**学术依据**：动力系统稳定要求特征根 / 衰减系数位于单位圆内（Strogatz, *Nonlinear Dynamics and Chaos*）。

#### 3.1.3 关系矩阵正反馈环

`_build_rel_state_coupling` 中，`AFFECTION → TRUST → SAFETY → AFFECTION` 形成正反馈环，对角线仅 0.90，总耦合强度可超过 1。即使没有外部冲突刺激，关系状态也会自发增长并饱和到 1.0。

**学术依据**：LTI 系统稳定的充要条件是所有特征值模 < 1，当前矩阵未做谱半径校验。

---

### 3.2 严重级：心理学模型覆盖不足

#### 3.2.1 情绪状态维度严重不足

当前 InternalState 仅有 8 维，且高度偏向"关系性"负性状态（stress/loneliness/insecurity/irritation/longing）。缺少：

- **基本情绪维度**：joy/sadness/fear/anger/disgust/surprise（Ekman / Plutchik）
- **核心情感维度**：valence / arousal / dominance（Russell circumplex / PAD）
- **评价维度**：novelty, relevance, goal_congruence, coping_potential, norm_compatibility（Scherer CPM / Lazarus）
- **社会情绪**：guilt, shame, pride, gratitude, disappointment, anticipation
- **认知状态**：uncertainty, rumination, hope, regret

这导致角色心理画像扁平化，无法区分"悲伤但平静"、"愤怒但压抑"、"嫉妒但装作无所谓"等 nuanced 状态。

#### 3.2.2 特质-状态关系建模错误

当前 Traits 是静态向量，且 `apply_decay` 始终向 `DEFAULT_INTERNAL` 回归，而不是向由 Traits 决定的个体化 setpoint 回归。高焦虑角色的 stress 理论上应该有更高基线，但系统会把它拉回 0.2。

这与现代人格动力学的核心发现冲突：

- **Whole Trait Theory**（Fleeson, 2001; Fleeson & Jayawickreme, 2015）：特质不是固定点，而是状态的密度分布；状态表达由情境与动机共同驱动。
- **Latent State-Trait Theory**（Steyer et al., 1999, 2015）：观察到的行为 = 稳定特质 + 情境特定状态 + 测量误差；ICC 通常仅 0.2~0.4，说明情境解释力远大于稳定特质。
- **Cybernetic Big Five Theory**（DeYoung, 2015）：人格应通过 setpoint、时间常数、刺激权重共同体现。

#### 3.2.3 缺少评价理论（Appraisal Theory）层

当前感知层直接输出 7 维刺激，缺少一个中间的评价层：事件对用户/角色的目标是否一致？是否可控？是否可预期？由谁造成？

这导致：
- 同样一句话"我先睡了"，无法区分"用户真的累了"与"用户故意冷落我"；
- 无法产生失望、庆幸、内疚等依赖"期望-结果"对比的情绪。

**学术依据**：OCC 模型（Ortony, Clore, Collins, 1988）、Scherer 的 Component Process Model（2001, 2009）、Lazarus 的评价理论均将认知评价作为情绪产生的核心条件。

#### 3.2.4 无目标/动机/行动倾向系统

引擎只有 `刺激 → 状态变化 → 表面表达`，没有：

- **目标层级**：维持亲密、寻求安慰、回避冲突、确认被爱
- **需求系统**：SDT 的 autonomy / competence / relatedness
- **行动倾向（action tendency）**：Frijda (1986) 认为情绪的功能性正在于产生行动倾向
- **内部独白/反刍**：两次对话之间角色完全静止

角色因此只是被动反应，不会主动追求、策划或反思。

#### 3.2.5 无用户模型与 Theory of Mind

`relationship_state` 是角色对关系的主观评估，但缺少一个独立的用户模型：

- 用户的人格特质、依恋风格
- 用户当前的情绪状态、目标、压力源
- 用户对角色的信念与期望
- 互惠性预期（reciprocity）

这导致角色无法区分"用户真的生气"与"用户在开玩笑"，无法形成信任推理，亲密关系的双向建模缺失。

**学术依据**：Bowlby (1969/1982) 的内部工作模型包含对自我和他人的双向表征；Theory of Mind（Premack & Woodruff, 1978）是社交互动的核心。

---

### 3.3 中等级：时间与记忆缺失

#### 3.3.1 没有真实时间模型

状态更新以"消息"为 tick，完全忽略 wall-clock 时间。`ENERGY`、`SOCIAL_BATTERY`、`MENTAL_FATIGUE` 的恢复/衰减与用户实际上一条消息间隔无关。

后果：
- 用户狂发 20 条与隔 8 小时发 1 条产生相同疲劳/恢复；
- 无法模拟睡眠、休息、想念随时间累积；
- "等待"是角色核心设定，但引擎没有任何等待时间维度。

**学术依据**：时间心理学（Zimbardo & Boyd, 1999）、情绪动力学（Kuppens et al., 2010）均强调时间尺度对情绪恢复的决定性作用。

#### 3.3.2 没有真正的记忆系统

系统仅保存 `messages` 与当前状态向量：

- 无 episodic memory（具体事件、承诺、共同经历）
- 无 semantic memory（用户偏好、事实、角色知识）
- 无记忆巩固、遗忘曲线、记忆检索
- `context_window = 4` 的感知上下文只能看到最近 2 轮对话

`relationship_state` 成为唯一"长期记忆"，但它维度极低且数学不稳定。

**学术依据**：Conway & Pleydell-Pearce (2000) 的自我记忆系统指出，长时记忆是人格一致性的根基；Tulving (1972) 区分情景记忆与语义记忆。

---

### 3.4 中等级：工程实现问题

#### 3.4.1 全硬编码权重，无可学习性

所有矩阵 `A, B, A_rel, B_rel`、门控系数、衰减偏置都是手写魔法数字，没有配置文件、没有学习机制、没有敏感性分析。调参困难，且无法针对不同角色复用。

#### 3.4.2 硬编码阈值造成不连续行为

`compute_dynamic_decay` 中大量使用：

```python
if stimuli[ST_ABANDONMENT] > 0.3 and traits[T_ATTACHMENT_ANXIETY] > 0.55:
    ...
```

0.299 与 0.301 的刺激强度导致截然不同的系统行为，与心理学中"连续变化"的直觉冲突。

#### 3.4.3 表面状态不持久化，无时间惯性

`SurfaceState` 被设计为"动态投影，不存储"。无法分析表达风格的长期变化，也无法将上一轮 surface 作为本轮输入，导致表情变化没有"余怒"或"余温"。

#### 3.4.4 state formatter 重新离散化连续状态

`_desc()` 的 5 级阈值把连续状态输出重新拉到 5 个桶，破坏了 State Engine 连续动力学的核心优势。

#### 3.4.5 感知层脆弱

- 依赖本地 7B 模型（qwen2.5:7b）做细粒度心理刺激提取；
- JSON 失败则整轮结束，无 fallback、无置信度；
- 上下文窗口仅 4 条；
- 验证只检查字段存在，不检查值域；
- 提示词硬编码角色背景，换角色需重写。

#### 3.4.6 无测试、无评估闭环

项目中没有任何 `*_test.py` 或 `test_*.py`，也没有对 LLM 输出是否真实符合状态描述的校验机制。角色一致性仅靠 prompt engineering，不可靠。

---

## 四、学术理论映射与研究方向

### 4.1 情感计算：从离散情绪到评价动力学

| 理论/模型 | 核心思想 | 对 Lunar 的启示 |
|----------|---------|----------------|
| **Ekman 基本情绪** | 6 种基本情绪：快乐、悲伤、恐惧、愤怒、惊讶、厌恶 | 刺激/内部状态应至少覆盖基本情绪类别 |
| **Plutchik 情感轮盘** | 8 种基本情绪 + 强度 + 组合 | 可扩展情绪空间，支持情绪混合 |
| **Russell Circumplex** | valence × arousal 二维连续空间 | 内部状态可加入核心情感维度 |
| **OCC 模型** | 情绪 = 对事件/行为/对象的评价结果 | 在感知与刺激之间加入评价层 |
| **Scherer CPM** | 情绪是多成分同步化过程，评价检查驱动行动倾向、表情、生理 | 构建 appraisal → action tendency → expression 的层级 |
| **Frijda 行动倾向** | 情绪的功能是产生行动准备 | 引入目标/行动系统，而不只是语言表达 |

**关键文献**：
- Ortony, A., Clore, G. L., & Collins, A. (1988). *The Cognitive Structure of Emotions*.
- Scherer, K. R. (2001, 2009). Component Process Model of Emotion.
- Russell, J. A. (1980). A circumplex model of affect.
- Frijda, N. H. (1986). *The Emotions*.

### 4.2 人格动力学：从静态特质到状态分布

| 理论 | 核心思想 | 对 Lunar 的启示 |
|------|---------|----------------|
| **Whole Trait Theory** | 特质是状态的密度分布，情境激活trait-relevant行为 | Traits 不应是固定点，而应决定状态分布参数（均值/方差/setpoint） |
| **Latent State-Trait Theory** | 观察行为 = 稳定特质 + 情境状态 + 误差 | 将 InternalState 显式分解为 trait-driven baseline + occasion-specific deviation |
| **Trait Activation Theory** | 情境线索激活特质相关目标与行为 | 感知层应输出"情境对哪些特质的激活强度" |
| **CAPS (Mischel & Shoda)** | if...then 情境-行为签名 | 可学习角色在特定情境下的稳定反应模式 |

**关键文献**：
- Fleeson, W. (2001). Toward a structure- and process-integrated view of personality.
- Fleeson, W., & Jayawickreme, E. (2015). Whole Trait Theory.
- Steyer, R., et al. (1999, 2015). Latent State-Trait Theory.
- Mischel, W., & Shoda, Y. (1995). A cognitive-affective system theory of personality.

### 4.3 依恋理论：内部工作模型的双向建模

Bowlby 的内部工作模型（IWM）包含两个核心表征：
- **自我模型**：我是否值得被爱？
- **他人模型**：他人是否可靠、可依赖？

Bartholomew & Horowitz (1991) 将成人依恋分为四类：安全型、痴迷型、恐惧型、疏离型。

对 Lunar 的启示：
- 当前 `attachment_anxiety / attachment_avoidance` 两维方向正确，但缺少 IWM 的双向结构；
- 关系状态应包含"对用户的 IWM"，并随互动经验更新；
- 防御性贬低（devaluation）、安全基地效应（secure base）、分离焦虑等依恋机制应显式建模。

### 4.4 叙事身份：McAdams 的三层人格模型

McAdams (2001) 将人格分为三层：
1. **Level 1：特质（Traits）** —— Big Five 等
2. **Level 2：特征性适应（Characteristic Adaptations）** —— 目标、价值观、防御机制
3. **Level 3：叙事身份（Narrative Identity）** —— 整合过去、现在、未来的生命故事

对 Lunar 的启示：
- 当前引擎只做到了 Level 1（Traits）+ 部分 Level 2（状态）；
- 长期记忆应以"叙事摘要"形式存在，让角色能回答"我们第一次聊天时发生了什么"；
- 可借鉴 McAdams Life Story Interview 的结构化提示，由 LLM 定期生成/更新角色自传摘要。

### 4.5 动机理论：Self-Determination Theory

Deci & Ryan 的 SDT 指出，人类有三种基本心理需求：
- **Autonomy（自主）**：感到行为出于自愿
- **Competence（胜任）**：感到自己能有效影响环境
- **Relatedness（联结）**：感到与他人有 meaningful connection

对 Lunar 的启示：
- 角色应有"需求状态"，驱动主动行为；
- 用户输入可被评价为满足/挫败哪种需求；
- 长期关系质量可通过三种需求的满足程度来预测。

### 4.6 情绪动力学：情感惯性、可变性、粒度

Kuppens & Verduyn (2017) 的情绪动力学研究识别出多个关键参数：
- **情绪惯性（emotional inertia）**：情绪的自回归强度
- **可变性（variability）**：情绪波动的幅度
- **不稳定性（instability）**：相邻时刻的情绪变化
- **情绪粒度（emotion granularity）**：区分不同情绪的能力
- **交叉滞后（cross-lagged effects）**：情绪之间的相互影响

对 Lunar 的启示：
-  decay/惯性参数应由人格特质决定；
- 应测量并可视化这些动力学指标，用于调参与诊断。

---

## 五、开源生态与工业实践

### 5.1 学术/研究型情感 Agent 架构

#### FAtiMA（FearNot! Affective Mind Architecture）

- **核心**：基于 BDI + OCC 评价理论
- **两层架构**：
  - Reactive Layer：快速情绪反应与行动倾向
  - Deliberative Layer：目标导向行为与规划
- **组件**：
  - Appraisal Derivation（评价推导）
  - Affect Derivation（情绪生成）
  - Autobiographical Memory（自传体记忆）
  - Knowledge Base（信念库）
  - Motivational Component（动机/内驱力）
  - Theory of Mind Component（他人心智建模）
- **对 Lunar 的启示**：完整的情感 Agent 需要评价层、记忆层、动机层、ToM 层，而不只是状态向量。

#### EMA / ALMA / FLAME

- EMA（Gratch & Marsella）：基于决策理论计划与 Lazarus 评价理论，支持情绪聚焦应对（emotion-focused coping）。
- ALMA：基于 OCC 的轻量情绪引擎，输出 PAD 空间。
- FLAME：基于 MDP 与模糊规则。

### 5.2 工业界 AI Companion / 角色引擎

| 产品/项目 | 记忆方案 | 可借鉴点 |
|----------|---------|---------|
| **Replika** | LSTM + 神经网络的混合架构，情感记忆 | 早期证明 LSTM 可用于陪伴对话 |
| **Nomi.ai** | 语义记忆，长期情感记忆 | 最佳长期记忆角色扮演产品之一 |
| **Character.AI** | 大规模角色微调 + 上下文 | 角色一致性工程化 |
| **Convai LTM** | RAG + 自定义排序 + 记忆树 | 情感影响、隐私隔离、记忆版本控制 |
| **Anione** | DeepSeek + 持久化层 | 事件、情感转变、承诺索引 |
| **SillyTavern** | JSON 角色配置 + SQLite 持久化 | 开源生态、插件化 |

### 5.3 LLM Agent 记忆基础设施

| 项目 | 类型 | 特点 |
|------|------|------|
| **Mem0** | 语义记忆 | 51k+ stars，自动去重，向量存储用户事实与偏好 |
| **Zep** | 时序+语义记忆 | 时间知识图谱 Graphiti，PostgreSQL |
| **Letta / MemGPT** | 全类型记忆 | LLM 管理内存分页，支持六种记忆类型 |
| **LangMem** | 语义+情景记忆 | LangChain 官方，结构化 memory manager |
| **Cognee** | 认知图记忆 | 自托管知识图谱 |
| **LangGraph Store** | 长期记忆 | 支持语义/情景/程序记忆，namespacing |
| **GraphRAG / LightRAG** | 图检索增强 | 关系推理，减少碎片化检索 |

**对 Lunar 的启示**：记忆应至少分为：
- **工作记忆 / 短期记忆**：当前对话上下文
- **语义记忆**：用户事实、偏好、角色知识
- **情景记忆**：具体事件、承诺、共同经历
- **程序记忆**：角色行为模式、系统提示演化

### 5.4 评估基准

| 基准 | 用途 |
|------|------|
| **CharacterEval** | 角色一致性 |
| **PERSIST** | 人格稳定性 |
| **RPEval** | 情绪理解、决策、角色一致性 |
| **CharacterBox** | 角色忠诚度行为轨迹 |
| **LoCoMo** | 长上下文对话记忆 |
| **LongMemEval** | 长期记忆评估 |
| **SpeechDRAMA** | 多轮语音角色扮演 |

---

## 六、改进路线图建议

基于以上诊断，建议按以下阶段推进：

### Phase 1：修复数学基础（1-2 周）

1. **严格限制 decay < 1**：内部状态 decay ∈ [0.80, 0.995]，关系 decay ∈ [0.95, 0.9995]，杜绝发散。
2. **引入 trait-dependent baseline**：用 Traits 计算每个内部状态的个体化基线，替代统一 `DEFAULT_INTERNAL`。
3. **修复 LSTM 门控**：引入 cell state 或强制 `f + i ≈ 1`，g_gate 仅作为 bias 权重而非独立加法项。
4. **校验 A_rel 谱半径**：确保关系矩阵所有特征值模 < 1，或引入归一化。
5. **增加不变量测试**：状态有界、长时间模拟不发散、门控在 [0,1]、decay < 1。

### Phase 2：扩展情绪与评价模型（2-3 周）

1. **增加基本情绪/核心情感维度**：在 Stimulus 或 InternalState 中加入 joy/sadness/fear/anger/surprise/disgust 或 valence/arousal。
2. **引入 Appraisal 层**：在感知输出前加入评价变量：goal_congruence, certainty, agency, coping_potential, expectation_deviation。
3. **扩展刺激维度**：加入 anticipation, guilt, disappointment, gratitude, curiosity 等。
4. **用 sigmoid/softmax 替代硬编码阈值**：确保连续过渡。

### Phase 3：构建时间与记忆系统（3-4 周）

1. **引入 wall-clock 时间**：每个状态向量带 `last_update_time`，用真实时间差 Δt 驱动衰减与恢复。
2. ** episodic memory**：存储重要对话片段、承诺、冲突、和解事件，带情感标签与时间戳。
3. **semantic memory**：存储用户事实、偏好、角色知识，使用 LangGraph Store / Chroma。
4. **记忆巩固**：定期由 LLM 将 episodic 片段压缩为语义摘要与叙事摘要。
5. **感知层扩展上下文**：4 → 12~20，并注入相关记忆检索结果。

### Phase 4：增加目标、动机与行动系统（3-4 周）

1. **目标层级**：短期目标（"确认ta是否爱我"）+ 长期目标（"建立深层信任"）。
2. **需求系统**：SDT 的 autonomy / competence / relatedness。
3. **行动倾向**：情绪产生 action tendency（靠近、逃避、攻击、求助、隐藏等）。
4. **内部独白/反刍**：在非对话时刻运行 reflection loop，更新 rumination、hope、regret 等。
5. **行为系统**：不仅生成语言，还能表达动作（"轻轻握住你的手"）。

### Phase 5：用户模型与 Theory of Mind（2-3 周）

1. **用户画像**：推断用户人格、依恋风格、偏好、压力源。
2. **用户情绪推断**：基于对话内容推断用户当前情绪。
3. **互惠性建模**：角色对用户行为的预期与归因。

### Phase 6：工程化与评估（持续）

1. **权重外部化**：YAML/JSON 配置文件，支持热加载与多角色。
2. **完整测试套件**：单元测试 + 长期模拟测试 + 角色一致性评估。
3. **State Formatter 连续化**：用权重式连续语义投影替代 5 级阈值。
4. **LLM 输出校验**：用 LLM-as-a-judge 评估回复是否符合状态描述。
5. **服务化**：完善 FastAPI 入口，支持多用户隔离与会话管理。

---

## 七、可直接参考的开源/论文资源

### 论文

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
2. Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to forget: Continual prediction with LSTM. *Neural Computation*, 12(10), 2451–2471.
3. Ortony, A., Clore, G. L., & Collins, A. (1988). *The Cognitive Structure of Emotions*.
4. Scherer, K. R. (2001). Appraisal considered as a process of multilevel sequential checking.
5. Scherer, K. R. (2009). The dynamic architecture of emotion: Evidence for the component process model. *Cognition & Emotion*, 23(7), 1307–1351.
6. Fleeson, W. (2001). Toward a structure- and process-integrated view of personality: Traits as density distributions of states. *Journal of Personality and Social Psychology*, 80(6), 1011–1027.
7. Fleeson, W., & Jayawickreme, E. (2015). Whole Trait Theory. *Journal of Research in Personality*, 56, 82–92.
8. Steyer, R., et al. (1999, 2015). Latent State-Trait Theory.
9. Bowlby, J. (1969/1982). *Attachment and Loss*.
10. Mikulincer, M., & Shaver, P. R. (2007). *Attachment in Adulthood*.
11. McAdams, D. P. (2001). The psychology of life stories. *Review of General Psychology*, 5(2), 100–122.
12. Deci, E. L., & Ryan, R. M. (1985, 2000). Self-Determination Theory.
13. Kuppens, P., & Verduyn, P. (2017). Emotion dynamics. *Current Opinion in Psychology*, 17, 22–26.
14. DeYoung, C. G. (2015). Cybernetic Big Five Theory. *Journal of Research in Personality*, 56, 33–58.
15. Marsella, S., Gratch, J., & Petta, P. (2010). Computational models of emotion. *A Blueprint for Affective Computing*.
16. Dias, J., Mascarenhas, S., & Paiva, A. (2014). FAtiMA Modular: Towards an Agent Architecture with a Generic Appraisal Framework.
17. Wang et al. (2024). *On the logic of agent's emotions* (OCC formalization).
18. *Beyond Fixed Psychological Personas: State Beats Trait, but Language Models are State-Blind* (arXiv:2601.15395).
19. *Driving Generative Agents With Their Personality* (arXiv:2402.14879).
20. *PersonaFuse: A Personality Activation-Driven Framework* (arXiv:2509.07370).

### 开源项目

1. **FAtiMA Toolkit** — `https://fatima-toolkit.eu/`
2. **Mem0** — `https://github.com/mem0ai/mem0`
3. **LangGraph Memory / LangMem** — `https://langchain-ai.github.io/langgraph/concepts/memory/`
4. **Letta (ex-MemGPT)** — `https://github.com/letta-ai/letta`
5. **Zep** — `https://github.com/getzep/zep`
6. **Cognee** — `https://github.com/topoteretes/cognee`
7. **SillyTavern** — `https://github.com/SillyTavern/SillyTavern`
8. **Awesome-GrokAni-VirtualMate** — 虚拟伴侣工具全景索引
9. **LOCOMO Benchmark** — 长上下文记忆评估
10. **LongMemEval** — 微软长期记忆评估套件

### 标准

1. **W3C EmotionML 1.0** — 情绪标注标准，支持 category/dimension/appraisal/action-tendency 四种描述方式。
2. **ISO 24617-2** — 对话行为标注，可与 EmotionML 结合使用。

---

## 八、结论

Lunar 状态引擎的**架构直觉是正确的**：用低维连续向量表示人格状态、用门控模拟防御机制、用 LTI/LSTM 混合动力学模拟情绪演化，这些都是情感计算领域的经典思路。但当前实现存在三类根本性问题：

1. **数学上不稳定**：三门控无约束、decay 可发散、关系矩阵正反馈，导致长期运行必然饱和或坍缩；
2. **心理学上不完整**：缺少评价层、基本情绪、目标/动机、记忆、时间、用户模型等核心构造；
3. **工程上不可扩展**：硬编码权重、无测试、无配置、无评估闭环。

修复路径应遵循"先底层、后上层"：

```
稳定动力系统 → 扩展情绪/评价模型 → 引入时间与记忆 → 增加动机与行动 → 构建用户模型 → 工程化与评估
```

只有在数学稳定、状态空间足够表达复杂心理、角色拥有"过去"与"想要"之后，Lunar 才能真正从"会说话的数值机"演化为"有连续心理生命的 AI 角色"。
