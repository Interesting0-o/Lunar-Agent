# Lunar 状态引擎架构

> 2026-06-19 | defense-based 残差动力学 | 3 阶段管线 + 时间衰减
>
> 架构变更历史：
> - **06-15**：旧 LSTM 三门控 + 动态衰减（已废弃）
> - **06-18**：γ·(sp-h) per-turn 稳态恢复移除 → γ 移至 `_decay.py`；A 矩阵替换为 10 条显式耦合规则；sigmoid 缩放 + β 调制修复
> - **06-19**：`DecayConfig.negative_decay_boost=1.8` 非对称衰减（Fading Affect Bias）

---

## 概览

Lunar 的状态引擎由 3 个主要阶段组成：

1. 防御剖面（Defense Profiles）
2. 残差动力学（Dynamics）
3. 表面投影（Surface Projection）

此外，系统有一个独立的时间衰减模块，用于对话间隔中的稳态恢复。

实现文件：
- `state_engine/_pipeline.py`
- `state_engine/_defenses.py`
- `state_engine/_dynamics.py`
- `state_engine/_surface.py`
- `state_engine/_decay.py`
- `state.py`

> **注意**：以下描述均为**当前**架构。旧 LSTM 三门控系统已被完全替换，参考 RESERCH 文档时请注意区分。

---

## 状态向量

### StimulusVector（7 维）
- `ST_ABANDONMENT`：被抛弃恐惧
- `ST_VALIDATION`：被认可/重视感
- `ST_CLOSENESS`：亲密靠近感
- `ST_CONFLICT`：冲突/对抗张力
- `ST_DEPENDENCY`：被依赖/被需要感
- `ST_TEASING`：被逗弄/被调侃
- `ST_EMOTIONAL_WEIGHT`：情绪冲击强度

### InternalState（8 维）
- `I_ENERGY`
- `I_STRESS`
- `I_LONELINESS`
- `I_INSECURITY`
- `I_IRRITATION`
- `I_LONGING`
- `I_SOCIAL_BATTERY`
- `I_MENTAL_FATIGUE`

### RelationshipState（6 维）
- `R_AFFECTION`
- `R_TRUST`
- `R_FAMILIARITY`
- `R_DEPENDENCY`
- `R_EMOTIONAL_SAFETY`
- `R_ROMANTIC_TENSION`

### SurfaceState（7 维）
- `S_EXPRESSIVENESS`
- `S_WARMTH`
- `S_SHARPNESS`
- `S_SOFTNESS`
- `S_ENTHUSIASM`
- `S_RESTRAINT`
- `S_VULNERABILITY`

### Traits（10 维）
- `T_SENSITIVITY`
- `T_PRIDE`
- `T_EMOTIONAL_OPENNESS`
- `T_EMOTIONAL_STABILITY`
- `T_OPTIMISM`
- `T_ANXIETY_PRONENESS`
- `T_ANGER_REACTIVITY`
- `T_JEALOUSY_SENSITIVITY`
- `T_ATTACHMENT_ANXIETY`
- `T_ATTACHMENT_AVOIDANCE`

---

## 管线流程

主入口：`state_engine/_pipeline.py` 的 `update_all()`。

输入：
- `current_internal`：当前内部状态
- `current_relationship`：当前关系状态
- `traits`：人格特质
- `stimuli`：心理刺激

输出：
- `internal_state`
- `relationship_state`
- `surface_state`

流程：

1. 计算防御剖面：`compute_defense_profiles(traits, current_relationship, current_internal)`
2. 应用防御：`apply_defenses(stimuli, profiles)` → 得到 `inner_stimuli` 和 `outer_stimuli`
3. 更新内部状态：`update_internal_state(...)`
4. 更新关系状态：`update_relationship_state(...)`
5. 投影表面状态：`project_surface(new_internal, new_relationship, traits, outer_stimuli)`

如果 `current_internal` 为空，则调用 `initialize_all(traits)`，用人格特质生成初始内部、关系、表面状态。

---

## 防御剖面

文件：`state_engine/_defenses.py`

### 目标

将原始刺激分成两类：
- `inner_stimuli`：进入内部动力学的"里"刺激
- `outer_stimuli`：进入表面表达的"表"刺激

这两个输出分别由两个防御剖面控制：
- `profiles[0]`：去激活（Deactivation）
- `profiles[1]`：过度激活（Hyperactivation）

### 计算公式

`profiles` 为 $(2, 7)$ 的向量，值域在 $[0, 1]$。

#### 去激活（Deactivation）

基于人格特质 $T_{PRIDE}$、$T_{ATTACHMENT\\_AVOIDANCE}$、$T_{EMOTIONAL\\_OPENNESS}$、$T_{EMOTIONAL\\_STABILITY}$
由当前关系状态 $R_{TRUST}$、$R_{EMOTIONAL\\_SAFETY}$ 调制
由当前内部状态 $I_{STRESS}$、$I_{INSECURITY}$ 推动

最终公式：

$$
profiles[0] = \sigma\bigl(5.0 \cdot (deact - 0.35)\bigr)
$$

其中 $deact$ 是基线 + 特质调制 + 关系调制 + 内部急性推动。

#### 过度激活（Hyperactivation）

基于人格特质 $T_{ATTACHMENT\_ANXIETY}$、$T_{JEALOUSY\_SENSITIVITY}$、$T_{SENSITIVITY}$
受 $T_{ATTACHMENT\_AVOIDANCE}$ 抑制
由当前关系状态 $R_{AFFECTION}$、$R_{ROMANTIC\_TENSION}$ 放大
由当前内部状态 $I_{INSECURITY}$、$I_{LONGING}$ 推动

最终公式：

$$
profiles[1] = \sigma\bigl(5.0 \cdot (hyper - 0.38)\bigr)
$$

#### 防御应用

$$
inner = stimuli \odot \bigl(1 + 0.50 \cdot hyper\bigr)
$$

$$
outer = inner \odot \bigl(1 - 0.70 \cdot deact\bigr)
$$

- $hyper$ 放大内心刺激
- $deact$ 削弱外表表达
- $inner$ 与 $outer$ 均使用 $\text{soft\_clamp}(\cdot, 0, 1)$ 限幅

---

## 残差动力学

文件：`state_engine/_dynamics.py`

### 核心思想

每轮对话中，状态更新只由刺激和耦合驱动。稳态恢复不在本阶段发生，改由时间衰减处理。

核心公式（γ 项已于 06-18 移除）：

$$
h_t = h_{t-1} + dt \cdot \bigl(\alpha \cdot \Delta_{coupling} + \beta \cdot \Delta_{stimulus}\bigr)
$$

### 内部状态更新

$update\_internal\_state(current, inner\_stimuli, traits, relationship, profiles)$

- $\alpha$：跨维度耦合速率
  - 受 $T_{EMOTIONAL\_OPENNESS}$、$T_{EMOTIONAL\_STABILITY}$、$R_{TRUST}$ 调制
- $\beta$：刺激接受速率
  - $0.05 + \mathrm{mean}(hyper) \cdot 0.35 - \mathrm{mean}(deact) \cdot 0.15$

β 调制修复后（06-18），β 有效变动从 0.042 提升到 0.289（×6.9），防御剖面现在真正控制响应速率。

耦合项：

$$
\Delta_{coupling} = coupling - SELF\_DECAY \odot current
$$

其中 $SELF\_DECAY$ 为每个内部维度常数 $0.15$。

示例耦合关系：
- energy 高 → stress 下降
- stress 高 → loneliness、irritation、mental\_fatigue 升高
- loneliness 高 → insecurity、longing 升高
- social\_battery 低 → irritation、mental\_fatigue 升高

刺激项：

$$
\Delta_{stimulus} = inner\_stimuli^{\mathsf{T}} \cdot INPUT\_INFLUENCE\_B
$$

最终更新：

$$
h_t = current + dt \cdot \bigl(\alpha \cdot \Delta_{coupling} + \beta \cdot \Delta_{stimulus}\bigr)
$$

结果再经 $\text{soft\_clamp}(\cdot, -1, 1)$ 限幅。

### 关系状态更新

$update\_relationship\_state(current, inner\_stimuli, traits)$

- $\alpha_{rel}$ 更小，表示关系变化比内部变化慢
- $\beta_{rel}$ 更小，表示关系对刺激缓冲更强

同样按照残差形式更新：

$$
r_t = r_{t-1} + dt \cdot \bigl(\alpha_{rel} \cdot \Delta_{coupling}^{rel} + \beta_{rel} \cdot \Delta_{stimulus}^{rel}\bigr)
$$

关系耦合项示例：
- $affection \to trust, familiarity$
- $trust \to dependency, emotional\_safety$
- $emotional\_safety \to affection$
- $dependency \to romantic\_tension$

刺激项：

$$
\Delta_{stimulus}^{rel} = inner\_stimuli^{\mathsf{T}} \cdot REL\_INPUT\_INFLUENCE\_B
$$

最终结果同样使用 $\text{soft\_clamp}(\cdot, -1, 1)$ 限幅。

### 稳态基线计算

用于初始化和时间衰减：

- $compute\_setpoint(traits)$ → 内部状态基线
- $compute\_rel\_setpoint(traits)$ → 关系状态基线

这些函数基于人格特质调整默认基线值。

---

## 表面投影

文件：`state_engine/_surface.py`

目标：将内部状态、关系状态与被抑制后的外表刺激投影为可感知的表面表达。

公式示例：

$$
\begin{aligned}
s[S\_EXPRESSIVENESS] &= -0.3 + 0.4 \cdot internal[I\_ENERGY] - 0.3 \cdot internal[I\_MENTAL\_FATIGUE] \\
s[S\_WARMTH] &= -0.2 + 0.4 \cdot relationship[R\_AFFECTION] - 0.2 \cdot internal[I\_STRESS] \\
s[S\_SHARPNESS] &= -0.1 + 0.5 \cdot internal[I\_IRRITATION] + 0.2 \cdot internal[I\_STRESS] \\
s[S\_SOFTNESS] &= -0.1 - 0.3 \cdot internal[I\_STRESS] + 0.2 \cdot relationship[R\_EMOTIONAL\_SAFETY] \\
s[S\_ENTHUSIASM] &= -0.2 + 0.5 \cdot internal[I\_ENERGY] - 0.3 \cdot internal[I\_MENTAL\_FATIGUE] \\
s[S\_RESTRAINT] &= -0.1 + 0.3 \cdot internal[I\_INSECURITY] + 0.2 \cdot traits[T\_PRIDE] \\
s[S\_VULNERABILITY] &= -0.5 + 0.3 \cdot internal[I\_LONELINESS] + 0.2 \cdot internal[I\_LONGING] - 0.2 \cdot traits[T\_PRIDE]
\end{aligned}
$$

外表刺激影响：

$$
\begin{aligned}
s[S\_WARMTH] &+= 0.30 \cdot outer\_stimuli[ST\_VALIDATION] \\
s[S\_SHARPNESS] &+= 0.25 \cdot outer\_stimuli[ST\_CONFLICT] \\
s[S\_SOFTNESS] &+= 0.20 \cdot outer\_stimuli[ST\_CLOSENESS] \\
s[S\_VULNERABILITY] &+= 0.15 \cdot outer\_stimuli[ST\_ABANDONMENT] \\
s[S\_RESTRAINT] &+= 0.20 \cdot outer\_stimuli[ST\_EMOTIONAL\_WEIGHT] \\
s[S\_SHARPNESS] &+= 0.10 \cdot outer\_stimuli[ST\_TEASING] \\
s[S\_WARMTH] &+= 0.10 \cdot outer\_stimuli[ST\_DEPENDENCY]
\end{aligned}
$$

特质修饰通过 sigmoid 软阈值进行连续加权。

最终结果使用 $\text{soft\_clamp}(\cdot, -1, 1)$ 限幅。

---

## 时间衰减

文件：`state_engine/_decay.py`

### 核心公式

$$
decayed[s] = baseline[s] + (current[s] - baseline[s]) \cdot e^{-\lambda_{eff}[s] \cdot \Delta t}
$$

其中：
- $baseline$ 为人格特质决定的 setpoint
- $\Delta t$ 为实际时间间隔（小时）
- $\lambda_{eff}[s] = \dfrac{\lambda_{base}[s] \cdot personality\_mod}{1 + k \cdot \Delta t}$

### 非对称衰减（06-19 新增）

关系态中，当 `current[s] < setpoint[s]`（负向偏离，即负面印象）时，λ_eff 乘以 `negative_decay_boost`（默认 1.8），使负面印象消退快于正面。这对应心理学中的 **Fading Affect Bias (FAB)**。

```python
negative_mask = deviation < 0
lam[negative_mask] *= config.negative_decay_boost
```

### 人格调制

内部状态：
- $T_{EMOTIONAL\_STABILITY}$、$T_{OPTIMISM}$ 提高恢复速度
- $T_{ANXIETY\_PRONENESS}$、$T_{ANGER\_REACTIVITY}$ 降低恢复速度
- $T_{EMOTIONAL\_OPENNESS}$ 轻微提高恢复速度

关系状态：
- $T_{ATTACHMENT\_AVOIDANCE}$ 提高衰减速度（更快疏远）
- $T_{ATTACHMENT\_ANXIETY}$ 降低衰减速度（更难放下）
- $T_{EMOTIONAL\_STABILITY}$ 轻微降低衰减速度（关系更稳定）

### 时间曲线

时间曲线参数 $k$ 用于让长时间间隔衰减速率逐渐放缓：

$$
\lambda_{eff} = \frac{\lambda_{base} \cdot personality\_mod}{1 + k \cdot \Delta t}
$$

默认：
- 内部 $k = 0.05$ → 渐近残余 exp(-λ_base·p_mod/k) 可达 9%~49%
- 关系 $k = 0.001$ → 渐近残余更显著（TRUST 最坏 66%）

这是幂律尾效应（affective chronometry），公式上 λ_eff → 0 当 Δt → ∞，状态永远不会完全收敛到 setpoint。这意味着严重的关系伤害可能留下"影子"。

### 接口

- `apply_time_decay_internal(...)`
- `apply_time_decay_relationship(...)`
- `apply_time_decay(...)`

> 这里的时间衰减是唯一负责"对话间隔内向人格基线回归"的机制。

---

## 关键设计原则

- 每轮对话中的状态更新不直接拉向人格基线，避免对话内"假性恢复"。
- 防御剖面控制的是刺激接受/表达速率，而不是直接修改状态值。
- 内部状态更新与关系状态更新采用同构残差形式，但关系节奏更慢。
- 表面状态由内部状态、关系状态与被压抑的外表刺激共同决定。
- 真正的稳态恢复由真实时间间隔驱动的指数衰减完成。
- 负向关系印象（受伤）消退快于正向（升温），boost=1.8×，由 Fading Affect Bias 支持。
- **soft_clamp** 默认 transition=0.1，输出在 ±1 上下有 ±0.1 的平滑过渡区。

---

## 文件映射

- `state_engine/_pipeline.py`：主流程调度
- `state_engine/_defenses.py`：防御剖面与 inner/outer 刺激
- `state_engine/_dynamics.py`：内部与关系状态残差更新、基线计算
- `state_engine/_surface.py`：表面表达投影
- `state_engine/_decay.py`：时间衰减与稳态恢复
- `state.py`：状态向量索引与默认基线

---

## 附录：前瞻方向与研究参考

> 以下内容来自之前的独立研究报告（06-15），经筛选保留仍适用的前瞻方向。
> **已解决的旧问题**（LSTM 三门控、decay>1、矩阵正反馈、硬编码阈值）已不再列出。

### A.1 情绪与评价模型扩展方向

当前 InternalState 偏向关系性负性状态（stress/loneliness/insecurity/irritation/longing），缺少：

- **基本情绪维度**：joy/sadness/fear/anger/disgust/surprise（Ekman / Plutchik）
- **核心情感维度**：valence / arousal / dominance（Russell circumplex / PAD）
- **评价维度**：novelty, relevance, goal_congruence, coping_potential（Scherer CPM / Lazarus）
- **社会情绪**：guilt, shame, pride, gratitude, disappointment
- **认知状态**：uncertainty, rumination, hope, regret

**关键文献**：
- Ortony, A., Clore, G. L., & Collins, A. (1988). *The Cognitive Structure of Emotions*.
- Scherer, K. R. (2001, 2009). Component Process Model of Emotion.
- Russell, J. A. (1980). A circumplex model of affect.
- Frijda, N. H. (1986). *The Emotions*.

### A.2 人格动力学方向

当前 Traits 是静态的。可扩展方向：
- **Whole Trait Theory**（Fleeson, 2001）：特质是状态的密度分布
- **Latent State-Trait Theory**（Steyer et al., 1999）：观察 = 特质 + 情境 + 误差
- **Trait Activation Theory**：情境激活特质相关目标
- **Cybernetic Big Five Theory**（DeYoung, 2015）：setpoint / 时间常数 / 刺激权重

### A.3 动机与目标系统

当前角色只有刺激 → 反应，缺少主动驱动。
- **SDT**（Deci & Ryan）：autonomy / competence / relatedness 需求
- **目标层级**：维持亲密、寻求安慰、回避冲突、确认被爱
- **行动倾向**（Frijda, 1986）：情绪产生 action tendency——靠近、逃避、攻击、求助
- **内部独白/反刍**：在非对话时刻运行 reflection loop

### A.4 用户模型与 Theory of Mind

- 用户的人格特质、依恋风格、情绪推断
- 互惠性预期（reciprocity）
- **Bowlby IWM**：对自我和对他人的双向表征
- **Theory of Mind**（Premack & Woodruff, 1978）

### A.5 开源与工业参考

| 类别 | 项目 | 参考价值 |
|------|------|---------|
| 学术情感架构 | FAtiMA（OCC+BDI） | 评价层 + 记忆 + 动机 + ToM |
| 工业 AI 陪伴 | Nomi.ai, Replika | 长期记忆与情感记忆 |
| 记忆基础设施 | Mem0, LangMem, Zep, Letta | 语义/情景记忆、记忆巩固 |
| 评估基准 | CharacterEval, PERSIST, RPEval | 角色一致性评测 |
| 开源引擎 | SillyTavern | JSON 配置 + SQLite 持久化 |

### A.6 路线图：从当前到完整心理主体

**Phase 1 ✅ 已完成**（数学基础修复）→ 当前架构

**Phase 2：扩展情绪与评价模型（2-3 周）**
- 增加基本情绪/核心情感维度
- 引入 Appraisal 层：goal_congruence, certainty, agency 等
- 扩展刺激维度

**Phase 3：记忆系统完善（2-3 周）**
- 记忆注入节点 + 记忆总结节点（当前为 stub）
- 情景记忆存储 + LLM 巩固
- 感知上下文窗口 4 → 12~20

**Phase 4：目标、动机与行动系统（3-4 周）**
- 目标层级（短期+长期）
- SDT 需求系统
- 行动倾向 → 行为系统

**Phase 5：用户模型与 ToM（2-3 周）**

**Phase 6：工程化（持续）**
- 权重外部化 YAML/JSON
- State Formatter 连续化
- LLM-as-a-judge 评估闭环
- FastAPI 服务化

### A.7 可直接参考的资源

**论文**：
1. Ortony, A., Clore, G. L., & Collins, A. (1988). *The Cognitive Structure of Emotions*.
2. Scherer, K. R. (2009). The dynamic architecture of emotion.
3. Fleeson, W., & Jayawickreme, E. (2015). Whole Trait Theory.
4. Bowlby, J. (1969/1982). *Attachment and Loss*.
5. McAdams, D. P. (2001). The psychology of life stories.
6. DeYoung, C. G. (2015). Cybernetic Big Five Theory.
7. Kuppens, P., & Verduyn, P. (2017). Emotion dynamics.
8. Marsella, S., Gratch, J., & Petta, P. (2010). Computational models of emotion.
9. Dias, J., et al. (2014). FAtiMA Modular.
10. *Beyond Fixed Psychological Personas: State Beats Trait* (arXiv:2601.15395).

**开源项目**：
1. FAtiMA Toolkit — `https://fatima-toolkit.eu/`
2. Mem0 — `https://github.com/mem0ai/mem0`
3. LangGraph Memory / LangMem — LangChain 官方
4. Letta (ex-MemGPT) — `https://github.com/letta-ai/letta`
5. Zep — `https://github.com/getzep/zep`
6. SillyTavern — `https://github.com/SillyTavern/SillyTavern`

**标准**：
- W3C EmotionML 1.0 — 情绪标注标准
- ISO 24617-2 — 对话行为标注
