# Surface Projection 研究综述

> 2026-06-22 | 面向 `project_surface()` 函数重构的跨学科理论基础调查
>
> 共两轮搜索，覆盖约 30 个研究领域，旨在为约束①（特质直连表面）和
> 约束④（outer_stimuli 跳过动力学层）的修复提供学术依据，
> 并为将来的惯性更新 + SSM 架构铺设理论基础。

---

## 目录

1. [核心心理学理论](#1-核心心理学理论)
2. [计算情感建模框架](#2-计算情感建模框架)
3. [人格与个体差异](#3-人格与个体差异)
4. [社会与人际互动](#4-社会与人际互动)
5. [生理与神经生物学](#5-生理与神经生物学)
6. [临床与计算交叉](#6-临床与计算交叉)
7. [文化与群体差异](#7-文化与群体差异)
8. [对 `project_surface` 重构的跨学科综合建议](#8-对-project_surface-重构的跨学科综合建议)
9. [参考文献清单](#9-参考文献清单)

---

## 1. 核心心理学理论

### 1.1 显示规则（Display Rules）

**起源**: Ekman & Friesen (1975) 神经文化模型

**定义**: 文化习得的"从私人情绪体验到公开情绪行为的映射规范"——规定谁可以对谁展示什么情绪、何时展示。

**四种修饰模式**:

| 模式 | 行为 | 感受 | 与你模型的关系 |
|------|------|------|-------------|
| **最小化（Minimization）** | 表达＜感受 | 抑制 | Deactivation 的输出——outer_stimuli |
| **夸大化（Exaggeration）** | 表达＞感受 | 放大 | Hyperactivation 的产物 |
| **中性化（Neutralization）** | 无表达 | 感受存在 | 极端克制——S_RESTRAINT 极高 |
| **替代（Substitution）** | 不同表达 | 感受存在 | 微笑掩盖焦虑——S_WARMTH↑ + I_STRESS↑ |

**对你的架构的直接确认**: outer_stimuli 是 deactivation 压抑后的"残余"是心理学上正确的设计。但每维刺激应该有不同的抑制系数（validation 的抑制程度 ≠ conflict 的抑制程度），这与你的防御配置函数对应。

**三维结构（Dawel et al., 2022）**:
- **Harmonious/Affiliative** — 促进社交联结（对应 S_WARMTH, S_SOFTNESS）
- **Vulnerable** — 传达需要或弱点（对应 S_VULNERABILITY）
- **Disharmonious/Dominant** — 宣示权力或破坏和谐（对应 S_SHARPNESS）

→ 你的 7 维 surface 可以自然归入这三维，验证了维度选择的合理性。

**人格×情境×关系交互（Matsumoto et al., 2005）**:
- 外向性 → 在远距离关系中增强抑制（最大化社交回报）
- 神经质 → 在公共场合增加抑制（社交焦虑驱动）

→ 显示规则是 **personality × context × relationship** 三者交互的产物，不是固定系数。

---

### 1.2 Gross 情绪调节过程模型

**1998 原始模型 → 2015 扩展过程模型（EPM）**

**五阶段**:
| # | 阶段 | 策略示例 | 在你的模型中 |
|---|------|---------|-----------|
| 1 | 情境选择 | 靠近/回避 | 外显行为层 |
| 2 | 情境修改 | 主动改变 | 同上 |
| 3 | 注意力部署 | 分心/专注 | Hyperactivation 前置 |
| 4 | **认知改变** | **认知重评** | 防御配置≈自动重评 |
| 5 | **反应调制** | **表达抑制** | **Deactivation → outer_stimuli** |

**2015 扩展**: 四阶段元调节循环
```
识别（要不要调）→ 选择（用什么策略）→ 执行（具体战术）→ 监控（评估效果）
```

**对你的启示**:
- 当前只实现了第⑤阶段（反应调制）的表达抑制，缺少前因阶段的建模
- **重评比抑制更适应**: 抑制不减少负面体验，增加交感激活、损害社交联系 → 你的"压抑后的 outer_stimuli"在心理学上正确——压抑不会消除内部感受
- **关键设计建议**: 增加 surface → internal 的反馈回路——长期 surface acting 应增加 stress/mental_fatigue（情绪劳动成本）

---

### 1.3 Hochschild 情绪劳动理论（The Managed Heart, 1983）

**核心概念与你的架构的直接映射**:

| Hochschild 术语 | 定义 | 你的模型对应 |
|----------------|------|------------|
| **Surface Acting** | 只改变外部表达，不改内部感受 | `project_surface()` 的线性映射 |
| **Deep Acting** | 先改变内部感受，表达自然跟随 | `_dynamics.py` 更新 + 防御配置 |
| **Feeling Rules** | 社会规范告诉你"应该"有什么感觉 | traits + ST_* 基线 |
| **Emotive Dissonance** | 内外不一致产生的紧张 | S_RESTRAINT 与 I_STRESS 同时高 |
| **自异化** | 长期 acting 后分不清真假感受 | 长期高 S_RESTRAINT→降低自我觉察 |

**对你的启示**:
- `project_surface` 的线性映射 = **surface acting**——合理但缺少切换到 **deep acting** 的条件机制
- 建议增加 `acting_mode` 参数（0 = surface acting, 1 = deep acting），由 trait 或资源消耗驱动

---

### 1.4 Affect Control Theory（Heise, 2007）

**核心**: 人们在互动中努力维持文化共享的意义——**EPA 三维空间**（Evaluation, Potency, Activity）。

**计算模型 INTERACT**:
- 使用大规模文化意义词典（EPA 评分）
- 多变量非线性方程模拟:
  - 人们的行动选择
  - 对事件的**情绪反应**
  - 重新定义自我/他人以解释意外行为

**BayesACT（Hoey & Schröder, 2023）**: 引入不确定性，允许文化意义随时间学习变化。

**对你的启示**:
- 你的 S_WARMTH ≈ Evaluation, S_SHARPNESS ≈ Potency/Activity——7D surface 本质是高维 EPA 空间
- ACT 的"情绪 = EPA 空间中的偏移"验证了投影映射的合理性
- **关键缺失**: ACT 中**事件改变意义**，而你的 surface 每轮独立计算——应添加 surface → internal 的反馈

---

## 2. 计算情感建模框架

### 2.1 EMA 模型（Marsella & Gratch, 2009）

**核心论文**: *EMA: A process model of appraisal dynamics*, Cognitive Systems Research, 10(1), 70-90.

**关键主张**:
- 反对多级评估假设，提出**单一自动评估过程**
- 动态**来自感知和推理过程对情境解释的变化**，而非多个评估系统
- 区分快速情感反应和慢速审慎反应

**Coping 机制**:
- 问题导向应对 → 改变环境
- 情绪导向应对 → 改变解释

**对你的启示**: EMA 支持"单一线性投影 + 非线性门控"的混合架构，但强调动态来自**时间积分**而非每轮独立计算。

---

### 2.2 OCC + Agent Personality 表达模型（Malatesta et al., NTUA）

**关键贡献**:
- 简化版 OCC 模型用于虚拟角色
- 显式建模**表达性参数**（面部、身体）从内部评估状态映射
- **人格特质（外向性/神经质）和情绪状态调制表达反应倾向**
- **关键设计**: 特质的调制通过改变映射的 **gain/slope**，而非直接加在表面上

→ 这与你的 sigmoid 门控一致，但他们的模型是**全局 trait → surface gain 调制**，而你是 trait 直接加在特定维度上。

---

### 2.3 分层情感架构（AME — Luo et al., 2011）

```
Attitude (长期) → Mood (中期) → Emotion (短期) → Expression
```

验证三层分离（traits / internal / surface）架构的正确性，但建议 mood 作为独立中间层。

### 2.4 Panksepp 情感神经科学的计算实现（Joseph & Levkowitz, 2011）

实现了全部 **7 个 Panksepp 系统**（SEEKING, RAGE, FEAR, PANIC, LUST, CARE, PLAY）作为**线性七变量动力学系统**，驱动 Pacman 智能体。

**关键发现**: 即使简单线性模型也复现了哺乳动物的基本情绪模式（单波段、双波段、三波段情绪状态随时间的变化）。

**对你的启示**:
- 你的 8D internal 与 Panksepp 7 系统的概念映射:
  - I_ENERGY ≈ SEEKING
  - I_IRRITATION ≈ RAGE
  - I_STRESS ≈ FEAR
  - I_LONELINESS ≈ PANIC
- Panksepp 系统的**相互作用**是关键动力学特征:
  - SEEKING 受阻 → RAGE 激活（你的 I_ENERGY↓ → I_IRRITATION↑ 应更显式）
  - CARE 激活 → 抑制 FEAR（R_AFFECTION → I_STRESS↓ 已有但不够强）

---

## 3. 人格与个体差异

### 3.1 CAPS 认知-情感处理系统（Mischel & Shoda, 1995, 1998）

**核心**: 人格 = 稳定的**认知-情感单元（CAUs）网络**，每种情境激活不同子集→产生不同的行为→ **If...Then 行为签名**。

| CAUs 类型 | 你的对应 |
|-----------|---------|
| 编码（Encodings） | outer_stimuli |
| 期望/信念 | traits |
| 情感（Affects） | internal state |
| 目标/价值 | **缺少！** |
| 自我调节计划 | **缺少！** |

**计算模型**: 平行约束满足网络（连接主义），固定权重 + 情境输入 → 稳定行为签名。

**对你的启示**:
- 固定权重设计（你的矩阵）在 CAPS 中已被验证能产生稳定的情境依赖行为
- 考虑**情境门控**——不是所有 CAU 都被激活，而是情境选择子集
- **最大缺失**: 目标单元和自我调节计划。AI 没有"我想这样表达"的动机层

---

### 3.2 特质激活理论（Tett & Burnett, 2003）

**核心**: 特质是**潜伏的潜能**，需要**情境线索**来激活才能在行为中表达。

**三层情境线索**:
| 层级 | 在你的模型中 |
|------|------------|
| 任务层 | 对话主题/用户输入 |
| 社交层 | relationship state |
| 组织层 | 无对应（可忽略） |

**五个功能特征**:
| 特征 | 效应 |
|------|------|
| Demand | 特质表达→正向结果 |
| Distracter | 特质表达→负向结果 |
| Constraint | 降低特质相关性 |
| Releaser | 撤销约束 |
| Facilitator | 放大线索显著性 |

**对你的启示**:
- 你的 sigmoid 门控已蕴含特质激活思想——但也只是简单的阈值
- 应使用情境向量（outer + relationship）来**门控哪些 trait→surface 连接被激活**:
  ```python
  context_safety = (1 - outer[ST_CONFLICT]) * (1 + relationship[R_AFFECTION])
  anxiety_gate = sigmoid(context_safety / threshold)  # 不安全情境才激活焦虑特质
  ```

---

### 3.3 自我差异理论（Higgins, 1987）

**三种自我状态的不一致产生特定情绪**:
- **现实 vs 理想自我** → 失望/不满（dejection）
- **现实 vs 应该自我** → 焦虑/内疚（agitation）

**对你的启示**: 可以计算自我差异并反馈到 internal:
```python
ideal_surface = compute_from_traits(traits, mode="ideal")
ought_surface = compute_from_traits(traits, mode="ought")
self_discrepancy = surface - ideal_surface  # 反馈到 internal
```

### 3.4 自决理论（Deci & Ryan, SDT）

**三个基本心理需求**:
- 自主性（Autonomy）
- 胜任感（Competence）
- 关系性（Relatedness）

**对你的启示**: 需求挫败是情绪的直接来源:
- 自主性挫败 → 愤怒/抗拒（ST_CONFLICT）
- 胜任感挫败 → 焦虑/羞耻（**缺少对应刺激维度**）
- 关系性挫败 → 孤独/抑郁（ST_ABANDONMENT）

**动态 SEM 研究（2026, J. Happiness Studies）**: 情绪整合（vs 抑制）与需求满足有双向关系。

---

## 4. 社会与人际互动

### 4.1 情绪传染与自动模仿（Hatfield, Cacioppo & Rapson, 1993）

**三步模型**:
1. **自动模仿**（125-200ms）—— 自动同步面部/声音/姿态
2. **反馈** —— 模仿的肌肉活动通过传入神经影响主观体验
3. **趋同** —— 情感状态向对方靠拢

**对你的启示**: 你的模型**缺少对用户情绪表达的反应**。建议添加从用户输入直接到 `project_surface` 的**情绪传染通道**（不经过动力学层）。

### 4.2 社交基线理论（Beckes & Coan, 2011）

**核心**: 人类大脑的基线假设是"周围有可预测的社交网络"。社交靠近**降低自身调节成本**:
- 风险分担
- 负荷共享

**对你模型的启示**:
- R_AFFECTION 高时应**全局提高 surface 的响应增益**（安全感使表达更自如）
- 不是独立加性效应，而是**全局增益调制**:
  ```python
  social_gain = 1.0 + beta * sigmoid(relationship[R_AFFECTION] / gain_scale)
  s = SURFACE_MAPPER.compute(sources) * social_gain
  ```

### 4.3 文化约束的 Affect Consistency（Schröder et al., 2013）

首次对非语言行为验证 ACT——120 被试、60 对，视频编码友好度/支配度/活跃度。ACT 计算模型成功预测了人际情感频率和时序。

### 4.4 虚拟智能体同理心模型（IEEE Process Model）

**四因素调制同理心反应强度**:
1. 智体间相似度
2. 情感联结（R_AFFECTION）
3. 当前情绪状态
4. 人格

**建议**: 将同理心建模为三个层次——情绪传染（直接映射）→ 情感共鸣（R_AFFECTION 门控）→ 认知共情（LLM 推理）。

---

## 5. 生理与神经生物学

### 5.1 Polyvagal Theory（Porges, 1995）

**三阶段自主神经层级**:

| 状态 | 神经系统 | 行为模式 | `project_surface` 对应 |
|------|---------|---------|---------------------|
| 社交参与 | 腹侧迷走神经 | 安全、连接 | I_STRESS↓ + S_WARMTH↑ |
| 动员 | 交感神经 | 战斗/逃跑 | I_STRESS中 + S_SHARPNESS↑ |
| 固定不动 | 背侧迷走神经 | 冻结/关闭 | I_STRESS↑ + S_EXPRESSIVENESS↓↓ |

**建议**: 在 `project_surface` 中加入基于 I_STRESS 的**质的状态切换**:
```python
if stress < 0.3: mode = "social_engagement"
elif stress < 0.7: mode = "mobilization"
else: mode = "immobilization"
```

### 5.2 Damasio 躯体标记假说

**双重表征**:
- **身体环路**: 实际生理变化
- **作为-身体环路**（as-if body loop）: 大脑直接模拟身体变化

**DARE 架构（Maçãs et al., 2001）**:
- 快速感知级（≈ 外刺激→表面直接通道）
- 慢速认知级（记忆匹配、预期模拟）

**对你的启示**: `project_surface` 的输出应回馈到 LLM 决策作为躯体标记——这已通过 `state_formatter → LLM` 实现。但缺少从 LLM 行动选择到 internal 的预期躯体标记反馈。

---

## 6. 临床与计算交叉

### 6.1 情绪动力学——惯性与变异性（Kuppens & Verduyn, 2015）

**VAR(1) 模型**: `s(t) = Φ · s(t-1) + ε(t)`

| 特征 | 定义 | 数学表示 |
|------|------|---------|
| **惯性** | 情绪持续趋势 | Φ 对角线 |
| **交叉滞后** | 情绪间预测 | Φ 非对角线 |
| **变异性** | 波动幅度 | ε 方差 |
| **颗粒度** | 区分能力 | 情绪间协方差 |

**对你的 `project_surface` 的直接启示**:
- **当前最大设计缺陷**: 完全无惯性——每轮独立计算
- 建议添加表面惯性:
  ```python
  s(t) = α · project_surface(internal, relationship, traits, outer_stimuli)
         + (1-α) · s(t-1)
  ```
  其中 α ∈ (0,1] 由当前情绪强度控制（高 arousal → 低惯性，更容易变化）
- 交叉滞后: 高 S_RESTRAINT 预测下轮 S_WARMTH↓（长期克制使人变冷）

### 6.2 Barrett 情绪建构理论（Conceptual Act Theory）

**核心**: 情绪不是被"触发"，而是在当下**从更基本成分建构**:
```
Core Affect（核心情感: 效价+唤醒）
 + Conceptual Knowledge（概念知识）
 + Categorization（自动分类）
= Constructed Emotion
```

**对你的启示**:
- 你的 8D internal = core affect 的高维版本
- surface 7D = 概念空间的分类标签
- 建议将 surface 投影重新概念化为**从核心情感空间到概念空间的分类映射**

### 6.3 述情障碍（Alexithymia）

**处理模型（2001, Cognitive Systems Research）**: 述情障碍 = 情绪处理组件之间的**信息传输失败**。

**对你模型的启示**:
- 高述情障碍 = trait→surface 连接噪声增大 + sigmoid 阈值提高（更难激活）
- 表面表达趋向"平淡"——与 S_RESTRAINT 不同，述情障碍是"没有感受可表达"而非"选择不表达"

### 6.4 情绪粒度（Emotional Granularity）

**Barrett (1995) + Kashdan et al. (2015)**:
- 高粒度 → 精细区分"失望"vs"悲伤"vs"孤独"
- 低粒度 → 笼统感受"糟糕"

**建议**: 添加情绪粒度参数 σ_granularity:
- 高 → surface 7 维协方差高（维度相关，表达笼统）
- 低 → 7 维独立变化（表达精细）
- 应与 T_EMOTIONAL_OPENNESS 关联

### 6.5 Affective Computing（Picard, 1997）

**关键框架**:
- 情感识别 × 表达合成 × 认知模型（OCC）
- 三层架构: 感知 + 认知 + 表达
- Believable Agents: 通过合适时机和怪癖创造"生命幻觉"

**对你的启示**: 3 步架构（Defense → Dynamics → Surface）符合 Picard 规范。但缺少**多模态表达**区分（面部/声音/语言共用 7D 空间）。

---

## 7. 文化与群体差异

### 7.1 大规模计算研究证据（McDuff et al., 2017）

计算机视觉分析 **740,984 名参与者 × 12 个国家**:
- **个人主义文化**: 表达更开放，但情境差异大
- **集体主义文化**: 表达更抑制，情境差异小
- 微笑依赖文化×环境交互

### 7.2 性别差异与显示规则

**模式**:
- 女性: 更高微笑、悲伤、恐惧表达
- 男性: 更高愤怒和自豪
- 集体主义文化中性别差异最小（规范约束强于性别角色）

---

## 8. 对 `project_surface` 重构的跨学科综合建议

### 8.1 设计目标（从用户需求提炼）

1. **惯性更新**: surface 与 internal/relationship 同等的惯性动力学，而非每轮独立重算
2. **SSM 准备**: 架构设计为将来 surface 比 internal/relationship 更新更频繁（更高时间分辨率）预留接口
3. **outer_stimuli 已是门控产物**: outer_stimuli 是 deactivation 压抑后的残余表达，不重复压抑
4. **保持内部状态影响**: surface 不是独立更新的，受 internal / relationship / traits 驱动

### 8.2 建议架构

**核心设计原则**:
- surface 的输入源: **internal + relationship + outer_stimuli**（不含 traits——traits 已通过 pipeline 间接实现）
- **双向链路**: internal ↔ surface（与 internal ↔ relationship 同级）
- 惯性更新: surface 有自己的时间积分，不每轮独立重算
- SSM 预留: 架构弹性支持未来 surface 细粒度微步

```
每轮（低频，对应每轮对话）:
  ┌──────────────────────────────────────────────────────────────┐
  │  ① 动力学更新                                                  │
  │    internal(t) = f_dynamics(internal(t-1), inner_stimuli,     │
  │                             relationship, surface_feedback)   │
  │    relationship(t) = g_dynamics(relationship(t-1), internal,  │
  │                                 relationship_stimuli)         │
  │                                                               │
  │  ② 表面投影（有惯性）                                            │
  │    raw_surface(t) = SURFACE_MAPPER.compose(                   │
  │        internal(t), relationship(t), outer_stimuli(t)         │
  │    )  # ← 不含 traits，traits 已通过 defense/dynamics 间接实现   │
  │                                                               │
  │    s(t) = α(t) · raw_surface(t) + (1 - α(t)) · s(t-1)        │
  │    其中 α(t) = f_alpha(internal)  # arousal 高→α高（变化快）   │
  │                                                               │
  │  ③ 表达反馈                                                    │
  │    surface_feedback = g_feedback(s(t), internal(t))           │
  │    # 反馈到下轮动力学（如长期高 stiffness → 增加 stress）         │
  └──────────────────────────────────────────────────────────────┘

将来 SSM 架构:
  （每轮可包含 N 个 surface 微步，每个微步：
     s[t+δ] = A_ssm · s[t] + B_ssm · raw_surface(t)
     internal/relationship 保持静默）
  这为将来 surface 更高时间分辨率预留了架构弹性。
```

### 8.3 关键设计参数来源

| 参数 | 推荐来源 | 心理学基础 |
|------|---------|-----------|
| α(t) 惯性系数 | Kuppens & Verduyn (2015) | 情绪惯性——高 arousal 时惯性降低 |
| social_gain | Beckes & Coan (2011) | 社交基线——安全关系放大表达增益 |
| sigmoid 阈值（已移除） | ~~Tett & Burnett (2003)~~ | traits 已通过 defense pipeline 间接实现 |
| **surface → internal 反馈** | **Gross (2015) / Hochschild (1983) / Hatfield et al. (1993)** | **情绪劳动成本 + 面部反馈假说——新设计** |
| 情绪传染通道 | Hatfield et al. (1993) | 自动模仿——125-200ms 同步 |

### 8.4 Internal ↔ Surface 双向链路设计

与现有的 internal ↔ relationship 双向耦合平级。

**Forward 方向（internal → surface）**:
- 已有实现: `project_surface()` 从 internal/relationship/outer_stimuli 投影
- 预期修改: 移除 traits 作为直接输入，改为通过 pipeline 间接影响

**Backward 方向（surface → internal）**:
- **新设计，尚未实现**
- 心理学基础:
  - **Hochschild (1983) 情绪劳动**: 长期 surface acting（S_RESTRAINT 持续高）导致 stress↑, mental_fatigue↑, burnout——表达压抑有代谢成本
  - **Hatfield et al. (1993) 面部反馈**: 表达温暖（S_WARMTH↑）实际上会反馈影响感受——表达改变体验
  - **Gross (2015) 反应调制循环**: response modulation 不是终结点，而是整个调节循环的一部分
  - **Damasio (1994) 作为-身体环路**: 表达本身就是身体状态的一部分，反馈到感知
- 可能的实现:
  ```python
  # surface → internal 反馈
  # 1. 情绪劳动成本: 表达与内部感受不一致 → 增加 stress
  dissonance = compute_dissonance(surface, internal)
  delta_stress += beta_cost * dissonance
  
  # 2. 表达放大感受: 表达温暖 → 实际感觉温暖（facial feedback）
  delta_warmth += gamma_feedback * s[S_WARMTH]
  
  # 3. 长期抑制成本: 长期高 S_RESTRAINT → mental_fatigue↑
  delta_fatigue += eta_suppression_cost * max(0, s[S_RESTRAINT] - threshold)
  ```

### 8.4 对约束①的修复路径

当前 trait 直连 surface（`_surface_weights.py:158-162` 标记为已知违规）:

**2026-06-22 更新: 设计决策** —— traits 不应直接作为 surface 的输入。

traits 已通过以下路径**间接影响 surface**:

```
traits ──→ defense profiles ──→ deactivation/hyperactivation ──→ outer/inner stimuli
traits ──→ dynamics (decay rate, setpoint modulation) ──→ internal
```

不需要再通过 `SURFACE_MAPPER` 添加 traits 到 surface 的直接连接。
`_surface_weights.py:158-162` 中标记为违规的直接 trait 连接（T_PRIDE → S_RESTRAINT, T_PRIDE → S_VULNERABILITY）应当被移除。

**替代方案**: 保留 sigmoid 门控中 trait 对 surface 的**增益调制**功能，但改为:
- trait 调制的是 internal/relationship/outer_stimuli 的响应增益（乘性），而非直接加性
- 或者完全依赖防御剖面和动力学层的间接路径，移除所有 trait 相关的非线性门控

**学术支持**: 
- Tett & Burnett (2003) 特质激活理论——特质需要情境线索激活，不应直接注入行为输出
- Mischel & Shoda (1995) CAPS——特质是 CAUs 网络的连接权重，不是直接输出成分
- 分层架构（AME, Luo et al. 2011; Picard 1997）——长期/中期/短期严格分层

### 8.5 对约束④的修复路径

当前 outer_stimuli 跳过动力学层（约束④违反）:

**学术建议**: 将 outer_stimuli 先送入动力学层作为外部驱动力:
```python
# 在 _dynamics.py 中新增 "表层内部状态" (Layer 2.5):
surface_internal(t) = surface_internal(t-1) 
    + Δt · (A_surf · surface_internal(t-1) + B_outer · outer_stimuli)
    
# 然后 surface 从 surface_internal 投影:
s(t) = SURFACE_MAPPER.compose(internal, relationship, surface_internal)
```

但这种做法与你的"outer_stimuli 已是门控压抑后的产物"设计理念可能冲突。
**替代方案**: outer_stimuli 保留为独立输入，但增加从 outer_stimuli 到 internal 的反馈:
```python
# outer_stimuli 不仅直接做 surface 映射，也反馈到 internal:
internal += B_outer_to_internal · outer_stimuli  # 梯度更新
```

---

## 9. 参考文献清单

### 核心文献

| # | 引用 | 领域 | 年份 |
|---|------|------|------|
| 1 | Marsella, S.C. & Gratch, J. EMA: A Process Model of Appraisal Dynamics. *Cognitive Systems Research*, 10(1), 70-90. | 计算情感建模 | 2009 |
| 2 | Ekman, P. & Friesen, W.V. Unmasking the Face. Prentice-Hall. | 显示规则 | 1975 |
| 3 | Dawel, A. et al. A Three-Dimensional Model of Emotional Display Rules. *Emotion*. | 显示规则维度 | 2022 |
| 4 | Matsumoto, D. et al. Integrating Personality, Context, Relationship, and Emotion Type into a Model of Display Rules. *J. Research in Personality*. | 人格×情境×显示规则 | 2005 |
| 5 | Gross, J.J. The Extended Process Model of Emotion Regulation. *Handbook of Personality*. | 情绪调节 | 2015 |
| 6 | Hochschild, A.R. The Managed Heart: Commercialization of Human Feeling. UC Press. | 情绪劳动 | 1983 |
| 7 | Mischel, W. & Shoda, Y. A Cognitive-Affective System Theory of Personality. *Psychological Review*, 102(2), 246-268. | CAPS | 1995 |
| 8 | Tett, R.P. & Burnett, D.D. A Personality Trait-Based Interactionist Model of Job Performance. *J. Applied Psychology*, 88(3), 500-517. | 特质激活 | 2003 |
| 9 | Kuppens, P. & Verduyn, P. Emotion Dynamics. *Current Opinion in Psychology*, 3, 22-26. | 情绪动力学 | 2015 |
| 10 | Krone, T. et al. A Multivariate Statistical Model for Emotion Dynamics. *Emotion*. | VAR-1 模型 | 2017 |
| 11 | Barrett, L.F. The Theory of Constructed Emotion. *Trends in Cognitive Sciences*. | 情绪建构 | 2017 |
| 12 | Kashdan, T.B. et al. Unpacking Emotion Differentiation. *Current Directions in Psychological Science*, 24(1), 10-16. | 情绪粒度 | 2015 |
| 13 | Hatfield, E. et al. Emotional Contagion. Cambridge University Press. | 情绪传染 | 1993 |
| 14 | Beckes, L. & Coan, J.A. Social Baseline Theory. *Social and Personality Psychology Compass*. | 社交基线 | 2011 |
| 15 | Porges, S.W. The Polyvagal Theory. *Biological Psychology*. | 自主神经层级 | 1995 |
| 16 | Panksepp, J. Affective Neuroscience. Oxford University Press. | 情感神经科学 | 1998 |
| 17 | Heise, D.R. Expressive Order: Confirming Sentiments in Social Actions. Springer. | Affect Control Theory | 2007 |
| 18 | Hoey, J. & Schröder, T. Bayesian Affect Control Theory. *American Behavioral Scientist*. | BayesACT | 2023 |
| 19 | Picard, R.W. Affective Computing. MIT Press. | 情感计算 | 1997 |
| 20 | Sheppes, G. et al. Emotion Regulation Choice. *J. Experimental Psychology: General*. | 策略选择 | 2014 |
| 21 | Higgins, E.T. Self-Discrepancy: A Theory Relating Self and Affect. *Psychological Review*, 94(3), 319-340. | 自我差异 | 1987 |
| 22 | Deci, E.L. & Ryan, R.M. Self-Determination Theory. *Handbook of Theories of Social Psychology*. | SDT | 2012 |
| 23 | McDuff, D. et al. Large-Scale Observational Evidence of Cross-Cultural Differences in Facial Behavior. *J. Nonverbal Behavior*. | 文化差异计算 | 2017 |
| 24 | Joseph, P.G. & Levkowitz, H. Patterns of Emotion Driven by Affect State and Environment. *PATTERNS*. | Panksepp 计算实现 | 2011 |
| 25 | Malatesta, L. et al. Agent Personality Traits in Virtual Environments Based on Appraisal Theory Predictions. *AAMAS*. | OCC + 人格 | 2010 |

### 次要引用

| # | 引用 | 年份 |
|---|------|------|
| 26 | Saarni, C. Children's Understanding of Display Rules. *Developmental Psychology*. | 1979 |
| 27 | Luo, et al. A Layered Model of Artificial Emotion Merging with Attitude (AME). | 2011 |
| 28 | Maçãs, et al. DARE: An Emotion-Based Agent Architecture. *FLAIRS*. | 2001 |
| 29 | Barthet, et al. Play with Emotion: Affect-Driven Reinforcement Learning. | 2022 |
| 30 | Tracey, et al. Default Defenses: Attachment Anxiety and Attachment Avoidance. *Current Psychology*. | 2023 |
| 31 | Forgas, J.P. Dual-Process Mood Management. *Psychological Inquiry*. | 2001 |
| 32 | Yanagisawa, et al. Free Energy Model of Emotional Valence in Dual-Process Perceptions. *Neural Networks*. | 2023 |
| 33 | Read & Miller. The Virtual Personality Model. *J. Psychopathology and Clinical Science*. | 2025 |
| 34 | Ekman, P., Levenson, R.W. & Friesen, W.V. Autonomic Nervous System Activity Distinguishes Among Emotions. *Science*, 221(4616), 1208-1210. | 1983 |
| 35 | Hofmann, W. et al. Working Memory Capacity and Self-Regulatory Behavior. *JPSP*. | 2008 |
| 36 | Aldao, A., Sheppes, G. & Gross, J.J. Emphasis on Flexibility in Emotion Regulation Strategy Use. | 2015 |

---

> 本文件会根据后续搜索和设计迭代持续更新。
