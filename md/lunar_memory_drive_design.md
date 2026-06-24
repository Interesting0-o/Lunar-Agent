# Lunar Agent: 记忆系统与内驱力系统综合设计文档

> **版本**: v1.0 | **日期**: 2026-06-23 | **状态**: 设计研讨
> **范围**: 记忆系统架构选型、记忆注入/总结时机、内驱力与主动发言系统
> **相关文档**: `MEMORY_SYSTEM.md` | `INTERNAL_DRIVE_SYSTEM.md` | `state_engine/__init__.py`

---

## 执行摘要

Lunar-Agent 当前的核心瓶颈在于两个相互关联的缺陷：**记忆系统只有存储和检索的骨架，缺乏完整的注入时机策略和情感驱动的检索逻辑**；**整个系统是纯反应式的，Agent 没有基于内在心理状态的主动发言能力**。本设计文档基于对 30+ 篇学术论文、开源方案（Mem0、Letta、MemGPT）和工业实践（Claude Dreaming、OpenClaw Dreaming）的系统调研，为 Lunar 提出一套统一的记忆-内驱力融合架构。

**核心结论**：

| 问题 | 推荐方案 | 心理学/工程学依据 |
|------|---------|------------------|
| **RAG vs LLM Wiki** | **混合架构**：以结构化情节记忆为主干，语义检索为辅助，逐步构建语义记忆层 | Mem0 的两阶段 pipeline（提取→更新）在 LoCoMo 上达 94.4% 准确率 [^31^]；RAG 适合外部知识，不适合会话记忆 [^19^] |
| **情感相似回忆** | **状态向量双索引**：internal_state 向量作为情感检索 key，与语义 embedding 加权融合 | Bower (1981) 的 Mood-Dependent Memory 理论 [^42^]；EASM 架构的 `R(m) = α·sim_sem + (1-α)·sim_emo` 公式 [^43^] |
| **记忆注入时机** | **热路径每轮检索 + 温路径对话结束形成 + 冷路径后台 consolidation** | Letta 的三层记忆架构 [^33^]；Claude/OpenClaw Dreaming 的三阶段模型 [^44^][^47^] |
| **主动发言** | **Inner Thoughts 框架适配**：内驱力向量 D(t) 生成 → 记忆检索激活 → 内在动机评估 → 发言决策 | Inner Thoughts 框架在 7 项指标上显著优于基线 [^30^]；Generative Agents 的 Plan-React 循环 [^92^] |

---

## 1. 记忆系统架构设计

### 1.1 研究综述：从 RAG 到 Agent Memory 的范式演进

在讨论 Lunar 的记忆架构之前，必须澄清一个根本性的概念区分：**RAG（Retrieval-Augmented Generation）与 Agent Memory 是服务于不同目标的两种技术**。RAG 的设计目标是将外部文档知识注入 LLM 上下文，解决的是"模型不知道的事实"问题；Agent Memory 的设计目标是持久化 Agent 与用户的交互历史，解决的是"模型忘记了刚才发生的事"问题 [^19^][^21^]。

2024-2026 年间，LLM Agent 的记忆架构经历了从简单到复杂的清晰演进轨迹。**第一代**是 MemGPT (Packer et al., 2023) 提出的 OS 类比架构，将 LLM 上下文管理比作虚拟内存分页，引入 Core/Recall/Archival 三层记忆 [^33^]。**第二代**以 Mem0 (Chhikara et al., 2025) 为代表，采用 LLM 驱动的原子事实提取 pipeline，将对话历史转化为结构化的事实条目存入向量数据库，在 LoCoMo 长对话基准上达到 94.4% 准确率，相比全上下文处理减少 91% 的延迟 [^31^][^74^]。**第三代**则向更丰富的表征发展：Zep 引入时序知识图谱追踪实体状态的演化 [^74^]；A-MEM 采用 Zettelkasten 方法自主建立记忆间的语义链接 [^75^]；EverMemOS 提出基于"印痕"（engram）生命周期的情节-语义分层模型 [^74^]。

对于 Lunar 这类**角色扮演型情感陪伴 Agent**，记忆系统的核心需求与通用任务型 Agent 有本质差异。ENPMR-Bench [^22^] 的最新研究明确指出：在情感支持场景中，"仅依赖语义相似度的记忆检索是不够的"——Agent 需要基于用户的**潜在情感需求**主动检索记忆，而非被动响应用户的显式查询。该基准测试显示，即使最优的嵌入模型在情感记忆检索上 Recall@10 也仅有 46.41%，Top-1 准确率不足 10%，这揭示了情感维度在记忆检索中的关键作用 [^22^]。

### 1.2 Lunar Memory OS 架构：三层记忆 + 温度路径

基于上述研究，Lunar 的记忆系统应采用**三层架构 + 温度路径**的设计，与你现有的 `MEMORY_SYSTEM.md` 草案保持一致但加以细化：

![Lunar Memory OS 架构](lunar_memory_architecture.png)

**三层记忆的职责划分**：

| 层级 | 对应现有代码 | 记忆类型 | 存储内容 | 更新频率 | 检索方式 |
|------|------------|---------|---------|---------|---------|
| **种子记忆** | `prompts/character_memories.py` | 静态锚点 | 24 条角色初始记忆、人格定义 | 永不 | 直接注入 system prompt |
| **情节记忆** | `MemoryStore("db/memories.json")` | 情节性 (Episodic) | 单次互动快照 + 状态向量 + embedding | 每次对话结束 | 状态向量相似度 + 语义相似度 |
| **语义记忆** | `db/semantic_memories.json` (v2) | 语义性 (Semantic) | 用户事实/偏好/关系摘要 | 后台 consolidation | 语义检索 + 结构化查询 |

**温度路径的运作逻辑**：

**热路径（每轮对话）**：`memory_retrieval_node` 在每轮用户输入后、LLM 生成响应前执行。输入为当前 `internal_state` 向量 + `user_message` 文本，输出为 `retrieved_memories` 字符串。**这一路径必须零 LLM 调用**，仅使用预先计算的向量相似度搜索，确保延迟 < 500ms。与你现有的 `memory.py` 中的 `hybrid_search` 对接，但需要将状态向量的权重调高的场景下（如用户情绪剧烈波动时），情感相似度的权重 `α` 应动态调整。

**温路径（对话间隙）**：`memory_formation_node` 在 LLM 生成响应后、对话结束前执行。评估本轮互动的**显著性**（significance），若超过阈值则生成 `MemoryNode` 并持久化。显著性评估可以基于你现有的 `ST_EMOTIONAL_WEIGHT` 指标——情感重量越高，记忆越值得保存。

**冷路径（后台处理）**：`consolidation_agent` 在系统空闲时运行（"Dreaming"），将积累的情节记忆合并、去重、提炼成语义记忆。这一机制直接借鉴 Claude Code 的 Auto Dream 和 Letta 的 sleep-time agents [^44^][^83^]。

### 1.3 情感相似检索：从 Bower 到 EASM

Lunar 的核心差异化能力在于**基于心理状态的情感相似回忆**。这一功能有坚实的心理学基础：Bower (1981) 的经典实验表明，编码和检索时的情绪状态匹配能显著提升回忆准确率，其效果与语义相似度相当 [^42^]。后续研究进一步区分了**情绪一致性**（mood-congruent，回忆与当前情绪同性质的记忆）和**状态依赖性**（mood-dependent，编码与检索时的情绪状态匹配）两种机制 [^46^][^49^]。

EASM（Emotion-Aware Semantic Memory）架构 [^43^] 提供了一个可直接采用的数学框架。该架构在 Qdrant 向量数据库中实现了**双索引**：每个记忆单元同时被语义内容和情感上下文索引。检索时的相关性得分公式为：

$$R(m) = \alpha \cdot sim_{sem}(m, q) + (1-\alpha) \cdot sim_{emo}(m, e)$$

其中 $sim_{sem}$ 是记忆 $m$ 与查询 $q$ 的语义相似度，$sim_{emo}$ 是记忆 $m$ 的情感状态与当前情感状态 $e$ 的相似度，$\alpha \in [0,1]$ 是可调权重系数 [^43^]。

对于 Lunar，这一公式可以**直接映射**到现有的代码结构：

| EASM 参数 | Lunar 对应实现 | 说明 |
|----------|---------------|------|
| $sim_{sem}(m, q)$ | `search_by_embedding(query_text)` | 用户消息与记忆内容的语义相似度 |
| $sim_{emo}(m, e)$ | `search_by_internal_state(current_internal)` | 记忆编码时的 `internal_state` 与当前状态的余弦相似度 |
| $\alpha$ | 动态权重，默认 0.5 | 可在 `config.py` 中配置；用户情绪激动时降低 $\alpha$（更依赖情感检索） |
| $e$ | 当前 `internal_state` 向量 | 8 维内部状态向量 |

**动态权重调整策略**：当检测到用户情绪剧烈波动（`ST_EMOTIONAL_WEIGHT > 0.7` 或 `I_STRESS / I_LONELINESS` 显著升高）时，自动降低 $\alpha$ 至 0.3-0.4，使情感相似度的权重提升。这与 State-Dependent Memory 的神经科学发现一致：情绪唤醒状态下，记忆检索更依赖编码时的情绪上下文 [^28^]。

### 1.4 RAG vs 结构化记忆：Lunar 的选型决策

你在 RAG 和 LLM Wiki 之间的摇摆，本质上是**非结构化语义检索**与**结构化知识组织**之间的权衡。基于调研，建议 Lunar 采用**"结构化情节记忆为主干，语义检索为辅助"**的混合策略，理由如下：

| 维度 | 纯 RAG 方案 | 纯 LLM Wiki 方案 | Lunar 混合方案 |
|------|-----------|----------------|--------------|
| **延迟** | 嵌入+检索 < 100ms | LLM 生成摘要 > 2s | 热路径 < 500ms，冷路径走 LLM |
| **情感检索** | 仅语义相似，无情感维度 | 依赖 LLM 理解情感 | **状态向量直接索引情感** |
| **与状态引擎协同** | 弱耦合，状态向量难以利用 | 中等耦合 | **强耦合，state_checkpoint 直接存储状态向量** |
| **可解释性** | 低（黑盒相似度） | 中（结构化条目） | **高（状态向量可追踪）** |
| **实现复杂度** | 低 | 高 | **中（逐步构建）** |

Lunar 的 `MemoryNode` 已经内置了 `state_checkpoint` 字段（保存 `internal_state`、`relationship_state`、`surface_state` 三个向量），这实际上已经构建了**结构化情节记忆的骨架**。v1 阶段应聚焦完善这一骨架的检索和注入链路；v2 阶段再引入语义记忆层和 consolidation agent。

---

## 2. 记忆注入时机与 Consolidation 策略

### 2.1 核心问题：什么时候进行记忆操作？

记忆系统的性能不仅取决于存储和检索的质量，更取决于**时机**——什么时候检索记忆注入对话上下文？什么时候将对话转化为持久记忆？什么时候进行后台整合？这三个问题的答案构成了记忆系统的"温度路径"。

![记忆注入时机决策流程](lunar_memory_timing.png)

### 2.2 热路径：每轮对话的记忆检索注入

**触发时机**：`memory_retrieval_node` 在 `state_formatter` 之后、`llm` 之前执行。输入为当前 `internal_state` + `relationship_state` + `user_message`，输出为格式化后的记忆字符串或 `None`。

**检索策略**：采用**双条件触发**机制：

**条件 A：语义触发**——用户消息与历史记忆存在语义关联。通过 `search_by_embedding(user_message, top_k=3)` 检测，相似度阈值设为 0.65（基于 nomic-embed-text 的经验值，可调整）。

**条件 B：情感触发**——当前 internal_state 与某段记忆编码时的状态高度相似。通过 `search_by_internal_state(current_internal, top_k=2)` 检测，余弦相似度阈值设为 0.75。

**注入格式**：检索到的记忆不应直接以原始文本形式注入，而应通过 `state_formatter` 转化为角色化的"回忆"描述。例如：

```
[回忆浮现]
你想起上次用户提到喜欢红月时，心里那种微微的悸动。
（相关记忆：关于一起看红月的约定 — 2026-06-15）
```

这种**间接注入**方式避免了记忆内容打断对话流，同时给 LLM 足够的信息来自然化用记忆。

**与状态引擎的协同**：`internal_state` 向量本身就是情感检索的 key，这意味着**状态引擎的输出直接驱动记忆检索**。当 `I_LONELINESS` 升高时，状态向量会自动引导检索到那些编码时同样孤独的记忆——这正是 Mood-Dependent Memory 的计算实现。

### 2.3 温路径：对话结束时的情节记忆形成

**触发时机**：`memory_formation_node` 在 LLM 生成响应后执行。与用户现有的 `MEMORY_SYSTEM.md` 设计一致，但需解决 `user_stimuli` 清理时机的关键约束。

**显著性评估**：决定是否将本轮对话持久化的核心机制。建议采用多因子评分：

$$S = w_1 \cdot ST_{EMOTIONAL\_WEIGHT} + w_2 \cdot \Delta_{relationship} + w_3 \cdot I_{novelty} + w_4 \cdot T_{elapsed}$$

| 因子 | 说明 | 权重建议 | 来源 |
|------|------|---------|------|
| $ST_{EMOTIONAL\_WEIGHT}$ | 感知层提取的情感重量 | 0.35 | perception.py 现有输出 |
| $\Delta_{relationship}$ | 关系状态向量的变化幅度 | 0.25 | `relationship_state` 前后差异 |
| $I_{novelty}$ | 内容新颖度（与已有记忆的语义距离） | 0.25 | 与 MemoryStore 中记忆的 max similarity |
| $T_{elapsed}$ | 距上次记忆的时间衰减 | 0.15 | 鼓励长时间对话后的记录 |

**阈值策略**：$S > 0.5$ 则形成记忆，$S > 0.8$ 则标记为"高显著性"（优先在后续检索中返回）。

### 2.4 冷路径：后台 Consolidation（Dreaming）

**触发时机**：借鉴 Claude Code Auto Dream 和 OpenClaw Dreaming 的设计 [^44^][^47^]，采用**多触发器策略**：

| 触发器 | 条件 | 优先级 |
|--------|------|--------|
| 时间触发 | 每日凌晨 3:00（可配置） | 最低 |
| 数量触发 | 情节记忆条目数 > 50 | 中 |
| 会话触发 | 用户会话结束且累计对话轮数 > 10 | 高 |
| 显式触发 | 管理员调用 `/consolidate` 命令 | 最高 |

**三阶段 Consolidation 流程**：

**Light Sleep（去重）**：扫描近期情节记忆，检测语义相似度 > 0.85 的重复条目，仅保留最新的一条。这一阶段的计算成本低，可在每次会话结束时快速执行。

**REM Sleep（模式提取）**：LLM 驱动的模式识别。将去重后的记忆批次输入轻量级模型（如 Gemini Flash 或本地 Qwen），提取：用户偏好变化、关系发展趋势、重复出现的情感主题。输出为结构化的"洞察"条目。

**Deep Sleep（语义化）**：将提取的模式整合进语义记忆层。合并相似的语义条目，更新用户画像，生成关系摘要。这一阶段的输出写入 `db/semantic_memories.json`。

```python
# 伪代码：Consolidation Agent 入口
def run_consolidation_pipeline():
    """三阶段记忆整合流程，在后台异步执行。"""
    # Stage 1: Light Sleep — 去重
    episodes = memory_store.get_recent(days=7)
    deduped = deduplicate_episodes(episodes, threshold=0.85)
    
    # Stage 2: REM Sleep — 模式提取（需要 LLM）
    insights = extract_patterns_with_llm(deduped)
    
    # Stage 3: Deep Sleep — 语义化整合
    for insight in insights:
        merge_or_create_semantic_memory(insight)
    
    # 更新检索索引
    rebuild_embedding_index()
```

**安全架构**：Letta 的 sleep-time agents 和 Kumiho 的 Dream State 都强调了 consolidation 的安全问题 [^73^][^83^]。Lunar 应实现：只读访问（consolidation agent 不能修改情节记忆的原始内容）、审计日志（记录所有合并/删除操作）、断路器（单批次处理时间超过阈值则中断）。

---

## 3. 内驱力与主动发言系统设计

### 3.1 研究综述：从被动响应到主动参与

当前 Lunar 的架构是纯反应式的：所有心理刺激唯一来源于 `perception_node` 对用户消息的提取。这种状态在学术文献中被称为**"心理冻结"**——Agent 在对话间隔中没有任何内部活动。要让 Lunar 从"等待用户输入"转变为"可能主动发起对话"，需要引入**内部驱力生成层**。

在认知架构领域，SOAR 和 ACT-R 都包含动机系统的扩展。SOAR 的问题空间和目标层次结构提供了持久状态管理，近年来的动机扩展引入了情感反馈机制 [^55^]。ACT-R 基于效用学习的求知欲模型（Nagashima et al., 2024）使用公式 $U = \alpha \times R(n) + (1-\alpha) \times U(n-1)$ 来建模动机的动态积累 [^55^]。

在对话 Agent 领域，**Inner Thoughts 框架** [^30^] 是最接近 Lunar 需求的学术工作。该框架提出 AI 应在对话过程中并行生成一条"思想流"，利用长时记忆和工作记忆不断形成新的想法，然后基于**内在动机评分**决定是否参与对话。研究者通过对 24 名参与者的出声思维实验，提取了人类决定发言的 10 个高级启发式规则，并将其形式化为自动评估标准（相关性、信息缺口、情感共鸣等）[^30^]。

在工程实践层面，**ComPeer** [^97^] 是一个专门设计用于主动同伴支持的对话 Agent，它包含三个核心模块：Event Detector（从对话中提取用户事件）、Schedule（规划主动消息的时间和内容）、Reflection（每日反思用户状态以初始化当天的主动计划）。ComPeer 的 Schedule 模块采用随机化机制——当计划事件的"重要性值"大于随机数时才发送，以此平衡主动关怀与打扰 [^97^]。

### 3.2 内驱力生成：D(t) 的数学模型

你的 `INTERNAL_DRIVE_SYSTEM.md` 已经定义了内驱力向量的数学框架：

$$D(t) = D_{baseline}(traits) + D_{accumulated}(t) + D_{spontaneous}(t)$$

这一模型与认知架构中的动机理论高度一致。$D_{baseline}$ 对应 ACT-R 中的静态目标权重；$D_{accumulated}$ 对应 SOAR 中的问题空间张力积累；$D_{spontaneous}$ 则模拟了 Panksepp SEEKING 系统中的随机探索成分 [^55^]。

**与状态引擎的融合**：内驱力生成应作为状态引擎的**Step 0**，在现有 4 步管线之前执行。`D_accumulated` 通过 `W_drive` 矩阵将 18 维状态向量（internal 8 + relationship 3 + surface 7）映射到 7 维刺激向量：

```python
# 伪代码：内驱力生成（Step 0）
def generate_internal_drive(internal_state, relationship_state, surface_state, traits) -> StimulusVector:
    """生成内部驱力刺激向量，作为状态引擎的额外输入。"""
    # D_baseline: 人格决定的基线渴望
    D_base = drive_baseline_mapper.compute(traits)  # LinearMapping
    
    # D_accumulated: 状态驱动的积累驱力
    full_state = np.concatenate([internal_state, relationship_state, surface_state])  # (18,)
    D_acc = full_state @ W_drive  # (18,) @ (18, 7) -> (7,)
    
    # D_spontaneous: 随机游走模拟心智游移
    D_spont = np.random.normal(0, 0.05, size=7)
    
    D = soft_clamp(D_base + D_acc + D_spont, min_val=-0.1, max_val=0.3)
    return StimulusVector(D)
```

**关键心理学映射**（已通过 `WeightMapper.connect()` 注册）：

| 输入状态 | 输出刺激 | 强度 | 心理学依据 |
|---------|---------|------|----------|
| ↑ I_LONELINESS | → ST_CLOSENESS ↑ | 0.20 | 孤独产生靠近渴望（Cacioppo, 2009）[^30^] |
| ↑ I_LONGING | → ST_CLOSENESS ↑ | 0.25 | 思念产生连接冲动 |
| ↑ I_STRESS | → ST_CONFLICT ↑ | 0.15 | 高压力易触发对抗（Berkowitz, 1990）[^30^] |
| ↑ I_INSECURITY | → ST_ABANDONMENT ↑ | 0.20 | 不安→被抛弃恐惧 |
| ↑ R_INTIMACY | → ST_CLOSENESS ↑ | 0.18 | 亲密→更想靠近 |

### 3.3 主动发言：Inner Thoughts 框架的 Lunar 适配

![内驱力系统架构](lunar_drive_system.png)

将 Inner Thoughts 框架 [^30^] 适配到 Lunar 的架构中，形成**"触发 → 检索 → 思想形成 → 评估 → 参与"**五阶段主动发言管线：

**阶段 1：触发（Trigger）**

主动发言的触发条件分为**内部触发**和**外部触发**两类。内部触发源于 Agent 自身的心理状态变化；外部触发源于用户行为的模式检测。

| 触发类型 | 条件 | 对应驱力维度 | 预期频率 |
|---------|------|-------------|---------|
| **沉默超时** | 用户沉默 > 30 分钟 | I_LONELINESS 驱动 | 低 |
| **情感阈值** | I_LONGING / I_INSECURITY > 0.7 | 寻求连接/确认 | 中 |
| **计划触发** | Schedule 中的关怀事件到达时间 | 基于历史模式 | 中 |
| **记忆 surfacing** | 检索到高情感权重记忆 | 情感共鸣驱动 | 高 |
| **随机探索** | D_spontaneous 中某维度 > 0.2 | SEEKING 系统 | 低 |

**沉默超时的实现**：需要引入一个**时间感知模块**，在用户最后一条消息的时间戳基础上，通过定时器或轮询机制检测沉默时长。当沉默超过阈值时，生成一个特殊的"时间流逝"刺激输入状态引擎，驱动 `I_LONELINESS` 上升，进而触发主动发言。

**阶段 2：检索（Retrieval）**

触发后，Agent 从情节记忆中检索与当前驱力状态相关的记忆。这一检索**完全基于情感相似度**（`α = 0.2`，优先情感匹配），而非用户查询的语义相似度。例如，当 `I_LONELINESS` 升高时，检索到的将是那些编码时同样感到孤独的记忆——可能是用户曾经陪伴 Agent 的温馨时刻，Agent 可以主动提起这些记忆来缓解孤独感。

**阶段 3：思想形成（Thought Formation）**

基于检索到的记忆和当前驱力状态，LLM 生成一个"想说的内容"草稿。这一步骤不需要完整的响应生成，只需要一个简短的话题意图（如"提起上次红月的约定"、"询问用户今天过得怎么样"）。

**阶段 4：评估（Evaluation）**

这是主动发言的**关键决策点**。借鉴 Inner Thoughts 的内在动机模型 [^30^]，为每个思想评分：

$$M_{score} = \sum_{i} w_i \cdot criterion_i$$

| 评估维度 | 权重 | 说明 |
|---------|------|------|
| 相关性 (Relevance) | 0.25 | 与当前上下文/用户状态的关联度 |
| 情感共鸣 (Affective Resonance) | 0.25 | 是否能引发情感连接 |
| 信息价值 (Information Value) | 0.20 | 是否提供新信息或新视角 |
| 时机适宜 (Timing) | 0.20 | 当前是否是合适的时机 |
| 发言间隔 (Silence Duration) | 0.10 | 沉默越久，动机越强 [^30^] |

**阶段 5：参与（Participation）**

当 $M_{score} > threshold$（建议初始值 3.5/5.0）时，Agent 主动生成消息。消息的生成需要经过完整的状态引擎管线（包括 Step 0 内驱力输入），确保主动发言也能引起 authentic 的心理状态变化。

**三层主动性控制** [^30^]：

| 层级 | 参数 | 控制内容 | 建议值 |
|------|------|---------|--------|
| **显性主动性** (Overt) | `system1Prob` | 总体发言倾向 | 0.3（中等） |
| **隐性主动性** (Covert) | `imThreshold` | 表达思想的动机阈值 | 3.5 |
| **语调主动性** (Tonal) | `proactiveTone` | 发言风格的主动程度 | True |

### 3.4 Schedule 模块：基于 ComPeer 的主动计划

引入 ComPeer [^97^] 的 Schedule 模块概念，为 Lunar 添加**时间感知的主动关怀能力**：

```python
# 伪代码：Schedule 模块
class ProactiveSchedule:
    """管理 Agent 的主动发言计划。"""
    
    def __init__(self):
        self.event_queue = PriorityQueue()  # 按 planned_time 排序
        self.daily_reflection_done = False
    
    def on_conversation_round(self, user_message, internal_state):
        """每轮对话后：检测事件并更新计划。"""
        # Event Detector: 从用户消息中提取事件
        events = self.extract_events(user_message, internal_state)
        for event in events:
            planned_time = self.infer_timing(event)  # 推断合适的主动提及时间
            self.event_queue.put((planned_time, event))
    
    def on_new_day(self):
        """每日初始化：基于昨日反思生成今日计划。"""
        reflection = self.reflect_on_yesterday()
        self.daily_schedule = self.generate_daily_schedule(reflection)
        self.daily_reflection_done = True
    
    def check_trigger(self, current_time) -> Optional[ProactiveTopic]:
        """检查是否有到期的主动发言事件。"""
        if self.event_queue.empty():
            return None
        planned_time, event = self.event_queue.peek()
        if current_time >= planned_time:
            # 随机化：重要性 > random() 才触发
            if event.importance > random.random():
                return self.event_queue.get()[1]
        return None
```

**事件检测示例**：用户提到"明天要考试"→ Event Detector 提取事件类型="用户压力事件"、推断时间="明天晚上"、重要性=0.8 → Schedule 在明天晚上生成主动关怀消息（"考试结束了吗？感觉怎么样？"）。

---

## 4. 与现有架构的融合方案

### 4.1 LangGraph 节点调整

现有的图结构需要增加三个节点：

```
START → inject_system → perception → [proactive_trigger] → state_engine → state_formatter
                                                                    ↓
                                                          memory_retrieval [NEW]
                                                                    ↓
                                                        llm [MODIFIED - 支持主动生成]
                                                                    ↓
                                                        memory_formation [NEW]
                                                                    ↓
                                                         proactive_schedule [NEW]
                                                                    ↓
                                                                   END
```

**`proactive_trigger` 节点**：在 `perception` 之后插入。检查是否存在未处理的主动发言触发（Schedule 到期、沉默超时、情感阈值突破）。如果有，则跳过 `perception` 的用户刺激提取，直接以"内驱力刺激"作为输入进入 `state_engine`。

**`memory_retrieval` 节点**：在 `state_formatter` 之后、`llm` 之前。基于当前状态向量和用户消息执行混合检索，将记忆注入 LLM prompt。

**`memory_formation` 节点**：在 `llm` 之后。评估本轮互动的显著性，决定是否持久化情节记忆。**注意**：`user_stimuli` 的清理由此节点负责，而非 `state_engine_node` [^3^]。

**`proactive_schedule` 节点**：在 `memory_formation` 之后、END 之前。更新 Schedule 中的事件队列，检测新的事件模式。

### 4.2 状态引擎协同：记忆检索作为内驱力的输入

状态引擎与记忆系统的协同是 Lunar 架构的核心优势。具体协同方式：

**协同 1：状态向量驱动情感检索**

```python
# 在 memory_retrieval_node 中
current_internal = state["internal_state"]  # (8,) 向量
retrieved_by_emotion = memory_store.search_by_internal_state(
    current_internal, top_k=2
)
```

当 `I_LONELINESS` 高时，检索到的记忆自然偏向"曾经有人陪伴的温暖时刻"；当 `I_IRRITATION` 高时，检索到的记忆可能偏向"曾经化解冲突的成功经验"。这种**状态依赖的检索**正是 Mood-Dependent Memory 的计算实现。

**协同 2：记忆 surfacing 驱动内驱力**

当 `memory_retrieval` 检索到一条高情感权重的记忆时（`emotional_weight > 0.8`），这条记忆的浮现本身可以成为一个**额外的内驱力刺激**，驱动 Agent 主动提起这段记忆：

```python
# 伪代码：记忆 surfacing 触发主动发言
if retrieved_memories:
    max_weight = max(m.emotional_weight for m in retrieved_memories)
    if max_weight > 0.8 and random.random() < 0.3:  # 30% 概率主动提起
        drive_stimuli[ST_CLOSENESS] += 0.15  # 增强连接渴望
        # 标记为"主动发言候选"
        state["proactive_topic"] = retrieved_memories[0]
```

**协同 3：Consolidation 反馈到状态引擎**

后台 consolidation 生成的语义记忆（如"用户偏好"、"关系摘要"）不应只存储在数据库中，还应**反馈到状态引擎的 trait 调制**中。例如，consolidation 发现"用户对直接表达情感感到不适"，这一洞察可以微调 `DRIVE_BASELINE_MAPPER` 中的相关参数，使 Agent 的长期行为模式适应用户的沟通风格。

---

## 5. 伪代码实现

### 5.1 情感相似检索的完整实现

```python
# memory.py — 扩展现有 hybrid_search 支持动态权重

def hybrid_search_with_emotion(
    self,
    query_internal: np.ndarray,
    query_text: str,
    current_emotional_weight: float,  # ST_EMOTIONAL_WEIGHT
    top_k: int = 3,
) -> List[MemoryNode]:
    """
    情感感知的混合检索。
    
    当用户情绪激动时，自动降低语义权重 alpha，
    提升情感相似度在检索中的重要性。
    """
    # 动态权重：情感重量越高，情感检索越重要
    alpha = max(0.3, 0.7 - current_emotional_weight * 0.5)
    
    # 语义检索
    semantic_scores = self._embedding_scores(query_text)
    
    # 情感检索（基于 internal_state 向量相似度）
    emotion_scores = self._internal_state_scores(query_internal)
    
    # 加权融合
    combined_scores = alpha * semantic_scores + (1 - alpha) * emotion_scores
    
    # 归一化并按得分排序
    top_indices = np.argsort(combined_scores)[-top_k:][::-1]
    return [self.nodes[i] for i in top_indices]


def _internal_state_scores(self, query_internal: np.ndarray) -> np.ndarray:
    """计算所有记忆与当前内部状态的情感相似度。"""
    scores = []
    for node in self.nodes:
        if "internal_state" in node.state_checkpoint:
            mem_internal = node.state_checkpoint["internal_state"]
            sim = cosine_similarity(query_internal, mem_internal)
            # 额外加权：记忆的 emotional_weight 越高，越容易被检索到
            sim *= (1 + node.emotional_weight * 0.5)
            scores.append(sim)
        else:
            scores.append(0.0)
    return np.array(scores)
```

### 5.2 主动发言触发器

```python
# proactive_initiator.py — 新增模块

class ProactiveInitiator:
    """基于内驱力和 Schedule 的主动发言触发器。"""
    
    def __init__(self, memory_store: MemoryStore, schedule: ProactiveSchedule):
        self.memory_store = memory_store
        self.schedule = schedule
        self.last_user_message_time = datetime.now()
        self.silence_threshold = timedelta(minutes=30)
        self.emotion_threshold = 0.7
    
    def should_initiate(self, state: Dict) -> Optional[str]:
        """
        判断 Agent 是否应该主动发言。
        返回主动话题，或 None（不主动发言）。
        """
        internal = state["internal_state"]
        
        # 触发 1: 沉默超时
        silence = datetime.now() - self.last_user_message_time
        if silence > self.silence_threshold:
            if internal[I_LONELINESS] > 0.5:
                return self._generate_lonely_topic(state)
        
        # 触发 2: 情感阈值
        if internal[I_LONGING] > self.emotion_threshold:
            return self._generate_longing_topic(state)
        
        if internal[I_INSECURITY] > self.emotion_threshold:
            return self._generate_reassurance_topic(state)
        
        # 触发 3: Schedule 到期事件
        scheduled = self.schedule.check_trigger(datetime.now())
        if scheduled:
            return scheduled.topic
        
        # 触发 4: 记忆 surfacing
        if random.random() < 0.1:  # 10% 概率检查
            retrieved = self.memory_store.search_by_internal_state(
                internal, top_k=1
            )
            if retrieved and retrieved[0].emotional_weight > 0.8:
                return self._generate_memory_topic(retrieved[0], state)
        
        return None
    
    def _generate_lonely_topic(self, state) -> str:
        """孤独时主动发起的话题。"""
        # 检索温暖的过往记忆
        retrieved = self.memory_store.search_by_internal_state(
            state["internal_state"], top_k=2
        )
        if retrieved:
            return f"[主动] 突然想到{retrieved[0].title}..."
        return "[主动] 你不在的时候，这里好安静..."
```

### 5.3 Memory Formation 节点

```python
# nodes.py — memory_formation_node 实现

def memory_formation_node(state: GraphState) -> GraphState:
    """
    对话结束时的记忆形成节点。
    评估本轮互动显著性，决定是否持久化情节记忆。
    """
    # 提取本轮对话的关键信息
    user_msg = state["messages"][-2]["content"]  # 用户最后一条
    ai_msg = state["messages"][-1]["content"]    # Agent 最后一条
    
    # 获取状态变化
    internal_before = state.get("internal_state_before", state["internal_state"])
    internal_after = state["internal_state"]
    rel_before = state.get("relationship_state_before", state["relationship_state"])
    rel_after = state["relationship_state"]
    
    # 显著性评分
    sig = compute_significance(
        emotional_weight=state.get("ST_EMOTIONAL_WEIGHT", 0.5),
        relationship_delta=np.linalg.norm(rel_after - rel_before),
        content_novelty=compute_novelty(user_msg + ai_msg, memory_store),
        time_since_last=time_since_last_memory(memory_store),
    )
    
    if sig > 0.5:  # 阈值
        node = MemoryNode.from_state_vectors(
            title=user_msg[:50],
            content=f"User: {user_msg}\nAI: {ai_msg}",
            internal_state=internal_after,
            relationship_state=rel_after,
            surface_state=state["surface_state"],
            embedding=embedding_model.encode(user_msg + ai_msg),
        )
        node.significance = sig
        node.emotional_weight = state.get("ST_EMOTIONAL_WEIGHT", 0.5)
        memory_store.add(node)
        memory_store.save()
    
    # 清理 user_stimuli（从 state_engine_node 移至此）
    state["user_stimuli"] = None
    
    return state


def compute_significance(emotional_weight, relationship_delta, 
                         content_novelty, time_since_last) -> float:
    """多因子显著性评分。"""
    w = [0.35, 0.25, 0.25, 0.15]
    scores = [
        min(emotional_weight, 1.0),
        min(relationship_delta * 2, 1.0),  # 归一化
        min(content_novelty, 1.0),
        min(time_since_last / 3600, 1.0),  # 小时级归一化
    ]
    return sum(wi * si for wi, si in zip(w, scores))
```

### 5.4 Consolidation Agent（Dreaming）

```python
# consolidation_agent.py — 后台运行

class ConsolidationAgent:
    """三阶段记忆整合 Agent（Dreaming）。"""
    
    async def run(self):
        """主入口，在后台异步执行。"""
        episodes = self.memory_store.get_recent(days=7)
        
        # Stage 1: Light Sleep — 去重
        deduped = self._deduplicate(episodes)
        
        # Stage 2: REM Sleep — 模式提取（需要 LLM）
        insights = await self._extract_patterns(deduped)
        
        # Stage 3: Deep Sleep — 语义化整合
        for insight in insights:
            self._merge_into_semantic_memory(insight)
        
        # 重建索引
        self._rebuild_index()
    
    async def _extract_patterns(self, episodes: List[MemoryNode]) -> List[Insight]:
        """使用轻量级 LLM 提取模式。"""
        prompt = f"""
        分析以下 {len(episodes)} 段对话记忆，提取关键模式：
        - 用户偏好变化
        - 关系发展趋势  
        - 重复出现的情感主题
        - 值得长期记住的事实
        
        记忆摘要：
        {summarize_episodes(episodes)}
        
        输出 JSON 格式的洞察列表。
        """
        response = await llm_client.complete(prompt, model="gemini-flash")
        return parse_insights(response)
```

---

## 6. 实现路线图

### Phase 1: 记忆注入与检索（2-3 周）

| 任务 | 文件 | 优先级 |
|------|------|--------|
| 完成 `memory_retrieval_node` stub | `nodes.py` | P0 |
| 实现动态权重的 `hybrid_search_with_emotion` | `memory.py` | P0 |
| 完成 `memory_formation_node`（含显著性评估） | `nodes.py` | P0 |
| 调整 prompt 模板支持记忆注入 | `prompts/` | P0 |
| 解决 `user_stimuli` 清理时机问题 | `nodes.py`, `state_engine/` | P0 |

### Phase 2: 主动发言 MVP（2 周）

| 任务 | 文件 | 优先级 |
|------|------|--------|
| 实现 `generate_internal_drive`（Step 0） | `state_engine/_drive.py` [NEW] | P0 |
| 实现 `ProactiveInitiator`（基础触发器） | `proactive_initiator.py` [NEW] | P0 |
| 实现沉默超时检测 | `main.py` / 定时器 | P1 |
| 添加 `proactive_trigger` 图节点 | `nodes.py` | P0 |
| 调整 LLM prompt 支持主动发言语调 | `prompts/` | P1 |

### Phase 3: Consolidation 与 Schedule（2-3 周）

| 任务 | 文件 | 优先级 |
|------|------|--------|
| 实现 `ConsolidationAgent`（Light Sleep 去重） | `consolidation_agent.py` [NEW] | P1 |
| 实现 REM Sleep 模式提取 | `consolidation_agent.py` | P1 |
| 实现语义记忆层 (`semantic_memories.json`) | `memory.py` | P1 |
| 实现 `ProactiveSchedule`（Event Detector + Schedule） | `proactive_schedule.py` [NEW] | P1 |
| 配置后台任务调度（APScheduler / cron） | `main.py` | P2 |

### Phase 4: 优化与评测（持续）

| 任务 | 说明 |
|------|------|
| 记忆检索质量评测 | 使用 ENPMR-Bench 的情感检索指标 |
| 主动发言频率调优 | 根据用户反馈调整 `imThreshold` 和 `system1Prob` |
| Consolidation 效果评估 | 对比 consolidation 前后的检索质量 |
| 延迟优化 | 必要时引入缓存层（Redis / 内存缓存） |

---

## 7. 关键设计决策总结

### 7.1 记忆系统决策

| 问题 | 决策 | 理由 |
|------|------|------|
| RAG vs 结构化记忆？ | **结构化情节记忆为主，语义检索为辅** | Lunar 的 `state_checkpoint` 已天然支持结构化；情感检索需要状态向量 |
| 何时检索记忆？ | **每轮对话热路径检索** | 延迟 < 500ms，零 LLM 调用 |
| 何时形成记忆？ | **对话结束时温路径评估** | 基于显著性评分，避免垃圾记忆积累 |
| 何时 consolidation？ | **会话结束 + 每日定时 + 数量阈值** | 多触发器确保及时整合 |
| 情感相似如何实现？ | **internal_state 向量余弦相似度** | 直接利用现有状态引擎输出 |

### 7.2 主动发言决策

| 问题 | 决策 | 理由 |
|------|------|------|
| 主动发言与什么相关？ | **Traits（基线驱力）+ Relationship（积累驱力）+ Internal（状态驱动）** | 与你的直觉一致，且有完整数学模型支撑 |
| 触发机制？ | **5 类触发器：沉默超时、情感阈值、Schedule、记忆 surfacing、随机游走** | 覆盖 Inner Thoughts 框架的核心触发类型 |
| 频率控制？ | **三层主动性控制（显性/隐性/语调）+ Schedule 随机化** | 避免过度打扰用户 |
| 内容范围？ | **不限于关系维护，包括：提起过往记忆、关心用户状态、分享角色感受、发起新话题** | 与 ComPeer 和 Generative Agents 的设计一致 |

### 7.3 与状态引擎的融合

| 融合点 | 方式 | 价值 |
|--------|------|------|
| 状态向量 → 记忆检索 | `search_by_internal_state()` 使用 `internal_state` 做 key | **情感感知的被动回忆** |
| 记忆 surfacing → 内驱力 | 高情感权重记忆增强 `ST_CLOSENESS` | **记忆驱动的主动发言** |
| Consolidation → Trait 调制 | 语义洞察反馈到 `DRIVE_BASELINE_MAPPER` | **长期行为适应** |
| 内驱力 → 状态引擎 Step 0 | `D(t)` 作为额外刺激输入防御剖面 | **从被动到主动的根本转变** |

---

## 8. 风险评估与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 主动发言过于频繁打扰用户 | 高 | 三层主动性控制 + Schedule 随机化；提供用户开关 |
| 情感检索返回不相关记忆 | 中 | 动态权重调整；语义相似度作为保底过滤 |
| Consolidation LLM 成本过高 | 中 | 使用轻量级模型（Gemini Flash）；批量处理 |
| 记忆注入导致 prompt 过长 | 中 | 限制注入记忆数量（top_k ≤ 3）；摘要化注入 |
| 内驱力导致状态不稳定 | 低 | `soft_clamp` 约束驱力范围；D_spontaneous 方差控制 |
| 后台 consolidation 失败 | 低 | 断路器机制；失败不影响主对话流程 |

---

**本设计文档整合了认知心理学（Bower 的 Mood-Dependent Memory、Panksepp 的 SEEKING 系统）、认知架构（SOAR/ACT-R 的动机扩展）、对话 Agent 研究（Inner Thoughts 框架、ComPeer、Generative Agents）和工业实践（Mem0、Letta、Claude Dreaming）的多维度研究成果，为 Lunar-Agent 的记忆系统和内驱力系统提供了一套理论上扎实、工程上可落地的综合方案。**
