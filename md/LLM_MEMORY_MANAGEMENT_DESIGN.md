# Lunar LLM-Wiki 记忆管理方案书

> 基于开源社区与学术前沿的深度研究，为 Lunar 角色扮演引擎设计 LLM 驱动的长期记忆管理系统。
>
> 研究日期：2026-06-17 | 状态：方案阶段

---

## 目录

1. [研究摘要](#1-研究摘要)
2. [领域全景扫描](#2-领域全景扫描)
3. [系统深度分析](#3-系统深度分析)
4. [对比矩阵](#4-对比矩阵)
5. [Lunar 差距分析](#5-lunar-差距分析)
6. [推荐架构：Lunar Memory OS](#6-推荐架构lunar-memory-os)
7. [实施路线图](#7-实施路线图)
8. [致谢与参考](#8-致谢与参考)

---

## 1. 研究摘要

### 1.1 核心发现

对 20 个来源、78 个主张进行提取和交叉验证后，确认 6 个高置信度发现：

1. **内存层级化是共识**：Letta（OS 启发式 Core/Recall/Archival 三层）、LangMem（Semantic/Episodic/Procedural 三层）、Zep（双时态知识图谱）——成熟系统不约而同采用分层架构
2. **LLM 参与度的光谱分布**：从 Agnaistic 的零 LLM（纯关键词匹配）到 Letta 的全 LLM（自主调用 memory tool），中间存在大量混合方案
3. **热/冷路径分离是延迟优化关键**：LangMem 的 `ReflectionExecutor` 将记忆整合推迟到后台，保证会话路径零 LLM 开销；Zep 的检索路径无需 LLM 调用
4. **"管理"阶段是最大空白**：Du (2025) 的 POMDP 框架指出——写入和读取已充分解决，但组织/合并/压缩/遗忘/冲突消解仍是"粗放启发式"
5. **Generative Agents 的 reflection 机制奠定基础**：重要性评分 + 递归 reflection → 高层次洞察，但为离线仿真设计，不适合实时对话
6. **MemR3 是 LangGraph 原生的参考实现**：Router → Retrieve → Reflect → Answer 的图节点结构可直接适配 Lunar 的 LangGraph 管线

### 1.2 对 Lunar 的关键启示

Lunar 的独特性——**实时角色扮演 + 第一人称记忆 + 心理状态引擎 + 本地 LLM**——意味着没有任何现成方案能直接套用。但三个设计原则已经明确：

- **检索路径零 LLM**：embedding 搜索 (<0.5s) 足以胜任，LLM 不应出现在热路径
- **记忆整合走后台**：对话间隙运行 consolidation agent，不阻塞用户交互
- **层级从简开始**：种子记忆 → 情节记忆 → 语义记忆，渐进式构建

---

## 2. 领域全景扫描

### 2.1 系统光谱：从全自动到全手动

```
LLM 参与度

100% ─┤  Letta/MemGPT (LLM 自主管理全部记忆操作)
      │
 75% ─┤  Generative Agents (LLM 写入+检索+reflection, 非实时)
      │  LangMem (LLM 负责提取+整合, 检索走 embedding)
 50% ─┤
      │  Zep/Graphiti (LLM 仅冲突消解, 检索纯确定)
 25% ─┤  CrewAI (LLM 通过 tool 读写共享记忆池)
      │  Mem0 (托管 API, 内部混合)
  0% ─┤  Agnaistic (零 LLM, 关键词匹配)
      │  SillyTavern (用户手工编辑 Lorebook)
```

**核心趋势**：2025 年以后的项目普遍从"全 LLM"回退到"热路径去 LLM 化"——即使 Letta 的 sleep-time compute 提案（被 refuted）也反映了对延迟的焦虑。

### 2.2 已排除的系统

以下系统相关但声明在交叉验证中被 refuted（2/3 反对票），故未作为可靠参考：

| 系统 | 被 refuted 的主张 | 原因 |
|------|------------------|------|
| Mem0 | benchmark 数据 | 来源为自发布 blog，无独立验证 |
| SillyTavern Lorebook | 架构细节 | DeepWiki 标注 unreliable |
| RisuAI HypaMemory v3 | 架构细节 | 来源为 DeepWiki secondary |
| CrewAI Cognitive Memory | 架构声明 | 官方 blog 被标记 unreliable |
| Cognee | 所有主张 | blog 比较文章不满足 primary source 标准 |

---

## 3. 系统深度分析

### 3.1 Letta/MemGPT — OS 启发式三层记忆

**来源**：[MemGPT paper (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560), [Letta docs](https://docs.letta.com/concepts/memgpt/), [Memory Blocks blog](https://www.letta.com/blog/memory-blocks)
**置信度**：HIGH (3-0 验证通过)

**架构**：

```
┌─────────────────────────────────────────┐
│             Main Context (RAM)           │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Persona │ │  Human   │ │  System  │  │  ← Core Memory blocks (每轮注入)
│  └─────────┘ └──────────┘ └──────────┘  │
│  + 最近 N 轮对话 (Recall Memory 检索)     │
│  + Archival Memory 检索结果              │
└─────────────────────────────────────────┘
                    ↕ LLM 自主 tool call 分页
┌─────────────────────────────────────────┐
│          External Context (Disk)         │
│  ┌──────────────┐ ┌───────────────────┐ │
│  │ Recall Memory│ │ Archival Memory   │ │
│  │ (全量消息历史) │ │ (向量DB, 无限容量)  │ │
│  │ 支持时间/角色  │ │ pgvector/Chroma   │ │
│  │ /语义过滤     │ │                   │ │
│  └──────────────┘ └───────────────────┘ │
└─────────────────────────────────────────┘
```

**LLM 参与方式**：LLM 在推理时通过 function calling 自主决定何时 `archival_memory_insert`、`archival_memory_search`、`core_memory_replace`。这意味着 LLM 同时扮演"CPU"和"内存控制器"。

**延迟特征**：每次 tool call 增加 1-3 次 LLM 往返。在本地 Ollama 环境下（qwen2.5:7b ≈ 3s/次），单轮记忆操作可能导致 **+3-9s 延迟**。

**Lunar 适用性**：⭐⭐（架构思想可借鉴，实时性不可接受）

**对 Lunar 的启示**：
- Core Memory 概念 → 可映射为 Lunar 已有的角色基座记忆（种子记忆 + 关系状态摘要）
- Archival Memory → 可映射为情节记忆库（向量检索）
- **但 LLM 自主 tool call 分页在延迟预算 500ms 内完全不可行**

### 3.2 LangMem — 三维记忆架构 + 热/冷路径分离

**来源**：[LangMem SDK launch](https://www.langchain.com/blog/langmem-sdk-launch), [GitHub](https://github.com/langchain-ai/langmem), [Conceptual Guide](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)
**置信度**：HIGH (3-0 验证通过)

**三维架构**：

```
Dimension 1: 时间路径         Dimension 2: 记忆类型       Dimension 3: API 层级
─────────────                ────────────               ────────────
┌─ Hot Path (同步) ─┐       Semantic (事实/偏好)        Core API (无状态)
│  写入 + 检索       │       Episodic (对话摘要)          create_memory_manager
│  保证实时响应       │       Procedural (行为规则)        → 返回 List[ExtractedMemory]
└───────────────────┘                                   → 不依赖任何存储
                                          ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
┌─ Cold Path (异步) ─┐                               Integration Layer (有状态)
│  ReflectionExecutor │                               create_memory_store_manager
│  debounce 防抖      │                               → 集成 LangGraph BaseStore
│  p95 ≈ 60s          │                               → 自动持久化 + 向量检索
└─────────────────────┘                               → 版本化历史
```

**关键创新 — ReflectionExecutor**：

```python
# debounce 机制：新消息到达时取消并重新调度
executor.submit(to_process, after_seconds=delay)
```

这保证了：用户快速连续发消息时，内存管理器不会在每次消息后立即触发，而是等对话"安静下来"再运行。

**延迟特征**：
- Hot path：无额外 LLM 开销（仅 embedding 检索）
- Cold path：p95 = 60s（LOCOMO benchmark），但离线运行不接受延迟约束

**Lunar 适用性**：⭐⭐⭐⭐（热/冷分离 + LangGraph 集成，高度匹配）

**对 Lunar 的启示**：
- **热/冷路径分离是 Lunar 必须采用的核心模式**
- LangMem 的 `create_memory_store_manager` 直接依赖 LangGraph 的 `BaseStore`，与 Lunar 的技术栈同源
- 注意：LangMem 目前是 pre-1.0（0.0.x），不应作为生产依赖，但可借鉴其接口设计

### 3.3 Zep/Graphiti — 双时态知识图谱

**来源**：[Zep paper (arXiv:2501.13956v1)](https://export.arxiv.org/abs/2501.13956v1), [GitHub](https://github.com/getzep/graphiti)
**置信度**：HIGH (3-0 验证通过)

**核心机制**：

```
每条事实边携带四个时间戳：
  t_valid, t_invalid    — 现实世界有效期（事实本身何时为真）
  t_created, t_expired  — 系统事务时间（何时记录/何时被标记过期）

冲突消解流程：
  LLM 检测新事实与已有边的矛盾
  → 旧边设 invalid_at + expired_at（不删除！）
  → 新建边覆盖新事实
  → 保留完整历史溯源能力
```

**关键判断**：检索路径无需 LLM 调用（声称的 ~300ms P95 被 1-2 refuted），但冲突消解依赖 LLM 做语义判断。

**Lunar 适用性**：⭐⭐⭐（冲突消解机制可借鉴，但全量知识图谱对单角色 over-engineered）

**对 Lunar 的启示**：
- **软删除而非硬删除**的记忆过期策略值得采纳
- 四时间戳模型可以简化为"创建时间 + 最后访问时间 + 有效性标记"
- 但 Lunar 只有 1 个角色 + 1 个用户的二元关系，不需要完整的实体-边图结构

### 3.4 Agnaistic — 角色扮演框架的极简记忆

**来源**：[Agnaistic Memory Books docs](https://agnai.guide/docs/memory/memory-books), [Chat Embeddings docs](https://agnai.guide/docs/memory/embeddings), [GitHub Issue #801](https://github.com/agnaistic/agnai/issues/801)
**置信度**：HIGH (3-0 验证通过)

**架构**：

```
Memory Books (关键词匹配)        Chat Embeddings (语义搜索)
─────────────────────          ──────────────────────
• 用户手工创建 entry            • 浏览器端 in-memory DB
• 关键词 + wildcard (*, ?)      • 仅索引 viewport 内消息
• 优先级排序 + 预算截断          • 页面刷新后全部丢失
• 零 LLM 参与                   • embedding 模型运行在浏览器

输出占位符: {{memory}}           输出占位符: {{chat_embed}}
```

**LLM 参与**：**完全为零**。GitHub Issue #801（自动摘要功能请求）至今 open，证明社区渴望 LLM 介入但尚未实现。

**Lunar 适用性**：⭐⭐（证明关键词匹配对角色扮演有效，但 Lunar 需要远超此能力）

**对 Lunar 的启示**：
- 优先级 + 预算截断的注入模式是正确的（context window 有限）
- 用户对"自动记忆管理"有明确需求（Issue #801 的高关注度）
- **纯客户端/纯确定性的道路不适合有独立后端和 LLM 的 Lunar**

### 3.5 Generative Agents — Reflection 记忆整合

**来源**：[Park et al. (arXiv:2304.03442)](https://arxiv.org/abs/2304.03442), ACM UIST 2023
**置信度**：HIGH (3-0 验证通过)

**记忆流架构**：

```
Write Path:
  事件 → LLM 评估重要性 (1-10) → 存入 memory stream

Read Path (检索):
  score = α_recency × recency(t) + α_importance × importance + α_relevance × cos_sim(q, m)

Reflection (整合):
  触发条件: Σ importance > threshold (基于最近 100 条记录)
  流程:
    1. LLM 生成 3 个高层次问题
    2. 每个问题 → embedding 检索相关记忆
    3. 每个问题 → LLM 合成 5 个洞察 (共 15 条)
    4. 洞察存入 memory stream (可递归 reflection)
```

**延迟**：Reflection 需要多次 LLM 调用（3 次问题生成 + 3×5 次洞察合成），设计用于离线仿真非实时对话。研究级代码，非生产优化。

**Lunar 适用性**：⭐⭐⭐（Reflection 概念是核心参考，但实现需要大幅简化）

**对 Lunar 的启示**：
- Reflection 的三阶段（提问 → 检索 → 合成）可压缩为 Lunar 的"单轮后台整合"
- 重要性评分机制可以改为由心理状态引擎的 arousal 指标自动决定
- Lunar 不需要 15 条洞察——1-3 条经过整合的高价值记忆足矣

### 3.6 MemR3 — LangGraph 原生的记忆增强推理

**来源**：[MemR3 paper (arXiv:2512.20237)](https://arxiv.org/abs/2512.20237), Dec 2025, MBZUAI & LangGraph Inc.
**置信度**：HIGH (3-0 验证通过)

**图结构**：

```
              ┌──────────┐
     START → │  Router   │ ← Evidence-Gap Tracker 控制循环
              └────┬─────┘
          ┌────────┼────────┐
          ▼        ▼        ▼
    ┌─────────┐ ┌───────┐ ┌────────┐
    │ Retrieve│ │Reflect│ │ Answer │
    │ 精炼查询 │ │逻辑推理│ │ 最终输出│
    │ mask去重 │ │无外部调│ │ gap为空 │
    └─────────┘ │  用    │ └────────┘
                └───────┘
    状态: (query, snippets, evidence, gaps, iter_idx)
```

**关键设计**：记忆检索 (Retrieve) 和推理回答 (Reflect, Answer) 是**独立的图节点**，通过 Router 控制循环。这天然适配 LangGraph 的 `StateGraph`。

**Lunar 适用性**：⭐⭐⭐⭐（同为 LangGraph 原生，节点分离模式可直接复用）

**对 Lunar 的启示**：
- Lunar 的 LangGraph 管线可以自然扩展一个 `memory_retrieve` 节点
- Router 控制循环（"还需要更多记忆吗？"）可以改造为 Lunar 的记忆补充判断
- Evidence-Gap Tracker 可映射为"心理状态缺乏解释 → 检索更多记忆"

---

## 4. 对比矩阵

| 维度 | Letta/MemGPT | LangMem | Zep/Graphiti | Agnaistic | Gen Agents | MemR3 | **Lunar 需求** |
|------|-------------|---------|-------------|-----------|------------|-------|---------------|
| **LLM 参与检索** | ✅ tool call | ❌ 纯 embedding | ❌ 确定融合 | ❌ 关键词 | ❌ embedding | ✅ Router控制 | **❌ 不可** |
| **LLM 参与写入** | ✅ tool call | ✅ 后台提取 | ✅ 冲突消解 | ❌ 手工 | ✅ 重要性评分 | ❌ | **✅ 后台** |
| **LLM 参与整合** | ✅ sleep-time(提) | ✅ ReflectionEx | ✅ 边消解 | ❌ | ✅ Reflection | ✅ 迭代循环 | **✅ 后台** |
| **记忆分层** | 3 层 (C/R/A) | 3 类 (S/E/P) | 图 + 时序 | 2 层 (书+嵌入) | 1 流 | 单层检索 | **2-3 层** |
| **延迟 (检索)** | +3-9s (LLM) | <0.1s | ~? | <0.01s | <0.1s | +3s (LLM) | **<0.5s** |
| **延迟 (整合)** | N/A (内联) | p95 60s (离线) | N/A | N/A | 分钟级 | N/A | **无实时约束** |
| **生产就绪** | ✅ 13k⭐ | ⚠️ pre-1.0 | ✅ 20k⭐ | ✅ 740⭐ | ❌ 研究级 | ❌ 研究级 | — |
| **与 LangGraph 集成** | ❌ | ✅ 原生 | ❌ | ❌ | ❌ | ✅ 原生 | **✅ 必须** |
| **开源协议** | AGPL | MIT | Apache 2.0 | AGPL-3.0 | CC BY | — | — |
| **为角色扮演设计** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | **✅** |

---

## 5. Lunar 差距分析

### 5.1 当前状态

Lunar 已具备：

- ✅ 功能完善的 `MemoryNode` (Pydantic) + `MemoryStore` (JSON + 原子写入)
- ✅ 三种检索方式（向量检索 / Embedding 检索 / 混合检索）
- ✅ bge-m3 embedding 模型（1024d, 0.14s/次, 73.7% 命中率）
- ✅ 16 条手工种子记忆（第一人称角色视角）
- ✅ LangGraph 管线（5 个节点 + 2 个 stub）
- ✅ 本地 qwen2.5:7b 可用于记忆管理

### 5.2 关键缺失

对照研究报告，Lunar 缺少四个核心能力：

| 缺失能力 | 研究依据 | 严重程度 |
|---------|---------|---------|
| **记忆生命周期管理** | Du (2025): "manage 阶段是最大空白" | P0 |
| **热/冷路径分离** | LangMem 已验证的模式 | P0 |
| **记忆冲突/合并** | Zep 的双时态消解 | P1 |
| **重要性驱动的检索排序** | Gen Agents: importance × recency × relevance | P1 |

### 5.3 当前 `memory_inject_node` 和 `memory_summery_node` Stub 的分析

两个 stub 节点恰好对应了 LangMem 的两个路径：

- `memory_inject_node` → **Hot Path**：在生成回复前注入相关记忆
- `memory_summery_node` → **Cold Path**：对话后总结并存储新记忆

现有架构的前瞻性是对的，问题在于实现。

---

## 6. 推荐架构：Lunar Memory OS

### 6.1 设计原则

1. **Hot path 零 LLM**：检索走 bge-m3 embedding (<0.3s)，满足 500ms 延迟预算
2. **Cold path 全 LLM**：记忆提取/合并/遗忘在后台异步运行
3. **渐进复杂度**：Phase 1 只做检索注入 → Phase 2 加记忆提取 → Phase 3 加整合/遗忘
4. **与心理状态引擎耦合**：记忆的重要性由 arousal/valence 自动评分，不依赖额外的 LLM 调用
5. **第一人称视角保持**：记忆始终以角色视角存储，post-retrieval 的 LLM 理解"这是我（角色）的记忆"

### 6.2 三层记忆层级

```
┌──────────────────────────────────────────────────────┐
│                 Layer 1: Core Memory                  │
│              (每轮无条件注入 context window)             │
│                                                        │
│  ┌──────────────────┐  ┌───────────────────────────┐  │
│  │ 角色基座记忆 (16条) │  │ 关系摘要 (从 RelationshipState │  │
│  │ 种子记忆 + 演化记忆 │  │ 动态生成的一小段文字)         │  │
│  └──────────────────┘  └───────────────────────────┘  │
│                    占用 token 预算: ~800 tokens         │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│               Layer 2: Episodic Memory                │
│          (Hot Path: embedding 检索 → 注入相关记忆)       │
│                                                        │
│  • 存储: 对话中提取的情节片段 (第一人称)                  │
│  • 索引: bge-m3 embedding (1024d)                     │
│  • 检索: 对话上下文 → embedding → top-3                │
│  • 注入: 按 relevance 排序, 控制预算 (≤3条, ≤500 tokens) │
│                                                        │
│                    延迟: <0.3s                          │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│               Layer 3: Semantic Memory                │
│         (Cold Path: LLM 后台整合 → 长期事实)            │
│                                                        │
│  • 触发: 用户空闲 N 秒后 / 每 K 轮对话后                 │
│  • 流程:                                                │
│    1. LLM 扫描最近的 Episodic 记忆                      │
│    2. 判断: 是否有值得提炼的事实/模式?                    │
│    3. 合并: 与已有 Semantic 记忆去重/更新                 │
│    4. 遗忘: 标记低价值 Episodic 记忆为 archived          │
│                                                        │
│                    运行时机: 后台, 不阻塞回复               │
└──────────────────────────────────────────────────────┘
```

### 6.3 LangGraph 管线扩展

```
当前管线:
  START → inject_system → perception → state_engine → state_formatter → llm → END
            (stub) ↓                                  (stub) ↓
         memory_inject                            memory_summery

扩展后管线:
  START → inject_system → perception → state_engine → state_formatter
                                                            │
                                              ┌─────────────┤
                                              ▼             ▼
                                     [Hot Path]     [Cold Path 触发]
                                     memory_retrieve  schedule_consolidation
                                              │             │
                                              ▼             │ (async)
                                     + memory_context      │
                                              │             ▼
                                              ▼       consolidation_agent
                                             llm         │
                                              │     ┌─────┴─────┐
                                              ▼     ▼           ▼
                                             END  extract    merge/forget
                                                   └─────┬─────┘
                                                         ▼
                                                    update_store
```

**新增节点说明**：

| 节点 | 路径 | 功能 | LLM 调用 | 延迟预算 |
|------|------|------|---------|---------|
| `memory_retrieve` | Hot | embedding 检索 + 注入 | ❌ | <300ms |
| `schedule_consolidation` | Hot→Cold | 判断是否需要触发整合 | ❌ (轮次计数) | <1ms |
| `consolidation_agent` | Cold | 提取/合并/遗忘 | ✅ (qwen2.5:7b) | 无约束 |

### 6.4 记忆生命周期状态机

```
                 ┌──────────┐
   对话事件 ───→ │ ephemeral │ (当前轮对话, 不持久化)
                 └────┬─────┘
                      │ consolidation_agent 判断值得记住
                      ▼
                 ┌──────────┐
                 │ episodic  │ (存入 Layer 2, 有 embedding)
                 └────┬─────┘
                      │ 多次检索命中 + importance 累积 → 升级
                      ▼
                 ┌──────────┐
                 │ semantic  │ (存入 Layer 3, 可注入 Core)
                 └────┬─────┘
                      │ 长期未被检索 + importance 衰减 → 归档
                      ▼
                 ┌──────────┐
                 │ archived  │ (保留但不注入, 可恢复)
                 └──────────┘
```

**重要性评分**（由心理状态引擎驱动，无需额外 LLM 调用）：

```python
importance = (
    arousal_weight  × abs(internal_state[I_STRESS] - prev_internal[I_STRESS])
  + valence_weight  × abs(internal_state[I_LONGING] - prev_internal[I_LONGING])
  + relation_weight × abs(relationship_state[R_ROMANTIC_TENSION] - prev_rel[R_ROMANTIC_TENSION])
)
```

与 Generative Agents 的 LLM 评分不同，Lunar 有心理状态这个天然信号源，可以实现**零额外 LLM 开销的重要性自动评分**。

### 6.5 检索函数设计

借鉴 Generative Agents 的三因子加权 + Zep 的多路融合：

```python
def retrieve_memories(query_text, current_internal, top_k=3):
    # ① Embedding 语义检索
    semantic_results = store.search_by_embedding(query_text, top_k=top_k*2)

    # ② 关键词/实体匹配 (boost 因子, 不需 LLM)
    for node, score in semantic_results:
        keyword_boost = sum(1 for kw in extract_keywords(query_text) if kw in node.content)
        score += 0.1 * keyword_boost

    # ③ 情感状态相似度 (Lunar 独有优势)
    for node, score in semantic_results:
        state_sim = cos_similarity(current_internal, node.state_checkpoint.get("internal_state"))
        score = 0.7 * score + 0.3 * state_sim  # 加权融合

    # ④ 按最终分数排序 + 预算截断
    return sort_and_truncate(semantic_results, top_k, max_tokens=500)
```

### 6.6 Consolidation Agent 的 Prompt 设计

```markdown
你是 Lunar 的记忆管理代理。你的职责是维护角色的长期记忆库。

## 输入
- 最近 N 条 episodic 记忆 (含标题、内容、时间戳、重要性)
- 当前 semantic 记忆列表

## 任务
1. **合并**: 新记忆与已有 semantic 记忆高度重叠？合并为一条，保留更完整的叙述
2. **升级**: episodic 记忆中是否有反复出现的模式？提炼为 semantic 记忆
3. **遗忘**: 哪些 episodic 记忆重要性衰减到阈值以下？标记为 archived
4. **冲突**: 新记忆是否与已有记忆矛盾？保留两者，标注时间线

## 约束
- 所有记忆保持角色第一人称 ("我") 视角
- 每次最多输出 3 条变更 (避免过度操作)
- 不确定时，宁可保留，不要删除
```

---

## 7. 实施路线图

### Phase 1: Memory Retriever（目标：命中率 85%+，延迟 <500ms）

| 任务 | 预估工时 | 依赖 |
|------|---------|------|
| 1.1 实现 `memory_retrieve` 节点 | 2h | bge-m3 已就绪 |
| 1.2 实现关键词 boost + 状态相似度融合检索 | 1h | — |
| 1.3 实现 memory context 注入 LLM prompt | 1h | — |
| 1.4 连接 LangGraph 管线（替换 `memory_inject_node` stub） | 1h | 1.1-1.3 |
| 1.5 命中率 + 延迟评测 | 1h | 1.4 |

**总工时**：~6h | **风险**：低（纯 embedding 检索，已在 test_rag.py 验证）

### Phase 2: Memory Extraction（目标：对话自动产生 episodic 记忆）

| 任务 | 预估工时 | 依赖 |
|------|---------|------|
| 2.1 设计 Memory Extraction prompt（第一人称 + Pydantic schema） | 1h | — |
| 2.2 实现 `consolidation_agent` 基础版（仅 extract） | 2h | 2.1 |
| 2.3 实现重要性自动评分（心理状态驱动） | 1h | state_engine |
| 2.4 实现 `schedule_consolidation` 触发逻辑 | 1h | — |
| 2.5 端到端测试：对话 → 自动记忆 → 检索 → 注入 | 2h | 2.2-2.4 |

**总工时**：~7h | **风险**：中（extraction 质量依赖 qwen2.5:7b 的指令遵循能力）

### Phase 3: Memory Lifecycle（目标：完整的记忆演化闭环）

| 任务 | 预估工时 | 依赖 |
|------|---------|------|
| 3.1 实现记忆合并/去重（LLM 判断 + cos_sim 阈值） | 2h | Phase 2 |
| 3.2 实现记忆遗忘（importance 衰减 + LLM 确认） | 1.5h | 3.1 |
| 3.3 实现冲突标注（软删除 + 时间线，借鉴 Zep 模式） | 1.5h | 3.1 |
| 3.4 实现 semantic memory 的 Core Memory 注入 | 1h | 3.1-3.3 |
| 3.5 完整评测：10 轮连续对话的记忆一致性 | 2h | 3.1-3.4 |

**总工时**：~8h | **风险**：中高（整合质量需要反复调优 prompt）

### Phase 4 (远期): 高级特性

- **记忆关联图**：记忆之间的因果/时序关系（简化版 Zep 图结构）
- **用户画像层**：从对话中提取用户偏好/习惯（LangMem Semantic 记忆的"human"面）
- **Dream 模式**：在长时间空闲时运行更深层的 reflection（Generative Agents 风格的递归整合）

---

## 8. 致谢与参考

### 已确认参考（通过 3-0 交叉验证）

| 来源 | 类型 | URL |
|------|------|-----|
| Packer et al. (2023) — MemGPT | 学术论文 | https://arxiv.org/abs/2310.08560 |
| Letta — Memory Blocks 架构 | 官方文档 | https://docs.letta.com/concepts/memgpt/ |
| Letta — Memory Blocks 博客 | 官方博客 | https://www.letta.com/blog/memory-blocks |
| LangChain — LangMem SDK 发布 | 官方博客 | https://www.langchain.com/blog/langmem-sdk-launch |
| LangChain — LangMem 概念指南 | 官方文档 | https://langchain-ai.github.io/langmem/concepts/conceptual_guide/ |
| LangChain — LangMem 仓库 | 开源代码 | https://github.com/langchain-ai/langmem |
| Zep AI — Graphiti 论文 | 学术论文 | https://export.arxiv.org/abs/2501.13956v1 |
| Zep AI — Graphiti 仓库 | 开源代码 | https://github.com/getzep/graphiti |
| Agnaistic — Memory Books 文档 | 官方文档 | https://agnai.guide/docs/memory/memory-books |
| Agnaistic — Chat Embeddings 文档 | 官方文档 | https://agnai.guide/docs/memory/embeddings |
| Park et al. (2023) — Generative Agents | 学术论文 | https://arxiv.org/abs/2304.03442 |
| MemR3 (2025) — LangGraph Memory Reasoning | 学术论文 | https://arxiv.org/abs/2512.20237 |

### 已审阅但主张被 refuted 的来源

Du (2025) — Agent Memory POMDP, Cognee — Competition Comparison, Mem0 — Benchmark Blog, SillyTavern — DeepWiki, RisuAI — DeepWiki, Letta — Sleep-time Compute Blog, CrewAI — Cognitive Memory Blog

### 研究统计

```
搜索角度:     5
来源抓取:     20
主张提取:     78
主张验证:     25 (3-vote adversarial)
确认:         11
证伪:         14
综合后保留:   6
Agent 调用:   104
总工时:       ~30 min (自动化研究)
```

---

> **方案书状态**: 待评审
> **下一步**: 确定 Phase 1 启动时间，开始实现 `memory_retrieve` 节点
