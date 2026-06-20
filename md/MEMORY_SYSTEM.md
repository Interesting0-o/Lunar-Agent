# 记忆系统设计

> 2026-06-19 | 合并自 MEMORY_SYSTEM_DESIGN.md（实现设计） + LLM_MEMORY_MANAGEMENT_DESIGN.md（研究路线）

---

## 一、问题与现状

### 1.1 要解决的问题

当前 Lunar 的角色是"金鱼脑"——每轮对话只看最近 4 条消息（perception context window），对过去的互动毫无记忆。`relationship_state` 虽然会跨轮累积，但那只是 6 个浮点数，不包含任何**情节内容**。

| 环节 | 做什么 | 状态 |
|------|--------|:----:|
| **存储** | 有意义的互动结束后，把摘要 + 情绪状态存下来 | ✅ MemoryStore 可用 |
| **检索** | 用户提过去 / 情绪触发联想时，找出相关记忆 | ✅ 三路检索实现 |
| **表达** | 把记忆注入 LLM prompt，引导自然回忆 | 🟡 LLM prompt 模板需要调整 |
| **管线集成** | 两个 memory 图节点 | 🟡 memory_inject_node 和 memory_summery_node 为 stub |

### 1.2 设计原则（来自研究调查）

对 20+ 来源交叉验证后确认三条核心原则：

1. **检索路径零 LLM**：embedding / cosine 搜索 (<0.5s) 足以胜任，LLM 不应出现在热路径
2. **记忆整合走后台**：对话间隙运行 consolidation agent，不阻塞用户交互
3. **层级从简开始**：种子记忆 → 情节记忆 → 语义记忆，渐进式构建

---

## 二、三层记忆架构（Lunar Memory OS）

参考 Letta 的 OS 启发式三层设计，结合 Lunar 的心理状态引擎特点：

```
┌─────────────────────────────────────────────────────┐
│            对话上下文（热路径，每轮注入）                │
│  ┌───────────┐ ┌──────────┐ ┌─────────────┐       │
│  │ 心理状态描述│ │ 状态表达  │ │ 检索到的记忆  │       │
│  └───────────┘ └──────────┘ └─────────────┘       │
│  + 最近 N 轮对话                                     │
└─────────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────────┐
│              非对话记忆（温/冷路径）                    │
│  ┌─────────────────┐ ┌──────────────────────────┐  │
│  │  情节记忆         │ │  语义记忆                 │  │
│  │  (episodic)      │ │  (semantic)              │  │
│  │  单次事件快照     │ │  用户事实 / 偏好 / 摘要   │  │
│  │  带状态向量       │ │  由 LLM 整合生成          │  │
│  └─────────────────┘ └──────────────────────────┘  │
│                    ↕ (后台 consolidation agent)      │
│  ┌──────────────────────────────────────────────┐  │
│  │  种子记忆                                     │  │
│  │  (seed_memories.py: 24 条初始记忆锚点)         │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

- **热路径**：状态描述 + 检索到的记忆 → 每轮注入 LLM，零额外 LLM 调用
- **温路径**：内存形成，在对话结束后评估显著性并持久化
- **冷路径**：consolidation agent，在空闲时运行，整合情节为语义

### 2.1 与记忆类型的映射

| 记忆类型 | 存储位置 | 依赖模型 | 特征 |
|---------|---------|---------|------|
| 种子记忆 | `prompts/character_memories.py` | 无（静态） | 24 条初始锚点，persona 定义 |
| 情节记忆 | `db/memories.json` | 无 | 状态向量 + 事件快照，确定性更新 |
| 语义记忆 | `db/semantic_memories.json` v2 | LLM | 由 consolidation agent 从情节提炼 |
| 程序记忆 | 不在 v1 范围 | — | 角色行为模式、系统 prompt 演化 |

---

## 三、图集成

### 3.1 节点位置

```
START → inject_system → perception → state_engine → state_formatter
                                                          ↓
                                            memory_retrieval [NEW]
                                                          ↓
                                                        llm [MODIFIED]
                                                          ↓
                                            memory_formation [NEW]
                                                          ↓
                                                         END
```

### 3.2 各节点职责

| 节点 | 位置 | 输入 | 输出 |
|------|------|------|------|
| `memory_retrieval_node` | state_formatter 后，llm 前 | internal_state + relationship_state + user_message | `retrieved_memories`（格式化字符串或 None） |
| `memory_formation_node` | llm 后，END 前 | user_stimuli + internal/relationship/surface + user/ai messages | 写入 `db/memories.json`，清空 user_stimuli |
| `memory_summery_node` | **stub** | 语义记忆整合 | 后台 consolidation |

### 3.3 关键约束

`state_engine_node` 当前在消费完 `user_stimuli` 后将其设为 `None`，但 `memory_formation_node` 需要读取 `ST_EMOTIONAL_WEIGHT`。需将 `user_stimuli` 清除移到 `memory_formation_node` 末尾。

`state_formatter` 和 `memory_retrieval` 不读取 `user_stimuli`，不受中间残留影响。

---

## 四、记忆存储

### 4.1 MemoryNode（Pydantic 模型）

```python
class MemoryNode(BaseModel):
    id: str
    title: str
    content: str                # 形成时的对话摘要
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    significance: float         # 综合显著性 [0, 1]
    emotional_weight: float     # ST_EMOTIONAL_WEIGHT 快照
    state_checkpoint: StateCheckpoint  # 内部/关系/表面状态向量
    embedding: Optional[list[float]] = None  # v2: 用于语义搜索
```

`StateCheckpoint` 保存三个状态向量（internal_state, relationship_state, surface_state），用于后续的"情绪指纹"检索。

### 4.2 MemoryStore（JSON 持久化）

```python
MemoryStore("db/memories.json")
  .load()   → List[MemoryNode]    # 文件不存在/损坏则返回 []
  .save()   → None                 # 原子写入（tmp → rename）
  .add(node) → None
  .get_all() → List[MemoryNode]
```

序列化自动处理 ndarray↔list 转换（Pydantic `model_dump`）。

### 4.3 v1 不做

- 记忆图谱（关联边）——先验证检索-注入链路
- LLM 自动摘要标题——直接用用户消息截断
- 记忆遗忘/衰减——500 条以内不影响性能
- Chroma 向量检索——bigram Jaccard 对中文够用

---

## 五、记忆检索（三路并行）

### 5.1 总体流程

```
输入：internal_state + user_message
输出：top-5 格式化记忆字符串 或 None

1. 加载所有记忆 → 空则返回 None
2. 三路并行：
   a. 近因检索：最近 2 条（score=0.5，始终执行）
   b. 内容检索：bigram Jaccard top-3（仅有时序关键词时）
   c. 状态检索：cosine similarity top-3（阈值 0.75，始终执行）
3. 合并去重 → 按分数取 top-5 → 格式化
```

### 5.2 近因检索

始终返回最近 2 条记忆。人类对话中天然倾向于引用近期事件。

### 5.3 内容检索（时序关键词触发）

**触发条件**：用户消息包含以下关键词时才执行：

```python
TEMPORAL_KEYWORDS = [
    "上次", "记得", "那时候", "之前", "曾经", "以前", "那次",
    "回忆", "想起来", "还记得", "没忘", "忘了", "忘记",
    "那一天", "有一天", "上次我们", "你记不记得", "你记得",
]
```

**算法**：字符级 bigram Jaccard 相似度

```python
def _content_score(query: str, memory: MemoryNode) -> float:
    q_bigrams = set(query[i:i+2] for i in range(len(query) - 1))
    m_bigrams = set(memory.content[:100])
    if not q_bigrams or not m_bigrams:
        return 0.0
    return len(q_bigrams & m_bigrams) / len(q_bigrams | m_bigrams)
```

### 5.4 状态检索（普鲁斯特效应）

使用 cosine similarity 比较当前 `internal_state` 与记忆快照。阈值 0.75，返回 top-3。

8 维情绪指纹（精力/压力/孤独/不安/烦躁/思念/社交电量/精神疲劳）提供足够区分度。

### 5.5 去重与排序

最多 5 条记忆，避免 context window 被记忆淹没。

---

## 六、记忆形成

### 6.1 显著性判断

```python
def compute_significance(state) -> float:
    stimuli = state["user_stimuli"]
    emotional_weight = stimuli[ST_EMOTIONAL_WEIGHT] if stimuli is not None else 0.0
    human_msg = _get_last_human_message(state["messages"])
    ai_msg = _get_last_ai_message(state["messages"])
    msg_len = min(len(human_msg) + len(ai_msg), 500) / 500.0
    return 0.7 * emotional_weight + 0.3 * msg_len
```

**阈值 = 0.35**。不同场景估算：

| 场景 | emotional_weight | 消息长度 | significance | 是否形成 |
|------|:---:|:---:|:---:|:---:|
| "你好"/"嗯" | 0.05 | 0.04 | 0.05 | ❌ |
| 轻松闲聊 | 0.15 | 0.3 | 0.20 | ❌ |
| 日常关心 | 0.25 | 0.4 | 0.30 | ❌ |
| 分享心情 | 0.40 | 0.5 | 0.43 | ✅ |
| 深情告白 | 0.75 | 0.6 | 0.71 | ✅ |
| 激烈争吵 | 0.85 | 0.8 | 0.84 | ✅ |

### 6.2 边界条件

- **首轮**（inject_system 后）：无 AI 回复 → 跳过
- **perception 失败**（error=True）：跳过
- **user_stimuli 为 None**：跳过
- **user_stimuli 清除**：在 `memory_formation_node` 返回时执行

---

## 七、Prompt 注入（LLM 记忆感知）

### 7.1 主动回忆模板

用户提及过去时（含时序关键词）：

```
【记忆线索】用户似乎在提及过去的事。以下是你能模糊记起的相关往事：

- [3天前] 关于"我们一起去看红月"的记忆
  那时你感到：深深的思念，渴望被陪伴，心情柔软而期待

【回忆表现指引】
- 先用"嗯……"或"……"停顿，表现正在努力回忆的样子
- 提及记忆时用模糊表达（"那次……""好像……"）
- 强调当时的感受，而非精确细节
- 记不清完整对话是自然的，不要编造
- 提及记忆后，自然地回到当下
```

### 7.2 被动回忆模板

没有时序关键词、纯状态触发：

```
【记忆浮现】当前心境让你隐约想起了一些过去片段。你不需要刻意提及，
但如果恰当，可以在回应中自然地流露似曾相识的感觉。
```

### 7.3 两种模板的设计理由

| 设计决策 | 理由 |
|----------|------|
| "模糊表达"指引 | 人回忆是模糊的，不是数据库查询 |
| "强调感受" | 情感记忆比事实更真实、更有感染力 |
| "不要编造" | 防止 LLM 幻觉 |
| "回到当下" | 回忆是调味品，不是主菜 |
| 两种模板分离 | 用户主动问"记得吗"时努力回忆 vs 状态触发时隐约流露 |

---

## 八、记忆巩固（Consolidation — v2）

当前为 stub。计划：

1. **整合 agents**：每 N 轮或空闲时，用 LLM 将高 significance 的情节记忆压缩为语义摘要
2. **重要性再评估**：遗忘曲线（艾宾浩斯）+ 用户互动频率
3. **冲突消解**：当新记忆与旧事实冲突时，标记为"不确定"

---

## 九、文件改动清单

| 文件 | 改动 | 内容 |
|------|:---:|------|
| `memory.py` | 已实现 | MemoryNode v2 + MemoryStore（JSON）+ 三路检索 + 显著性计算 |
| `state.py` | +1 字段 | `retrieved_memories: Optional[str]` |
| `nodes.py` | 2 stub 节点 | `memory_inject_node`（retrieval）和 `memory_summery_node`（consolidation）未完成 |
| `graph/_builder.py` | 待改 | 注册 2 个新节点、改连线 |
| `prompts/memory_summery.py` | 已实现 | 主动/被动回忆模板 + memory 总结 prompt |
| `prompts/character_memories.py` | 已存在 | 24 条种子记忆 |

---

## 十、附录：研究参考（来自 LLM 记忆管理调查）

### 10.1 核心发现

1. **内存层级化是共识**：Letta（Core/Recall/Archival）、LangMem（Semantic/Episodic/Procedural）、Zep（双时态知识图谱）
2. **LLM 参与度的光谱分布**：从 Agnaistic 的零 LLM 到 Letta 的全 LLM
3. **热/冷路径分离是延迟优化关键**：LangMem 的 `ReflectionExecutor` 将整合推迟到后台
4. **"管理"阶段是最大空白**：写入和读取已充分解决，但组织/合并/压缩/遗忘仍是"启发式"
5. **Generative Agents 的 reflection**：重要性评分 + 递归 → 高层次洞察，但离线仿真设计
6. **MemR3 是 LangGraph 原生**：Router → Retrieve → Reflect → Answer 可直接适配

### 10.2 系统光谱

```
LLM 参与度
100% → Letta/MemGPT
 75% → Generative Agents, LangMem
 50% → Zep/Graphiti
 25% → CrewAI, Mem0
  0% → Agnaistic, SillyTavern (关键词匹配/手工编辑)
```

2025 年后普遍从"全 LLM"回退到"热路径去 LLM 化"。

### 10.3 参考项目

| 项目 | 类型 | 对 Lunar 的参考 |
|------|------|---------------|
| FAtiMA | 学术情感架构 | OCC 评价 + 自传体记忆 + 动机层 + ToM |
| Letta/MemGPT | 三层记忆 | Core/Recall/Archival 启发 Lunar Memory OS |
| LangMem | LangChain 官方 | Semantic/Episodic/Procedural 三层 + ReflectionExecutor |
| Mem0 | 语义记忆 | 自动去重、向量存储用户事实与偏好 |
| Zep | 时序+语义 | 时间知识图谱 Graphiti，PostgreSQL |
| Nomi.ai | 工业 AI 陪伴 | 最佳长期记忆角色扮演产品之一 |
| SillyTavern | 开源引擎 | JSON 角色配置 + SQLite 持久化 |

### 10.4 评估基准

| 基准 | 用途 |
|------|------|
| CharacterEval | 角色一致性 |
| PERSIST | 人格稳定性 |
| RPEval | 情绪理解、决策、角色一致性 |
| CharacterBox | 角色忠诚度行为轨迹 |
| LoCoMo | 长上下文对话记忆 |
| LongMemEval | 长期记忆评估 |

### 10.5 参考文献

1. Letta/MemGPT — arXiv:2310.08560
2. Generative Agents — arXiv:2304.03442
3. Du et al. (2025) — POMDP 框架
4. Conway & Pleydell-Pearce (2000) — 自我记忆系统
5. Tulving (1972) — 情景记忆与语义记忆
6. MemR3 — LangGraph 原生记忆参考
7. W3C EmotionML 1.0 — 情绪标注标准
8. ISO 24617-2 — 对话行为标注
