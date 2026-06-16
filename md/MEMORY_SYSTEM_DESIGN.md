# 记忆系统设计方案

> 目标效果：  
> 用户：「你还记得上次我们一起出去的那次吗？」  
> 月下：「嗯……那次…………」

---

## 一、问题分析

当前 Lunar 的角色是"金鱼脑"——每轮对话只看最近 4 条消息（perception context window），对过去的互动毫无记忆。`relationship_state` 虽然会跨轮累积（好感度、信任度慢慢变化），但那是 6 个浮点数，不包含任何**情节内容**。

要做出"回忆"效果，需要三件事：

| 环节 | 做什么 | 对应问题 |
|------|--------|----------|
| **存储** | 有意义的互动结束后，把摘要 + 当时情绪状态存下来 | "有没有东西可回忆？" |
| **检索** | 用户提到过去 / 当前情绪触发联想时，找出相关记忆 | "回忆什么？" |
| **表达** | 把记忆注入 LLM prompt，引导角色用"……"、模糊措辞、情绪共鸣来表达 | "怎么回忆才自然？" |

---

## 二、架构：在图里插两个新节点

当前图结构：

```
START → inject_system → perception → state_engine → state_formatter → llm → END
```

改造后：

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

### 为什么放在这里

- **memory_retrieval** 在 state_formatter 之后、llm 之前：
  - 此时 `internal_state` / `relationship_state` 已更新完毕，可以做状态相似度检索
  - 检索结果需要在 llm 调用前注入 prompt
  
- **memory_formation** 在 llm 之后：
  - 此时一整轮交互（用户消息 + 角色回复）已完成，可以判断是否值得形成记忆
  - 需要读取 `user_stimuli` 中的 `emotional_weight` 来判断显著性

### 一个关键的小改动

当前 `state_engine_node` 在消费完 `user_stimuli` 后立即将其设为 `None`。但 `memory_formation_node` 需要读取 `ST_EMOTIONAL_WEIGHT` 来判断这轮对话的情感重量。所以：

- **改前**：`state_engine_node` 中 `result["user_stimuli"] = None`
- **改后**：移到 `memory_formation_node` 末尾清除

`state_formatter` 和 `memory_retrieval` 都不读取 `user_stimuli`，不会被中间残留影响。

---

## 三、记忆存储

### 3.1 存储格式

一个 JSON 文件 `db/memories.json`，内容是一个数组：

```json
[
  {
    "id": "a1b2c3d4-...",
    "title": "关于一起去看红月的对话",
    "created_at": "2026-06-15T21:30:00",
    "user_message": "下次我们一起去看红月吧？",
    "character_response": "……红月吗。和你一起的话，应该会很美吧。",
    "emotional_weight": 0.72,
    "significance": 0.65,
    "internal_state": [0.55, 0.3, 0.2, 0.15, 0.1, 0.7, 0.5, 0.2],
    "relationship_state": [0.6, 0.55, 0.5, 0.35, 0.5, 0.55]
  }
]
```

各字段说明：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | UUID 字符串 | 唯一标识 |
| `title` | str | 用户消息截断到 50 字作为标题 |
| `created_at` | ISO 8601 | 记忆形成时间（用于计算"3天前"） |
| `user_message` | str | 用户消息截断到 300 字 |
| `character_response` | str | 角色回复截断到 300 字 |
| `emotional_weight` | float [0,1] | 形成时的 ST_EMOTIONAL_WEIGHT |
| `significance` | float [0,1] | 综合显著性分数 |
| `internal_state` | float[8] | 形成时的内部状态快照（用于状态检索） |
| `relationship_state` | float[6] | 形成时的关系状态快照 |

### 3.2 MemoryStore 类

在 `memory.py` 中实现：

```
MemoryStore("db/memories.json")
  .load()   → List[MemoryNode]    # 从文件读取，文件不存在/损坏则返回 []
  .save()   → None                 # 原子写入（先写临时文件再 rename）
  .add(node) → None                # load → append → save
  .get_all() → List[MemoryNode]
```

序列化：Pydantic `model_dump()` 自动把 numpy 数组转成 list（和 test.json 一致）。反序列化时 `_ensure_ndarray` 转回来。

### 3.3 不做什么（v1）

- **不做记忆图谱**（Edge/关联边）——v1 先验证检索-注入链路，关联图谱是锦上添花
- **不做自动摘要**（LLM 生成标题）——直接用用户消息截断当标题，零延迟
- **不做记忆遗忘/衰减**——v1 不做自动清理，500 条以内不会影响性能
- **不做 Chroma 向量检索**——bigram Jaccard 对中文够用，零外部依赖

---

## 四、记忆检索（三路并行）

`memory_retrieval_node` 的核心逻辑：

```
输入：state（含 messages, internal_state）
输出：retrieved_memories（格式化字符串 或 None）

1. 从 db/memories.json 加载所有记忆
2. 如果记忆为空 → 返回 None
3. 三路检索并行：
   a. 近因路：取最近 2 条                  ← 永远执行
   b. 内容路：bigram Jaccard 搜 top 3      ← 仅在有时序关键词时执行
   c. 状态路：cosine similarity 搜 top 3   ← 永远执行，阈值 0.75
4. 合并去重，按综合分数排序，取 top 5
5. 格式化为 prompt 文本
```

### 4.1 近因检索

```
recent = sorted(memories, key=created_at, reverse=True)[:2]
score = 0.5  （固定中等分数，确保不会被内容/状态结果淹没）
```

人类对话中天然倾向于引用近期事件。即使没有关键词触发，近因记忆也始终参与。

### 4.2 内容检索（时序关键词检测）

**触发条件**：用户消息中包含以下任一关键词时才执行内容检索：

```python
TEMPORAL_KEYWORDS = [
    "上次", "记得", "那时候", "之前", "曾经", "以前", "那次",
    "回忆", "想起来", "还记得", "没忘", "忘了", "忘记",
    "那一天", "有一天", "上次我们", "你记不记得", "你记得",
]
```

**匹配算法**：字符级 bigram Jaccard 相似度

```python
def _bigrams(text: str) -> set[str]:
    return {text[i:i+2] for i in range(len(text) - 1)}

def _content_score(query: str, memory: MemoryNode) -> float:
    q_bigrams = _bigrams(query)
    m_text = memory.user_message + " " + memory.title
    m_bigrams = _bigrams(m_text)
    if not q_bigrams or not m_bigrams:
        return 0.0
    return len(q_bigrams & m_bigrams) / len(q_bigrams | m_bigrams)
```

举例：
- 用户"上次我们一起去看的红月" vs 记忆"下次我们一起去看红月吧？" 
- 共同 bigram：`我们` `们一` `一起` `起去` `去看` `红月` → 高匹配
- 用户"今天天气不错" vs 记忆"下次我们一起去看红月" → 几乎 0 匹配

**为什么不用 jieba 分词**：jieba 是额外依赖，引入安装负担。中文 bigram 的特性是——两个相邻字构成的片段有很强的主题相关性（"红月""一起""那次"都是最小语义单元），Jaccard 对短文本效果不差。v2 可以换成 jieba 或 Chroma embedding。

### 4.3 状态检索（普鲁斯特效应）

使用已有的 `cos_similarity` 函数（`memory.py:79-88`），计算当前 `internal_state` 与每条记忆的 `internal_state` 快照之间的余弦相似度：

```python
def search_by_state(
    internal: np.ndarray,
    nodes: List[MemoryNode],
    threshold: float = 0.75,
    k: int = 3
) -> List[tuple[MemoryNode, float]]:
    scored = []
    for node in nodes:
        sim = cos_similarity(internal, node.internal_state)
        if sim >= threshold:
            scored.append((node, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
```

**阈值 0.75 的含义**：只有当当前情绪与记忆中的情绪**明显相似**时才触发。这避免了对每条记忆都返回的"泛检索"问题。8 个维度（精力、压力、孤独、不安、烦躁、思念、社交电量、精神疲劳）构成的情绪指纹有足够的区分度。

举例：
- 当前状态：loneliness=0.7, longing=0.8, insecurity=0.6（孤独思念状态）
- 记忆 A：loneliness=0.65, longing=0.75, insecurity=0.55 → cosine ≈ 0.95 → **触发**
- 记忆 B：energy=0.8, stress=0.1, irritation=0.05（轻松愉快状态）→ cosine ≈ 0.2 → **不触发**

### 4.4 去重与排序

```python
# 合并三路结果
all_results = recent_results + content_results + state_results

# 去重（按 id）
seen = set()
unique = []
for memory, score in sorted(all_results, key=lambda x: x[1], reverse=True):
    if memory.id not in seen:
        seen.add(memory.id)
        unique.append(memory)
        if len(unique) >= 5:
            break
```

最多 5 条记忆，避免 context window 被记忆淹没。

---

## 五、记忆形成

### 5.1 显著性判断

```python
def compute_significance(state: State) -> float:
    # 因子1：本轮对话的情感重量（来自 perception 的 emotional_weight_stimulus）
    stimuli = state["user_stimuli"]
    emotional_weight = stimuli[ST_EMOTIONAL_WEIGHT] if stimuli is not None else 0.0
    
    # 因子2：消息长度归一化（实质性对话比简短对话更值得记住）
    human_msg = _get_last_human_message(state["messages"])
    ai_msg = _get_last_ai_message(state["messages"])
    msg_len = min(len(human_msg) + len(ai_msg), 500) / 500.0
    
    return 0.7 * emotional_weight + 0.3 * msg_len
```

**阈值 = 0.35**：低于此值不形成记忆。

不同场景的显著性估算：

| 场景 | emotional_weight | 消息长度 | significance | 是否形成记忆 |
|------|:---:|:---:|:---:|:---:|
| "你好" / "嗯" | 0.05 | 0.04 | 0.05 | ❌ |
| 轻松闲聊 | 0.15 | 0.3 | 0.20 | ❌ |
| 日常关心 | 0.25 | 0.4 | 0.30 | ❌ |
| 分享心情 | 0.40 | 0.5 | 0.43 | ✅ |
| 深情告白 | 0.75 | 0.6 | 0.71 | ✅ |
| 激烈争吵 | 0.85 | 0.8 | 0.84 | ✅ |

### 5.2 创建记忆

```python
node = MemoryNode(
    id=str(uuid.uuid4()),
    title=human_msg[:50].strip(),
    created_at=datetime.now().isoformat(),
    user_message=human_msg[:300],
    character_response=ai_msg[:300],
    emotional_weight=float(stimuli[ST_EMOTIONAL_WEIGHT]),
    significance=sig,
    internal_state=internal_state.copy(),
    relationship_state=relationship_state.copy(),
)
store.add(node)
```

### 5.3 边界条件

- **首轮（inject_system 后）**：没有 AI 回复 → 跳过形成
- **perception 失败（error=True）**：跳过形成
- **user_stimuli 为 None**：跳过形成
- **user_stimuli 清除**：在 `memory_formation_node` 返回时执行 `{"user_stimuli": None}`

---

## 六、记忆注入（Prompt 工程）

这是做出"嗯……那次…………"效果的核心。

### 6.1 llm_node 改造

当前 llm_node 把 `state_description` 作为 SystemMessage 注入。改造后，把 `retrieved_memories` 也合并进去：

```python
def llm_node(state: State) -> dict:
    parts = []
    
    state_desc = state.get("state_description")
    if state_desc:
        parts.append(state_desc)
    
    memories = state.get("retrieved_memories")
    if memories:
        parts.append(memories)
    
    if parts:
        combined = "\n\n".join(parts)
        inject_msg = SystemMessage(content=combined)
        res = model.invoke([inject_msg] + messages)
    else:
        res = model.invoke(messages)
    
    return {"messages": [res]}
```

**为什么不分开两条 SystemMessage**：多条 SystemMessage 的优先级语义不明确，合并为一条保证所有上下文被平等对待。

### 6.2 主动回忆模板

当用户消息中包含时序关键词（"你还记得……""上次……"）时，使用主动回忆模板：

```
【记忆线索】
用户似乎在提及过去的事。以下是你能模糊记起的相关往事：

- [3天前] 关于"我们一起去看红月"的记忆
  当时用户说："下次我们一起去看红月吧？"
  你回应了："……红月吗。和你一起的话，应该会很美吧。"
  那时你感到：深深的思念，渴望被陪伴，心情柔软而期待

- [1周前] 关于"你还会等我吗"的记忆
  ...

【回忆表现指引】
- 先用"嗯……"或"……"停顿，表现正在努力回忆的样子
- 提及记忆时用模糊的表达（"那次……""好像……""我记得似乎……"）
- 强调当时的感受，而非精确的细节
- 记不清完整对话是自然的，不要编造你不记得的细节
- 如果记忆唤起了相似的情绪，让它在语气中自然流露
- 提及记忆后，自然地回到当下的对话，不要一直沉溺在过去
```

### 6.3 被动回忆模板

当没有时序关键词、纯粹是状态相似触发的记忆时，使用被动模板：

```
【记忆浮现】
当前心境让你隐约想起了过去的一些片段。你不需要刻意提及这些记忆，
但如果恰当，可以在回应中自然地流露出似曾相识的感觉。
用"……"表达回忆时的停顿和不确定。

- [5天前] 关于"..."的记忆
  ...
```

### 6.4 为什么这样设计 Prompt

| 设计决策 | 理由 |
|----------|------|
| "模糊表达"指引 | 避免角色像数据库查询一样精确复述——真实的人回忆是模糊的 |
| "强调感受而非细节" | 情感记忆比事实记忆更真实、更有感染力 |
| "不要编造" | 防止 LLM 幻觉——记不清就说记不清 |
| "回到当下" | 回忆是调味品，不是主菜——不要让每次对话都变成怀旧大会 |
| 两种模板分离 | 用户主动问"记得吗"时角色应该努力回忆；状态触发时应该只是隐约流露 |

---

## 七、文件改动清单

| 文件 | 改动类型 | 内容 |
|------|:---:|------|
| `memory.py` | **重写** | MemoryNode v2（去掉 Edge/edges）、MemoryStore（JSON 持久化）、三路检索函数、显著性计算、记忆格式化 |
| `state.py` | **+1 字段** | State TypedDict 新增 `retrieved_memories: Optional[str]` |
| `nodes.py` | **+2 节点，改 2 节点** | 新增 `memory_retrieval_node`、`memory_formation_node`；修改 `llm_node`（合并记忆注入）；修改 `state_engine_node`（移除 user_stimuli 清除） |
| `graph/_builder.py` | **改连线** | 注册 2 个新节点；`state_formatter → memory_retrieval → llm → memory_formation → END` |
| `prompts/memory.py` | **新建** | 主动/被动回忆模板 |
| `prompts/__init__.py` | **+导出** | 导出新模板 |
| `agent.py` | **无需改动** | 图编译自动包含新节点 |

---

## 八、验证方案

### 8.1 单元级验证（可脱离图独立测试）

| 测试项 | 验证内容 |
|--------|----------|
| MemoryStore save/load 来回 | 写入→读取→字段一致 |
| `search_by_content("上次我们一起出去")` | 返回内容相关的记忆 |
| `search_by_state(lonely_vector)` | 返回情绪相似的记忆 |
| `has_temporal_reference("你还记得吗？")` | True |
| `has_temporal_reference("今天天气不错")` | False |
| `compute_significance(emotional_weight=0.1)` | < 0.35，不形成记忆 |
| `compute_significance(emotional_weight=0.6, msg_len=0.5)` | > 0.35，形成记忆 |

### 8.2 集成验证（通过 agent.py TUI）

| 场景 | 预期行为 |
|------|----------|
| 3 轮"你好""吃了没""嗯" | 无新记忆文件或记忆数不变 |
| 用户分享一段心情故事 | `db/memories.json` 新增一条记录，state snapshot 正确 |
| 用户说"你还记得上次我和你说的那个事吗" | 内容检索命中 → LLM 响应含"……""那次……" |
| 用户谈论孤独话题，角色正好处于高 loneliness 状态 | 普鲁斯特效应触发过往孤独记忆 → 角色语气流露似曾相识 |
| `db/memories.json` 不存在（首次运行） | 不报错，正常运行 |
| `db/memories.json` 内容损坏 | 捕获异常，返回空列表，下次形成记忆时覆盖重建 |

### 8.3 品质验证（人工判断）

- 角色回忆时是否有"……"停顿？是否使用了省略号？
- 角色是否在回忆**感受**而非精确复述对话？
- 回忆后是否自然回到当下？还是沉溺在过去？
- 角色是否编造了它不可能知道的内容？
- 当用户没提过去时，回忆是否喧宾夺主？

---

## 九、风险与缓解

| 风险 | 缓解 |
|------|------|
| bigram Jaccard 对中文语义匹配太粗糙 | v1 可用。如果需要，后续替换为 jieba 分词（无需改接口） |
| LLM 忽略回忆 prompt 指引 | 注入为 SystemMessage（最高优先级）。已有 state_formatter 证明 DeepSeek 遵循 SystemMessage 指令 |
| 记忆形成过多或过少 | significance 阈值 0.35 是主要调参旋钮，可随时调整 |
| 记忆文件无限增长 | v1 假设几百条量级（人工对话场景），v2 加 cap=500 + 最低显著性淘汰 |
| 8 维 internal_state 区分度不够 | 阈值设 0.75 已经很高。如需增强，可以 concatenate relationship_state 形成 14 维向量 |
| user_stimuli 清除位置变化引入 bug | state_formatter / memory_retrieval 均不读取 user_stimuli，残留无影响 |

---

## 十、后续增强（v2+，本期不做）

- **记忆图谱**：恢复 Edge 关联边，使检索可以沿关联扩散（"那次吵架之后我们就……"）
- **LLM 摘要标题**：用一次轻量 LLM 调用把用户消息总结为更自然的标题
- **记忆遗忘曲线**：随时间推移降低检索分数（艾宾浩斯效应）
- **Chroma 向量检索**：替换 bigram Jaccard，真正的语义检索
- **用户心理档案**：独立维护用户人格/情绪，实现 Theory of Mind 记忆
- **记忆合并**：把相关记忆自动合并为"故事线"
