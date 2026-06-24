"""
nodes —— LangGraph 图节点函数

所有图节点实现在此，agent.py 只负责注册和连线。

注意：state 中存储的 np.ndarray 在通过 JSON 序列化/反序列化
（如 SQLite checkpoint 恢复）后可能变为 Python list，
所有 node 函数在读取 state 时负责确保类型正确。
"""

import numpy as np
from prompts import SYSTEM_PROMPT, MEMORY_SYSTEM_PROMPT, SEED_MEMORIES
from config import PERCEPTION_CONFIG
import logging
from langchain.messages import SystemMessage,HumanMessage
from llm import model,memory_summry_model
from state import State, DEFAULT_TRAITS, DEFAULT_INTERNAL, DEFAULT_RELATIONSHIP
from perception import extract_recent_context, call_perception_with_retry
from state_engine import update_all
from state_formatter import format_state_for_node
from memory import MemoryStore, MemoryNode



logger = logging.getLogger(__name__)


def _ensure_array(v, dtype=np.float64) -> np.ndarray:
    """确保值为 numpy 数组（兼容 Python list/json 反序列化）。"""
    if v is None:
        return None#type:ignore
    if isinstance(v, np.ndarray):
        return v
    return np.asarray(v, dtype=dtype)


def _seed_character_memories(memory_id: str) -> int:
    """将角色的核心设定故事写入长期记忆。

    仅当 MemoryStore 为空时执行（幂等——不会重复写入）。
    使用 DEFAULT_* 基线作为状态快照，因为这些是身份锚点记忆。
    返回写入的记忆数量。
    """
    store = MemoryStore(memory_id=memory_id)
    if store.count() > 0:
        return 0

    from memory import compute_embedding

    nodes: list[MemoryNode] = []
    for item in SEED_MEMORIES:
        embedding = compute_embedding(item["content"])
        node = MemoryNode.from_state_vectors(
            title=item["title"],
            content=item["content"],
            internal_state=DEFAULT_INTERNAL.copy(),
            relationship_state=DEFAULT_RELATIONSHIP.copy(),
            embedding=embedding,
        )
        nodes.append(node)

    if nodes:
        store.add_batch(nodes)
        logger.info("已写入 %d 条种子记忆到 %s", len(nodes), memory_id)

    return len(nodes)


def inject_system_node(state: State) -> dict:
    """首次运行：注入角色系统提示词 + 默认人格特质 + 种子记忆。"""
    result = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT)],
        "has_inject_system_prompt": True,
    }
    # 如果 state 已有 traits（来自 test.json），不覆盖
    if not state.get("traits"):
        result["traits"] = DEFAULT_TRAITS

    # 写入角色背景故事作为种子记忆
    memory_id = state.get("memory_id") or "main"
    _seed_character_memories(memory_id)

    return result


def perception_node(state: State) -> dict:
    """感知节点：从用户输入中直接提取心理刺激强度。

    将 user_stimuli 写入图 State，state_engine_node 消费后会立即清理。
    同时写入 stimulus_metadata（约束②），包含置信度/来源/衰减因子。
    失败时设置 error=True，条件边据此引导到 END。
    """
    cfg = PERCEPTION_CONFIG
    context = extract_recent_context(state["messages"], cfg["context_window"])

    # 注入状态上下文（辅助感知判断）
    internal_state = _ensure_array(state.get("internal_state"))
    relationship_state = _ensure_array(state.get("relationship_state"))

    result = call_perception_with_retry(
        context, cfg,
        internal_state=internal_state,
        relationship_state=relationship_state,
    )

    if result is None:
        return {"error": True}

    return {
        "user_stimuli": result["user_stimuli"],
        "stimulus_metadata": result.get("stimulus_metadata"),
        "error": False,
    }


def state_engine_node(state: State) -> dict:
    """状态引擎节点：根据感知输出的心理刺激 + 当前状态 + Traits 更新所有状态层。

    流程：
      1. 从 checkpoint 加载持久化数据（decay_modulator, 时间戳）
      2. 用新轮的 stimulus_metadata 更新 decay_modulator（约束②）
      3. 计算自上次更新以来的 Δt，应用时间衰减（_decay.py）
      4. 用衰减后的状态 + 用户刺激运行 update_all
      5. 持久化衰减调制因子 + 新时间戳，清理临时字段（user_stimuli）
    """
    if state.get("error"):
        logger.info("state_engine 跳过本轮（perception 已标记 error）")
        return {}

    import time as _time
    from state_engine import apply_time_decay

    # ── 确保所有数值状态为 numpy 数组（兼容 json/list 反序列化） ──
    traits = _ensure_array(state.get("traits"))
    stimuli = _ensure_array(state.get("user_stimuli"))

    internal = _ensure_array(state.get("internal_state"))
    relationship = _ensure_array(state.get("relationship_state"))
    prev_surface = _ensure_array(state.get("surface_state"))

    # ── 加载持久化衰减调制因子 ──
    persistent_modulator = _ensure_array(state.get("current_decay_modulator"))

    # ── 重建 StimulusMetadata & 合并 decay_modulator ──
    meta_dict = state.get("stimulus_metadata")
    new_modulator = None
    if meta_dict is not None and isinstance(meta_dict, dict):
        from state import StimulusMetadata
        metadata = StimulusMetadata(
            confidence=np.asarray(meta_dict["confidence"], dtype=np.float64),
            source=np.asarray(meta_dict["source"], dtype=np.int8),
            decay_modulator=np.asarray(meta_dict["decay_modulator"], dtype=np.float64),
            timestamp=float(meta_dict.get("timestamp", 0.0)),
        )
        new_modulator = metadata.decay_modulator

    # ── 更新持久化衰减调制因子（EWMA 平滑，防止单轮剧烈波动） ──
    if new_modulator is not None:
        if persistent_modulator is None:
            persistent_modulator = new_modulator.copy()
        else:
            # 指数加权移动平均：新轮贡献 30%，历史贡献 70%
            persistent_modulator = 0.7 * persistent_modulator + 0.3 * new_modulator
        persistent_modulator = np.clip(persistent_modulator, 0.1, 2.0)

    # ── 计算自上次更新以来的 Δt ──
    last_ts = state.get("last_update_timestamp")
    now = _time.time()
    delta_hours = 0.0
    if last_ts is not None:
        delta_hours = max(0.0, (now - last_ts) / 3600.0)

    # ── 时间衰减（仅在有间隔时生效，max 24 小时防止数值溢出） ──
    if delta_hours > 0.01 and internal is not None and relationship is not None:
        decayed = apply_time_decay(
            internal, relationship, traits,
            delta_hours=min(delta_hours, 24.0),
            decay_modulator=persistent_modulator,
        )
        internal = decayed["internal_state"]
        relationship = decayed["relationship_state"]

    # ── 双速动力学计数器（关系态每 REL_BUFFER_INTERVAL 轮更新一次） ──
    rel_counter = state.get("rel_update_counter") or 0

    # ── 运行主引擎（用衰减后的状态） ──
    result = update_all(
        current_internal=internal,
        current_relationship=relationship,
        traits=traits,
        stimuli=stimuli,
        prev_surface=prev_surface,
        stimulus_metadata=metadata if meta_dict is not None else None,
        delta_hours=delta_hours,
        rel_counter=rel_counter,
    )

    # ── 持久化衰减调制因子和时间戳，清理临时字段 ──
    result["current_decay_modulator"] = persistent_modulator
    result["last_update_timestamp"] = now
    result["user_stimuli"] = None
    result["stimulus_metadata"] = None

    return result


def state_formatter_node(state: State) -> dict:
    """状态格式化节点：将 State Engine 输出的数值状态翻译为自然语言描述。

    写入 state_description 字段，供 llm_node 注入到模型调用中。
    如果 error=True（perception 失败），跳过格式化。
    """
    if state.get("error"):
        logger.info("state_formatter 跳过本轮（perception 已标记 error）")
        return {}

    state_description = format_state_for_node(state)
    return {"state_description": state_description}


def llm_node(state: State) -> dict:
    """LLM 回复节点：用消息历史 + 当前状态描述调用模型生成回复。"""
    messages = state["messages"]
    state_desc = state.get("state_description")

    if state_desc:
        inject_msg = SystemMessage(content=state_desc)
        res = model.invoke([inject_msg] + messages)
    else:
        res = model.invoke(messages)

    return {"messages": [res]}


def _format_memory_context(
    memories: list[tuple[MemoryNode, float]],
) -> str | None:
    """将检索到的记忆格式化为可注入 LLM 的上下文文本。

    以角色的第一人称视角呈现，让她"想起"这些过往片段。
    空列表返回 None。

    Args:
        memories: (MemoryNode, 相似度) 列表，按相似度降序。

    Returns:
        格式化的记忆上下文文本，或 None。
    """
    if not memories:
        return None

    lines = [
        "【记忆浮现】",
        "以下是你忽然想起的、与此刻情境相似的过往记忆。请自然地让这些回忆影响你接下来的回应——",
        "可以提及、可以暗指、也可以只是让它们悄无声息地改变你说话的温度。",
        "",
    ]

    for i, (node, score) in enumerate(memories, 1):
        # 相似度 → 模糊的"熟悉感"描述
        # 注意：状态值改为 [-1, 1] 后 cosine 可为负，阈值相应下调
        if score >= 0.90:
            feeling = "几乎一模一样"
        elif score >= 0.70:
            feeling = "非常相似"
        elif score >= 0.50:
            feeling = "有些熟悉"
        else:
            feeling = "隐约相关"

        lines.append(f"■ 记忆{i} · {node.title}（{feeling}）")
        lines.append(f"  {node.content}")
        lines.append("")

    lines.append("—— 以上记忆仅供参考，请自然地融入回应，不必逐条复述或刻意引用。")

    return "\n".join(lines)


def memory_inject_node(state: State) -> dict:
    """记忆注入节点：根据用户最新消息检索相关记忆，以 SystemMessage 注入消息流。

    在 llm_node 之前执行。返回的 SystemMessage 通过 add_messages 拼入消息历史，
    llm_node 调用模型时自然携带此上下文。无匹配记忆时返回空 dict。
    """
    # 逆序遍历获取最后一条用户消息
    messages = state["messages"]
    user_message = ""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            user_message = messages[i].content
            break

    if not user_message:
        return {}

    # 检索相关记忆
    memory_id = state.get("memory_id")
    results = MemoryStore(memory_id=memory_id).search_by_embedding(
        query_text=user_message,
        top_k=3,
        threshold=0.7,
    )

    memory_context = _format_memory_context(results)

    if memory_context:
        return {"messages": [SystemMessage(content=memory_context)]}
    return {}


def _parse_memory_json(text: str) -> tuple[str, str]:
    """从 LLM 输出中提取记忆的 title 和 summary。

    处理【思考】前导段落、```json 包裹、裸 JSON 等格式。
    返回 (title, summary)，解析失败返回 ("", "")。
    """
    import json
    import re
    from state import _strip_json_fence

    # 统一剥除 ```json ... ``` 包裹
    cleaned = _strip_json_fence(text)

    # 找到第一个含 "title" 的 JSON 对象
    match = re.search(r"\{[^{}]*\"title\"\s*:\s*\"[^\"]*\"[^{}]*\}", cleaned, re.DOTALL)
    if not match:
        return "", ""

    try:
        data = json.loads(match.group(0))
        return data.get("title", ""), data.get("summary", "")
    except json.JSONDecodeError:
        return "", ""



def memory_summery_node(state: State) -> dict:
    """记忆总结节点：在 llm 回复后执行，将本轮对话总结为记忆并持久化。

    以角色第一人称视角生成标题和摘要（调用 LLM），
    捕获当前心理状态快照 + 计算文本嵌入，存入 MemoryStore。
    解析失败或对话过短时静默跳过。
    """
    # 过滤所有 SystemMessage，保留纯对话
    messages = [m for m in state["messages"] if not isinstance(m, SystemMessage)]

    # 至少需要一轮对话（user + ai 各一条）
    if len(messages) < 2:
        return {}

    messages.insert(0, SystemMessage(content=MEMORY_SYSTEM_PROMPT))

    # 调用 LLM 生成记忆总结
    try:
        res = memory_summry_model.invoke(messages)
        text = str(res.content).strip() if res.content else ""
    except Exception:
        logger.warning("记忆总结 LLM 调用失败")
        return {}

    if not text:
        return {}

    # 解析 JSON 提取 title / summary
    title, summary = _parse_memory_json(text)
    if not title or not summary:
        logger.warning("记忆总结 JSON 解析失败，原始输出前200字: %s", text[:200])
        return {}

    # 捕获当前心理状态快照
    internal = _ensure_array(state.get("internal_state"))
    relationship = _ensure_array(state.get("relationship_state"))
    surface = _ensure_array(state.get("surface_state"))

    # 计算文本嵌入
    from memory import compute_embedding
    embedding = compute_embedding(summary)

    # 构建并保存记忆
    node = MemoryNode.from_state_vectors(
        title=title,
        content=summary,
        internal_state=internal,
        relationship_state=relationship,
        surface_state=surface,
        embedding=embedding,
    )

    memory_id = state.get("memory_id") or "main"
    MemoryStore(memory_id=memory_id).add(node)

    logger.info("记忆已保存: %s", title)
    return {}



