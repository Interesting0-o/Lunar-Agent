"""
nodes —— LangGraph 图节点函数

所有图节点实现在此，agent.py 只负责注册和连线。

注意：state 中存储的 np.ndarray 在通过 JSON 序列化/反序列化
（如 test.json 加载、SQLite checkpoint 恢复）后可能变为 Python list，
所有 node 函数在读取 state 时负责确保类型正确。
"""

import numpy as np
from character_prompt import SYSTEM_PROMPT
from config import PERCEPTION_CONFIG
import logging
from langchain.messages import SystemMessage
from model import model
from state import State
from perception import extract_recent_context, call_perception_with_retry
from state_engine import update_all, DEFAULT_TRAITS

logger = logging.getLogger(__name__)


def _ensure_array(v, dtype=np.float64) -> np.ndarray:
    """确保值为 numpy 数组（兼容 Python list/json 反序列化）。"""
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v
    return np.asarray(v, dtype=dtype)


def inject_system_node(state: State) -> dict:
    """首次运行：注入角色系统提示词 + 默认人格特质（如果状态中还没有）。"""
    result = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT)],
        "has_inject_system_prompt": True,
    }
    # 如果 state 已有 traits（来自 test.json），不覆盖
    if not state.get("traits"):
        result["traits"] = DEFAULT_TRAITS
    return result


def perception_node(state: State) -> dict:
    """感知节点：分析用户输入的社交意义。

    失败时设置 error=True，条件边据此引导到 END。
    """
    cfg = PERCEPTION_CONFIG
    context = extract_recent_context(state["messages"], cfg["context_window"])
    result = call_perception_with_retry(context, cfg)

    if result is None:
        return {"error": True}

    return {
        "user_signals": result["user_signals"],
        "user_interaction_impact": result["user_interaction_impact"],
        "error": False,
    }


def state_engine_node(state: State) -> dict:
    """状态引擎节点：根据感知输出 + 当前状态 + Traits 更新所有状态层。

    如果 perception 已标记 error，则跳过本轮状态更新。
    triggered_events 仅用于日志，不写入 State。
    """
    if state.get("error"):
        logger.info("state_engine 跳过本轮（perception 已标记 error）")
        return {}

    # ── 确保所有数值状态为 numpy 数组（兼容 json/list 反序列化） ──
    traits = _ensure_array(state.get("traits"))
    signals = _ensure_array(state.get("user_signals"))
    impact = _ensure_array(state.get("user_interaction_impact"))

    result = update_all(
        current_internal=_ensure_array(state.get("internal_state")),
        current_relationship=_ensure_array(state.get("relationship_state")),
        current_hidden=_ensure_array(state.get("hidden_state")),
        traits=traits,
        signals=signals,
        impact=impact,
    )

    # triggered_events 仅日志，不返回给图（State 无此字段）
    events = result.pop("triggered_events", [])
    if events:
        logger.info("State Engine 事件: %s", events)

    return result


def llm_node(state: State) -> dict:
    """LLM 回复节点：用消息历史调用模型生成回复。"""
    messages = state["messages"]
    res = model.invoke(messages)
    return {"messages": [res]}
