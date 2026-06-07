"""
nodes —— LangGraph 图节点函数

所有图节点实现在此，agent.py 只负责注册和连线。
"""

import config
import logging
from langchain.messages import SystemMessage
from model import model
from state import State
from perception import extract_recent_context, call_perception_with_retry
from state_engine import update_all

logger = logging.getLogger(__name__)


def inject_system_node(state: State) -> dict:
    """首次运行：注入角色系统提示词。"""
    return {
        "messages": [SystemMessage(content=config.SYSTEM_PROMPT)],
        "has_inject_system_prompt": True,
    }


def perception_node(state: State) -> dict:
    """感知节点：分析用户输入的社交意义。

    失败时设置 error=True，条件边据此引导到 END。
    """
    cfg = config.PERCEPTION_CONFIG
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

    result = update_all(
        current_internal=state.get("internal_state"),
        current_relationship=state.get("relationship_state"),
        current_hidden=state.get("hidden_state"),
        traits=state["traits"],
        signals=state["user_signals"],
        impact=state["user_interaction_impact"],
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
