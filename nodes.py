"""
nodes —— LangGraph 图节点函数

所有图节点实现在此，agent.py 只负责注册和连线。

注意：state 中存储的 np.ndarray 在通过 JSON 序列化/反序列化
（如 test.json 加载、SQLite checkpoint 恢复）后可能变为 Python list，
所有 node 函数在读取 state 时负责确保类型正确。
"""

import numpy as np
from prompts import SYSTEM_PROMPT
from config import PERCEPTION_CONFIG
import logging
from langchain.messages import SystemMessage
from llm import model
from state import State, DEFAULT_TRAITS
from perception import extract_recent_context, call_perception_with_retry
from state_engine import update_all
from state_formatter import format_state_for_node

logger = logging.getLogger(__name__)


def _ensure_array(v, dtype=np.float64) -> np.ndarray:
    """确保值为 numpy 数组（兼容 Python list/json 反序列化）。"""
    if v is None:
        return None#type:ignore
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
    """感知节点：从用户输入中直接提取心理刺激强度。

    将 user_stimuli 写入图 State，state_engine_node 消费后会立即清理。
    失败时设置 error=True，条件边据此引导到 END。
    """
    cfg = PERCEPTION_CONFIG
    context = extract_recent_context(state["messages"], cfg["context_window"])
    result = call_perception_with_retry(context, cfg)

    if result is None:
        return {"error": True}

    return {
        "user_stimuli": result["user_stimuli"],
        "error": False,
    }


def state_engine_node(state: State) -> dict:
    """状态引擎节点：根据感知输出的心理刺激 + 当前状态 + Traits 更新所有状态层。

    读取 state 中的 user_stimuli 后将其置为 None，避免 checkpoint 残留。
    """
    if state.get("error"):
        logger.info("state_engine 跳过本轮（perception 已标记 error）")
        return {}

    # ── 确保所有数值状态为 numpy 数组（兼容 json/list 反序列化） ──
    traits = _ensure_array(state.get("traits"))
    stimuli = _ensure_array(state.get("user_stimuli"))

    result = update_all(
        current_internal=_ensure_array(state.get("internal_state")),
        current_relationship=_ensure_array(state.get("relationship_state")),
        traits=traits,
        stimuli=stimuli,
    )

    # 消费后清理中间数据，避免 checkpoint 残留
    result["user_stimuli"] = None

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
        # 将状态描述作为 SystemMessage 注入（不持久化到 state.messages）
        inject_msg = SystemMessage(content=state_desc)
        res = model.invoke([inject_msg] + messages)
    else:
        res = model.invoke(messages)

    return {"messages": [res]}
