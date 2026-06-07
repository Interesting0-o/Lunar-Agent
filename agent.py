"""
agent —— LangGraph 图定义与编译

职责：注册节点 → 连线 → 编译导出。
模型初始化在 model.py，节点实现在 nodes.py，
感知逻辑在 perception.py，状态引擎在 state_engine.py。
"""

import sqlite3
import logging
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from state import State
from nodes import (
    inject_system_node,
    perception_node,
    state_engine_node,
    llm_node,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── 构建图 ──
graph_builder = StateGraph(State)

graph_builder.add_node("inject_system", inject_system_node)
graph_builder.add_node("perception", perception_node)
graph_builder.add_node("state_engine", state_engine_node)
graph_builder.add_node("llm", llm_node)


# ── 路由函数 ──

def route_after_start(state: State) -> str:
    """首次运行先注入系统提示词，之后直接走感知。"""
    if state.get("has_inject_system_prompt"):
        return "perception"
    return "inject_system"


def route_after_perception(state: State) -> str:
    """感知成功 → 状态引擎 → LLM；感知失败 → 结束本轮。"""
    if state.get("error"):
        return "end"
    return "state_engine"


# ── 连线 ──
graph_builder.add_conditional_edges(
    START,
    route_after_start,
    {"inject_system": "inject_system", "perception": "perception"},
)
graph_builder.add_edge("inject_system", "perception")
graph_builder.add_conditional_edges(
    "perception",
    route_after_perception,
    {"state_engine": "state_engine", "end": END},
)
graph_builder.add_edge("state_engine", "llm")
graph_builder.add_edge("llm", END)

compiled_graph = graph_builder.compile()


if __name__ == "__main__":
    connection = sqlite3.connect("./db/luna.db", check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
