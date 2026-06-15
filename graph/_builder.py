"""图构建器 —— 注册节点、连线、编译。"""

import logging
from langgraph.graph import StateGraph, START, END
from state import State
from nodes import (
    inject_system_node,
    perception_node,
    state_engine_node,
    state_formatter_node,
    llm_node,
)
from ._routing import route_after_start, route_after_perception

logger = logging.getLogger(__name__)

# ── 构建图 ──
graph_builder = StateGraph(State)

graph_builder.add_node("inject_system", inject_system_node)
graph_builder.add_node("perception", perception_node)
graph_builder.add_node("state_engine", state_engine_node)
graph_builder.add_node("state_formatter", state_formatter_node)
graph_builder.add_node("llm", llm_node)

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
graph_builder.add_edge("state_engine", "state_formatter")
graph_builder.add_edge("state_formatter", "llm")
graph_builder.add_edge("llm", END)

compiled_graph = graph_builder.compile()
