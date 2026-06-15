"""图路由函数 —— 条件边的决策逻辑。"""

from state import State


def route_after_start(state: State) -> str:
    """首次运行先注入系统提示词，之后直接走感知。"""
    if state.get("has_inject_system_prompt"):
        return "perception"
    return "inject_system"


def route_after_perception(state: State) -> str:
    """感知成功 → 状态引擎；感知失败 → 结束本轮。"""
    if state.get("error"):
        return "end"
    return "state_engine"
