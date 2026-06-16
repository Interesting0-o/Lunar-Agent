"""
agent —— TUI 交互入口

图定义与编译已移入 graph/ 包，本文件只负责终端交互。
模型初始化在 llm.py，节点实现在 nodes.py，
感知逻辑在 perception.py，状态引擎在 state_engine/。
"""
import dotenv
import sqlite3
import logging
from graph import compiled_graph, graph_builder

dotenv.load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def tui_test():
    import json
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langchain.messages import HumanMessage
    connection = sqlite3.connect("./db/lunar.db", check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()

    check_graph = graph_builder.compile(saver)

    config = {"configurable": {"thread_id": "useasdr_123"}}

    state = check_graph.get_state(config)  # type: ignore
    print(state)
    with open("test.json", "r") as f:
        test_data = json.load(f)

    is_inject_system = False

    # 打印已有对话历史
    if state and state.values:
        for msg in state.values.get("messages", []):
            if hasattr(msg, "type") and hasattr(msg, "content"):
                if msg.type == "human":
                    print(f"[User]: {msg.content}")
                elif msg.type == "ai":
                    print(f"[Lunar]: {msg.content}")

    while True:
        print("[User]:", end="")
        user_input = input()
        print()

        if user_input in ["quit", "exit", "q"]:
            break

        if not is_inject_system:
            test_data["messages"].append(HumanMessage(content=user_input))
            # 移除 has_inject_system_prompt 让 router 走 inject_system_node
            # （否则首跳跳过注入角色 system prompt → LLM 无角色上下文 → 性格崩坏）
            test_data.pop("has_inject_system_prompt", None)
            res = check_graph.stream(test_data, config)  # type: ignore
            is_inject_system = True

        else:
            res = check_graph.stream({"messages": [HumanMessage(content=user_input)]}, config)  # type: ignore

        print("[Lunar]:", end="")
        for chunk in res:
            if isinstance(chunk, dict):
                for val in chunk.values():
                    if isinstance(val, dict):
                        for msg in val.get("messages", []):
                            if hasattr(msg, "type") and msg.type == "ai":
                                print(msg.content, end="")
        print()


if __name__ == "__main__":
    tui_test()
