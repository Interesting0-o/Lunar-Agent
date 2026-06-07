import os
import dotenv
import config
import sqlite3
from typing import TypedDict, List, Optional, Annotated
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from state import State
dotenv.load_dotenv()

model = init_chat_model(
    model="qwen2.5:7b",
    model_provider="ollama",
    base_url="http://localhost:11434",
    api_key="",
)

perception_model = init_chat_model(
    model="qwen2.5:7b",
    model_provider="ollama",
    base_url="http://localhost:11434",
    api_key="",
)


graph_builder = StateGraph(State)


def inject_system_node(state: State):
    return {"messages": [SystemMessage(content=config.SYSTEM_PROMPT)], "has_inject_system_prompt": True}


def llm_node(state: State):
    messages = state["messages"]
    res = model.invoke(messages)
    return {"messages": [res]}


def route_after_start(state: State) -> str:
    """如果 has_inject_system_prompt 为 True 表示已注入，否则先注入。"""
    if state.get("has_inject_system_prompt"):
        return "llm"
    return "inject_system"

def perception_node(state: State):
    user_input = state["messages"][-1]


    res = model.invoke([
        SystemMessage(content=config.PERCEPTION_SYSTEM_PROMPT),
        user_input
    ])


def system_node(state: State):
    pass




graph_builder.add_node("inject_system", inject_system_node)
graph_builder.add_node("llm", llm_node)
graph_builder.add_node("perception",perception_node)


graph_builder.add_conditional_edges(
    START,
    route_after_start,
    {"inject_system": "inject_system", "llm": "llm"},
)
graph_builder.add_edge("inject_system", "llm")
graph_builder.add_edge("llm", END)

compiled_graph = graph_builder.compile()


if __name__ == "__main__":
    connection = sqlite3.connect("./db/luna.db", check_same_thread=False)
    sql_saver = SqliteSaver(connection)
    sql_saver.setup()

