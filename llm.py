"""model —— LLM 模型单例初始化。"""
import os
import dotenv
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings

dotenv.load_dotenv()

# Ollama 默认地址：优先读环境变量，fallback 到 WSL 可访问的 Windows 宿主机
_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://172.18.96.1:11434")

model = init_chat_model(
    model="deepseek-v4-pro",
    model_provider="deepseek",
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
)

perception_model = init_chat_model(
    model="qwen2.5:7b",
    model_provider="ollama",
    base_url=_OLLAMA_URL,
    api_key="",
)

memory_summry_model = init_chat_model(
    model="qwen2.5:7b",
    model_provider="ollama",
    base_url=_OLLAMA_URL,
    api_key="",
)

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:8b", 
    base_url=_OLLAMA_URL, 
)