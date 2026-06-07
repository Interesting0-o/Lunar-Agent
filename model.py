"""model —— LLM 模型单例初始化。"""

import dotenv
from langchain.chat_models import init_chat_model

dotenv.load_dotenv()

model = init_chat_model(
    model="qwen2.5:7b",
    model_provider="ollama",
    base_url="http://localhost:11434",
    api_key="",
)
