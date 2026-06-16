"""model —— LLM 模型单例初始化。"""
import os
import dotenv
from langchain.chat_models import init_chat_model

dotenv.load_dotenv()

model = init_chat_model(
    model="deepseek-v4-pro",
    model_provider="deepseek",
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
)

# perception_model = init_chat_model(
#     model="qwen2.5:7b",
#     model_provider="ollama",
#     base_url="http://localhost:11434",
#     api_key="",
# )

preception_model = model


if __name__ == "__main__":
    print(model.invoke("你好，请简短介绍你自己"))