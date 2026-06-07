"""
perception —— 感知节点工具集

职责：提取对话上下文、调用模型进行社交意义理解、验证并重试。
所有与 LLM 调用相关的感知逻辑集中于此，不涉及状态更新。
"""

import json
import logging
import config
from typing import Optional
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from model import model

logger = logging.getLogger(__name__)


# ── SocialSignals 所有必需字段 ──
REQUIRED_SIGNAL_KEYS = [
    "affection_signal", "attention_signal", "intimacy_signal", "approval_signal",
    "rejection_signal", "abandonment_signal",
    "dependency_signal", "teasing_signal", "conflict_signal",
]

REQUIRED_IMPACT_KEYS = [
    "emotional_weight", "memorability",
    "trust_impact", "closeness_impact",
]


def extract_recent_context(messages: list, max_messages: int) -> list:
    """提取最近 N 条 Human/AI 消息作为感知上下文，跳过 SystemMessage。"""
    relevant = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    recent = relevant[-max_messages:] if len(relevant) > max_messages else relevant
    return recent if recent else messages[-1:]


def validate_perception_result(data: dict) -> bool:
    """验证反序列化结果是否包含 user_signals 和 user_interaction_impact 的所有字段。"""
    if not isinstance(data, dict):
        return False
    signals = data.get("user_signals")
    impact = data.get("user_interaction_impact")
    if not isinstance(signals, dict) or not isinstance(impact, dict):
        return False
    for key in REQUIRED_SIGNAL_KEYS:
        if key not in signals or not isinstance(signals[key], (int, float)):
            return False
    for key in REQUIRED_IMPACT_KEYS:
        if key not in impact or not isinstance(impact[key], (int, float)):
            return False
    return True


def call_perception_with_retry(user_context: list, cfg: dict) -> Optional[dict]:
    """调用感知模型并自动重试。全部失败返回 None，由调用方设置 error=True。"""
    system_prompt = config.PERCEPTION_SYSTEM_PROMPT
    max_attempts = cfg["max_retries"]
    emphases = cfg["retry_emphases"]
    last_error = None

    for attempt in range(max_attempts):
        emphasis = emphases[attempt] if attempt < len(emphases) else emphases[-1]

        msgs = [SystemMessage(content=system_prompt)]
        if emphasis:
            msgs.append(SystemMessage(content=emphasis))
        msgs.extend(user_context)

        try:
            raw = model.invoke(msgs)
            text = raw.content.strip()

            # 抹掉 ```json ... ``` 包裹
            if text.startswith("```"):
                text = text.strip("`").strip()
                if text.startswith("json"):
                    text = text[4:].strip()

            data = json.loads(text)

            if validate_perception_result(data):
                return data

            last_error = (
                f"字段缺失或类型错误: {list(data.keys()) if isinstance(data, dict) else type(data)}"
            )

        except (json.JSONDecodeError, Exception) as e:
            last_error = str(e)
            logger.warning("perception 第 %d 次尝试失败: %s", attempt + 1, last_error)

    logger.error("perception 全部 %d 次尝试失败，跳过本轮。最后错误: %s", max_attempts, last_error)
    return None
