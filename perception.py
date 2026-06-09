"""
perception —— 感知节点工具集

职责：提取对话上下文、调用模型进行社交意义理解、验证并重试。
所有与 LLM 调用相关的感知逻辑集中于此，不涉及状态更新。

perception_node 的输出格式：
  {
    "user_signals": np.ndarray,              # 9 维 SocialSignals 数组
    "user_interaction_impact": np.ndarray,   # 4 维 InteractionImpact 数组
  }
"""

import json
import logging
import numpy as np
from perception_prompt import PERCEPTION_SYSTEM_PROMPT
from typing import Optional
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from model import perception_model
from state import SS_LABELS, II_LABELS, signals_from_dict, impact_from_dict

logger = logging.getLogger(__name__)


def extract_recent_context(messages: list, max_messages: int) -> list:
    """提取最近 N 条 Human/AI 消息作为感知上下文，跳过 SystemMessage。"""
    relevant = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    recent = relevant[-max_messages:] if len(relevant) > max_messages else relevant
    return recent if recent else messages[-1:]


def validate_perception_result(data: dict) -> bool:
    """验证反序列化结果是否包含 user_signals 和 user_interaction_impact 的所有字段。

    注意：这里验证的是原始 JSON 字典，验证通过后由
    call_perception_with_retry 转换为 numpy 数组。
    """
    if not isinstance(data, dict):
        return False
    signals = data.get("user_signals")
    impact = data.get("user_interaction_impact")
    if not isinstance(signals, dict) or not isinstance(impact, dict):
        return False
    for key in SS_LABELS:
        if key not in signals or not isinstance(signals[key], (int, float)):
            return False
    for key in II_LABELS:
        if key not in impact or not isinstance(impact[key], (int, float)):
            return False
    return True


def call_perception_with_retry(user_context: list, cfg: dict) -> Optional[dict]:
    """调用感知模型并自动重试。

    成功返回（numpy 数组格式）：
      {
        "user_signals": np.ndarray,              # 9 维，用 SS_* 索引
        "user_interaction_impact": np.ndarray,   # 4 维，用 II_* 索引
      }

    全部失败返回 None，由调用方设置 error=True。
    """
    system_prompt = PERCEPTION_SYSTEM_PROMPT
    max_attempts = cfg["max_retries"]
    emphases = cfg["retry_emphases"]
    last_error = None

    for attempt in range(max_attempts):
        emphasis = emphases[attempt] if attempt < len(emphases) else emphases[-1]

        # 合并 emphasis 到 system prompt 中，避免多条 SystemMessage
        combined_prompt = system_prompt + ("\n\n" + emphasis if emphasis else "")
        msgs = [SystemMessage(content=combined_prompt)]
        msgs.extend(user_context)

        try:
            raw = perception_model.invoke(msgs)
            text = raw.content.strip()  # type: ignore

            # 抹掉 ```json ... ``` 包裹
            if text.startswith("```"):
                text = text.strip("`").strip()
                if text.startswith("json"):
                    text = text[4:].strip()

            data = json.loads(text)

            if validate_perception_result(data):
                # 将模型输出的键值对字典转换为 numpy 数组
                return {
                    "user_signals": signals_from_dict(data["user_signals"]),
                    "user_interaction_impact": impact_from_dict(data["user_interaction_impact"]),
                }

            last_error = (
                f"字段缺失或类型错误: {list(data.keys()) if isinstance(data, dict) else type(data)}"
            )

        except (json.JSONDecodeError, Exception) as e:
            last_error = str(e)
            logger.warning("perception 第 %d 次尝试失败: %s", attempt + 1, last_error)

    logger.error("perception 全部 %d 次尝试失败，跳过本轮。最后错误: %s", max_attempts, last_error)
    return None
