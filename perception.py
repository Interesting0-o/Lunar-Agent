"""
perception —— 感知节点工具集

职责：提取对话上下文、调用模型进行心理刺激提取、验证并重试。
所有与 LLM 调用相关的感知逻辑集中于此，不涉及状态更新。

perception_node 的输出格式：
  {
    "user_stimuli": np.ndarray,     # 7 维 StimulusVector 数组
    "stimulus_metadata": dict,      # StimulusMetadata 的 JSON 兼容表示（约束②）
  }
"""

import json
import logging
import numpy as np
from prompts import PERCEPTION_SYSTEM_PROMPT
from typing import Optional
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from llm import perception_model
from state import ST_LABELS, stimuli_from_dict, StimulusMetadata

logger = logging.getLogger(__name__)


def extract_recent_context(messages: list, max_messages: int) -> list:
    """提取最近 N 条 Human/AI 消息作为感知上下文，跳过 SystemMessage。"""
    relevant = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    recent = relevant[-max_messages:] if len(relevant) > max_messages else relevant
    return recent if recent else messages[-1:]


def validate_perception_result(data: dict) -> bool:
    """验证反序列化结果是否包含 user_stimuli 的所有字段。

    注意：这里验证的是原始 JSON 字典，验证通过后由
    call_perception_with_retry 转换为 numpy 数组。
    """
    if not isinstance(data, dict):
        return False
    stimuli = data.get("user_stimuli")
    if not isinstance(stimuli, dict):
        return False
    for key in ST_LABELS:
        if key not in stimuli or not isinstance(stimuli[key], (int, float)):
            return False
    return True


def call_perception_with_retry(
    user_context: list,
    cfg: dict,
    internal_state: Optional[np.ndarray] = None,
    relationship_state: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """调用感知模型并自动重试。

    成功返回（numpy 数组格式）：
      {
        "user_stimuli": np.ndarray,  # 7 维，用 ST_* 索引
      }

    全部失败返回 None，由调用方设置 error=True。

    Args:
        user_context: 对话上下文消息列表
        cfg: 感知配置字典（max_retries, context_window, retry_emphases）
        internal_state: 当前内部状态 (8,)，用于注入状态上下文
        relationship_state: 当前关系状态 (3,)，用于注入状态上下文
    """
    system_prompt = PERCEPTION_SYSTEM_PROMPT

    # 注入状态上下文辅助感知判断
    if internal_state is not None or relationship_state is not None:
        state_note = "\n\n## 当前角色状态（影响刺激解读）\n"
        if internal_state is not None:
            # 将 8 维内部状态转为简洁文本
            i_labels = ["energy", "stress", "loneliness", "insecurity",
                        "irritation", "longing", "social_battery", "mental_fatigue"]
            i_summary = " | ".join(f"{lbl}={internal_state[i]:+.2f}"
                                   for i, lbl in enumerate(i_labels))
            state_note += f"内部状态: {i_summary}\n"
        if relationship_state is not None:
            r_labels = ["affection", "trust_bond", "intimacy"]
            r_summary = " | ".join(f"{lbl}={relationship_state[i]:+.2f}"
                                   for i, lbl in enumerate(r_labels))
            state_note += f"关系状态: {r_summary}\n"
        state_note += ("当前状态会影响角色对同一句话的感受："
                       "高压力时更易感到被抛弃/被攻击，高好感时更易感到被认可/被靠近。\n")
        system_prompt = system_prompt.rstrip() + state_note
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

            # 统一剥除 ```json ... ``` 包裹
            from state import _strip_json_fence
            text = _strip_json_fence(text)

            data = json.loads(text)

            if validate_perception_result(data):
                # 将模型输出的键值对字典转换为 numpy 数组
                stimuli_array = stimuli_from_dict(data["user_stimuli"])

                # 构建 StimulusMetadata（约束②）
                confidence = _compute_confidence(attempt, max_attempts)
                # 低置信度时，按维度幅度微调：极端值（≈0或1）的置信度略降
                # 因为 LLM 在打极端分时更不可靠
                per_dim_confidence = np.full(ST_SIZE, confidence, dtype=np.float64)
                extreme_mask = (stimuli_array < 0.05) | (stimuli_array > 0.95)
                per_dim_confidence[extreme_mask] *= 0.85  # 极端值再降15%

                metadata = StimulusMetadata(
                    confidence=per_dim_confidence,
                    source=np.zeros(ST_SIZE, dtype=np.int8),
                    decay_modulator=np.ones(ST_SIZE, dtype=np.float64),
                    timestamp=raw.response_metadata.get("created", None) if hasattr(raw, "response_metadata") else None,
                )

                return {
                    "user_stimuli": stimuli_array,
                    "stimulus_metadata": {
                        "confidence": metadata.confidence.tolist(),
                        "source": metadata.source.tolist(),
                        "decay_modulator": metadata.decay_modulator.tolist(),
                        "timestamp": metadata.timestamp or __import__("time").time(),
                    },
                }

            last_error = (
                f"字段缺失或类型错误: {list(data.keys()) if isinstance(data, dict) else type(data)}"
            )

        except (json.JSONDecodeError, Exception) as e:
            last_error = str(e)
            logger.warning("perception 第 %d 次尝试失败: %s", attempt + 1, last_error)

    logger.error("perception 全部 %d 次尝试失败，跳过本轮。最后错误: %s", max_attempts, last_error)
    return None


def _compute_confidence(attempt: int, max_attempts: int) -> float:
    """根据重试次数计算感知置信度。

    第 0 次尝试（首次成功）→ 0.90
    第 1 次重试成功    → 0.70
    第 2+ 次重试成功   → 0.50

    这反映一个直觉：LLM 第一次就能正确提取时最可信，
    需要多次重试才成功意味着输入模糊或模型不确定。
    """
    if attempt == 0:
        return 0.90
    elif attempt == 1:
        return 0.70
    else:
        return 0.50
