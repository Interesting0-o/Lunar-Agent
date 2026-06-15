"""② Internal Dynamics + ⑤ Relationship Dynamics —— 状态更新核心。

内部状态: LSTM 式 3 门控更新（快速变量）
  h_t = f ⊙ h_{t-1} + i ⊙ (A·h_{t-1} + B·e_t) + g ⊙ bias

关系状态: LTI 更新（超慢变量）
  rel_t = A_rel · rel_{t-1} + B_rel · e_t
"""

import numpy as np
from state import (
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE,
    T_SENSITIVITY, T_PRIDE, T_EMOTIONAL_OPENNESS, T_EMOTIONAL_STABILITY,
    T_OPTIMISM, T_ANXIETY_PRONENESS, T_ANGER_REACTIVITY,
    T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE,
    R_AFFECTION, R_TRUST, R_EMOTIONAL_SAFETY,
    G_SUPPRESSION, G_ATTACHMENT,
)
from ._utils import soft_clamp, _sigmoid_gate
from ._matrices import (
    STATE_COUPLING_A, INPUT_INFLUENCE_B, PERSONALITY_BIAS_C,
    REL_STATE_COUPLING_A, REL_INPUT_INFLUENCE_B,
)


def update_internal_dynamics(
    current: np.ndarray,
    gated_stimuli: np.ndarray,
    traits: np.ndarray,
    relationship: np.ndarray,
    gates: np.ndarray,
) -> np.ndarray:
    """② 内部动力系统——LSTM 式 3 门控更新。

    主公式:
        h_t = f ⊙ h_{t-1} + i ⊙ (A·h_{t-1} + B·e_t) + g ⊙ bias
    三个门控由显式心理学变量构造，sigmoid 软阈值激活。

    调参经验：同一特质不要在多个门控中"对冲"
      - 情绪稳定主导 f_gate（遗忘），不在 i_gate 中再抑制
      - 衰减项作为 f_gate 的"后盾"，增强稳定者真的能"放下"的能力

    参数:
      current:       当前内部状态 h_{t-1} (8 维)
      gated_stimuli: 已被门控调制过的"里表情"刺激 e_t (7 维)
      traits:        角色特质 (10 维)
      relationship:  当前关系状态 (6 维)——用于 g_gate 构造
      gates:         三门控值 (3 维)
    返回:
      更新后的内部状态 h_t (8 维)
    """
    # LTI 风格的"建议值"
    coupling = STATE_COUPLING_A @ current
    influence = gated_stimuli @ INPUT_INFLUENCE_B
    raw_dynamics = coupling + influence

    # 遗忘门 f：高情绪稳定+高乐观 → 强；高依恋焦虑+高敏感 → 弱
    f_signal = (
        traits[T_EMOTIONAL_STABILITY] * 0.6
        + traits[T_OPTIMISM] * 0.3
        - traits[T_ATTACHMENT_ANXIETY] * 0.3
        - traits[T_SENSITIVITY] * 0.3
    )
    f_gate = _sigmoid_gate(f_signal + 0.3)

    # 接受门 i：高压抑 → 弱；高依恋敏感+高信任 → 强；高开放 → 强
    i_signal = (
        -gates[G_SUPPRESSION] * 0.4
        + gates[G_ATTACHMENT] * 0.2
        + relationship[R_TRUST] * 0.2
        + relationship[R_EMOTIONAL_SAFETY] * 0.2
        + (traits[T_EMOTIONAL_OPENNESS] - 0.5) * 0.3
    )
    i_gate = _sigmoid_gate(i_signal + 0.3)

    # 自生门 g：关系好 → 内部生正面；高焦虑 → 内部生压力
    g_signal = (
        (relationship[R_AFFECTION] - 0.5) * 0.4
        + (relationship[R_TRUST] - 0.5) * 0.3
        - (1.0 - relationship[R_TRUST]) * 0.2
        - traits[T_ANXIETY_PRONENESS] * 0.2
        + traits[T_OPTIMISM] * 0.2
    )
    g_gate = _sigmoid_gate(g_signal + 0.5)

    # 人格偏置（per-tick baseline）
    bias = PERSONALITY_BIAS_C.copy()
    bias[I_ENERGY]     += (traits[T_OPTIMISM] - 0.5) * 0.02
    bias[I_STRESS]     -= (traits[T_OPTIMISM] - 0.5) * 0.01
    bias[I_STRESS]     += (traits[T_ANXIETY_PRONENESS] - 0.5) * 0.02
    bias[I_INSECURITY] += (traits[T_ANXIETY_PRONENESS] - 0.5) * 0.01

    # LSTM 式 3 门控更新
    new_state = (
        f_gate * current          # 遗忘：保留多少旧里表情
        + i_gate * raw_dynamics   # 接受：新刺激进入多少
        + g_gate * bias           # 自生：内心基线偏移
    )
    return soft_clamp(new_state, 0.0, 1.0)


def update_relationship_dynamics(
    current: np.ndarray,
    gated_stimuli: np.ndarray,
) -> np.ndarray:
    """⑤ 关系动力系统：LTI 风格（不 LSTM 化，关系是超慢变量）。"""
    coupling = REL_STATE_COUPLING_A @ current
    influence = gated_stimuli @ REL_INPUT_INFLUENCE_B
    new_state = coupling + influence
    return soft_clamp(new_state, 0.0, 1.0)
