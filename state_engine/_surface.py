"""Surface Projection —— 内部状态 + 外表情刺激 → 表面表达。

SurfaceState 每轮从内部状态 + outer_stimuli 动态投影。
表面表达 = 内部状态基线 + 外表情刺激影响 + 特质修饰

关键：外表情刺激是被压抑后的版本（由 apply_defenses 的 deactivation 输出）
  - validation 被压抑 → 表面"温度"自然降低
  - conflict 被压抑 → 表面"尖锐度"自然降低
"""

import numpy as np
from state import (
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_MENTAL_FATIGUE,
    R_AFFECTION, R_TRUST_BOND,
    S_EXPRESSIVENESS, S_WARMTH, S_SHARPNESS, S_SOFTNESS,
    S_ENTHUSIASM, S_RESTRAINT, S_VULNERABILITY, S_SIZE,
    T_PRIDE, T_EMOTIONAL_OPENNESS, T_OPTIMISM,
    ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT,
    ST_DEPENDENCY, ST_TEASING, ST_EMOTIONAL_WEIGHT,
)
from ._utils import soft_clamp, _sigmoid


def project_surface(
    internal: np.ndarray,
    relationship: np.ndarray,
    traits: np.ndarray,
    outer_stimuli: np.ndarray,
) -> np.ndarray:
    """④ 表面投影：内部状态 + outer_stimuli → 表面表达（动态计算）。"""
    s = np.zeros(S_SIZE, dtype=np.float64)

    # 内部状态基线（值域 [-1, 1]）
    # stress 分散出口：只影响 warmth(-0.15)/sharpness(+0.15)/restraint(+0.20)，
    # 不再三路叠加 warmth↓+sharpness↑+softness↓。restraint 是中性出口（克制≠负面）。
    # softness 改为由 emotional_safety 单独主导。
    # energy 与 fatigue 的表面对冲减弱：fatigue 系数从 0.30 降至 0.15，
    # 避免等高对冲掩盖 energy 的正向表达。
    s[S_EXPRESSIVENESS] = -0.3 + internal[I_ENERGY] * 0.4 - internal[I_MENTAL_FATIGUE] * 0.15
    s[S_WARMTH]         = -0.2 + relationship[R_AFFECTION] * 0.4 - internal[I_STRESS] * 0.15
    s[S_SHARPNESS]      = -0.1 + internal[I_IRRITATION] * 0.5 + internal[I_STRESS] * 0.15
    s[S_SOFTNESS]       = -0.1 + relationship[R_TRUST_BOND] * 0.2   # trust+safety 合并
    s[S_ENTHUSIASM]     = -0.2 + internal[I_ENERGY] * 0.5 - internal[I_MENTAL_FATIGUE] * 0.15
    s[S_RESTRAINT]      = -0.1 + internal[I_INSECURITY] * 0.3 + traits[T_PRIDE] * 0.2 + internal[I_STRESS] * 0.20
    s[S_VULNERABILITY]  = -0.5 + internal[I_LONELINESS] * 0.3 + internal[I_LONGING] * 0.2 - traits[T_PRIDE] * 0.2

    # 外表情刺激的直接影响（被压抑后的版本，stimuli 仍在 [0, 1]）
    s[S_WARMTH]         += outer_stimuli[ST_VALIDATION]  * 0.30
    s[S_SHARPNESS]      += outer_stimuli[ST_CONFLICT]    * 0.25
    s[S_SOFTNESS]       += outer_stimuli[ST_CLOSENESS]   * 0.20
    s[S_VULNERABILITY]  += outer_stimuli[ST_ABANDONMENT] * 0.15
    s[S_RESTRAINT]      += outer_stimuli[ST_EMOTIONAL_WEIGHT] * 0.20
    s[S_SHARPNESS]      += outer_stimuli[ST_TEASING]     * 0.10
    s[S_WARMTH]         += outer_stimuli[ST_DEPENDENCY]  * 0.10
    s[S_ENTHUSIASM]     += outer_stimuli[ST_VALIDATION]  * 0.15  # 被认可→活力上升

    # 特质修饰 — 连续贡献（sigmoid 软阈值，traits 已以 0 为中心）
    pride_active = _sigmoid(traits[T_PRIDE] / 0.30)
    s[S_SHARPNESS]     += pride_active * traits[T_PRIDE] * 0.10
    s[S_VULNERABILITY] -= pride_active * traits[T_PRIDE] * 0.15

    openness_active = _sigmoid(traits[T_EMOTIONAL_OPENNESS] / 0.30)
    s[S_EXPRESSIVENESS] += openness_active * traits[T_EMOTIONAL_OPENNESS] * 0.10
    s[S_RESTRAINT]      -= openness_active * traits[T_EMOTIONAL_OPENNESS] * 0.10

    optimism_active = _sigmoid(traits[T_OPTIMISM] / 0.30)
    s[S_ENTHUSIASM]    += optimism_active * traits[T_OPTIMISM] * 0.10

    return soft_clamp(s, -1.0, 1.0)
