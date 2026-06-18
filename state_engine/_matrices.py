"""输入影响矩阵 B —— 心理刺激 → 状态维度的线性映射。

所有 B 矩阵集中于此，便于后续外置化为 JSON/YAML 配置。

跨维度耦合已迁移至 _dynamics.py 中的显式命名规则（替代旧 A 矩阵）。
"""

import numpy as np
from state import (
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE, I_SIZE,
    R_AFFECTION, R_TRUST, R_FAMILIARITY, R_DEPENDENCY,
    R_EMOTIONAL_SAFETY, R_ROMANTIC_TENSION, R_SIZE,
    ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT,
    ST_DEPENDENCY, ST_TEASING, ST_EMOTIONAL_WEIGHT, ST_SIZE,
)


def _build_input_influence() -> np.ndarray:
    """输入影响矩阵 B（ST_SIZE × I_SIZE）：B[s, i] = 刺激 s 对状态 i 的影响权重。"""
    B = np.zeros((ST_SIZE, I_SIZE), dtype=np.float64)

    # abandoned → insecurity↑, loneliness↑, stress↑, longing↑, energy↓
    B[ST_ABANDONMENT, I_INSECURITY] = 0.25
    B[ST_ABANDONMENT, I_LONELINESS] = 0.18
    B[ST_ABANDONMENT, I_STRESS] = 0.12
    B[ST_ABANDONMENT, I_LONGING] = 0.18
    B[ST_ABANDONMENT, I_ENERGY] = -0.12
    # validation → insecurity↓, energy↑, loneliness↓
    B[ST_VALIDATION, I_INSECURITY] = -0.20
    B[ST_VALIDATION, I_ENERGY] = 0.15
    B[ST_VALIDATION, I_LONELINESS] = -0.15
    # closeness → loneliness↓, longing↓, social_battery↓, energy↑
    B[ST_CLOSENESS, I_LONELINESS] = -0.25
    B[ST_CLOSENESS, I_LONGING] = -0.10
    B[ST_CLOSENESS, I_SOCIAL_BATTERY] = -0.10
    B[ST_CLOSENESS, I_ENERGY] = 0.08
    # conflict → stress↑, irritation↑, energy↓, fatigue↑, social_battery↓
    B[ST_CONFLICT, I_STRESS] = 0.35
    B[ST_CONFLICT, I_IRRITATION] = 0.30
    B[ST_CONFLICT, I_ENERGY] = -0.20
    B[ST_CONFLICT, I_MENTAL_FATIGUE] = 0.25
    B[ST_CONFLICT, I_SOCIAL_BATTERY] = -0.25
    # dependency → social_battery↓, loneliness↓, energy↑
    B[ST_DEPENDENCY, I_SOCIAL_BATTERY] = -0.08
    B[ST_DEPENDENCY, I_LONELINESS] = -0.15
    B[ST_DEPENDENCY, I_ENERGY] = 0.05
    # teasing → social_battery↓, irritation↑, energy↑
    B[ST_TEASING, I_SOCIAL_BATTERY] = -0.08
    B[ST_TEASING, I_IRRITATION] = 0.05
    B[ST_TEASING, I_ENERGY] = 0.05
    # emotional_weight → stress↑, mental_fatigue↑
    B[ST_EMOTIONAL_WEIGHT, I_STRESS] = 0.20
    B[ST_EMOTIONAL_WEIGHT, I_MENTAL_FATIGUE] = 0.15

    return B


INPUT_INFLUENCE_B = _build_input_influence()


# ═══════════════════════════════════════════════════════════════
# Relationship 输入影响矩阵
# ═══════════════════════════════════════════════════════════════


def _build_rel_input_influence() -> np.ndarray:
    """关系输入影响矩阵 B_rel（ST_SIZE × R_SIZE）。"""
    B = np.zeros((ST_SIZE, R_SIZE), dtype=np.float64)

    # abandoned → trust↓, safety↓, tension↑, dependency↑
    B[ST_ABANDONMENT, R_TRUST] = -0.08
    B[ST_ABANDONMENT, R_EMOTIONAL_SAFETY] = -0.10
    B[ST_ABANDONMENT, R_ROMANTIC_TENSION] = 0.06
    B[ST_ABANDONMENT, R_DEPENDENCY] = 0.08
    # validation → affection↑, trust↑
    B[ST_VALIDATION, R_AFFECTION] = 0.12
    B[ST_VALIDATION, R_TRUST] = 0.10
    # closeness → affection↑, familiarity↑, safety↑, tension↑
    B[ST_CLOSENESS, R_AFFECTION] = 0.10
    B[ST_CLOSENESS, R_FAMILIARITY] = 0.12
    B[ST_CLOSENESS, R_EMOTIONAL_SAFETY] = 0.08
    B[ST_CLOSENESS, R_ROMANTIC_TENSION] = 0.06
    # conflict → trust↓, safety↓, affection↓, tension↓
    B[ST_CONFLICT, R_TRUST] = -0.18
    B[ST_CONFLICT, R_EMOTIONAL_SAFETY] = -0.20
    B[ST_CONFLICT, R_AFFECTION] = -0.08
    B[ST_CONFLICT, R_ROMANTIC_TENSION] = -0.08
    # dependency → dependency↑, familiarity↑, safety↑
    B[ST_DEPENDENCY, R_DEPENDENCY] = 0.18
    B[ST_DEPENDENCY, R_FAMILIARITY] = 0.06
    B[ST_DEPENDENCY, R_EMOTIONAL_SAFETY] = 0.05
    # teasing → familiarity↑, tension↑
    B[ST_TEASING, R_FAMILIARITY] = 0.08
    B[ST_TEASING, R_ROMANTIC_TENSION] = 0.08

    return B


REL_INPUT_INFLUENCE_B = _build_rel_input_influence()
