"""Surface Projection —— 内部状态 + 外表情刺激 → 表面表达。

SurfaceState 从内部态 + 关系态 + outer_stimuli 动态投影，带惯性混合。

s(t) = alpha * raw_projection(t) + (1 - alpha) * s(t-1)

关键：
  - outer_stimuli 是被压抑后的版本（由 apply_defenses 的 deactivation 输出）
  - traits 不直接作为输入（已通过 defense profiles + dynamics 间接影响）
  - 表面→内部反馈（情绪失调成本 + 面部/躯体反馈 + 表达消耗）在下轮 dynamics 生效

约束合规：
  - 约束①修复：traits 不再直接输入 surface（通过 pipeline 间接）
  - 约束④维持：outer_stimuli 是 defenses 压抑后产物，作为正常输入
  - 线性系数由 SURFACE_MAPPER（`_surface_weights.py`）管理
  - 惯性混合保留为显式非线性动力学代码
"""

import numpy as np
from state import (
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_MENTAL_FATIGUE,
    R_AFFECTION, R_TRUST_BOND,
    S_EXPRESSIVENESS, S_WARMTH, S_SHARPNESS, S_SOFTNESS,
    S_ENTHUSIASM, S_RESTRAINT, S_VULNERABILITY, S_SIZE,
)
from ._utils import soft_clamp
from ._surface_weights import (
    SURFACE_MAPPER,
    SURFACE_FEEDBACK_MATRIX,
)


def _compute_surface_alpha(internal: np.ndarray) -> float:
    """计算表面惯性混合系数 alpha。

    高压力 → alpha 下降（表面更"僵"，惯性更强，不易变化）
    高精力 → alpha 上升（表面更"灵"，响应更快）

    alpha ∈ [0.1, 0.9]，基值 0.5
    """
    stress_component = max(0.0, float(internal[I_STRESS]))
    energy_component = max(0.0, float(internal[I_ENERGY]))
    alpha = 0.5 - 0.3 * stress_component + 0.2 * energy_component
    return float(np.clip(alpha, 0.1, 0.9))


def project_surface(
    internal: np.ndarray,
    relationship: np.ndarray,
    outer_stimuli: np.ndarray,
    prev_surface: np.ndarray | None = None,
) -> np.ndarray:
    """④ 表面投影：内部态+关系态+外刺激 → 表面表达（带惯性混合）。

    s(t) = alpha * raw(t) + (1 - alpha) * s(t-1)

    Args:
        internal: 内部状态 (8,)
        relationship: 关系状态 (3,)
        outer_stimuli: 压抑后的外表情刺激 (7,)
        prev_surface: 前一帧表面状态 (7,)，None 表示首帧/初始化

    Returns:
        surface_state (7,)
    """
    # ── 线性基线：由 SURFACE_MAPPER 计算（不含 traits）──
    sources = np.concatenate([internal, relationship, outer_stimuli])
    raw = SURFACE_MAPPER.compute(sources).copy()  # (7,)

    # ── 惯性混合 ──
    if prev_surface is not None:
        alpha = _compute_surface_alpha(internal)
        s = alpha * raw + (1.0 - alpha) * prev_surface
    else:
        s = raw

    return soft_clamp(s, -1.0, 1.0)


def compute_surface_feedback(
    surface: np.ndarray,
    internal: np.ndarray,
) -> np.ndarray:
    """⑤ 表面→内部反馈：情绪失调成本 + 面部/躯体反馈 + 表达消耗。

    与 update_relationship_state 中的跨尺度耦合模式一致
    （CROSS_SCALE_COUPLING: internal(8,) @ M(8,3) → (3,)）。

    此处: surface(7,) @ M(7,8) → feedback_delta(8,)

    注意：仅 surface 的正值（主动表达）产生反馈。
    负值（如 S_RESTRAINT < 0 → "不克制"）不应反向降低 stress，
    因为情绪劳动/面部反馈效应要求"正在做某事"才有代谢成本。

    Args:
        surface: 当前表面状态 (7,)
        internal: 当前内部状态 (8,) — 为未来状态相关门控预留

    Returns:
        feedback_delta (8,)，下轮加到内部状态
    """
    active = np.maximum(surface, 0.0)  # 仅正值产生反馈
    return active @ SURFACE_FEEDBACK_MATRIX  # (7,) @ (7,8) → (8,)
