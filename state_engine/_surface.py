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
    SURFACE_FEEDBACK_NEG,
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
    delta_hours: float = 0.0,
) -> np.ndarray:
    """④ 表面投影：内部态+关系态+外刺激 → 表面表达（带惯性混合 + 时间衰减）。

    s(t) = alpha * raw(t) + (1 - alpha) * s'(t-1)

    其中 s'(t-1) 是 prev_surface 经时间衰减后的版本：
      s'(t-1) = raw + (prev - raw) · exp(-0.5 · Δt)
    衰减半衰期约 1.4 小时——长时间间隔后表面表达"清零"重新初始化。

    Args:
        internal: 内部状态 (8,)
        relationship: 关系状态 (3,)
        outer_stimuli: 压抑后的外表情刺激 (7,)
        prev_surface: 前一帧表面状态 (7,)，None 表示首帧/初始化
        delta_hours: 自上次表面投影以来的时间（小时），0 表示连续对话

    Returns:
        surface_state (7,)
    """
    # ── 线性基线：由 SURFACE_MAPPER 计算（不含 traits）──
    sources = np.concatenate([internal, relationship, outer_stimuli])
    raw = SURFACE_MAPPER.compute(sources).copy()  # (7,)

    # ── 惯性混合（含时间衰减） ──
    if prev_surface is not None:
        # 时间衰减：prev_surface 向 raw 回归（表面"遗忘"效应）
        if delta_hours > 0.01:
            surface_decay = np.exp(-0.5 * delta_hours)
            prev_adj = raw + (prev_surface - raw) * surface_decay
        else:
            prev_adj = prev_surface

        alpha = _compute_surface_alpha(internal)
        s = alpha * raw + (1.0 - alpha) * prev_adj
    else:
        s = raw

    return soft_clamp(s, -1.0, 1.0)


def compute_surface_feedback(
    surface: np.ndarray,
    internal: np.ndarray,
) -> np.ndarray:
    """⑤ 表面→内部反馈：情绪失调成本 + 面部/躯体反馈 + 表达消耗 + 压抑成本。

    与 update_relationship_state 中的跨尺度耦合模式一致
    （CROSS_SCALE_COUPLING: internal(8,) @ M(8,3) → (3,)）。

    此处: surface(7,) @ M(7,8) → feedback_delta(8,)

    使用两个矩阵：
      - SURFACE_FEEDBACK_MATRIX: 正值（主动表达）→ 失调成本/面部反馈/表达消耗
      - SURFACE_FEEDBACK_NEG: 负值（压抑/伪装）→ 压抑成本/社交代谢

    Args:
        surface: 当前表面状态 (7,)
        internal: 当前内部状态 (8,) — 为未来状态相关门控预留

    Returns:
        feedback_delta (8,)，下轮加到内部状态
    """
    pos = np.maximum(surface, 0.0)   # 正值：主动表达
    neg = np.minimum(surface, 0.0)   # 负值：压抑/伪装
    return pos @ SURFACE_FEEDBACK_MATRIX + neg @ SURFACE_FEEDBACK_NEG
