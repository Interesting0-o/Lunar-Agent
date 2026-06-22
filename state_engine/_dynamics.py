"""State Dynamics —— 残差式状态更新（刺激+耦合驱动，无 per-turn 稳态恢复）。

核心公式:
  h_t = h_{t-1} + Δt · (α·Δ_coupling + Δ_stimulus_modulated)

两个参数 + 一个逐维度调制:
  α — 跨维度耦合速率 (traits + relationship)
  β — 刺激接受速率，逐刺激维度 (defense profiles: hyper↑每维, deact↓每维)

稳态恢复（拉到人格基线）已移除——职责转移到 _decay.py 的时间衰减。

约束合规:
  - SELF_DECAY / REL_SELF_DECAY / DECAY_TARGETS → WeightVector in _dynamics_weights.py
  - 耦合系数 (internal/relationship/cross-scale) → WeightMapper in _dynamics_weights.py
  - 所有参数带完整 provenance
"""

import numpy as np
from state import (
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE,
    R_AFFECTION, R_TRUST_BOND, R_INTIMACY,
)
from ._utils import soft_clamp
from ._matrices import INPUT_INFLUENCE_B, REL_INPUT_INFLUENCE_B
from ._dynamics_weights import (
    SELF_DECAY, DECAY_TARGETS, REL_SELF_DECAY,
    INTERNAL_COUPLING, RELATIONSHIP_COUPLING, CROSS_SCALE_COUPLING,
    ALPHA_MAPPER, ALPHA_REL_MAPPER, BETA_REL_MAPPER,
    BETA_BASE, HYPER_BETA_GAIN, DEACT_BETA_GAIN,
    SETPOINT_MAPPER, REL_SETPOINT_MAPPER,
)


def compute_setpoint(traits: np.ndarray) -> np.ndarray:
    """计算人格决定的内部情绪稳态基线。

    通过 SETPOINT_MAPPER(LinearMapping) 计算:
      sp = DEFAULT_INTERNAL + traits @ W  (W 带 provenance)
    """
    return np.clip(SETPOINT_MAPPER.compute(traits), -0.9, 0.9)


def compute_rel_setpoint(traits: np.ndarray) -> np.ndarray:
    """计算人格决定的关系稳态基线（3 维版）。

    通过 REL_SETPOINT_MAPPER(LinearMapping) 计算:
      sp = DEFAULT_RELATIONSHIP + traits @ W  (W 带 provenance)
    """
    return np.clip(REL_SETPOINT_MAPPER.compute(traits), -0.96, 0.96)


def update_internal_state(
    current: np.ndarray,
    inner_stimuli: np.ndarray,
    traits: np.ndarray,
    relationship: np.ndarray,
    profiles: np.ndarray,  # (2, 7) defense profiles
    dt: float = 1.0,
) -> np.ndarray:
    """残差式内部状态更新。

    h_t = h_{t-1} + dt · (α·Δ_coupling + Δ_stimulus_modulated)

    Args:
        current: 当前内部状态 h_{t-1} (8,)
        inner_stimuli: 防御过滤后的"里"刺激 (7,)
        traits: 人格特质 (10,)
        relationship: 关系状态 (3,)
        profiles: 防御剖面 (2, 7)
        dt: 时间步长

    Returns:
        更新后的内部状态 h_t (8,)
    """
    deact = profiles[0]  # 去激活 (7,)
    hyper = profiles[1]  # 过度激活 (7,)

    # ── α: 跨维度耦合速率 ──
    # 由 ALPHA_MAPPER(LinearMapping) 计算，带完整 provenance。
    alpha = ALPHA_MAPPER.compute(np.concatenate([traits, relationship]))[0]
    alpha = soft_clamp(alpha, 0.02, 0.35)

    # ── β: 刺激接受速率（逐刺激维度）──
    # β_base + hyper * hyper_gain + deact * deact_gain
    # 所有系数来自 _dynamics_weights.py（WeightVector 带 provenance）
    beta_stim = BETA_BASE + hyper * HYPER_BETA_GAIN + deact * DEACT_BETA_GAIN
    beta_stim = np.clip(beta_stim, 0.01, 0.35)

    # ── Δ_coupling: 跨维度耦合 + 每维度自阻尼 ──
    # 耦合矩阵通过 WeightMapper 构建（_dynamics_weights.py），带完整 provenance。
    # SELF_DECAY 是每维度独立自阻尼率。
    coupling = current @ INTERNAL_COUPLING  # current (8,) @ M (8,8) → (8,)
    delta_coupling = coupling - SELF_DECAY * (current - DECAY_TARGETS)

    # ② 刺激输入：逐维度 β 调制 → B 矩阵映射
    modulated_stimuli = beta_stim * inner_stimuli  # (7,), 逐元素乘
    delta_stimulus = modulated_stimuli @ INPUT_INFLUENCE_B

    # ③ 稳态恢复已移除——拉到 setpoint 的职责交给 _decay.py

    # ── 残差更新 ──
    delta = alpha * delta_coupling + delta_stimulus
    return soft_clamp(current + dt * delta, -1.0, 1.0)


def update_relationship_state(
    current: np.ndarray,
    inner_stimuli: np.ndarray,
    traits: np.ndarray,
    dt: float = 1.0,
    current_internal: np.ndarray | None = None,
) -> np.ndarray:
    """残差式关系状态更新（时间常数比内部状态慢 5-10 倍）。

    与 update_internal_state 同构，但:
      - α_rel 更小（关系变化极慢）
      - β_rel 更小（刺激对关系的影响有缓冲）
      - 可选的 current_internal 参数提供跨尺度耦合（内→关）
    """
    # ── α_rel: 关系跨维度耦合速率 ──
    # 由 ALPHA_REL_MAPPER(LinearMapping) 计算，带完整 provenance。
    alpha = ALPHA_REL_MAPPER.compute(np.concatenate([traits, current]))[0]
    alpha = soft_clamp(alpha, 0.005, 0.06)

    # ── β_rel: 关系刺激接受速率 ──
    # 由 BETA_REL_MAPPER(LinearMapping) 计算，带完整 provenance。
    beta = BETA_REL_MAPPER.compute(traits)[0]
    beta = soft_clamp(beta, 0.002, 0.06)

    # ── Δ_coupling: 关系耦合 + 自阻尼 + 跨尺度 ──
    # 关系耦合矩阵 (3×3, WeightMapper 构建)
    rel_coupling = current @ RELATIONSHIP_COUPLING  # current (3,) @ M (3,3) → (3,)

    # 跨尺度耦合（内→关）
    if current_internal is not None:
        cross = current_internal @ CROSS_SCALE_COUPLING  # internal (8,) @ M (8,3) → (3,)
        rel_coupling = rel_coupling + cross

    delta_coupling = rel_coupling - REL_SELF_DECAY * current

    # ── 刺激输入 ──
    delta_stimulus = inner_stimuli @ REL_INPUT_INFLUENCE_B
    delta_stimulus *= beta

    delta = alpha * delta_coupling + delta_stimulus
    return soft_clamp(current + dt * delta, -1.0, 1.0)
