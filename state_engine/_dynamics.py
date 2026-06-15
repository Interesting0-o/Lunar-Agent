"""State Dynamics —— 残差式状态更新 + 内建稳态恢复。

核心公式:
  h_t = h_{t-1} + Δt · (α·Δ_coupling + β·Δ_stimulus + γ·Δ_homeostatic)

三个速率参数分别由不同的人格/关系/防御因素调制:
  α — 跨维度耦合速率 (traits + relationship)
  β — 刺激接受速率    (defense profiles: attachment↑, suppression↓)
  γ — 稳态恢复速率    (traits + suppression↓)

门控不直接乘到状态值上，而是控制变化速率。
这保证: ① 不同防御水平最终收敛到同一稳态（仅速度不同）
         ② 残差形式天然保持长程稳定性
         ③ 稳态恢复永远向 setpoint 拉（γ > 0）
"""

import numpy as np
from state import (
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE,
    R_AFFECTION, R_TRUST, R_FAMILIARITY, R_DEPENDENCY,
    R_EMOTIONAL_SAFETY, R_ROMANTIC_TENSION,
    T_SENSITIVITY, T_PRIDE, T_EMOTIONAL_OPENNESS, T_EMOTIONAL_STABILITY,
    T_OPTIMISM, T_ANXIETY_PRONENESS, T_ANGER_REACTIVITY,
    T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE,
    DEFAULT_INTERNAL, DEFAULT_RELATIONSHIP,
)
from ._utils import soft_clamp
from ._matrices import (
    STATE_COUPLING_A, INPUT_INFLUENCE_B,
    REL_STATE_COUPLING_A, REL_INPUT_INFLUENCE_B,
)


def compute_setpoint(traits: np.ndarray) -> np.ndarray:
    """计算人格决定的内部情绪稳态基线。

    不同人格有不同的"正常"情绪水平:
      - 高焦虑 → stress/insecurity 基线高
      - 高乐观 → energy 基线高
      - 高依恋焦虑 → longing 基线高
      - 高易怒 → irritation 基线高

    setpoint 是系统在无外部刺激时的长期收敛目标。
    """
    sp = DEFAULT_INTERNAL.copy()
    t_dev = traits - 0.5

    sp[I_ENERGY]     += t_dev[T_OPTIMISM] * 0.15 - t_dev[T_ANXIETY_PRONENESS] * 0.08
    sp[I_STRESS]     += t_dev[T_ANXIETY_PRONENESS] * 0.20 + t_dev[T_ANGER_REACTIVITY] * 0.05
    sp[I_LONELINESS] += t_dev[T_ATTACHMENT_ANXIETY] * 0.10 - t_dev[T_OPTIMISM] * 0.05
    sp[I_INSECURITY] += t_dev[T_ATTACHMENT_ANXIETY] * 0.20 + t_dev[T_ANXIETY_PRONENESS] * 0.10
    sp[I_IRRITATION] += t_dev[T_ANGER_REACTIVITY] * 0.15 - t_dev[T_EMOTIONAL_STABILITY] * 0.10
    sp[I_LONGING]    += t_dev[T_ATTACHMENT_ANXIETY] * 0.15
    sp[I_SOCIAL_BATTERY] += t_dev[T_EMOTIONAL_STABILITY] * 0.05
    sp[I_MENTAL_FATIGUE] -= t_dev[T_EMOTIONAL_STABILITY] * 0.08 + t_dev[T_ANXIETY_PRONENESS] * 0.05

    return soft_clamp(sp, 0.05, 0.95)


def compute_rel_setpoint(traits: np.ndarray) -> np.ndarray:
    """计算人格决定的关系稳态基线。

    依恋回避 → 信任/好感/依赖基线低
    依恋焦虑 → 依赖基线高
    """
    sp = DEFAULT_RELATIONSHIP.copy()
    t_dev = traits - 0.5

    sp[R_TRUST]             -= t_dev[T_ATTACHMENT_AVOIDANCE] * 0.15
    sp[R_AFFECTION]         -= t_dev[T_ATTACHMENT_AVOIDANCE] * 0.10
    sp[R_DEPENDENCY]        -= t_dev[T_ATTACHMENT_AVOIDANCE] * 0.15
    sp[R_DEPENDENCY]        += t_dev[T_ATTACHMENT_ANXIETY] * 0.10
    sp[R_EMOTIONAL_SAFETY]  -= t_dev[T_ATTACHMENT_AVOIDANCE] * 0.12
    sp[R_FAMILIARITY]       -= t_dev[T_ATTACHMENT_AVOIDANCE] * 0.05
    sp[R_ROMANTIC_TENSION]  += t_dev[T_ATTACHMENT_ANXIETY] * 0.05

    return soft_clamp(sp, 0.02, 0.98)


def update_internal_state(
    current: np.ndarray,
    inner_stimuli: np.ndarray,
    traits: np.ndarray,
    relationship: np.ndarray,
    profiles: np.ndarray,  # (3, 7) defense profiles
    dt: float = 1.0,
) -> np.ndarray:
    """残差式内部状态更新。

    h_t = h_{t-1} + dt · (α·Δ_coupling + β·Δ_stimulus + γ·Δ_homeostatic)

    门控 (profiles) 控制 β 和 γ 的速率，不控制状态比例。

    Args:
        current: 当前内部状态 h_{t-1} (8,)
        inner_stimuli: 防御过滤后的"里"刺激 (7,)
        traits: 人格特质 (10,)
        relationship: 关系状态 (6,)
        profiles: 防御剖面 (3, 7)
        dt: 时间步长

    Returns:
        更新后的内部状态 h_t (8,)
    """
    supp = profiles[0]
    vuln = profiles[1]
    att  = profiles[2]

    # ── α: 跨维度耦合速率 ──
    # 由 traits 决定。开放→耦合快，稳定→耦合慢（更独立），信任→耦合快。
    alpha = (
        traits[T_EMOTIONAL_OPENNESS] * 0.30 +
        (1.0 - traits[T_EMOTIONAL_STABILITY]) * 0.15 +
        relationship[R_TRUST] * 0.12
    )
    alpha = soft_clamp(alpha, 0.02, 0.35)

    # ── β: 刺激接受速率 ──
    # 由防御剖面决定。依恋→接受快，脆弱→接受快，压抑→接受慢。
    beta = 0.10
    beta += att.mean() * 0.12
    beta += vuln.mean() * 0.08
    beta -= supp.mean() * 0.10
    beta = soft_clamp(beta, 0.01, 0.35)

    # ── γ: 稳态恢复速率 ──
    # 稳定→恢复快，乐观→恢复快，焦虑→恢复慢，压抑→恢复慢（放不下）。
    gamma = 0.08
    gamma += traits[T_EMOTIONAL_STABILITY] * 0.10
    gamma += traits[T_OPTIMISM] * 0.06
    gamma -= traits[T_ANXIETY_PRONENESS] * 0.06
    gamma -= supp.mean() * 0.08  # 压抑的人更难恢复
    gamma = soft_clamp(gamma, 0.01, 0.25)

    # ── 构建三项 Δ ──
    # ① 跨维度耦合（差值形式: A·h − h）
    coupling_effect = STATE_COUPLING_A @ current
    delta_coupling = coupling_effect - current

    # ② 刺激输入
    delta_stimulus = inner_stimuli @ INPUT_INFLUENCE_B

    # ③ 稳态恢复（永远向 setpoint 拉）
    setpoint = compute_setpoint(traits)
    delta_homeostatic = setpoint - current

    # ── 残差更新 ──
    delta = alpha * delta_coupling + beta * delta_stimulus + gamma * delta_homeostatic
    return soft_clamp(current + dt * delta, 0.0, 1.0)


def update_relationship_state(
    current: np.ndarray,
    inner_stimuli: np.ndarray,
    traits: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    """残差式关系状态更新（时间常数比内部状态慢 5-10 倍）。

    与 update_internal_state 同构，但:
      - α_rel 更小（关系变化极慢）
      - β_rel 更小（刺激对关系的影响有缓冲）
      - γ_rel 更小（关系稳态恢复极慢）
    """
    # ── α_rel: 关系跨维度耦合速率 ──
    alpha = (
        traits[T_EMOTIONAL_OPENNESS] * 0.04 +
        current[R_TRUST] * 0.03 +
        current[R_EMOTIONAL_SAFETY] * 0.02
    )
    alpha = soft_clamp(alpha, 0.005, 0.06)

    # ── β_rel: 关系刺激接受速率 ──
    beta = 0.02 + traits[T_ATTACHMENT_ANXIETY] * 0.015
    beta = soft_clamp(beta, 0.002, 0.06)

    # ── γ_rel: 关系稳态恢复速率 ──
    gamma = 0.005 + traits[T_EMOTIONAL_STABILITY] * 0.005
    gamma = soft_clamp(gamma, 0.001, 0.02)

    # ── 三项 Δ ──
    delta_coupling = REL_STATE_COUPLING_A @ current - current
    delta_stimulus = inner_stimuli @ REL_INPUT_INFLUENCE_B
    setpoint = compute_rel_setpoint(traits)
    delta_homeostatic = setpoint - current

    delta = alpha * delta_coupling + beta * delta_stimulus + gamma * delta_homeostatic
    return soft_clamp(current + dt * delta, 0.0, 1.0)
