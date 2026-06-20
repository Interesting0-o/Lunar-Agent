"""State Dynamics —— 残差式状态更新（刺激+耦合驱动，无 per-turn 稳态恢复）。

核心公式:
  h_t = h_{t-1} + Δt · (α·Δ_coupling + Δ_stimulus_modulated)

两个参数 + 一个逐维度调制:
  α — 跨维度耦合速率 (traits + relationship)
  β — 刺激接受速率，逐刺激维度 (defense profiles: hyper↑每维, deact↓每维)
      不再使用 hyper.mean() 全局标量，保留防御剖面的刺激特异性

稳态恢复（拉到人格基线）已移除——职责转移到 _decay.py 的时间衰减。
也就是说，每轮对话中状态完全由刺激和耦合驱动，不向 setpoint 拉。
回 base 靠的是真实时间流逝（_decay.apply_time_decay）。

设计理由:
  - 持续单一刺激下情绪应累积（耦合平衡点 h_eq），不应被 per-turn 恢复抵消
  - 时间衰减才是现实世界中情绪回归基线的通道（affective chronometry）
  - 这避免了 h_eq ≠ setpoint 的系统性偏离问题（见测试报告 2.2/2.3）
"""

import numpy as np
from state import (
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE, I_SIZE,
    R_AFFECTION, R_TRUST, R_FAMILIARITY, R_DEPENDENCY,
    R_EMOTIONAL_SAFETY, R_ROMANTIC_TENSION, R_SIZE,
    ST_SIZE,
    T_EMOTIONAL_OPENNESS, T_EMOTIONAL_STABILITY,
    T_OPTIMISM, T_ANXIETY_PRONENESS, T_ANGER_REACTIVITY,
    T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE,
    DEFAULT_INTERNAL, DEFAULT_RELATIONSHIP,
)
from ._utils import soft_clamp
from ._matrices import INPUT_INFLUENCE_B, REL_INPUT_INFLUENCE_B


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

    sp[I_ENERGY]     += traits[T_OPTIMISM] * 0.15 - traits[T_ANXIETY_PRONENESS] * 0.08
    sp[I_STRESS]     += traits[T_ANXIETY_PRONENESS] * 0.20 + traits[T_ANGER_REACTIVITY] * 0.05
    sp[I_LONELINESS] += traits[T_ATTACHMENT_ANXIETY] * 0.10 - traits[T_OPTIMISM] * 0.05
    sp[I_INSECURITY] += traits[T_ATTACHMENT_ANXIETY] * 0.20 + traits[T_ANXIETY_PRONENESS] * 0.10
    sp[I_IRRITATION] += traits[T_ANGER_REACTIVITY] * 0.15 - traits[T_EMOTIONAL_STABILITY] * 0.10
    sp[I_LONGING]    += traits[T_ATTACHMENT_ANXIETY] * 0.15
    sp[I_SOCIAL_BATTERY] += traits[T_EMOTIONAL_STABILITY] * 0.05
    sp[I_MENTAL_FATIGUE] -= traits[T_EMOTIONAL_STABILITY] * 0.08 + traits[T_ANXIETY_PRONENESS] * 0.05

    return np.clip(sp, -0.9, 0.9)


def compute_rel_setpoint(traits: np.ndarray) -> np.ndarray:
    """计算人格决定的关系稳态基线。

    依恋回避 → 信任/好感/依赖基线低
    依恋焦虑 → 依赖基线高
    """
    sp = DEFAULT_RELATIONSHIP.copy()

    sp[R_TRUST]             -= traits[T_ATTACHMENT_AVOIDANCE] * 0.15
    sp[R_AFFECTION]         -= traits[T_ATTACHMENT_AVOIDANCE] * 0.10
    sp[R_DEPENDENCY]        -= traits[T_ATTACHMENT_AVOIDANCE] * 0.15
    sp[R_DEPENDENCY]        += traits[T_ATTACHMENT_ANXIETY] * 0.10
    sp[R_EMOTIONAL_SAFETY]  -= traits[T_ATTACHMENT_AVOIDANCE] * 0.12
    sp[R_FAMILIARITY]       -= traits[T_ATTACHMENT_AVOIDANCE] * 0.05
    sp[R_ROMANTIC_TENSION]  += traits[T_ATTACHMENT_ANXIETY] * 0.05

    return np.clip(sp, -0.96, 0.96)


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

    β 已不再是全局标量——每个刺激维度有自己的接受率，融入 Δ_stimulus_modulated。

    Args:
        current: 当前内部状态 h_{t-1} (8,)
        inner_stimuli: 防御过滤后的"里"刺激 (7,)
        traits: 人格特质 (10,)
        relationship: 关系状态 (6,)
        profiles: 防御剖面 (2, 7)
        dt: 时间步长

    Returns:
        更新后的内部状态 h_t (8,)
    """
    deact = profiles[0]  # 去激活 (7,)
    hyper = profiles[1]  # 过度激活 (7,)

    # ── α: 跨维度耦合速率 ──
    # 由 traits 决定。开放→耦合快，稳定→耦合慢（更独立），信任→耦合快。
    alpha = 0.285
    alpha += traits[T_EMOTIONAL_OPENNESS] * 0.15
    alpha -= traits[T_EMOTIONAL_STABILITY] * 0.075
    alpha += relationship[R_TRUST] * 0.06
    alpha = soft_clamp(alpha, 0.02, 0.35)

    # ── β: 刺激接受速率（逐刺激维度）──
    # 每个刺激维度有自己的接受率，由该维度的防御剖面调制。
    #   hyper[ST_CLOSENESS]高 → closeness 被强烈接受
    #   deact[ST_CONFLICT]高  → conflict 被情感压制
    # 相比 hyper.mean() 全局标量方案，保留防御剖面在具体刺激类型上的选择性。
    # 注意：apply_defenses 通过 inner=stimuli*(1+hyper) 做幅度放大，
    # 这里是速率调制——两者独立，并用更合理。
    beta_base = np.full(ST_SIZE, 0.05)
    beta_stim = beta_base + hyper * 0.35 - deact * 0.15
    beta_stim = np.clip(beta_stim, 0.01, 0.35)

    # ── Δ_coupling: 跨维度耦合 + 每维度自阻尼 ──
    # 替代旧的 A 矩阵（STATE_COUPLING_A @ h − h）:
    # 自阻尼系数对应原 A 矩阵对角线 0.85 → 每轮向 0 回归 15%
    SELF_DECAY = np.full(I_SIZE, 0.15)

    # 跨维度耦合：显式命名规则，每条附心理学依据
    coupling = np.zeros(I_SIZE, dtype=np.float64)
    coupling[I_STRESS]     += current[I_ENERGY] * (-0.05)        # 精力充沛→压力降低
    coupling[I_STRESS]     += current[I_INSECURITY] * 0.10       # 不安全感→压力
    coupling[I_LONELINESS] += current[I_ENERGY] * (-0.05)        # 精力充沛→孤独感降低
    coupling[I_LONELINESS] += current[I_STRESS] * 0.08           # 压力→孤独感
    coupling[I_INSECURITY] += current[I_LONELINESS] * 0.12       # 孤独→不安
    coupling[I_IRRITATION] += current[I_STRESS] * 0.15           # 压力积累→易怒
    coupling[I_IRRITATION] += current[I_SOCIAL_BATTERY] * (-0.08) # 社交电量低→烦躁
    coupling[I_LONGING]    += current[I_LONELINESS] * 0.15       # 孤独→思念
    coupling[I_SOCIAL_BATTERY] += current[I_ENERGY] * 0.08       # 精力充沛→电量恢复
    coupling[I_MENTAL_FATIGUE] += current[I_STRESS] * 0.10       # 压力→精神疲劳
    coupling[I_MENTAL_FATIGUE] += current[I_SOCIAL_BATTERY] * (-0.10) # 社交电量低→疲劳

    # 自阻尼目标：每维度向谁收敛
    # 大部分维度向 0（中性基线），social_battery 向 0.20（健康基线）
    # setpoint 0.20 ≈ DEFAULT_INTERNAL[I_SOCIAL_BATTERY] + stability*0.05
    # 这里用固定值 0.20 作为阻尼目标，人格化 setpoint 由 _decay.py 的时间衰减负责
    DECAY_TARGETS = np.zeros(I_SIZE, dtype=np.float64)
    DECAY_TARGETS[I_SOCIAL_BATTERY] = 0.20
    delta_coupling = coupling - SELF_DECAY * (current - DECAY_TARGETS)

    # ② 刺激输入：逐维度 β 调制 → B 矩阵映射
    # 每个刺激先按自己的接受率缩放，再映射到内部状态。
    # 这保留了防御剖面对具体刺激类型的选择性响应。
    modulated_stimuli = beta_stim * inner_stimuli  # (7,), 逐元素乘
    delta_stimulus = modulated_stimuli @ INPUT_INFLUENCE_B

    # ③ 稳态恢复已移除——拉到 setpoint 的职责交给 _decay.py
    #    （时间衰减，由真实时间 Δt 驱动）。
    #    每轮对话中，状态完全由刺激和耦合驱动。

    # ── 残差更新 ──
    delta = alpha * delta_coupling + delta_stimulus  # β 已融入 modulated_stimuli
    return soft_clamp(current + dt * delta, -1.0, 1.0)


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
    """
    # ── α_rel: 关系跨维度耦合速率 ──
    alpha = 0.045
    alpha += traits[T_EMOTIONAL_OPENNESS] * 0.02
    alpha += current[R_TRUST] * 0.015
    alpha += current[R_EMOTIONAL_SAFETY] * 0.01
    alpha = soft_clamp(alpha, 0.005, 0.06)

    # ── β_rel: 关系刺激接受速率 ──
    beta = 0.0275
    beta += traits[T_ATTACHMENT_ANXIETY] * 0.0075
    beta = soft_clamp(beta, 0.002, 0.06)

    # ── γ_rel: 关系稳态恢复速率（已移除） ──
    # 关系稳态恢复由 _decay.py 的时间衰减负责。

    # ── Δ_coupling: 关系跨维度耦合 + 自阻尼 ──
    # 替代旧的 REL_STATE_COUPLING_A @ h − h
    # 原矩阵经谱归一化（缩放 0.9441），自阻尼 ≈ 0.1503
    REL_SELF_DECAY = np.full(R_SIZE, 0.15033222)

    rel_coupling = np.zeros(R_SIZE, dtype=np.float64)
    rel_coupling[R_TRUST]            += current[R_AFFECTION] * 0.075526      # 好感→信任
    rel_coupling[R_TRUST]            += current[R_EMOTIONAL_SAFETY] * 0.047204  # 安全感→信任
    rel_coupling[R_FAMILIARITY]      += current[R_AFFECTION] * 0.047204     # 好感→熟悉度
    rel_coupling[R_DEPENDENCY]       += current[R_TRUST] * 0.047204         # 信任→依赖
    rel_coupling[R_EMOTIONAL_SAFETY] += current[R_TRUST] * 0.094408        # 信任→安全感
    rel_coupling[R_EMOTIONAL_SAFETY] += current[R_FAMILIARITY] * 0.075526  # 熟悉→安全感
    rel_coupling[R_AFFECTION]        += current[R_EMOTIONAL_SAFETY] * 0.047204  # 安全感→好感
    rel_coupling[R_AFFECTION]        += current[R_ROMANTIC_TENSION] * 0.028322  # 暧昧→好感
    rel_coupling[R_ROMANTIC_TENSION] += current[R_DEPENDENCY] * 0.047204   # 依赖→暧昧
    rel_coupling[R_ROMANTIC_TENSION] += current[R_AFFECTION] * 0.035       # 好感→紧张（喜欢一个人自然会紧张）
    rel_coupling[R_ROMANTIC_TENSION] += current[R_TRUST] * 0.025          # 信任→期待（越信任越在意对方）

    delta_coupling = rel_coupling - REL_SELF_DECAY * current

    # ── 刺激输入 ──
    delta_stimulus = inner_stimuli @ REL_INPUT_INFLUENCE_B

    delta = alpha * delta_coupling + beta * delta_stimulus  # 无 gamma 项
    return soft_clamp(current + dt * delta, -1.0, 1.0)
