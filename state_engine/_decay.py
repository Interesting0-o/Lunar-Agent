"""③ Dynamic Decay —— 人格驱动的动态衰减（纯稳态恢复）。

衰减速率由以下因素动态调制:
  - 人格基线: 高傲→记仇（恢复慢），情绪稳定→恢复快
  - 关系语境: 信任高→压力消退快（安全基地效应）
  - 急性状态: 高压下所有负面情绪消退变慢（压力锁定）
  - 刺激-特质共振: 特定刺激遇到特定特质时，恢复进一步放缓

稳定性保证:
  - 所有 decay < 1.0 严格成立（contractive，向基线回归）
  - decay = 1.0 表示零恢复（保持当前偏离量）
  - 永远不会 > 1.0（不会背离基线，情绪自我增强由动力学 A 矩阵负责）

门控与衰减的协作:
  - 紧门控 + 慢衰减 = 压抑爆炸型（进得少但放不下）
  - 松门控 + 快衰减 = 表达恢复型（进得多但也放得下）
  - 紧门控 + 快衰减 = 冷漠超然型
  - 松门控 + 慢衰减 = 敏感内耗型

注意: 旧代码曾允许 decay > 1.0 作为"情绪自我增强"机制，
已废弃——自我增强应通过跨维度耦合（A 矩阵）实现，
不应通过破坏衰减的稳定语义来实现。
"""

import numpy as np
from state import (
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE,
    R_AFFECTION, R_TRUST, R_FAMILIARITY, R_DEPENDENCY,
    R_EMOTIONAL_SAFETY, R_ROMANTIC_TENSION,
    T_SENSITIVITY, T_PRIDE, T_EMOTIONAL_STABILITY,
    T_OPTIMISM, T_ANXIETY_PRONENESS, T_ANGER_REACTIVITY,
    T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE,
    ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT, ST_TEASING,
)
from ._utils import soft_clamp
from ._matrices import _INTERNAL_BASE_DECAY, _RELATIONSHIP_BASE_DECAY


# 衰减系数严格上界 — 保证永远向基线回归而非背离
_MAX_INTERNAL_DECAY = 0.99   # 内部状态最快可接近"不恢复"但绝不能放大
_MAX_RELATIONSHIP_DECAY = 0.999  # 关系状态同理，上界更接近 1（关系变化极慢）
_MIN_INTERNAL_DECAY = 0.70
_MIN_RELATIONSHIP_DECAY = 0.95


def compute_dynamic_decay(
    traits: np.ndarray,
    relationship: np.ndarray,
    internal: np.ndarray,
    stimuli: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """③ 人格驱动的动态衰减计算。

    返回 (internal_decay, relationship_decay)，可能 > 1.0 表示情绪自我增强。
    """
    idcy = _INTERNAL_BASE_DECAY.copy()
    rdcy = _RELATIONSHIP_BASE_DECAY.copy()

    # ── 人格调制 ──
    # 自尊：高 → 烦躁消退慢（记仇），不安全感消退也慢（不愿承认脆弱）
    pride_dev = traits[T_PRIDE] - 0.5
    idcy[I_IRRITATION] += pride_dev * 0.12
    idcy[I_INSECURITY] += pride_dev * 0.06

    # 情绪稳定：→ 所有负面情绪消退快
    stability_dev = traits[T_EMOTIONAL_STABILITY] - 0.5
    idcy[I_STRESS]       -= stability_dev * 0.15
    idcy[I_IRRITATION]   -= stability_dev * 0.12
    idcy[I_MENTAL_FATIGUE] -= stability_dev * 0.08
    idcy[I_INSECURITY]   -= stability_dev * 0.08
    idcy[I_LONELINESS]   -= stability_dev * 0.06

    # 依恋焦虑：→ 不安全感/渴望消退慢（总怕被丢下）
    attach_dev = traits[T_ATTACHMENT_ANXIETY] - 0.5
    idcy[I_INSECURITY] += attach_dev * 0.10
    idcy[I_LONGING]    += attach_dev * 0.08

    # 乐观：→ 孤独/不安全感消退快（天然自我调节）
    optimism_dev = traits[T_OPTIMISM] - 0.5
    idcy[I_LONELINESS] -= optimism_dev * 0.08
    idcy[I_INSECURITY] -= optimism_dev * 0.06

    # 易怒：→ 烦躁消退慢
    anger_dev = traits[T_ANGER_REACTIVITY] - 0.5
    idcy[I_IRRITATION] += anger_dev * 0.10

    # 敏感：→ 所有情绪体验更深，消退慢
    sensitivity_dev = traits[T_SENSITIVITY] - 0.5
    idcy[I_STRESS]     += sensitivity_dev * 0.04
    idcy[I_LONELINESS] += sensitivity_dev * 0.04
    idcy[I_INSECURITY] += sensitivity_dev * 0.04

    # 依恋回避：→ 关系衰减加速（回避型更难建立深层关系）
    avoidance_dev = traits[T_ATTACHMENT_AVOIDANCE] - 0.5
    rdcy[R_AFFECTION]       -= avoidance_dev * 0.004
    rdcy[R_TRUST]           -= avoidance_dev * 0.003
    rdcy[R_EMOTIONAL_SAFETY] -= avoidance_dev * 0.003

    # ── 关系调制 ──
    # 信任高 → 压力/不安全感消退更快（安全基地效应）
    idcy[I_STRESS]     -= relationship[R_TRUST] * 0.06
    idcy[I_INSECURITY] -= relationship[R_EMOTIONAL_SAFETY] * 0.06
    # 熟悉度高 → 关系状态更稳定
    familiarity_bonus = relationship[R_FAMILIARITY] * 0.003
    rdcy[R_AFFECTION] += familiarity_bonus
    rdcy[R_TRUST]     += familiarity_bonus
    # 情感安全高 → 关系各方面都更稳定
    safety_bonus = relationship[R_EMOTIONAL_SAFETY] * 0.002
    rdcy[R_AFFECTION]       += safety_bonus
    rdcy[R_TRUST]           += safety_bonus
    rdcy[R_EMOTIONAL_SAFETY] += safety_bonus
    # 浪漫张力高 → 好感消退慢（越在意越放不下）
    tension_effect = relationship[R_ROMANTIC_TENSION] * 0.002
    rdcy[R_AFFECTION] += tension_effect

    # ── 急性状态调制 ──
    # 高压力 → 所有负面情绪消退变慢（压力锁定效应）
    stress_penalty = internal[I_STRESS] * 0.04
    idcy[I_IRRITATION] += stress_penalty
    idcy[I_INSECURITY] += stress_penalty
    idcy[I_LONELINESS] += stress_penalty * 0.5
    # 精神疲劳 → 精力和社交电量恢复变慢
    fatigue_penalty = internal[I_MENTAL_FATIGUE] * 0.04
    idcy[I_ENERGY]        -= fatigue_penalty
    idcy[I_SOCIAL_BATTERY] -= fatigue_penalty

    # ── 刺激-特质共振（条件放大/加速消退） ──
    # 被抛弃 + 高依恋焦虑 → 不安全感自我增强
    if stimuli[ST_ABANDONMENT] > 0.3 and traits[T_ATTACHMENT_ANXIETY] > 0.55:
        boost = stimuli[ST_ABANDONMENT] * traits[T_ATTACHMENT_ANXIETY] * 0.05
        idcy[I_INSECURITY] += boost
        idcy[I_LONELINESS] += boost * 0.6
    # 被逗弄 + 高自尊 → 烦躁增强
    if stimuli[ST_TEASING] > 0.2 and traits[T_PRIDE] > 0.55:
        idcy[I_IRRITATION] += stimuli[ST_TEASING] * traits[T_PRIDE] * 0.06
    # 冲突 + 高易怒 → 压力/烦躁共振放大
    if stimuli[ST_CONFLICT] > 0.3 and traits[T_ANGER_REACTIVITY] > 0.55:
        boost = stimuli[ST_CONFLICT] * traits[T_ANGER_REACTIVITY] * 0.05
        idcy[I_IRRITATION] += boost
        idcy[I_STRESS]     += boost * 0.5
    # 被认可 + 高敏感 → 不安全感加速消退
    if stimuli[ST_VALIDATION] > 0.3 and traits[T_SENSITIVITY] > 0.55:
        idcy[I_INSECURITY] -= stimuli[ST_VALIDATION] * traits[T_SENSITIVITY] * 0.04
    # 亲密靠近 + 高依恋焦虑 → 渴望消退变慢
    if stimuli[ST_CLOSENESS] > 0.3 and traits[T_ATTACHMENT_ANXIETY] > 0.55:
        idcy[I_LONGING] += stimuli[ST_CLOSENESS] * traits[T_ATTACHMENT_ANXIETY] * 0.04

    # 最终 clamp — 严格 < 1.0，永不背离基线
    idcy = soft_clamp(idcy, _MIN_INTERNAL_DECAY, _MAX_INTERNAL_DECAY)
    rdcy = soft_clamp(rdcy, _MIN_RELATIONSHIP_DECAY, _MAX_RELATIONSHIP_DECAY)

    return idcy, rdcy


def apply_decay(state: np.ndarray, decay: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """③ 衰减：严格向基线回归（contractive mapping）。

    state[t] = baseline + (state[t-1] - baseline) × decay

    decay ∈ (0, 1) → 收缩向基线
    decay = 1     → 保持不变（零恢复）
    decay < 1 必成立（调用方保证），绝不背离基线。

    情绪自我增强由动力学 A 矩阵的跨维度耦合负责，
    不通过破坏衰减语义来实现。
    """
    if not np.all(decay < 1.0):
        violators = np.where(decay >= 1.0)[0]
        raise ValueError(
            f"衰减系数必须 < 1.0，发现违规维度: {violators.tolist()}, "
            f"decay={decay[violators]}"
        )
    return soft_clamp(baseline + (state - baseline) * decay, 0.0, 1.0)
