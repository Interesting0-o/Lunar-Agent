"""Defense Profiles —— 二维防御机制建模。

基于 Bowlby (1980) 的依恋防御二分法 + Richardson et al. (2023, 2025) 的实证验证:

  profiles[0, :] — Deactivation (去激活): 削减外在表达
    高回避 → 情感疏离、压抑表达。关联特质: pride↑, avoidance↑, stability↓
    由 suppression + 逆 vulnerability 合并而来。

  profiles[1, :] — Hyperactivation (过度激活): 放大内心感受
    高焦虑 → 放大关系威胁/亲近信号。关联特质: attachment_anxiety↑, jealousy↑
    原 attachment 剖面。

每个剖面是 7 维敏感度向量 ∈ [0, 1]，对不同类型的心理刺激独立激活。

扩展方式: 增加新防御维度（如 boundary）需验证:
  1. 概念判别效度 — 与 deactivation/hyperactivation 不是同一构念
  2. 因子/相关性检查 — |r| < 0.3 才值得独立成维度
  3. 交互机制 — 在 apply_defenses 中有不同于现有维度的数学操作
验证通过后将 profiles 扩展为 (3, 7)，不影响现有逻辑。
"""

import numpy as np
from state import (
    T_SENSITIVITY, T_PRIDE, T_EMOTIONAL_OPENNESS, T_EMOTIONAL_STABILITY,
    T_OPTIMISM, T_ANXIETY_PRONENESS, T_ANGER_REACTIVITY, T_JEALOUSY_SENSITIVITY,
    T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE,
    I_STRESS, I_INSECURITY, I_LONELINESS, I_LONGING,
    R_AFFECTION, R_TRUST, R_FAMILIARITY,
    R_EMOTIONAL_SAFETY, R_ROMANTIC_TENSION,
    ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT,
    ST_DEPENDENCY, ST_TEASING, ST_EMOTIONAL_WEIGHT, ST_SIZE,
)
from ._utils import soft_clamp, _sigmoid


def compute_defense_profiles(
    traits: np.ndarray,
    relationship: np.ndarray,
    internal: np.ndarray,
) -> np.ndarray:
    """计算二维防御剖面: 去激活 + 过度激活。

    Returns:
        profiles: (2, 7) np.ndarray
            profiles[0, :] — deactivation: 削减各类刺激外在表达的程度
            profiles[1, :] — hyperactivation: 放大各类刺激内心感受的程度

    每个剖面 ∈ [0, 1]，值越高 = 该防御对该类刺激越活跃。
    """
    profiles = np.zeros((2, ST_SIZE), dtype=np.float64)

    # ═══════════════════════════════════════════════════════════
    # Profile 0 — Deactivation (去激活)
    #
    #   合并了原 suppression + 逆 vulnerability:
    #     - 高 Pride → 隐藏"暴露脆弱"的刺激（被抛弃、被认可、依赖）
    #     - 高 Avoidance → 全局情感疏离
    #     - 高 Stability → 全局去激活低（真淡定，不是装的）
    #     - 高 Openness → 去激活低（愿意流露）
    #     - 低 Trust / 低 Safety → 去激活高（不信任时不示弱）
    #     - 高 Stress / 高 Insecurity → 去激活高（越难受越藏）
    #
    #   Bowlby: 去激活策略 = 情感疏离 + 最小化痛苦表达 + 转移注意力
    #   Richardson (2023): 回避独有防御 —— distancing, disengagement, vulnerability suppression
    # ═══════════════════════════════════════════════════════════
    deact = np.zeros(ST_SIZE, dtype=np.float64)

    # 人格基线: 每种刺激天然被隐藏的程度不同
    deact[ST_ABANDONMENT] = 0.30 + traits[T_PRIDE] * 0.225 + traits[T_JEALOUSY_SENSITIVITY] * 0.09
    deact[ST_VALIDATION]  = 0.25 + traits[T_PRIDE] * 0.20
    deact[ST_DEPENDENCY]  = 0.28 + traits[T_PRIDE] * 0.19 + traits[T_ATTACHMENT_AVOIDANCE] * 0.10
    deact[ST_CLOSENESS]   = 0.15 + traits[T_PRIDE] * 0.09 + traits[T_ATTACHMENT_AVOIDANCE] * 0.075
    deact[ST_CONFLICT]    = 0.20 + traits[T_PRIDE] * 0.14 + traits[T_ANGER_REACTIVITY] * 0.11
    deact[ST_TEASING]     = 0.20 + traits[T_PRIDE] * 0.16
    deact[ST_EMOTIONAL_WEIGHT] = 0.25 + traits[T_PRIDE] * 0.14

    # 去激活调制器（全局）
    # 情绪稳定 → 真淡定，不需要去激活
    deact -= traits[T_EMOTIONAL_STABILITY] * 0.11
    # 情绪开放 → 愿意流露，去激活低
    deact -= traits[T_EMOTIONAL_OPENNESS] * 0.06
    # 依恋回避 → 全局增强去激活（情感疏离是回避的核心特征）
    deact += traits[T_ATTACHMENT_AVOIDANCE] * 0.09

    # 关系调制: 信任和情感安全感降低去激活（安全基地效应）
    rel_loosen = 1.0 - relationship[R_TRUST] * 0.11 - relationship[R_EMOTIONAL_SAFETY] * 0.09
    deact *= rel_loosen

    # 内部急性推动: 压力和不安加剧去激活（越难受越藏）
    deact += internal[I_STRESS] * 0.05 + internal[I_INSECURITY] * 0.04

    profiles[0] = _sigmoid(deact - 0.48)

    # ═══════════════════════════════════════════════════════════
    # Profile 1 — Hyperactivation (过度激活)
    #
    #   原 attachment 剖面:
    #     - 高 Attachment Anxiety → 放大"关系威胁/亲近"刺激
    #     - 高 Jealousy → 对被抛弃更敏感
    #     - 低 Avoidance → 不回避亲密信号
    #     - 高 Sensitivity → 全局更敏感
    #
    #   Bowlby: 过度激活策略 = 夸大痛苦表达 + 持续监控 + 投射
    #   Richardson (2023): 焦虑独有防御 —— splitting, projective identification,
    #     anticipation, acting out, passive-aggression, reaction formation
    # ═══════════════════════════════════════════════════════════
    hyper = np.zeros(ST_SIZE, dtype=np.float64)

    hyper[ST_ABANDONMENT] = 0.45 + traits[T_ATTACHMENT_ANXIETY] * 0.275 + traits[T_JEALOUSY_SENSITIVITY] * 0.15
    hyper[ST_CLOSENESS]   = 0.30 + traits[T_ATTACHMENT_ANXIETY] * 0.25
    hyper[ST_DEPENDENCY]  = 0.35 + traits[T_ATTACHMENT_ANXIETY] * 0.20
    hyper[ST_VALIDATION]  = 0.15 + traits[T_ATTACHMENT_ANXIETY] * 0.10
    hyper[ST_CONFLICT]    = 0.15 + traits[T_ATTACHMENT_ANXIETY] * 0.15
    hyper[ST_TEASING]     = 0.10 + traits[T_JEALOUSY_SENSITIVITY] * 0.10
    hyper[ST_EMOTIONAL_WEIGHT] = 0.20 + traits[T_ATTACHMENT_ANXIETY] * 0.15

    # 敏感度: 全局增强过度激活（敏感的人所有刺激都感受更深）
    hyper += traits[T_SENSITIVITY] * 0.04
    # 依恋回避 → 全局降低过度激活（回避型抑制依恋系统激活）
    hyper -= traits[T_ATTACHMENT_AVOIDANCE] * 0.15

    # 关系调制: 好感/浪漫张力 → 放大依恋系统的反应
    hyper *= 1.0 + relationship[R_AFFECTION] * 0.09 + relationship[R_ROMANTIC_TENSION] * 0.05

    # 内部急性推动: 不安全感/渴望 → 依恋系统激活
    hyper += internal[I_INSECURITY] * 0.06 + internal[I_LONGING] * 0.04

    profiles[1] = _sigmoid(hyper - 0.50)

    return soft_clamp(profiles, 0.0, 1.0)


def apply_defenses(
    stimuli: np.ndarray,
    profiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """应用二维防御剖面，生成 inner / outer 刺激。

    每一类心理刺激经过其对应的防御剖面维度:

      inner[s]  = stimuli[s] × (1 + hyperactivation[s] × 0.50)
        → 过度激活放大内心感受: "我比看起来更在意"

      outer[s]  = inner[s] × (1 − deactivation[s] × 0.70)
        → 去激活削减外在表达: "我不想让人看出来"

    deactivation 控制 outer 削减，hyperactivation 控制 inner 放大。
    两者独立——可以内心翻江倒海但表面波澜不惊（高 hyper + 高 deact），
    也可以内心平静且表里如一（低 hyper + 低 deact）。

    Args:
        stimuli: 原始心理刺激 (7,)
        profiles: 防御剖面 (2, 7)

    Returns:
        (inner_stimuli, outer_stimuli) — 均为 (7,)
    """
    deact = profiles[0]  # (7,)
    hyper = profiles[1]  # (7,)

    # Inner: 过度激活放大内心感受
    inner = stimuli * (1.0 + hyper * 0.50)

    # Outer: 去激活削减外在表达
    outer = inner * (1.0 - deact * 0.70)

    inner = soft_clamp(inner, 0.0, 1.0)
    outer = soft_clamp(outer, 0.0, 1.0)

    return inner, outer
