"""Defense Profiles —— 统一防御机制建模。

核心改进：每个防御机制不是全局标量，而是一个 7 维"敏感度剖面"——
对不同类型的心理刺激有不同的激活程度。

剖面 = trait_baseline(7) × relationship_modulation + internal_push

当前防御: suppression, vulnerability, attachment
扩展方式: 新增一行 profile 即可，不影响现有门控。
"""

import numpy as np
from state import (
    T_SENSITIVITY, T_PRIDE, T_EMOTIONAL_OPENNESS, T_EMOTIONAL_STABILITY,
    T_OPTIMISM, T_ANXIETY_PRONENESS, T_ANGER_REACTIVITY, T_JEALOUSY_SENSITIVITY,
    T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE,
    I_STRESS, I_INSECURITY, I_LONELINESS, I_LONGING,
    R_AFFECTION, R_TRUST, R_FAMILIARITY, R_DEPENDENCY,
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
    """计算所有防御机制的逐维度敏感度剖面。

    Returns:
        profiles: (3, 7) np.ndarray
            profiles[0, :] — suppression: 压抑各类刺激外显的程度
            profiles[1, :] — vulnerability: 允许各类刺激"泄露"到表面的程度
            profiles[2, :] — attachment: 放大各类刺激内部感受的程度

    每个剖面 ∈ [0, 1]，值越高 = 该防御对该类刺激越活跃。
    """
    profiles = np.zeros((3, ST_SIZE), dtype=np.float64)
    t_dev = traits - 0.5  # 特质偏离中性

    # ═══════════════════════════════════════════
    # Profile 0 — Suppression（压抑）
    #   高 Pride → 压抑"暴露脆弱"的刺激（被抛弃、被认可、依赖）
    #   高 Anger → 压抑冲突（虽然生气但压着）
    #   高 Stability → 全局压抑低（情绪稳定的人不压抑，是真淡定）
    # ═══════════════════════════════════════════
    supp = np.zeros(ST_SIZE, dtype=np.float64)

    # 人格基线
    supp[ST_ABANDONMENT] = 0.35 + t_dev[T_PRIDE] * 0.50 + t_dev[T_JEALOUSY_SENSITIVITY] * 0.20
    supp[ST_VALIDATION]  = 0.25 + t_dev[T_PRIDE] * 0.40
    supp[ST_DEPENDENCY]  = 0.30 + t_dev[T_PRIDE] * 0.40 + t_dev[T_ATTACHMENT_AVOIDANCE] * 0.20
    supp[ST_CLOSENESS]   = 0.15 + t_dev[T_PRIDE] * 0.20
    supp[ST_CONFLICT]    = 0.20 + t_dev[T_PRIDE] * 0.30 + t_dev[T_ANGER_REACTIVITY] * 0.25
    supp[ST_TEASING]     = 0.20 + t_dev[T_PRIDE] * 0.35
    supp[ST_EMOTIONAL_WEIGHT] = 0.25 + t_dev[T_PRIDE] * 0.30
    # 情绪稳定 → 全局压低压抑（不需要压抑，是真稳定）
    supp -= t_dev[T_EMOTIONAL_STABILITY] * 0.20

    # 关系调制: 信任和安全感松动压抑
    rel_loosen = 1.0 - relationship[R_TRUST] * 0.25 - relationship[R_EMOTIONAL_SAFETY] * 0.18
    supp *= rel_loosen

    # 内部急性推动: 压力/不安 → 更压抑
    supp += internal[I_STRESS] * 0.10 + internal[I_INSECURITY] * 0.08

    profiles[0] = _sigmoid(supp - 0.50)

    # ═══════════════════════════════════════════
    # Profile 1 — Vulnerability（脆弱/示弱）
    #   高 Sensitivity → 更容易被情感连接触动
    #   高 Openness → 更愿意示弱
    #   低 Pride → 不介意暴露脆弱
    # ═══════════════════════════════════════════
    vuln = np.zeros(ST_SIZE, dtype=np.float64)

    vuln[ST_VALIDATION]  = 0.30 + t_dev[T_SENSITIVITY] * 0.50 + t_dev[T_EMOTIONAL_OPENNESS] * 0.30
    vuln[ST_CLOSENESS]   = 0.40 + t_dev[T_SENSITIVITY] * 0.40 + t_dev[T_EMOTIONAL_OPENNESS] * 0.30
    vuln[ST_DEPENDENCY]  = 0.30 + t_dev[T_SENSITIVITY] * 0.40
    vuln[ST_ABANDONMENT] = 0.15 + t_dev[T_SENSITIVITY] * 0.30
    vuln[ST_CONFLICT]    = 0.10
    vuln[ST_TEASING]     = 0.15 + t_dev[T_SENSITIVITY] * 0.10
    vuln[ST_EMOTIONAL_WEIGHT] = 0.25 + t_dev[T_SENSITIVITY] * 0.30
    # 高自尊 → 全局压制脆弱
    vuln -= t_dev[T_PRIDE] * 0.25

    # 关系调制: 情感安全和熟悉鼓励示弱
    vuln *= 1.0 + relationship[R_EMOTIONAL_SAFETY] * 0.20 + relationship[R_FAMILIARITY] * 0.12

    # 内部急性推动: 孤独/渴望 → 更想示弱（想被注意到）
    vuln += internal[I_LONELINESS] * 0.12 + internal[I_LONGING] * 0.10

    profiles[1] = _sigmoid(vuln - 0.45)

    # ═══════════════════════════════════════════
    # Profile 2 — Attachment（依恋敏感）
    #   高 Attachment Anxiety → 放大"关系威胁/亲近"刺激
    #   高 Jealousy → 被抛弃更敏感
    #   低 Avoidance → 不回避亲密信号
    # ═══════════════════════════════════════════
    att = np.zeros(ST_SIZE, dtype=np.float64)

    att[ST_ABANDONMENT] = 0.45 + t_dev[T_ATTACHMENT_ANXIETY] * 0.55 + t_dev[T_JEALOUSY_SENSITIVITY] * 0.30
    att[ST_CLOSENESS]   = 0.30 + t_dev[T_ATTACHMENT_ANXIETY] * 0.50
    att[ST_DEPENDENCY]  = 0.35 + t_dev[T_ATTACHMENT_ANXIETY] * 0.40
    att[ST_VALIDATION]  = 0.15 + t_dev[T_ATTACHMENT_ANXIETY] * 0.20
    att[ST_CONFLICT]    = 0.15 + t_dev[T_ATTACHMENT_ANXIETY] * 0.30
    att[ST_TEASING]     = 0.10 + t_dev[T_JEALOUSY_SENSITIVITY] * 0.20
    att[ST_EMOTIONAL_WEIGHT] = 0.20 + t_dev[T_ATTACHMENT_ANXIETY] * 0.30
    # 依恋回避 → 全局降低依恋敏感
    att -= t_dev[T_ATTACHMENT_AVOIDANCE] * 0.30

    # 关系调制: 好感/浪漫张力 → 放大依恋
    att *= 1.0 + relationship[R_AFFECTION] * 0.18 + relationship[R_ROMANTIC_TENSION] * 0.10

    # 内部急性推动: 不安全感/渴望 → 依恋系统激活
    att += internal[I_INSECURITY] * 0.12 + internal[I_LONGING] * 0.08

    profiles[2] = _sigmoid(att - 0.50)

    return soft_clamp(profiles, 0.0, 1.0)


def apply_defenses(
    stimuli: np.ndarray,
    profiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """应用防御剖面，生成 inner / outer 刺激。

    每一类心理刺激经过其对应的防御剖面维度:
      - inner[s] = stimuli[s] × (1 + attachment[s] × gain)
      - outer[s] = inner[s] × (1 − suppression[s]) × vulnerability_leak[s]

    suppression / vulnerability / attachment 现在都是 7 维向量，
    每维独立作用于对应的刺激类型。

    Returns:
        (inner_stimuli, outer_stimuli) — 均为 7 维
    """
    supp = profiles[0]  # (7,)
    vuln = profiles[1]  # (7,)
    att  = profiles[2]  # (7,)

    # Inner: 依恋敏感放大特定刺激的内部感受
    inner = stimuli * (1.0 + att * 0.50)

    # Outer: 从 inner 出发，经压抑衰减和脆弱泄露
    #   压抑: 高 suppression[s] → outer[s] 被大幅削减（"口是心非"）
    #   脆弱: 高 vulnerability[s] → 部分 bypass 压抑（"忍不住流露"）
    outer = inner * (1.0 - supp * 0.70)
    outer = outer * (0.30 + vuln * 0.70)  # vuln 高时接近原始，低时只剩 30%

    inner = soft_clamp(inner, 0.0, 1.0)
    outer = soft_clamp(outer, 0.0, 1.0)

    return inner, outer
