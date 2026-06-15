"""① Gate Control —— 心理刺激三向门控。

输出 4 维 GateVector:
  G_SUPPRESSION (压抑强度):   决定多少情绪被压到里表情
  G_VULNERABILITY (脆弱度):   决定是否愿意示弱
  G_ATTACHMENT   (依恋敏感):  决定依恋类刺激放大倍数
  G_LEAKAGE      (保留索引，不再使用)

公式模式: gate = sigmoid( trait_baseline × rel_mod + internal_push )
  - trait_baseline: 人格决定的基线值
  - rel_mod: 关系对基线的调制因子（信任→降低压抑，安全→鼓励示弱）
  - internal_push: 内部状态的急性推动（压力→更压抑，孤独→更渴望表达）
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
    G_SUPPRESSION, G_VULNERABILITY, G_ATTACHMENT, G_LEAKAGE, G_SIZE,
)
from ._utils import soft_clamp, _sigmoid_gate


def compute_gates(
    traits: np.ndarray,
    relationship: np.ndarray,
    current_internal: np.ndarray,
) -> np.ndarray:
    """① 门控计算：特质 × 关系 × 内部状态 → 3 维门控值。"""
    gates = np.zeros(G_SIZE, dtype=np.float64)

    # 压抑强度
    # 特质：高自尊+低开放+低稳定 → 压抑强
    trait_supp = (
        traits[T_PRIDE] * 0.4
        + (1.0 - traits[T_EMOTIONAL_OPENNESS]) * 0.3
        + (1.0 - traits[T_EMOTIONAL_STABILITY]) * 0.3
    )
    # 关系：信任和安全感 → 松动防御
    rel_supp = 1.0 - relationship[R_TRUST] * 0.20 - relationship[R_EMOTIONAL_SAFETY] * 0.15
    # 急性：压力/不安 → 加重压抑
    internal_supp = current_internal[I_STRESS] * 0.10 + current_internal[I_INSECURITY] * 0.08
    gates[G_SUPPRESSION] = _sigmoid_gate(trait_supp * rel_supp + internal_supp)

    # 脆弱度
    # 特质：低自尊+高开放+高敏感 → 示弱基线高
    trait_vuln = (
        (1.0 - traits[T_PRIDE]) * 0.5
        + traits[T_EMOTIONAL_OPENNESS] * 0.3
        + traits[T_SENSITIVITY] * 0.2
    )
    # 关系：情感安全和熟悉 → 鼓励示弱
    rel_vuln = 1.0 + relationship[R_EMOTIONAL_SAFETY] * 0.15 + relationship[R_FAMILIARITY] * 0.10
    # 急性：孤独和渴望 → push 示弱意愿
    internal_vuln = current_internal[I_LONELINESS] * 0.12 + current_internal[I_LONGING] * 0.10
    gates[G_VULNERABILITY] = _sigmoid_gate(trait_vuln * rel_vuln + internal_vuln)

    # 依恋敏感
    # 特质：高依恋焦虑 + 低依恋回避 → 依恋敏感基线高
    trait_att = (
        traits[T_ATTACHMENT_ANXIETY] * 0.6
        + (1.0 - traits[T_ATTACHMENT_AVOIDANCE]) * 0.4
    )
    # 关系：好感和浪漫张力 → 放大依恋敏感
    rel_att = 1.0 + relationship[R_AFFECTION] * 0.12 + relationship[R_ROMANTIC_TENSION] * 0.08
    # 急性：不安全感和渴望 → 急性放大
    internal_att = current_internal[I_INSECURITY] * 0.10 + current_internal[I_LONGING] * 0.08
    gates[G_ATTACHMENT] = _sigmoid_gate(trait_att * rel_att + internal_att)

    return gates


def _compute_stimulus_modulation(
    traits: np.ndarray,
    relationship: np.ndarray,
) -> np.ndarray:
    """计算 7 维刺激的调制系数 = (1 + trait_amp) × (1 + rel_amp)。"""
    trait_dev = traits - 0.5
    trait_amp = np.zeros(ST_SIZE, dtype=np.float64)

    # 依恋焦虑 → 放大被抛弃恐惧和亲密靠近
    trait_amp[ST_ABANDONMENT] += trait_dev[T_ATTACHMENT_ANXIETY] * 0.5
    trait_amp[ST_CLOSENESS]   += trait_dev[T_ATTACHMENT_ANXIETY] * 0.3
    # 嫉妒敏感 → 放大被抛弃和逗弄
    trait_amp[ST_ABANDONMENT] += trait_dev[T_JEALOUSY_SENSITIVITY] * 0.4
    trait_amp[ST_TEASING]     += trait_dev[T_JEALOUSY_SENSITIVITY] * 0.2
    # 易怒 → 放大冲突
    trait_amp[ST_CONFLICT]    += trait_dev[T_ANGER_REACTIVITY] * 0.5
    trait_amp[ST_ABANDONMENT] += trait_dev[T_ANGER_REACTIVITY] * 0.2
    # 自尊 → 抑制被认可，放大被逗弄
    trait_amp[ST_VALIDATION]  += trait_dev[T_PRIDE] * -0.2
    trait_amp[ST_TEASING]     += trait_dev[T_PRIDE] * 0.3
    # 情绪稳定 → 抑制冲突和抛弃恐惧
    trait_amp[ST_CONFLICT]    += trait_dev[T_EMOTIONAL_STABILITY] * -0.3
    trait_amp[ST_ABANDONMENT] += trait_dev[T_EMOTIONAL_STABILITY] * -0.2

    rel_amp = np.zeros(ST_SIZE, dtype=np.float64)
    rel_amp[ST_ABANDONMENT] += relationship[R_EMOTIONAL_SAFETY] * -0.5
    rel_amp[ST_VALIDATION]  += relationship[R_AFFECTION] * 0.3
    rel_amp[ST_CLOSENESS]   += relationship[R_EMOTIONAL_SAFETY] * 0.2
    rel_amp[ST_CONFLICT]    += relationship[R_TRUST] * -0.3
    rel_amp[ST_DEPENDENCY]  += relationship[R_DEPENDENCY] * 0.3

    return (1.0 + trait_amp) * (1.0 + rel_amp)


def apply_gates(
    stimuli: np.ndarray,
    gates: np.ndarray,
    traits: np.ndarray,
    relationship: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """①.b 应用门控，输出 (inner_stimuli, outer_stimuli) 两套。

    里外差异化:
      - 高压抑: outer 远弱于 inner（口是心非）
      - 低压抑: outer ≈ inner（心口一致）
      - 高脆弱: outer 可超过 inner（主动示弱）
    """
    mod = _compute_stimulus_modulation(traits, relationship)
    base = stimuli * mod

    suppression = gates[G_SUPPRESSION]
    attachment = gates[G_ATTACHMENT]
    vulnerability = gates[G_VULNERABILITY]

    # 里表情：角色真正"心理上"接收到的强度（依恋类受 attachment 调制）
    inner = base.copy()
    inner[ST_ABANDONMENT] *= attachment
    inner[ST_CLOSENESS]   *= attachment

    # 外表情：角色"实际表现出"的强度
    outer = base.copy()
    outer[ST_ABANDONMENT] *= attachment
    outer[ST_CLOSENESS]   *= attachment
    # 脆弱门：示弱意愿高 → validation/closeness 更有影响
    outer[ST_VALIDATION]  *= (0.5 + vulnerability * 0.5)
    outer[ST_CLOSENESS]   *= (0.5 + vulnerability * 0.5)
    # 压抑衰减：整体压向外表情
    outer *= (1.0 - suppression * 0.6)

    inner = soft_clamp(inner, 0.0, 1.0)
    outer = soft_clamp(outer, 0.0, 1.0)

    return inner, outer
