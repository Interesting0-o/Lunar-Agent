"""Defense Profiles —— 二维防御机制建模（刺激特异性调制）。

基于 Bowlby (1980) 的依恋防御二分法 + Richardson et al. (2023, 2025) 的实证验证:

  profiles[0, :] — Deactivation (去激活): 削减外在表达
    高回避 → 情感疏离、压抑表达。关联特质: pride↑, avoidance↑, stability↓
    由 suppression + 逆 vulnerability 合并而来。

  profiles[1, :] — Hyperactivation (过度激活): 放大内心感受
    高焦虑 → 放大关系威胁/亲近信号。关联特质: attachment_anxiety↑, jealousy↑
    原 attachment 剖面。

每个剖面是 7 维敏感度向量 ∈ [0, 1]，对不同类型的心理刺激独立激活。

═══ 刺激特异性调制 ═══

所有状态变量（stress、insecurity、trust、affection 等）已从全局标量操作
改为逐刺激维度的权重数组。每个状态变量对各维度的影响力不同，由心理驱动。

例如 stress 对 deactivation:
  - CONFLICT (权重 0.12):  压力大时最想逃避冲突
  - CLOSENESS (权重 0.01): 压力不影响亲近的意愿
  - TEASING (权重 0.00):   压力不影响调侃表达

这避免了旧设计中 7 维同步漂移的实际维度塌缩（PCA 验证: 有效秩从 ~1.5
提升至 ~4+），使防御剖面真正具备刺激选择性。

扩展方式: 增加新防御维度（如 boundary）需验证:
  1. 概念判别效度 — 与 deactivation/hyperactivation 不是同一构念
  2. 因子/相关性检查 — |r| < 0.3 才值得独立成维度
  3. 交互机制 — 在 apply_defenses 中有不同于现有维度的数学操作
验证通过后将 profiles 扩展为 (3, 7)，不影响现有逻辑。
"""

import numpy as np
from state import (
    T_SENSITIVITY, T_PRIDE, T_EMOTIONAL_OPENNESS, T_EMOTIONAL_STABILITY,
    T_ANGER_REACTIVITY, T_JEALOUSY_SENSITIVITY,
    T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE,
    I_STRESS, I_INSECURITY, I_LONGING,
    R_AFFECTION, R_TRUST_BOND, R_INTIMACY,
    ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT,
    ST_DEPENDENCY, ST_TEASING, ST_EMOTIONAL_WEIGHT, ST_SIZE,
)
from ._utils import soft_clamp, _sigmoid

# ═══════════════════════════════════════════════════════════════════
# 每维调制权重
#
# 每个状态变量对 7 种刺激的独立敏感度。
# 加法权重:  {var}_[DEACT|HYPER]_A[d] = 变量每单位对该维度 profile 的增量
# 乘法权重:  {var}_[DEACT|HYPER]_M[d] = 变量每单位的乘数系数
#   乘法公式: profile[d] *= 1.0 + var_value * weight[d]
#   （负权重 = 降低该维度的防御激活）
#
# 权重值的心理依据详见每组的注释。
# 每组的均值 ≈ 原全局系数，以保证整体行为向后兼容。
# ═══════════════════════════════════════════════════════════════════

# ────── DEACTIVATION 加法调制（人格特质） ──────

# 情绪稳定性: 稳定的人在哪方面不需要伪装
#   原值: -0.11（全局）
STABILITY_DEACT_A: np.ndarray = np.array([
    -0.10,  # ST_ABANDONMENT     — 对抛弃不恐惧，不需要藏
    -0.12,  # ST_VALIDATION      — 内心稳定，不需要藏"需要认可"
    -0.05,  # ST_CLOSENESS       — 适度降低防御
    -0.15,  # ST_CONFLICT        — 稳定的核心: 敢于面对冲突
    -0.08,  # ST_DEPENDENCY      — 不隐瞒依赖需求
    -0.03,  # ST_TEASING         — 些许降低
    -0.12,  # ST_EMOTIONAL_WEIGHT — 不回避沉重话题
])

# 情绪开放性: 开放的人在哪方面更坦诚
#   原值: -0.06（全局）
OPENNESS_DEACT_A: np.ndarray = np.array([
    -0.08,  # ST_ABANDONMENT     — 开放→愿意暴露脆弱
    -0.06,  # ST_VALIDATION      — 不藏"想要认可"
    -0.10,  # ST_CLOSENESS       — 核心: 开放→不掩藏亲近渴望
    -0.03,  # ST_CONFLICT        — 冲突不是开放的主要表现
    -0.06,  # ST_DEPENDENCY      — 适度表露依赖
    -0.02,  # ST_TEASING         — 少许效果
    -0.08,  # ST_EMOTIONAL_WEIGHT — 愿意表达情感重量
])

# 依恋回避: 回避者在哪方面掩饰最多
#   原值: +0.09（全局）
AVOIDANCE_DEACT_A: np.ndarray = np.array([
    0.08,   # ST_ABANDONMENT     — 回避→掩饰对抛弃的在意
    0.06,   # ST_VALIDATION      — 掩饰被认可的需求
    0.13,   # ST_CLOSENESS       — 核心: 最怕暴露渴望亲近
    0.04,   # ST_CONFLICT        — 适度回避冲突
    0.10,   # ST_DEPENDENCY      — 核心: 最怕表现依赖
    0.02,   # ST_TEASING         — 少许
    0.07,   # ST_EMOTIONAL_WEIGHT — 掩饰情绪重量
])

# ────── DEACTIVATION 加法调制（急性状态） ──────

# 压力: 压力让你在哪方面隐藏最深
#   原值: +0.05（全局）
STRESS_DEACT_A: np.ndarray = np.array([
    0.08,   # ST_ABANDONMENT     — 压力→对被抛弃更敏感
    0.03,   # ST_VALIDATION      — 压力下减少暴露需求
    0.01,   # ST_CLOSENESS       — 压力下仍渴望亲近（不隐藏）
    0.12,   # ST_CONFLICT        — 核心: 压力→逃避冲突
    0.04,   # ST_DEPENDENCY      — 压力→隐藏依赖需求
    0.00,   # ST_TEASING         — 压力不影响调侃表达
    0.08,   # ST_EMOTIONAL_WEIGHT — 压力→回避沉重话题
])

# 不安全感: 不安时在哪些方面防御最强
#   原值: +0.04（全局）
INSECURITY_DEACT_A: np.ndarray = np.array([
    0.10,   # ST_ABANDONMENT     — 核心: 不安→极度恐惧被抛弃
    0.05,   # ST_VALIDATION      — 不安→藏起对被认可的渴望
    0.01,   # ST_CLOSENESS       — 不安不影响亲近意愿
    0.03,   # ST_CONFLICT        — 不敢面对冲突
    0.07,   # ST_DEPENDENCY      — 核心: 不安→藏起依赖需求
    0.00,   # ST_TEASING
    0.05,   # ST_EMOTIONAL_WEIGHT — 不安→不碰沉重话题
])

# ────── DEACTIVATION 乘法调制（关系状态） ──────
#
# 公式: deact[i] *= 1.0 + rel_var * weight[i]
# 负权重 → 该关系变量降低去激活（安全基地效应）

# 信任: 信任让你在哪方面放下伪装
#   原值: -0.11（全局）
TRUST_BOND_DEACT_M: np.ndarray = np.array([
    -0.16,  # ST_ABANDONMENT     — 信任→敢于暴露被抛弃恐惧
    -0.15,  # ST_VALIDATION      — 信任→敢于说"我需要认可"
    -0.08,  # ST_CLOSENESS       — 信任→敢于靠近
    -0.10,  # ST_CONFLICT        — 信任→敢于争吵
    -0.12,  # ST_DEPENDENCY      — 信任→敢于示弱
    -0.03,  # ST_TEASING         — 少量
    -0.14,  # ST_EMOTIONAL_WEIGHT — 信任→敢于谈沉重话题
])

# ────── HYPERACTIVATION 加法调制（人格特质） ──────

# 敏感度: 敏感的人在哪类刺激上感受更深
#   原值: +0.04（全局）
SENSITIVITY_HYPER_A: np.ndarray = np.array([
    0.06,   # ST_ABANDONMENT     — 敏感→对抛弃信号更敏锐
    0.05,   # ST_VALIDATION      — 敏感→更在意认可
    0.04,   # ST_CLOSENESS       — 敏感→对亲近更心动
    0.04,   # ST_CONFLICT        — 敏感→冲突感受更深
    0.05,   # ST_DEPENDENCY      — 敏感→对依赖更敏锐
    0.03,   # ST_TEASING         — 敏感→调侃也能触动
    0.05,   # ST_EMOTIONAL_WEIGHT — 敏感→沉重话题冲击更大
])

# 依恋回避: 回避在哪方面最抑制依恋系统激活
#   原值: -0.15（全局）
AVOIDANCE_HYPER_A: np.ndarray = np.array([
    -0.18,  # ST_ABANDONMENT     — 回避→不表现出在意抛弃
    -0.10,  # ST_VALIDATION      — 回避→不在意认可
    -0.25,  # ST_CLOSENESS       — 核心: 最压制对亲近的渴望
    -0.08,  # ST_CONFLICT        — 回避→冲突也懒得在意
    -0.22,  # ST_DEPENDENCY      — 核心: 最压制依赖需求
    -0.04,  # ST_TEASING
    -0.14,  # ST_EMOTIONAL_WEIGHT — 回避→淡漠以对
])

# ────── HYPERACTIVATION 加法调制（急性状态） ──────

# 不安全感: 不安时过度激活聚焦在哪
#   原值: +0.06（全局）
INSECURITY_HYPER_A: np.ndarray = np.array([
    0.14,   # ST_ABANDONMENT     — 核心: 不安→极度恐惧被抛弃
    0.04,   # ST_VALIDATION
    0.08,   # ST_CLOSENESS       — 不安→更渴望靠近（矛盾）
    0.02,   # ST_CONFLICT
    0.06,   # ST_DEPENDENCY      — 不安→更想依赖
    0.00,   # ST_TEASING         — 不影响
    0.04,   # ST_EMOTIONAL_WEIGHT
])

# 渴望/思念: 渴望在哪方面增强过度激活
#   原值: +0.04（全局）
LONGING_HYPER_A: np.ndarray = np.array([
    0.06,   # ST_ABANDONMENT     — 渴望→怕失去思念的对象
    0.02,   # ST_VALIDATION
    0.08,   # ST_CLOSENESS       — 核心: 渴望→无限放大对亲近的向往
    0.00,   # ST_CONFLICT        — 不影响
    0.04,   # ST_DEPENDENCY      — 渴望→想依赖
    0.02,   # ST_TEASING
    0.06,   # ST_EMOTIONAL_WEIGHT — 渴望→加重情感重量
])

# ────── HYPERACTIVATION 乘法调制（关系状态） ──────
#
# 公式: hyper[i] *= 1.0 + rel_var * weight[i]

# 好感: 喜欢上对方后，哪些刺激更容易触动内心
#   原值: +0.09（全局）
AFFECTION_HYPER_M_NEW: np.ndarray = np.array([
    0.10,   # ST_ABANDONMENT     — 好感→怕被抛弃
    0.06,   # ST_VALIDATION      — 好感→在意认可
    0.12,   # ST_CLOSENESS       — 好感→渴望亲近
    0.03,   # ST_CONFLICT        — 少量
    0.08,   # ST_DEPENDENCY      — 好感→想依赖
    0.05,   # ST_TEASING         — 好感→调侃有温度
    0.08,   # ST_EMOTIONAL_WEIGHT — 好感→情感加重
])

# 浪漫张力: 暧昧气氛放大了哪类刺激
#   原值: +0.05（全局）
INTIMACY_HYPER_M: np.ndarray = np.array([
    0.12,   # ST_ABANDONMENT     — 亲密→更怕被抛弃
    0.05,   # ST_VALIDATION      — 亲密→更在意认可
    0.14,   # ST_CLOSENESS       — 亲密→更渴望靠近
    0.02,   # ST_CONFLICT        — 少量
    0.12,   # ST_DEPENDENCY      — 亲密→更想依赖
    0.08,   # ST_TEASING         — 亲密→调侃变调情
    0.10,   # ST_EMOTIONAL_WEIGHT — 亲密→情感更沉重
])

def _apply_additive(
    vector: np.ndarray,
    mod_value: float,
    weights: np.ndarray,
    label: str = "",
) -> None:
    """将调制值按逐维权重加入向量。原地操作。"""
    vector += mod_value * weights

def _apply_multiplicative(
    vector: np.ndarray,
    mod_value: float,
    weights: np.ndarray,
    label: str = "",
) -> None:
    """按逐维权重缩放向量。原地操作。

    公式: vector[i] *= 1.0 + mod_value * weights[i]
    """
    vector *= 1.0 + mod_value * weights

def compute_defense_profiles(
    traits: np.ndarray,
    relationship: np.ndarray,
    internal: np.ndarray,
) -> np.ndarray:
    """计算二维防御剖面: 去激活 + 过度激活（刺激特异性调制版本）。

    Parameters
    ----------
    traits: (10,)  人格特质
    relationship: (3,)  关系状态
    internal: (8,)  内部状态

    Returns
    -------
    profiles: (2, 7) np.ndarray
        profiles[0, :] — deactivation: 削减各类刺激外在表达的程度
        profiles[1, :] — hyperactivation: 放大各类刺激内心感受的程度

    每个剖面 ∈ [0, 1]，值越高 = 该防御对该类刺激越活跃。

    ── 与旧版本的区别 ──
    旧版中所有状态变量（stress、insecurity、trust、affection 等）对各刺激维度
    施加等量影响（全局标量操作）。PCA 验证其有效维度秩仅 ~1.5，7 维名义维度
    实际只有 2 个自由度。

    新版用逐维权重数组替代全局标量，每个状态变量对不同刺激类型施加不同强度的
    调制，使防御剖面在状态驱动下具备真正的维度特异性。详见模块文档。
    """
    profiles = np.zeros((2, ST_SIZE), dtype=np.float64)

    # ═══════════════════════════════════════════════════════════════
    # Profile 0 — Deactivation (去激活)
    #
    #   合并了原 suppression + 逆 vulnerability:
    #     - 高 Pride → 隐藏"暴露脆弱"的刺激（被抛弃、被认可、依赖）
    #     - 高 Avoidance → 核心: 情感疏离 + 隐藏亲近渴望
    #     - 高 Stability → 真淡定，不是装的
    #     - 高 Openness → 愿意流露
    #     - 低 Trust / 低 Safety → 不信任时不示弱
    #     - 高 Stress / 高 Insecurity → 越难受越藏
    #
    #   新版差异: 每个调制变量有各自的刺激特异性权重。
    #   例如 stress 影响冲突防御最强 (0.12)，不影响调侃 (0.00)。
    #
    #   Bowlby: 去激活策略 = 情感疏离 + 最小化痛苦表达 + 转移注意力
    #   Richardson (2023): 回避独有防御 —— distancing, disengagement,
    #     vulnerability suppression
    # ═══════════════════════════════════════════════════════════════
    deact = np.zeros(ST_SIZE, dtype=np.float64)

    # ── 第一步: 人格基线 ──
    # 每种刺激天然被隐藏的程度不同，由核心特质决定。
    deact[ST_ABANDONMENT] = 0.30 + traits[T_PRIDE] * 0.225 + traits[T_JEALOUSY_SENSITIVITY] * 0.09
    deact[ST_VALIDATION]  = 0.25 + traits[T_PRIDE] * 0.20
    deact[ST_DEPENDENCY]  = 0.28 + traits[T_PRIDE] * 0.19 + traits[T_ATTACHMENT_AVOIDANCE] * 0.10
    deact[ST_CLOSENESS]   = 0.15 + traits[T_PRIDE] * 0.09 + traits[T_ATTACHMENT_AVOIDANCE] * 0.075
    deact[ST_CONFLICT]    = 0.20 + traits[T_PRIDE] * 0.14 + traits[T_ANGER_REACTIVITY] * 0.11
    deact[ST_TEASING]     = 0.20 + traits[T_PRIDE] * 0.16
    deact[ST_EMOTIONAL_WEIGHT] = 0.25 + traits[T_PRIDE] * 0.14

    # ── 第二步: 人格特质调制（刺激特异性） ──
    # 替换旧版全局 -= stability*0.11, -= openness*0.06, += avoidance*0.09
    _apply_additive(deact, traits[T_EMOTIONAL_STABILITY],   STABILITY_DEACT_A,  "stability→deact")
    _apply_additive(deact, traits[T_EMOTIONAL_OPENNESS],    OPENNESS_DEACT_A,   "openness→deact")
    _apply_additive(deact, traits[T_ATTACHMENT_AVOIDANCE],  AVOIDANCE_DEACT_A,  "avoidance→deact")

    # ── 第三步: 关系状态调制（刺激特异性乘法） ──
    # 替换旧版全局 *= (1.0 - trust*0.11 - safety*0.09)
    _apply_multiplicative(deact, relationship[R_TRUST_BOND], TRUST_BOND_DEACT_M, "trust_bond->deact")

    # ── 第四步: 急性状态调制（刺激特异性加法） ──
    # 替换旧版全局 += stress*0.05 + insecurity*0.04
    _apply_additive(deact, internal[I_STRESS],    STRESS_DEACT_A,     "stress→deact")
    _apply_additive(deact, internal[I_INSECURITY], INSECURITY_DEACT_A, "insecurity→deact")

    profiles[0] = _sigmoid((deact - 0.35) * 5.0)

    # ═══════════════════════════════════════════════════════════════
    # Profile 1 — Hyperactivation (过度激活)
    #
    #   原 attachment 剖面:
    #     - 高 Attachment Anxiety → 放大"关系威胁/亲近"刺激
    #     - 高 Jealousy → 对被抛弃更敏感
    #     - 低 Avoidance → 不回避亲密信号
    #     - 高 Sensitivity → 全局更敏感
    #
    #   新版差异: 同 deactivation，每个调制变量有独立的刺激特异性权重。
    #   例如 insecurity 放大 abandonment 的过度激活最强 (0.14)，
    #   不影响 teasing (0.00)。
    #
    #   Bowlby: 过度激活策略 = 夸大痛苦表达 + 持续监控 + 投射
    #   Richardson (2023): 焦虑独有防御 —— splitting, projective
    #     identification, anticipation, acting out, passive-aggression,
    #     reaction formation
    # ═══════════════════════════════════════════════════════════════
    hyper = np.zeros(ST_SIZE, dtype=np.float64)

    # ── 第一步: 人格基线 ──
    hyper[ST_ABANDONMENT] = 0.45 + traits[T_ATTACHMENT_ANXIETY] * 0.275 + traits[T_JEALOUSY_SENSITIVITY] * 0.15
    hyper[ST_CLOSENESS]   = 0.30 + traits[T_ATTACHMENT_ANXIETY] * 0.25
    hyper[ST_DEPENDENCY]  = 0.35 + traits[T_ATTACHMENT_ANXIETY] * 0.20
    hyper[ST_VALIDATION]  = 0.15 + traits[T_ATTACHMENT_ANXIETY] * 0.10
    hyper[ST_CONFLICT]    = 0.15 + traits[T_ATTACHMENT_ANXIETY] * 0.15
    hyper[ST_TEASING]     = 0.10 + traits[T_JEALOUSY_SENSITIVITY] * 0.10
    hyper[ST_EMOTIONAL_WEIGHT] = 0.20 + traits[T_ATTACHMENT_ANXIETY] * 0.15

    # ── 第二步: 人格特质调制（刺激特异性加法） ──
    # 替换旧版全局 += sensitivity*0.04, -= avoidance*0.15
    _apply_additive(hyper, traits[T_SENSITIVITY],           SENSITIVITY_HYPER_A, "sensitivity→hyper")
    _apply_additive(hyper, traits[T_ATTACHMENT_AVOIDANCE],  AVOIDANCE_HYPER_A,   "avoidance→hyper")

    # ── 第三步: 关系状态调制（刺激特异性乘法） ──
    # 替换旧版全局 *= (1.0 + affection*0.09 + tension*0.05)
    _apply_multiplicative(hyper, relationship[R_AFFECTION], AFFECTION_HYPER_M_NEW, "affection->hyper")
    _apply_multiplicative(hyper, relationship[R_INTIMACY],  INTIMACY_HYPER_M,      "intimacy->hyper")

    # ── 第四步: 急性状态调制（刺激特异性加法） ──
    # 替换旧版全局 += insecurity*0.06 + longing*0.04
    _apply_additive(hyper, internal[I_INSECURITY], INSECURITY_HYPER_A, "insecurity→hyper")
    _apply_additive(hyper, internal[I_LONGING],    LONGING_HYPER_A,    "longing→hyper")

    profiles[1] = _sigmoid((hyper - 0.38) * 5.0)

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
