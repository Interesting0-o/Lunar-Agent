"""Dynamics 权重 —— 自阻尼/衰减目标/耦合系数的 provenance 集中管理。

约束⑤(语义映射层) + 约束⑧(参数审计) 合规：
  - SELF_DECAY / REL_SELF_DECAY / DECAY_TARGETS → WeightVector
  - 内部耦合 / 关系耦合 / 跨尺度耦合 → WeightMapper (稀疏矩阵)
  - 所有条目带心理学依据和起源
"""

import numpy as np
from state import (
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE,
    I_SIZE, I_LABELS,
    R_AFFECTION, R_TRUST_BOND, R_INTIMACY,
    R_SIZE, R_LABELS,
)
from ._validator import WeightVector, WeightMapper, LinearMapping, register_mapper


# ═══════════════════════════════════════════════════════════════════
# SELF_DECAY — 内部状态自阻尼率（每维独立）
# ═══════════════════════════════════════════════════════════════════

def _build_self_decay() -> WeightVector:
    """内部状态每维自阻尼率，控制状态向 0 收敛的速度。"""
    vec = WeightVector(
        "SELF_DECAY", I_LABELS,
        "内部状态自阻尼率（每维独立，替代旧统一 0.15）",
    )
    # 正值 setpoint 维度阻尼更小，避免隐性"税"
    vec.connect("energy", I_ENERGY, 0.10, "weak", (0.05, 0.15),
                "高基线(+0.4)→慢衰减，防空转税", "calibrated", "2026-06-20")
    vec.connect("stress", I_STRESS, 0.12, "weak", (0.08, 0.18),
                "负基线(-0.56)→中等衰减", "calibrated", "2026-06-20")
    vec.connect("loneliness", I_LONELINESS, 0.12, "weak", (0.08, 0.18),
                "负基线(-0.40)→中等衰减", "calibrated", "2026-06-20")
    vec.connect("insecurity", I_INSECURITY, 0.12, "weak", (0.08, 0.18),
                "负基线(-0.46)→中等衰减", "calibrated", "2026-06-20")
    vec.connect("irritation", I_IRRITATION, 0.12, "weak", (0.08, 0.18),
                "负基线(-0.80)→中等持续", "calibrated", "2026-06-20")
    vec.connect("longing", I_LONGING, 0.12, "weak", (0.08, 0.18),
                "负基线(-0.19)→中等衰减", "calibrated", "2026-06-20")
    vec.connect("social_battery", I_SOCIAL_BATTERY, 0.10, "weak", (0.05, 0.15),
                "正基线(+0.20)→慢衰减（由DECAY_TARGETS补正）", "calibrated", "2026-06-20")
    vec.connect("mental_fatigue", I_MENTAL_FATIGUE, 0.12, "weak", (0.08, 0.18),
                "负基线(-0.71)→中等衰减", "calibrated", "2026-06-20")

    vec.build()
    register_mapper(vec)
    return vec


# ═══════════════════════════════════════════════════════════════════
# DECAY_TARGETS — 自阻尼收敛目标（非零维度：social_battery）
# ═══════════════════════════════════════════════════════════════════

def _build_decay_targets() -> WeightVector:
    """自阻尼收敛目标。大部分维度向 0 收敛，social_battery 有健康基线。"""
    vec = WeightVector(
        "DECAY_TARGETS", I_LABELS,
        "自阻尼收敛目标（只有 social_battery 非零）",
    )
    # social_battery 的健康基线 ≈ DEFAULT_INTERNAL 中值 + 稳定性增益
    vec.connect("social_battery", I_SOCIAL_BATTERY, 0.20, "weak", (0.10, 0.30),
                "社交电量健康基线——允许恢复而非归零", "calibrated", "2026-06-20")

    vec.build()
    register_mapper(vec)
    return vec


# ═══════════════════════════════════════════════════════════════════
# REL_SELF_DECAY — 关系状态自阻尼率
# ═══════════════════════════════════════════════════════════════════

def _build_rel_self_decay() -> WeightVector:
    """关系状态自阻尼率（3 维独立）。"""
    vec = WeightVector(
        "REL_SELF_DECAY", R_LABELS,
        "关系状态自阻尼率（替代旧统一 0.1503）",
    )
    vec.connect("affection", R_AFFECTION, 0.12, "weak", (0.08, 0.18),
                "好感衰减适中", "calibrated", "2026-06-20")
    vec.connect("trust_bond", R_TRUST_BOND, 0.12, "weak", (0.08, 0.18),
                "信任/安全感衰减慢", "calibrated", "2026-06-20")
    vec.connect("intimacy", R_INTIMACY, 0.10, "weak", (0.05, 0.15),
                "亲密张力衰减最慢（慢热慢冷）", "calibrated", "2026-06-20")

    vec.build()
    register_mapper(vec)
    return vec


# ═══════════════════════════════════════════════════════════════════
# 内部耦合矩阵 — 8→8 稀疏权重
# ═══════════════════════════════════════════════════════════════════

def _build_internal_coupling() -> np.ndarray:
    """内部跨维度耦合（10 条规则，每行附带心理学依据）。"""
    mapper = WeightMapper(
        "INTERNAL_COUPLING",
        source_labels=I_LABELS, target_labels=I_LABELS,
        description="内部态→内部态跨维度耦合（8×8 稀疏）",
    )
    # 精力→压力↓
    mapper.connect(I_ENERGY, I_STRESS, -0.05, "trace", (-0.10, -0.02),
                   "精力充沛→压力降低", "theory", "2026-06-21")
    # 不安全感→压力↑
    mapper.connect(I_INSECURITY, I_STRESS, 0.10, "weak", (0.04, 0.16),
                   "不安全感→压力上升", "theory", "2026-06-21")
    # 精力→孤独↓
    mapper.connect(I_ENERGY, I_LONELINESS, -0.05, "trace", (-0.10, -0.02),
                   "精力充沛→孤独感降低", "calibrated", "2026-06-21")
    # 压力→孤独↑
    mapper.connect(I_STRESS, I_LONELINESS, 0.08, "trace", (0.03, 0.14),
                   "压力→孤独感", "theory", "2026-06-21")
    # 孤独→不安↑
    mapper.connect(I_LONELINESS, I_INSECURITY, 0.12, "weak", (0.05, 0.18),
                   "孤独→不安全感上升", "theory", "2026-06-21")
    # 压力→烦躁↑
    mapper.connect(I_STRESS, I_IRRITATION, 0.15, "weak", (0.08, 0.22),
                   "压力积累→易怒", "theory", "2026-06-21")
    # 社交电量→烦躁↓
    mapper.connect(I_SOCIAL_BATTERY, I_IRRITATION, -0.08, "trace", (-0.14, -0.03),
                   "社交电量低→烦躁", "theory", "2026-06-21")
    # 孤独→思念↑
    mapper.connect(I_LONELINESS, I_LONGING, 0.15, "weak", (0.08, 0.22),
                   "孤独→思念", "theory", "2026-06-21")
    # 精力→社交电量↑
    mapper.connect(I_ENERGY, I_SOCIAL_BATTERY, 0.08, "trace", (0.03, 0.14),
                   "精力充沛→电量恢复", "calibrated", "2026-06-21")
    # 压力→精神疲劳↑
    mapper.connect(I_STRESS, I_MENTAL_FATIGUE, 0.10, "weak", (0.04, 0.16),
                   "压力→精神疲劳", "theory", "2026-06-21")
    # 社交电量→精神疲劳↓
    mapper.connect(I_SOCIAL_BATTERY, I_MENTAL_FATIGUE, -0.10, "weak", (-0.16, -0.04),
                   "社交电量低→疲劳", "theory", "2026-06-21")

    M = mapper.build_matrix(
        (I_SIZE, I_SIZE),
        skip_sparsity=True,       # 命名规则稀疏性由规则数保证
        skip_orthogonality=True,  # 耦合矩阵不要求正交
        skip_spectral=True,       # 谱半径由非线性 soft_clamp 保证
    )
    register_mapper(mapper)
    return M


# ═══════════════════════════════════════════════════════════════════
# 关系耦合矩阵 — 3→3 稀疏权重
# ═══════════════════════════════════════════════════════════════════

def _build_relationship_coupling() -> np.ndarray:
    """关系耦合（6 条规则：4 正 + 2 负拮抗）。"""
    mapper = WeightMapper(
        "RELATIONSHIP_COUPLING",
        source_labels=R_LABELS, target_labels=R_LABELS,
        description="关系态→关系态耦合（3×3）",
    )
    # 好感→信任↑
    mapper.connect(R_AFFECTION, R_TRUST_BOND, 0.08, "trace", (0.03, 0.14),
                   "好感构建信任感", "theory", "2026-06-21")
    # 信任→好感↑（安全基地效应）
    mapper.connect(R_TRUST_BOND, R_AFFECTION, 0.04, "trace", (0.01, 0.08),
                   "安全基地效应：信任让人更亲近", "theory", "2026-06-21")
    # 好感→亲密↑
    mapper.connect(R_AFFECTION, R_INTIMACY, 0.035, "trace", (0.01, 0.07),
                   "喜欢让人想靠近", "theory", "2026-06-21")
    # 信任→亲密↑
    mapper.connect(R_TRUST_BOND, R_INTIMACY, 0.04, "trace", (0.01, 0.08),
                   "信任允许深入", "theory", "2026-06-21")
    # 亲密→信任↓（拮抗负边）
    mapper.connect(R_INTIMACY, R_TRUST_BOND, -0.02, "trace", (-0.05, -0.01),
                   "过度亲密→安全感下降（拮抗）", "theory", "2026-06-21")
    # 亲密→好感↓（拮抗负边）
    mapper.connect(R_INTIMACY, R_AFFECTION, -0.02, "trace", (-0.05, -0.01),
                   "张力过载→伤好感（拮抗）", "theory", "2026-06-21")

    M = mapper.build_matrix(
        (R_SIZE, R_SIZE),
        skip_sparsity=True,
        skip_orthogonality=True,
        skip_spectral=True,
    )
    register_mapper(mapper)
    return M


# ═══════════════════════════════════════════════════════════════════
# 跨尺度耦合矩阵 — 8→3 内部态→关系态
# ═══════════════════════════════════════════════════════════════════

def _build_cross_scale_coupling() -> np.ndarray:
    """跨尺度耦合（内部态→关系态，5 条规则）。"""
    mapper = WeightMapper(
        "CROSS_SCALE_COUPLING",
        source_labels=I_LABELS, target_labels=R_LABELS,
        description="内部态→关系态跨尺度耦合（8×3）",
    )
    # 压力→信任↓
    mapper.connect(I_STRESS, R_TRUST_BOND, -0.03, "trace", (-0.06, -0.01),
                   "压力→信任感下降", "theory", "2026-06-21")
    # 压力→亲密↑
    mapper.connect(I_STRESS, R_INTIMACY, 0.015, "trace", (0.005, 0.03),
                   "压力→张力上升", "calibrated", "2026-06-21")
    # 精力→好感↑
    mapper.connect(I_ENERGY, R_AFFECTION, 0.015, "trace", (0.005, 0.03),
                   "精力充沛→好感上升", "calibrated", "2026-06-21")
    # 不安→亲密↑
    mapper.connect(I_INSECURITY, R_INTIMACY, 0.02, "trace", (0.01, 0.04),
                   "不安→渴望靠近", "theory", "2026-06-21")
    # 孤独→张力↑
    mapper.connect(I_LONELINESS, R_INTIMACY, 0.02, "trace", (0.01, 0.04),
                   "孤独→张力上升", "theory", "2026-06-21")

    M = mapper.build_matrix(
        (I_SIZE, R_SIZE),
        skip_sparsity=True,
        skip_orthogonality=True,
    )
    register_mapper(mapper)
    return M


# ═══════════════════════════════════════════════════════════════════
# α (alpha) — 内部态跨维度耦合速率
# ═══════════════════════════════════════════════════════════════════

def _build_alpha_internal() -> LinearMapping:
    """内部态 α 速率: 0.285 + openness*0.15 - stability*0.075 + trust_bond*0.06

    α 决定跨维度耦合的强度。由人格特质和当前信任关系调制。
    """
    from state import T_LABELS
    lm = LinearMapping("ALPHA_INTERNAL", ["alpha"],
                       "内部态跨维度耦合速率 (traits+relationship 调制)")
    lm.add_source_group("traits", T_LABELS)         # offset 0, 10
    lm.add_source_group("relationship", R_LABELS)    # offset 10, 3

    lm.set_bias(0, 0.285, (0.10, 0.40),
                "基础耦合速率", "calibrated", "2026-06-21")
    lm.connect("traits", "emotional_openness", 0, 0.15, "weak", (0.05, 0.25),
               "开放→耦合加快（喜怒形于色）", "theory", "2026-06-21")
    lm.connect("traits", "emotional_stability", 0, -0.075, "trace", (-0.15, -0.02),
               "稳定→耦合更慢（情绪独立）", "theory", "2026-06-21")
    lm.connect("relationship", "trust_bond", 0, 0.06, "trace", (0.02, 0.12),
               "信任→耦合加快（敞开心扉）", "theory", "2026-06-21")

    lm.build(skip_sparsity=True, skip_orthogonality=True)
    register_mapper(lm)
    return lm


# ═══════════════════════════════════════════════════════════════════
# α_rel — 关系态跨维度耦合速率
# ═══════════════════════════════════════════════════════════════════

def _build_alpha_relationship() -> LinearMapping:
    """关系态 α_rel 速率: 0.045 + openness*0.02 + trust_bond*0.015

    关系变化比内部状态慢 5-10 倍（α_rel << α）。
    """
    from state import T_LABELS
    lm = LinearMapping("ALPHA_RELATIONSHIP", ["alpha_rel"],
                       "关系态跨维度耦合速率 (traits+relationship 调制)")
    lm.add_source_group("traits", T_LABELS)
    lm.add_source_group("relationship", R_LABELS)

    lm.set_bias(0, 0.045, (0.01, 0.08),
                "基础关系耦合速率（慢速）", "calibrated", "2026-06-21")
    lm.connect("traits", "emotional_openness", 0, 0.02, "trace", (0.005, 0.04),
               "开放→关系变化略快", "calibrated", "2026-06-21")
    lm.connect("relationship", "trust_bond", 0, 0.015, "trace", (0.005, 0.03),
               "信任→关系更容易被影响（信任加速一切）", "theory", "2026-06-21")

    lm.build(skip_sparsity=True, skip_orthogonality=True)
    register_mapper(lm)
    return lm


# ═══════════════════════════════════════════════════════════════════
# β_rel — 关系态刺激接受速率
# ═══════════════════════════════════════════════════════════════════

def _build_beta_relationship() -> LinearMapping:
    """β_rel: 0.0275 + anxiety*0.0075

    高依恋焦虑→对关系信号更敏感。
    """
    from state import T_LABELS
    lm = LinearMapping("BETA_RELATIONSHIP", ["beta_rel"],
                       "关系态刺激接受速率 (traits 调制)")
    lm.add_source_group("traits", T_LABELS)

    lm.set_bias(0, 0.0275, (0.01, 0.05),
                "基础关系刺激接受率（慢速）", "calibrated", "2026-06-21")
    lm.connect("traits", "attachment_anxiety", 0, 0.0075, "trace", (0.002, 0.015),
               "焦虑→对关系信号更敏感", "theory", "2026-06-21")

    lm.build(skip_sparsity=True, skip_orthogonality=True)
    register_mapper(lm)
    return lm


# ═══════════════════════════════════════════════════════════════════
# β_stim 基础值 + 防御增益 — 内部态刺激接受向量
# ═══════════════════════════════════════════════════════════════════

def _build_beta_base() -> WeightVector:
    """β 基础接受率（每刺激维度 0.05）。

    β_stim[i] = BETA_BASE[i] + hyper[i] * HYPER_BETA_GAIN + deact[i] * DEACT_BETA_GAIN
    """
    from state import ST_LABELS
    vec = WeightVector("BETA_BASE", ST_LABELS,
                       "刺激接受率基础值（每维 0.05，由防御剖面调制）")
    for i, label in enumerate(ST_LABELS):
        vec.connect("beta_base", i, 0.05, "trace", (0.02, 0.10),
                    f"{label} 基础接受率", "calibrated", "2026-06-21")
    vec.build()
    register_mapper(vec)
    return vec


def _build_hyper_beta_gain() -> WeightVector:
    """hyper→β 增益系数: 0.35

    hyperactivation[i] 每增加 1.0，β_stim[i] 增加 0.35。
    """
    vec = WeightVector("HYPER_BETA_GAIN", ["gain"],
                       "过度激活→β 增益（hyper 每 +1，β 上升 0.35）")
    vec.connect("hyper_beta_gain", 0, 0.35, "moderate", (0.20, 0.50),
                "hyper 放大刺激接受——核心机制", "theory", "2026-06-21")
    vec.build()
    register_mapper(vec)
    return vec


def _build_deact_beta_gain() -> WeightVector:
    """deact→β 增益系数: -0.15

    deactivation[i] 每增加 1.0，β_stim[i] 降低 0.15。
    """
    vec = WeightVector("DEACT_BETA_GAIN", ["gain"],
                       "去激活→β 增益（deact 每 +1，β 下降 0.15）")
    vec.connect("deact_beta_gain", 0, -0.15, "weak", (-0.25, -0.05),
                "deact 压制刺激接受——核心机制", "theory", "2026-06-21")
    vec.build()
    register_mapper(vec)
    return vec


# ═══════════════════════════════════════════════════════════════════
# Setpoint — 内部态人格基线
# ═══════════════════════════════════════════════════════════════════

def _build_setpoint_mapper() -> LinearMapping:
    """内部态 setpoint: DEFAULT_INTERNAL + traits 调制。

    不同人格有不同的"正常"情绪水平。
    """
    from state import T_LABELS, DEFAULT_INTERNAL
    lm = LinearMapping("SETPOINT_INTERNAL", I_LABELS,
                       "内部态人格基线（DEFAULT_INTERNAL + traits 调制）")
    lm.add_source_group("traits", T_LABELS)

    # 偏置 = DEFAULT_INTERNAL
    lm.set_bias(I_ENERGY, DEFAULT_INTERNAL[I_ENERGY], (-1.0, 1.0),
                "精力基线", "theory", "2026-06-21")
    lm.set_bias(I_STRESS, DEFAULT_INTERNAL[I_STRESS], (-1.0, 1.0),
                "压力基线", "theory", "2026-06-21")
    lm.set_bias(I_LONELINESS, DEFAULT_INTERNAL[I_LONELINESS], (-1.0, 1.0),
                "孤独基线", "theory", "2026-06-21")
    lm.set_bias(I_INSECURITY, DEFAULT_INTERNAL[I_INSECURITY], (-1.0, 1.0),
                "不安基线", "theory", "2026-06-21")
    lm.set_bias(I_IRRITATION, DEFAULT_INTERNAL[I_IRRITATION], (-1.0, 1.0),
                "烦躁基线", "theory", "2026-06-21")
    lm.set_bias(I_LONGING, DEFAULT_INTERNAL[I_LONGING], (-1.0, 1.0),
                "思念基线", "theory", "2026-06-21")
    lm.set_bias(I_SOCIAL_BATTERY, DEFAULT_INTERNAL[I_SOCIAL_BATTERY], (-1.0, 1.0),
                "社交电量基线", "theory", "2026-06-21")
    lm.set_bias(I_MENTAL_FATIGUE, DEFAULT_INTERNAL[I_MENTAL_FATIGUE], (-1.0, 1.0),
                "精神疲劳基线", "theory", "2026-06-21")

    # trait → internal 调制
    lm.connect("traits", "optimism", I_ENERGY, 0.15, "weak", (0.05, 0.25),
               "乐观→精力基线上升", "theory", "2026-06-21")
    lm.connect("traits", "anxiety_proneness", I_ENERGY, -0.08, "trace", (-0.15, -0.03),
               "焦虑→精力基线下降", "theory", "2026-06-21")

    lm.connect("traits", "anxiety_proneness", I_STRESS, 0.20, "weak", (0.10, 0.30),
               "焦虑→压力基线上升", "theory", "2026-06-21")
    lm.connect("traits", "anger_reactivity", I_STRESS, 0.05, "trace", (0.01, 0.10),
               "易怒→用压力表现", "calibrated", "2026-06-21")

    lm.connect("traits", "attachment_anxiety", I_LONELINESS, 0.10, "weak", (0.04, 0.18),
               "依恋焦虑→孤独基线上升", "theory", "2026-06-21")
    lm.connect("traits", "optimism", I_LONELINESS, -0.05, "trace", (-0.12, -0.01),
               "乐观→孤独基线下降", "calibrated", "2026-06-21")

    lm.connect("traits", "attachment_anxiety", I_INSECURITY, 0.20, "weak", (0.10, 0.30),
               "依恋焦虑→不安基线上升", "theory", "2026-06-21")
    lm.connect("traits", "anxiety_proneness", I_INSECURITY, 0.10, "weak", (0.04, 0.18),
               "焦虑→不安基线上升", "theory", "2026-06-21")

    lm.connect("traits", "anger_reactivity", I_IRRITATION, 0.15, "weak", (0.05, 0.25),
               "易怒→烦躁基线上升", "theory", "2026-06-21")
    lm.connect("traits", "emotional_stability", I_IRRITATION, -0.10, "weak", (-0.18, -0.04),
               "稳定→烦躁基线下降", "theory", "2026-06-21")

    lm.connect("traits", "attachment_anxiety", I_LONGING, 0.15, "weak", (0.05, 0.25),
               "依恋焦虑→思念基线上升", "theory", "2026-06-21")

    lm.connect("traits", "emotional_stability", I_SOCIAL_BATTERY, 0.05, "trace", (0.01, 0.10),
               "稳定→社交电量基线上升", "calibrated", "2026-06-21")

    lm.connect("traits", "emotional_stability", I_MENTAL_FATIGUE, -0.08, "trace", (-0.15, -0.02),
               "稳定→精神疲劳基线下降", "theory", "2026-06-21")
    lm.connect("traits", "anxiety_proneness", I_MENTAL_FATIGUE, -0.05, "trace", (-0.10, -0.01),
               "焦虑→精神疲劳基线也下降（保持警觉）", "calibrated", "2026-06-21")

    lm.build(skip_sparsity=True, skip_orthogonality=True)
    register_mapper(lm)
    return lm


# ═══════════════════════════════════════════════════════════════════
# Setpoint — 关系态人格基线
# ═══════════════════════════════════════════════════════════════════

def _build_rel_setpoint_mapper() -> LinearMapping:
    """关系态 setpoint: DEFAULT_RELATIONSHIP + traits 调制

    依恋回避→信任/亲密↓，依恋焦虑→亲密↑。
    """
    from state import T_LABELS, DEFAULT_RELATIONSHIP
    lm = LinearMapping("SETPOINT_RELATIONSHIP", R_LABELS,
                       "关系态人格基线（DEFAULT_RELATIONSHIP + traits 调制）")
    lm.add_source_group("traits", T_LABELS)

    lm.set_bias(R_AFFECTION, DEFAULT_RELATIONSHIP[R_AFFECTION], (-1.0, 1.0),
                "好感基线", "theory", "2026-06-21")
    lm.set_bias(R_TRUST_BOND, DEFAULT_RELATIONSHIP[R_TRUST_BOND], (-1.0, 1.0),
                "信任基线", "theory", "2026-06-21")
    lm.set_bias(R_INTIMACY, DEFAULT_RELATIONSHIP[R_INTIMACY], (-1.0, 1.0),
                "亲密基线", "theory", "2026-06-21")

    lm.connect("traits", "attachment_avoidance", R_TRUST_BOND, -0.27, "moderate", (-0.40, -0.15),
               "回避→信任基线下降（含 safety 合并项）", "theory", "2026-06-21")

    lm.connect("traits", "attachment_avoidance", R_INTIMACY, -0.12, "weak", (-0.20, -0.04),
               "回避→亲密基线下降", "theory", "2026-06-21")
    lm.connect("traits", "attachment_anxiety", R_INTIMACY, 0.10, "weak", (0.04, 0.18),
               "焦虑→亲密基线上升（焦虑型依赖）", "theory", "2026-06-21")

    lm.build(skip_sparsity=True, skip_orthogonality=True)
    register_mapper(lm)
    return lm


# ═══════════════════════════════════════════════════════════════════
# 衰减 — 基础衰减率 λ (每小时)
# ═══════════════════════════════════════════════════════════════════

def _build_decay_internal_lambda() -> WeightVector:
    """内部状态基础衰减率 λ_base (/小时)。"""
    vec = WeightVector("DECAY_INTERNAL_LAMBDA", I_LABELS,
                       "内部状态基础衰减率（/小时）")
    vec.connect("energy", I_ENERGY, 0.35, "moderate", (0.20, 0.50),
                "精力半衰期 ~2h", "calibrated", "2026-06-21")
    vec.connect("stress", I_STRESS, 0.23, "weak", (0.12, 0.35),
                "压力半衰期 ~3h", "calibrated", "2026-06-21")
    vec.connect("loneliness", I_LONELINESS, 0.17, "weak", (0.08, 0.28),
                "孤独半衰期 ~4h", "calibrated", "2026-06-21")
    vec.connect("insecurity", I_INSECURITY, 0.14, "weak", (0.06, 0.24),
                "不安半衰期 ~5h", "calibrated", "2026-06-21")
    vec.connect("irritation", I_IRRITATION, 0.69, "strong", (0.50, 0.90),
                "烦躁半衰期 ~1h（最快消退）", "calibrated", "2026-06-21")
    vec.connect("longing", I_LONGING, 0.12, "weak", (0.05, 0.22),
                "思念半衰期 ~6h（最慢）", "calibrated", "2026-06-21")
    vec.connect("social_battery", I_SOCIAL_BATTERY, 0.35, "moderate", (0.20, 0.50),
                "社交电量半衰期 ~2h", "calibrated", "2026-06-21")
    vec.connect("mental_fatigue", I_MENTAL_FATIGUE, 0.23, "weak", (0.12, 0.35),
                "精神疲劳半衰期 ~3h", "calibrated", "2026-06-21")
    vec.build()
    register_mapper(vec)
    return vec


def _build_decay_relationship_lambda() -> WeightVector:
    """关系状态基础衰减率 λ_base（/小时，天级）。"""
    vec = WeightVector("DECAY_RELATIONSHIP_LAMBDA", R_LABELS,
                       "关系状态基础衰减率（/小时，天级半衰期）")
    vec.connect("affection", R_AFFECTION, 0.0021, "trace", (0.001, 0.005),
                "好感 ~14d 半衰期", "calibrated", "2026-06-21")
    vec.connect("trust_bond", R_TRUST_BOND, 0.0021, "trace", (0.001, 0.005),
                "信任 ~14d 半衰期（trust+safety 原值平均）", "calibrated", "2026-06-21")
    vec.connect("intimacy", R_INTIMACY, 0.0041, "trace", (0.002, 0.008),
                "亲密 ~7d 半衰期（张力消退更快）", "calibrated", "2026-06-21")
    vec.build()
    register_mapper(vec)
    return vec


# ═══════════════════════════════════════════════════════════════════
# 衰减 — 时间曲线 + 非对称参数
# ═══════════════════════════════════════════════════════════════════

def _build_time_curve_k(label: str, value: float, desc: str) -> float:
    """时间曲线参数 k: λ_eff = λ_base / (1 + k × Δt)"""
    vec = WeightVector(f"TIME_CURVE_K_{label}", ["k"],
                       f"时间曲线参数 k——{desc}")
    mag = "trace" if abs(value) < 0.01 else "weak"
    low = max(0.0, value * 0.5)
    high = value * 2.0
    vec.connect("k", 0, value, mag, (low, high), desc, "calibrated", "2026-06-21")
    vec.build()
    register_mapper(vec)
    return vec.values[0]


def _build_negative_decay_boost() -> float:
    """非对称衰减倍率: 负向偏离（current < setpoint）的 λ 倍率。"""
    vec = WeightVector("NEGATIVE_DECAY_BOOST", ["boost"],
                       "非对称衰减倍率——负面印象消退快于正面（FAB）")
    vec.connect("boost", 0, 1.8, "moderate", (1.2, 2.5),
                "Fading Affect Bias: 负面情绪比正面消退快 1.8×", "theory", "2026-06-21")
    vec.build()
    register_mapper(vec)
    return vec.values[0]


# ═══════════════════════════════════════════════════════════════════
# 衰减 — 人格调制因子
# ═══════════════════════════════════════════════════════════════════

def _build_internal_personality_mod() -> LinearMapping:
    """内部态人格调制: mod = 1.0 + stability*0.15 + optimism*0.075
                        - anxiety*0.125 - anger*0.05 + openness*0.05"""
    from state import T_LABELS
    lm = LinearMapping("INT_PERSONALITY_MOD", ["mod"],
                       "内部态衰减人格调制因子（缩放 λ_base）")
    lm.add_source_group("traits", T_LABELS)

    lm.set_bias(0, 1.0, (0.5, 1.5), "基础调制=1.0（不缩放）", "theory", "2026-06-21")
    lm.connect("traits", "emotional_stability", 0, 0.15, "weak", (0.05, 0.25),
               "稳定→恢复快", "theory", "2026-06-21")
    lm.connect("traits", "optimism", 0, 0.075, "trace", (0.02, 0.15),
               "乐观→恢复快", "theory", "2026-06-21")
    lm.connect("traits", "anxiety_proneness", 0, -0.125, "weak", (-0.22, -0.04),
               "焦虑→放不下", "theory", "2026-06-21")
    lm.connect("traits", "anger_reactivity", 0, -0.05, "trace", (-0.12, -0.01),
               "易怒→恢复慢", "theory", "2026-06-21")
    lm.connect("traits", "emotional_openness", 0, 0.05, "trace", (0.01, 0.12),
               "开放→恢复快", "theory", "2026-06-21")

    lm.build(skip_sparsity=True, skip_orthogonality=True)
    register_mapper(lm)
    return lm


def _build_relationship_personality_mod() -> LinearMapping:
    """关系态人格调制: mod = 1.0 + avoidance*0.175 - anxiety*0.10 - stability*0.05"""
    from state import T_LABELS
    lm = LinearMapping("REL_PERSONALITY_MOD", ["mod"],
                       "关系态衰减人格调制因子（缩放 λ_base）")
    lm.add_source_group("traits", T_LABELS)

    lm.set_bias(0, 1.0, (0.5, 1.5), "基础调制=1.0", "theory", "2026-06-21")
    lm.connect("traits", "attachment_avoidance", 0, 0.175, "weak", (0.08, 0.28),
               "回避→疏远快", "theory", "2026-06-21")
    lm.connect("traits", "attachment_anxiety", 0, -0.10, "weak", (-0.20, -0.03),
               "焦虑→放不下", "theory", "2026-06-21")
    lm.connect("traits", "emotional_stability", 0, -0.05, "trace", (-0.12, -0.01),
               "稳定→关系稳定（衰减慢）", "theory", "2026-06-21")

    lm.build(skip_sparsity=True, skip_orthogonality=True)
    register_mapper(lm)
    return lm


# ── 模块加载时构建 ──
SELF_DECAY = _build_self_decay().values  # np.ndarray(8,)
DECAY_TARGETS = _build_decay_targets().values  # np.ndarray(8,)
REL_SELF_DECAY = _build_rel_self_decay().values  # np.ndarray(3,)
INTERNAL_COUPLING = _build_internal_coupling()  # (8, 8) sources@M → coupling
RELATIONSHIP_COUPLING = _build_relationship_coupling()  # (3, 3)
CROSS_SCALE_COUPLING = _build_cross_scale_coupling()  # (8, 3)

# α/β 速率参数
ALPHA_MAPPER = _build_alpha_internal()
ALPHA_REL_MAPPER = _build_alpha_relationship()
BETA_REL_MAPPER = _build_beta_relationship()
BETA_BASE = _build_beta_base().values  # (7,)
HYPER_BETA_GAIN = _build_hyper_beta_gain().values[0]  # scalar
DEACT_BETA_GAIN = _build_deact_beta_gain().values[0]  # scalar

# Setpoint 映射器
SETPOINT_MAPPER = _build_setpoint_mapper()
REL_SETPOINT_MAPPER = _build_rel_setpoint_mapper()

# 衰减参数
DECAY_INTERNAL_LAMBDA = _build_decay_internal_lambda().values  # (8,)
DECAY_RELATIONSHIP_LAMBDA = _build_decay_relationship_lambda().values  # (3,)
DECAY_INTERNAL_TIME_CURVE_K = _build_time_curve_k("INTERNAL", 0.05, "内部时间曲线放缓系数")
DECAY_REL_TIME_CURVE_K = _build_time_curve_k("RELATIONSHIP", 0.001, "关系时间曲线放缓系数（几乎不放缓）")
DECAY_NEGATIVE_BOOST = _build_negative_decay_boost()
INT_PERSONALITY_MOD = _build_internal_personality_mod()
REL_PERSONALITY_MOD = _build_relationship_personality_mod()
