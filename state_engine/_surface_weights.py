"""Surface 投影权重 —— 所有线性系数的 provenance 集中管理。

约束⑤(语义映射层) + 约束⑧(参数审计) 合规：
  - 每个系数通过 LinearMapping.connect() 注册，带心理学依据和起源
  - 每个偏置通过 LinearMapping.set_bias() 注册，带心理学依据和起源
  - build() 时自动执行约束检查并注册到 ConstraintRegistry

注意：sigmoid 门控特质项已于 2026-06-22 移除
保留在 _surface.py 中作为显式非线性代码（不属于线性映射层）。

用法:
    from ._surface_weights import SURFACE_MAPPER
    sources = np.concatenate([internal, relationship, outer_stimuli])
    surface_linear = SURFACE_MAPPER.compute(sources)
"""

import numpy as np
from state import (
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_MENTAL_FATIGUE,
    R_AFFECTION, R_TRUST_BOND, R_INTIMACY,
    S_EXPRESSIVENESS, S_WARMTH, S_SHARPNESS, S_SOFTNESS,
    S_ENTHUSIASM, S_RESTRAINT, S_VULNERABILITY,
)
from ._validator import LinearMapping, WeightMapper, register_mapper


def _build_surface_mapper() -> LinearMapping:
    """构建 surface 投影的线性映射 (28→7)，带完整 provenance。

    源向量拼接顺序: [internal(8), relationship(3), outer_stimuli(7), traits(10)]
    """
    from state import I_LABELS, R_LABELS, ST_LABELS

    lm = LinearMapping(
        "SURFACE_PROJECTION",
        target_labels=[
            "expressiveness", "warmth", "sharpness", "softness",
            "enthusiasm", "restraint", "vulnerability",
        ],
        description="内部态+关系态+外刺激+特质 → 7维表面表达",
    )

    # ── 注册 3 个源组 ──
    lm.add_source_group("internal", I_LABELS)             # offset 0,  8 dims
    lm.add_source_group("relationship", R_LABELS)          # offset 8,  3 dims
    lm.add_source_group("outer_stimuli", ST_LABELS)        # offset 11, 7 dims

    # 共 18 个源 → 7 个 target（移除 traits——traits 通过 defense/dynamics 间接实现）

    # ════════════════════════════════════════════════════════════════
    # 偏置项（每个 surface 维度的基线偏移）
    # ════════════════════════════════════════════════════════════════
    lm.set_bias(S_EXPRESSIVENESS, -0.3, (-0.5, 0.0),
                "默认表达偏内敛", "calibrated", "2026-06-21")
    lm.set_bias(S_WARMTH, -0.2, (-0.4, 0.0),
                "默认语气偏凉", "calibrated", "2026-06-21")
    lm.set_bias(S_SHARPNESS, -0.1, (-0.3, 0.1),
                "默认锋芒中性略低", "calibrated", "2026-06-21")
    lm.set_bias(S_SOFTNESS, -0.1, (-0.3, 0.1),
                "默认柔软度中等", "calibrated", "2026-06-21")
    lm.set_bias(S_ENTHUSIASM, -0.2, (-0.4, 0.0),
                "默认活力偏低调", "calibrated", "2026-06-21")
    lm.set_bias(S_RESTRAINT, -0.1, (-0.3, 0.1),
                "默认克制度中等", "calibrated", "2026-06-21")
    lm.set_bias(S_VULNERABILITY, -0.5, (-0.7, -0.2),
                "默认脆弱感低（自尊保护）", "theory", "2026-06-21")

    # ════════════════════════════════════════════════════════════════
    # 内部态 → 表面（核心：真实感受的外泄程度）
    # ════════════════════════════════════════════════════════════════

    # 精力充沛→表达外露↑ 活力↑
    lm.connect("internal", "energy", S_EXPRESSIVENESS, 0.4,
               "moderate", (0.25, 0.50), "精力→表达欲上升", "theory", "2026-06-21")
    lm.connect("internal", "energy", S_ENTHUSIASM, 0.5,
               "strong", (0.30, 0.60), "精力→活力四射", "theory", "2026-06-21")

    # 精神疲劳→表达收窄↓ 活力↓
    lm.connect("internal", "mental_fatigue", S_EXPRESSIVENESS, -0.15,
               "weak", (-0.25, -0.05), "疲惫→表达减少", "calibrated", "2026-06-21")
    lm.connect("internal", "mental_fatigue", S_ENTHUSIASM, -0.15,
               "weak", (-0.25, -0.05), "疲惫→活力下降", "calibrated", "2026-06-21")

    # 烦躁→尖锐↑
    lm.connect("internal", "irritation", S_SHARPNESS, 0.5,
               "strong", (0.30, 0.65), "烦躁→话语带刺", "theory", "2026-06-21")

    # 压力→温暖↓ 尖锐↑ 克制↑
    lm.connect("internal", "stress", S_WARMTH, -0.15,
               "weak", (-0.25, -0.05), "压力→温度下降", "calibrated", "2026-06-21")
    lm.connect("internal", "stress", S_SHARPNESS, 0.15,
               "weak", (0.05, 0.25), "压力→锋芒显现", "calibrated", "2026-06-21")
    lm.connect("internal", "stress", S_RESTRAINT, 0.20,
               "moderate", (0.10, 0.30), "压力→克制上升（伪装）", "theory", "2026-06-21")

    # 不安全感→克制↑
    lm.connect("internal", "insecurity", S_RESTRAINT, 0.3,
               "moderate", (0.15, 0.40), "不安→字斟句酌", "theory", "2026-06-21")

    # 孤独→脆弱↑
    lm.connect("internal", "loneliness", S_VULNERABILITY, 0.3,
               "moderate", (0.15, 0.40), "孤独→防御下降", "theory", "2026-06-21")

    # 思念→脆弱↑
    lm.connect("internal", "longing", S_VULNERABILITY, 0.2,
               "weak", (0.10, 0.30), "思念→流露脆弱", "theory", "2026-06-21")

    # ════════════════════════════════════════════════════════════════
    # 关系态 → 表面（对用户的感受投射）
    # ════════════════════════════════════════════════════════════════

    # 好感→温暖↑
    lm.connect("relationship", "affection", S_WARMTH, 0.4,
               "moderate", (0.25, 0.55), "好感→语气温暖", "theory", "2026-06-21")

    # 信任→柔软↑
    lm.connect("relationship", "trust_bond", S_SOFTNESS, 0.2,
               "weak", (0.10, 0.30), "信任→态度柔软", "theory", "2026-06-21")

    # ════════════════════════════════════════════════════════════════
    # 外刺激 → 表面（被压抑后的残余表达）
    # ════════════════════════════════════════════════════════════════

    # 被认可→温暖↑ 活力↑
    lm.connect("outer_stimuli", "validation_stimulus", S_WARMTH, 0.30,
               "moderate", (0.15, 0.40), "被认可→温暖回应", "theory", "2026-06-21")
    lm.connect("outer_stimuli", "validation_stimulus", S_ENTHUSIASM, 0.15,
               "weak", (0.05, 0.25), "被认可→活力上升", "calibrated", "2026-06-21")

    # 冲突→尖锐↑
    lm.connect("outer_stimuli", "conflict_stimulus", S_SHARPNESS, 0.25,
               "moderate", (0.10, 0.40), "冲突→话语带刺", "theory", "2026-06-21")

    # 靠近→柔软↑
    lm.connect("outer_stimuli", "closeness_stimulus", S_SOFTNESS, 0.20,
               "weak", (0.10, 0.30), "亲近→态度柔软", "calibrated", "2026-06-21")

    # 被抛弃→脆弱↑
    lm.connect("outer_stimuli", "abandonment_stimulus", S_VULNERABILITY, 0.15,
               "weak", (0.05, 0.25), "被抛弃恐惧→脆弱外露", "theory", "2026-06-21")

    # 情感重量→克制↑
    lm.connect("outer_stimuli", "emotional_weight_stimulus", S_RESTRAINT, 0.20,
               "weak", (0.10, 0.30), "沉重话题→字斟句酌", "theory", "2026-06-21")

    # 调侃→尖锐↑
    lm.connect("outer_stimuli", "teasing_stimulus", S_SHARPNESS, 0.10,
               "trace", (0.03, 0.18), "调侃→微带锋芒", "calibrated", "2026-06-21")

    # 被依赖→温暖↑
    lm.connect("outer_stimuli", "dependency_stimulus", S_WARMTH, 0.10,
               "trace", (0.03, 0.18), "被需要→温度上升", "calibrated", "2026-06-21")

    # ════════════════════════════════════════════════════════════════
    # 特质直连（约束①违规——2026-06-22 移除）
    # traits 已通过 defense profiles + dynamics 间接影响 surface，
    # 不再需要直连。参见 md/SURFACE_PROJECTION_RESEARCH.md §8.4
    # ════════════════════════════════════════════════════════════════

    # ── 构建 ──
    lm.build(skip_sparsity=True)  # surface 映射非纯矩阵，跳过密度检查
    register_mapper(lm)

    return lm


# ═══════════════════════════════════════════════════════════════════
# Sigmoid 门控系数（2026-06-22 移除——traits 已间接实现）
# 这些系数曾是 pride/openness/optimism 的门控幅度和缩放参数，
# 移除后相关功能通过 defense profiles + dynamics 间接实现。
# 参见 md/SURFACE_PROJECTION_RESEARCH.md §8.4
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# 表面→内部反馈矩阵
# ═══════════════════════════════════════════════════════════════════

def _build_surface_feedback_matrix() -> np.ndarray:
    """表面→内部反馈映射 (7×8)：情绪失调成本 + 面部/躯体反馈 + 表达消耗。

    与 CROSS_SCALE_COUPLING (internal→relationship) 模式一致。
    所有系数为 trace 量级 (0.03–0.06)，仅提供微小逐轮调制。
    """
    from state import I_SIZE, S_SIZE, S_LABELS, I_LABELS

    mapper = WeightMapper(
        "SURFACE_FEEDBACK",
        source_labels=S_LABELS,
        target_labels=I_LABELS,
        description="表面→内部反馈（失调成本/面部反馈/表达消耗）",
    )

    # ── ① 情绪失调成本：表面与内部的差异产生内部压力 ──
    mapper.connect(S_RESTRAINT, I_STRESS, 0.06, "trace", (0.02, 0.10),
                   "克制真实感受→压力上升", "calibrated", "2026-06-22")
    mapper.connect(S_WARMTH, I_STRESS, 0.04, "trace", (0.01, 0.08),
                   "强颜欢笑→压力微升", "calibrated", "2026-06-22")

    # ── ② 面部/躯体反馈：表达改变内在感受 ──
    mapper.connect(S_SHARPNESS, I_IRRITATION, 0.05, "trace", (0.02, 0.09),
                   "面露锋芒→烦躁微增", "calibrated", "2026-06-22")
    mapper.connect(S_WARMTH, I_LONELINESS, -0.04, "trace", (-0.08, -0.01),
                   "温暖表达→孤独感降", "calibrated", "2026-06-22")
    mapper.connect(S_ENTHUSIASM, I_ENERGY, 0.05, "trace", (0.02, 0.09),
                   "活力外显→精力互促", "calibrated", "2026-06-22")
    mapper.connect(S_VULNERABILITY, I_LONGING, 0.04, "trace", (0.01, 0.08),
                   "流露脆弱→渴望微增", "calibrated", "2026-06-22")

    # ── ③ 表达消耗成本：活跃表达消耗精力 ──
    mapper.connect(S_EXPRESSIVENESS, I_ENERGY, -0.06, "trace", (-0.10, -0.02),
                   "表达外露→精力消耗", "calibrated", "2026-06-22")
    mapper.connect(S_EXPRESSIVENESS, I_MENTAL_FATIGUE, 0.05, "trace", (0.02, 0.09),
                   "表达外露→疲劳微增", "calibrated", "2026-06-22")
    mapper.connect(S_RESTRAINT, I_MENTAL_FATIGUE, 0.04, "trace", (0.01, 0.08),
                   "克制压抑→精神疲劳", "calibrated", "2026-06-22")

    M = mapper.build_matrix(
        (S_SIZE, I_SIZE),   # (7, 8)
        skip_sparsity=True,
        skip_orthogonality=True,
        skip_rank=True,
    )
    register_mapper(mapper)
    return M


# ── 模块加载时构建 ──
SURFACE_MAPPER = _build_surface_mapper()
SURFACE_FEEDBACK_MATRIX = _build_surface_feedback_matrix()
