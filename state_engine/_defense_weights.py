"""防御剖面权重 —— 秩-1 分解：基线 + 强度 × PC1 方向。

约束⑤(语义映射层) + 约束⑧(参数审计) 合规。

基于 100,000 随机样本 PCA 审计（2026-06-22）:
  去激活:   PC1 解释方差 83%,  秩-1 R²=84.3%
  过度激活: PC1 解释方差 92%,  秩-1 R²=94.7%

关键参数:
  profile_raw = BASELINE + INTENSITY × PC1_DIR    # pre-sigmoid
  profile     = sigmoid((profile_raw - SHIFT) × SCALE)
"""

import numpy as np
from state import (
    ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT,
    ST_DEPENDENCY, ST_TEASING, ST_EMOTIONAL_WEIGHT, ST_SIZE, ST_LABELS,
)
from ._validator import LinearMapping, WeightVector, register_mapper


# ═══════════════════════════════════════════════════════════════════
# 基线（pre-sigmoid 均值）
# ═══════════════════════════════════════════════════════════════════

DEACT_BASELINE_RAW: np.ndarray = np.array([
    -0.2498,   # ST_ABANDONMENT     — 最常隐藏的
    -0.5001,   # ST_VALIDATION      — 藏起"在意认可"
    -1.0028,   # ST_CLOSENESS       — 伪装最少（渴望的露馅）
    -0.7521,   # ST_CONFLICT        — 中等防御
    -0.3526,   # ST_DEPENDENCY      — 口是心非
    -0.7484,   # ST_TEASING         — 被调侃也绷着
    -0.5013,   # ST_EMOTIONAL_WEIGHT — 不愿深谈
], dtype=np.float64)

HYPER_BASELINE_RAW: np.ndarray = np.array([
     0.3523,   # ST_ABANDONMENT     — 抛弃天然最强烈
    -1.1493,   # ST_VALIDATION      — 认可不愿承认
    -0.3972,   # ST_CLOSENESS       — 渴望亲近
    -1.1499,   # ST_CONFLICT        — 冲突不多想
    -0.1477,   # ST_DEPENDENCY      — 被需要→中高唤起
    -1.3979,   # ST_TEASING         — 调侃不放心上
    -0.8985,   # ST_EMOTIONAL_WEIGHT — 沉重话题走心
], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════
# PC1 方向（单位向量）
# ═══════════════════════════════════════════════════════════════════

DEACT_PC1_DIR: np.ndarray = np.array([
    0.4785,    # ST_ABANDONMENT     — 抛弃恐惧变化最大
    0.3960,    # ST_VALIDATION      — 认可随整体增减
    0.3147,    # ST_CLOSENESS       — 亲近同步变化
    0.3552,    # ST_CONFLICT        — 冲突同步
    0.4595,    # ST_DEPENDENCY      — 依赖同步
    0.2173,    # ST_TEASING         — 调侃变化最小（社交面具）
    0.3616,    # ST_EMOTIONAL_WEIGHT — 沉重同步
], dtype=np.float64)
# 单位化: norm = 1.0000

HYPER_PC1_DIR: np.ndarray = np.array([
    0.5527,    # ST_ABANDONMENT     — 抛弃放大最剧烈
    0.2194,    # ST_VALIDATION      — 认可放大有限
    0.5369,    # ST_CLOSENESS       — 亲近放大剧烈
    0.2355,    # ST_CONFLICT        — 冲突放大中等
    0.4443,    # ST_DEPENDENCY      — 依赖放大强
    0.0673,    # ST_TEASING         — 调侃几乎无放大
    0.3173,    # ST_EMOTIONAL_WEIGHT — 沉重放大中等
], dtype=np.float64)
# 单位化: norm = 1.0000


# ═══════════════════════════════════════════════════════════════════
# 强度 Mapper: inputs(21) → intensity(scalar)
# ═══════════════════════════════════════════════════════════════════

def _build_deact_intensity() -> LinearMapping:
    """去激活强度: scalar intensity = w·[traits, relationship, internal]

    正值增强防御，负值削弱防御。该标量 × PC1_DIR 分布到 7 维。
    回归系数来自 100,000 随机样本 OLS 拟合，R²=99.54%。
    """
    from state import T_LABELS, I_LABELS, R_LABELS
    lm = LinearMapping("DEACT_INTENSITY", ["intensity"],
                       "去激活强度（21→1，正值=更多防御）")
    lm.add_source_group("traits", T_LABELS)
    lm.add_source_group("relationship", R_LABELS)
    lm.add_source_group("internal", I_LABELS)

    # ── Traits: 人格塑形基调制强度 ──
    lm.connect("traits", "pride", 0, 2.00, "strong", (1.50, 2.50),
               "高自尊→更多防御（Bowlby 自尊保护的防御动机）",
               "calibrated", "2026-06-22")
    lm.connect("traits", "emotional_openness", 0, -0.80, "moderate", (-1.20, -0.40),
               "情绪开放→减少防御（开放性的坦诚倾向）",
               "calibrated", "2026-06-22")
    lm.connect("traits", "emotional_stability", 0, -1.20, "strong", (-1.60, -0.80),
               "情绪稳定→大幅减少防御（安全型依恋特质）",
               "calibrated", "2026-06-22")
    lm.connect("traits", "anger_reactivity", 0, 0.20, "weak", (0.10, 0.35),
               "易怒→略微增加防御", "calibrated", "2026-06-22")
    lm.connect("traits", "jealousy_sensitivity", 0, 0.20, "weak", (0.10, 0.35),
               "嫉妒→略微增加防御", "calibrated", "2026-06-22")
    lm.connect("traits", "attachment_avoidance", 0, 1.30, "strong", (1.00, 1.60),
               "依恋回避→大幅增加防御（疏离策略的核心驱动）",
               "calibrated", "2026-06-22")

    # ── Relationship: 信任降低防御 ──
    lm.connect("relationship", "trust_bond", 0, -0.40, "moderate", (-0.60, -0.20),
               "信任纽带→降低防御（安全基地效应）",
               "calibrated", "2026-06-22")

    # ── Internal: 急性状态驱动 ──
    lm.connect("internal", "stress", 0, 0.70, "moderate", (0.50, 0.90),
               "压力→增加防御（应激防御增强）",
               "calibrated", "2026-06-22")
    lm.connect("internal", "insecurity", 0, 0.65, "moderate", (0.45, 0.85),
               "不安全感→增加防御（脆弱性补偿）",
               "calibrated", "2026-06-22")

    lm.build(skip_sparsity=True, skip_orthogonality=True)
    register_mapper(lm)
    return lm


def _build_hyper_state_modulation() -> LinearMapping:
    """过度激活状态调制: internal[8] → hyper_delta[7] (维度特异性)

    在人格基线（秩-1 标量强度）之上，添加情绪状态驱动的维度特异性调制。
    使 hyper 剖面可以产生"某些维度↑、另一些维度↓"的交叉模式。

    设计原则:
      - 稀疏连接: 每维 internal 只影响 1-2 个 hyper 维度
      - 心理学解释: 每条连接关联到具体理论
      - 状态与人格解耦: 人格差异仍由 HYPER_INTENSITY(21→1)×PC1_DIR 处理
    """
    from state import I_LABELS, ST_LABELS
    from state import (
        ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT,
        ST_DEPENDENCY, ST_TEASING, ST_EMOTIONAL_WEIGHT,
    )
    lm = LinearMapping("HYPER_STATE_MODULATION", ST_LABELS,
                       "内部状态→过度激活维度特异性调制 (8→7)")
    lm.add_source_group("internal", I_LABELS)

    # ── Stress（压力）──
    lm.connect("internal", "stress", ST_CONFLICT, 0.20, "moderate", (0.10, 0.30),
               "压力→交感激活增强威胁知觉，放大冲突感受 (Lazarus & Folkman, 1984)",
               "theory", "2026-06-22")
    lm.connect("internal", "stress", ST_CLOSENESS, -0.12, "weak", (-0.20, -0.05),
               "压力→注意力转向威胁监测，减少对亲近信号的接收 (Ford et al., 2010)",
               "theory", "2026-06-22")
    lm.connect("internal", "stress", ST_VALIDATION, -0.08, "trace", (-0.15, -0.03),
               "压力→认知资源消耗，降低对认可的敏感性",
               "theory", "2026-06-22")

    # ── Irritation（烦躁）──
    lm.connect("internal", "irritation", ST_CONFLICT, 0.30, "strong", (0.20, 0.42),
               "烦躁→冲突触发阈值显著下降，一触即炸 (Berkowitz, 1989 挫折-攻击理论)",
               "theory", "2026-06-22")
    lm.connect("internal", "irritation", ST_TEASING, 0.20, "moderate", (0.10, 0.30),
               "烦躁→调侃被重新解释为攻击/挑衅 (Keltner et al., 2001)",
               "theory", "2026-06-22")

    # ── Insecurity（不安全感）──
    lm.connect("internal", "insecurity", ST_ABANDONMENT, 0.50, "strong", (0.35, 0.65),
               "不安全感→直接放大被抛弃恐惧 (Bowlby, 1988; Mikulincer & Shaver, 2003)",
               "theory", "2026-06-22")

    # ── Loneliness（孤独）──
    lm.connect("internal", "loneliness", ST_CLOSENESS, 0.20, "moderate", (0.10, 0.30),
               "孤独→对亲密的渴望增强，社会重连驱力 (Cacioppo & Patrick, 2008)",
               "theory", "2026-06-22")
    lm.connect("internal", "loneliness", ST_DEPENDENCY, 0.10, "weak", (0.04, 0.18),
               "孤独→被需要感增强，关系锚定需求 (Baumeister & Leary, 1995)",
               "theory", "2026-06-22")

    # ── Longing（思念）──
    lm.connect("internal", "longing", ST_CLOSENESS, 0.25, "strong", (0.15, 0.35),
               "思念→放大亲近驱力，预期性依恋系统激活 (Mikulincer & Shaver, 2003)",
               "theory", "2026-06-22")
    lm.connect("internal", "longing", ST_DEPENDENCY, 0.12, "weak", (0.05, 0.20),
               "思念→被需要感也是连接的一种形式",
               "theory", "2026-06-22")

    # ── Social Battery（社交电量）──
    lm.connect("internal", "social_battery", ST_CLOSENESS, 0.15, "moderate", (0.08, 0.25),
               "社交电充足→对亲近开放；电量低→回避亲密接触 (Eysenck, 1967)",
               "theory", "2026-06-22")
    lm.connect("internal", "social_battery", ST_DEPENDENCY, 0.10, "weak", (0.04, 0.18),
               "社交电充足→对被需要更积极回应",
               "theory", "2026-06-22")

    # ── Mental Fatigue（精神疲劳）──
    lm.connect("internal", "mental_fatigue", ST_ABANDONMENT, -0.08, "trace", (-0.15, -0.03),
               "精神疲劳→钝化威胁反应，情绪耗竭 (Baumeister, 1998)",
               "theory", "2026-06-22")
    lm.connect("internal", "mental_fatigue", ST_CONFLICT, -0.10, "weak", (-0.18, -0.04),
               "精神疲劳→无力应对冲突，反应钝化",
               "theory", "2026-06-22")
    lm.connect("internal", "mental_fatigue", ST_CLOSENESS, -0.10, "weak", (-0.18, -0.04),
               "精神疲劳→社交退缩，对亲近的响应下降",
               "theory", "2026-06-22")

    lm.build(skip_sparsity=True, skip_orthogonality=True)
    register_mapper(lm)
    return lm


def _build_hyper_intensity() -> LinearMapping:
    """过度激活强度: scalar intensity = w·[traits, relationship, internal]

    正值放大内心感受，负值抑制放大。
    OLS 拟合 R²=99.36%。
    """
    from state import T_LABELS, I_LABELS, R_LABELS
    lm = LinearMapping("HYPER_INTENSITY", ["intensity"],
                       "过度激活强度（14→1，仅人格基线；状态调制由 HYPER_STATE_MODULATION 处理）")
    lm.add_source_group("traits", T_LABELS)
    lm.add_source_group("relationship", R_LABELS)

    # ── Traits ──
    lm.connect("traits", "sensitivity", 0, 0.55, "moderate", (0.40, 0.70),
               "敏感→放大感受（感知敏锐→情绪共鸣更深）",
               "calibrated", "2026-06-22")
    lm.connect("traits", "jealousy_sensitivity", 0, 0.45, "moderate", (0.30, 0.60),
               "嫉妒敏感→放大关系威胁感受",
               "calibrated", "2026-06-22")
    lm.connect("traits", "attachment_anxiety", 0, 2.40, "strong", (1.80, 3.00),
               "依恋焦虑→核心驱动：极大放大所有关系相关感受",
               "calibrated", "2026-06-22")
    lm.connect("traits", "attachment_avoidance", 0, -2.10, "strong", (-2.80, -1.60),
               "依恋回避→核心抑制：否认/疏离所有情感信号",
               "calibrated", "2026-06-22")

    # ── Relationship ──
    lm.connect("relationship", "affection", 0, 0.30, "moderate", (0.20, 0.45),
               "好感→放大情感共鸣", "calibrated", "2026-06-22")
    lm.connect("relationship", "intimacy", 0, 0.40, "moderate", (0.25, 0.55),
               "亲密张力→放大关系投入度", "calibrated", "2026-06-22")

    lm.build(skip_sparsity=True, skip_orthogonality=True)
    register_mapper(lm)
    return lm


# ═══════════════════════════════════════════════════════════════════
# Sigmoid 缩放 & 偏移（保留原语义）
# ═══════════════════════════════════════════════════════════════════

def _build_deact_sigmoid_shift() -> float:
    vec = WeightVector("DEACT_SIGMOID_SHIFT", ["shift"],
                       "去激活 sigmoid 偏移量")
    vec.connect("shift", 0, 0.35, "moderate", (0.20, 0.50),
                "presigmoid 围绕 0 ≈ 中等防御", "calibrated", "2026-06-21")
    vec.build(); register_mapper(vec)
    return vec.values[0]

def _build_deact_sigmoid_scale() -> float:
    vec = WeightVector("DEACT_SIGMOID_SCALE", ["scale"],
                       "去激活 sigmoid 缩放倍数")
    vec.connect("scale", 0, 5.0, "strong", (3.0, 8.0),
                "5× 放大使差异在 [0,1] 间展开", "calibrated", "2026-06-21")
    vec.build(); register_mapper(vec)
    return vec.values[0]

def _build_hyper_sigmoid_shift() -> float:
    vec = WeightVector("HYPER_SIGMOID_SHIFT", ["shift"],
                       "过度激活 sigmoid 偏移量")
    vec.connect("shift", 0, 0.38, "moderate", (0.20, 0.55),
                "过度激活基线偏移", "calibrated", "2026-06-21")
    vec.build(); register_mapper(vec)
    return vec.values[0]

def _build_hyper_sigmoid_scale() -> float:
    vec = WeightVector("HYPER_SIGMOID_SCALE", ["scale"],
                       "过度激活 sigmoid 缩放倍数")
    vec.connect("scale", 0, 5.0, "strong", (3.0, 8.0),
                "5× 放大", "calibrated", "2026-06-21")
    vec.build(); register_mapper(vec)
    return vec.values[0]

def _build_hyper_apply_gain() -> float:
    vec = WeightVector("HYPER_APPLY_GAIN", ["gain"],
                       "过度激活→inner 放大增益")
    vec.connect("gain", 0, 0.50, "moderate", (0.30, 0.70),
                "hyper 每 +1，inner 放大 50%", "theory", "2026-06-21")
    vec.build(); register_mapper(vec)
    return vec.values[0]

def _build_deact_apply_gain() -> float:
    vec = WeightVector("DEACT_APPLY_GAIN", ["gain"],
                       "去激活→outer 压制增益")
    vec.connect("gain", 0, 0.70, "moderate", (0.50, 0.90),
                "deact 每 +1，outer 压至原值 30%", "theory", "2026-06-21")
    vec.build(); register_mapper(vec)
    return vec.values[0]


# ── 模块加载时构建 ──
DEACT_INTENSITY = _build_deact_intensity()
HYPER_INTENSITY = _build_hyper_intensity()
HYPER_STATE_MODULATION = _build_hyper_state_modulation()
DEACT_SIGMOID_SHIFT = _build_deact_sigmoid_shift()
DEACT_SIGMOID_SCALE = _build_deact_sigmoid_scale()
HYPER_SIGMOID_SHIFT = _build_hyper_sigmoid_shift()
HYPER_SIGMOID_SCALE = _build_hyper_sigmoid_scale()
HYPER_APPLY_GAIN = _build_hyper_apply_gain()
DEACT_APPLY_GAIN = _build_deact_apply_gain()
