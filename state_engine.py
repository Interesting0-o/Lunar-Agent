"""
state_engine —— 连续人格动力系统

角色"潜意识"的核心，将外部社交刺激转化为连续心理变化。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
架构分层（数据流从上到下）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Perception (user_signals + interaction_impact)
       ↓
  ① Stimulus Construction        社交信号 → 心理意义空间
       ↓
  ② Trait Modulation             人格特质放大/衰减心理刺激
       ↓
  ③ Relationship Modulation       关系改变刺激的"含义"
       ↓
  ④ Gate Control (核心)           压抑/脆弱/依恋/泄漏 门控
       ↓
  ⑤ Internal Dynamics            h_t = A·h_{t-1} + B·e_t + c
       ↓
  ⑥ Decay                        各维度不同衰减速率
       ↓
  ⑦ Hidden Accumulation          被压抑的情绪进入隐藏层
       ↓
  ⑧ Event Trigger                阈值检查 → 离散人格事件
       ↓
  ⑨ Surface Projection           内部状态 → 表面表达 (动态投影，不存储)
       ↓
  LLM Generation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
设计原则：
  - 纯函数：给定相同输入，始终返回相同输出
  - 可组合：每层职责单一，可独立测试
  - 向量化：尽可能用矩阵运算代替逐元素操作
  - 压抑机制：h_t = f(x_t, p, r) 算出"真实感受"，
            然后 Gate 决定多少进入表达、多少压入隐藏层
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
from state import (
    # ── 内部状态 ──
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE, I_SIZE,
    # ── 关系状态 ──
    R_AFFECTION, R_TRUST, R_FAMILIARITY, R_DEPENDENCY,
    R_EMOTIONAL_SAFETY, R_ROMANTIC_TENSION, R_SIZE,
    # ── 隐藏状态 ──
    H_SUPPRESSED_SADNESS, H_SUPPRESSED_ANGER, H_HIDDEN_AFFECTION, H_SIZE,
    # ── 表面状态 ──
    S_EXPRESSIVENESS, S_WARMTH, S_SHARPNESS, S_SOFTNESS,
    S_ENTHUSIASM, S_RESTRAINT, S_VULNERABILITY, S_SIZE,
    # ── 特质 ──
    T_SENSITIVITY, T_PRIDE, T_EMOTIONAL_OPENNESS, T_EMOTIONAL_STABILITY,
    T_OPTIMISM, T_ANXIETY_PRONENESS, T_ANGER_REACTIVITY, T_JEALOUSY_SENSITIVITY,
    T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE, T_SIZE,
    # ── 社交信号 ──
    SS_AFFECTION, SS_ATTENTION, SS_INTIMACY, SS_APPROVAL,
    SS_REJECTION, SS_ABANDONMENT, SS_DEPENDENCY, SS_TEASING, SS_CONFLICT, SS_SIZE,
    # ── 互动影响 ──
    II_EMOTIONAL_WEIGHT, II_TRUST_IMPACT, II_CLOSENESS_IMPACT,
    # ── 心理意义空间 ──
    ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT,
    ST_DEPENDENCY, ST_TEASING, ST_EMOTIONAL_WEIGHT, ST_SIZE,
    # ── 门控 ──
    G_SUPPRESSION, G_VULNERABILITY, G_ATTACHMENT, G_LEAKAGE, G_SIZE,
)
from typing import Optional

# ============================================================
# 0. 默认初始值
# ============================================================

DEFAULT_TRAITS: np.ndarray = np.array([
    0.7,    # T_SENSITIVITY
    0.65,   # T_PRIDE
    0.6,    # T_EMOTIONAL_OPENNESS
    0.5,    # T_EMOTIONAL_STABILITY
    0.55,   # T_OPTIMISM
    0.6,    # T_ANXIETY_PRONENESS
    0.5,    # T_ANGER_REACTIVITY
    0.7,    # T_JEALOUSY_SENSITIVITY
    0.55,   # T_ATTACHMENT_ANXIETY
    0.2,    # T_ATTACHMENT_AVOIDANCE
], dtype=np.float64)

DEFAULT_INTERNAL: np.ndarray = np.array([
    0.7,    # I_ENERGY
    0.2,    # I_STRESS
    0.3,    # I_LONELINESS
    0.25,   # I_INSECURITY
    0.1,    # I_IRRITATION
    0.4,    # I_LONGING
    0.6,    # I_SOCIAL_BATTERY
    0.15,   # I_MENTAL_FATIGUE
], dtype=np.float64)

DEFAULT_RELATIONSHIP: np.ndarray = np.array([
    0.3,    # R_AFFECTION
    0.3,    # R_TRUST
    0.2,    # R_FAMILIARITY
    0.15,   # R_DEPENDENCY
    0.25,   # R_EMOTIONAL_SAFETY
    0.2,    # R_ROMANTIC_TENSION
], dtype=np.float64)

DEFAULT_HIDDEN: np.ndarray = np.array([
    0.0,    # H_SUPPRESSED_SADNESS
    0.0,    # H_SUPPRESSED_ANGER
    0.05,   # H_HIDDEN_AFFECTION
], dtype=np.float64)

# ── 默认门控状态（全部开放） ──
DEFAULT_GATES: np.ndarray = np.array([
    0.0,    # G_SUPPRESSION    = 不压抑
    0.5,    # G_VULNERABILITY  = 适度示弱
    0.5,    # G_ATTACHMENT     = 依恋敏感度居中
    0.2,    # G_LEAKAGE        = 少量泄漏
], dtype=np.float64)


# ============================================================
# ① Stimulus Construction —— 社交信号 → 心理意义空间
# ============================================================
#
# 将"社交信号"（affection、rejection……）映射为"心理刺激"
#（abandonment_stimulus、validation_stimulus……）。
#
# 公式: stimuli[s] = Σ_ss signals[ss] × W_signal_to_stimulus[ss, s]
# 其中 emotional_weight_stimulus 从 impact 直接取得。

def _build_signal_to_stimulus() -> np.ndarray:
    """构建社交信号→心理刺激映射矩阵（SS_SIZE × ST_SIZE）。"""
    W = np.zeros((SS_SIZE, ST_SIZE), dtype=np.float64)

    # abandonment_stimulus: 被抛弃恐惧
    W[SS_REJECTION, ST_ABANDONMENT] = 0.7
    W[SS_ABANDONMENT, ST_ABANDONMENT] = 1.2

    # validation_stimulus: 被认可/被重视
    W[SS_APPROVAL, ST_VALIDATION] = 0.8
    W[SS_AFFECTION, ST_VALIDATION] = 0.3

    # closeness_stimulus: 亲密靠近
    W[SS_INTIMACY, ST_CLOSENESS] = 0.9
    W[SS_ATTENTION, ST_CLOSENESS] = 0.2

    # conflict_stimulus: 冲突张力
    W[SS_CONFLICT, ST_CONFLICT] = 1.0
    W[SS_REJECTION, ST_CONFLICT] = 0.3

    # dependency_stimulus: 被依赖
    W[SS_DEPENDENCY, ST_DEPENDENCY] = 0.8

    # teasing_stimulus: 被逗弄
    W[SS_TEASING, ST_TEASING] = 0.7

    return W


SIGNAL_TO_STIMULUS = _build_signal_to_stimulus()


def construct_stimuli(
    signals: np.ndarray,
    impact: np.ndarray,
) -> np.ndarray:
    """① 刺激构造：社交信号 → 心理意义空间

    参数：
      signals: SocialSignals (9 维)
      impact:  InteractionImpact (4 维)

    返回：
      stimuli: StimulusVector (7 维)
    """
    stimuli = signals @ SIGNAL_TO_STIMULUS  # (9,) @ (9,7) → (7,)
    # emotional_weight 直接从 impact 映射
    stimuli[ST_EMOTIONAL_WEIGHT] = impact[II_EMOTIONAL_WEIGHT]
    return stimuli


# ============================================================
# ② Trait Modulation —— 人格特质放大/衰减心理刺激
# ============================================================
#
# 每个心理刺激被相关特质调制。
# 公式: modulated[s] = stimulus[s] × (1 + Σ_t (traits[t]-0.5) × M[t, s])
# 其中 M[t, s] > 0 表示特质 t 会放大刺激 s。
#
# 例: 高 attachment_anxiety (0.7) → abandonment 效果乘 (1 + 0.2×0.5) = 1.1x

def _build_trait_modulation() -> np.ndarray:
    """构建特质调制矩阵（T_SIZE × ST_SIZE）。

    值 = 特质每变化 0.1 对刺激的放大倍率。
    """
    M = np.zeros((T_SIZE, ST_SIZE), dtype=np.float64)

    # 依恋焦虑 → 放大被抛弃恐惧和亲密靠近
    M[T_ATTACHMENT_ANXIETY, ST_ABANDONMENT] = 0.5
    M[T_ATTACHMENT_ANXIETY, ST_CLOSENESS] = 0.3

    # 嫉妒敏感 → 放大被抛弃和逗弄
    M[T_JEALOUSY_SENSITIVITY, ST_ABANDONMENT] = 0.4
    M[T_JEALOUSY_SENSITIVITY, ST_TEASING] = 0.2

    # 易怒 → 放大冲突
    M[T_ANGER_REACTIVITY, ST_CONFLICT] = 0.5
    M[T_ANGER_REACTIVITY, ST_ABANDONMENT] = 0.2

    # 自尊 → 抑制被认可（不屑于），放大被逗弄（不爽）
    M[T_PRIDE, ST_VALIDATION] = -0.2
    M[T_PRIDE, ST_TEASING] = 0.3

    # 情绪稳定 → 抑制冲突和抛弃恐惧
    M[T_EMOTIONAL_STABILITY, ST_CONFLICT] = -0.3
    M[T_EMOTIONAL_STABILITY, ST_ABANDONMENT] = -0.2

    return M


TRAIT_MODULATION = _build_trait_modulation()


def modulate_by_traits(
    stimuli: np.ndarray,
    traits: np.ndarray,
) -> np.ndarray:
    """② 人格调制：stimulus[s] × (1 + Σ traits_dev[t] × M[t,s])"""
    trait_dev = traits - 0.5                             # (T,)
    amp = 1.0 + trait_dev @ TRAIT_MODULATION             # (T,)@(T,ST) → (ST,)
    return stimuli * amp


# ============================================================
# ③ Relationship Modulation —— 关系状态调制刺激
# ============================================================
#
# 同一句话，陌生人 vs 亲密对象，心理意义不同。
#
# 公式: modulated[s] = stimulus[s] × rel_factor[s]
# 其中 rel_factor[s] 由关系状态计算。

def _build_rel_modulation() -> np.ndarray:
    """构建关系调制矩阵（R_SIZE × ST_SIZE）。

    值 = 关系维度对刺激的放大系数。
    """
    M = np.zeros((R_SIZE, ST_SIZE), dtype=np.float64)

    # 情感安全感越高 → 被抛弃恐惧越低
    M[R_EMOTIONAL_SAFETY, ST_ABANDONMENT] = -0.5

    # 好感度越高 → 被认可感越强
    M[R_AFFECTION, ST_VALIDATION] = 0.3

    # 情感安全感越高 → 亲密靠近越自然
    M[R_EMOTIONAL_SAFETY, ST_CLOSENESS] = 0.2

    # 信任度越高 → 冲突感越低
    M[R_TRUST, ST_CONFLICT] = -0.3

    # 依赖度越高 → 被需要感越强
    M[R_DEPENDENCY, ST_DEPENDENCY] = 0.3

    return M


REL_MODULATION = _build_rel_modulation()


def modulate_by_relationship(
    stimuli: np.ndarray,
    relationship: np.ndarray,
) -> np.ndarray:
    """③ 关系调制：stimulus[s] × (1 + Σ relationship[r] × M[r,s])"""
    amp = 1.0 + relationship @ REL_MODULATION            # (R,)@(R,ST) → (ST,)
    return stimuli * amp


# ============================================================
# ④ Gate Control —— 压抑/脆弱/依恋/泄漏 门控
# ============================================================
#
# 这是"角色潜意识"的核心——决定多少心理刺激允许进入内心。
#
# Gate 的计算取决于 traits + 当前 hidden 状态：
#   - suppression_gate:  压抑程度，traits 决定基线，hidden 累积会拉高
#   - vulnerability_gate: 示弱意愿，pride↓ + emotional_openness↑
#   - attachment_gate:   依恋敏感度
#   - leakage_gate:      压抑太久后的泄漏

def compute_gates(
    traits: np.ndarray,
    hidden: np.ndarray,
) -> np.ndarray:
    """④ 门控计算：根据特质和隐藏状态计算各门控值。"""
    gates = np.zeros(G_SIZE, dtype=np.float64)

    # Suppression Gate: 高自尊 + 低情绪开放 → 压抑
    #               + 隐藏情绪积累会拉高压抑
    base_suppression = (
        traits[T_PRIDE] * 0.4
        + (1.0 - traits[T_EMOTIONAL_OPENNESS]) * 0.3
        + (1.0 - traits[T_EMOTIONAL_STABILITY]) * 0.3
    )
    hidden_burden = (
        hidden[H_SUPPRESSED_SADNESS] * 0.2
        + hidden[H_SUPPRESSED_ANGER] * 0.3
        + hidden[H_HIDDEN_AFFECTION] * 0.1
    )
    gates[G_SUPPRESSION] = np.clip(base_suppression + hidden_burden, 0.0, 1.0)

    # Vulnerability Gate: 低自尊 + 高开放 → 示弱
    gates[G_VULNERABILITY] = np.clip(
        (1.0 - traits[T_PRIDE]) * 0.5
        + traits[T_EMOTIONAL_OPENNESS] * 0.3
        + traits[T_SENSITIVITY] * 0.2,
        0.0, 1.0,
    )

    # Attachment Gate: 依恋焦虑高 → 对关系刺激更敏感
    gates[G_ATTACHMENT] = np.clip(
        traits[T_ATTACHMENT_ANXIETY] * 0.6
        + (1.0 - traits[T_ATTACHMENT_AVOIDANCE]) * 0.4,
        0.0, 1.0,
    )

    # Leakage Gate: 隐藏状态积累太多 → 开始泄漏
    total_hidden = (hidden[H_SUPPRESSED_SADNESS] + hidden[H_SUPPRESSED_ANGER]
                    + hidden[H_HIDDEN_AFFECTION]) / 3.0
    gates[G_LEAKAGE] = np.clip(total_hidden * 1.2 - 0.2, 0.0, 1.0)

    return gates


def apply_gates(
    stimuli: np.ndarray,
    gates: np.ndarray,
) -> np.ndarray:
    """④ 应用门控：stimulus[s] × gate_factor[s]

    被压抑的刺激减弱，依恋相关的刺激受 attachment_gate 调制。
    """
    gated = stimuli.copy()

    # 整体压抑：所有刺激被衰减
    gated *= (1.0 - gates[G_SUPPRESSION] * 0.6)

    # 依恋门：abandonment 和 closeness 额外受 attachment 调制
    gated[ST_ABANDONMENT] *= gates[G_ATTACHMENT]
    gated[ST_CLOSENESS] *= gates[G_ATTACHMENT]

    # 脆弱门：示弱意愿高 → validation 和 closeness 更有影响
    gated[ST_VALIDATION] *= (0.5 + gates[G_VULNERABILITY] * 0.5)
    gated[ST_CLOSENESS] *= (0.5 + gates[G_VULNERABILITY] * 0.5)

    return gated


# ============================================================
# ⑤ Internal Dynamics —— 内部动力系统
# ============================================================
#
# h_t = A·h_{t-1} + B·e_t + c
#
# 其中：
#   A = 状态耦合矩阵（stress → irritation, loneliness → insecurity……）
#   B = 输入影响矩阵（abandonment_stimulus → insecurity↑ 等）
#   c = 人格偏置向量（高 optimism 自然恢复，高 anxiety 基线更高）

def _build_state_coupling() -> np.ndarray:
    """构建内部状态耦合矩阵 A（I_SIZE × I_SIZE）。

    A[i, j] = h_{t-1}[j] 对 h_t[i] 的影响。
    正值 = 正耦合（stress↑ → irritation↑），负值 = 负耦合（energy↓ → stress↑）。
    """
    A = np.zeros((I_SIZE, I_SIZE), dtype=np.float64)

    # 压力 → 烦躁、疲劳、孤独
    A[I_IRRITATION, I_STRESS] = 0.15
    A[I_MENTAL_FATIGUE, I_STRESS] = 0.10
    A[I_LONELINESS, I_STRESS] = 0.08

    # 孤独 → 不安全感、渴望
    A[I_INSECURITY, I_LONELINESS] = 0.12
    A[I_LONGING, I_LONELINESS] = 0.15

    # 社交电量耗尽 → 疲劳、烦躁
    A[I_MENTAL_FATIGUE, I_SOCIAL_BATTERY] = -0.10
    A[I_IRRITATION, I_SOCIAL_BATTERY] = -0.08

    # 精力充沛 → 积极状态
    A[I_STRESS, I_ENERGY] = -0.05
    A[I_LONELINESS, I_ENERGY] = -0.05

    # 不安全感 → 压力
    A[I_STRESS, I_INSECURITY] = 0.10

    # 对角线 = 自保持（惯性）
    np.fill_diagonal(A, 0.85)

    return A


def _build_input_influence() -> np.ndarray:
    """构建输入影响矩阵 B（ST_SIZE × I_SIZE）。

    B[s, i] = 刺激 s 对状态 i 的影响权重。
    """
    B = np.zeros((ST_SIZE, I_SIZE), dtype=np.float64)

    # abandoned → insecurity↑, loneliness↑, stress↑, longing↑
    B[ST_ABANDONMENT, I_INSECURITY] = 0.30
    B[ST_ABANDONMENT, I_LONELINESS] = 0.20
    B[ST_ABANDONMENT, I_STRESS] = 0.15
    B[ST_ABANDONMENT, I_LONGING] = 0.20
    B[ST_ABANDONMENT, I_ENERGY] = -0.15

    # validation → insecurity↓, energy↑
    B[ST_VALIDATION, I_INSECURITY] = -0.20
    B[ST_VALIDATION, I_ENERGY] = 0.15
    B[ST_VALIDATION, I_LONELINESS] = -0.15

    # closeness → loneliness↓, longing↑, social_battery↓
    B[ST_CLOSENESS, I_LONELINESS] = -0.25
    B[ST_CLOSENESS, I_LONGING] = -0.10
    B[ST_CLOSENESS, I_SOCIAL_BATTERY] = -0.10
    B[ST_CLOSENESS, I_ENERGY] = 0.08

    # conflict → stress↑, irritation↑, energy↓, fatigue↑, social_battery↓
    B[ST_CONFLICT, I_STRESS] = 0.35
    B[ST_CONFLICT, I_IRRITATION] = 0.30
    B[ST_CONFLICT, I_ENERGY] = -0.20
    B[ST_CONFLICT, I_MENTAL_FATIGUE] = 0.25
    B[ST_CONFLICT, I_SOCIAL_BATTERY] = -0.25

    # dependency → social_battery↓, loneliness↓, energy↑
    B[ST_DEPENDENCY, I_SOCIAL_BATTERY] = -0.08
    B[ST_DEPENDENCY, I_LONELINESS] = -0.15
    B[ST_DEPENDENCY, I_ENERGY] = 0.05

    # teasing → social_battery↓, irritation↑, energy↑
    B[ST_TEASING, I_SOCIAL_BATTERY] = -0.08
    B[ST_TEASING, I_IRRITATION] = 0.05
    B[ST_TEASING, I_ENERGY] = 0.05

    # emotional_weight → stress↑, mental_fatigue↑
    B[ST_EMOTIONAL_WEIGHT, I_STRESS] = 0.20
    B[ST_EMOTIONAL_WEIGHT, I_MENTAL_FATIGUE] = 0.15

    return B


def _build_personality_bias() -> np.ndarray:
    """构建人格偏置向量 c（I_SIZE）。

    决定角色"自然状态"的基线。
    高 optimism → 正能量自然恢复；高 anxiety → 基线压力偏高。
    """
    c = np.zeros(I_SIZE, dtype=np.float64)
    # 这些是"每轮自然偏移"，会被 A 矩阵和 clamp 约束
    c[I_ENERGY] = 0.01            # 活跃角色自然恢复
    c[I_LONELINESS] = -0.005      # 孤独自然缓解
    c[I_IRRITATION] = -0.01       # 烦躁自然消退
    return c


STATE_COUPLING_A = _build_state_coupling()
INPUT_INFLUENCE_B = _build_input_influence()
PERSONALITY_BIAS_C = _build_personality_bias()


def update_internal_dynamics(
    current: np.ndarray,
    gated_stimuli: np.ndarray,
    traits: np.ndarray,
) -> np.ndarray:
    """⑤ 内部动力系统：h_t = A·h_{t-1} + B·e_t + c

    参数：
      current:       当前内部状态 (8 维)
      gated_stimuli: 门控后的心理刺激 (7 维)
      traits:        特质 (10 维)

    返回：
      更新后的内部状态 (8 维)
    """
    # 状态耦合项
    coupling = STATE_COUPLING_A @ current                     # (I,)@(I,I) → (I,)

    # 输入影响项
    influence = gated_stimuli @ INPUT_INFLUENCE_B             # (ST,)@(ST,I) → (I,)

    # 人格偏置（受 traits 影响）
    bias = PERSONALITY_BIAS_C.copy()
    # 高 optimism → 额外正偏置
    bias[I_ENERGY] += (traits[T_OPTIMISM] - 0.5) * 0.02
    bias[I_STRESS] -= (traits[T_OPTIMISM] - 0.5) * 0.01
    # 高 anxiety_proneness → 额外压力偏置
    bias[I_STRESS] += (traits[T_ANXIETY_PRONENESS] - 0.5) * 0.02
    bias[I_INSECURITY] += (traits[T_ANXIETY_PRONENESS] - 0.5) * 0.01

    new_state = coupling + influence + bias
    return np.clip(new_state, 0.0, 1.0)


# ============================================================
# ⑤b. Relationship Dynamics —— 关系动力系统
# ============================================================
#
# relationship_t = A_rel · relationship_{t-1} + B_rel · e_t + impact_direct

def _build_rel_state_coupling() -> np.ndarray:
    """构建关系状态耦合矩阵 A_rel（R_SIZE × R_SIZE）。"""
    A = np.zeros((R_SIZE, R_SIZE), dtype=np.float64)

    # 好感 → 信任、熟悉
    A[R_TRUST, R_AFFECTION] = 0.08
    A[R_FAMILIARITY, R_AFFECTION] = 0.05

    # 信任 → 情感安全、依赖
    A[R_EMOTIONAL_SAFETY, R_TRUST] = 0.10
    A[R_DEPENDENCY, R_TRUST] = 0.05

    # 熟悉 → 情感安全
    A[R_EMOTIONAL_SAFETY, R_FAMILIARITY] = 0.08

    # 情感安全 → 好感、信任
    A[R_AFFECTION, R_EMOTIONAL_SAFETY] = 0.05
    A[R_TRUST, R_EMOTIONAL_SAFETY] = 0.05

    # 浪漫张力 → 好感
    A[R_AFFECTION, R_ROMANTIC_TENSION] = 0.03

    # 依赖 → 浪漫张力
    A[R_ROMANTIC_TENSION, R_DEPENDENCY] = 0.05

    # 对角线
    np.fill_diagonal(A, 0.90)

    return A


def _build_rel_input_influence() -> np.ndarray:
    """构建关系输入影响矩阵 B_rel（ST_SIZE × R_SIZE）。"""
    B = np.zeros((ST_SIZE, R_SIZE), dtype=np.float64)

    # abandoned → trust↓, safety↓, tension↑, dependency↑
    B[ST_ABANDONMENT, R_TRUST] = -0.12
    B[ST_ABANDONMENT, R_EMOTIONAL_SAFETY] = -0.15
    B[ST_ABANDONMENT, R_ROMANTIC_TENSION] = 0.08
    B[ST_ABANDONMENT, R_DEPENDENCY] = 0.10

    # validation → affection↑, trust↑
    B[ST_VALIDATION, R_AFFECTION] = 0.12
    B[ST_VALIDATION, R_TRUST] = 0.10

    # closeness → affection↑, familiarity↑, safety↑, tension↑
    B[ST_CLOSENESS, R_AFFECTION] = 0.10
    B[ST_CLOSENESS, R_FAMILIARITY] = 0.12
    B[ST_CLOSENESS, R_EMOTIONAL_SAFETY] = 0.08
    B[ST_CLOSENESS, R_ROMANTIC_TENSION] = 0.06

    # conflict → trust↓, safety↓, affection↓, tension↓
    B[ST_CONFLICT, R_TRUST] = -0.18
    B[ST_CONFLICT, R_EMOTIONAL_SAFETY] = -0.20
    B[ST_CONFLICT, R_AFFECTION] = -0.08
    B[ST_CONFLICT, R_ROMANTIC_TENSION] = -0.08

    # dependency → dependency↑, familiarity↑, safety↑
    B[ST_DEPENDENCY, R_DEPENDENCY] = 0.18
    B[ST_DEPENDENCY, R_FAMILIARITY] = 0.06
    B[ST_DEPENDENCY, R_EMOTIONAL_SAFETY] = 0.05

    # teasing → familiarity↑, tension↑
    B[ST_TEASING, R_FAMILIARITY] = 0.08
    B[ST_TEASING, R_ROMANTIC_TENSION] = 0.08

    return B


REL_STATE_COUPLING_A = _build_rel_state_coupling()
REL_INPUT_INFLUENCE_B = _build_rel_input_influence()


def update_relationship_dynamics(
    current: np.ndarray,
    gated_stimuli: np.ndarray,
    impact: np.ndarray,
) -> np.ndarray:
    """⑤b. 关系动力系统：rel_t = A_rel·rel_{t-1} + B_rel·e_t + impact_direct"""
    coupling = REL_STATE_COUPLING_A @ current
    influence = gated_stimuli @ REL_INPUT_INFLUENCE_B

    new_state = coupling + influence

    # impact 直接效应
    ci = impact[II_CLOSENESS_IMPACT]
    if ci > 0:
        new_state[R_FAMILIARITY] += ci * 0.15
        new_state[R_EMOTIONAL_SAFETY] += ci * 0.12
    elif ci < 0:
        new_state[R_EMOTIONAL_SAFETY] += ci * 0.15
        new_state[R_TRUST] += ci * 0.10

    ti = impact[II_TRUST_IMPACT]
    if ti != 0:
        new_state[R_TRUST] += ti * 0.15

    return np.clip(new_state, 0.0, 1.0)


# ============================================================
# ⑥ Decay —— 各维度不同衰减速率
# ============================================================
#
# 不同心理维度有不同的时间尺度：
#   - irritation:  快衰减（容易消气）
#   - longing:     慢衰减（思念难消）
#   - trust:       极慢衰减（信任一旦建立）

INTERNAL_DECAY: np.ndarray = np.array([
    0.98,   # I_ENERGY — 精力自然恢复
    0.92,   # I_STRESS — 压力缓慢消解
    0.95,   # I_LONELINESS — 孤独缓慢消解
    0.95,   # I_INSECURITY — 不安全感缓慢消解
    0.85,   # I_IRRITATION — 烦躁消退快
    0.97,   # I_LONGING — 思念消退慢
    0.93,   # I_SOCIAL_BATTERY — 社交电量恢复
    0.90,   # I_MENTAL_FATIGUE — 精神疲劳消解
], dtype=np.float64)

RELATIONSHIP_DECAY: np.ndarray = np.array([
    0.995,  # R_AFFECTION — 好感衰减极慢
    0.990,  # R_TRUST — 信任衰减极慢
    0.985,  # R_FAMILIARITY — 熟悉缓慢衰减
    0.980,  # R_DEPENDENCY — 依赖缓慢衰减
    0.990,  # R_EMOTIONAL_SAFETY — 安全感衰减极慢
    0.970,  # R_ROMANTIC_TENSION — 张力适度衰减
], dtype=np.float64)

HIDDEN_DECAY: np.ndarray = np.array([
    0.93,   # H_SUPPRESSED_SADNESS
    0.90,   # H_SUPPRESSED_ANGER
    0.95,   # H_HIDDEN_AFFECTION
], dtype=np.float64)

# 使衰减以每轮为基准，decay 值越接近 1 越持久
# 通过 state = (state - mean) * decay + mean 的方式，
# 让状态自然回归基线而非归零


def apply_decay(state: np.ndarray, decay: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """⑥ 衰减：向基线回归而非归零

    state[t] = baseline + (state[t-1] - baseline) × decay
    """
    return np.clip(baseline + (state - baseline) * decay, 0.0, 1.0)


# ============================================================
# ⑦ Hidden Accumulation —— 被压抑的情绪积累
# ============================================================
#
# 经 Gate 判断为"不应表达"的情绪进入隐藏层。
# Δhidden = raw_emotion × suppression_gate
#
# raw_emotion 从内部状态变化推断：
#   - loneliness↑ + stress↑   → 悲伤
#   - irritation↑              → 愤怒
#   - longing↑, loneliness↓   → 好感

def accumulate_hidden(
    current_hidden: np.ndarray,
    new_internal: np.ndarray,
    old_internal: np.ndarray,
    gated_stimuli: np.ndarray,
    gates: np.ndarray,
    traits: np.ndarray,
) -> np.ndarray:
    """⑦ 隐藏情绪积累：被压抑的情绪进入隐藏层。"""
    h = current_hidden.copy()
    suppression = gates[G_SUPPRESSION]

    # 从内部状态变化推导"原始情绪"
    delta = new_internal - old_internal

    # 悲伤：loneliness↑ + stress↑
    raw_sadness = max(0.0, delta[I_LONELINESS]) + max(0.0, delta[I_STRESS]) * 0.5
    h[H_SUPPRESSED_SADNESS] += raw_sadness * suppression * 0.3

    # 愤怒：irritation↑
    raw_anger = max(0.0, delta[I_IRRITATION])
    h[H_SUPPRESSED_ANGER] += raw_anger * suppression * 0.3

    # 好感：来自 closeness/validation 刺激（如果被压抑）
    raw_affection = (gated_stimuli[ST_CLOSENESS] + gated_stimuli[ST_VALIDATION]) * 0.15
    h[H_HIDDEN_AFFECTION] += raw_affection * suppression

    # 特质修饰：高自尊额外压抑好感
    if traits[T_PRIDE] > 0.6:
        pride_extra = (traits[T_PRIDE] - 0.6) * 2
        h[H_HIDDEN_AFFECTION] += raw_affection * pride_extra * 0.3

    return np.clip(h, 0.0, 1.0)


# ============================================================
# ⑧ Event Trigger —— 离散人格事件
# ============================================================

def check_events(hidden: np.ndarray, traits: np.ndarray) -> list:
    """⑧ 检查隐藏状态是否积累到突破阈值。"""
    events = []

    if hidden[H_HIDDEN_AFFECTION] > 0.85:
        events.append("AFFECTION_BREAKTHROUGH")
    if hidden[H_SUPPRESSED_SADNESS] > 0.85:
        events.append("SADNESS_BREAKTHROUGH")
    if hidden[H_SUPPRESSED_ANGER] > 0.80:
        events.append("ANGER_BREAKTHROUGH")
    if (hidden[H_SUPPRESSED_SADNESS] > 0.6
            and traits[T_ATTACHMENT_ANXIETY] > 0.6
            and hidden[H_HIDDEN_AFFECTION] > 0.5):
        events.append("CLINGY_BREAKTHROUGH")

    return events


# ============================================================
# ⑨ Surface Projection —— 内部状态 → 表层表达
# ============================================================
#
# SurfaceState 不存储，每轮从内部状态动态投影。
# y_t = P · [internal, relationship, hidden, traits]
#
# 这里 P 不是纯粹的线性矩阵——有非线性 clamp 和条件逻辑。
# 但核心思想是：表面表达 = 内部状态的"可见部分"。

def project_surface(
    internal: np.ndarray,
    relationship: np.ndarray,
    hidden: np.ndarray,
    traits: np.ndarray,
    gates: np.ndarray,
) -> np.ndarray:
    """⑨ 表层投影：内部状态 → 表面表达（动态计算，不存储）。"""
    s = np.zeros(S_SIZE, dtype=np.float64)

    # ── 内部 → 表面基线 ──
    s[S_EXPRESSIVENESS] = 0.3 + internal[I_ENERGY] * 0.4 - internal[I_MENTAL_FATIGUE] * 0.3
    s[S_WARMTH] = 0.3 + relationship[R_AFFECTION] * 0.4 - internal[I_STRESS] * 0.2
    s[S_SHARPNESS] = 0.1 + internal[I_IRRITATION] * 0.5 + internal[I_STRESS] * 0.2
    s[S_SOFTNESS] = 0.2 + (1.0 - internal[I_STRESS]) * 0.3 + relationship[R_EMOTIONAL_SAFETY] * 0.2
    s[S_ENTHUSIASM] = 0.3 + internal[I_ENERGY] * 0.5 - internal[I_MENTAL_FATIGUE] * 0.3
    s[S_RESTRAINT] = 0.2 + internal[I_INSECURITY] * 0.3 + traits[T_PRIDE] * 0.2
    s[S_VULNERABILITY] = 0.1 + internal[I_LONELINESS] * 0.3 + internal[I_LONGING] * 0.2 - traits[T_PRIDE] * 0.2

    # ── 隐藏状态泄漏效应 ──
    leakage = gates[G_LEAKAGE]

    if hidden[H_SUPPRESSED_SADNESS] > 0.4:
        s[S_WARMTH] -= hidden[H_SUPPRESSED_SADNESS] * 0.15 * (0.5 + leakage * 0.5)
        s[S_RESTRAINT] += hidden[H_SUPPRESSED_SADNESS] * 0.10 * (0.5 + leakage * 0.5)

    if hidden[H_SUPPRESSED_ANGER] > 0.3:
        s[S_SHARPNESS] += hidden[H_SUPPRESSED_ANGER] * 0.20 * (0.5 + leakage * 0.5)
        s[S_WARMTH] -= hidden[H_SUPPRESSED_ANGER] * 0.10 * (0.5 + leakage * 0.5)

    if hidden[H_HIDDEN_AFFECTION] > 0.5:
        s[S_VULNERABILITY] += hidden[H_HIDDEN_AFFECTION] * 0.10 * leakage
        s[S_RESTRAINT] += hidden[H_HIDDEN_AFFECTION] * 0.15 * (1.0 - leakage)

    # ── 特质修饰 ──
    if traits[T_PRIDE] > 0.6:
        s[S_SHARPNESS] += traits[T_PRIDE] * 0.10
        s[S_VULNERABILITY] -= traits[T_PRIDE] * 0.15

    if traits[T_EMOTIONAL_OPENNESS] > 0.6:
        s[S_EXPRESSIVENESS] += traits[T_EMOTIONAL_OPENNESS] * 0.10
        s[S_RESTRAINT] -= traits[T_EMOTIONAL_OPENNESS] * 0.10

    if traits[T_OPTIMISM] > 0.6:
        s[S_ENTHUSIASM] += traits[T_OPTIMISM] * 0.10

    return np.clip(s, 0.0, 1.0)


# ============================================================
# 主入口：pipeline 编排
# ============================================================

def initialize_all(traits: np.ndarray) -> dict:
    """首次运行：用 Traits 初始化所有状态层为合理默认值。"""
    internal = DEFAULT_INTERNAL.copy()
    relationship = DEFAULT_RELATIONSHIP.copy()
    hidden = DEFAULT_HIDDEN.copy()
    gates = DEFAULT_GATES.copy()
    surface = project_surface(internal, relationship, hidden, traits, gates)

    return {
        "internal_state": internal,
        "relationship_state": relationship,
        "hidden_state": hidden,
        "surface_state": surface,
    }


def update_all(
    current_internal: Optional[np.ndarray],
    current_relationship: Optional[np.ndarray],
    current_hidden: Optional[np.ndarray],
    traits: np.ndarray,
    signals: np.ndarray,
    impact: np.ndarray,
) -> dict:
    """State Engine 主入口——完整分层 pipeline。

    参数：
      current_*: 当前各状态层（首次为 None 则自动初始化）
      traits:    角色特质 (10 维)
      signals:   SocialSignals (9 维)
      impact:    InteractionImpact (4 维)

    返回：
      {
        "internal_state":      np.ndarray,   # 更新后的内部状态
        "relationship_state":   np.ndarray,   # 更新后的关系状态
        "hidden_state":         np.ndarray,   # 更新后的隐藏状态
        "surface_state":        np.ndarray,   # 本轮表面表达（动态投影，不存储）
        "triggered_events":     list[str],    # 触发的事件列表
      }
    """
    # ── 首次运行初始化 ──
    if current_internal is None:
        return initialize_all(traits)

    # ── ① 刺激构造 ──
    stimuli = construct_stimuli(signals, impact)

    # ── ② 特质调制 ──
    stimuli = modulate_by_traits(stimuli, traits)

    # ── ③ 关系调制 ──
    stimuli = modulate_by_relationship(stimuli, current_relationship)

    # ── ④ 门控 ──
    gates = compute_gates(traits, current_hidden)
    gated_stimuli = apply_gates(stimuli, gates)

    # ── ⑤ 内部动力系统 ──
    new_internal = update_internal_dynamics(current_internal, gated_stimuli, traits)
    new_internal = apply_decay(new_internal, INTERNAL_DECAY, DEFAULT_INTERNAL)

    # ── ⑤b. 关系动力系统 ──
    new_relationship = update_relationship_dynamics(current_relationship, gated_stimuli, impact)
    new_relationship = apply_decay(new_relationship, RELATIONSHIP_DECAY, DEFAULT_RELATIONSHIP)

    # ── ⑦ 隐藏情绪积累 ──
    new_hidden = accumulate_hidden(
        current_hidden, new_internal, current_internal,
        gated_stimuli, gates, traits,
    )
    new_hidden = apply_decay(new_hidden, HIDDEN_DECAY, DEFAULT_HIDDEN)

    # ── ⑧ 事件检测 ──
    triggered_events = check_events(new_hidden, traits)
    if triggered_events:
        import logging
        logging.getLogger(__name__).info("State Engine 触发事件: %s", triggered_events)

    # ── ⑨ 表层投影（不存储，仅返回供 LLM 参考） ──
    surface = project_surface(new_internal, new_relationship, new_hidden, traits, gates)

    return {
        "internal_state": new_internal,
        "relationship_state": new_relationship,
        "hidden_state": new_hidden,
        "surface_state": surface,
        "triggered_events": triggered_events,
    }
