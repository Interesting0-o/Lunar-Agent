"""
state_engine —— 连续人格动力系统

角色"潜意识"的核心,将外部心理刺激转化为连续心理变化。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
架构分层(数据流从上到下)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Perception (user_stimuli, 7 维)
       ↓
  ① Gate Control (融合 trait/rel/gate 三层)
       │   - sigmoid 软阈值门控(其他环节保持线性)
       ├─→ inner_stimuli (7 维)            进入内部状态
       └─→ outer_stimuli (7 维)            进入表面表达
       ↓
  ② Internal Dynamics                  LSTM 式 3 门控更新
       │   h_t = f ⊙ h_{t-1} + i ⊙ (A·h_{t-1} + B·e_t) + g ⊙ bias
       ↓  f=遗忘门, i=接受门, g=自生门
  ③ Dynamic Decay                      人格驱动的软衰减
       ↓
  ④ Surface Projection                 内部状态 + outer_stimuli → 表面(动态投影)
       ↓
  ⑤ Relationship Drift (LTI 保持)       关系状态慢演化
       ↓
  LLM Generation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心设计原则
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 纯函数:相同输入 → 相同输出
2. 可组合:每层职责单一,可独立测试
3. 激活 vs 饱和分离:
   - 门控用 sigmoid(软开关)
   - 边界用 soft_clamp(渐近饱和)
   - 其他环节保持线性(可解释、可调参)
4. 里/外表情分离:
   - inner_stimuli 决定状态变化(进入动力系统)
   - outer_stimuli 决定表面表达(进入 LLM prompt)
   - 差异 = 压抑强度(口是心非)
5. LSTM 式 3 门控:
   - f 遗忘门:放下过去(情绪稳定↑、乐观↑ → 强)
   - i 接受门:被新刺激触动(高压抑 → 弱,高依恋敏感 → 强)
   - g 自生门:内心自生情绪(关系好 → 正面,高焦虑 → 压力)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
from state import (
    # 内部状态
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE, I_SIZE,
    # 关系状态
    R_AFFECTION, R_TRUST, R_FAMILIARITY, R_DEPENDENCY,
    R_EMOTIONAL_SAFETY, R_ROMANTIC_TENSION, R_SIZE,
    # 表面状态
    S_EXPRESSIVENESS, S_WARMTH, S_SHARPNESS, S_SOFTNESS,
    S_ENTHUSIASM, S_RESTRAINT, S_VULNERABILITY, S_SIZE,
    # 特质
    T_SENSITIVITY, T_PRIDE, T_EMOTIONAL_OPENNESS, T_EMOTIONAL_STABILITY,
    T_OPTIMISM, T_ANXIETY_PRONENESS, T_ANGER_REACTIVITY, T_JEALOUSY_SENSITIVITY,
    T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE, T_SIZE,
    # 心理刺激
    ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT,
    ST_DEPENDENCY, ST_TEASING, ST_EMOTIONAL_WEIGHT, ST_SIZE,
    # 门控
    G_SUPPRESSION, G_VULNERABILITY, G_ATTACHMENT, G_LEAKAGE, G_SIZE,
)
from typing import Optional
from default_state import (
    DEFAULT_TRAITS, DEFAULT_INTERNAL, DEFAULT_RELATIONSHIP,
)


# ============================================================
# 工具函数
# ============================================================

def soft_clamp(
    x: np.ndarray,
    low: float = 0.0,
    high: float = 1.0,
    transition: float = 0.1,
) -> np.ndarray:
    """软饱和裁剪。

    [low, high] 区间内 = np.clip(完全兼容)。
    区间外用 tanh 平滑压回,保留"超出量"信息,有明确渐近线。

    行为(transition=0.1, low=0, high=1):
      x=1.00   → 1.0000
      x=1.05   → 0.9975
      x=1.10   → 0.9999
      x=2.00   → 1.0000
      x=10.0   → 1.0000
      x=-10.0  → 0.0000
    """
    upper_delta = x - high
    upper_output = high - transition * np.tanh(upper_delta / transition)

    lower_delta = low - x
    lower_output = low + transition * np.tanh(lower_delta / transition)

    return np.where(
        x > high, upper_output,
        np.where(x < low, lower_output, x)
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid,处理 ±∞ 和大数不产生 NaN。"""
    pos_mask = x >= 0
    result = np.empty_like(x, dtype=np.float64)
    result[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    neg_x = x[~pos_mask]
    result[~pos_mask] = np.exp(neg_x) / (1.0 + np.exp(neg_x))
    return result


def _sigmoid_gate(raw: np.ndarray) -> np.ndarray:
    """门控专用 sigmoid 激活:中点居中(raw-0.5),值域 (0, 1)。

    与 np.clip 的差异:硬阈值 → 软阈值,符合心理学"防御机制软启动"。
    """
    return _sigmoid(raw - 0.5)


# ============================================================
# ① Gate Control —— 心理刺激三向门控
# ============================================================
#
# 输出:4 维 GateVector
#   G_SUPPRESSION (压抑强度):  决定多少情绪被压到里表情
#   G_VULNERABILITY (脆弱度):  决定是否愿意示弱
#   G_ATTACHMENT   (依恋敏感): 决定依恋类刺激(abandonment/closeness)放大倍数
#   G_LEAKAGE      (保留索引,不再使用)
#
# 公式模式: gate = sigmoid( trait_baseline × rel_mod + internal_push )
#   - trait_baseline: 人格决定的基线值
#   - rel_mod: 关系对基线的调制因子(信任→降低压抑,安全→鼓励示弱)
#   - internal_push: 内部状态的急性推动(压力→更压抑,孤独→更渴望表达)

def compute_gates(
    traits: np.ndarray,
    relationship: np.ndarray,
    current_internal: np.ndarray,
) -> np.ndarray:
    """① 门控计算:特质 × 关系 × 内部状态 → 3 维门控值。"""
    gates = np.zeros(G_SIZE, dtype=np.float64)

    # 压抑强度
    # 特质:高自尊+低开放+低稳定 → 压抑强
    trait_supp = (
        traits[T_PRIDE] * 0.4
        + (1.0 - traits[T_EMOTIONAL_OPENNESS]) * 0.3
        + (1.0 - traits[T_EMOTIONAL_STABILITY]) * 0.3
    )
    # 关系:信任和安全感 → 松动防御
    rel_supp = 1.0 - relationship[R_TRUST] * 0.20 - relationship[R_EMOTIONAL_SAFETY] * 0.15
    # 急性:压力/不安 → 加重压抑
    internal_supp = current_internal[I_STRESS] * 0.10 + current_internal[I_INSECURITY] * 0.08
    gates[G_SUPPRESSION] = _sigmoid_gate(trait_supp * rel_supp + internal_supp)

    # 脆弱度
    # 特质:低自尊+高开放+高敏感 → 示弱基线高
    trait_vuln = (
        (1.0 - traits[T_PRIDE]) * 0.5
        + traits[T_EMOTIONAL_OPENNESS] * 0.3
        + traits[T_SENSITIVITY] * 0.2
    )
    # 关系:情感安全和熟悉 → 鼓励示弱
    rel_vuln = 1.0 + relationship[R_EMOTIONAL_SAFETY] * 0.15 + relationship[R_FAMILIARITY] * 0.10
    # 急性:孤独和渴望 → push 示弱意愿
    internal_vuln = current_internal[I_LONELINESS] * 0.12 + current_internal[I_LONGING] * 0.10
    gates[G_VULNERABILITY] = _sigmoid_gate(trait_vuln * rel_vuln + internal_vuln)

    # 依恋敏感
    # 特质:高依恋焦虑 + 低依恋回避 → 依恋敏感基线高
    trait_att = (
        traits[T_ATTACHMENT_ANXIETY] * 0.6
        + (1.0 - traits[T_ATTACHMENT_AVOIDANCE]) * 0.4
    )
    # 关系:好感和浪漫张力 → 放大依恋敏感
    rel_att = 1.0 + relationship[R_AFFECTION] * 0.12 + relationship[R_ROMANTIC_TENSION] * 0.08
    # 急性:不安全感和渴望 → 急性放大
    internal_att = current_internal[I_INSECURITY] * 0.10 + current_internal[I_LONGING] * 0.08
    gates[G_ATTACHMENT] = _sigmoid_gate(trait_att * rel_att + internal_att)

    return gates


# ============================================================
# ①.b 刺激调制系数(原 trait/rel 调制,合并)
# ============================================================
#
# 合并理由:特质调制和关系调制本质上都是"刺激 × 角色状态"的乘法缩放
# 每个系数 ∈ [0, 2.0]:
#   - 1.0 = 中性
#   - > 1.0 = 放大
#   - < 1.0 = 缩小

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
    # 自尊 → 抑制被认可,放大被逗弄
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
    """①.b 应用门控,输出 (inner_stimuli, outer_stimuli) 两套。

    里外差异化:
      - 高压抑: outer 远弱于 inner(口是心非)
      - 低压抑: outer ≈ inner(心口一致)
      - 高脆弱: outer 可超过 inner(主动示弱)
    """
    mod = _compute_stimulus_modulation(traits, relationship)
    base = stimuli * mod

    suppression = gates[G_SUPPRESSION]
    attachment = gates[G_ATTACHMENT]
    vulnerability = gates[G_VULNERABILITY]

    # 里表情:角色真正"心理上"接收到的强度(依恋类受 attachment 调制)
    inner = base.copy()
    inner[ST_ABANDONMENT] *= attachment
    inner[ST_CLOSENESS]   *= attachment

    # 外表情:角色"实际表现出"的强度
    outer = base.copy()
    outer[ST_ABANDONMENT] *= attachment
    outer[ST_CLOSENESS]   *= attachment
    # 脆弱门:示弱意愿高 → validation/closeness 更有影响
    outer[ST_VALIDATION]  *= (0.5 + vulnerability * 0.5)
    outer[ST_CLOSENESS]   *= (0.5 + vulnerability * 0.5)
    # 压抑衰减:整体压向外表情
    outer *= (1.0 - suppression * 0.6)

    inner = soft_clamp(inner, 0.0, 1.0)
    outer = soft_clamp(outer, 0.0, 1.0)

    return inner, outer


# ============================================================
# ② Internal Dynamics —— LSTM 式 3 门控更新
# ============================================================
#
# h_t = f ⊙ h_{t-1} + i ⊙ raw_dynamics + g ⊙ bias
# raw_dynamics = A·h_{t-1} + B·e_t  (LTI 风格的"建议值")
#
# 三个门控的心理学语义:
#   f 遗忘门 ——"放下过去"的能力
#   i 接受门 ——"被触动"的程度
#   g 自生门 ——"内心自生"情绪

def _build_state_coupling() -> np.ndarray:
    """内部状态耦合矩阵 A(I_SIZE × I_SIZE)。

    A[i, j] = h_{t-1}[j] 对 h_t[i] 的影响。
    正值=正耦合,负值=负耦合,对角线=自保持(惯性)。
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

    np.fill_diagonal(A, 0.85)  # 自保持(惯性)
    return A


def _build_input_influence() -> np.ndarray:
    """输入影响矩阵 B(ST_SIZE × I_SIZE):B[s, i] = 刺激 s 对状态 i 的影响权重。"""
    B = np.zeros((ST_SIZE, I_SIZE), dtype=np.float64)

    # abandoned → insecurity↑, loneliness↑, stress↑, longing↑, energy↓
    B[ST_ABANDONMENT, I_INSECURITY] = 0.25
    B[ST_ABANDONMENT, I_LONELINESS] = 0.18
    B[ST_ABANDONMENT, I_STRESS] = 0.12
    B[ST_ABANDONMENT, I_LONGING] = 0.18
    B[ST_ABANDONMENT, I_ENERGY] = -0.12
    # validation → insecurity↓, energy↑, loneliness↓
    B[ST_VALIDATION, I_INSECURITY] = -0.20
    B[ST_VALIDATION, I_ENERGY] = 0.15
    B[ST_VALIDATION, I_LONELINESS] = -0.15
    # closeness → loneliness↓, longing↓, social_battery↓, energy↑
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
    """人格偏置向量 c(I_SIZE):每轮自然偏移,会被 A 矩阵和 clamp 约束。"""
    c = np.zeros(I_SIZE, dtype=np.float64)
    c[I_ENERGY] = 0.01       # 活跃角色自然恢复
    c[I_LONELINESS] = -0.005 # 孤独自然缓解
    c[I_IRRITATION] = -0.01  # 烦躁自然消退
    return c


STATE_COUPLING_A = _build_state_coupling()
INPUT_INFLUENCE_B = _build_input_influence()
PERSONALITY_BIAS_C = _build_personality_bias()


def update_internal_dynamics(
    current: np.ndarray,
    gated_stimuli: np.ndarray,
    traits: np.ndarray,
    relationship: np.ndarray,
    gates: np.ndarray,
) -> np.ndarray:
    """② 内部动力系统——LSTM 式 3 门控更新。

    主公式:
        h_t = f ⊙ h_{t-1} + i ⊙ (A·h_{t-1} + B·e_t) + g ⊙ bias
    三个门控由显式心理学变量构造,sigmoid 软阈值激活。

    调参经验:同一特质不要在多个门控中"对冲"
      - 情绪稳定主导 f_gate(遗忘),不在 i_gate 中再抑制
      - 衰减项作为 f_gate 的"后盾",增强稳定者真的能"放下"的能力

    参数:
      current:       当前内部状态 h_{t-1} (8 维)
      gated_stimuli: 已被门控调制过的"里表情"刺激 e_t (7 维)
      traits:        角色特质 (10 维)
      relationship:  当前关系状态 (6 维)——用于 g_gate 构造
      gates:         三门控值(3 维)
    返回:
      更新后的内部状态 h_t (8 维)
    """
    # LTI 风格的"建议值"
    coupling = STATE_COUPLING_A @ current
    influence = gated_stimuli @ INPUT_INFLUENCE_B
    raw_dynamics = coupling + influence

    # 遗忘门 f:高情绪稳定+高乐观 → 强;高依恋焦虑+高敏感 → 弱
    f_signal = (
        traits[T_EMOTIONAL_STABILITY] * 0.6
        + traits[T_OPTIMISM] * 0.3
        - traits[T_ATTACHMENT_ANXIETY] * 0.3
        - traits[T_SENSITIVITY] * 0.3
    )
    f_gate = _sigmoid_gate(f_signal + 0.3)

    # 接受门 i:高压抑 → 弱;高依恋敏感+高信任 → 强;高开放 → 强
    i_signal = (
        -gates[G_SUPPRESSION] * 0.4
        + gates[G_ATTACHMENT] * 0.2
        + relationship[R_TRUST] * 0.2
        + relationship[R_EMOTIONAL_SAFETY] * 0.2
        + (traits[T_EMOTIONAL_OPENNESS] - 0.5) * 0.3
    )
    i_gate = _sigmoid_gate(i_signal + 0.3)

    # 自生门 g:关系好 → 内部生正面;高焦虑 → 内部生压力
    g_signal = (
        (relationship[R_AFFECTION] - 0.5) * 0.4
        + (relationship[R_TRUST] - 0.5) * 0.3
        - (1.0 - relationship[R_TRUST]) * 0.2
        - traits[T_ANXIETY_PRONENESS] * 0.2
        + traits[T_OPTIMISM] * 0.2
    )
    g_gate = _sigmoid_gate(g_signal + 0.5)

    # 人格偏置(per-tick baseline)
    bias = PERSONALITY_BIAS_C.copy()
    bias[I_ENERGY]     += (traits[T_OPTIMISM] - 0.5) * 0.02
    bias[I_STRESS]     -= (traits[T_OPTIMISM] - 0.5) * 0.01
    bias[I_STRESS]     += (traits[T_ANXIETY_PRONENESS] - 0.5) * 0.02
    bias[I_INSECURITY] += (traits[T_ANXIETY_PRONENESS] - 0.5) * 0.01

    # LSTM 式 3 门控更新
    new_state = (
        f_gate * current          # 遗忘:保留多少旧里表情
        + i_gate * raw_dynamics   # 接受:新刺激进入多少
        + g_gate * bias           # 自生:内心基线偏移
    )
    return soft_clamp(new_state, 0.0, 1.0)


# ============================================================
# ③ Dynamic Decay —— 人格驱动的动态衰减
# ============================================================
#
# 衰减速率由以下因素动态调制:
#   - 人格基线:高傲→记仇,情绪稳定→恢复快
#   - 关系语境:信任高→压力消退快(安全基地效应)
#   - 急性状态:高压下所有负面情绪消退变慢(压力锁定)
#   - 刺激-特质共振:特定刺激遇到特定特质时,情绪自我增强
#
# decay < 1.0 → 向基线回归
# decay = 1.0 → 保持不变
# decay > 1.0 → 背离基线(情绪自我增强)
#
# 门控与衰减的 2×2 协同:
#   - 紧门控 + 慢衰减 = 压抑爆炸型
#   - 松门控 + 快衰减 = 表达恢复型
#   - 紧门控 + 快衰减 = 冷漠超然型
#   - 松门控 + 慢衰减 = 敏感内耗型

_INTERNAL_BASE_DECAY = np.array([
    0.98, 0.92, 0.95, 0.95, 0.85, 0.97, 0.93, 0.90
], dtype=np.float64)  # energy, stress, loneliness, insecurity, irritation, longing, battery, fatigue

_RELATIONSHIP_BASE_DECAY = np.array([
    0.995, 0.990, 0.985, 0.980, 0.990, 0.970
], dtype=np.float64)  # affection, trust, familiarity, dependency, safety, tension


def compute_dynamic_decay(
    traits: np.ndarray,
    relationship: np.ndarray,
    internal: np.ndarray,
    stimuli: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """③ 人格驱动的动态衰减计算。

    返回 (internal_decay, relationship_decay),可能 > 1.0 表示情绪自我增强。
    """
    idcy = _INTERNAL_BASE_DECAY.copy()
    rdcy = _RELATIONSHIP_BASE_DECAY.copy()

    # ── 人格调制 ──
    # 自尊:高 → 烦躁消退慢(记仇),不安全感消退也慢(不愿承认脆弱)
    pride_dev = traits[T_PRIDE] - 0.5
    idcy[I_IRRITATION] += pride_dev * 0.12
    idcy[I_INSECURITY] += pride_dev * 0.06

    # 情绪稳定:→ 所有负面情绪消退快
    stability_dev = traits[T_EMOTIONAL_STABILITY] - 0.5
    idcy[I_STRESS]       -= stability_dev * 0.15
    idcy[I_IRRITATION]   -= stability_dev * 0.12
    idcy[I_MENTAL_FATIGUE] -= stability_dev * 0.08
    idcy[I_INSECURITY]   -= stability_dev * 0.08
    idcy[I_LONELINESS]   -= stability_dev * 0.06

    # 依恋焦虑:→ 不安全感/渴望消退慢(总怕被丢下)
    attach_dev = traits[T_ATTACHMENT_ANXIETY] - 0.5
    idcy[I_INSECURITY] += attach_dev * 0.10
    idcy[I_LONGING]    += attach_dev * 0.08

    # 乐观:→ 孤独/不安全感消退快(天然自我调节)
    optimism_dev = traits[T_OPTIMISM] - 0.5
    idcy[I_LONELINESS] -= optimism_dev * 0.08
    idcy[I_INSECURITY] -= optimism_dev * 0.06

    # 易怒:→ 烦躁消退慢
    anger_dev = traits[T_ANGER_REACTIVITY] - 0.5
    idcy[I_IRRITATION] += anger_dev * 0.10

    # 敏感:→ 所有情绪体验更深,消退慢
    sensitivity_dev = traits[T_SENSITIVITY] - 0.5
    idcy[I_STRESS]     += sensitivity_dev * 0.04
    idcy[I_LONELINESS] += sensitivity_dev * 0.04
    idcy[I_INSECURITY] += sensitivity_dev * 0.04

    # 依恋回避:→ 关系衰减加速(回避型更难建立深层关系)
    avoidance_dev = traits[T_ATTACHMENT_AVOIDANCE] - 0.5
    rdcy[R_AFFECTION]       -= avoidance_dev * 0.004
    rdcy[R_TRUST]           -= avoidance_dev * 0.003
    rdcy[R_EMOTIONAL_SAFETY] -= avoidance_dev * 0.003

    # ── 关系调制 ──
    # 信任高 → 压力/不安全感消退更快(安全基地效应)
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
    # 浪漫张力高 → 好感消退慢(越在意越放不下)
    tension_effect = relationship[R_ROMANTIC_TENSION] * 0.002
    rdcy[R_AFFECTION] += tension_effect

    # ── 急性状态调制 ──
    # 高压力 → 所有负面情绪消退变慢(压力锁定效应)
    stress_penalty = internal[I_STRESS] * 0.04
    idcy[I_IRRITATION] += stress_penalty
    idcy[I_INSECURITY] += stress_penalty
    idcy[I_LONELINESS] += stress_penalty * 0.5
    # 精神疲劳 → 精力和社交电量恢复变慢
    fatigue_penalty = internal[I_MENTAL_FATIGUE] * 0.04
    idcy[I_ENERGY]        -= fatigue_penalty
    idcy[I_SOCIAL_BATTERY] -= fatigue_penalty

    # ── 刺激-特质共振(条件放大/加速消退) ──
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

    # 最终 clamp
    idcy = soft_clamp(idcy, 0.70, 1.05)
    rdcy = soft_clamp(rdcy, 0.95, 1.005)

    return idcy, rdcy


def apply_decay(state: np.ndarray, decay: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """③ 衰减/增强:向基线回归或背离。

    state[t] = baseline + (state[t-1] - baseline) × decay
    """
    return soft_clamp(baseline + (state - baseline) * decay, 0.0, 1.0)


# ============================================================
# ④ Surface Projection —— 内部状态 + 外表情刺激 → 表面表达
# ============================================================
#
# SurfaceState 不存储,每轮从内部状态 + outer_stimuli 动态投影。
# 表面表达 = 内部状态基线 + 外表情刺激影响 + 特质修饰
#
# 关键:外表情刺激是被压抑后的版本(由 apply_gates 输出)
#   - validation 被压抑 → 表面"温度"自然降低
#   - conflict 被压抑 → 表面"尖锐度"自然降低

def project_surface(
    internal: np.ndarray,
    relationship: np.ndarray,
    traits: np.ndarray,
    outer_stimuli: np.ndarray,
) -> np.ndarray:
    """④ 表面投影:内部状态 + outer_stimuli → 表面表达(动态计算)。"""
    s = np.zeros(S_SIZE, dtype=np.float64)

    # 内部状态基线
    s[S_EXPRESSIVENESS] = 0.3 + internal[I_ENERGY] * 0.4 - internal[I_MENTAL_FATIGUE] * 0.3
    s[S_WARMTH]         = 0.3 + relationship[R_AFFECTION] * 0.4 - internal[I_STRESS] * 0.2
    s[S_SHARPNESS]      = 0.1 + internal[I_IRRITATION] * 0.5 + internal[I_STRESS] * 0.2
    s[S_SOFTNESS]       = 0.2 + (1.0 - internal[I_STRESS]) * 0.3 + relationship[R_EMOTIONAL_SAFETY] * 0.2
    s[S_ENTHUSIASM]     = 0.3 + internal[I_ENERGY] * 0.5 - internal[I_MENTAL_FATIGUE] * 0.3
    s[S_RESTRAINT]      = 0.2 + internal[I_INSECURITY] * 0.3 + traits[T_PRIDE] * 0.2
    s[S_VULNERABILITY]  = 0.1 + internal[I_LONELINESS] * 0.3 + internal[I_LONGING] * 0.2 - traits[T_PRIDE] * 0.2

    # 外表情刺激的直接影响(被压抑后的版本)
    s[S_WARMTH]         += outer_stimuli[ST_VALIDATION]  * 0.30
    s[S_SHARPNESS]      += outer_stimuli[ST_CONFLICT]    * 0.25
    s[S_SOFTNESS]       += outer_stimuli[ST_CLOSENESS]   * 0.20
    s[S_VULNERABILITY]  += outer_stimuli[ST_ABANDONMENT] * 0.15
    s[S_RESTRAINT]      += outer_stimuli[ST_EMOTIONAL_WEIGHT] * 0.20
    s[S_SHARPNESS]      += outer_stimuli[ST_TEASING]     * 0.10
    s[S_WARMTH]         += outer_stimuli[ST_DEPENDENCY]  * 0.10

    # 特质修饰
    if traits[T_PRIDE] > 0.6:
        s[S_SHARPNESS]     += traits[T_PRIDE] * 0.10
        s[S_VULNERABILITY] -= traits[T_PRIDE] * 0.15
    if traits[T_EMOTIONAL_OPENNESS] > 0.6:
        s[S_EXPRESSIVENESS] += traits[T_EMOTIONAL_OPENNESS] * 0.10
        s[S_RESTRAINT]      -= traits[T_EMOTIONAL_OPENNESS] * 0.10
    if traits[T_OPTIMISM] > 0.6:
        s[S_ENTHUSIASM]    += traits[T_OPTIMISM] * 0.10

    return soft_clamp(s, 0.0, 1.0)


# ============================================================
# ⑤ Relationship Dynamics —— 关系动力系统(LTI)
# ============================================================
#
# 关系状态保持 LTI 不 LSTM 化——因为信任/好感建立需要几十轮,是超慢变量
# rel_t = A_rel · rel_{t-1} + B_rel · e_t (消费 inner_stimuli)

def _build_rel_state_coupling() -> np.ndarray:
    """关系状态耦合矩阵 A_rel(R_SIZE × R_SIZE)。"""
    A = np.zeros((R_SIZE, R_SIZE), dtype=np.float64)

    A[R_TRUST, R_AFFECTION] = 0.08
    A[R_FAMILIARITY, R_AFFECTION] = 0.05
    A[R_EMOTIONAL_SAFETY, R_TRUST] = 0.10
    A[R_DEPENDENCY, R_TRUST] = 0.05
    A[R_EMOTIONAL_SAFETY, R_FAMILIARITY] = 0.08
    A[R_AFFECTION, R_EMOTIONAL_SAFETY] = 0.05
    A[R_TRUST, R_EMOTIONAL_SAFETY] = 0.05
    A[R_AFFECTION, R_ROMANTIC_TENSION] = 0.03
    A[R_ROMANTIC_TENSION, R_DEPENDENCY] = 0.05

    np.fill_diagonal(A, 0.90)
    return A


def _build_rel_input_influence() -> np.ndarray:
    """关系输入影响矩阵 B_rel(ST_SIZE × R_SIZE)。"""
    B = np.zeros((ST_SIZE, R_SIZE), dtype=np.float64)

    # abandoned → trust↓, safety↓, tension↑, dependency↑
    B[ST_ABANDONMENT, R_TRUST] = -0.08
    B[ST_ABANDONMENT, R_EMOTIONAL_SAFETY] = -0.10
    B[ST_ABANDONMENT, R_ROMANTIC_TENSION] = 0.06
    B[ST_ABANDONMENT, R_DEPENDENCY] = 0.08
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
) -> np.ndarray:
    """⑤ 关系动力系统:LTI 风格(不 LSTM 化,关系是超慢变量)。"""
    coupling = REL_STATE_COUPLING_A @ current
    influence = gated_stimuli @ REL_INPUT_INFLUENCE_B
    new_state = coupling + influence
    return soft_clamp(new_state, 0.0, 1.0)


# ============================================================
# 主入口:Pipeline 编排
# ============================================================

def initialize_all(traits: np.ndarray) -> dict:
    """首次运行:用 Traits 初始化所有状态层,outer_stimuli 用 0 向量。"""
    internal = DEFAULT_INTERNAL.copy()
    relationship = DEFAULT_RELATIONSHIP.copy()
    outer_zero = np.zeros(ST_SIZE, dtype=np.float64)
    surface = project_surface(internal, relationship, traits, outer_zero)

    return {
        "internal_state": internal,
        "relationship_state": relationship,
        "surface_state": surface,
    }


def update_all(
    current_internal: Optional[np.ndarray],
    current_relationship: Optional[np.ndarray],
    traits: np.ndarray,
    stimuli: np.ndarray,
) -> dict:
    """State Engine 主入口:4 步 Pipeline。

    步骤:
      ① 三向门控 → (inner_stimuli, outer_stimuli)
      ② 内部动力系统(LSTM 式 3 门控) + 衰减
      ③ 关系动力系统(LTI) + 衰减
      ④ 表面投影

    返回:
      {
        "internal_state":     np.ndarray,  # 8 维
        "relationship_state":  np.ndarray,  # 6 维
        "surface_state":       np.ndarray,  # 7 维
      }
    """
    if current_internal is None:
        return initialize_all(traits)

    # ① 门控
    gates = compute_gates(traits, current_relationship, current_internal)
    inner_stimuli, outer_stimuli = apply_gates(
        stimuli, gates, traits, current_relationship,
    )

    # ② 内部动力系统 + 衰减
    new_internal = update_internal_dynamics(
        current_internal, inner_stimuli, traits, current_relationship, gates,
    )
    internal_decay, rel_decay = compute_dynamic_decay(
        traits, current_relationship, current_internal, inner_stimuli,
    )
    new_internal = apply_decay(new_internal, internal_decay, DEFAULT_INTERNAL)

    # ③ 关系动力系统 + 衰减
    new_relationship = update_relationship_dynamics(current_relationship, inner_stimuli)
    new_relationship = apply_decay(new_relationship, rel_decay, DEFAULT_RELATIONSHIP)

    # ④ 表面投影
    surface = project_surface(new_internal, new_relationship, traits, outer_stimuli)

    return {
        "internal_state": new_internal,
        "relationship_state": new_relationship,
        "surface_state": surface,
    }
