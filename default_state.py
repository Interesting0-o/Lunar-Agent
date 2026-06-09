"""
default_state —— 人格默认初始值

所有状态层的默认基线值集中于此。
这些值定义了角色在没有任何既往状态时的心理基线。
导入方式：
  from default_state import DEFAULT_TRAITS, DEFAULT_INTERNAL, ...
"""

import numpy as np
from state import (
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE,
    R_AFFECTION, R_TRUST, R_FAMILIARITY, R_DEPENDENCY,
    R_EMOTIONAL_SAFETY, R_ROMANTIC_TENSION,
    T_SENSITIVITY, T_PRIDE, T_EMOTIONAL_OPENNESS, T_EMOTIONAL_STABILITY,
    T_OPTIMISM, T_ANXIETY_PRONENESS, T_ANGER_REACTIVITY, T_JEALOUSY_SENSITIVITY,
    T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE,
)

# ── 人格特质（10 维） ──
# 长期稳定的性格参数，目前以「月下誓约」人设为基准
DEFAULT_TRAITS: np.ndarray = np.array([
    0.7,    # T_SENSITIVITY — 敏感度（偏高，容易感知情绪变化）
    0.65,   # T_PRIDE — 自尊心（偏强，口是心非的资本）
    0.6,    # T_EMOTIONAL_OPENNESS — 情绪开放性
    0.5,    # T_EMOTIONAL_STABILITY — 情绪稳定性
    0.55,   # T_OPTIMISM — 乐观倾向
    0.6,    # T_ANXIETY_PRONENESS — 焦虑倾向（偏高，害怕被抛弃）
    0.5,    # T_ANGER_REACTIVITY — 易怒倾向
    0.7,    # T_JEALOUSY_SENSITIVITY — 嫉妒敏感度（偏高，独占欲强）
    0.55,   # T_ATTACHMENT_ANXIETY — 依恋焦虑
    0.2,    # T_ATTACHMENT_AVOIDANCE — 依恋回避（低，渴望亲近）
], dtype=np.float64)

# ── 内部状态（8 维） ──
# 底层心理指标基线
DEFAULT_INTERNAL: np.ndarray = np.array([
    0.7,    # I_ENERGY — 精力充沛
    0.2,    # I_STRESS — 压力较低
    0.3,    # I_LONELINESS — 略有孤独感
    0.25,   # I_INSECURITY — 轻微不安
    0.1,    # I_IRRITATION — 平静
    0.4,    # I_LONGING — 有一定思念/渴望
    0.6,    # I_SOCIAL_BATTERY — 社交电量尚可
    0.15,   # I_MENTAL_FATIGUE — 精神清醒
], dtype=np.float64)

# ── 关系状态（6 维） ──
# 对用户的关系感知基线
DEFAULT_RELATIONSHIP: np.ndarray = np.array([
    0.3,    # R_AFFECTION — 初始好感偏低
    0.3,    # R_TRUST — 初始信任中立
    0.2,    # R_FAMILIARITY — 初始陌生
    0.15,   # R_DEPENDENCY — 初始独立
    0.25,   # R_EMOTIONAL_SAFETY — 情感安全感偏低
    0.2,    # R_ROMANTIC_TENSION — 浪漫张力较低
], dtype=np.float64)
