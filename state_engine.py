"""
state_engine —— 核心状态更新引擎

职责：
  接收 perception_node 输出的社交信号（SocialSignals + InteractionImpact），
  结合角色的 Traits 和当前各状态层，更新所有内部状态并计算表面表达。

设计原则：
  - 纯函数：给定相同输入，始终返回相同输出
  - 可组合：每个子函数职责单一，可独立测试
  - 特质介导：相同的社交信号因 Traits 不同导致不同的状态变化
  - 压抑机制：内部状态 ≠ 表面表达，隐藏层作为缓冲
"""

from state import (
    InternalState, RelationshipState, HiddenState, SurfaceState,
    Traits, SocialSignals, InteractionImpact,
)
from typing import Optional

# ═══════════════════════════════════════════════════════════
# 1. 默认初始值
# ═══════════════════════════════════════════════════════════

DEFAULT_TRAITS: Traits = {
    "sensitivity": 0.7,
    "pride": 0.65,
    "emotional_openness": 0.6,
    "emotional_stability": 0.5,
    "optimism": 0.55,
    "anxiety_proneness": 0.6,
    "anger_reactivity": 0.5,
    "jealousy_sensitivity": 0.7,
    "attachment_anxiety": 0.55,
    "attachment_avoidance": 0.2,
}

DEFAULT_INTERNAL: InternalState = {
    "energy": 0.7,
    "stress": 0.2,
    "loneliness": 0.3,
    "insecurity": 0.25,
    "irritation": 0.1,
    "longing": 0.4,
    "social_battery": 0.6,
    "mental_fatigue": 0.15,
}

DEFAULT_RELATIONSHIP: RelationshipState = {
    "affection": 0.3,
    "trust": 0.3,
    "familiarity": 0.2,
    "dependency": 0.15,
    "emotional_safety": 0.25,
    "romantic_tension": 0.2,
}

DEFAULT_HIDDEN: HiddenState = {
    "suppressed_sadness": 0.0,
    "suppressed_anger": 0.0,
    "hidden_affection": 0.05,
}

DEFAULT_SURFACE: SurfaceState = {
    "expressiveness": 0.5,
    "warmth": 0.5,
    "sharpness": 0.2,
    "softness": 0.4,
    "enthusiasm": 0.4,
    "restraint": 0.4,
    "vulnerability": 0.2,
}

# ═══════════════════════════════════════════════════════════
# 2. 权重规则表
# ═══════════════════════════════════════════════════════════
#
# 每个社交信号对各个状态维度的基础影响权重。
# 正值 = 增加，负值 = 减少。
# 最终影响 = 信号强度 × 权重，再经特质修饰器调整。

# ── 内部状态更新规则 ──
INTERNAL_UPDATE_RULES = {
    "affection_signal": {
        "energy": 0.15,
        "loneliness": -0.25,
        "insecurity": -0.20,
        "longing": -0.10,
    },
    "attention_signal": {
        "social_battery": -0.10,
        "loneliness": -0.15,
    },
    "intimacy_signal": {
        "loneliness": -0.30,
        "insecurity": -0.15,
        "energy": 0.10,
    },
    "approval_signal": {
        "energy": 0.10,
        "insecurity": -0.10,
    },
    "rejection_signal": {
        "insecurity": 0.40,
        "stress": 0.30,
        "irritation": 0.20,
        "loneliness": 0.25,
        "mental_fatigue": 0.20,
    },
    "abandonment_signal": {
        "insecurity": 0.35,
        "longing": 0.25,
        "energy": -0.15,
        "stress": 0.20,
    },
    "dependency_signal": {
        "energy": 0.05,
        "loneliness": -0.15,
    },
    "teasing_signal": {
        "social_battery": -0.05,
        "irritation": 0.03,
        "energy": 0.03,
    },
    "conflict_signal": {
        "stress": 0.40,
        "irritation": 0.35,
        "energy": -0.20,
        "mental_fatigue": 0.30,
        "social_battery": -0.30,
    },
}

# ── 关系状态更新规则 ──
RELATIONSHIP_UPDATE_RULES = {
    "affection_signal": {
        "affection": 0.15,
        "trust": 0.08,
        "familiarity": 0.05,
        "romantic_tension": 0.05,
    },
    "attention_signal": {
        "familiarity": 0.05,
    },
    "intimacy_signal": {
        "affection": 0.10,
        "familiarity": 0.15,
        "emotional_safety": 0.10,
        "romantic_tension": 0.08,
    },
    "approval_signal": {
        "affection": 0.08,
        "trust": 0.10,
    },
    "rejection_signal": {
        "trust": -0.15,
        "emotional_safety": -0.20,
        "affection": -0.08,
    },
    "abandonment_signal": {
        "trust": -0.10,
        "dependency": 0.15,
        "romantic_tension": 0.10,
    },
    "dependency_signal": {
        "dependency": 0.20,
        "familiarity": 0.08,
        "emotional_safety": 0.05,
    },
    "teasing_signal": {
        "familiarity": 0.08,
        "romantic_tension": 0.10,
    },
    "conflict_signal": {
        "trust": -0.20,
        "emotional_safety": -0.25,
        "affection": -0.10,
        "romantic_tension": -0.10,
    },
}

# ── 隐藏状态更新规则 ──
# 高自尊/高克制角色会把部分情感压入隐藏层而非直接表达
HIDDEN_UPDATE_RULES = {
    "affection_signal": {"hidden_affection": 0.08},
    "rejection_signal": {"suppressed_sadness": 0.15, "suppressed_anger": 0.10},
    "abandonment_signal": {"suppressed_sadness": 0.20},
    "conflict_signal": {"suppressed_anger": 0.25},
}

# ── 特质修饰器 ──
# 特定特质会放大或抑制某些信号对状态的影响
# 值 = 该特质每高于 0.5 一单位时对信号影响的额外乘数
# 例: attachment_anxiety=0.7, 高于0.5的0.2 → abandonment_signal 影响 +0.2×0.5=+10%
TRAIT_AMPLIFIERS = {
    "attachment_anxiety": {
        "abandonment_signal": 0.5,
        "rejection_signal": 0.3,
    },
    "jealousy_sensitivity": {
        "abandonment_signal": 0.4,
        "teasing_signal": 0.2,
    },
    "anger_reactivity": {
        "conflict_signal": 0.5,
        "rejection_signal": 0.2,
    },
    "pride": {
        "affection_signal": -0.2,     # 高自尊 → 不愿承认被好感影响
        "teasing_signal": 0.3,        # 高自尊 → 被逗弄更易不爽
    },
    "emotional_stability": {
        "conflict_signal": -0.3,      # 高稳定 → 冲突影响减弱
        "rejection_signal": -0.2,
    },
}

# ═══════════════════════════════════════════════════════════
# 3. 工具函数
# ═══════════════════════════════════════════════════════════

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """将值约束在 [lo, hi] 范围内。"""
    return max(lo, min(hi, value))


def _apply_signal_rules(
    current: dict,
    signals: dict,
    rules: dict,
    traits: dict,
    trait_amps: dict,
) -> dict:
    """通用规则引擎：对每个信号，查表 + 特质修饰 → 更新各维度。"""
    result = dict(current)

    for signal_key, dim_rules in rules.items():
        strength = signals.get(signal_key, 0.0)
        if strength == 0.0:
            continue

        # 基础影响
        for dim, weight in dim_rules.items():
            old_val = result.get(dim, 0.5)
            result[dim] = clamp(old_val + strength * weight)

        # 特质二次修饰
        for trait_name, affected_signals in trait_amps.items():
            if signal_key in affected_signals:
                trait_deviation = traits.get(trait_name, 0.5) - 0.5
                amp_factor = affected_signals[signal_key]
                extra = strength * trait_deviation * amp_factor
                if extra != 0.0:
                    for dim, weight in dim_rules.items():
                        result[dim] = clamp(result.get(dim, 0.5) + extra * weight)

    return result


# ═══════════════════════════════════════════════════════════
# 4. 核心更新函数
# ═══════════════════════════════════════════════════════════

def _update_internal(
    current: InternalState,
    signals: SocialSignals,
    impact: InteractionImpact,
    traits: Traits,
) -> InternalState:
    """根据社交信号更新内部心理状态。"""
    new_state = _apply_signal_rules(current, signals, INTERNAL_UPDATE_RULES, traits, TRAIT_AMPLIFIERS)

    # interaction_impact 放大效应：情绪越重的交互，状态变化幅度越大
    weight_scale = 1.0 + impact["emotional_weight"] * 0.5
    if weight_scale > 1.0:
        for k in new_state:
            base = current.get(k, 0.5)
            new_state[k] = clamp(base + (new_state[k] - base) * weight_scale)

    return new_state


def _update_relationship(
    current: RelationshipState,
    signals: SocialSignals,
    impact: InteractionImpact,
    traits: Traits,
) -> RelationshipState:
    """根据社交信号更新关系状态。"""
    new_state = _apply_signal_rules(current, signals, RELATIONSHIP_UPDATE_RULES, traits, TRAIT_AMPLIFIERS)

    # closeness_impact 直接影响亲密度和安全感
    ci = impact["closeness_impact"]
    if ci > 0:
        new_state["familiarity"] = clamp(new_state["familiarity"] + ci * 0.15)
        new_state["emotional_safety"] = clamp(new_state["emotional_safety"] + ci * 0.12)
    elif ci < 0:
        new_state["emotional_safety"] = clamp(new_state["emotional_safety"] + ci * 0.15)
        new_state["trust"] = clamp(new_state["trust"] + ci * 0.10)

    # trust_impact 直接影响信任
    ti = impact["trust_impact"]
    if ti != 0:
        new_state["trust"] = clamp(new_state["trust"] + ti * 0.15)

    return new_state


def _update_hidden(
    current: HiddenState,
    signals: SocialSignals,
    traits: Traits,
) -> HiddenState:
    """根据社交信号更新隐藏（压抑）状态。"""
    new_state = _apply_signal_rules(current, signals, HIDDEN_UPDATE_RULES, traits, {})

    # 自然衰减：隐藏情感会随时间缓慢消解
    for k in new_state:
        new_state[k] = clamp(new_state[k] - 0.01)

    # 特质修饰：高自尊者压抑更多情感
    if traits["pride"] > 0.6:
        pride_extra = (traits["pride"] - 0.6) * 2  # 0~0.8
        # 收到好感时，高自尊会把更多压抑到隐藏层
        a = signals.get("affection_signal", 0.0)
        new_state["hidden_affection"] = clamp(new_state["hidden_affection"] + a * pride_extra * 0.08)

    return new_state


def _compute_surface(
    internal: InternalState,
    relationship: RelationshipState,
    hidden: HiddenState,
    traits: Traits,
) -> SurfaceState:
    """根据内部状态、关系状态、隐藏状态和特质，计算表面表达状态。

    这是"口是心非"机制的核心——表面 ≠ 内心。
    """
    s: SurfaceState = dict(DEFAULT_SURFACE)  # type: ignore

    # ── 内部状态 → 表面基线 ──
    s["expressiveness"] = clamp(0.3 + internal["energy"] * 0.4 - internal["mental_fatigue"] * 0.3)
    s["warmth"] = clamp(0.3 + relationship["affection"] * 0.4 - internal["stress"] * 0.2)
    s["sharpness"] = clamp(0.1 + internal["irritation"] * 0.5 + internal["stress"] * 0.2)
    s["softness"] = clamp(0.2 + (1.0 - internal["stress"]) * 0.3 + relationship["emotional_safety"] * 0.2)
    s["enthusiasm"] = clamp(0.3 + internal["energy"] * 0.5 - internal["mental_fatigue"] * 0.3)
    s["restraint"] = clamp(0.2 + internal["insecurity"] * 0.3 + traits["pride"] * 0.2)
    s["vulnerability"] = clamp(0.1 + internal["loneliness"] * 0.3 + internal["longing"] * 0.2 - traits["pride"] * 0.2)

    # ── 隐藏状态压抑效应 ──
    if hidden["suppressed_sadness"] > 0.4:
        s["warmth"] = clamp(s["warmth"] - hidden["suppressed_sadness"] * 0.15)
        s["restraint"] = clamp(s["restraint"] + hidden["suppressed_sadness"] * 0.10)

    if hidden["suppressed_anger"] > 0.3:
        s["sharpness"] = clamp(s["sharpness"] + hidden["suppressed_anger"] * 0.20)
        s["warmth"] = clamp(s["warmth"] - hidden["suppressed_anger"] * 0.10)

    if hidden["hidden_affection"] > 0.5:
        # 隐藏好感 → 表面更克制（口是心非），但偶尔泄露脆弱
        s["vulnerability"] = clamp(s["vulnerability"] + hidden["hidden_affection"] * 0.10)
        s["restraint"] = clamp(s["restraint"] + hidden["hidden_affection"] * 0.15)

    # ── 特质修饰 ──
    if traits["pride"] > 0.6:
        s["sharpness"] = clamp(s["sharpness"] + traits["pride"] * 0.10)
        s["vulnerability"] = clamp(s["vulnerability"] - traits["pride"] * 0.15)

    if traits["emotional_openness"] > 0.6:
        s["expressiveness"] = clamp(s["expressiveness"] + traits["emotional_openness"] * 0.10)
        s["restraint"] = clamp(s["restraint"] - traits["emotional_openness"] * 0.10)

    if traits["optimism"] > 0.6:
        s["enthusiasm"] = clamp(s["enthusiasm"] + traits["optimism"] * 0.10)

    return s


def _check_breakthroughs(hidden: HiddenState, traits: Traits) -> list:
    """检查隐藏状态是否积累到突破阈值，返回触发的事件列表。"""
    events = []

    if hidden["hidden_affection"] > 0.85:
        events.append("AFFECTION_BREAKTHROUGH")
    if hidden["suppressed_sadness"] > 0.85:
        events.append("SADNESS_BREAKTHROUGH")
    if hidden["suppressed_anger"] > 0.80:
        events.append("ANGER_BREAKTHROUGH")

    # 复合条件：高依恋焦虑 + 压抑悲伤 + 隐藏好感 → "黏人"突破
    if (hidden["suppressed_sadness"] > 0.6
            and traits["attachment_anxiety"] > 0.6
            and hidden["hidden_affection"] > 0.5):
        events.append("CLINGY_BREAKTHROUGH")

    return events


# ═══════════════════════════════════════════════════════════
# 5. 主入口
# ═══════════════════════════════════════════════════════════

def initialize_all(traits: Traits) -> dict:
    """首次运行：用 Traits 初始化所有状态层为合理默认值。"""
    internal = dict(DEFAULT_INTERNAL)
    relationship = dict(DEFAULT_RELATIONSHIP)
    hidden = dict(DEFAULT_HIDDEN)
    surface = _compute_surface(internal, relationship, hidden, traits)

    return {
        "internal_state": internal,
        "relationship_state": relationship,
        "hidden_state": hidden,
        "surface_state": surface,
    }


def update_all(
    current_internal: Optional[InternalState],
    current_relationship: Optional[RelationshipState],
    current_hidden: Optional[HiddenState],
    traits: Traits,
    signals: SocialSignals,
    impact: InteractionImpact,
) -> dict:
    """State Engine 主入口。

    参数：
      current_*: 当前各状态层（首次为 None 则自动初始化）
      traits:    角色特质（稳定参数）
      signals:   perception_node 输出的社交信号
      impact:    perception_node 输出的互动影响指标

    返回：
      {
        "internal_state": InternalState,
        "relationship_state": RelationshipState,
        "hidden_state": HiddenState,
        "surface_state": SurfaceState,
        "triggered_events": list[str],
      }
    """
    # ── 首次运行初始化 ──
    if current_internal is None:
        return initialize_all(traits)

    # ── 1. 更新内部状态 ──
    new_internal = _update_internal(current_internal, signals, impact, traits)

    # ── 2. 更新关系状态 ──
    new_relationship = _update_relationship(current_relationship, signals, impact, traits)

    # ── 3. 更新隐藏状态 ──
    new_hidden = _update_hidden(current_hidden, signals, traits)

    # ── 4. 计算表面表达 ──
    new_surface = _compute_surface(new_internal, new_relationship, new_hidden, traits)

    # ── 5. 检查阈值突破 ──
    triggered_events = _check_breakthroughs(new_hidden, traits)

    if triggered_events:
        import logging
        logging.getLogger(__name__).info("State Engine 触发事件: %s", triggered_events)

    return {
        "internal_state": new_internal,
        "relationship_state": new_relationship,
        "hidden_state": new_hidden,
        "surface_state": new_surface,
        "triggered_events": triggered_events,
    }
