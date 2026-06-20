"""Time-Aware Decay —— 时间感知状态衰减（稳态恢复的唯一通道）。

基于情感动力学的时间衰减组件，以真实时间戳驱动状态向基线回归。
与残差动力学（_dynamics.py）的分工:
  - _dynamics:  每轮对话内的刺激+耦合驱动（不向 setpoint 拉）
  - _decay:     对话间隔中的时间衰减（向 setpoint 拉）

学术基础:
  - 指数衰减: Rutledge et al. (2014), Vanhasbroeck et al. (2024)
  - 人格调制衰减率: Schuyler et al. (2014), Lücke et al. (2024)
  - 多时间尺度: DER 模型 (Tanguy et al., 2007)
  - 关系衰减: Bhattacharya et al. (2017), Pellegrini (1977)

核心公式:
  decayed[s] = baseline[s] + (current[s] - baseline[s]) × exp(-λ_eff[s] × Δt)

其中 λ_eff 由维度基础衰减率、人格调制、时间曲线三者共同决定。
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from state import (
    # 内部状态索引
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE, I_SIZE,
    # 关系状态索引
    R_AFFECTION, R_TRUST, R_FAMILIARITY, R_DEPENDENCY,
    R_EMOTIONAL_SAFETY, R_ROMANTIC_TENSION, R_SIZE,
    # 特质索引
    T_EMOTIONAL_STABILITY, T_OPTIMISM, T_ANXIETY_PRONENESS,
    T_ANGER_REACTIVITY, T_EMOTIONAL_OPENNESS,
    T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE,
)
from ._dynamics import compute_setpoint, compute_rel_setpoint
from ._utils import soft_clamp


# ═══════════════════════════════════════════════════════════════
# 衰减配置
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DecayConfig:
    """时间衰减配置，可外部化为 JSON/YAML。"""

    # ── 基础衰减率 λ_base (/小时) ──
    # 内部状态: 小时级
    internal_lambda: np.ndarray = field(default_factory=lambda: np.array([
        0.35,  # I_ENERGY         — 精力, 半衰期 ~2h
        0.23,  # I_STRESS         — 压力, ~3h
        0.17,  # I_LONELINESS     — 孤独, ~4h
        0.14,  # I_INSECURITY     — 不安全, ~5h
        0.69,  # I_IRRITATION     — 烦躁, ~1h (最快)
        0.12,  # I_LONGING        — 思念, ~6h (最慢)
        0.35,  # I_SOCIAL_BATTERY — 社交电量, ~2h
        0.23,  # I_MENTAL_FATIGUE — 精神疲劳, ~3h
    ], dtype=np.float64))

    # 关系状态: 天级 (λ 很小)
    relationship_lambda: np.ndarray = field(default_factory=lambda: np.array([
        0.0021,  # R_AFFECTION        — 好感, 半衰期 ~14d
        0.0014,  # R_TRUST            — 信任, ~21d
        0.0041,  # R_FAMILIARITY      — 熟悉, ~7d
        0.0029,  # R_DEPENDENCY       — 依赖, ~10d
        0.0021,  # R_EMOTIONAL_SAFETY — 情感安全, ~14d
        0.0058,  # R_ROMANTIC_TENSION — 浪漫张力, ~5d
    ], dtype=np.float64))

    # ── 时间曲线参数 ──
    # k: 衰减速率随 Δt 放缓的强度
    # λ_eff = λ_base × personality_mod / (1 + k × Δt)
    internal_time_curve_k: float = 0.05   # 内部: 弱放缓
    relationship_time_curve_k: float = 0.001  # 关系: 几乎不额外放缓

    # ── 非对称衰减 ──
    # 关系状态中，负向偏离（current[s] < setpoint[s]）的 λ 倍率。
    # > 1.0 → 负面印象消退快于正面（Fading Affect Bias），关系韧性。
    # 仅用于关系状态（内部状态不区分方向）。
    negative_decay_boost: float = 1.8

    # ── 最小时间间隔 (小时) ──
    # 低于此值的间隔不触发衰减，避免每轮微小计算
    min_delta_hours: float = 0.01  # ~36 秒


# 默认配置单例
DEFAULT_DECAY_CONFIG = DecayConfig()


# ═══════════════════════════════════════════════════════════════
# 人格调制因子
# ═══════════════════════════════════════════════════════════════

def _compute_internal_personality_mod(traits: np.ndarray) -> float:
    """内部状态的人格调制因子。

    返回值缩放 λ_base:
      > 1.0 → 衰减更快 (情绪稳定的人恢复快)
      < 1.0 → 衰减更慢 (高焦虑的人放不下)

    文献: Schuyler et al. (2014) — 杏仁核恢复速度预测神经质;
          Lücke et al. (2024) — 高神经质 → 更慢的压力恢复
    """
    mod = 1.0
    mod += traits[T_EMOTIONAL_STABILITY] * 0.15   # 稳定→恢复快
    mod += traits[T_OPTIMISM]            * 0.075  # 乐观→恢复快
    mod -= traits[T_ANXIETY_PRONENESS]   * 0.125  # 焦虑→恢复慢
    mod -= traits[T_ANGER_REACTIVITY]    * 0.05   # 易怒→恢复慢
    mod += traits[T_EMOTIONAL_OPENNESS]  * 0.05   # 开放→恢复快

    return soft_clamp(mod, 0.3, 2.0)  # 最快 2×, 最慢 0.3×


def _compute_relationship_personality_mod(traits: np.ndarray) -> float:
    """关系状态的人格调制因子。

    文献: Bhattacharya et al. (2017) — 回避型更容易疏远;
          Pellegrini (1977) — 依恋焦虑→放不下
    """
    mod = 1.0
    mod += traits[T_ATTACHMENT_AVOIDANCE] * 0.175  # 回避→疏远快
    mod -= traits[T_ATTACHMENT_ANXIETY]   * 0.10   # 焦虑→放不下
    mod -= traits[T_EMOTIONAL_STABILITY]  * 0.05   # 稳定→关系稳定

    return soft_clamp(mod, 0.3, 2.0)


# ═══════════════════════════════════════════════════════════════
# 有效衰减率计算
# ═══════════════════════════════════════════════════════════════

def _compute_lambda_effective(
    lambda_base: np.ndarray,
    personality_mod: float,
    delta_hours: float,
    time_curve_k: float,
) -> np.ndarray:
    """计算有效衰减率。

    λ_eff[s] = λ_base[s] × personality_mod / (1 + time_curve_k × Δt)

    时间曲线 1/(1+k·Δt):
      - Δt → 0:    λ_eff ≈ λ_base × personality_mod  (全速衰减)
      - Δt → ∞:    λ_eff → 0  (衰减速率趋零，模拟幂律尾)

    文献: Hong & Zhang (2025) — 幂律衰减的长尾效应;
          Vanhasbroeck et al. (2024) — 拟双曲衰减
    """
    time_damping = 1.0 / (1.0 + time_curve_k * delta_hours)
    return lambda_base * personality_mod * time_damping


# ═══════════════════════════════════════════════════════════════
# 主 API
# ═══════════════════════════════════════════════════════════════

def apply_time_decay_internal(
    current: np.ndarray,
    setpoint: np.ndarray,
    traits: np.ndarray,
    delta_hours: float,
    config: DecayConfig = DEFAULT_DECAY_CONFIG,
) -> np.ndarray:
    """对内部状态应用时间衰减。

    Args:
        current: 当前内部状态 (8,)
        setpoint: 人格决定的稳态基线 (8,)
        traits: 人格特质 (10,)
        delta_hours: 自上次更新以来的实际时间 (小时)
        config: 衰减参数配置

    Returns:
        衰减后的内部状态 (8,)
    """
    if delta_hours < config.min_delta_hours:
        return current

    p_mod = _compute_internal_personality_mod(traits)
    lam = _compute_lambda_effective(
        config.internal_lambda, p_mod, delta_hours,
        config.internal_time_curve_k,
    )

    # 核心衰减公式
    deviation = current - setpoint
    decay_factor = np.exp(-lam * delta_hours)
    decayed = setpoint + deviation * decay_factor

    return soft_clamp(decayed, -1.0, 1.0)


def apply_time_decay_relationship(
    current: np.ndarray,
    setpoint: np.ndarray,
    traits: np.ndarray,
    delta_hours: float,
    config: DecayConfig = DEFAULT_DECAY_CONFIG,
) -> np.ndarray:
    """对关系状态应用时间衰减。

    Args:
        current: 当前关系状态 (6,)
        setpoint: 人格决定的关系稳态基线 (6,)
        traits: 人格特质 (10,)
        delta_hours: 自上次更新以来的实际时间 (小时)
        config: 衰减参数配置

    Returns:
        衰减后的关系状态 (6,)
    """
    if delta_hours < config.min_delta_hours:
        return current

    p_mod = _compute_relationship_personality_mod(traits)
    lam = _compute_lambda_effective(
        config.relationship_lambda, p_mod, delta_hours,
        config.relationship_time_curve_k,
    )

    deviation = current - setpoint

    # 非对称衰减：负向偏离（current < setpoint，即负面印象）加速恢复
    negative_mask = deviation < 0
    lam[negative_mask] *= config.negative_decay_boost

    decay_factor = np.exp(-lam * delta_hours)
    decayed = setpoint + deviation * decay_factor

    return soft_clamp(decayed, -1.0, 1.0)


def apply_time_decay(
    current_internal: np.ndarray,
    current_relationship: np.ndarray,
    traits: np.ndarray,
    delta_hours: float,
    config: DecayConfig = DEFAULT_DECAY_CONFIG,
) -> dict:
    """对内部和关系状态同时应用时间衰减（便捷接口）。

    从 _dynamics 导入 compute_setpoint / compute_rel_setpoint，
    消除 setpoint 重复代码（之前版本有重复实现）。

    Args:
        current_internal: 当前内部状态 (8,)
        current_relationship: 当前关系状态 (6,)
        traits: 人格特质 (10,)
        delta_hours: 自上次更新以来的实际时间 (小时)
        config: 衰减参数配置

    Returns:
        {"internal_state": (8,), "relationship_state": (6,)}
    """
    internal_sp = compute_setpoint(traits)
    rel_sp = compute_rel_setpoint(traits)

    return {
        "internal_state": apply_time_decay_internal(
            current_internal, internal_sp, traits, delta_hours, config,
        ),
        "relationship_state": apply_time_decay_relationship(
            current_relationship, rel_sp, traits, delta_hours, config,
        ),
    }
