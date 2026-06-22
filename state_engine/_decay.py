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
from state import I_SIZE, R_SIZE
from ._dynamics import compute_setpoint, compute_rel_setpoint
from ._utils import soft_clamp
from ._dynamics_weights import (
    DECAY_INTERNAL_LAMBDA, DECAY_RELATIONSHIP_LAMBDA,
    DECAY_INTERNAL_TIME_CURVE_K, DECAY_REL_TIME_CURVE_K,
    DECAY_NEGATIVE_BOOST,
    INT_PERSONALITY_MOD, REL_PERSONALITY_MOD,
)


# ═══════════════════════════════════════════════════════════════
# 衰减配置
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DecayConfig:
    """时间衰减配置，参数来自 _dynamics_weights.py（带 provenance）。"""

    # ── 基础衰减率 λ_base (/小时) ──
    # 来自 DECAY_INTERNAL_LAMBDA（WeightVector 带 provenance）
    internal_lambda: np.ndarray = field(default_factory=lambda: DECAY_INTERNAL_LAMBDA)

    # 来自 DECAY_RELATIONSHIP_LAMBDA（WeightVector 带 provenance）
    relationship_lambda: np.ndarray = field(default_factory=lambda: DECAY_RELATIONSHIP_LAMBDA)

    # ── 时间曲线参数 ──
    # 来自 TIME_CURVE_K_INTERNAL（WeightVector 带 provenance）
    internal_time_curve_k: float = DECAY_INTERNAL_TIME_CURVE_K
    # 来自 TIME_CURVE_K_RELATIONSHIP（WeightVector 带 provenance）
    relationship_time_curve_k: float = DECAY_REL_TIME_CURVE_K

    # ── 非对称衰减 ──
    # 来自 NEGATIVE_DECAY_BOOST（WeightVector 带 provenance）
    negative_decay_boost: float = DECAY_NEGATIVE_BOOST

    # ── 最小时间间隔 (小时) ──
    min_delta_hours: float = 0.01  # ~36 秒


# 默认配置单例（各参数从 WeightVector 获取，带 provenance）
DEFAULT_DECAY_CONFIG = DecayConfig()


# ═══════════════════════════════════════════════════════════════
# 人格调制因子
# ═══════════════════════════════════════════════════════════════

def _compute_internal_personality_mod(traits: np.ndarray) -> float:
    """内部状态的人格调制因子（通过 INT_PERSONALITY_MOD 计算，带 provenance）。

    返回值缩放 λ_base:
      > 1.0 → 衰减更快 (情绪稳定的人恢复快)
      < 1.0 → 衰减更慢 (高焦虑的人放不下)
    """
    return soft_clamp(INT_PERSONALITY_MOD.compute(traits)[0], 0.3, 2.0)


def _compute_relationship_personality_mod(traits: np.ndarray) -> float:
    """关系状态的人格调制因子（通过 REL_PERSONALITY_MOD 计算，带 provenance）。"""
    return soft_clamp(REL_PERSONALITY_MOD.compute(traits)[0], 0.3, 2.0)


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
    decay_modulator: Optional[np.ndarray] = None,
) -> np.ndarray:
    """对内部状态应用时间衰减。

    Args:
        current: 当前内部状态 (8,)
        setpoint: 人格决定的稳态基线 (8,)
        traits: 人格特质 (10,)
        delta_hours: 自上次更新以来的实际时间 (小时)
        config: 衰减参数配置
        decay_modulator: 刺激衰减调制因子 (7,) [0,1]，约束②参数。
                         值<1 减慢衰减（情绪冲击残留），>1 加速恢复。
                         未提供时无影响。

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

    # 约束②：decay_modulator → 缩放有效衰减率
    # 将 (7,) 刺激级调制映射到 (8,) 状态级：取均值
    if decay_modulator is not None:
        mod = float(np.clip(np.mean(decay_modulator), 0.1, 2.0))
        lam = lam * mod

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
    decay_modulator: Optional[np.ndarray] = None,
) -> np.ndarray:
    """对关系状态应用时间衰减。

    Args:
        current: 当前关系状态 (3,)
        setpoint: 人格决定的关系稳态基线 (3,)
        traits: 人格特质 (10,)
        delta_hours: 自上次更新以来的实际时间 (小时)
        config: 衰减参数配置
        decay_modulator: 刺激衰减调制因子 (7,) [0,1]，约束②参数。

    Returns:
        衰减后的关系状态 (3,)
    """
    if delta_hours < config.min_delta_hours:
        return current

    p_mod = _compute_relationship_personality_mod(traits)
    lam = _compute_lambda_effective(
        config.relationship_lambda, p_mod, delta_hours,
        config.relationship_time_curve_k,
    )

    # 约束②：decay_modulator → 缩放有效衰减率
    if decay_modulator is not None:
        mod = float(np.clip(np.mean(decay_modulator), 0.1, 2.0))
        lam = lam * mod

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
    decay_modulator: Optional[np.ndarray] = None,
) -> dict:
    """对内部和关系状态同时应用时间衰减（便捷接口）。

    从 _dynamics 导入 compute_setpoint / compute_rel_setpoint，
    消除 setpoint 重复代码（之前版本有重复实现）。

    Args:
        current_internal: 当前内部状态 (8,)
        current_relationship: 当前关系状态 (3,)
        traits: 人格特质 (10,)
        delta_hours: 自上次更新以来的实际时间 (小时)
        config: 衰减参数配置
        decay_modulator: 刺激衰减调制因子 (7,) [0,1]，约束②参数。

    Returns:
        {"internal_state": (8,), "relationship_state": (3,)}
    """
    internal_sp = compute_setpoint(traits)
    rel_sp = compute_rel_setpoint(traits)

    return {
        "internal_state": apply_time_decay_internal(
            current_internal, internal_sp, traits, delta_hours, config,
            decay_modulator=decay_modulator,
        ),
        "relationship_state": apply_time_decay_relationship(
            current_relationship, rel_sp, traits, delta_hours, config,
            decay_modulator=decay_modulator,
        ),
    }
