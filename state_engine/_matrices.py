"""状态引擎的矩阵常量与工厂函数。

所有耦合矩阵、影响矩阵、基线衰减率集中于此，便于后续外置化为 JSON/YAML 配置。

稳定性保证:
  - 所有耦合矩阵经过谱归一化，ρ(A) < 1.0
  - 运行时可通过 validate_matrices() 验证不变量
"""

import logging
import numpy as np
from state import (
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE, I_SIZE,
    R_AFFECTION, R_TRUST, R_FAMILIARITY, R_DEPENDENCY,
    R_EMOTIONAL_SAFETY, R_ROMANTIC_TENSION, R_SIZE,
    ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT,
    ST_DEPENDENCY, ST_TEASING, ST_EMOTIONAL_WEIGHT, ST_SIZE,
)

logger = logging.getLogger(__name__)

# 目标谱半径 — 所有耦合矩阵缩放到此值以下
_TARGET_SPECTRAL_RADIUS = 0.95


def _spectral_normalize(matrix: np.ndarray, name: str) -> np.ndarray:
    """谱归一化：确保矩阵谱半径 ≤ 目标值。

    若 ρ(matrix) > target，整体缩放 matrix *= target/ρ。
    保持矩阵内部相对权重不变，仅缩放整体影响强度。

    返回（可能缩放后的）矩阵。
    """
    eigenvalues = np.linalg.eigvals(matrix)
    rho = max(abs(ev) for ev in eigenvalues)

    if rho >= _TARGET_SPECTRAL_RADIUS:
        scale = _TARGET_SPECTRAL_RADIUS / rho
        logger.warning(
            "%s 谱半径 %.4f ≥ %.2f，应用谱归一化（缩放系数 %.4f）",
            name, rho, _TARGET_SPECTRAL_RADIUS, scale,
        )
        return matrix * scale

    logger.debug("%s 谱半径 %.4f < %.2f，跳过归一化", name, rho, _TARGET_SPECTRAL_RADIUS)
    return matrix


def validate_matrices() -> dict:
    """运行时验证所有耦合矩阵的稳定性不变量。

    返回 {"ok": bool, "results": [...]}
    可在 pipeline 每轮调用，也可仅在初始化时调用一次。
    """
    results = []
    all_ok = True

    for name, matrix in [
        ("STATE_COUPLING_A", STATE_COUPLING_A),
        ("REL_STATE_COUPLING_A", REL_STATE_COUPLING_A),
    ]:
        eigenvalues = np.linalg.eigvals(matrix)
        rho = max(abs(ev) for ev in eigenvalues)
        ok = rho < 1.0 and np.all(np.isfinite(matrix))
        if not ok:
            logger.error("%s 不稳定: ρ=%.4f", name, rho)
        results.append({"matrix": name, "spectral_radius": float(rho), "stable": bool(ok)})
        all_ok = all_ok and ok

    return {"ok": all_ok, "results": results}


# ═══════════════════════════════════════════════════════════════
# ② Internal Dynamics 矩阵
# ═══════════════════════════════════════════════════════════════


def _build_state_coupling() -> np.ndarray:
    """内部状态耦合矩阵 A（I_SIZE × I_SIZE）。

    A[i, j] = h_{t-1}[j] 对 h_t[i] 的影响。
    正值=正耦合，负值=负耦合，对角线=自保持（惯性）。
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

    np.fill_diagonal(A, 0.85)  # 自保持（惯性）
    return A


def _build_input_influence() -> np.ndarray:
    """输入影响矩阵 B（ST_SIZE × I_SIZE）：B[s, i] = 刺激 s 对状态 i 的影响权重。"""
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
    """人格偏置向量 c（I_SIZE）：每轮自然偏移，会被 A 矩阵和 clamp 约束。"""
    c = np.zeros(I_SIZE, dtype=np.float64)
    c[I_ENERGY] = 0.01       # 活跃角色自然恢复
    c[I_LONELINESS] = -0.005 # 孤独自然缓解
    c[I_IRRITATION] = -0.01  # 烦躁自然消退
    return c


STATE_COUPLING_A = _spectral_normalize(_build_state_coupling(), "STATE_COUPLING_A")
INPUT_INFLUENCE_B = _build_input_influence()
PERSONALITY_BIAS_C = _build_personality_bias()


# ═══════════════════════════════════════════════════════════════
# ③ Dynamic Decay 基础衰减率
# ═══════════════════════════════════════════════════════════════

_INTERNAL_BASE_DECAY = np.array([
    0.98, 0.92, 0.95, 0.95, 0.85, 0.97, 0.93, 0.90
], dtype=np.float64)  # energy, stress, loneliness, insecurity, irritation, longing, battery, fatigue

_RELATIONSHIP_BASE_DECAY = np.array([
    0.995, 0.990, 0.985, 0.980, 0.990, 0.970
], dtype=np.float64)  # affection, trust, familiarity, dependency, safety, tension


# ═══════════════════════════════════════════════════════════════
# ⑤ Relationship Dynamics 矩阵
# ═══════════════════════════════════════════════════════════════


def _build_rel_state_coupling() -> np.ndarray:
    """关系状态耦合矩阵 A_rel（R_SIZE × R_SIZE）。"""
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
    """关系输入影响矩阵 B_rel（ST_SIZE × R_SIZE）。"""
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


REL_STATE_COUPLING_A = _spectral_normalize(_build_rel_state_coupling(), "REL_STATE_COUPLING_A")
REL_INPUT_INFLUENCE_B = _build_rel_input_influence()
