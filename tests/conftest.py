"""共享测试 fixtures 和工具函数。"""

import numpy as np
import pytest


def pytest_addoption(parser):
    """Register optional flags for sensitivity analysis and other heavy tests."""
    parser.addoption(
        "--run-full-sensitivity",
        action="store_true",
        default=False,
        help="Run full sensitivity analysis (30 scenarios) and print report.",
    )


from state import (
    DEFAULT_TRAITS, DEFAULT_INTERNAL, DEFAULT_RELATIONSHIP,
    T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE,
    T_EMOTIONAL_STABILITY, T_OPTIMISM, T_ANXIETY_PRONENESS,
    T_ANGER_REACTIVITY, T_PRIDE, T_SENSITIVITY, T_JEALOUSY_SENSITIVITY,
    T_EMOTIONAL_OPENNESS,
    ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT,
    ST_DEPENDENCY, ST_TEASING, ST_EMOTIONAL_WEIGHT, ST_SIZE,
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE, I_SIZE,
    R_AFFECTION, R_TRUST_BOND, R_INTIMACY, R_SIZE,
    S_EXPRESSIVENESS, S_WARMTH, S_SHARPNESS, S_SOFTNESS,
    S_ENTHUSIASM, S_RESTRAINT, S_VULNERABILITY, S_SIZE,
    I_LABELS, R_LABELS, S_LABELS, ST_LABELS, T_LABELS,
)
from state_engine import update_all, initialize_all


@pytest.fixture
def default_traits():
    """默认人格特质 (10,)。"""
    return DEFAULT_TRAITS.copy()


@pytest.fixture
def default_internal():
    """默认内部状态 (8,)。"""
    return DEFAULT_INTERNAL.copy()


@pytest.fixture
def default_relationship():
    """默认关系状态 (3,)。"""
    return DEFAULT_RELATIONSHIP.copy()


@pytest.fixture
def zero_stimuli():
    """零刺激向量 (7,)。"""
    return np.zeros(ST_SIZE, dtype=np.float64)


@pytest.fixture
def extreme_stimuli():
    """极端高强度刺激 (7,) — 所有维度拉到 0.9。"""
    return np.full(ST_SIZE, 0.9, dtype=np.float64)


@pytest.fixture
def extreme_negative_stimuli():
    """极端负面刺激 — 被抛弃 + 冲突 + 高情绪重量。"""
    s = np.zeros(ST_SIZE, dtype=np.float64)
    s[ST_ABANDONMENT] = 0.9
    s[ST_CONFLICT] = 0.9
    s[ST_EMOTIONAL_WEIGHT] = 0.9
    return s


@pytest.fixture
def extreme_positive_stimuli():
    """极端正面刺激 — 被认可 + 亲密 + 被依赖。"""
    s = np.zeros(ST_SIZE, dtype=np.float64)
    s[ST_VALIDATION] = 0.9
    s[ST_CLOSENESS] = 0.9
    s[ST_DEPENDENCY] = 0.9
    return s


@pytest.fixture
def single_stimuli():
    """每种刺激单独为 1.0 的刺激矩阵 (7, 7) — 每行对应一种刺激。"""
    return np.eye(ST_SIZE, dtype=np.float64)


@pytest.fixture
def random_state(rng):
    """随机种子工厂 — 每次调用生成不同的 rng。"""
    return np.random.default_rng


# ── 大规模测试用的随机刺激生成器 ──

def generate_random_stimuli(rng: np.random.Generator, n: int = 1000) -> np.ndarray:
    """生成 n 组随机刺激向量 (n, 7)。

    使用多种分布混合以覆盖不同场景:
      - beta(0.5, 0.5): U型分布，覆盖极端值
      - beta(2, 2): 钟形分布，覆盖中等值
      - uniform: 均匀分布
      - 稀疏: 大量零 + 少量高强度
    """
    result = np.empty((n, ST_SIZE), dtype=np.float64)

    # 25% beta(0.5, 0.5) — 两极分化
    n1 = n // 4
    result[:n1] = rng.beta(0.5, 0.5, size=(n1, ST_SIZE))

    # 25% beta(2, 2) — 集中在中间
    n2 = n // 4
    result[n1:n1+n2] = rng.beta(2, 2, size=(n2, ST_SIZE))

    # 25% uniform
    n3 = n // 4
    result[n1+n2:n1+n2+n3] = rng.uniform(0, 1, size=(n3, ST_SIZE))

    # 25% 稀疏 — 70% 零 + 30% 高强度
    n4 = n - n1 - n2 - n3
    sparse = rng.uniform(0, 1, size=(n4, ST_SIZE))
    mask = rng.uniform(0, 1, size=(n4, ST_SIZE)) < 0.7
    sparse[mask] = 0.0
    result[n1+n2+n3:] = sparse

    rng.shuffle(result)
    return result


def generate_random_traits(rng: np.random.Generator, n: int = 500) -> np.ndarray:
    """生成 n 组随机人格特质 (n, 10)，全部 ∈ [-1, 1]。"""
    return rng.uniform(-1, 1, size=(n, 10))


def generate_random_states(rng: np.random.Generator, n: int = 500) -> tuple:
    """生成 n 组随机初始状态，全部 ∈ [-1, 1]。"""
    internal = rng.uniform(-1, 1, size=(n, I_SIZE))
    relationship = rng.uniform(-1, 1, size=(n, R_SIZE))
    return internal, relationship


def describe_violations(
    values: np.ndarray,
    labels: list,
    low: float = -1.0,
    high: float = 1.0,
    tol: float = 1e-10,
) -> list[str]:
    """检测并描述超出范围的异常值。

    Returns:
        违规描述列表，每个元素如 "energy: 1.05 (max) at index 42"
    """
    violations = []
    for i, label in enumerate(labels):
        col = values[:, i] if values.ndim > 1 else values
        if values.ndim > 1:
            col = values[:, i]
        else:
            col = values

        below = col < low - tol
        above = col > high + tol
        for idx in np.where(below)[0]:
            violations.append(
                f"  {label}: {col[idx]:.6f} (below {low}) at sample {idx}"
            )
        for idx in np.where(above)[0]:
            violations.append(
                f"  {label}: {col[idx]:.6f} (above {high}) at sample {idx}"
            )
    return violations


# ── 全局 rng fixture ──

@pytest.fixture
def rng():
    """固定种子的随机数生成器，保证可复现。"""
    return np.random.default_rng(42)
