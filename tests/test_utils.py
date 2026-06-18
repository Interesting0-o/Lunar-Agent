"""Layer 1: 工具函数测试 — soft_clamp, _sigmoid, _sigmoid_gate。

大量数据测试: 覆盖极端值、边界、数值稳定性。
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_less
from state_engine._utils import soft_clamp, _sigmoid, _sigmoid_gate


# ═══════════════════════════════════════════════════════════════
# soft_clamp
# ═══════════════════════════════════════════════════════════════

class TestSoftClampBounds:
    """测试 soft_clamp 的恒等和边界软饱和行为（默认 [-1, 1]）。"""

    def test_identity_within_bounds_single(self):
        """区间 [-1, 1] 内恒等通过（无压缩）。"""
        for x in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            result = soft_clamp(np.array([x]))
            assert result[0] == pytest.approx(x, abs=1e-10), f"x={x} → {result[0]}"

    def test_identity_within_bounds_massive(self, rng):
        """区间 [-1, 1] 内大量随机值：全部恒等通过。"""
        x = rng.uniform(-1.0, 1.0, size=50_000)
        result = soft_clamp(x)
        assert_array_almost_equal(result, x, decimal=12)

    def test_upper_suppression(self, rng):
        """上界压制：x > 1 被 tanh 平滑压缩到 [1.0, ~1.1]，不产生 NaN。"""
        x = rng.uniform(1.0, 10.0, size=20_000)
        result = soft_clamp(x)
        assert np.all(np.isfinite(result)), "出现 NaN/Inf"
        assert np.all(result >= 1.0 - 1e-10), "低于下界"
        # 超出越多，渐进逼近 high+transition = 1.1
        assert np.all(result <= 1.0 + 0.11), "过度超出"

    def test_lower_suppression(self, rng):
        """下界压制：x < -1 被 tanh 平滑压缩到 [~-1.1, -1.0]。"""
        x = rng.uniform(-5.0, -1.0, size=20_000)
        result = soft_clamp(x)
        assert np.all(np.isfinite(result)), "出现 NaN/Inf"
        assert np.all(result >= -1.0 - 0.11), "过度超出"
        assert np.all(result <= -1.0 + 1e-10), "超过上界"

    def test_extreme_values_no_nan(self):
        """极端值 ±∞ 不产生 NaN。"""
        for x in [1e10, -1e10, 1e100, -1e100]:
            result = soft_clamp(np.array([x]))
            assert np.isfinite(result[0]), f"x={x} → NaN"
        # inf 本身
        result = soft_clamp(np.array([np.inf]))
        assert np.isfinite(result[0]), "inf → NaN"


class TestSoftClampMonotonicity:
    """soft_clamp 的单调性。

    新版 soft_clamp 区间内恒等，边界外 tanh 软饱和，全程 C¹ 单调。
    """

    def test_monotonic_within_bounds(self, rng):
        """区间 (-0.95, 0.95) 内严格单调 — 恒等映射保持单调。"""
        x = rng.uniform(-0.95, 0.95, size=5000)
        x.sort()
        result = soft_clamp(x)
        assert np.all(np.diff(result) >= -1e-15), "区间内非单调"

    def test_monotonic_below_low(self, rng):
        """下界以下单调递增（朝 low 收敛）。"""
        x = rng.uniform(-3.0, -1.2, size=5000)  # 全部低于下界 -1.0
        x.sort()
        result = soft_clamp(x)
        diffs = np.diff(result)
        assert np.all(diffs >= -1e-15), f"下界外非单调递增: min diff={diffs.min():.2e}"

    def test_monotonic_above_high(self, rng):
        """上界以上单调递增（朝 high 收敛）。"""
        x = rng.uniform(1.5, 5.0, size=5000)  # 充分高于上界
        x.sort()
        result = soft_clamp(x)
        diffs = np.diff(result)
        assert np.all(diffs >= -1e-15), f"上界外非单调递增: min diff={diffs.min():.2e}"

    def test_global_monotonic(self):
        """验证 soft_clamp 全局单调。"""
        x = np.linspace(-1.5, 1.5, 1000)
        result = soft_clamp(x)
        diffs = np.diff(result)
        assert np.all(diffs >= -1e-15), \
            f"全局非单调: min diff={diffs.min():.2e} at idx={np.argmin(diffs)}"


class TestSoftClampCustomBounds:
    """自定义上下界的 soft_clamp。"""

    def test_custom_bounds_identity(self):
        """自定义 [-0.5, 0.5] 区间内恒等通过。"""
        x = np.linspace(-0.5, 0.5, 1000)
        result = soft_clamp(x, low=-0.5, high=0.5)
        assert_array_almost_equal(result, x, decimal=12)

    def test_custom_bounds_symmetric(self):
        """[-1, 1] 对称区间：超出部分 tanh 压制到 ~±1.1。"""
        result = soft_clamp(np.array([2.0, -2.0]), low=-1.0, high=1.0)
        assert 1.0 < result[0] <= 1.11, f"上界压制异常: {result[0]}"
        assert -1.11 <= result[1] < -1.0, f"下界压制异常: {result[1]}"

    def test_old_default_bounds(self):
        """验证旧默认 [0, 1] 区间内恒等通过。"""
        result = soft_clamp(np.array([0.0, 0.5, 1.0]), low=0.0, high=1.0)
        assert result[0] == pytest.approx(0.0, abs=1e-10)  # x=0 → 0 (恒等)
        assert result[1] == pytest.approx(0.5, abs=1e-10)  # x=0.5 → 0.5
        assert result[2] == pytest.approx(1.0, abs=1e-10)  # x=1 → 1

    def test_wide_transition(self, rng):
        """transition 越宽，超出部分被允许浮动越远。

        narrow (t=0.1): tanh 渐近 high+0.1
        wide (t=0.5): tanh 渐近 high+0.5
        所以 wide > narrow（宽 transition 允许更远的超出）。
        """
        x = rng.uniform(1.5, 5.0, size=2000)  # 充分高于上界
        narrow = soft_clamp(x, transition=0.1)
        wide = soft_clamp(x, transition=0.5)
        assert np.all(wide > narrow), \
            f"宽transition应浮动更远: wide=[{wide.min():.4f}, {wide.max():.4f}], narrow=[{narrow.min():.4f}, {narrow.max():.4f}]"


# ═══════════════════════════════════════════════════════════════
# _sigmoid
# ═══════════════════════════════════════════════════════════════

class TestSigmoid:
    """_sigmoid 的数值稳定性和数学性质。"""

    def test_center(self):
        """σ(0) = 0.5。"""
        assert _sigmoid(np.array([0.0]))[0] == pytest.approx(0.5, abs=1e-12)

    def test_symmetry(self):
        """σ(-x) = 1 - σ(x)。"""
        x = np.linspace(-5, 5, 1000)
        assert_array_almost_equal(
            _sigmoid(-x), 1.0 - _sigmoid(x), decimal=12,
        )

    def test_monotonic(self, rng):
        """严格单调递增。"""
        x = rng.uniform(-10, 10, size=5000)
        x.sort()
        result = _sigmoid(x)
        assert np.all(np.diff(result) > 0), "sigmoid 不单调"

    def test_large_positive_no_overflow(self):
        """大正数不溢出，结果 ≈ 1。"""
        for x in [10, 50, 100, 500, 1000]:
            result = _sigmoid(np.array([x]))
            assert np.isfinite(result[0]), f"x={x} → 非有限"
            assert result[0] > 0.999, f"x={x} → {result[0]}"

    def test_large_negative_no_underflow(self):
        """大负数不产生 NaN，结果 ≈ 0。"""
        for x in [-10, -50, -100, -500, -1000]:
            result = _sigmoid(np.array([x]))
            assert np.isfinite(result[0]), f"x={x} → 非有限"
            assert result[0] < 0.001, f"x={x} → {result[0]}"

    def test_range(self, rng):
        """所有输出 ∈ [0, 1]（允许恰好 0 或 1，float64 精度有限）。

        已知: x ≥ 37 时 σ(x) 在 float64 下恰好为 1.0。
        """
        x = rng.uniform(-50, 50, size=50_000)
        result = _sigmoid(x)
        assert np.all(result >= 0.0), "出现 <0"
        assert np.all(result <= 1.0), "出现 >1"

    def test_bulk_monotonic_many_distributions(self, rng):
        """多种分布下均保持单调。"""
        for dist_name, dist in [
            ("uniform", rng.uniform(-20, 20, size=5000)),
            ("normal", rng.normal(0, 3, size=5000)),
            ("beta_u", rng.beta(0.5, 0.5, size=5000) * 20 - 10),
        ]:
            dist.sort()
            result = _sigmoid(dist)
            assert np.all(np.diff(result) > 0), f"{dist_name} 不单调"


# ═══════════════════════════════════════════════════════════════
# _sigmoid_gate
# ═══════════════════════════════════════════════════════════════

class TestSigmoidGate:
    """门控专用的 sigmoid 激活。"""

    def test_range(self, rng):
        """输出 ∈ (0, 1)。"""
        x = rng.uniform(-10, 10, size=5000)
        result = _sigmoid_gate(x)
        assert np.all(result > 0.0)
        assert np.all(result < 1.0)

    def test_midpoint(self):
        """raw=0 时输出 ≈ 0.5（sigoid 中点）。"""
        result = _sigmoid_gate(np.array([0.0]))
        assert result[0] == pytest.approx(0.5, abs=1e-12)

    def test_monotonic(self, rng):
        """单调递增。"""
        x = rng.uniform(-5, 5, size=3000)
        x.sort()
        result = _sigmoid_gate(x)
        assert np.all(np.diff(result) >= -1e-15)


# ═══════════════════════════════════════════════════════════════
# 综合压力测试
# ═══════════════════════════════════════════════════════════════

class TestStressSoftClamp:
    """soft_clamp 在大量随机参数下的压力测试。"""

    def test_random_bounds_identity(self, rng):
        """随机上下界 + 区间内值 → 恒等通过。"""
        for _ in range(500):
            low = rng.uniform(-3, 2)
            high = low + rng.uniform(0.5, 5)
            x = rng.uniform(low, high, size=200)
            result = soft_clamp(x, low=low, high=high)
            # 验证在 [low, high] 内且恒等
            assert_array_almost_equal(result, x, decimal=12)
            assert np.all(result >= low - 1e-10), f"低于下界 low={low:.2f}"
            assert np.all(result <= high + 1e-10), f"超过上界 high={high:.2f}"

    def test_random_bounds_suppression(self, rng):
        """随机上下界 + 区间外值 被压制且不产生 NaN。"""
        for _ in range(500):
            low = rng.uniform(-3, 2)
            high = low + rng.uniform(0.5, 5)
            # 值在区间外
            x = np.concatenate([
                rng.uniform(high, high + 5, size=100),
                rng.uniform(low - 5, low, size=100),
            ])
            result = soft_clamp(x, low=low, high=high, transition=0.5)
            assert np.all(np.isfinite(result)), f"NaN at low={low:.2f} high={high:.2f}"

    def test_tiny_transition_no_nan(self, rng):
        """极小 transition 不产生 NaN。"""
        x = rng.uniform(-10, 10, size=2000)
        result = soft_clamp(x, transition=1e-6)
        assert np.all(np.isfinite(result))
