"""Layer 4: 时间衰减测试 — 指数衰减、人格调制、半衰期验证。"""

import numpy as np
import pytest
from state import (
    DEFAULT_TRAITS, DEFAULT_INTERNAL, DEFAULT_RELATIONSHIP,
    I_ENERGY, I_STRESS, I_IRRITATION,
    T_EMOTIONAL_STABILITY, T_OPTIMISM, T_ANXIETY_PRONENESS,
    T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE,
)
from state_engine._decay import (
    DecayConfig,
    apply_time_decay_internal,
    apply_time_decay_relationship,
    apply_time_decay,
    _compute_internal_personality_mod,
    _compute_relationship_personality_mod,
    _compute_lambda_effective,
    _compute_setpoint_for_decay,
    _compute_rel_setpoint_for_decay,
)


class TestTimeDecayZero:
    """Δt = 0 时无变化。"""

    def test_internal_zero_delta(self, default_traits):
        setpoint = _compute_setpoint_for_decay(default_traits)
        result = apply_time_decay_internal(
            DEFAULT_INTERNAL, setpoint, default_traits, delta_hours=0.0,
        )
        assert np.allclose(result, DEFAULT_INTERNAL, atol=1e-12), \
            f"Δt=0 应原样返回: diff={np.abs(result - DEFAULT_INTERNAL).max():.2e}"

    def test_relationship_zero_delta(self, default_traits):
        setpoint = _compute_rel_setpoint_for_decay(default_traits)
        result = apply_time_decay_relationship(
            DEFAULT_RELATIONSHIP, setpoint, default_traits, delta_hours=0.0,
        )
        assert np.allclose(result, DEFAULT_RELATIONSHIP, atol=1e-12)


class TestTimeDecayConvergence:
    """Δt → ∞ 时趋近 setpoint。"""

    def test_internal_long_time_converges(self, default_traits):
        """5000 小时后应非常接近 setpoint（考虑最慢衰减维度 I_LONGING λ=0.12/h）。"""
        setpoint = _compute_setpoint_for_decay(default_traits)
        # 从中等偏离开始
        current = np.clip(setpoint + 0.2, -1.0, 1.0)
        result = apply_time_decay_internal(
            current, setpoint, default_traits, delta_hours=5000.0,
        )
        deviation = np.abs(result - setpoint).max()
        assert deviation < 0.05, f"5000h 后偏差={deviation:.6f}"

    def test_relationship_long_time_converges(self, default_traits):
        """10000 小时后应接近 setpoint（关系衰减更慢）。"""
        setpoint = _compute_rel_setpoint_for_decay(default_traits)
        current = np.clip(setpoint + 0.3, -1.0, 1.0)
        result = apply_time_decay_relationship(
            current, setpoint, default_traits, delta_hours=10000.0,
        )
        deviation = np.abs(result - setpoint).max()
        assert deviation < 0.1, f"10000h 后偏差={deviation:.6f}"


class TestTimeDecayDirection:
    """衰减方向: 始终向 setpoint 靠近。"""

    def test_internal_moves_toward_setpoint(self, rng, default_traits):
        """任意初始状态 + 任意 Δt: 衰减后离 setpoint 更近。"""
        setpoint = _compute_setpoint_for_decay(default_traits)
        n = 10_000
        currents = rng.uniform(-1, 1, size=(n, 8))
        deltas = rng.uniform(0.1, 50, size=n)

        for i in range(n):
            result = apply_time_decay_internal(
                currents[i], setpoint, default_traits, delta_hours=deltas[i],
            )
            before_dist = np.linalg.norm(currents[i] - setpoint)
            after_dist = np.linalg.norm(result - setpoint)
            assert after_dist <= before_dist + 1e-12, (
                f"Δt={deltas[i]:.1f}h 离 setpoint 更远了: {before_dist:.4f} → {after_dist:.4f}"
            )

    def test_relationship_moves_toward_setpoint(self, rng, default_traits):
        """关系状态同样向 setpoint 靠近。"""
        setpoint = _compute_rel_setpoint_for_decay(default_traits)
        n = 10_000
        currents = rng.uniform(-1, 1, size=(n, 6))
        deltas = rng.uniform(1, 100, size=n)

        for i in range(n):
            result = apply_time_decay_relationship(
                currents[i], setpoint, default_traits, delta_hours=deltas[i],
            )
            before_dist = np.linalg.norm(currents[i] - setpoint)
            after_dist = np.linalg.norm(result - setpoint)
            assert after_dist <= before_dist + 1e-12, (
                f"Δt={deltas[i]:.1f}h 离 setpoint 更远了: {before_dist:.4f} → {after_dist:.4f}"
            )


class TestPersonalityModulation:
    """人格调制因子: 情绪稳定→恢复快, 焦虑→恢复慢。"""

    def test_stable_recovers_faster(self):
        """高情绪稳定性 → 相同 Δt 下离 setpoint 更近。"""
        t_low = DEFAULT_TRAITS.copy(); t_low[T_EMOTIONAL_STABILITY] = -0.8
        t_high = DEFAULT_TRAITS.copy(); t_high[T_EMOTIONAL_STABILITY] = 0.8

        setpoint_low = _compute_setpoint_for_decay(t_low)
        setpoint_high = _compute_setpoint_for_decay(t_high)

        current = np.full(8, 0.8)  # 远高于 setpoint

        r_low = apply_time_decay_internal(current, setpoint_low, t_low, delta_hours=1.0)
        r_high = apply_time_decay_internal(current, setpoint_high, t_high, delta_hours=1.0)

        dev_low = np.linalg.norm(r_low - setpoint_low)
        dev_high = np.linalg.norm(r_high - setpoint_high)
        assert dev_high < dev_low, (
            f"高稳定应恢复更快: stable dev={dev_high:.4f} ≥ unstable dev={dev_low:.4f}"
        )

    def test_anxious_recovers_slower(self):
        """高焦虑 → 恢复更慢。

        比较相同初始偏差下的恢复比例（因为不同人格有不同 setpoint）。
        """
        t_low = DEFAULT_TRAITS.copy(); t_low[T_ANXIETY_PRONENESS] = -0.8
        t_high = DEFAULT_TRAITS.copy(); t_high[T_ANXIETY_PRONENESS] = 0.8

        setpoint_low = _compute_setpoint_for_decay(t_low)
        setpoint_high = _compute_setpoint_for_decay(t_high)

        # 对各自的 setpoint 产生相同的偏离
        current_low = np.clip(setpoint_low + 0.3, -1.0, 1.0)
        current_high = np.clip(setpoint_high + 0.3, -1.0, 1.0)

        r_low = apply_time_decay_internal(current_low, setpoint_low, t_low, delta_hours=1.0)
        r_high = apply_time_decay_internal(current_high, setpoint_high, t_high, delta_hours=1.0)

        # 用恢复比例替代绝对偏差
        initial_dev_low = np.linalg.norm(current_low - setpoint_low)
        initial_dev_high = np.linalg.norm(current_high - setpoint_high)
        recovery_low = 1.0 - np.linalg.norm(r_low - setpoint_low) / initial_dev_low
        recovery_high = 1.0 - np.linalg.norm(r_high - setpoint_high) / initial_dev_high

        assert recovery_high < recovery_low, (
            f"高焦虑恢复比例应更低: anxious={recovery_high:.4f} ≥ calm={recovery_low:.4f}"
        )

    def test_personality_mod_range(self, rng):
        """人格调制因子 ∈ [0.3, 2.0]。"""
        n = 20_000
        traits = rng.uniform(-1, 1, size=(n, 10))
        for i in range(n):
            mod_i = _compute_internal_personality_mod(traits[i])
            mod_r = _compute_relationship_personality_mod(traits[i])
            assert 0.25 <= float(mod_i) <= 2.05, f"internal mod={mod_i:.3f}"
            assert 0.25 <= float(mod_r) <= 2.05, f"rel mod={mod_r:.3f}"


class TestHalfLife:
    """半衰期验证: Δt = ln(2)/λ_eff 后 deviation 减半。"""

    def test_internal_half_life_irritation(self):
        """I_IRRITATION 的 λ_base=0.69/hr, 半衰期 ≈ ln(2)/0.69 ≈ 1h。"""
        config = DecayConfig()
        lam_base = config.internal_lambda[I_IRRITATION]
        expected_hl = np.log(2) / lam_base
        print(f"\n  I_IRRITATION λ_base={lam_base:.2f}/h, 期望半衰期={expected_hl:.2f}h")

        setpoint = _compute_setpoint_for_decay(DEFAULT_TRAITS)
        current = DEFAULT_INTERNAL.copy()
        current[I_IRRITATION] = 0.6  # 偏离 setpoint

        initial_dev = current[I_IRRITATION] - setpoint[I_IRRITATION]

        result = apply_time_decay_internal(
            current, setpoint, DEFAULT_TRAITS, delta_hours=expected_hl,
        )
        after_dev = result[I_IRRITATION] - setpoint[I_IRRITATION]

        assert after_dev < initial_dev * 0.6, (
            f"半衰期 {expected_hl:.2f}h 后 deviate={after_dev:.4f}, "
            f"期望≤{initial_dev*0.5:.4f}（因人格调制可能有偏差）"
        )


class TestDecayOutputBounds:
    """时间衰减输出 ∈ [-1, 1]。"""

    def test_internal_bounds_bulk(self, rng, default_traits):
        """大量随机输入: 衰减后不越界。"""
        n = 20_000
        setpoint = _compute_setpoint_for_decay(default_traits)
        currents = rng.uniform(-1, 1, size=(n, 8))
        deltas = rng.uniform(0, 200, size=n)

        for i in range(n):
            result = apply_time_decay_internal(
                currents[i], setpoint, default_traits, delta_hours=deltas[i],
            )
            assert np.all(np.isfinite(result)), f"[{i}] NaN"
            assert result.min() >= -1.0 - 1e-10, f"[{i}] min={result.min()}"
            assert result.max() <= 1.0 + 1e-10, f"[{i}] max={result.max()}"

    def test_relationship_bounds_bulk(self, rng, default_traits):
        """大量随机输入: 关系衰减后不越界。"""
        n = 20_000
        setpoint = _compute_rel_setpoint_for_decay(default_traits)
        currents = rng.uniform(-1, 1, size=(n, 6))
        deltas = rng.uniform(0, 500, size=n)

        for i in range(n):
            result = apply_time_decay_relationship(
                currents[i], setpoint, default_traits, delta_hours=deltas[i],
            )
            assert np.all(np.isfinite(result)), f"[{i}] NaN"
            assert result.min() >= -1.0 - 1e-10, f"[{i}] min={result.min()}"
            assert result.max() <= 1.0 + 1e-10, f"[{i}] max={result.max()}"

    def test_apply_time_decay_combined(self, default_traits):
        """apply_time_decay 便捷接口正常。"""
        result = apply_time_decay(
            DEFAULT_INTERNAL, DEFAULT_RELATIONSHIP, default_traits, delta_hours=1.0,
        )
        assert "internal_state" in result
        assert "relationship_state" in result
        assert result["internal_state"].shape == (8,)
        assert result["relationship_state"].shape == (6,)


class TestLambdaEffective:
    """有效衰减率计算。"""

    def test_time_damping(self):
        """时间曲线: Δt 越大阻尼越强。"""
        lam_base = np.ones(8) * 0.5
        mod = 1.0

        lam_small = _compute_lambda_effective(lam_base, mod, delta_hours=0.1, time_curve_k=0.05)
        lam_large = _compute_lambda_effective(lam_base, mod, delta_hours=100.0, time_curve_k=0.05)

        assert np.all(lam_small > lam_large), (
            f"小Δt={lam_small[0]:.6f} 应 > 大Δt={lam_large[0]:.6f}"
        )
