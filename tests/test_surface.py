"""Layer 4: 表面投影测试 — project_surface 的范围、方向性、惯性混合与反馈。"""

import numpy as np
import pytest
from state import (
    DEFAULT_TRAITS, DEFAULT_INTERNAL, DEFAULT_RELATIONSHIP,
    S_EXPRESSIVENESS, S_WARMTH, S_SHARPNESS, S_SOFTNESS,
    S_ENTHUSIASM, S_RESTRAINT, S_VULNERABILITY, S_SIZE,
    S_LABELS,
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_MENTAL_FATIGUE,
    R_AFFECTION, R_TRUST_BOND,
    ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT,
    ST_DEPENDENCY, ST_TEASING, ST_EMOTIONAL_WEIGHT, ST_SIZE,
)
from state_engine._surface import project_surface


class TestSurfaceProjectionBounds:
    """表面投影输出必须在 [-1, 1]。"""

    def test_default_output(self, default_internal, default_relationship):
        zero_outer = np.zeros(ST_SIZE)
        surface = project_surface(default_internal, default_relationship, zero_outer)
        assert surface.shape == (S_SIZE,), f"shape={surface.shape}"
        assert np.all(surface >= -1.0 - 0.11), f"min={surface.min()}"
        assert np.all(surface <= 1.0), f"max={surface.max()}"

    def test_bulk_random(self, rng):
        """5000 组随机输入: surface ∈ [0, 1]。"""
        n = 20_000
        internal = rng.uniform(-1, 1, size=(n, 8))
        relationship = rng.uniform(-1, 1, size=(n, 3))
        outer_stimuli = rng.uniform(0, 1, size=(n, 7))

        violations = 0
        nan_count = 0
        for i in range(n):
            s = project_surface(internal[i], relationship[i], outer_stimuli[i])
            if not np.all(np.isfinite(s)):
                nan_count += 1
            elif s.min() < -1.0 - 0.11 or s.max() > 1.0 + 0.11:
                violations += 1

        assert nan_count == 0, f"NaN/Inf: {nan_count}/{n}"
        assert violations == 0, f"越界: {violations}/{n}"

    def test_extreme_inputs(self):
        """极端输入不崩溃。"""
        surfaces = []
        for internal_val in [-1.0, 0.0, 1.0]:
            for rel_val in [-1.0, 0.0, 1.0]:
                for outer_val in [0.0, 0.5, 1.0]:
                    s = project_surface(
                        np.full(8, internal_val),
                        np.full(3, rel_val),
                        np.full(7, outer_val),
                    )
                    surfaces.append(s)

        all_s = np.array(surfaces)
        assert np.all(np.isfinite(all_s)), "NaN in extreme inputs"
        assert np.all(all_s >= -1.0 - 0.11) and np.all(all_s <= 1.0 + 0.11), \
            f"range=[{all_s.min():.6f}, {all_s.max():.6f}]"


class TestSurfaceDirectionality:
    """表面投影的心理方向性。"""

    def test_high_energy_high_enthusiasm(self, default_relationship):
        """高精力 → 高热情。"""
        outer = np.zeros(ST_SIZE)
        i_low = DEFAULT_INTERNAL.copy(); i_low[I_ENERGY] = -0.8
        i_high = DEFAULT_INTERNAL.copy(); i_high[I_ENERGY] = 0.8

        s_low = project_surface(i_low, default_relationship, outer)
        s_high = project_surface(i_high, default_relationship, outer)

        assert s_high[S_ENTHUSIASM] > s_low[S_ENTHUSIASM], (
            f"高精力 enthusiasm={s_high[S_ENTHUSIASM]:.3f} ≤ 低精力={s_low[S_ENTHUSIASM]:.3f}"
        )

    def test_high_irritation_high_sharpness(self, default_relationship):
        """高烦躁 → 高尖锐度。"""
        outer = np.zeros(ST_SIZE)
        i_low = DEFAULT_INTERNAL.copy(); i_low[I_IRRITATION] = -0.8
        i_high = DEFAULT_INTERNAL.copy(); i_high[I_IRRITATION] = 0.8

        s_low = project_surface(i_low, default_relationship, outer)
        s_high = project_surface(i_high, default_relationship, outer)

        assert s_high[S_SHARPNESS] > s_low[S_SHARPNESS], (
            f"高烦躁 sharpness={s_high[S_SHARPNESS]:.3f} ≤ 低烦躁={s_low[S_SHARPNESS]:.3f}"
        )

    def test_high_affection_high_warmth(self, default_internal):
        """高好感 → 高温暖。"""
        outer = np.zeros(ST_SIZE)
        r_low = DEFAULT_RELATIONSHIP.copy(); r_low[R_AFFECTION] = -0.8
        r_high = DEFAULT_RELATIONSHIP.copy(); r_high[R_AFFECTION] = 0.8

        s_low = project_surface(default_internal, r_low, outer)
        s_high = project_surface(default_internal, r_high, outer)

        assert s_high[S_WARMTH] > s_low[S_WARMTH], (
            f"高好感 warmth={s_high[S_WARMTH]:.3f} ≤ 低好感={s_low[S_WARMTH]:.3f}"
        )

    def test_high_stress_low_warmth(self, default_relationship):
        """高压力 → 低温暖。"""
        outer = np.zeros(ST_SIZE)
        i_low = DEFAULT_INTERNAL.copy(); i_low[I_STRESS] = -0.8
        i_high = DEFAULT_INTERNAL.copy(); i_high[I_STRESS] = 0.8

        s_low = project_surface(i_low, default_relationship, outer)
        s_high = project_surface(i_high, default_relationship, outer)

        assert s_high[S_WARMTH] < s_low[S_WARMTH], (
            f"高压力 warmth={s_high[S_WARMTH]:.3f} ≥ 低压力={s_low[S_WARMTH]:.3f}"
        )

    def test_high_fatigue_low_expressiveness(self, default_relationship):
        """高疲劳 → 低外露度。"""
        outer = np.zeros(ST_SIZE)
        i_low = DEFAULT_INTERNAL.copy(); i_low[I_MENTAL_FATIGUE] = -0.8
        i_high = DEFAULT_INTERNAL.copy(); i_high[I_MENTAL_FATIGUE] = 0.8

        s_low = project_surface(i_low, default_relationship, outer)
        s_high = project_surface(i_high, default_relationship, outer)

        assert s_high[S_EXPRESSIVENESS] < s_low[S_EXPRESSIVENESS], (
            f"高疲劳 expressiveness={s_high[S_EXPRESSIVENESS]:.3f} ≥ 低疲劳={s_low[S_EXPRESSIVENESS]:.3f}"
        )


class TestOuterStimuliEffect:
    """outer_stimuli 对表面表达的直接影响。"""

    def test_validation_outer_increases_warmth(
        self, default_internal, default_relationship,
    ):
        """被认可（outer）→ 增加温暖。"""
        outer_zero = np.zeros(ST_SIZE)
        outer_val = np.zeros(ST_SIZE); outer_val[ST_VALIDATION] = 0.9

        s_zero = project_surface(default_internal, default_relationship, outer_zero)
        s_val = project_surface(default_internal, default_relationship, outer_val)

        assert s_val[S_WARMTH] > s_zero[S_WARMTH], (
            f"validation outer 应增加 warmth: {s_zero[S_WARMTH]:.3f} → {s_val[S_WARMTH]:.3f}"
        )

    def test_conflict_outer_increases_sharpness(
        self, default_internal, default_relationship,
    ):
        """冲突（outer）→ 增加尖锐度。"""
        outer_zero = np.zeros(ST_SIZE)
        outer_con = np.zeros(ST_SIZE); outer_con[ST_CONFLICT] = 0.9

        s_zero = project_surface(default_internal, default_relationship, outer_zero)
        s_con = project_surface(default_internal, default_relationship, outer_con)

        assert s_con[S_SHARPNESS] > s_zero[S_SHARPNESS], (
            f"conflict outer 应增加 sharpness: {s_zero[S_SHARPNESS]:.3f} → {s_con[S_SHARPNESS]:.3f}"
        )

    def test_outer_stimuli_different_from_inner_only(self, rng):
        """验证 outer_stimuli 确实独立影响了 surface（不只是内部状态的翻版）。"""
        internal = DEFAULT_INTERNAL.copy()
        relationship = DEFAULT_RELATIONSHIP.copy()

        s1 = project_surface(internal, relationship, np.zeros(ST_SIZE))
        s2 = project_surface(internal, relationship, np.ones(ST_SIZE) * 0.5)

        assert not np.allclose(s1, s2, atol=1e-6), \
            "不同 outer_stimuli 应产生不同 surface"


class TestSurfaceStatistics:
    """表面投影的统计特征。"""

    def test_surface_distribution(self, rng):
        """大规模随机输入的 surface 分布。"""
        n = 20_000
        internal = rng.uniform(-1, 1, size=(n, 8))
        relationship = rng.uniform(-1, 1, size=(n, 3))
        outer = rng.uniform(0, 1, size=(n, 7))

        all_surfaces = np.empty((n, S_SIZE))
        for i in range(n):
            all_surfaces[i] = project_surface(internal[i], relationship[i], outer[i])

        print()
        for dim in range(S_SIZE):
            col = all_surfaces[:, dim]
            print(f"  {S_LABELS[dim]:>16s}: mean={col.mean():.3f} std={col.std():.3f} "
                  f"[{col.min():.3f}, {col.max():.3f}]")

        for dim in range(S_SIZE):
            col = all_surfaces[:, dim]
            assert col.std() > 0.01, f"{S_LABELS[dim]} std={col.std():.4f} 太小，可能卡在阈值"


class TestSurfaceInertia:
    """表面惯性混合：prev_surface 影响当前输出。"""

    def test_inertia_changes_output(self, default_internal, default_relationship):
        """相同 raw 输入，不同 prev_surface → 不同输出。"""
        outer = np.zeros(ST_SIZE)
        prev_a = np.ones(S_SIZE) * 0.8
        prev_b = np.ones(S_SIZE) * (-0.8)
        s_a = project_surface(default_internal, default_relationship, outer, prev_a)
        s_b = project_surface(default_internal, default_relationship, outer, prev_b)
        assert not np.allclose(s_a, s_b, atol=1e-4), \
            "不同 prev_surface 应产生不同 surface"

    def test_no_prev_differs_from_zero_prev(
        self, default_internal, default_relationship,
    ):
        """prev_surface=None 和全零 prev 应有差异（None 无拖拽，全零有拖拽）。"""
        outer = np.zeros(ST_SIZE)
        s_none = project_surface(default_internal, default_relationship, outer)
        s_zero = project_surface(
            default_internal, default_relationship, outer,
            prev_surface=np.zeros(S_SIZE),
        )
        assert not np.allclose(s_none, s_zero, atol=1e-4), \
            "None 和全零 prev_surface 应有差异"

    def test_high_stress_increases_inertia(
        self, default_relationship,
    ):
        """高压力下 surface 更接近 prev_surface（惯性更强，alpha 更低）。"""
        outer = np.zeros(ST_SIZE)
        prev = np.ones(S_SIZE) * 0.9
        i_stressed = DEFAULT_INTERNAL.copy(); i_stressed[I_STRESS] = 0.9
        i_calm = DEFAULT_INTERNAL.copy(); i_calm[I_STRESS] = -0.9

        s_stressed = project_surface(i_stressed, default_relationship, outer, prev)
        s_calm = project_surface(i_calm, default_relationship, outer, prev)

        dist_stressed = np.abs(s_stressed - prev).max()
        dist_calm = np.abs(s_calm - prev).max()
        assert dist_stressed <= dist_calm + 1e-6, \
            f"高压力下 surface 应更靠近 prev: {dist_stressed:.4f} vs {dist_calm:.4f}"


class TestSurfaceAlpha:
    """表面惯性系数 _compute_surface_alpha 的数学性质。"""

    def test_alpha_bounds(self, rng):
        from state_engine._surface import _compute_surface_alpha as _alpha
        """alpha ∈ [0.1, 0.9] for all valid internal states."""
        n = 2000
        for _ in range(n):
            internal = rng.uniform(-1, 1, size=8)
            a = _alpha(internal)
            assert 0.1 <= a <= 0.9, f"alpha={a:.6f} 超出 [0.1, 0.9]"

    def test_alpha_default(self):
        from state_engine._surface import _compute_surface_alpha as _alpha
        """默认内部状态下 alpha 应为基值 0.58（0.5 + 0.2*max(0,0.4)）。"""
        a = _alpha(DEFAULT_INTERNAL)
        # DEFAULT_INTERNAL: I_STRESS=-0.6, I_ENERGY=0.4
        # alpha = 0.5 - 0.3*max(0,-0.6) + 0.2*max(0,0.4) = 0.5 + 0.08 = 0.58
        assert abs(a - 0.58) < 0.01, f"alpha 偏离预期: {a:.4f}"

    def test_alpha_stress_reduces_alpha(self):
        from state_engine._surface import _compute_surface_alpha as _alpha
        """正压力 → alpha 下降（表面更僵）。"""
        i_low = DEFAULT_INTERNAL.copy(); i_low[I_STRESS] = -0.8
        i_high = DEFAULT_INTERNAL.copy(); i_high[I_STRESS] = 0.8
        a_low = _alpha(i_low)
        a_high = _alpha(i_high)
        assert a_high < a_low, \
            f"高压力 alpha={a_high:.4f} >= 低压力 alpha={a_low:.4f}"

    def test_alpha_energy_increases_alpha(self):
        from state_engine._surface import _compute_surface_alpha as _alpha
        """正精力 → alpha 上升（表面更灵）。"""
        i_low = DEFAULT_INTERNAL.copy(); i_low[I_ENERGY] = -0.8
        i_high = DEFAULT_INTERNAL.copy(); i_high[I_ENERGY] = 0.8
        a_low = _alpha(i_low)
        a_high = _alpha(i_high)
        assert a_high > a_low, \
            f"高精力 alpha={a_high:.4f} <= 低精力 alpha={a_low:.4f}"


class TestSurfaceFeedback:
    """表面→内部反馈的数值性质。"""

    def test_feedback_shape(self):
        from state_engine._surface import compute_surface_feedback as _feedback
        from state import I_SIZE
        """反馈应为 (8,) 向量。"""
        surface = np.zeros(S_SIZE)
        internal = DEFAULT_INTERNAL.copy()
        fb = _feedback(surface, internal)
        assert fb.shape == (I_SIZE,), f"shape={fb.shape}"
        assert np.all(np.isfinite(fb)), "NaN in feedback"

    def test_feedback_magnitude(self, rng):
        from state_engine._surface import compute_surface_feedback as _feedback
        from state import I_SIZE
        """任何有效输入下反馈幅度不大 (max|fb| < 0.2)。"""
        n = 2000
        for _ in range(n):
            surface = rng.uniform(-1, 1, size=S_SIZE)
            internal = rng.uniform(-1, 1, size=I_SIZE)
            fb = _feedback(surface, internal)
            assert np.all(np.abs(fb) < 0.2), \
                f"反馈值过大: max|fb|={np.abs(fb).max():.4f}"

    def test_feedback_zero_with_zero_surface(self):
        from state_engine._surface import compute_surface_feedback as _feedback
        """surface=0 → feedback ≈ 0。"""
        fb = _feedback(np.zeros(S_SIZE), DEFAULT_INTERNAL.copy())
        assert np.all(np.abs(fb) < 1e-15), f"零 surface 反馈: {fb}"
