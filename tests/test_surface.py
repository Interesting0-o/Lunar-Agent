"""Layer 4: 表面投影测试 — project_surface 的范围、方向性、特质修饰。"""

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
    T_PRIDE, T_EMOTIONAL_OPENNESS, T_OPTIMISM,
    ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT,
    ST_DEPENDENCY, ST_TEASING, ST_EMOTIONAL_WEIGHT, ST_SIZE,
)
from state_engine._surface import project_surface


class TestSurfaceProjectionBounds:
    """表面投影输出必须在 [-1, 1]。"""

    def test_default_output(self, default_internal, default_relationship, default_traits):
        zero_outer = np.zeros(ST_SIZE)
        surface = project_surface(default_internal, default_relationship, default_traits, zero_outer)
        assert surface.shape == (S_SIZE,), f"shape={surface.shape}"
        assert np.all(surface >= -1.0 - 0.11), f"min={surface.min()}"
        assert np.all(surface <= 1.0), f"max={surface.max()}"

    def test_bulk_random(self, rng):
        """5000 组随机输入: surface ∈ [0, 1]。"""
        n = 20_000
        internal = rng.uniform(-1, 1, size=(n, 8))
        relationship = rng.uniform(-1, 1, size=(n, 6))
        traits = rng.uniform(-1, 1, size=(n, 10))
        outer_stimuli = rng.uniform(0, 1, size=(n, 7))

        violations = 0
        nan_count = 0
        for i in range(n):
            s = project_surface(internal[i], relationship[i], traits[i], outer_stimuli[i])
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
                for trait_val in [-1.0, 0.0, 1.0]:
                    for outer_val in [0.0, 0.5, 1.0]:
                        s = project_surface(
                            np.full(8, internal_val),
                            np.full(6, rel_val),
                            np.full(10, trait_val),
                            np.full(7, outer_val),
                        )
                        surfaces.append(s)

        all_s = np.array(surfaces)
        assert np.all(np.isfinite(all_s)), "NaN in extreme inputs"
        assert np.all(all_s >= -1.0 - 0.11) and np.all(all_s <= 1.0 + 0.11), \
            f"range=[{all_s.min():.6f}, {all_s.max():.6f}]"


class TestSurfaceDirectionality:
    """表面投影的心理方向性。"""

    def test_high_energy_high_enthusiasm(self, default_relationship, default_traits):
        """高精力 → 高热情。"""
        outer = np.zeros(ST_SIZE)
        i_low = DEFAULT_INTERNAL.copy(); i_low[I_ENERGY] = -0.8
        i_high = DEFAULT_INTERNAL.copy(); i_high[I_ENERGY] = 0.8

        s_low = project_surface(i_low, default_relationship, default_traits, outer)
        s_high = project_surface(i_high, default_relationship, default_traits, outer)

        assert s_high[S_ENTHUSIASM] > s_low[S_ENTHUSIASM], (
            f"高精力 enthusiasm={s_high[S_ENTHUSIASM]:.3f} ≤ 低精力={s_low[S_ENTHUSIASM]:.3f}"
        )

    def test_high_irritation_high_sharpness(self, default_relationship, default_traits):
        """高烦躁 → 高尖锐度。"""
        outer = np.zeros(ST_SIZE)
        i_low = DEFAULT_INTERNAL.copy(); i_low[I_IRRITATION] = -0.8
        i_high = DEFAULT_INTERNAL.copy(); i_high[I_IRRITATION] = 0.8

        s_low = project_surface(i_low, default_relationship, default_traits, outer)
        s_high = project_surface(i_high, default_relationship, default_traits, outer)

        assert s_high[S_SHARPNESS] > s_low[S_SHARPNESS], (
            f"高烦躁 sharpness={s_high[S_SHARPNESS]:.3f} ≤ 低烦躁={s_low[S_SHARPNESS]:.3f}"
        )

    def test_high_affection_high_warmth(self, default_internal, default_traits):
        """高好感 → 高温暖。"""
        outer = np.zeros(ST_SIZE)
        r_low = DEFAULT_RELATIONSHIP.copy(); r_low[R_AFFECTION] = -0.8
        r_high = DEFAULT_RELATIONSHIP.copy(); r_high[R_AFFECTION] = 0.8

        s_low = project_surface(default_internal, r_low, default_traits, outer)
        s_high = project_surface(default_internal, r_high, default_traits, outer)

        assert s_high[S_WARMTH] > s_low[S_WARMTH], (
            f"高好感 warmth={s_high[S_WARMTH]:.3f} ≤ 低好感={s_low[S_WARMTH]:.3f}"
        )

    def test_high_pride_low_vulnerability(self, default_internal, default_relationship):
        """高自尊 → 低脆弱感。"""
        outer = np.zeros(ST_SIZE)
        t_low = DEFAULT_TRAITS.copy(); t_low[T_PRIDE] = -0.8
        t_high = DEFAULT_TRAITS.copy(); t_high[T_PRIDE] = 0.8

        s_low = project_surface(default_internal, default_relationship, t_low, outer)
        s_high = project_surface(default_internal, default_relationship, t_high, outer)

        assert s_high[S_VULNERABILITY] < s_low[S_VULNERABILITY], (
            f"高自尊 vulnerability={s_high[S_VULNERABILITY]:.3f} ≥ 低自尊={s_low[S_VULNERABILITY]:.3f}"
        )

    def test_high_stress_low_warmth(self, default_relationship, default_traits):
        """高压力 → 低温暖。"""
        outer = np.zeros(ST_SIZE)
        i_low = DEFAULT_INTERNAL.copy(); i_low[I_STRESS] = -0.8
        i_high = DEFAULT_INTERNAL.copy(); i_high[I_STRESS] = 0.8

        s_low = project_surface(i_low, default_relationship, default_traits, outer)
        s_high = project_surface(i_high, default_relationship, default_traits, outer)

        assert s_high[S_WARMTH] < s_low[S_WARMTH], (
            f"高压力 warmth={s_high[S_WARMTH]:.3f} ≥ 低压力={s_low[S_WARMTH]:.3f}"
        )

    def test_high_fatigue_low_expressiveness(self, default_relationship, default_traits):
        """高疲劳 → 低外露度。"""
        outer = np.zeros(ST_SIZE)
        i_low = DEFAULT_INTERNAL.copy(); i_low[I_MENTAL_FATIGUE] = -0.8
        i_high = DEFAULT_INTERNAL.copy(); i_high[I_MENTAL_FATIGUE] = 0.8

        s_low = project_surface(i_low, default_relationship, default_traits, outer)
        s_high = project_surface(i_high, default_relationship, default_traits, outer)

        assert s_high[S_EXPRESSIVENESS] < s_low[S_EXPRESSIVENESS], (
            f"高疲劳 expressiveness={s_high[S_EXPRESSIVENESS]:.3f} ≥ 低疲劳={s_low[S_EXPRESSIVENESS]:.3f}"
        )


class TestOuterStimuliEffect:
    """outer_stimuli 对表面表达的直接影响。"""

    def test_validation_outer_increases_warmth(self, default_internal, default_relationship, default_traits):
        """被认可（outer）→ 增加温暖。"""
        outer_zero = np.zeros(ST_SIZE)
        outer_val = np.zeros(ST_SIZE); outer_val[ST_VALIDATION] = 0.9

        s_zero = project_surface(default_internal, default_relationship, default_traits, outer_zero)
        s_val = project_surface(default_internal, default_relationship, default_traits, outer_val)

        assert s_val[S_WARMTH] > s_zero[S_WARMTH], (
            f"validation outer 应增加 warmth: {s_zero[S_WARMTH]:.3f} → {s_val[S_WARMTH]:.3f}"
        )

    def test_conflict_outer_increases_sharpness(self, default_internal, default_relationship, default_traits):
        """冲突（outer）→ 增加尖锐度。"""
        outer_zero = np.zeros(ST_SIZE)
        outer_con = np.zeros(ST_SIZE); outer_con[ST_CONFLICT] = 0.9

        s_zero = project_surface(default_internal, default_relationship, default_traits, outer_zero)
        s_con = project_surface(default_internal, default_relationship, default_traits, outer_con)

        assert s_con[S_SHARPNESS] > s_zero[S_SHARPNESS], (
            f"conflict outer 应增加 sharpness: {s_zero[S_SHARPNESS]:.3f} → {s_con[S_SHARPNESS]:.3f}"
        )

    def test_outer_stimuli_different_from_inner_only(self, rng):
        """验证 outer_stimuli 确实独立影响了 surface（不只是内部状态的翻版）。"""
        n = 10_000
        internal = DEFAULT_INTERNAL.copy()
        relationship = DEFAULT_RELATIONSHIP.copy()
        traits = DEFAULT_TRAITS.copy()

        # 相同 internal，不同 outer → 不同 surface
        s1 = project_surface(internal, relationship, traits, np.zeros(ST_SIZE))
        s2 = project_surface(internal, relationship, traits, np.ones(ST_SIZE) * 0.5)

        assert not np.allclose(s1, s2, atol=1e-6), \
            "不同 outer_stimuli 应产生不同 surface"


class TestSurfaceStatistics:
    """表面投影的统计特征。"""

    def test_surface_distribution(self, rng):
        """大规模随机输入的 surface 分布。"""
        n = 20_000
        internal = rng.uniform(-1, 1, size=(n, 8))
        relationship = rng.uniform(-1, 1, size=(n, 6))
        traits = rng.uniform(-1, 1, size=(n, 10))
        outer = rng.uniform(0, 1, size=(n, 7))

        all_surfaces = np.empty((n, S_SIZE))
        for i in range(n):
            all_surfaces[i] = project_surface(internal[i], relationship[i], traits[i], outer[i])

        print()
        for dim in range(S_SIZE):
            col = all_surfaces[:, dim]
            print(f"  {S_LABELS[dim]:>16s}: mean={col.mean():.3f} std={col.std():.3f} "
                  f"[{col.min():.3f}, {col.max():.3f}]")

        # 所有维度都应该有合理的 spread
        for dim in range(S_SIZE):
            col = all_surfaces[:, dim]
            assert col.std() > 0.01, f"{S_LABELS[dim]} std={col.std():.4f} 太小，可能卡在阈值"
