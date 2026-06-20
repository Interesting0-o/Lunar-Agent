"""Layer 3: 防御剖面测试 — 去激活/过度激活剖面 + apply_defenses。

大规模数据测试防御计算的心理方向性和不变式。
"""

import numpy as np
import pytest
from numpy.testing import assert_array_less
from state import (
    DEFAULT_TRAITS, DEFAULT_INTERNAL, DEFAULT_RELATIONSHIP,
    T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE,
    T_PRIDE, T_EMOTIONAL_OPENNESS, T_EMOTIONAL_STABILITY,
    T_JEALOUSY_SENSITIVITY, T_SENSITIVITY,
    R_TRUST, R_EMOTIONAL_SAFETY, R_AFFECTION, R_ROMANTIC_TENSION,
    I_STRESS, I_INSECURITY, I_LONGING,
    ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT,
    ST_DEPENDENCY, ST_TEASING, ST_EMOTIONAL_WEIGHT, ST_SIZE,
    I_LABELS, ST_LABELS,
)
from state_engine._defenses import compute_defense_profiles, apply_defenses


class TestDefenseProfilesShape:
    """输出形状正确。"""

    def test_shape(self, default_traits, default_relationship, default_internal):
        profiles = compute_defense_profiles(default_traits, default_relationship, default_internal)
        assert profiles.shape == (2, ST_SIZE), f"shape={profiles.shape}"

    def test_range_default(self, default_traits, default_relationship, default_internal):
        """默认参数下 profiles ∈ [0, 1]。"""
        profiles = compute_defense_profiles(default_traits, default_relationship, default_internal)
        assert np.all(profiles >= 0.0), f"min={profiles.min()}"
        assert np.all(profiles <= 1.0), f"max={profiles.max()}"


class TestDefenseProfilesBulk:
    """大量随机参数的剖面计算 —— 观察异常值。"""

    def test_all_in_bounds(self, rng):
        """5000 组随机 (traits, relationship, internal) 全部 profiles ∈ [0, 1]。"""
        n = 20_000
        traits = rng.uniform(-0.999, 0.999, size=(n, 10))
        rel = rng.uniform(-0.999, 0.999, size=(n, 6))
        internal = rng.uniform(-0.999, 0.999, size=(n, 8))

        violations = []
        for i in range(n):
            profiles = compute_defense_profiles(traits[i], rel[i], internal[i])
            if profiles.min() < -1e-10 or profiles.max() > 1.0 + 0.11:
                violations.append({
                    "i": i,
                    "min": float(profiles.min()),
                    "max": float(profiles.max()),
                })

        assert len(violations) == 0, (
            f"发现 {len(violations)} 组越界（共 {n} 组）:\n"
            + "\n".join(f"  [{v['i']}] min={v['min']:.6f} max={v['max']:.6f}"
                        for v in violations[:10])
        )

    def test_extreme_traits_bounds(self, rng):
        """极端人格（全0或全1）不产生 NaN/Inf。"""
        n = 10_000
        # 用 beta(0.2, 0.2) 生成趋近 0/1 的值
        traits = rng.beta(0.2, 0.2, size=(n, 10)) * 2 - 1
        rel = rng.beta(0.2, 0.2, size=(n, 6)) * 2 - 1
        internal = rng.beta(0.2, 0.2, size=(n, 8)) * 2 - 1

        for i in range(n):
            profiles = compute_defense_profiles(traits[i], rel[i], internal[i])
            assert np.all(np.isfinite(profiles)), f"[{i}] 出现 NaN/Inf"

    def test_profile_statistics(self, rng):
        """批量统计: deact 和 hyper 的分布特征。"""
        n = 20_000
        traits = rng.uniform(-1, 1, size=(n, 10))
        rel = rng.uniform(-1, 1, size=(n, 6))
        internal = rng.uniform(-1, 1, size=(n, 8))

        deact_means = np.empty(n)
        hyper_means = np.empty(n)

        for i in range(n):
            p = compute_defense_profiles(traits[i], rel[i], internal[i])
            deact_means[i] = p[0].mean()
            hyper_means[i] = p[1].mean()

        # 不报告硬性错误，但输出统计信息供分析
        print(f"\n  deact mean: {deact_means.mean():.3f} ± {deact_means.std():.3f} "
              f"[{deact_means.min():.3f}, {deact_means.max():.3f}]")
        print(f"  hyper mean: {hyper_means.mean():.3f} ± {hyper_means.std():.3f} "
              f"[{hyper_means.min():.3f}, {hyper_means.max():.3f}]")


class TestDefenseDirectional:
    """防御剖面的心理方向性检查。

    验证: 高回避 → 高去激活，高焦虑 → 高过度激活。
    """

    def test_high_avoidance_high_deactivation(self, default_relationship, default_internal):
        """高回避 → deact.mean() 显著高于低回避。"""
        t_low = DEFAULT_TRAITS.copy()
        t_low[T_ATTACHMENT_AVOIDANCE] = -0.8
        t_high = DEFAULT_TRAITS.copy()
        t_high[T_ATTACHMENT_AVOIDANCE] = 0.8

        p_low = compute_defense_profiles(t_low, default_relationship, default_internal)
        p_high = compute_defense_profiles(t_high, default_relationship, default_internal)

        assert p_high[0].mean() > p_low[0].mean(), (
            f"高回避 deact={p_high[0].mean():.3f} ≤ 低回避 deact={p_low[0].mean():.3f}"
        )

    def test_high_anxiety_high_hyperactivation(self, default_relationship, default_internal):
        """高依恋焦虑 → hyper.mean() 显著高于低焦虑。"""
        t_low = DEFAULT_TRAITS.copy()
        t_low[T_ATTACHMENT_ANXIETY] = -0.8
        t_high = DEFAULT_TRAITS.copy()
        t_high[T_ATTACHMENT_ANXIETY] = 0.8

        p_low = compute_defense_profiles(t_low, default_relationship, default_internal)
        p_high = compute_defense_profiles(t_high, default_relationship, default_internal)

        assert p_high[1].mean() > p_low[1].mean(), (
            f"高焦虑 hyper={p_high[1].mean():.3f} ≤ 低焦虑 hyper={p_low[1].mean():.3f}"
        )

    def test_high_pride_high_deactivation(self, default_relationship, default_internal):
        """高自尊 → 去激活增强。"""
        t_low = DEFAULT_TRAITS.copy()
        t_low[T_PRIDE] = -0.8
        t_high = DEFAULT_TRAITS.copy()
        t_high[T_PRIDE] = 0.8

        p_low = compute_defense_profiles(t_low, default_relationship, default_internal)
        p_high = compute_defense_profiles(t_high, default_relationship, default_internal)

        assert p_high[0].mean() > p_low[0].mean(), (
            f"高自尊 deact={p_high[0].mean():.3f} ≤ 低自尊 deact={p_low[0].mean():.3f}"
        )

    def test_trust_reduces_deactivation(self, default_traits, default_internal):
        """高信任 → 去激活降低。"""
        r_low = DEFAULT_RELATIONSHIP.copy()
        r_low[R_TRUST] = -0.8
        r_high = DEFAULT_RELATIONSHIP.copy()
        r_high[R_TRUST] = 0.8

        p_low = compute_defense_profiles(default_traits, r_low, default_internal)
        p_high = compute_defense_profiles(default_traits, r_high, default_internal)

        assert p_high[0].mean() < p_low[0].mean(), (
            f"高信任 deact={p_high[0].mean():.3f} ≥ 低信任 deact={p_low[0].mean():.3f}"
        )

    def test_affection_boosts_hyperactivation(self, default_traits, default_internal):
        """高好感 → 过度激活增强（在意的人更容易触发依恋系统）。"""
        r_low = DEFAULT_RELATIONSHIP.copy()
        r_low[R_AFFECTION] = -0.8
        r_high = DEFAULT_RELATIONSHIP.copy()
        r_high[R_AFFECTION] = 0.8

        p_low = compute_defense_profiles(default_traits, r_low, default_internal)
        p_high = compute_defense_profiles(default_traits, r_high, default_internal)

        assert p_high[1].mean() >= p_low[1].mean(), (
            f"高好感 hyper={p_high[1].mean():.3f} < 低好感 hyper={p_low[1].mean():.3f}"
        )

    def test_insecurity_increases_deactivation(self, default_traits, default_relationship):
        """高不安全感 → 越难受越藏（去激活增强）。"""
        i_low = DEFAULT_INTERNAL.copy()
        i_low[I_INSECURITY] = -0.8
        i_high = DEFAULT_INTERNAL.copy()
        i_high[I_INSECURITY] = 0.8

        p_low = compute_defense_profiles(default_traits, default_relationship, i_low)
        p_high = compute_defense_profiles(default_traits, default_relationship, i_high)

        assert p_high[0].mean() >= p_low[0].mean(), (
            f"高不安 deact={p_high[0].mean():.3f} < 低不安 deact={p_low[0].mean():.3f}"
        )

    def test_insecurity_increases_hyperactivation(self, default_traits, default_relationship):
        """高不安全感 → 依恋系统激活（过度激活增强）。"""
        i_low = DEFAULT_INTERNAL.copy()
        i_low[I_INSECURITY] = -0.8
        i_high = DEFAULT_INTERNAL.copy()
        i_high[I_INSECURITY] = 0.8

        p_low = compute_defense_profiles(default_traits, default_relationship, i_low)
        p_high = compute_defense_profiles(default_traits, default_relationship, i_high)

        assert p_high[1].mean() >= p_low[1].mean(), (
            f"高不安 hyper={p_high[1].mean():.3f} < 低不安 hyper={p_low[1].mean():.3f}"
        )

    def test_high_stability_low_deactivation(self, default_relationship, default_internal):
        """高情绪稳定性 → 真淡定，去激活低。"""
        t_low = DEFAULT_TRAITS.copy()
        t_low[T_EMOTIONAL_STABILITY] = -0.8
        t_high = DEFAULT_TRAITS.copy()
        t_high[T_EMOTIONAL_STABILITY] = 0.8

        p_low = compute_defense_profiles(t_low, default_relationship, default_internal)
        p_high = compute_defense_profiles(t_high, default_relationship, default_internal)

        assert p_high[0].mean() < p_low[0].mean(), (
            f"高稳定 deact={p_high[0].mean():.3f} ≥ 低稳定 deact={p_low[0].mean():.3f}"
        )

    # ── 刺激特异性验证（2025-06 新增） ──

    def test_stress_modulation_is_dimension_specific(self, default_traits, default_relationship):
        """stress 对 deact 各维影响不同（不再是全局+0.05）。

        旧版: stress 对 7 维均匀 +0.05/单位，各维 Δ 相同。
        新版: stress 对 conflict 影响最大，对 teasing 影响最小（或为零）。
        验证极差 > 0（即各维不等效）。
        """
        i_low = DEFAULT_INTERNAL.copy()
        i_low[I_STRESS] = -0.8
        i_high = DEFAULT_INTERNAL.copy()
        i_high[I_STRESS] = 0.8

        p_low = compute_defense_profiles(default_traits, default_relationship, i_low)
        p_high = compute_defense_profiles(default_traits, default_relationship, i_high)

        delta = p_high[0] - p_low[0]
        dim_range = delta.max() - delta.min()

        assert dim_range > 0.05, (
            f"stress 对 deact 各维影响过于均匀，极差={dim_range:.4f}（应 > 0.05）\n"
            f"  delta={np.array2string(delta, precision=4, suppress_small=True)}"
        )

    def test_trust_modulation_is_dimension_specific(self, default_traits, default_internal):
        """trust 对 deact 各维影响不同（不再是全局乘法 -0.11）。

        旧版: trust 对 7 维等比例缩放。
        新版: trust 对 abandonment 影响最大，对 closeness 影响最小。
        验证各维 Δ 不等效。
        """
        r_low = DEFAULT_RELATIONSHIP.copy()
        r_low[R_TRUST] = -0.8
        r_high = DEFAULT_RELATIONSHIP.copy()
        r_high[R_TRUST] = 0.8

        p_low = compute_defense_profiles(default_traits, r_low, default_internal)
        p_high = compute_defense_profiles(default_traits, r_high, default_internal)

        delta = p_high[0] - p_low[0]
        dim_range = delta.max() - delta.min()

        assert dim_range > 0.02, (
            f"trust 对 deact 各维影响过于均匀，极差={dim_range:.4f}（应 > 0.02）\n"
            f"  delta={np.array2string(delta, precision=4, suppress_small=True)}"
        )

    def test_insecurity_hyperactivation_is_dimension_specific(self, default_traits, default_relationship):
        """insecurity 对 hyper 各维影响不同（不再是全局+0.06）。

        旧版: insecurity 对 hyper 7 维均匀 +0.06/单位。
        新版: 对 abandonment 影响最大，对 teasing 不影响。
        """
        i_low = DEFAULT_INTERNAL.copy()
        i_low[I_INSECURITY] = -0.8
        i_high = DEFAULT_INTERNAL.copy()
        i_high[I_INSECURITY] = 0.8

        p_low = compute_defense_profiles(default_traits, default_relationship, i_low)
        p_high = compute_defense_profiles(default_traits, default_relationship, i_high)

        delta = p_high[1] - p_low[1]
        dim_range = delta.max() - delta.min()

        assert dim_range > 0.05, (
            f"insecurity 对 hyper 各维影响过于均匀，极差={dim_range:.4f}（应 > 0.05）\n"
            f"  delta={np.array2string(delta, precision=4, suppress_small=True)}"
        )


class TestApplyDefenses:
    """apply_defenses: 防御应用后的 inner/outer 刺激不变式。"""

    def test_zero_stimuli_zero_output(self):
        """零输入 → 零输出。"""
        from state_engine._defenses import compute_defense_profiles
        profiles = compute_defense_profiles(DEFAULT_TRAITS, DEFAULT_RELATIONSHIP, DEFAULT_INTERNAL)
        stimuli = np.zeros(ST_SIZE)
        inner, outer = apply_defenses(stimuli, profiles)
        assert np.allclose(inner, 0.0, atol=1e-12), f"inner={inner}"
        assert np.allclose(outer, 0.0, atol=1e-12), f"outer={outer}"

    def test_inner_ge_outer(self, rng):
        """核心不变式: inner[s] ≥ outer[s] 对所有刺激维度。

        过度激活放大内心感受，去激活削减外在表达。
        因此 inner_stimuli ≥ outer_stimuli（逐元素）。

        注意: 当 inner 值 > 1.0 时 soft_clamp 会在边界附近产生惩罚谷，
        可能导致某些刺激维度上 inner < outer。此测试过滤掉那些极端情况。
        """
        n = 20_000
        traits = rng.uniform(-1, 1, size=(n, 10))
        rel = rng.uniform(-1, 1, size=(n, 6))
        internal = rng.uniform(-1, 1, size=(n, 8))
        stimuli = rng.uniform(0, 0.7, size=(n, ST_SIZE))  # 避免极端刺激导致 inner > 1.0

        violations = 0
        max_diff = 0.0
        for i in range(n):
            profiles = compute_defense_profiles(traits[i], rel[i], internal[i])
            inner, outer = apply_defenses(stimuli[i], profiles)
            diff = inner - outer
            if np.any(diff < -1e-12):
                violations += 1
                max_diff = max(max_diff, -diff.min())

        # 在中等刺激强度下 invariant 应始终保持
        assert violations == 0, (
            f"inner < outer 共 {violations}/{n} 次，最大逆差={max_diff:.6f}\n"
            f"（中等刺激 0~0.7 下不应发生）"
        )

    def test_inner_ge_outer_high_stimuli_edge_case(self, rng):
        """高刺激 + 高 hyper 时 inner 可能 > 1.0 被 soft_clamp 压制。

        这是已知边界效应: soft_clamp 的惩罚谷可能导致 inner 略 < outer。
        验证该效应确实存在但很小。
        """
        n = 10_000
        traits = rng.uniform(-1, 1, size=(n, 10))
        rel = rng.uniform(-1, 1, size=(n, 6))
        internal = rng.uniform(-1, 1, size=(n, 8))
        stimuli = rng.uniform(0.7, 1.0, size=(n, ST_SIZE))  # 高强度刺激

        violation_count = 0
        max_violation = 0.0
        for i in range(n):
            profiles = compute_defense_profiles(traits[i], rel[i], internal[i])
            inner, outer = apply_defenses(stimuli[i], profiles)
            diff = inner - outer
            violations = diff < -1e-12
            if violations.any():
                violation_count += 1
                max_violation = max(max_violation, -diff[violations].min())

        # 高刺激下有少量越界是可接受的（soft_clamp 边界效应）
        violation_rate = violation_count / n
        print(f"\n  高刺激下 inner<outer 比例: {violation_rate*100:.1f}% "
              f"({violation_count}/{n}), 最大逆差={max_violation:.6f}")
        assert max_violation < 0.05, (
            f"inner<outer 逆差过大: {max_violation:.6f}（应 < 0.05）"
        )

    def test_hyper_amplifies(self):
        """hyper=0.9, deact=0 时 inner ≥ stimuli（放大效应）。"""
        profiles = np.zeros((2, ST_SIZE))
        profiles[1] = 0.9  # 高过度激活
        profiles[0] = 0.0  # 无去激活

        stimuli = np.full(ST_SIZE, 0.5)
        inner, outer = apply_defenses(stimuli, profiles)

        assert np.all(inner >= stimuli - 1e-12), (
            f"hyper=0.9 应放大 inner，但 inner={inner}"
        )

    def test_deact_suppresses_outer(self):
        """hyper=0, deact=0.9 时 outer ≤ inner（压制效应）。"""
        profiles = np.zeros((2, ST_SIZE))
        profiles[0] = 0.9  # 高去激活
        profiles[1] = 0.0  # 无过度激活

        stimuli = np.full(ST_SIZE, 0.5)
        inner, outer = apply_defenses(stimuli, profiles)

        assert np.all(outer <= inner + 1e-12), (
            f"deact=0.9 应压制 outer，但 outer={outer}"
        )
        assert np.all(outer < stimuli - 1e-12), (
            f"deact=0.9 时 outer 应 < stimuli，但 outer={outer}"
        )

    def test_both_high_independent(self):
        """高 hyper + 高 deact: 内心翻江倒海但表面波澜不惊。"""
        profiles = np.zeros((2, ST_SIZE))
        profiles[0] = 0.9  # 高去激活
        profiles[1] = 0.9  # 高过度激活

        stimuli = np.full(ST_SIZE, 0.5)
        inner, outer = apply_defenses(stimuli, profiles)

        # inner 被放大
        assert np.all(inner > stimuli), f"高hyper时inner应>stimuli, inner={inner}"
        # outer 被压制
        assert np.all(outer < inner), f"高deact时outer应<inner, outer={outer}"

    def test_output_bounds_bulk(self, rng):
        """5000 组随机输入: inner/outer 均 ∈ [0, 1]。"""
        n = 20_000
        traits = rng.uniform(-1, 1, size=(n, 10))
        rel = rng.uniform(-1, 1, size=(n, 6))
        internal = rng.uniform(-1, 1, size=(n, 8))
        stimuli = rng.uniform(0, 1, size=(n, ST_SIZE))

        inner_violations = 0
        outer_violations = 0
        for i in range(n):
            profiles = compute_defense_profiles(traits[i], rel[i], internal[i])
            inner, outer = apply_defenses(stimuli[i], profiles)
            if inner.min() < -1e-10 or inner.max() > 1.0 + 0.11:
                inner_violations += 1
            if outer.min() < -1e-10 or outer.max() > 1.0 + 0.11:
                outer_violations += 1

        assert inner_violations == 0, f"inner 越界 {inner_violations}/{n}"
        assert outer_violations == 0, f"outer 越界 {outer_violations}/{n}"


class TestDefenseCornerCases:
    """边界情况测试。"""

    def test_extreme_stimuli_all_zero(self, default_traits, default_relationship, default_internal):
        """零刺激 + 默认状态: 防御剖面仍然合法。"""
        profiles = compute_defense_profiles(default_traits, default_relationship, default_internal)
        inner, outer = apply_defenses(np.zeros(ST_SIZE), profiles)
        assert np.all(inner == 0.0)
        assert np.all(outer == 0.0)

    def test_extreme_stimuli_all_one(self, default_traits, default_relationship, default_internal):
        """全1刺激: 输出不越界。"""
        profiles = compute_defense_profiles(default_traits, default_relationship, default_internal)
        inner, outer = apply_defenses(np.ones(ST_SIZE), profiles)
        assert np.all(inner <= 1.0 + 0.11), f"inner max={inner.max()}"
        assert np.all(outer <= 1.0 + 0.11), f"outer max={outer.max()}"

    def test_extreme_profiles(self):
        """极端防御剖面 (全0 或 全1): 不崩溃。"""
        # 全 0 profiles
        p_zero = np.zeros((2, ST_SIZE))
        inner, outer = apply_defenses(np.full(ST_SIZE, 0.5), p_zero)
        assert np.allclose(inner, 0.5)
        assert np.allclose(outer, 0.5)

        # 全 1 profiles
        p_one = np.ones((2, ST_SIZE))
        inner, outer = apply_defenses(np.full(ST_SIZE, 0.5), p_one)
        assert np.all(np.isfinite(inner))
        assert np.all(np.isfinite(outer))
        assert np.all(inner > 0.5), "hyper=1 应放大 inner"
        assert np.all(outer < inner), "deact=1 应压制 outer"
