"""Layer 4: 残差动力学测试 — 状态更新、稳态收敛、刺激方向性。

核心测试:
  1. setpoint 计算范围
  2. 稳态收敛（零刺激下多轮迭代）
  3. 刺激方向性（每种刺激对状态维度的正/负影响）
  4. 防御剖面调制速率（hyper 加速, deact 减速）
  5. 大规模随机输入边界
"""

import numpy as np
import pytest
from numpy.testing import assert_array_less
from state import (
    DEFAULT_TRAITS, DEFAULT_INTERNAL, DEFAULT_RELATIONSHIP,
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE, I_SIZE,
    R_AFFECTION, R_TRUST_BOND, R_INTIMACY, R_SIZE,
    ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT,
    ST_DEPENDENCY, ST_TEASING, ST_EMOTIONAL_WEIGHT, ST_SIZE,
    T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE,
    T_EMOTIONAL_STABILITY, T_OPTIMISM, T_ANXIETY_PRONENESS,
    T_ANGER_REACTIVITY, T_PRIDE, T_JEALOUSY_SENSITIVITY,
    T_SIZE,
    I_LABELS, R_LABELS, ST_LABELS,
)
from state_engine._dynamics import (
    compute_setpoint, compute_rel_setpoint,
    update_internal_state, update_relationship_state,
)
from state_engine._defenses import compute_defense_profiles, apply_defenses
from state_engine import update_all


# ═══════════════════════════════════════════════════════════════
# Setpoint 计算
# ═══════════════════════════════════════════════════════════════

class TestSetpoint:
    """setpoint 计算的范围和心理方向。"""

    def test_internal_setpoint_range(self, rng):
        """大量随机人格: compute_setpoint ∈ [-0.9, 0.9]."""
        n = 20_000
        traits = rng.uniform(-1, 1, size=(n, 10))

        violations = []
        for i in range(n):
            sp = compute_setpoint(traits[i])
            mn, mx = sp.min(), sp.max()
            if mn < -0.91 or mx > 0.91:
                violations.append((i, float(mn), float(mx)))

        assert len(violations) == 0, (
            f"setpoint 越界 {len(violations)}/{n}:\n"
            + "\n".join(f"  [{i}] [{mn:.4f}, {mx:.4f}]" for i, mn, mx in violations[:10])
        )

    def test_rel_setpoint_range(self, rng):
        """大量随机人格: compute_rel_setpoint ∈ [-0.96, 0.96]。"""
        n = 20_000
        traits = rng.uniform(-1, 1, size=(n, 10))

        violations = []
        for i in range(n):
            sp = compute_rel_setpoint(traits[i])
            mn, mx = sp.min(), sp.max()
            if mn < -0.97 or mx > 0.97:
                violations.append((i, float(mn), float(mx)))

        assert len(violations) == 0, (
            f"rel_setpoint 越界 {len(violations)}/{n}"
        )

    def test_high_anxiety_higher_stress_setpoint(self):
        """高焦虑 → stress setpoint 更高。"""
        t_low = DEFAULT_TRAITS.copy()
        t_low[T_ANXIETY_PRONENESS] = -0.8
        t_high = DEFAULT_TRAITS.copy()
        t_high[T_ANXIETY_PRONENESS] = 0.8

        sp_low = compute_setpoint(t_low)
        sp_high = compute_setpoint(t_high)
        assert sp_high[I_STRESS] > sp_low[I_STRESS], (
            f"高焦虑 stress sp={sp_high[I_STRESS]:.3f} ≤ 低焦虑={sp_low[I_STRESS]:.3f}"
        )

    def test_high_avoidance_lower_trust_setpoint(self):
        """高回避 → trust setpoint 更低。"""
        t_low = DEFAULT_TRAITS.copy()
        t_low[T_ATTACHMENT_AVOIDANCE] = -0.8
        t_high = DEFAULT_TRAITS.copy()
        t_high[T_ATTACHMENT_AVOIDANCE] = 0.8

        sp_low = compute_rel_setpoint(t_low)
        sp_high = compute_rel_setpoint(t_high)
        assert sp_high[R_TRUST_BOND] < sp_low[R_TRUST_BOND], (
            f"高回避 trust sp={sp_high[R_TRUST_BOND]:.3f} ≥ 低回避={sp_low[R_TRUST_BOND]:.3f}"
        )

    def test_default_setpoint_finite(self, default_traits):
        """默认人格的 setpoint 全部有限。"""
        sp = compute_setpoint(default_traits)
        assert np.all(np.isfinite(sp))
        rsp = compute_rel_setpoint(default_traits)
        assert np.all(np.isfinite(rsp))


# ═══════════════════════════════════════════════════════════════
# 稳定性: 零刺激下收敛到耦合平衡（非 setpoint）
# ═══════════════════════════════════════════════════════════════

class TestConvergence:
    """零刺激下状态应收敛到耦合平衡点（A·h=h, 即 h→0）。

    注意: 与旧设计不同，稳态恢复（拉到 setpoint）已移至 _decay.py
    的时间衰减。每轮动态中，零刺激下状态只受耦合矩阵驱动。
    ρ(A)=0.95<1，因此 h 自然收敛到 0，而非 setpoint。
    这是设计意图——见 _dynamics.py 文档和测试报告 2.2/2.3。
    """

    def test_internal_converges_stable(self, default_traits, default_relationship):
        """零刺激下迭代 2000 轮 → 状态稳定在耦合平衡点（近 0 而非 setpoint）。

        验证: 最终 100 步的 L2 变化极小（已到达不动点）。
        """
        current = np.full(I_SIZE, -0.4)
        profiles = np.zeros((2, ST_SIZE))
        zero_stimuli = np.zeros(ST_SIZE)

        for _ in range(1900):
            current = update_internal_state(
                current, zero_stimuli, default_traits, default_relationship, profiles,
            )

        # 再跑 100 步看是否已稳定
        final_vals = []
        for _ in range(100):
            current = update_internal_state(
                current, zero_stimuli, default_traits, default_relationship, profiles,
            )
            final_vals.append(current.copy())

        final_arr = np.array(final_vals)
        step_diffs = np.abs(np.diff(final_arr, axis=0))
        assert np.max(step_diffs) < 0.001, (
            f"零刺激下未收敛到稳定点: 最大步变化={np.max(step_diffs):.6f}"
        )
        # 收敛点应接近 0（耦合矩阵的不动点）
        assert np.linalg.norm(current) < 0.5, (
            f"零刺激下收敛点远离0: L2={np.linalg.norm(current):.4f}"
        )

    def test_internal_converges_from_extremes(self, default_traits, default_relationship):
        """从不同极端状态出发都收敛到同一稳定区域。

        验证: 不同起始点都应收敛到相近的 L2 范围。
        """
        profiles = np.zeros((2, ST_SIZE))
        zero_stimuli = np.zeros(ST_SIZE)

        starts = [
            np.full(I_SIZE, -0.9),
            np.full(I_SIZE, 0.9),
            np.full(I_SIZE, 0.0),
            np.full(I_SIZE, -0.5),
        ]

        final_l2s = []
        for j, start in enumerate(starts):
            current = start.copy()
            for _ in range(2000):
                current = update_internal_state(
                    current, zero_stimuli, default_traits, default_relationship, profiles,
                )
            final_l2s.append(np.linalg.norm(current))

        # 所有起点应收敛到相近的 L2 区域
        l2_arr = np.array(final_l2s)
        assert l2_arr.max() - l2_arr.min() < 0.3, (
            f"不同起点收敛到不同区域: L2={final_l2s}"
        )

    def test_relationship_converges_stable(self, default_traits):
        """零刺激下关系状态收敛到耦合平衡点。

        关系状态从任意起点收敛到近 0。
        """
        current = np.full(R_SIZE, 0.5)
        zero_stimuli = np.zeros(ST_SIZE)

        for _ in range(2900):
            current = update_relationship_state(
                current, zero_stimuli, default_traits,
            )

        # 再跑 100 步验证已稳定
        for _ in range(100):
            nxt = update_relationship_state(
                current.copy(), zero_stimuli, default_traits,
            )
            step_diff = np.max(np.abs(nxt - current))
            assert step_diff < 0.001, f"关系状态未收敛: 步变化={step_diff:.6f}"
            current = nxt

    def test_zero_stimulus_no_divergence(self, default_traits, default_relationship):
        """零刺激下 2000 轮不出现 NaN/Inf，状态不越界。"""
        current = np.full(I_SIZE, -0.6)
        profiles = np.zeros((2, ST_SIZE))
        zero_stimuli = np.zeros(ST_SIZE)

        for _ in range(2000):
            current = update_internal_state(
                current, zero_stimuli, default_traits, default_relationship, profiles,
            )
            assert np.all(np.isfinite(current)), "出现 NaN/Inf"
            assert current.min() >= -1.1 and current.max() <= 1.1, \
                f"状态越界: [{current.min():.4f}, {current.max():.4f}]"


# ═══════════════════════════════════════════════════════════════
# 刺激方向性: 每种刺激对状态的正/负影响
# ═══════════════════════════════════════════════════════════════

class TestStimulusDirectionality:
    """验证每种心理刺激对状态维度的方向性影响符合心理学常识。

    使用中性防御剖面 (profiles=0) 以隔离刺激的纯粹方向性效应。
    默认防御剖面会偏转甚至反转刺激效果。
    """

    @pytest.fixture(autouse=True)
    def setup(self, default_traits, default_relationship, default_internal):
        self.traits = default_traits
        self.relationship = default_relationship
        self.internal = default_internal
        # 使用中性防御以隔离刺激效应
        self.neutral_profiles = np.zeros((2, ST_SIZE))

    def _apply_stimulus(self, stim_array: np.ndarray):
        """应用刺激并返回内部状态变化量（中性防御）。"""
        inner, _ = apply_defenses(stim_array, self.neutral_profiles)
        new_internal = update_internal_state(
            self.internal, inner, self.traits, self.relationship, self.neutral_profiles,
        )
        return new_internal - self.internal

    def _apply_rel_stimulus(self, stim_array: np.ndarray):
        """应用刺激并返回关系状态变化量。"""
        inner, _ = apply_defenses(stim_array, self.neutral_profiles)
        new_rel = update_relationship_state(
            self.relationship, inner, self.traits,
        )
        return new_rel - self.relationship

    # ── 内部状态（直接验证 B 矩阵方向） ──

    def test_abandonment_increases_insecurity(self):
        """B[ST_ABANDONMENT, I_INSECURITY] > 0 — 被抛弃增加不安全感。"""
        from state_engine._matrices import INPUT_INFLUENCE_B
        assert INPUT_INFLUENCE_B[ST_ABANDONMENT, I_INSECURITY] > 0
        assert INPUT_INFLUENCE_B[ST_ABANDONMENT, I_LONELINESS] > 0

    def test_validation_reduces_insecurity(self):
        """B[ST_VALIDATION, I_INSECURITY] < 0 — 被认可减少不安全感。"""
        from state_engine._matrices import INPUT_INFLUENCE_B
        assert INPUT_INFLUENCE_B[ST_VALIDATION, I_INSECURITY] < 0

    def test_closeness_reduces_loneliness(self):
        from state_engine._matrices import INPUT_INFLUENCE_B
        assert INPUT_INFLUENCE_B[ST_CLOSENESS, I_LONELINESS] < 0

    def test_conflict_increases_stress(self):
        from state_engine._matrices import INPUT_INFLUENCE_B
        assert INPUT_INFLUENCE_B[ST_CONFLICT, I_STRESS] > 0
        assert INPUT_INFLUENCE_B[ST_CONFLICT, I_IRRITATION] > 0
        assert INPUT_INFLUENCE_B[ST_CONFLICT, I_ENERGY] < 0

    def test_dependency_reduces_loneliness(self):
        from state_engine._matrices import INPUT_INFLUENCE_B
        assert INPUT_INFLUENCE_B[ST_DEPENDENCY, I_LONELINESS] < 0

    def test_emotional_weight_increases_stress(self):
        from state_engine._matrices import INPUT_INFLUENCE_B
        assert INPUT_INFLUENCE_B[ST_EMOTIONAL_WEIGHT, I_STRESS] > 0
        assert INPUT_INFLUENCE_B[ST_EMOTIONAL_WEIGHT, I_MENTAL_FATIGUE] > 0

    # ── 关系状态（B_rel） ──

    def test_rel_abandonment_reduces_trust_bond(self):
        """验证新显式命名规则: abandonment → trust_bond--."""
        from state_engine._dynamics import update_relationship_state
        from state_engine._defenses import apply_defenses
        curr = DEFAULT_RELATIONSHIP.copy()
        stimuli = np.zeros(ST_SIZE)
        stimuli[ST_ABANDONMENT] = 0.8
        traits = DEFAULT_TRAITS.copy()
        inner_s, _ = apply_defenses(stimuli, np.ones((2, ST_SIZE)) * 0.3)
        result = update_relationship_state(curr, inner_s, traits)
        assert result[R_TRUST_BOND] < curr[R_TRUST_BOND], \
            "abandonment 应减少信任纽带"

    def test_rel_validation_increases_affection(self):
        """验证新显式命名规则: validation → affection++."""
        from state_engine._dynamics import update_relationship_state
        from state_engine._defenses import apply_defenses
        curr = DEFAULT_RELATIONSHIP.copy()
        stimuli = np.zeros(ST_SIZE)
        stimuli[ST_VALIDATION] = 0.8
        traits = DEFAULT_TRAITS.copy()
        inner_s, _ = apply_defenses(stimuli, np.ones((2, ST_SIZE)) * 0.3)
        result = update_relationship_state(curr, inner_s, traits)
        assert result[R_AFFECTION] > curr[R_AFFECTION], \
            "validation 应增加好感度"

    def test_rel_closeness_increases_intimacy(self):
        """验证新显式命名规则: closeness → intimacy++."""
        from state_engine._dynamics import update_relationship_state
        from state_engine._defenses import apply_defenses
        curr = DEFAULT_RELATIONSHIP.copy()
        stimuli = np.zeros(ST_SIZE)
        stimuli[ST_CLOSENESS] = 0.8
        traits = DEFAULT_TRAITS.copy()
        inner_s, _ = apply_defenses(stimuli, np.ones((2, ST_SIZE)) * 0.3)
        result = update_relationship_state(curr, inner_s, traits)
        assert result[R_INTIMACY] > curr[R_INTIMACY], \
            "closeness 应增加亲密张力"

    def test_rel_conflict_reduces_trust_bond(self):
        """验证新显式命名规则: conflict → trust_bond--."""
        from state_engine._dynamics import update_relationship_state
        from state_engine._defenses import apply_defenses
        curr = DEFAULT_RELATIONSHIP.copy()
        stimuli = np.zeros(ST_SIZE)
        stimuli[ST_CONFLICT] = 0.8
        traits = DEFAULT_TRAITS.copy()
        inner_s, _ = apply_defenses(stimuli, np.ones((2, ST_SIZE)) * 0.3)
        result = update_relationship_state(curr, inner_s, traits)
        assert result[R_TRUST_BOND] < curr[R_TRUST_BOND], \
            "conflict 应减少信任纽带"


# ═══════════════════════════════════════════════════════════════
# 防御剖面调制速率
# ═══════════════════════════════════════════════════════════════

class TestDefenseRateModulation:
    """防御剖面控制变化速率（β, γ），而非状态比例。"""

    def test_hyper_increases_stimulus_acceptance(self, default_traits, default_relationship):
        """hyper=0.9 → β 更大，同样的刺激产生更大的状态变化。"""
        current = DEFAULT_INTERNAL.copy()
        stimuli = np.zeros(ST_SIZE)
        stimuli[ST_ABANDONMENT] = 0.7

        # 高 hyper, 低 deact
        p_high = np.zeros((2, ST_SIZE))
        p_high[1] = 0.9
        p_high[0] = 0.1
        inner_h, _ = apply_defenses(stimuli, p_high)
        new_h = update_internal_state(
            current, inner_h, default_traits, default_relationship, p_high,
        )

        # 低 hyper, 低 deact
        p_low = np.zeros((2, ST_SIZE))
        p_low[1] = 0.1
        p_low[0] = 0.1
        inner_l, _ = apply_defenses(stimuli, p_low)
        new_l = update_internal_state(
            current, inner_l, default_traits, default_relationship, p_low,
        )

        # 高 hyper → 更大的 absolute change
        change_h = np.abs(new_h - current).sum()
        change_l = np.abs(new_l - current).sum()
        assert change_h > change_l, (
            f"高 hyper 应有更大的变化: hyper=0.9 Δ={change_h:.4f} ≤ hyper=0.1 Δ={change_l:.4f}"
        )

    def test_deact_reduces_stimulus_response(self, default_traits, default_relationship):
        """deact 降低刺激接受速率 β。

        旧测试检验 deact→γ 的调制已被移除。
        新设计: defense profiles 仅调制 β（刺激接受速率），不调制 γ。
        hyper 高 → 刺激响应更大，deact 高 → 刺激响应更小。
        """
        setpoint = compute_setpoint(default_traits)
        current = setpoint.copy()  # 从 setpoint 出发
        stim = np.zeros(ST_SIZE); stim[ST_CONFLICT] = 0.5

        # 高 deact + 低 hyper — 刺激响应应较小
        p_high_deact = np.zeros((2, ST_SIZE))
        p_high_deact[0] = 0.9  # deact
        p_high_deact[1] = 0.1  # hyper
        new_high = update_internal_state(
            current.copy(), stim, default_traits, default_relationship, p_high_deact,
        )

        # 低 deact + 高 hyper — 刺激响应应较大
        p_low_deact = np.zeros((2, ST_SIZE))
        p_low_deact[0] = 0.1  # deact
        p_low_deact[1] = 0.9  # hyper

        new_low = update_internal_state(
            current.copy(), stim, default_traits, default_relationship, p_low_deact,
        )

        # 低 deact 组应偏离 setpoint 更多（β 更大，刺激影响更强）
        dev_high = np.linalg.norm(new_high - setpoint)
        dev_low = np.linalg.norm(new_low - setpoint)
        assert dev_low > dev_high, (
            f"高 hyper+低 deact 应响应更强: deact=0.9 dev={dev_high:.4f}, "
            f"deact=0.1 dev={dev_low:.4f}"
        )


# ═══════════════════════════════════════════════════════════════
# 大规模随机测试
# ═══════════════════════════════════════════════════════════════

class TestDynamicsBulk:
    """大量随机输入: 所有输出必须在 [-1, 1]，不产生 NaN。"""

    def test_internal_update_bounds(self, rng):
        """5000 组随机输入: update_internal_state 输出 ∈ [-1, 1]。"""
        n = 20_000
        traits = rng.uniform(-1, 1, size=(n, 10))
        rel = rng.uniform(-1, 1, size=(n, R_SIZE))
        current = rng.uniform(-1, 1, size=(n, 8))
        stimuli = rng.uniform(0, 1, size=(n, 7))
        # profiles 也用随机值（但由 traits 决定，这里直接随机化）
        deact_random = rng.uniform(0, 1, size=(n, 7))
        hyper_random = rng.uniform(0, 1, size=(n, 7))

        violations = 0
        nan_count = 0
        for i in range(n):
            profiles = np.stack([deact_random[i], hyper_random[i]])
            new_internal = update_internal_state(
                current[i], stimuli[i], traits[i], rel[i], profiles,
            )
            if not np.all(np.isfinite(new_internal)):
                nan_count += 1
            elif new_internal.min() < -1.0 - 0.11 or new_internal.max() > 1.0 + 0.11:
                violations += 1

        assert nan_count == 0, f"NaN/Inf: {nan_count}/{n}"
        assert violations == 0, f"越界: {violations}/{n}"

    def test_relationship_update_bounds(self, rng):
        """5000 组随机输入: update_relationship_state 输出 ∈ [-1, 1]。"""
        n = 20_000
        traits = rng.uniform(-1, 1, size=(n, 10))
        current = rng.uniform(-1, 1, size=(n, R_SIZE))
        stimuli = rng.uniform(0, 1, size=(n, 7))

        violations = 0
        nan_count = 0
        for i in range(n):
            new_rel = update_relationship_state(current[i], stimuli[i], traits[i])
            if not np.all(np.isfinite(new_rel)):
                nan_count += 1
            elif new_rel.min() < -1.0 - 0.11 or new_rel.max() > 1.0 + 0.11:
                violations += 1

        assert nan_count == 0, f"NaN/Inf: {nan_count}/{n}"
        assert violations == 0, f"越界: {violations}/{n}"

    def test_time_scale_separation(self, default_traits, default_relationship, default_internal):
        """关系状态变化幅度 < 内部状态变化幅度（同一轮）。"""
        stimuli = np.ones(ST_SIZE) * 0.5
        profiles = compute_defense_profiles(default_traits, default_relationship, default_internal)
        inner, _ = apply_defenses(stimuli, profiles)

        new_internal = update_internal_state(
            default_internal, inner, default_traits, default_relationship, profiles,
        )
        new_relationship = update_relationship_state(
            default_relationship, inner, default_traits,
        )

        internal_change = np.abs(new_internal - default_internal).max()
        rel_change = np.abs(new_relationship - default_relationship).max()

        assert rel_change < internal_change, (
            f"关系变化幅度({rel_change:.6f})应 < 内部变化幅度({internal_change:.6f})"
        )

    def test_change_magnitudes_statistics(self, rng):
        """大规模统计: 单轮变化幅度的分布。"""
        n = 15_000
        traits = rng.uniform(-1, 1, size=(n, 10))
        rel = rng.uniform(-1, 1, size=(n, R_SIZE))
        current = rng.uniform(-1, 1, size=(n, 8))
        stimuli = rng.uniform(0, 1, size=(n, 7))

        max_changes = np.empty(n)
        for i in range(n):
            profiles = np.zeros((2, ST_SIZE))  # 中性防御
            new_internal = update_internal_state(
                current[i], stimuli[i], traits[i], rel[i], profiles,
            )
            max_changes[i] = np.abs(new_internal - current[i]).max()

        print(f"\n  内部状态单轮最大变化: "
              f"mean={max_changes.mean():.4f} std={max_changes.std():.4f} "
              f"[{max_changes.min():.4f}, {max_changes.max():.4f}]")

        # 不应有单轮变化 > 0.5（过于剧烈）
        extreme = max_changes > 0.5
        assert extreme.sum() <= 5, (
            f"异常剧烈变化: {extreme.sum()} 组 > 0.5 (max={max_changes.max():.4f})"
        )


class TestLongConvergenceStress:
    """长时间收敛压力测试 —— 验证系统在数千轮迭代后的稳定性。"""

    def test_thousand_rounds_no_nan(self, default_traits, default_internal, default_relationship):
        """1000 轮混合刺激后全部有限。"""
        rng = np.random.default_rng(12345)
        current_internal = default_internal.copy()
        current_rel = default_relationship.copy()

        for i in range(1000):
            stim = rng.uniform(0, 1, size=ST_SIZE)
            result = update_internal_state(
                current_internal, stim, default_traits, current_rel,
                np.zeros((2, ST_SIZE)),
            )
            assert np.all(np.isfinite(result)), f"第{i}轮内部 NaN"
            current_internal = result

            current_rel = update_relationship_state(current_rel, stim, default_traits)
            assert np.all(np.isfinite(current_rel)), f"第{i}轮关系 NaN"

    def test_alternating_scenarios_no_divergence(self, default_traits, default_internal, default_relationship):
        """交替极端场景 500 轮不发散。"""
        rng = np.random.default_rng(67890)
        current_internal = default_internal.copy()
        current_rel = default_relationship.copy()

        history_l2 = []

        for i in range(500):
            # 交替正面和负面刺激
            if i % 2 == 0:
                stim = np.zeros(ST_SIZE)
                stim[ST_CONFLICT] = rng.uniform(0.5, 1.0)
                stim[ST_ABANDONMENT] = rng.uniform(0.3, 0.7)
            else:
                stim = np.zeros(ST_SIZE)
                stim[ST_VALIDATION] = rng.uniform(0.5, 1.0)
                stim[ST_CLOSENESS] = rng.uniform(0.3, 0.7)

            result = update_all(current_internal, current_rel, default_traits, stim)
            current_internal = result["internal_state"]
            current_rel = result["relationship_state"]

            history_l2.append(np.linalg.norm(current_internal))
            history_l2.append(np.linalg.norm(current_rel))

            assert current_internal.min() >= -1.0 - 0.11  # soft_clamp 过渡区允许 ±1.1
            assert current_internal.max() <= 1.0 + 0.11
            assert current_rel.min() >= -1.0 - 0.11
            assert current_rel.max() <= 1.0 + 0.11

        # L2 范数不应发散
        l2_arr = np.array(history_l2)
        assert l2_arr.max() < 5.0, f"状态 L2 范数过大: {l2_arr.max():.2f}"
        print(f"\n  500轮交替场景: internal L2 ∈ [{l2_arr[::2].min():.2f}, {l2_arr[::2].max():.2f}], "
              f"rel L2 ∈ [{l2_arr[1::2].min():.2f}, {l2_arr[1::2].max():.2f}]")


# ═══════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════

def soft_clamp_s(x):
    """简单 clamp 到 [-1, 1]。"""
    return np.clip(x, -1.0, 1.0)


# ═══════════════════════════════════════════════════════════════
# 单轮变化率硬上限
# ═══════════════════════════════════════════════════════════════

class TestPerTurnRateLimit:
    """验证每轮状态变化不超过安全上限。

    上限基于 500k 极端随机采样的实测 max 并留 50% 余量。
    这确保任何重构不引入异常跳变。
    """

    # 内部状态上限（含 50% 安全余量）
    INTERNAL_LIMITS = {
        "energy": 0.19, "stress": 0.36, "loneliness": 0.33,
        "insecurity": 0.28, "irritation": 0.29, "longing": 0.26,
        "social_battery": 0.21, "mental_fatigue": 0.25,
    }
    # 关系状态上限
    RELATIONSHIP_LIMITS = {
        "affection": 0.04, "trust_bond": 0.05, "intimacy": 0.04,
    }

    def test_internal_rate_limits(self, rng):
        """20,000 组随机输入，内部状态每轮变化不超过上限。"""
        n = 20_000
        traits_batch = rng.uniform(-1, 1, (n, T_SIZE))
        internal_batch = rng.uniform(-1, 1, (n, I_SIZE))
        rel_batch = rng.uniform(-1, 1, (n, R_SIZE))
        stimuli_batch = rng.beta(0.5, 0.5, (n, ST_SIZE))

        max_deltas = np.zeros(I_SIZE)
        for i in range(n):
            profiles = compute_defense_profiles(
                traits_batch[i], rel_batch[i], internal_batch[i])
            inner, _ = apply_defenses(stimuli_batch[i], profiles)
            new_int = update_internal_state(
                internal_batch[i], inner, traits_batch[i],
                rel_batch[i], profiles)
            delta = np.abs(new_int - internal_batch[i])
            np.maximum(max_deltas, delta, out=max_deltas)

        for j, name in enumerate(I_LABELS):
            limit = self.INTERNAL_LIMITS[name]
            assert max_deltas[j] <= limit, (
                f"|Δ{name}| = {max_deltas[j]:.4f} > {limit}")

    def test_relationship_rate_limits(self, rng):
        """20,000 组随机输入，关系状态每轮变化不超过上限。"""
        n = 20_000
        traits_batch = rng.uniform(-1, 1, (n, T_SIZE))
        internal_batch = rng.uniform(-1, 1, (n, I_SIZE))
        rel_batch = rng.uniform(-1, 1, (n, R_SIZE))
        stimuli_batch = rng.beta(0.5, 0.5, (n, ST_SIZE))

        max_deltas = np.zeros(R_SIZE)
        for i in range(n):
            profiles = compute_defense_profiles(
                traits_batch[i], rel_batch[i], internal_batch[i])
            inner, _ = apply_defenses(stimuli_batch[i], profiles)
            new_rel = update_relationship_state(
                rel_batch[i], inner, traits_batch[i],
                current_internal=internal_batch[i])
            delta = np.abs(new_rel - rel_batch[i])
            np.maximum(max_deltas, delta, out=max_deltas)

        for j, name in enumerate(R_LABELS):
            limit = self.RELATIONSHIP_LIMITS[name]
            assert max_deltas[j] <= limit, (
                f"|Δ{name}| = {max_deltas[j]:.4f} > {limit}")

    def test_time_scale_separation(self, rng):
        """关系态变化速度比内部态慢 2 倍以上（α/α_rel 保证）。"""
        n = 10_000
        alpha_vals = np.zeros(n)
        alpha_rel_vals = np.zeros(n)
        for i in range(n):
            t = rng.uniform(-1, 1, T_SIZE)
            rel = rng.uniform(-1, 1, R_SIZE)
            from state_engine._dynamics_weights import ALPHA_MAPPER, ALPHA_REL_MAPPER
            alpha = ALPHA_MAPPER.compute(np.concatenate([t, rel]))[0]
            alpha_rel = ALPHA_REL_MAPPER.compute(np.concatenate([t, rel]))[0]
            alpha_vals[i] = np.clip(alpha, 0.02, 0.35)
            alpha_rel_vals[i] = np.clip(alpha_rel, 0.005, 0.06)

        ratio = alpha_vals.mean() / alpha_rel_vals.mean()
        assert ratio >= 2.0, (
            f"α/α_rel = {ratio:.1f}x < 2.0x —— 时间尺度分离不足")


# ═══════════════════════════════════════════════════════════════
# 刺激强度单调性
# ═══════════════════════════════════════════════════════════════

class TestStimulusIntensityMonotonicity:
    """刺激强度增加 → 受影响的状态维度单调变化。

    只测试 B 矩阵有直接连接的路径，耦合间接路径可能因交互非单调。
    B 矩阵有 16 条非零连接。
    """

    # B_int 矩阵的非零连接（来自 _matrices.py）
    B_PATHS = [
        (ST_ABANDONMENT, I_INSECURITY),
        (ST_ABANDONMENT, I_LONELINESS),
        (ST_ABANDONMENT, I_LONGING),
        (ST_VALIDATION, I_ENERGY),
        (ST_VALIDATION, I_INSECURITY),
        (ST_CLOSENESS, I_ENERGY),
        (ST_CLOSENESS, I_LONELINESS),
        (ST_CLOSENESS, I_SOCIAL_BATTERY),
        (ST_CONFLICT, I_STRESS),
        (ST_CONFLICT, I_IRRITATION),
        (ST_CONFLICT, I_ENERGY),
        (ST_DEPENDENCY, I_SOCIAL_BATTERY),
        (ST_DEPENDENCY, I_LONELINESS),
        (ST_TEASING, I_IRRITATION),
        (ST_EMOTIONAL_WEIGHT, I_STRESS),
        (ST_EMOTIONAL_WEIGHT, I_MENTAL_FATIGUE),
    ]

    @pytest.mark.parametrize("stim_idx,state_idx", B_PATHS)
    def test_internal_monotonic(self, stim_idx, state_idx, rng):
        """刺激强度单调 → 内部状态变化单调（B 矩阵直连）。"""
        intensities = np.linspace(0.0, 1.0, 15)
        deltas = np.zeros(15)

        traits = rng.uniform(-1, 1, T_SIZE)
        internal = rng.uniform(-1, 1, I_SIZE)
        rel = rng.uniform(-1, 1, R_SIZE)

        for k, intensity in enumerate(intensities):
            stimuli = np.zeros(ST_SIZE)
            stimuli[stim_idx] = intensity
            profiles = compute_defense_profiles(traits, rel, internal)
            inner, _ = apply_defenses(stimuli, profiles)
            new_int = update_internal_state(internal, inner, traits, rel, profiles)
            deltas[k] = new_int[state_idx] - internal[state_idx]

        # 总效应必须单调（非零 diff 同号）
        diffs = np.diff(deltas)
        signs = np.sign(diffs)
        nonzero = signs[signs != 0]
        if len(nonzero) > 0:
            assert np.all(nonzero == nonzero[0]), (
                f"{ST_LABELS[stim_idx]} → {I_LABELS[state_idx]}: 非单调")

    def test_internal_monotonic_bulk(self, rng):
        """散弹测试：对所有路径多组随机 trait 验证单调性。"""
        for stim_idx, state_idx in self.B_PATHS:
            for _ in range(10):
                intensities = np.linspace(0.0, 1.0, 10)
                deltas = np.zeros(10)
                traits = rng.uniform(-1, 1, T_SIZE)
                internal = rng.uniform(-1, 1, I_SIZE)
                rel = rng.uniform(-1, 1, R_SIZE)
                for k, intensity in enumerate(intensities):
                    stimuli = np.zeros(ST_SIZE)
                    stimuli[stim_idx] = intensity
                    profiles = compute_defense_profiles(traits, rel, internal)
                    inner, _ = apply_defenses(stimuli, profiles)
                    new_int = update_internal_state(internal, inner, traits, rel, profiles)
                    deltas[k] = new_int[state_idx] - internal[state_idx]
                diffs = np.diff(deltas)
                signs = np.sign(diffs)
                nonzero = signs[signs != 0]
                if len(nonzero) > 0:
                    assert np.all(nonzero == nonzero[0]), (
                        f"{ST_LABELS[stim_idx]}→{I_LABELS[state_idx]}: buld 非单调")

    def test_zero_stimulus_no_change(self):
        """零刺激 → 状态变化极小（仅耦合驱动）。"""
        internal = DEFAULT_INTERNAL.copy()
        for _ in range(50):
            profiles = compute_defense_profiles(
                DEFAULT_TRAITS, DEFAULT_RELATIONSHIP, internal)
            inner, _ = apply_defenses(np.zeros(ST_SIZE), profiles)
            new_int = update_internal_state(
                internal, inner, DEFAULT_TRAITS,
                DEFAULT_RELATIONSHIP, profiles)
            delta = np.max(np.abs(new_int - internal))
            internal = new_int
        assert delta < 0.15, f"零刺激下变化过大: {delta:.4f}"


# ═══════════════════════════════════════════════════════════════
# 双刺激交互边界
# ═══════════════════════════════════════════════════════════════

class TestStimulusInteraction:
    """验证双刺激交互的心理方向正确性。

    交互效应是期望的（如 validation 补偿 abandonment），
    而非线性 soft_clamp 导致的小幅非加性是可接受的。
    """

    def test_validation_compensates_abandonment(self, rng):
        """validation 缓解 abandonment 引起的不安。"""
        for _ in range(30):
            t = rng.uniform(-1, 1, T_SIZE)
            internal = rng.uniform(-0.5, 0.5, I_SIZE)  # 远离边界避免 soft_clamp
            rel = rng.uniform(-0.5, 0.5, R_SIZE)

            # abandonment 单独
            sa = np.zeros(ST_SIZE); sa[ST_ABANDONMENT] = 0.7
            pa = compute_defense_profiles(t, rel, internal)
            ia, _ = apply_defenses(sa, pa)
            new_a = update_internal_state(internal, ia, t, rel, pa)

            # abandonment + validation 联合
            sj = np.zeros(ST_SIZE); sj[ST_ABANDONMENT] = 0.7; sj[ST_VALIDATION] = 0.5
            pj = compute_defense_profiles(t, rel, internal)
            ij, _ = apply_defenses(sj, pj)
            new_j = update_internal_state(internal, ij, t, rel, pj)

            # validation 应缓解不安
            insecurity_a = new_a[I_INSECURITY] - internal[I_INSECURITY]
            insecurity_j = new_j[I_INSECURITY] - internal[I_INSECURITY]
            assert insecurity_j <= insecurity_a + 0.02, (
                f"validation 未缓解不安: 单独={insecurity_a:.4f}, 联合={insecurity_j:.4f}")

    def test_abandonment_worsens_conflict_stress(self, rng):
        """abandonment + conflict 联合压力不弱于单独 conflict。"""
        for _ in range(30):
            t = rng.uniform(-1, 1, T_SIZE)
            internal = rng.uniform(-0.5, 0.5, I_SIZE)
            rel = rng.uniform(-0.5, 0.5, R_SIZE)

            sc = np.zeros(ST_SIZE); sc[ST_CONFLICT] = 0.7
            pc = compute_defense_profiles(t, rel, internal)
            ic, _ = apply_defenses(sc, pc)
            new_c = update_internal_state(internal, ic, t, rel, pc)

            sj = np.zeros(ST_SIZE); sj[ST_CONFLICT] = 0.7; sj[ST_ABANDONMENT] = 0.5
            pj = compute_defense_profiles(t, rel, internal)
            ij, _ = apply_defenses(sj, pj)
            new_j = update_internal_state(internal, ij, t, rel, pj)

            stress_c = new_c[I_STRESS] - internal[I_STRESS]
            stress_j = new_j[I_STRESS] - internal[I_STRESS]
            # AT LEAST the same stress as conflict alone
            assert stress_j >= stress_c - 0.02, (
                f"联合压力低于单独 conflict: {stress_c:.4f} vs {stress_j:.4f}")

    def test_validation_plus_closeness_boosts_energy(self, rng):
        """validation + closeness 联合使 energy 上升至少不弱于单独 validation。"""
        for _ in range(30):
            t = rng.uniform(-1, 1, T_SIZE)
            internal = rng.uniform(-0.5, 0.5, I_SIZE)
            rel = rng.uniform(-0.5, 0.5, R_SIZE)

            sv = np.zeros(ST_SIZE); sv[ST_VALIDATION] = 0.7
            pv = compute_defense_profiles(t, rel, internal)
            iv, _ = apply_defenses(sv, pv)
            new_v = update_internal_state(internal, iv, t, rel, pv)

            sj = np.zeros(ST_SIZE); sj[ST_VALIDATION] = 0.7; sj[ST_CLOSENESS] = 0.5
            pj = compute_defense_profiles(t, rel, internal)
            ij, _ = apply_defenses(sj, pj)
            new_j = update_internal_state(internal, ij, t, rel, pj)

            energy_v = new_v[I_ENERGY] - internal[I_ENERGY]
            energy_j = new_j[I_ENERGY] - internal[I_ENERGY]
            assert energy_j >= energy_v - 0.02, (
                f"联合 energy < 单独 validation: {energy_v:.4f} vs {energy_j:.4f}")

    def test_non_additivity_bounded(self, rng):
        """屏障测试：非加性偏差不超过 soft_clamp 过渡区宽度。"""
        violations = 0
        pairs = [(ST_ABANDONMENT, ST_VALIDATION), (ST_CONFLICT, ST_CLOSENESS)]
        n_trials = 50
        for _ in range(n_trials):
            t = rng.uniform(-1, 1, T_SIZE)
            internal = rng.uniform(-0.8, 0.8, I_SIZE)  # 充分安全距离
            rel = rng.uniform(-0.8, 0.8, R_SIZE)
            for a, b in pairs:
                for ai, bi in [(0.3, 0.3), (0.5, 0.5)]:
                    stimuli = np.zeros(ST_SIZE); stimuli[a] = ai; stimuli[b] = bi
                    profiles = compute_defense_profiles(t, rel, internal)
                    inner, _ = apply_defenses(stimuli, profiles)
                    new_j = update_internal_state(internal, inner, t, rel, profiles)

                    sa = np.zeros(ST_SIZE); sa[a] = ai
                    pa = compute_defense_profiles(t, rel, internal)
                    ia, _ = apply_defenses(sa, pa)
                    new_a = update_internal_state(internal, ia, t, rel, pa)

                    sb = np.zeros(ST_SIZE); sb[b] = bi
                    pb = compute_defense_profiles(t, rel, internal)
                    ib, _ = apply_defenses(sb, pb)
                    new_b = update_internal_state(internal, ib, t, rel, pb)

                    dj = new_j - internal
                    ds = (new_a - internal) + (new_b - internal)
                    diff = np.max(np.abs(dj - ds))
                    if diff > 0.10:  # soft_clamp 过渡区 ~0.11
                        violations += 1

        assert violations / (n_trials * 2 * 2) < 0.10, (
            f"非加性 >0.1 的比例 {violations}/{(n_trials*2*2)}")

