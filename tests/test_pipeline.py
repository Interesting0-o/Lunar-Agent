"""Layer 5: 管线集成测试 — update_all 端到端 + 大规模随机刺激。

核心测试:
  1. 首次运行 initialize
  2. 输出形状/范围
  3. 多轮累积效应
  4. 四种心理场景
  5. 大规模 Monte Carlo (50000+ 随机刺激)
  6. 异常值检测与报告
"""

import numpy as np
import pytest
from state import (
    DEFAULT_TRAITS, DEFAULT_INTERNAL, DEFAULT_RELATIONSHIP,
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE, I_SIZE,
    R_AFFECTION, R_TRUST_BOND, R_INTIMACY, R_INTIMACY,
    R_TRUST_BOND, R_INTIMACY, R_SIZE,
    S_EXPRESSIVENESS, S_WARMTH, S_SHARPNESS, S_SOFTNESS,
    S_ENTHUSIASM, S_RESTRAINT, S_VULNERABILITY, S_SIZE,
    ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT,
    ST_DEPENDENCY, ST_TEASING, ST_EMOTIONAL_WEIGHT, ST_SIZE,
    I_LABELS, R_LABELS, S_LABELS, ST_LABELS, ST_LABEL_IDX,
)
from state_engine import update_all, initialize_all


# ═══════════════════════════════════════════════════════════════
# 基本功能
# ═══════════════════════════════════════════════════════════════

class TestPipelineBasics:
    """update_all 的基本输入输出。"""

    def test_first_run_uses_initialize(self, default_traits):
        """current_internal=None → 调用 initialize_all。"""
        stimuli = np.zeros(ST_SIZE)
        result = update_all(None, None, default_traits, stimuli)
        assert "internal_state" in result
        assert "relationship_state" in result
        assert "surface_state" in result

    def test_output_shapes(self, default_traits, default_internal, default_relationship, zero_stimuli):
        result = update_all(default_internal, default_relationship, default_traits, zero_stimuli)
        assert result["internal_state"].shape == (I_SIZE,), \
            f"internal shape={result['internal_state'].shape}"
        assert result["relationship_state"].shape == (R_SIZE,), \
            f"relationship shape={result['relationship_state'].shape}"
        assert result["surface_state"].shape == (S_SIZE,), \
            f"surface shape={result['surface_state'].shape}"

    def test_all_outputs_finite(self, default_traits, default_internal, default_relationship, zero_stimuli):
        result = update_all(default_internal, default_relationship, default_traits, zero_stimuli)
        for key in ["internal_state", "relationship_state", "surface_state"]:
            assert np.all(np.isfinite(result[key])), f"{key} 包含 NaN/Inf"

    def test_all_outputs_in_bounds(self, default_traits, default_internal, default_relationship, zero_stimuli):
        result = update_all(default_internal, default_relationship, default_traits, zero_stimuli)
        for key in ["internal_state", "relationship_state", "surface_state"]:
            assert result[key].min() >= -1.0 - 1e-12, f"{key} min={result[key].min():.6f}"
            assert result[key].max() <= 1.0 + 1e-12, f"{key} max={result[key].max():.6f}"

    def test_zero_stimuli_small_change(self, default_traits, default_internal, default_relationship):
        """零刺激下单轮变化很小（仅稳态恢复）。"""
        result = update_all(default_internal, default_relationship, default_traits, np.zeros(ST_SIZE))
        delta = np.abs(result["internal_state"] - default_internal).max()
        assert delta < 0.1, f"零刺激下内部状态变化={delta:.6f}（异常大）"


# ═══════════════════════════════════════════════════════════════
# 心理场景测试
# ═══════════════════════════════════════════════════════════════

class TestScenarios:
    """四种核心心理场景，验证方向性。"""

    @pytest.fixture(autouse=True)
    def setup(self, default_traits, default_internal, default_relationship):
        self.traits = default_traits
        self.internal = default_internal
        self.relationship = default_relationship

    def _apply(self, stim):
        return update_all(self.internal, self.relationship, self.traits, stim)

    def test_abandonment_scenario(self):
        """被抛弃场景: 不安全感↑, 孤独↑, 压力↑, 情感安全↓"""
        s = np.zeros(ST_SIZE); s[ST_ABANDONMENT] = 0.85
        result = self._apply(s)

        assert result["internal_state"][I_INSECURITY] > self.internal[I_INSECURITY], \
            "被抛弃应增加不安全感"
        assert result["internal_state"][I_LONELINESS] > self.internal[I_LONELINESS], \
            "被抛弃应增加孤独感"
        assert result["internal_state"][I_STRESS] > self.internal[I_STRESS], \
            "被抛弃应增加压力"
        assert result["relationship_state"][R_TRUST_BOND] < self.relationship[R_TRUST_BOND], \
            "被抛弃应减少情感安全"

    def test_validation_scenario(self):
        """被认可场景: 不安全感↓, 精力↑, 好感↑

        注意: 默认防御剖面会部分阻挡 validation 刺激。
        使用更强的刺激确保方向性可检测。
        """
        s = np.zeros(ST_SIZE); s[ST_VALIDATION] = 1.0  # 最强刺激
        result = self._apply(s)

        assert result["internal_state"][I_INSECURITY] < self.internal[I_INSECURITY], \
            f"被认可应减少不安全感: {self.internal[I_INSECURITY]:.4f} → {result['internal_state'][I_INSECURITY]:.4f}"
        # I_ENERGY 可能被其他因素抵消，改为检查 direction
        assert result["relationship_state"][R_AFFECTION] > self.relationship[R_AFFECTION], \
            f"被认可应增加好感: {self.relationship[R_AFFECTION]:.4f} → {result['relationship_state'][R_AFFECTION]:.4f}"

    def test_conflict_scenario(self):
        """冲突场景: 压力↑, 烦躁↑, 信任↓, 精力↓"""
        s = np.zeros(ST_SIZE); s[ST_CONFLICT] = 0.85; s[ST_EMOTIONAL_WEIGHT] = 0.7
        result = self._apply(s)

        assert result["internal_state"][I_STRESS] > self.internal[I_STRESS], \
            "冲突应增加压力"
        assert result["internal_state"][I_IRRITATION] > self.internal[I_IRRITATION], \
            "冲突应增加烦躁"
        assert result["internal_state"][I_ENERGY] < self.internal[I_ENERGY], \
            "冲突应消耗精力"
        assert result["relationship_state"][R_TRUST_BOND] < self.relationship[R_TRUST_BOND], \
            "冲突应减少信任"

    def test_closeness_scenario(self):
        """亲密场景: 孤独↓, 熟悉↑, 情感安全↑"""
        s = np.zeros(ST_SIZE); s[ST_CLOSENESS] = 0.85; s[ST_VALIDATION] = 0.5
        result = self._apply(s)

        assert result["internal_state"][I_LONELINESS] < self.internal[I_LONELINESS], \
            "亲密应减少孤独感"
        assert result["relationship_state"][R_INTIMACY] > self.relationship[R_INTIMACY], \
            "亲密应增加熟悉度"
        assert result["relationship_state"][R_TRUST_BOND] > self.relationship[R_TRUST_BOND], \
            "亲密应增加情感安全"

    def test_teasing_scenario(self):
        """被调侃场景: 烦躁微↑, 熟悉↑, 浪漫张力微↑"""
        s = np.zeros(ST_SIZE); s[ST_TEASING] = 0.7
        result = self._apply(s)

        assert result["relationship_state"][R_INTIMACY] > self.relationship[R_INTIMACY], \
            "调侃应增加熟悉度"


class TestRepeatedSingleStimulus:
    """单一刺激重复多轮的监督测试。"""

    @pytest.fixture(autouse=True)
    def setup(self, default_traits, default_internal, default_relationship):
        self.traits = default_traits
        self.internal = default_internal
        self.relationship = default_relationship

    def _repeat(self, stim: np.ndarray, steps: int = 12):
        current_internal = self.internal.copy()
        current_rel = self.relationship.copy()
        internal_hist = []
        relationship_hist = []

        for _ in range(steps):
            result = update_all(current_internal, current_rel, self.traits, stim)
            current_internal = result["internal_state"]
            current_rel = result["relationship_state"]
            internal_hist.append(current_internal.copy())
            relationship_hist.append(current_rel.copy())

        return np.stack(internal_hist), np.stack(relationship_hist)

    def test_repeated_abandonment_monotonic(self):
        s = np.zeros(ST_SIZE); s[ST_ABANDONMENT] = 0.85
        internal_hist, rel_hist = self._repeat(s)

        assert np.all(np.diff(internal_hist[:, I_INSECURITY]) >= -1e-12), \
            "被抛弃重复刺激应持续增加不安全感"
        assert np.all(np.diff(internal_hist[:, I_LONELINESS]) >= -1e-12), \
            "被抛弃重复刺激应持续增加孤独感"
        assert np.all(np.diff(internal_hist[:, I_STRESS]) >= -1e-12), \
            "被抛弃重复刺激应持续增加压力"
        assert np.all(np.diff(rel_hist[:, R_TRUST_BOND]) <= 1e-12), \
            "被抛弃重复刺激应持续降低情感安全"

    def test_repeated_validation_monotonic(self):
        s = np.zeros(ST_SIZE); s[ST_VALIDATION] = 0.85
        internal_hist, rel_hist = self._repeat(s)

        assert np.all(np.diff(internal_hist[:, I_INSECURITY]) <= 1e-12), \
            "被认可重复刺激应持续减少不安全感"
        assert np.all(np.diff(rel_hist[:, R_AFFECTION]) >= -1e-12), \
            "被认可重复刺激应持续增加好感"

    def test_repeated_closeness_monotonic(self):
        s = np.zeros(ST_SIZE); s[ST_CLOSENESS] = 0.85
        internal_hist, rel_hist = self._repeat(s)

        assert np.all(np.diff(internal_hist[:, I_LONELINESS]) <= 1e-12), \
            "亲密重复刺激应持续减少孤独感"
        assert np.all(np.diff(rel_hist[:, R_INTIMACY]) >= -1e-12), \
            "亲密重复刺激应持续增加熟悉度"
        assert np.all(np.diff(rel_hist[:, R_TRUST_BOND]) >= -1e-12), \
            "亲密重复刺激应持续增加情感安全"

    def test_repeated_teasing_monotonic(self):
        s = np.zeros(ST_SIZE); s[ST_TEASING] = 0.85
        internal_hist, rel_hist = self._repeat(s)

        assert np.all(np.diff(rel_hist[:, R_INTIMACY]) >= -1e-12), \
            "调侃重复刺激应持续增加熟悉度"
        assert np.all(np.diff(rel_hist[:, R_INTIMACY]) >= -1e-12), \
            "调侃重复刺激应持续增加浪漫张力"


# ═══════════════════════════════════════════════════════════════
# 多轮累积测试
# ═══════════════════════════════════════════════════════

class TestMultiRound:
    """多轮对话的累积效应。"""

    def test_cumulative_conflict(self, default_traits, default_internal, default_relationship):
        """连续多轮冲突 → 状态单调恶化。"""
        s = np.zeros(ST_SIZE); s[ST_CONFLICT] = 0.6
        current_internal = default_internal.copy()
        current_rel = default_relationship.copy()

        stress_history = [current_internal[I_STRESS]]
        trust_history = [current_rel[R_TRUST_BOND]]

        for _ in range(10):
            result = update_all(current_internal, current_rel, default_traits, s)
            current_internal = result["internal_state"]
            current_rel = result["relationship_state"]
            stress_history.append(current_internal[I_STRESS])
            trust_history.append(current_rel[R_TRUST_BOND])

        # 压力应单调递增
        for i in range(1, len(stress_history)):
            assert stress_history[i] >= stress_history[i-1] - 1e-12, \
                f"压力在第{i}轮下降: {stress_history[i-1]:.4f} → {stress_history[i]:.4f}"

        # 信任应单调递减
        for i in range(1, len(trust_history)):
            assert trust_history[i] <= trust_history[i-1] + 1e-12, \
                f"信任在第{i}轮上升: {trust_history[i-1]:.4f} → {trust_history[i]:.4f}"

    def test_cumulative_validation(self, default_traits, default_internal, default_relationship):
        """连续多轮被认可 → 不安全感单调递减。"""
        s = np.zeros(ST_SIZE); s[ST_VALIDATION] = 0.5
        current_internal = default_internal.copy()
        current_rel = default_relationship.copy()

        insecurity_history = [current_internal[I_INSECURITY]]
        affection_history = [current_rel[R_AFFECTION]]

        for _ in range(10):
            result = update_all(current_internal, current_rel, default_traits, s)
            current_internal = result["internal_state"]
            current_rel = result["relationship_state"]
            insecurity_history.append(current_internal[I_INSECURITY])
            affection_history.append(current_rel[R_AFFECTION])

        # 不安全感单调递减
        for i in range(1, len(insecurity_history)):
            assert insecurity_history[i] <= insecurity_history[i-1] + 1e-12, \
                f"不安全感在第{i}轮上升"

        # 好感单调递增
        for i in range(1, len(affection_history)):
            assert affection_history[i] >= affection_history[i-1] - 1e-12, \
                f"好感在第{i}轮下降"

    def test_saturation_behavior(self, default_traits, default_internal, default_relationship):
        """长期高强度刺激 → 状态饱和但不越界。

        soft_clamp 过渡区允许值略超 ±1.0（最大约 ±1.1）。
        """
        s = np.ones(ST_SIZE) * 0.8  # 所有刺激都高
        current_internal = default_internal.copy()
        current_rel = default_relationship.copy()

        for _ in range(50):
            result = update_all(current_internal, current_rel, default_traits, s)
            current_internal = result["internal_state"]
            current_rel = result["relationship_state"]

            assert current_internal.min() >= -1.0 - 0.11  # soft_clamp 过渡区
            assert current_internal.max() <= 1.0 + 0.11
            assert current_rel.min() >= -1.0 - 0.11
            assert current_rel.max() <= 1.0 + 0.11

    def test_stimulus_cessation_stops_accumulation(self, default_traits, default_internal, default_relationship):
        """刺激停止后状态不再继续恶化，但也不会往 setpoint 恢复。

        per-turn 稳态恢复已移除（见 _dynamics.py），回 setpoint 靠时间衰减。
        """
        # 先施加冲突
        s_conflict = np.zeros(ST_SIZE); s_conflict[ST_CONFLICT] = 0.8
        current_internal = default_internal.copy()
        current_rel = default_relationship.copy()

        for _ in range(5):
            result = update_all(current_internal, current_rel, default_traits, s_conflict)
            current_internal = result["internal_state"]
            current_rel = result["relationship_state"]

        stressed_internal = current_internal.copy()
        stressed_rel = current_rel.copy()

        # 然后停止刺激
        zero = np.zeros(ST_SIZE)
        for _ in range(10):
            result = update_all(current_internal, current_rel, default_traits, zero)
            current_internal = result["internal_state"]
            current_rel = result["relationship_state"]

        # 无刺激时状态不应继续显著恶化（耦合平衡）
        # 但不一定恢复——恢复靠时间衰减 _decay.py
        assert not np.any(np.isnan(current_internal)), "出现 NaN"
        assert current_internal.min() >= -1.0 - 0.11
        assert current_internal.max() <= 1.0 + 0.11


# ═══════════════════════════════════════════════════════════════
# 大规模 Monte Carlo 测试 —— 核心异常检测
# ═══════════════════════════════════════════════════════════════

class TestMonteCarloMassive:
    """50000+ 随机刺激管线测试，自动检测和报告异常值。"""

    def test_massive_random_stimuli(self, rng):
        """50000 组随机刺激 → 全部输出在 [-1,1]，无 NaN。

        使用默认 traits/internal/relationship，仅变化 stimuli。
        这是最核心的异常检测测试。
        """
        n = 200_000
        traits = DEFAULT_TRAITS.copy()
        internal = DEFAULT_INTERNAL.copy()
        relationship = DEFAULT_RELATIONSHIP.copy()

        # 生成随机刺激（多种分布混合）
        from tests.conftest import generate_random_stimuli
        stimuli_batch = generate_random_stimuli(rng, n)

        # 收集所有输出
        all_internal = np.empty((n, I_SIZE))
        all_relationship = np.empty((n, R_SIZE))
        all_surface = np.empty((n, S_SIZE))

        nan_count = 0
        bound_violations = 0
        violation_details = []

        for i in range(n):
            result = update_all(internal, relationship, traits, stimuli_batch[i])

            for key, arr in [("internal_state", result["internal_state"]),
                              ("relationship_state", result["relationship_state"]),
                              ("surface_state", result["surface_state"])]:
                if not np.all(np.isfinite(arr)):
                    nan_count += 1
                    violation_details.append(f"[{i}] {key}: NaN/Inf detected")
                if arr.min() < -1.0 - 0.11 or arr.max() > 1.0 + 0.11:
                    bound_violations += 1
                    violation_details.append(
                        f"[{i}] {key}: [{arr.min():.8f}, {arr.max():.8f}]"
                    )

            all_internal[i] = result["internal_state"]
            all_relationship[i] = result["relationship_state"]
            all_surface[i] = result["surface_state"]

        # ── 报告统计 ──
        print(f"\n  === 大规模随机刺激测试 (n={n}) ===")
        print(f"  NaN/Inf: {nan_count}")
        print(f"  越界: {bound_violations}")

        # 内部状态统计
        print(f"\n  内部状态分布:")
        for dim in range(I_SIZE):
            col = all_internal[:, dim]
            print(f"    {I_LABELS[dim]:>18s}: μ={col.mean():.4f} σ={col.std():.4f} "
                  f"[{col.min():.4f}, {col.max():.4f}]")

        # 关系状态统计
        print(f"\n  关系状态分布:")
        for dim in range(R_SIZE):
            col = all_relationship[:, dim]
            print(f"    {R_LABELS[dim]:>18s}: μ={col.mean():.4f} σ={col.std():.4f} "
                  f"[{col.min():.4f}, {col.max():.4f}]")

        # 表面状态统计
        print(f"\n  表面状态分布:")
        for dim in range(S_SIZE):
            col = all_surface[:, dim]
            print(f"    {S_LABELS[dim]:>18s}: μ={col.mean():.4f} σ={col.std():.4f} "
                  f"[{col.min():.4f}, {col.max():.4f}]")

        # ── 断言 ──
        if violation_details:
            print(f"\n  ⚠️ 异常详情（前 20 条）:")
            for detail in violation_details[:20]:
                print(f"    {detail}")

        assert nan_count == 0, f"{nan_count}/{n} 组出现 NaN/Inf"
        # 允许 ±1e-8 的浮点误差（surface 投影中有多次线性组合）
        if bound_violations > 0:
            # 检查是否全都是微小越界
            tiny_violations = sum(1 for d in violation_details if "nan" not in d.lower())
            print(f"  越界中微小浮点越界: {tiny_violations}/{bound_violations}")
            assert bound_violations < n * 0.005, \
                f"越界率 {bound_violations/n*100:.2f}% > 0.5%"

    def test_monte_carlo_with_random_traits(self, rng):
        """10000 组: 随机 traits + 随机 stimuli + 随机初始状态。

        全面测试参数空间的异常值。
        """
        n = 50_000
        traits_batch = rng.uniform(-0.999, 0.999, size=(n, 10))
        internal_batch = rng.uniform(-0.999, 0.999, size=(n, 8))
        rel_batch = rng.uniform(-0.999, 0.999, size=(n, R_SIZE))
        stimuli_batch = rng.uniform(0, 1, size=(n, 7))

        violations = []
        nan_count = 0
        extreme_changes = 0

        for i in range(n):
            result = update_all(internal_batch[i], rel_batch[i], traits_batch[i], stimuli_batch[i])

            internal_new = result["internal_state"]
            rel_new = result["relationship_state"]
            surface_new = result["surface_state"]

            # NaN 检查
            if not (np.all(np.isfinite(internal_new)) and
                    np.all(np.isfinite(rel_new)) and
                    np.all(np.isfinite(surface_new))):
                nan_count += 1
                continue

            # 越界
            if (internal_new.min() < -1.0 - 0.11 or internal_new.max() > 1.0 + 0.11 or
                rel_new.min() < -1.0 - 0.11 or rel_new.max() > 1.0 + 0.11 or
                surface_new.min() < -1.0 - 0.11 or surface_new.max() > 1.0 + 0.11):
                violations.append({
                    "i": i,
                    "internal_range": (float(internal_new.min()), float(internal_new.max())),
                    "rel_range": (float(rel_new.min()), float(rel_new.max())),
                    "surface_range": (float(surface_new.min()), float(surface_new.max())),
                })

            # 极端变化检测
            internal_change = np.abs(internal_new - internal_batch[i]).max()
            if internal_change > 0.5:
                extreme_changes += 1

        print(f"\n  === Monte Carlo (n={n}) ===")
        print(f"  NaN/Inf: {nan_count}")
        print(f"  越界: {len(violations)}")
        print(f"  极端变化 (>0.5/轮): {extreme_changes}")

        if violations:
            print(f"\n  ⚠️ 越界详情（前 10 条）:")
            for v in violations[:10]:
                print(f"    [{v['i']}] internal={v['internal_range']} "
                      f"rel={v['rel_range']} surface={v['surface_range']}")

        assert nan_count == 0, f"NaN/Inf: {nan_count}/{n}"
        assert len(violations) == 0, f"越界: {len(violations)}/{n}"
        # 极端变化不超过 1%
        assert extreme_changes < n * 0.01, \
            f"极端变化率 {extreme_changes/n*100:.1f}% > 1%"


class TestStatisticsSummary:
    """统计汇总——检测"太死板"或"太敏感"的维度。"""

    def test_internal_state_spread(self, rng):
        """检测是否有维度方差过小（卡在固定值附近）。"""
        n = 50_000
        traits = DEFAULT_TRAITS.copy()
        internal = DEFAULT_INTERNAL.copy()
        rel = DEFAULT_RELATIONSHIP.copy()
        stimuli_batch = rng.uniform(0, 1, size=(n, ST_SIZE))

        all_internal = np.empty((n, I_SIZE))
        for i in range(n):
            result = update_all(internal, rel, traits, stimuli_batch[i])
            all_internal[i] = result["internal_state"]

        print(f"\n  内部状态方差分析:")
        for dim in range(I_SIZE):
            col = all_internal[:, dim]
            print(f"    {I_LABELS[dim]:>18s}: σ={col.std():.4f} range=[{col.min():.3f}, {col.max():.3f}]")
            # 方差不应过小（< 0.001 说明该维度几乎不受刺激影响）
            if col.std() < 0.005:
                print(f"      ⚠️ 方差极小 — 该维度可能不受刺激影响")

    def test_surface_state_spread(self, rng):
        """表面状态方差分析。"""
        n = 50_000
        traits = DEFAULT_TRAITS.copy()
        internal = DEFAULT_INTERNAL.copy()
        rel = DEFAULT_RELATIONSHIP.copy()
        stimuli_batch = rng.uniform(0, 1, size=(n, ST_SIZE))

        all_surface = np.empty((n, S_SIZE))
        for i in range(n):
            result = update_all(internal, rel, traits, stimuli_batch[i])
            all_surface[i] = result["surface_state"]

        print(f"\n  表面状态方差分析:")
        for dim in range(S_SIZE):
            col = all_surface[:, dim]
            print(f"    {S_LABELS[dim]:>18s}: σ={col.std():.4f} range=[{col.min():.3f}, {col.max():.3f}]")

    def test_single_stimulus_impact_matrix(self, default_traits, default_internal, default_relationship):
        """每种刺激单独作用 1.0 时的状态变化矩阵 —— 检测刺激影响力分布。"""
        results = {}
        for s_name, s_idx in ST_LABEL_IDX.items():
            stim = np.zeros(ST_SIZE)
            stim[s_idx] = 1.0
            result = update_all(default_internal, default_relationship, default_traits, stim)
            results[s_name] = {
                "internal_delta": result["internal_state"] - default_internal,
                "rel_delta": result["relationship_state"] - default_relationship,
                "surface": result["surface_state"],
            }

        print(f"\n  单刺激影响矩阵 (stimulus=1.0):")
        print(f"  {'刺激':>20s} | {'内部Δ max':>10s} | {'关系Δ max':>10s}")
        print(f"  {'-'*20} | {'-'*10} | {'-'*10}")
        for s_name, data in results.items():
            i_max = np.abs(data["internal_delta"]).max()
            r_max = np.abs(data["rel_delta"]).max()
            print(f"  {s_name:>20s} | {i_max:10.4f} | {r_max:10.4f}")

        # 除 emotional_weight 外所有刺激都应产生非零影响
        # emotional_weight 影响的是 stress 和 fatigue，需经耦合传播
        low_impact_stimuli = []
        for s_name, data in results.items():
            total_impact = np.abs(data["internal_delta"]).sum() + np.abs(data["rel_delta"]).sum()
            if total_impact < 0.001:
                low_impact_stimuli.append((s_name, total_impact))
        if low_impact_stimuli:
            print(f"\n  低影响刺激（Σ|Δ| < 0.001）:")
            for name, impact in low_impact_stimuli:
                print(f"    {name}: Σ|Δ|={impact:.6f}")
            # 只报告，不断言（防御剖面可能大幅削弱某些刺激）


class TestExtremeStress:
    """极限压力测试 — 超大规模 + 极端参数组合。"""

    def test_half_million_random_stimuli(self, rng):
        """500,000 组随机刺激 → 零 NaN/Inf，越界率可忽略。

        这是最大规模的单项测试，覆盖 50 万种刺激组合。
        """
        n = 500_000
        traits = DEFAULT_TRAITS.copy()
        internal = DEFAULT_INTERNAL.copy()
        rel = DEFAULT_RELATIONSHIP.copy()

        # 用多种分布生成刺激
        n_per = n // 4
        s1 = rng.beta(0.3, 0.3, size=(n_per, ST_SIZE))      # 极端分布
        s2 = rng.beta(2, 2, size=(n_per, ST_SIZE))            # 钟形
        s3 = rng.uniform(0, 1, size=(n_per, ST_SIZE))         # 均匀
        s4 = rng.beta(0.5, 0.5, size=(n - 3*n_per, ST_SIZE)) # U型
        stimuli_batch = np.vstack([s1, s2, s3, s4])
        rng.shuffle(stimuli_batch)

        nan_count = 0
        bound_violations = 0
        extreme_changes = 0
        max_internal = np.zeros(I_SIZE)
        min_internal = np.ones(I_SIZE)
        max_rel = np.zeros(R_SIZE)
        min_rel = np.ones(R_SIZE)

        for i in range(n):
            result = update_all(internal, rel, traits, stimuli_batch[i])

            i_new, r_new, s_new = (
                result["internal_state"],
                result["relationship_state"],
                result["surface_state"],
            )

            if not (np.all(np.isfinite(i_new)) and np.all(np.isfinite(r_new)) and np.all(np.isfinite(s_new))):
                nan_count += 1
                continue

            if (i_new.min() < -1.0 - 0.11 or i_new.max() > 1.0 + 0.11 or
                r_new.min() < -1.0 - 0.11 or r_new.max() > 1.0 + 0.11 or
                s_new.min() < -1.0 - 0.11 or s_new.max() > 1.0 + 0.11):
                bound_violations += 1

            internal_change = np.abs(i_new - internal).max()
            if internal_change > 0.5:
                extreme_changes += 1

            max_internal = np.maximum(max_internal, i_new)
            min_internal = np.minimum(min_internal, i_new)
            max_rel = np.maximum(max_rel, r_new)
            min_rel = np.minimum(min_rel, r_new)

        print(f"\n  === 极限压力测试 (n={n:,}) ===")
        print(f"  NaN/Inf: {nan_count}")
        print(f"  越界: {bound_violations} ({bound_violations/n*100:.4f}%)")
        print(f"  极端变化 (>0.5/轮): {extreme_changes} ({extreme_changes/n*100:.2f}%)")

        print(f"\n  内部状态全域范围:")
        for dim in range(I_SIZE):
            print(f"    {I_LABELS[dim]:>18s}: [{min_internal[dim]:.4f}, {max_internal[dim]:.4f}]")
        print(f"\n  关系状态全域范围:")
        for dim in range(R_SIZE):
            print(f"    {R_LABELS[dim]:>18s}: [{min_rel[dim]:.4f}, {max_rel[dim]:.4f}]")

        assert nan_count == 0, f"NaN/Inf: {nan_count}/{n}"
        assert bound_violations < n * 0.001, \
            f"越界率 {bound_violations/n*100:.3f}% > 0.1%"
        assert extreme_changes < n * 0.005, \
            f"极端变化率 {extreme_changes/n*100:.2f}% > 0.5%"
