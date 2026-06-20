"""监督测试：时间衰减模块的行为覆盖。

监督面 (Coverage Dimensions):
  1. 基础衰减公式 — 指数衰减的数值正确性
  2. 非对称衰减 — 负向偏离加速 (Fading Affect Bias)
  3. 人格调制 — traits 对 λ_eff 的影响范围与方向
  4. 时间曲线 — λ_eff 随 Δt 的幂律放缓
  5. 边界稳健性 — 极端值 / 零时间 / 长间隔
  6. 批量统计 — 大规模随机场景下的分布异常检测
  7. 可视化 — 趋势图保存到 tests/result/

度量标准:
  - 所有状态 ∈ [-1, 1]
  - 非对称衰减比例 ≈ negative_decay_boost (1.8)
  - 人格调制因子 ∈ [0.3, 2.0]
  - 单调性：Δt 越长 → 越接近 setpoint
"""

import numpy as np
import pytest
from dataclasses import replace

from state import (
    DEFAULT_TRAITS, DEFAULT_INTERNAL, DEFAULT_RELATIONSHIP,
    T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE,
    T_EMOTIONAL_STABILITY, T_OPTIMISM, T_ANXIETY_PRONENESS,
    T_ANGER_REACTIVITY, T_EMOTIONAL_OPENNESS,
    R_TRUST, R_AFFECTION, R_FAMILIARITY, R_DEPENDENCY,
    R_EMOTIONAL_SAFETY, R_ROMANTIC_TENSION, R_SIZE,
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SIZE,
    R_LABELS,
)
from state_engine._decay import (
    apply_time_decay_internal, apply_time_decay_relationship,
    apply_time_decay,
    DecayConfig, DEFAULT_DECAY_CONFIG,
    _compute_internal_personality_mod, _compute_relationship_personality_mod,
    _compute_lambda_effective,
)
from state_engine._dynamics import compute_setpoint, compute_rel_setpoint


# ═══════════════════════════════════════════════════════════════
# 测试数据
# ═══════════════════════════════════════════════════════════════

# 一个典型的关系受伤场景（信任/好感受损）
DAMAGED_REL = np.array([
    -0.5,   # R_AFFECTION — 好感降低
    -0.6,   # R_TRUST — 信任受伤
    -0.3,   # R_FAMILIARITY — 略疏远
    -0.4,   # R_DEPENDENCY — 不敢依赖
    -0.6,   # R_EMOTIONAL_SAFETY — 不安全
    -0.3,   # R_ROMANTIC_TENSION — 张力降低
], dtype=np.float64)

# 一个典型的关系升温场景（所有维度高正值）
ENHANCED_REL = np.array([
     0.7,   # R_AFFECTION
     0.8,   # R_TRUST
     0.6,   # R_FAMILIARITY
     0.5,   # R_DEPENDENCY
     0.7,   # R_EMOTIONAL_SAFETY
     0.6,   # R_ROMANTIC_TENSION
], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════
# 1. 基础衰减公式
# ═══════════════════════════════════════════════════════════════

class TestBasicDecay:
    """指数衰减的数值正确性。"""

    def test_zero_delta_no_change(self, default_traits, default_internal, default_relationship):
        """Δt = 0 → 无变化。"""
        sp_i = compute_setpoint(default_traits)
        sp_r = compute_rel_setpoint(default_traits)
        r1 = apply_time_decay_internal(default_internal, sp_i, default_traits, 0.0)
        r2 = apply_time_decay_relationship(default_relationship, sp_r, default_traits, 0.0)
        np.testing.assert_array_equal(r1, default_internal)
        np.testing.assert_array_equal(r2, default_relationship)

    def test_microscopic_delta_no_change(self, default_traits, default_internal):
        """Δt < min_delta_hours → 跳过衰减。"""
        sp = compute_setpoint(default_traits)
        r = apply_time_decay_internal(default_internal, sp, default_traits, 0.001)
        np.testing.assert_array_equal(r, default_internal)

    def test_at_setpoint_no_change(self, default_traits):
        """处于 setpoint → 衰减后仍在 setpoint。"""
        sp = compute_setpoint(default_traits)
        sp_r = compute_rel_setpoint(default_traits)
        r1 = apply_time_decay_internal(sp, sp, default_traits, 24.0)
        r2 = apply_time_decay_relationship(sp_r, sp_r, default_traits, 24.0)
        np.testing.assert_array_equal(r1, sp)
        np.testing.assert_array_equal(r2, sp_r)

    def test_convergence_toward_setpoint(self, default_traits):
        """长 Δt 下状态趋近 setpoint（受幂律尾渐近限制）。"""
        sp_i = compute_setpoint(default_traits)
        sp_r = compute_rel_setpoint(default_traits)

        extreme = np.full(I_SIZE, 0.9, dtype=np.float64)
        extreme_rel = np.full(R_SIZE, 0.9, dtype=np.float64)

        # 时间曲线 1/(1+k·Δt) 使 λ_eff → 0 当 Δt → ∞
        # 渐近残余 exp(-λ_base * p_mod / k) 在内部 (k=0.05) 和关系 (k=0.001) 均存在
        p_mod_i = _compute_internal_personality_mod(default_traits)
        p_mod_r = _compute_relationship_personality_mod(default_traits)
        r_i = apply_time_decay_internal(extreme, sp_i, default_traits, 10000)
        r_r = apply_time_decay_relationship(extreme_rel, sp_r, default_traits, 10000)

        # 内部态：λ*large / k=0.05 → exp(-0.12~0.69*mod/0.05)
        for d in range(I_SIZE):
            dev = extreme[d] - sp_i[d]
            asymp = np.exp(
                -DEFAULT_DECAY_CONFIG.internal_lambda[d] * p_mod_i
                / DEFAULT_DECAY_CONFIG.internal_time_curve_k
            )
            max_residual = abs(dev) * asymp * 1.05 + 0.02
            actual_residual = abs(r_i[d] - sp_i[d])
            assert actual_residual <= max_residual + 0.05, (
                f"I_{d}: 残余 {actual_residual:.4f} 超出包络 {max_residual:.4f}"
                f"(asymp={asymp:.4f})"
            )

        # 关系态：k=0.001 也有渐近残余 exp(-λ*mod/0.001)，最小 ≈ exp(-0.0014*0.3/0.001)=0.66
        for d in range(R_SIZE):
            dev = extreme_rel[d] - sp_r[d]
            asymp = np.exp(
                -DEFAULT_DECAY_CONFIG.relationship_lambda[d] * p_mod_r
                / DEFAULT_DECAY_CONFIG.relationship_time_curve_k
            )
            max_residual = abs(dev) * asymp * 1.05 + 0.02
            actual_residual = abs(r_r[d] - sp_r[d])
            assert actual_residual <= max_residual + 0.05, (
                f"R_{d}: 残余 {actual_residual:.4f} 超出包络 {max_residual:.4f}"
                f"(asymp={asymp:.4f})"
            )

    def test_monotonicity_wrt_delta(self, default_traits):
        """同一初始状态，Δt 越大 → 越接近 setpoint（单调性）。"""
        sp_i = compute_setpoint(default_traits)
        sp_r = compute_rel_setpoint(default_traits)

        extreme = np.full(I_SIZE, 0.9, dtype=np.float64)
        extreme_rel = np.full(R_SIZE, 0.9, dtype=np.float64)

        deltas = [0.5, 1, 3, 6, 12, 24, 48]
        prev_dist_i = np.inf
        prev_dist_r = np.inf
        for dt in deltas:
            r_i = apply_time_decay_internal(extreme, sp_i, default_traits, dt)
            r_r = apply_time_decay_relationship(extreme_rel, sp_r, default_traits, dt)
            dist_i = np.max(np.abs(r_i - sp_i))
            dist_r = np.max(np.abs(r_r - sp_r))
            assert dist_i < prev_dist_i + 1e-12, f"内部状态在 Δt={dt} 时逆单调"
            assert dist_r < prev_dist_r + 1e-12, f"关系状态在 Δt={dt} 时逆单调"
            prev_dist_i = dist_i
            prev_dist_r = dist_r


# ═══════════════════════════════════════════════════════════════
# 2. 非对称衰减（核心新功能）
# ═══════════════════════════════════════════════════════════════

class TestAsymmetricDecay:
    """Fading Affect Bias: 负向偏离衰减快于正向偏离。"""

    def test_negative_faster_than_positive(self, default_traits):
        """同一维度的负向偏离衰减快于正向偏离。"""
        sp = compute_rel_setpoint(default_traits)

        # 固定一个维度（以 R_TRUST 为例），其他人 setpoint
        def make_state(dim, val):
            s = sp.copy()
            s[dim] = val
            return s

        dt = 72  # 3 天

        for dim in range(R_SIZE):
            # 负向偏离（低于 setpoint 0.5）
            neg_mag = max(-0.99, sp[dim] - 0.5)
            pos_mag = min(0.99, sp[dim] + 0.5)

            neg_state = make_state(dim, neg_mag)
            pos_state = make_state(dim, pos_mag)

            r_neg = apply_time_decay_relationship(neg_state, sp, default_traits, dt)
            r_pos = apply_time_decay_relationship(pos_state, sp, default_traits, dt)

            # 负向偏离的绝对变化量 vs 正向偏离的绝对变化量
            # 注意：偏离量不同时，绝对变化量未必直接可比，但比例应反映 1.8× 加速
            neg_dev = sp[dim] - neg_mag   # 正数
            pos_dev = pos_mag - sp[dim]   # 正数
            neg_delta = abs(r_neg[dim] - neg_mag)
            pos_delta = abs(r_pos[dim] - pos_mag)

            # 归一化：恢复比例 = 已恢复 / 总偏离
            neg_ratio = neg_delta / neg_dev if neg_dev > 1e-10 else 0
            pos_ratio = pos_delta / pos_dev if pos_dev > 1e-10 else 0

            # 负向恢复比例应明显大于正向
            assert neg_ratio > pos_ratio * 1.5, (
                f"{R_LABELS[dim]}: neg_ratio={neg_ratio:.4f} (偏离 {neg_dev:.3f}) 应 "
                f"> pos_ratio={pos_ratio:.4f} (偏离 {pos_dev:.3f})"
            )

    def test_asymmetry_ratio_approaches_boost(self, default_traits):
        """衰减比例应接近配置 boost=1.8。"""
        sp = compute_rel_setpoint(default_traits)
        np.random.seed(42)

        ratios = []
        for dim in range(R_SIZE):
            for _ in range(20):
                delta = np.random.uniform(0.2, 0.8)
                neg_mag = max(-0.99, sp[dim] - delta)
                pos_mag = min(0.99, sp[dim] + delta)

                dt = np.random.uniform(12, 168)
                neg_state = sp.copy(); neg_state[dim] = neg_mag
                pos_state = sp.copy(); pos_state[dim] = pos_mag

                r_neg = apply_time_decay_relationship(neg_state, sp, default_traits, dt)
                r_pos = apply_time_decay_relationship(pos_state, sp, default_traits, dt)

                neg_dev = abs(sp[dim] - neg_mag)
                pos_dev = abs(pos_mag - sp[dim])
                if neg_dev < 1e-10 or pos_dev < 1e-10:
                    continue
                neg_frac = abs(r_neg[dim] - neg_mag) / neg_dev
                pos_frac = abs(r_pos[dim] - pos_mag) / pos_dev
                if pos_frac < 1e-10:
                    continue
                ratios.append(neg_frac / pos_frac)

        ratios = np.array(ratios)
        mean_ratio = ratios.mean()

        # 均值应在 1.8 ± 0.5 范围内（受时间曲线、人格调制影响而略偏离）
        assert 1.3 <= mean_ratio <= 2.5, (
            f"平均不对称比例 {mean_ratio:.3f} 超出预期范围 [1.3, 2.5] "
            f"(min={ratios.min():.3f}, max={ratios.max():.3f})"
        )

    def test_internal_state_not_affected(self, default_traits, default_internal):
        """内部状态不应受 negative_decay_boost 影响。"""
        sp = compute_setpoint(default_traits)

        # 构造负向偏离的内部状态
        neg_internal = default_internal.copy()
        neg_internal[I_STRESS] = 0.8      # 高压（负向）
        neg_internal[I_ENERGY] = -0.7     # 低能（负向）
        neg_internal[I_IRRITATION] = 0.9  # 烦躁（负向）

        # 正向偏离
        pos_internal = default_internal.copy()
        pos_internal[I_STRESS] = -0.7
        pos_internal[I_ENERGY] = 0.8
        pos_internal[I_IRRITATION] = -0.8

        dt = 24
        r_neg = apply_time_decay_internal(neg_internal, sp, default_traits, dt)
        r_pos = apply_time_decay_internal(pos_internal, sp, default_traits, dt)

        # 验证：内部状态不应有非对称加速，负向和正向的衰减应对称
        # 使用对称偏离量
        def sym_dev(state, dim):
            return abs(state[dim] - sp[dim])

        for dim in [I_STRESS, I_ENERGY, I_IRRITATION]:
            nd = sym_dev(neg_internal, dim)
            pd = sym_dev(pos_internal, dim)
            if abs(nd - pd) < 1e-6:
                nf = abs(r_neg[dim] - neg_internal[dim])
                pf = abs(r_pos[dim] - pos_internal[dim])
                ratio = nf / pf if pf > 1e-10 else 1
                assert 0.9 <= ratio <= 1.1, (
                    f"I_{dim} 内部状态非对称 ratio={ratio:.3f}，应为 ~1.0"
                )

    def test_config_change_boost(self, default_traits):
        """修改 boost 参数应改变衰减速率。"""
        sp = compute_rel_setpoint(default_traits)

        # 信任受伤
        damaged = sp.copy()
        damaged[R_TRUST] = -0.7

        dt = 48

        config_1x = replace(DEFAULT_DECAY_CONFIG, negative_decay_boost=1.0)
        config_2x = replace(DEFAULT_DECAY_CONFIG, negative_decay_boost=2.0)
        config_3x = replace(DEFAULT_DECAY_CONFIG, negative_decay_boost=3.0)

        r_1x = apply_time_decay_relationship(damaged, sp, default_traits, dt, config_1x)
        r_2x = apply_time_decay_relationship(damaged, sp, default_traits, dt, config_2x)
        r_3x = apply_time_decay_relationship(damaged, sp, default_traits, dt, config_3x)

        # boost 越大 → 恢复越多（离 setpoint 更近）
        d1 = abs(r_1x[R_TRUST] - sp[R_TRUST])
        d2 = abs(r_2x[R_TRUST] - sp[R_TRUST])
        d3 = abs(r_3x[R_TRUST] - sp[R_TRUST])

        assert d1 > d2 > d3, (
            f"boost 1x 残差 {d1:.6f}，2x {d2:.6f}，3x {d3:.6f}，应单调递减"
        )

    def test_each_dimension_independent(self, default_traits):
        """每维独立判断：同一维度负向恢复比例 > 正向（未加速）恢复比例。"""
        sp = compute_rel_setpoint(default_traits)
        dt = 72

        for dim in range(R_SIZE):
            # 同一维度：正向偏离 vs 负向偏离
            neg_state = sp.copy()
            neg_state[dim] = sp[dim] - 0.4  # 负向
            pos_state = sp.copy()
            pos_state[dim] = sp[dim] + 0.4  # 正向

            r_neg = apply_time_decay_relationship(neg_state, sp, default_traits, dt)
            r_pos = apply_time_decay_relationship(pos_state, sp, default_traits, dt)

            nd = abs(sp[dim] - neg_state[dim])
            pd = abs(pos_state[dim] - sp[dim])
            neg_rec = abs(r_neg[dim] - neg_state[dim]) / nd if nd > 1e-10 else 0
            pos_rec = abs(r_pos[dim] - pos_state[dim]) / pd if pd > 1e-10 else 0

            assert neg_rec > pos_rec, (
                f"{R_LABELS[dim]}: 负向恢复 {neg_rec:.4f} 应快于正向 {pos_rec:.4f}"
            )


# ═══════════════════════════════════════════════════════════════
# 3. 人格调制
# ═══════════════════════════════════════════════════════════════

class TestPersonalityModulation:
    """人格特质对衰减速率的调制。"""

    def test_internal_mod_range(self):
        """内部人格调制因子在有效范围内。"""
        for _ in range(1000):
            traits = np.random.uniform(-1, 1, 10)
            mod = _compute_internal_personality_mod(traits)
            assert 0.3 <= mod <= 2.0, f"调制因子 {mod:.4f} 超出 [0.3, 2.0]"

    def test_relationship_mod_range(self):
        """关系人格调制因子在有效范围内。"""
        for _ in range(1000):
            traits = np.random.uniform(-1, 1, 10)
            mod = _compute_relationship_personality_mod(traits)
            assert 0.3 <= mod <= 2.0, f"调制因子 {mod:.4f} 超出 [0.3, 2.0]"

    def test_internal_mod_direction(self):
        """内部调制方向：高稳定/乐观 → 快衰减；高焦虑/易怒 → 慢衰减。"""
        stable = np.zeros(10); stable[T_EMOTIONAL_STABILITY] = 1.0
        anxious = np.zeros(10); anxious[T_ANXIETY_PRONENESS] = 1.0
        angry = np.zeros(10); angry[T_ANGER_REACTIVITY] = 1.0
        optimistic = np.zeros(10); optimistic[T_OPTIMISM] = 1.0

        mod_stable = _compute_internal_personality_mod(stable)
        mod_anxious = _compute_internal_personality_mod(anxious)
        mod_angry = _compute_internal_personality_mod(angry)
        mod_opt = _compute_internal_personality_mod(optimistic)

        assert mod_stable > 1.0, f"稳定→快衰减: {mod_stable}"
        assert mod_opt > 1.0, f"乐观→快衰减: {mod_opt}"
        assert mod_anxious < 1.0, f"焦虑→慢衰减: {mod_anxious}"
        assert mod_angry < 1.0, f"易怒→慢衰减: {mod_angry}"

    def test_relationship_mod_direction(self):
        """关系调制方向：回避→快衰减；焦虑→慢衰减。"""
        avoidant = np.zeros(10); avoidant[T_ATTACHMENT_AVOIDANCE] = 1.0
        anxious = np.zeros(10); anxious[T_ATTACHMENT_ANXIETY] = 1.0

        mod_av = _compute_relationship_personality_mod(avoidant)
        mod_anx = _compute_relationship_personality_mod(anxious)

        assert mod_av > 1.0, f"回避→快衰减: {mod_av}"
        assert mod_anx < 1.0, f"焦虑→慢衰减: {mod_anx}"

    def test_personality_affects_recovery_rate(self, default_traits):
        """不同人格在同一受伤场景下恢复速度不同。"""
        sp = compute_rel_setpoint(default_traits)
        damaged = DAMAGED_REL.copy()

        dt = 72  # 3 天

        # 高回避人格
        high_avoid = default_traits.copy()
        high_avoid[T_ATTACHMENT_AVOIDANCE] = 0.8
        sp_av = compute_rel_setpoint(high_avoid)

        # 高焦虑人格
        high_anx = default_traits.copy()
        high_anx[T_ATTACHMENT_ANXIETY] = 0.8
        sp_anx = compute_rel_setpoint(high_anx)

        r_av = apply_time_decay_relationship(damaged, sp_av, high_avoid, dt)
        r_anx = apply_time_decay_relationship(damaged, sp_anx, high_anx, dt)

        # 注意：不同人格的 setpoint 不同，比较归一化恢复比例
        def recovery(state, sp, damaged):
            dev = abs(sp - damaged).sum()
            rec = abs(state - damaged).sum()
            return rec / dev if dev > 0 else 0

        rec_av = recovery(r_av, sp_av, damaged)
        rec_anx = recovery(r_anx, sp_anx, damaged)

        # 回避型恢复更快
        assert rec_av > rec_anx, (
            f"回避 {rec_av:.4f} 应快于焦虑 {rec_anx:.4f}"
        )


# ═══════════════════════════════════════════════════════════════
# 4. 时间曲线
# ═══════════════════════════════════════════════════════════════

class TestTimeCurve:
    """λ_eff 随 Δt 的幂律放缓。"""

    def test_lambda_decreases_with_delta(self, default_traits):
        """Δt 增加 → λ_eff 减小。"""
        p_mod = _compute_internal_personality_mod(default_traits)
        deltas = [0.1, 1, 6, 24, 72, 168, 720]
        lams_prev = np.inf
        for dt in deltas:
            lam = _compute_lambda_effective(
                DEFAULT_DECAY_CONFIG.internal_lambda, p_mod, dt,
                DEFAULT_DECAY_CONFIG.internal_time_curve_k,
            )
            assert lam.max() < lams_prev + 1e-12, f"λ_eff 在 Δt={dt} 时逆单调"
            lams_prev = lam.max()

    def test_very_long_delta_lambda_approaches_zero(self, default_traits):
        """Δt → ∞ 时 λ_eff → 0（但仍 > 0）。"""
        p_mod = _compute_internal_personality_mod(default_traits)
        lam = _compute_lambda_effective(
            DEFAULT_DECAY_CONFIG.internal_lambda, p_mod, 1e6,
            DEFAULT_DECAY_CONFIG.internal_time_curve_k,
        )
        assert lam.max() < 1e-3, f"长 Δt 下 λ_eff 应很小: {lam.max()}"
        assert lam.min() > 0, "λ_eff 应始终为正"

    def test_time_curve_difference(self):
        """内部曲线放缓(k=0.05) > 关系曲线放缓(k=0.001)。"""
        deltas = [1, 24, 168]
        for dt in deltas:
            internal_damping = 1.0 / (1.0 + 0.05 * dt)
            rel_damping = 1.0 / (1.0 + 0.001 * dt)
            assert internal_damping < rel_damping, (
                f"Δt={dt}: 内部阻尼 {internal_damping:.4f} < 关系 {rel_damping:.4f}"
            )


# ═══════════════════════════════════════════════════════════════
# 5. 边界稳健性
# ═══════════════════════════════════════════════════════════════

class TestBoundaryRobustness:
    """极端输入下的稳定性。"""

    def test_extreme_values_stay_in_bounds(self, default_traits):
        """极端输入值经衰减后仍在近 [-1, 1] 范围内。

        注意: soft_clamp 使用 sigmoid 平滑过渡，输入超出边界时
        输出可能略超 ±1（在过渡宽度内）。这与 soft_clamp 的设计一致。
        """
        sp_i = compute_setpoint(default_traits)
        sp_r = compute_rel_setpoint(default_traits)

        # soft_clamp 默认 transition=0.1，输出可到 low±0.1 / high±0.1
        SOFT_CLAMP_TOL = 0.1

        for dt in [0.5, 6, 72, 10000]:
            for val in [-1.5, -1.0, 1.0, 1.5]:
                extreme_i = np.full(I_SIZE, val, dtype=np.float64)
                extreme_r = np.full(R_SIZE, val, dtype=np.float64)

                r_i = apply_time_decay_internal(extreme_i, sp_i, default_traits, dt)
                r_r = apply_time_decay_relationship(extreme_r, sp_r, default_traits, dt)

                assert r_i.min() >= -1.0 - SOFT_CLAMP_TOL, (
                    f"内部下界 {r_i.min():.3f} 超出容忍 {SOFT_CLAMP_TOL} at dt={dt}, val={val}"
                )
                assert r_i.max() <= 1.0 + SOFT_CLAMP_TOL, (
                    f"内部上界 {r_i.max():.3f} 超出容忍 at dt={dt}, val={val}"
                )
                assert r_r.min() >= -1.0 - SOFT_CLAMP_TOL, (
                    f"关系下界 {r_r.min():.3f} 超出容忍 at dt={dt}, val={val}"
                )
                assert r_r.max() <= 1.0 + SOFT_CLAMP_TOL, (
                    f"关系上界 {r_r.max():.3f} 超出容忍 at dt={dt}, val={val}"
                )

    def test_all_traits_extremes(self):
        """极端人格下衰减不崩溃。"""
        for traits_arr in [
            np.full(10, -1.0),
            np.full(10, 1.0),
            np.random.uniform(-1, 1, 10),
        ]:
            sp_i = compute_setpoint(traits_arr)
            sp_r = compute_rel_setpoint(traits_arr)
            for dt in [0.1, 24, 720]:
                r_i = apply_time_decay_internal(
                    np.full(I_SIZE, 0.9), sp_i, traits_arr, dt,
                )
                r_r = apply_time_decay_relationship(
                    np.full(R_SIZE, 0.9), sp_r, traits_arr, dt,
                )
                assert r_i.min() >= -1.0 and r_i.max() <= 1.0
                assert r_r.min() >= -1.0 and r_r.max() <= 1.0

    def test_single_dimension_decay_isolation(self, default_traits):
        """只有一个维度偏离时，其他维度不串扰。"""
        sp = compute_rel_setpoint(default_traits)

        for test_dim in range(R_SIZE):
            perturbed = sp.copy()
            perturbed[test_dim] = sp[test_dim] - 0.6  # 仅该维度负向偏离

            dt = 96
            result = apply_time_decay_relationship(perturbed, sp, default_traits, dt)

            for d in range(R_SIZE):
                if d == test_dim:
                    continue  # 非测试维度应保持 setpoint
                assert abs(result[d] - sp[d]) < 1e-10, (
                    f"维度 {d} 被串扰：{result[d]:.6f} ≠ {sp[d]:.6f}"
                )


# ═══════════════════════════════════════════════════════════════
# 6. 批量统计（异常检测）
# ═══════════════════════════════════════════════════════════════

class TestBulkStatistics:
    """大规模随机场景下的分布监督和异常检测。

    检测项:
      - 全域违规（超出 [-1, 1]）
      - 维度级异常分布
      - 非对称比例稳定性
      - setpoint 收敛性
    """

    N_SAMPLES = 5000

    @pytest.fixture
    def bulk_data(self):
        """生成批量随机测试数据 (N_SAMPLES x dim)。"""
        rng = np.random.default_rng(42)
        traits = rng.uniform(-1, 1, size=(self.N_SAMPLES, 10))
        internal = rng.uniform(-1, 1, size=(self.N_SAMPLES, I_SIZE))
        relationship = rng.uniform(-1, 1, size=(self.N_SAMPLES, R_SIZE))
        deltas = rng.uniform(0.5, 720, size=self.N_SAMPLES)
        return traits, internal, relationship, deltas

    def test_no_boundary_violations(self, bulk_data):
        """所有样本衰减后值域 ∈ [-1, 1]。"""
        traits, internal, relationship, deltas = bulk_data
        violations_count = 0
        for i in range(self.N_SAMPLES):
            sp_i = compute_setpoint(traits[i])
            sp_r = compute_rel_setpoint(traits[i])
            r_i = apply_time_decay_internal(internal[i], sp_i, traits[i], deltas[i])
            r_r = apply_time_decay_relationship(relationship[i], sp_r, traits[i], deltas[i])
            if r_i.min() < -1.0 - 1e-10 or r_i.max() > 1.0 + 1e-10:
                violations_count += 1
            if r_r.min() < -1.0 - 1e-10 or r_r.max() > 1.0 + 1e-10:
                violations_count += 1

        assert violations_count == 0, f"{violations_count} 个边界违规"

    def test_setpoint_convergence(self, bulk_data):
        """极长时间 Δt 下状态收敛到 setpoint 的渐近误差包络内。"""
        rng = np.random.default_rng(42)
        internal = rng.uniform(-1, 1, size=(500, I_SIZE))
        relationship = rng.uniform(-1, 1, size=(500, R_SIZE))
        traits = rng.uniform(-1, 1, size=(500, 10))

        max_error_i = 0.0
        max_error_r = 0.0
        for i in range(500):
            sp_i = compute_setpoint(traits[i])
            sp_r = compute_rel_setpoint(traits[i])
            r_i = apply_time_decay_internal(internal[i], sp_i, traits[i], 10000)
            r_r = apply_time_decay_relationship(relationship[i], sp_r, traits[i], 10000)
            max_error_i = max(max_error_i, abs(r_i - sp_i).max())
            max_error_r = max(max_error_r, abs(r_r - sp_r).max())

        # 内部态：时间曲线 k=0.05 导致渐近残余 ≈ exp(-λ*p/k)
        # 最慢维度 (LONGING λ=0.12) 的渐近残余 ~ exp(-0.12/0.05) ≈ 0.09 → 9%
        # 加上极端人格调制范围 [0.3, 2.0]，最坏 ~ exp(-0.12*0.3/0.05) = exp(-0.72) = 0.49
        # 所以 max_error 可达 ~0.5
        assert max_error_i < 0.6, (
            f"内部最大收敛误差 {max_error_i:.4f}（渐近残余包络 ~0.5）"
        )

        # 关系态：k=0.001 也有渐近残余，最慢维 R_TRUST(λ=0.0014) 在 p_mod=0.3 时
        # asymp = exp(-0.0014*0.3/0.001) = exp(-0.42) ≈ 0.66，累积大量随机样本后可达 ~0.5
        assert max_error_r < 0.6, (
            f"关系最大收敛误差 {max_error_r:.4f}（渐近包络 ~0.5）"
        )

    def test_asymmetry_anomaly_detection(self, default_traits):
        """异常检测：不对称比例是否有异常值（如 NaN/Inf/负值）。"""
        sp = compute_rel_setpoint(default_traits)
        rng = np.random.default_rng(42)

        ratios = []
        for dim in range(R_SIZE):
            for _ in range(200):
                delta = rng.uniform(0.1, 0.9)
                neg_mag = max(-0.99, sp[dim] - delta)
                pos_mag = min(0.99, sp[dim] + delta)
                dt = rng.uniform(6, 336)

                neg_state = sp.copy(); neg_state[dim] = neg_mag
                pos_state = sp.copy(); pos_state[dim] = pos_mag

                r_neg = apply_time_decay_relationship(neg_state, sp, default_traits, dt)
                r_pos = apply_time_decay_relationship(pos_state, sp, default_traits, dt)

                nd = abs(r_neg[dim] - neg_mag)
                pd = abs(r_pos[dim] - pos_mag)
                if pd < 1e-12:
                    continue
                ratio = nd / pd

                assert np.isfinite(ratio), f"非有限 ratio: {ratio}"
                assert ratio > 0, f"非正 ratio: {ratio}"
                ratios.append(ratio)

        # 报告统计量
        ratios = np.array(ratios)
        stats = {
            "mean": ratios.mean(),
            "std": ratios.std(),
            "min": ratios.min(),
            "max": ratios.max(),
            "q25": np.percentile(ratios, 25),
            "q75": np.percentile(ratios, 75),
            "p05": np.percentile(ratios, 5),
            "p95": np.percentile(ratios, 95),
        }

        # 打印统计摘要（pytest -v 可看到）
        print(f"\n  不对称比例统计 (n={len(ratios)}):")
        for k, v in stats.items():
            print(f"    {k}: {v:.4f}")

        # 不应有极端异常值（boost=1.8 时，ratio 应在 0.5~5 之间）
        outliers = ratios[(ratios < 0.5) | (ratios > 5.0)]
        assert len(outliers) < 10, (
            f"发现 {len(outliers)} 个异常不对称比例: {outliers[:5]}"
        )

    def test_no_nan_in_outputs(self, default_traits):
        """所有场景下不产生 NaN。"""
        sp_i = compute_setpoint(default_traits)
        sp_r = compute_rel_setpoint(default_traits)

        for dt in [0.01, 0.5, 1, 24, 168, 720]:
            for extreme_state in [
                np.full(I_SIZE, 0.9),
                np.full(I_SIZE, -0.9),
            ]:
                r_i = apply_time_decay_internal(extreme_state, sp_i, default_traits, dt)
                assert not np.any(np.isnan(r_i)), f"内部 NaN at dt={dt}"

            for extreme_rel in [
                np.full(R_SIZE, 0.9),
                np.full(R_SIZE, -0.9),
            ]:
                r_r = apply_time_decay_relationship(extreme_rel, sp_r, default_traits, dt)
                assert not np.any(np.isnan(r_r)), f"关系 NaN at dt={dt}"

    def test_internal_no_asymmetry_in_bulk(self, default_traits):
        """批量验证内部状态衰减不包含非对称性。"""
        sp = compute_setpoint(default_traits)
        rng = np.random.default_rng(42)

        symmetric_ratios = []
        for _ in range(500):
            # 等距正负偏离
            offset = rng.uniform(0.1, 0.7)
            dim = rng.integers(0, I_SIZE)
            neg_state = sp.copy(); neg_state[dim] = sp[dim] - offset
            pos_state = sp.copy(); pos_state[dim] = sp[dim] + offset
            dt = rng.uniform(6, 168)

            r_neg = apply_time_decay_internal(neg_state, sp, default_traits, dt)
            r_pos = apply_time_decay_internal(pos_state, sp, default_traits, dt)

            nd = abs(r_neg[dim] - neg_state[dim])
            pd = abs(r_pos[dim] - pos_state[dim])
            if pd > 1e-12:
                symmetric_ratios.append(nd / pd)

        ratios = np.array(symmetric_ratios)
        mean = ratios.mean()
        assert 0.95 <= mean <= 1.05, (
            f"内部状态衰减不对称 mean={mean:.4f}，应 ≈1.0"
        )

    def test_convenience_api_alignment(self, default_traits, default_internal, default_relationship):
        """apply_time_decay 便捷接口与独立接口结果一致。"""
        sp_i = compute_setpoint(default_traits)
        sp_r = compute_rel_setpoint(default_traits)

        dt = 48
        combined = apply_time_decay(default_internal, default_relationship, default_traits, dt)
        independent_i = apply_time_decay_internal(default_internal, sp_i, default_traits, dt)
        independent_r = apply_time_decay_relationship(default_relationship, sp_r, default_traits, dt)

        np.testing.assert_array_equal(combined["internal_state"], independent_i)
        np.testing.assert_array_equal(combined["relationship_state"], independent_r)


# ═══════════════════════════════════════════════════════════════
# 7. 可视化（保存到 tests/result/）
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(True, reason="可视化：手动查看，不作为 CI 断言")
class TestVisualization:
    """生成可视化图表并保存到 tests/result/。

    通过 `pytest tests/test_decay.py::TestVisualization -v --no-header -s` 单独运行。
    """

    def _get_fig_dir(self):
        from pathlib import Path
        d = Path("tests/result")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_lambda_base_comparison(self):
        """图1: λ_base 对比 — 内部状态各维度衰减率。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["Energy", "Stress", "Loneliness", "Insecurity",
                   "Irritation", "Longing", "SocBat", "Fatigue"]
        lambdas = DEFAULT_DECAY_CONFIG.internal_lambda

        # 分类：正向/中性 vs 负向
        positive = [0, 5, 6]   # ENERGY, LONGING, SOCIAL_BATTERY
        negative = [1, 2, 3, 4, 7]  # STRESS, LONELINESS, INSECURITY, IRRITATION, FATIGUE

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#2ecc71" if i in positive else "#e74c3c" for i in range(len(labels))]
        bars = ax.bar(labels, lambdas, color=colors, edgecolor="gray", linewidth=0.5)

        ax.set_ylabel("λ_base (/hour)")
        ax.set_title("Internal State Decay Rates (λ_base)", fontsize=13)
        ax.axhline(y=0.35, color="gray", linestyle="--", alpha=0.5, label="Positive avg (~0.35)")
        ax.legend()

        # 在柱上标数值
        for bar, v in zip(bars, lambdas):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        fig.savefig(str(self._get_fig_dir() / "lambda_base_comparison.png"), dpi=150)
        plt.close(fig)
        print(f"  已保存: lambda_base_comparison.png")

    def test_asymmetric_decay_curves(self, default_traits):
        """图2: 非对称衰减曲线 — 信任受伤 vs 信任升温随时间演化。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sp = compute_rel_setpoint(default_traits)
        trust_sp = sp[R_TRUST]

        hours = np.linspace(0, 336, 100)  # 2 周
        damaged_trust = np.full(R_SIZE, trust_sp)
        damaged_trust[R_TRUST] = -0.6
        enhanced_trust = np.full(R_SIZE, trust_sp)
        enhanced_trust[R_TRUST] = 0.8

        damaged_curve = []
        enhanced_curve = []
        for dt in hours:
            r1 = apply_time_decay_relationship(damaged_trust, sp, default_traits, dt)
            r2 = apply_time_decay_relationship(enhanced_trust, sp, default_traits, dt)
            damaged_curve.append(r1[R_TRUST])
            enhanced_curve.append(r2[R_TRUST])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(hours / 24, damaged_curve, color="#e74c3c", linewidth=2,
                label='Trust Damaged (-0.6)')
        ax.plot(hours / 24, enhanced_curve, color="#2ecc71", linewidth=2,
                label='Trust Enhanced (+0.8)')
        ax.axhline(y=trust_sp, color="gray", linestyle="--", alpha=0.7,
                   label=f"Setpoint ({trust_sp:.2f})")

        # 标注半衰期
        damaged_dev = -0.6 - trust_sp
        enhanced_dev = 0.8 - trust_sp
        half_damaged = trust_sp + damaged_dev * 0.5
        half_enhanced = trust_sp + enhanced_dev * 0.5

        # 找最接近半衰期的时间点
        for curve, target, label, color in [
            (damaged_curve, half_damaged, "damaged half", "#c0392b"),
            (enhanced_curve, half_enhanced, "enhanced half", "#27ae60"),
        ]:
            idx = np.argmin(np.abs(np.array(curve) - target))
            ax.scatter(hours[idx] / 24, curve[idx], color=color, zorder=5, s=50)
            ax.annotate(f"{hours[idx]/24:.1f}d", (hours[idx]/24, curve[idx]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8, color=color)

        ax.set_xlabel("Time (days)")
        ax.set_ylabel("R_TRUST value")
        ax.set_title("Asymmetric Decay: Damaged vs Enhanced Trust\n"
                     "(Negative deviation recovers faster)", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(self._get_fig_dir() / "asymmetric_decay_curves.png"), dpi=150)
        plt.close(fig)
        print(f"  已保存: asymmetric_decay_curves.png")

    def test_personality_modulation_impact(self):
        """图3: 人格调制因子的分布与影响。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(42)
        internal_mods = []
        rel_mods = []
        for _ in range(5000):
            traits = rng.uniform(-1, 1, 10)
            internal_mods.append(_compute_internal_personality_mod(traits))
            rel_mods.append(_compute_relationship_personality_mod(traits))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(internal_mods, bins=50, color="#3498db", edgecolor="gray", alpha=0.8)
        axes[0].axvline(x=1.0, color="red", linestyle="--", alpha=0.5)
        axes[0].set_xlabel("Modulation Factor")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"Internal Personality Mod\n"
                          f"μ={np.mean(internal_mods):.3f}, σ={np.std(internal_mods):.3f}")

        axes[1].hist(rel_mods, bins=50, color="#9b59b6", edgecolor="gray", alpha=0.8)
        axes[1].axvline(x=1.0, color="red", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Modulation Factor")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Relationship Personality Mod\n"
                          f"μ={np.mean(rel_mods):.3f}, σ={np.std(rel_mods):.3f}")

        fig.tight_layout()
        fig.savefig(str(self._get_fig_dir() / "personality_modulation_impact.png"), dpi=150)
        plt.close(fig)
        print(f"  已保存: personality_modulation_impact.png")

    def test_time_curve_damping(self):
        """图4: 时间曲线阻尼 — λ_eff 随 Δt 的衰减。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        deltas = np.logspace(-1, 4, 200)  # 0.1h ~ 10000h

        internal_damping = 1.0 / (1.0 + DEFAULT_DECAY_CONFIG.internal_time_curve_k * deltas)
        rel_damping = 1.0 / (1.0 + DEFAULT_DECAY_CONFIG.relationship_time_curve_k * deltas)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(deltas, internal_damping, label=f"Internal (k={DEFAULT_DECAY_CONFIG.internal_time_curve_k})",
                color="#3498db", linewidth=2)
        ax.plot(deltas, rel_damping, label=f"Relationship (k={DEFAULT_DECAY_CONFIG.relationship_time_curve_k})",
                color="#9b59b6", linewidth=2)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax.text(deltas[-1], 0.52, "50% damping", fontsize=9, color="gray")

        ax.set_xscale("log")
        ax.set_xlabel("Δt (hours, log scale)")
        ax.set_ylabel("Damping Factor (1/(1 + k·Δt))")
        ax.set_title("Time Curve Damping: λ_eff decay over Δt", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(self._get_fig_dir() / "time_curve_damping.png"), dpi=150)
        plt.close(fig)
        print(f"  已保存: time_curve_damping.png")

    def test_anomaly_heatmap(self, default_traits):
        """图5: 异常值热力图 — 多维度·多Δt×多偏离距离下的残余偏差。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sp = compute_rel_setpoint(default_traits)
        dims = range(R_SIZE)
        deltas = np.array([6, 24, 72, 168, 336])
        offsets = np.array([0.2, 0.4, 0.6, 0.8])

        # 对于每个维度、每个 Δt、每个偏离量：计算负向 vs 正向残余偏差比
        fig, axes = plt.subplots(len(dims), 1, figsize=(12, 3 * len(dims)))

        for di, dim in enumerate(dims):
            ratio_matrix = np.zeros((len(offsets), len(deltas)))
            for oi, off in enumerate(offsets):
                neg_mag = max(-0.99, sp[dim] - off)
                pos_mag = min(0.99, sp[dim] + off)
                for ti, dt in enumerate(deltas):
                    neg_state = sp.copy(); neg_state[dim] = neg_mag
                    pos_state = sp.copy(); pos_state[dim] = pos_mag
                    r_neg = apply_time_decay_relationship(neg_state, sp, default_traits, dt)
                    r_pos = apply_time_decay_relationship(pos_state, sp, default_traits, dt)
                    nd = abs(r_neg[dim] - neg_mag)
                    pd = abs(r_pos[dim] - pos_mag)
                    ratio_matrix[oi, ti] = nd / pd if pd > 1e-12 else 0

            im = axes[di].imshow(ratio_matrix, aspect="auto", cmap="RdYlGn",
                                 vmin=0.5, vmax=3.0,
                                 extent=[deltas[0]-1, deltas[-1]+1, offsets[-1]+0.05, offsets[0]-0.05])
            axes[di].set_title(f"{R_LABELS[dim]} — Neg/Pos recovery ratio")
            axes[di].set_xlabel("Δt (hours)")
            axes[di].set_ylabel("Deviation offset")
            axes[di].set_xscale("log")
            for oi in range(len(offsets)):
                for ti in range(len(deltas)):
                    val = ratio_matrix[oi, ti]
                    axes[di].text(deltas[ti], offsets[oi], f"{val:.2f}",
                                  ha="center", va="center", fontsize=7,
                                  color="white" if val > 2.0 or val < 0.8 else "black")

        fig.colorbar(im, ax=axes, label="Neg/Pos ratio", shrink=0.8)
        fig.tight_layout()
        fig.savefig(str(self._get_fig_dir() / "anomaly_heatmap.png"), dpi=150)
        plt.close(fig)
        print(f"  已保存: anomaly_heatmap.png")

    def test_asymmetry_boost_sweep(self, default_traits):
        """图6: boost 参数扫描 — 不同 boost 下受伤信任恢复曲线。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sp = compute_rel_setpoint(default_traits)

        hours = np.linspace(0, 336, 100)
        damaged = sp.copy()
        damaged[R_TRUST] = -0.6

        fig, ax = plt.subplots(figsize=(10, 6))

        boosts = [1.0, 1.5, 1.8, 2.5, 4.0]
        colors = ["#95a5a6", "#f39c12", "#e74c3c", "#c0392b", "#8e44ad"]

        for boost, color in zip(boosts, colors):
            config = replace(DEFAULT_DECAY_CONFIG, negative_decay_boost=boost)
            curve = []
            for dt in hours:
                r = apply_time_decay_relationship(damaged, sp, default_traits, dt, config)
                curve.append(r[R_TRUST])
            ax.plot(hours / 24, curve, color=color, linewidth=2, label=f"boost={boost}")

        ax.axhline(y=sp[R_TRUST], color="gray", linestyle="--", alpha=0.5,
                   label=f"Setpoint ({sp[R_TRUST]:.2f})")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("R_TRUST value")
        ax.set_title("Boost Parameter Sweep: Recovery from Trust Damage (-0.6)", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(self._get_fig_dir() / "boost_sweep.png"), dpi=150)
        plt.close(fig)
        print(f"  已保存: boost_sweep.png")
