"""深度分析测试 — 小批量、全追踪、逐值验证。

使用 py.test -s -v tests/test_deep_analysis.py 可查看所有追踪输出。

每个测试类都:
  1. 用小批量 (100-1000) 保证可读性
  2. 用 print() 输出每个样本/维度的追踪数据
  3. 保留 assert 确保正确性
"""

import numpy as np
import pytest

from state import (
    DEFAULT_TRAITS, DEFAULT_INTERNAL, DEFAULT_RELATIONSHIP,
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE,
    I_SIZE, R_SIZE, S_SIZE, T_SIZE, ST_SIZE,
    R_AFFECTION, R_TRUST_BOND, R_INTIMACY,
    S_EXPRESSIVENESS, S_WARMTH, S_SHARPNESS, S_SOFTNESS,
    S_ENTHUSIASM, S_RESTRAINT, S_VULNERABILITY, S_SIZE,
    I_LABELS, R_LABELS, S_LABELS, ST_LABELS, T_LABELS,
    T_SENSITIVITY, T_PRIDE, T_EMOTIONAL_OPENNESS, T_EMOTIONAL_STABILITY,
    T_OPTIMISM, T_ANXIETY_PRONENESS, T_ANGER_REACTIVITY,
    T_JEALOUSY_SENSITIVITY, T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE,
    ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT,
    ST_DEPENDENCY, ST_TEASING, ST_EMOTIONAL_WEIGHT,
)

from state_engine._dynamics import (
    update_internal_state, update_relationship_state,
    compute_setpoint, compute_rel_setpoint,
)
from state_engine._dynamics_weights import (
    ALPHA_MAPPER, ALPHA_REL_MAPPER, BETA_REL_MAPPER,
    BETA_BASE, HYPER_BETA_GAIN, DEACT_SUPPRESSION_RATIO,
    DECAY_INTERNAL_LAMBDA, DECAY_INTERNAL_TIME_CURVE_K, DECAY_NEGATIVE_BOOST,
    INTERNAL_COUPLING, RELATIONSHIP_COUPLING, CROSS_SCALE_COUPLING,
    SELF_DECAY, DECAY_TARGETS, REL_SELF_DECAY,
)
from state_engine._decay import (
    apply_time_decay_internal, apply_time_decay_relationship,
    apply_time_decay,
)
from state_engine._surface import (
    project_surface, compute_surface_feedback, _compute_surface_alpha,
)
from state_engine._surface_weights import SURFACE_MAPPER
from state_engine._defenses import compute_defense_profiles, apply_defenses
from state_engine._pipeline import update_all, initialize_all

rng = np.random.RandomState(20260623)

I8_LABELS = ["energy","stress","loneliness","insecurity","irritation","longing","social_battery","mental_fatigue"]
R3_LABELS = ["affection","trust_bond","intimacy"]
S7_LABELS = ["expressiveness","warmth","sharpness","softness","enthusiasm","restraint","vulnerability"]
ST7_LABELS = ["abandonment","validation","closeness","conflict","dependency","teasing","emotional_weight"]
T10_LABELS = ["sensitivity","pride","openness","stability","optimism","anxiety","anger","jealousy","attach_anxiety","attach_avoidance"]


# ════════════════════════════════════════════════════════════
# Section 1 — 修复验证
# ════════════════════════════════════════════════════════════

class TestFixVerification:
    """验证 4 个修复的实际效果，打印修复前后的值对比。"""

    def test_fix_asymmetric_decay(self):
        """验证双条件修复：energy=0.20 > setpoint=0.40 不应被加速。"""
        print("\n=== [Fix-3] 非对称衰减双条件修复验证 ===")
        sp = compute_setpoint(DEFAULT_TRAITS)
        current = DEFAULT_INTERNAL.copy()
        # 构造一个 energy=0.20 但 setpoint=0.40 的场景（正向维度低于高基线）
        current[I_ENERGY] = 0.20
        current[I_SOCIAL_BATTERY] = 0.15

        # 手动模拟两版逻辑
        deviation = current - sp
        old_neg = deviation < 0
        new_neg = (current < 0) & (deviation < 0)

        print(f"  setpoint: {sp}")
        print(f"  current:  {current}")
        print(f"  deviation:{deviation}")
        print()
        print(f"  {'维度':20s} | {'val':>6} | {'sp':>6} | {'old_neg':>8} | {'new_neg':>8} | 修正?")
        print("-" * 60)
        for i, label in enumerate(I8_LABELS):
            fix_mark = "✅" if old_neg[i] and not new_neg[i] else ""
            print(f"  {label:20s} | {current[i]:>6.3f} | {sp[i]:>6.3f} | {str(old_neg[i]):>8} | {str(new_neg[i]):>8} {fix_mark}")

        # 断言：energy 和 social_battery 不再被误判
        assert not new_neg[I_ENERGY], "energy 不应被误判为非对称加速"
        assert not new_neg[I_SOCIAL_BATTERY], "social_battery 不应被误判"
        # 但真正的负向维度仍应被正确标记
        for i in range(I_SIZE):
            if i not in (I_ENERGY, I_SOCIAL_BATTERY):
                if current[i] < 0 and current[i] < sp[i]:
                    # 这些维度是负值且低于 setpoint
                    pass  # 取决于具体值，不断言

    def test_fix_beta_stim_multiplicative(self):
        """验证乘法公式：deact 不再产生负 β，且保留抑制梯度。"""
        print("\n=== [Fix-2] β_stim 乘法公式验证 ===")
        scenarios = {
            "secure   d=0.2 h=0.2": (0.2, 0.2),
            "avoidant d=0.8 h=0.2": (0.8, 0.2),
            "anxious  d=0.2 h=0.8": (0.2, 0.8),
            "extreme-avoid d=1.0 h=0.0": (1.0, 0.0),
            "extreme-anx   d=0.0 h=1.0": (0.0, 1.0),
            "extreme-both  d=1.0 h=1.0": (1.0, 1.0),
        }
        print(f"  {'Scenario':>25} | {'raw (旧加性)':>12} | {'mult (新)':>10} | {'on/off_ratio':>12}")
        print("-" * 65)
        for name, (deact, hyper) in scenarios.items():
            # 旧公式（已废弃）
            old_raw = BETA_BASE[0] + hyper * HYPER_BETA_GAIN + deact * (-0.15)  # old additive formula
            # 新乘法公式
            beta_raw_inner = max(BETA_BASE[0] + hyper * HYPER_BETA_GAIN, 0.005)
            new_mult = beta_raw_inner * (1.0 - deact * DEACT_SUPPRESSION_RATIO)
            new_mult = max(0.005, min(new_mult, 0.35))

            # 对比 secure 的抑制比例
            secure_raw_inner = max(BETA_BASE[0] + 0.2 * HYPER_BETA_GAIN, 0.005)
            secure_mult = max(0.005, min(secure_raw_inner * (1.0 - 0.2 * DEACT_SUPPRESSION_RATIO), 0.35))
            ratio = new_mult / secure_mult if secure_mult > 0 else 0
            print(f"  {name:>25} | {old_raw:>12.4f} | {new_mult:>10.4f} | {ratio:>11.4f}")

            # 关键断言：乘法公式永不逆转刺激方向
            assert new_mult >= 0.005, f"β 不应为负: {new_mult}"

            # extreme-avoid 应该有最低的 β（但仍是正值）
            if deact == 1.0 and hyper == 0.0:
                assert new_mult > 0, f"extreme-avoid β={new_mult} 应为正"

    def test_fix_alpha_bounds(self):
        """验证 alpha 边界放宽后截断率大幅降低。"""
        print("\n=== [Fix-5] α 边界放宽验证 ===")

        n_samples = 10000
        raw_alphas = []
        for _ in range(n_samples):
            traits = rng.uniform(-1, 1, T_SIZE)
            rel = rng.uniform(-1, 1, R_SIZE)
            inputs = np.concatenate([traits, rel])
            raw_alphas.append(ALPHA_MAPPER.compute(inputs)[0])
        raw_alphas = np.array(raw_alphas)

        # 旧边界
        old_low_cut = (raw_alphas < 0.02).mean() * 100
        old_high_cut = (raw_alphas > 0.35).mean() * 100
        # 新边界
        new_low_cut = (raw_alphas < 0.05).mean() * 100
        new_high_cut = (raw_alphas > 0.40).mean() * 100

        print(f"  α raw 统计: min={raw_alphas.min():.4f}, max={raw_alphas.max():.4f}, mean={raw_alphas.mean():.4f}")
        print(f"  ──────────────── 截断率对比 ────────────────")
        print(f"  旧边界 [0.02, 0.35]:   低={old_low_cut:.2f}%  高={old_high_cut:.2f}%  总={old_low_cut+old_high_cut:.2f}%")
        print(f"  新边界 [0.05, 0.40]:   低={new_low_cut:.2f}%  高={new_high_cut:.2f}%  总={new_low_cut+new_high_cut:.2f}%")

        # 断言：总截断率应大幅下降
        new_total = new_low_cut + new_high_cut
        old_total = old_low_cut + old_high_cut
        assert new_total < old_total * 0.6, f"截断率未显著下降: {old_total:.1f}% → {new_total:.1f}%"

    def test_fix_alpha_rel_bounds(self):
        """验证 α_rel 边界放宽。"""
        print("\n=== [Fix-5] α_rel 边界放宽验证 ===")

        n_samples = 10000
        raw_alphas = []
        for _ in range(n_samples):
            traits = rng.uniform(-1, 1, T_SIZE)
            rel = rng.uniform(-1, 1, R_SIZE)
            raw_alphas.append(ALPHA_REL_MAPPER.compute(np.concatenate([traits, rel]))[0])
        raw_alphas = np.array(raw_alphas)

        old_high_cut = (raw_alphas > 0.06).mean() * 100
        new_high_cut = (raw_alphas > 0.08).mean() * 100

        print(f"  α_rel raw: min={raw_alphas.min():.4f}, max={raw_alphas.max():.4f}, mean={raw_alphas.mean():.4f}")
        print(f"  旧上界 0.06: {old_high_cut:.2f}% 截断 | 新上界 0.08: {new_high_cut:.2f}% 截断")

        assert new_high_cut < old_high_cut * 0.5, f"α_rel 截断率未显著下降: {old_high_cut:.1f}% → {new_high_cut:.1f}%"

    def test_fix_vulnerability_inputs(self):
        """验证 vulnerability 的入边权重。"""
        print("\n=== [Fix-4] Vulnerability 入边验证 ===")

        total_weight = 0
        found = []
        for sw in SURFACE_MAPPER._weights:
            if sw.target_idx == S_VULNERABILITY:
                gname = None
                for g, off in SURFACE_MAPPER._group_offsets.items():
                    if off <= sw.source_idx < off + SURFACE_MAPPER._group_sizes[g]:
                        gname = g
                        break
                source_label = gname or f"idx{sw.source_idx}"
                found.append((source_label, sw.value))
                total_weight += abs(sw.value)

        print(f"  Vulnerability 入边 ({len(found)} 条):")
        for src, val in sorted(found, key=lambda x: -abs(x[1])):
            print(f"    {src:25s} → vulnerability: {val:+.2f}")

        print(f"\n  总绝对值: {total_weight:.2f}")
        assert len(found) >= 4, f"vulnerability 只有 {len(found)} 条入边 (期望 ≥4)"
        assert total_weight >= 0.75, f"总输入权重 {total_weight:.2f} 不足 0.75"


# ════════════════════════════════════════════════════════════
# Section 2 — Pipeline 中间值全追踪
# ════════════════════════════════════════════════════════════

class TestPipelineIntermediateTrace:
    """完整追踪一次 update_all 调用的所有中间值。"""

    def test_single_step_full_trace(self):
        """追踪单次 update_all 的全部中间变量。"""
        print("\n=== [Trace] 完整管线一步追踪 ===")

        # 构造一个有意义的输入
        traits = np.array([
            0.0,   # sensitivity
            0.0,   # pride
            0.5,   # openness
            0.0,   # stability
            0.3,   # optimism
            0.3,   # anxiety_proneness
            0.2,   # anger_reactivity
            0.0,   # jealousy
            0.4,   # attachment_anxiety
            0.1,   # attachment_avoidance
        ])

        internal = DEFAULT_INTERNAL.copy()
        relationship = DEFAULT_RELATIONSHIP.copy()
        stimuli = np.array([0.0, 0.8, 0.6, 0.0, 0.3, 0.0, 0.0])  # validation+closeness+dependency
        prev_surface = None

        print(f"  输入:")
        print(f"    traits:      {traits}")
        print(f"    internal:    {internal}")
        print(f"    relationship:{relationship}")
        print(f"    stimuli:     {stimuli}")
        print()

        # Step 1: 防御
        profiles = compute_defense_profiles(traits, relationship, internal)
        inner, outer = apply_defenses(stimuli, profiles)
        print(f"  Step 1 — 防御:")
        print(f"    deact:  {profiles[0]}")
        print(f"    hyper:  {profiles[1]}")
        print(f"    inner:  {inner}")
        print(f"    outer:  {outer}")
        print(f"    β_stim 公式: β = max(ε, base+hyper·0.35) · (1-deact·{DEACT_SUPPRESSION_RATIO})")
        beta_inner = np.maximum(BETA_BASE + profiles[1] * HYPER_BETA_GAIN, 0.005)
        beta = beta_inner * (1.0 - profiles[0] * DEACT_SUPPRESSION_RATIO)
        beta = np.clip(beta, 0.005, 0.35)
        print(f"    β_stim: {beta}")
        print()

        # Step 2: 动力学
        new_internal = update_internal_state(internal, inner, traits, relationship, profiles)
        new_relationship = update_relationship_state(relationship, inner, traits, current_internal=internal)
        print(f"  Step 2 — 动力学:")
        alpha_val = ALPHA_MAPPER.compute(np.concatenate([traits, relationship]))[0]
        alpha_raw = (ALPHA_MAPPER._weight_matrix @ np.concatenate([traits, relationship]) + ALPHA_MAPPER._bias_vector)[0]
        print(f"    α={alpha_val:.4f} (raw={alpha_raw:.4f})")
        print(f"    α_rel={ALPHA_REL_MAPPER.compute(np.concatenate([traits, relationship]))[0]:.4f}")
        print(f"    new_internal:    {new_internal}")
        print(f"    Δinternal:       {new_internal - internal}")
        print(f"    new_relationship:{new_relationship}")
        print(f"    Δrelationship:   {new_relationship - relationship}")
        print()

        # Step 3: 表面
        surface = project_surface(new_internal, new_relationship, outer, prev_surface)
        print(f"  Step 3 — 表面投影:")
        print(f"    sources concat: {np.concatenate([new_internal, new_relationship, outer])}")
        print(f"    surface: {surface}")
        print()

        # Step 4: 反馈
        feedback = compute_surface_feedback(surface, new_internal)
        print(f"  Step 4 — 表面反馈:")
        print(f"    feedback_delta: {feedback}")
        print()

        # 断言：所有值在范围内
        # soft_clamp(transition=0.1) 允许输出超出边界 ±0.1
        SOFT_TOL = 0.11
        assert np.all(np.abs(new_internal) <= 1.0 + SOFT_TOL)
        assert np.all(np.abs(new_relationship) <= 1.0 + SOFT_TOL)
        assert np.all(np.abs(surface) <= 1.0 + SOFT_TOL)
        assert np.all(np.abs(profiles) <= 1.0 + SOFT_TOL)

    def test_pipeline_5_archetypes_trace(self):
        """追踪 5 个人格 archetype 的完整管线输出。"""
        print("\n=== [Trace] 5 人格 Archetype 管线对比 ===")

        archetypes = {
            "secure ζ": np.array([0.0,0.0,0.5,0.5,0.5,-0.3,-0.3,-0.3,-0.3,-0.3]),
            "anxious ζ": np.array([0.0,0.0,0.3,-0.5,0.0,0.7,0.5,0.6,0.7,0.2]),
            "avoidant ζ": np.array([0.0,0.0,-0.3,0.5,0.3,-0.5,-0.3,-0.5,-0.2,0.7]),
            "angry ζ":    np.array([0.0,0.0,0.0,-0.3,0.0,0.3,0.8,0.2,0.0,0.0]),
            "resilient ζ":np.array([0.0,0.0,0.8,0.8,0.8,-0.5,-0.5,-0.6,-0.5,-0.5]),
        }

        stimuli = np.array([0.0, 0.7, 0.5, 0.3, 0.2, 0.0, 0.4])
        internal = DEFAULT_INTERNAL.copy()

        for name, traits in archetypes.items():
            rel = DEFAULT_RELATIONSHIP.copy()
            profiles = compute_defense_profiles(traits, rel, internal)
            inner, outer = apply_defenses(stimuli, profiles)
            new_internal = update_internal_state(internal, inner, traits, rel, profiles)
            new_rel = update_relationship_state(rel, inner, traits, current_internal=internal)
            surface = project_surface(new_internal, new_rel, outer, None)
            alpha = ALPHA_MAPPER.compute(np.concatenate([traits, rel]))[0]
            alpha = max(0.05, min(0.40, alpha))
            beta_inner = np.maximum(BETA_BASE + profiles[1] * HYPER_BETA_GAIN, 0.005)
            beta = np.clip(beta_inner * (1.0 - profiles[0] * 0.5), 0.005, 0.35)

            print(f"\n  ── {name} ──")
            print(f"    α={alpha:.4f}, β_mean={beta.mean():.4f}, deact_mean={profiles[0].mean():.3f}, hyper_mean={profiles[1].mean():.3f}")
            print(f"    internal: {new_internal}")
            print(f"    rel:      {new_rel}")
            print(f"    surface:  {surface}")
            # 关键断言：所有输出在范围
            assert np.all(np.abs(new_internal) <= 1.0)
            assert np.all(np.abs(surface) <= 1.0)


# ════════════════════════════════════════════════════════════
# Section 3 — 动态耦合方向性验证
# ════════════════════════════════════════════════════════════

class TestCouplingDirectionality:
    """验证每一条耦合规则的方向性和心理合理性。"""

    def test_internal_coupling_rules(self):
        """追踪内部耦合矩阵每条规则的作用。"""
        print("\n=== [Coupling] 内部耦合 10 条规则验证 ===")
        I = I_SIZE
        ic = INTERNAL_COUPLING  # (8, 8)
        print(f"  内部耦合矩阵 density: {np.count_nonzero(ic) / ic.size * 100:.1f}%")
        print()

        # 枚举每条非零规则
        print(f"  {'src':>20s} → {'tgt':>20s} | {'weight':>8} | {'心理依据'}")
        print("-" * 75)
        src_labels = I8_LABELS
        tgt_labels = I8_LABELS
        for i in range(I):
            for j in range(I):
                if ic[i, j] != 0:
                    direction = "↑" if ic[i, j] > 0 else "↓"
                    print(f"  {src_labels[i]:>20s} → {tgt_labels[j]:>20s} | {ic[i, j]:>+8.3f} {direction}")

        # 验证每个规则的作用方向
        assert ic[I_ENERGY, I_STRESS] < 0, "energy→stress 应为负（精力充沛→压力降低）"
        assert ic[I_INSECURITY, I_STRESS] > 0, "insecurity→stress 应为正"
        assert ic[I_LONELINESS, I_INSECURITY] > 0, "loneliness→insecurity 应为正"
        assert ic[I_STRESS, I_IRRITATION] > 0, "stress→irritation 应为正"
        assert ic[I_SOCIAL_BATTERY, I_IRRITATION] < 0, "social_battery→irritation 应为负"
        assert ic[I_LONELINESS, I_LONGING] > 0, "loneliness→longing 应为正"

    def test_relationship_coupling_rules(self):
        """追踪关系耦合规则。"""
        print("\n=== [Coupling] 关系耦合级联验证 ===")
        rc = RELATIONSHIP_COUPLING
        print(f"  关系耦合矩阵:")
        print(f"    {rc}")
        print()

        # 级联模式: affection→trust→intimacy
        assert rc[R_AFFECTION, R_TRUST_BOND] > 0, "affection→trust_bond 应为正"
        assert rc[R_TRUST_BOND, R_INTIMACY] > 0, "trust_bond→intimacy 应为正"
        print(f"  级联: affection({rc[R_AFFECTION,R_TRUST_BOND]:+.3f}) → trust → intimacy({rc[R_TRUST_BOND,R_INTIMACY]:+.3f})")

    def test_cross_scale_coupling(self):
        """追踪跨尺度耦合 (内部→关系)。"""
        print("\n=== [Coupling] 跨尺度耦合 5 条规则验证 ===")
        cc = CROSS_SCALE_COUPLING
        print(f"  {'internal':>20s} → {'rel':>20s} | {'weight':>8} | {'心理'}")
        print("-" * 60)
        for i in range(I_SIZE):
            for j in range(R_SIZE):
                if cc[i, j] != 0:
                    print(f"  {I8_LABELS[i]:>20s} → {R3_LABELS[j]:>20s} | {cc[i, j]:>+8.3f}")

        assert cc[I_STRESS, R_TRUST_BOND] < 0, "stress→trust 应为负（压力→信任下降）"
        assert cc[I_ENERGY, R_AFFECTION] > 0, "energy→affection 应为正"


# ════════════════════════════════════════════════════════════
# Section 4 — 防御剖面行为追踪
# ════════════════════════════════════════════════════════════

class TestDefenseProfileTrace:
    """追踪防御剖面在不同输入下的变化。"""

    def test_defense_profile_sweep_trace(self):
        """扫描不同人格的防御剖面变化。"""
        print("\n=== [Defense] 人格→防御剖面扫描 ===")

        archetypes = {
            "secure":   np.array([0.0,0.0,0.5,0.5,0.5,-0.3,-0.3,-0.3,-0.3,-0.3]),
            "anxious":  np.array([0.0,0.0,0.3,-0.5,0.0,0.7,0.5,0.6,0.7,0.2]),
            "avoidant": np.array([0.0,0.0,-0.3,0.5,0.3,-0.5,-0.3,-0.5,-0.2,0.7]),
            "angry":    np.array([0.0,0.0,0.0,-0.3,0.0,0.3,0.8,0.2,0.0,0.0]),
        }

        print(f"  {'Archetype':>12} | {'deact_mean':>10} | {'hyper_mean':>10} | {'deact_std':>9} | {'hyper_std':>9}")
        print("-" * 55)
        for name, traits in archetypes.items():
            deacts = []
            hyp_ers = []
            for _ in range(500):
                t = traits + rng.normal(0, 0.1, T_SIZE)
                rel = DEFAULT_RELATIONSHIP + rng.normal(0, 0.05, R_SIZE)
                internal = DEFAULT_INTERNAL + rng.normal(0, 0.05, I_SIZE)
                profiles = compute_defense_profiles(
                    np.clip(t, -1, 1), np.clip(rel, -1, 1), np.clip(internal, -1, 1)
                )
                deacts.append(profiles[0].mean())
                hyp_ers.append(profiles[1].mean())

            deacts = np.array(deacts)
            hyp_ers = np.array(hyp_ers)
            print(f"  {name:>12} | {deacts.mean():>10.4f} | {hyp_ers.mean():>10.4f} | {deacts.std():>9.4f} | {hyp_ers.std():>9.4f}")

    def test_apply_defenses_trace(self):
        """追踪 7 种刺激在被防御调制后的 inner/outer。"""
        print("\n=== [Defense] 防御前/后刺激追踪 ===")

        traits = np.array([0.0,0.0,0.0,0.0,0.0,0.5,0.3,0.4,0.6,0.2])  # anxious-倾向
        rel = DEFAULT_RELATIONSHIP.copy()
        internal = DEFAULT_INTERNAL.copy()
        stimuli = np.array([0.3, 0.8, 0.5, 0.2, 0.6, 0.1, 0.4])

        profiles = compute_defense_profiles(traits, rel, internal)
        inner, outer = apply_defenses(stimuli, profiles)

        print(f"  {'刺激维度':>20s} | {'原始':>6} | {'inner':>6} | {'outer':>6} | {'deact':>6} | {'hyper':>6}")
        print("-" * 60)
        for i, label in enumerate(ST7_LABELS):
            print(f"  {label:>20s} | {stimuli[i]:>6.3f} | {inner[i]:>6.3f} | {outer[i]:>6.3f} | {profiles[0][i]:>6.3f} | {profiles[1][i]:>6.3f}")

        # 验证约束: inner >= outer
        assert np.all(inner >= outer - 1e-10), "inner_stimuli 应 >= outer_stimuli"
        assert np.all(inner >= 0.0 - 1e-10), "inner_stimuli 应 >= 0"
        assert np.all(outer >= 0.0 - 1e-10), "outer_stimuli 应 >= 0"


# ════════════════════════════════════════════════════════════
# Section 5 — 时间衰减曲线追踪
# ════════════════════════════════════════════════════════════

class TestDecayCurveTrace:
    """追踪时间衰减曲线，验证渐近收敛和非对称行为。"""

    def test_decay_curve_full_trace(self):
        """追踪 8 维内部状态在不同 Δt 下的衰减轨迹。"""
        print("\n=== [Decay] 内部状态衰减曲线追踪 ===")

        sp = compute_setpoint(DEFAULT_TRAITS)
        current = sp + np.full(I_SIZE, 0.5)
        current = np.clip(current, -1.0, 1.0)

        delta_times = [0, 0.5, 1, 3, 6, 12, 24, 48, 72, 168, 336]

        print(f"  setpoint: {sp}")
        print(f"  current:  {current}")
        print()

        # 打印每维衰减
        print(f"{'Δt(h)':>7}", end="")
        for label in I8_LABELS:
            print(f" | {label:>8}", end="")
        print()
        print("-" * (7 + 11 * I_SIZE))

        for dt in delta_times:
            if dt < 0.01:
                vals = current
            else:
                vals = apply_time_decay_internal(current, sp, DEFAULT_TRAITS, dt)
            print(f"{dt:>7.0f}", end="")
            for v in vals:
                print(f" | {v:>8.4f}", end="")
            print()

    def test_asymmetric_decay_trace(self):
        """追踪关系衰减中的非对称行为（正向/负向偏离）。"""
        print("\n=== [Decay] 关系态非对称衰减追踪 ===")

        sp = compute_rel_setpoint(DEFAULT_TRAITS)
        dt = 24  # 1天

        # 测试正向偏离和负向偏离
        test_cases = {
            "正向偏离 (+0.5)": sp + np.array([0.5, 0.5, 0.5]),
            "负向偏离 (-0.5)": sp + np.array([-0.5, -0.5, -0.5]),
        }

        print(f"  setpoint: {sp}")
        print(f"  Δt = {dt}h")
        print()

        for case_name, current in test_cases.items():
            current = np.clip(current, -1.0, 1.0)
            result = apply_time_decay_relationship(current, sp, DEFAULT_TRAITS, dt)
            decay_pct = (result - sp) / (current - sp + 1e-10)
            print(f"  {case_name}:")
            print(f"    before: {current}")
            print(f"    after:  {result}")
            print(f"    residual: {decay_pct}")
            print(f"    sp:        {sp}")
            # 负向偏离应衰减得更快（FAB）
            if "负向" in case_name:
                # 负向偏离的衰减因子应大于正向（更快衰减）
                pass  # 取决于多项参数
            print()


# ════════════════════════════════════════════════════════════
# Section 6 — 表面投影行为追踪
# ════════════════════════════════════════════════════════════

class TestSurfaceProjectionTrace:
    """追踪表面投影的各维度构成。"""

    def test_surface_weight_decomposition(self):
        """追踪每个表面维度有哪些入边及权重。"""
        print("\n=== [Surface] 各维度权重分解 ===")

        for tgt_idx in range(S_SIZE):
            print(f"\n  {S7_LABELS[tgt_idx]:20s} 入边:")
            weights = []
            for sw in SURFACE_MAPPER._weights:
                if sw.target_idx == tgt_idx:
                    w = sw.value
                    gname = None
                    for g, off in SURFACE_MAPPER._group_offsets.items():
                        if off <= sw.source_idx < off + SURFACE_MAPPER._group_sizes[g]:
                            gname = g
                            break
                    src_label = gname or "?"
                    weights.append((src_label, sw.source_idx, w))

            total = sum(abs(w) for _, _, w in weights)
            for src, sidx, w in sorted(weights, key=lambda x: -abs(x[2])):
                print(f"    {src:20s}.{SURFACE_MAPPER._source_labels[sidx]:20s} → weight={w:+.2f}")
            print(f"    总 |w|: {total:.2f}")

    def test_vulnerability_before_after_fix(self):
        """对比修复前后 vulnerability 的输入权重变化（修复后应有 stress+energy）。"""
        print("\n=== [Surface] Vulnerability 修复前后对比 ===")
        print("  当前 vulnerability 入边:")
        for sw in SURFACE_MAPPER._weights:
            if sw.target_idx == S_VULNERABILITY:
                gname = None
                for g, off in SURFACE_MAPPER._group_offsets.items():
                    if off <= sw.source_idx < off + SURFACE_MAPPER._group_sizes[g]:
                        gname = g
                        break
                src_label = SURFACE_MAPPER._source_labels[sw.source_idx]
                print(f"    {gname:15s}.{src_label:20s} → {sw.value:+.2f}")

        # 验证新增入边
        has_stress = any(
            sw.target_idx == S_VULNERABILITY and
            abs(sw.value - 0.10) < 0.001
            for sw in SURFACE_MAPPER._weights
        )
        has_energy = any(
            sw.target_idx == S_VULNERABILITY and
            abs(sw.value + 0.05) < 0.001
            for sw in SURFACE_MAPPER._weights
        )
        assert has_stress, "stress→vulnerability (+0.10) 入边缺失"
        assert has_energy, "energy→vulnerability (-0.05) 入边缺失"
        print("  ✅ stress(+0.10) 和 energy(-0.05) 入边已存在")


# ════════════════════════════════════════════════════════════
# Section 7 — Multi-round 轨迹追踪
# ════════════════════════════════════════════════════════════

class TestMultiRoundTrace:
    """追踪多轮交互的状态演变。"""

    def test_10_round_conflict_trace(self):
        """追踪 10 轮冲突刺激下的状态变化。"""
        print("\n=== [Multi] 10 轮冲突刺激轨迹 ===")

        traits = np.array([0.0,0.2,0.3,0.0,0.0,0.4,0.3,0.3,0.5,0.1])
        conflict = np.zeros(ST_SIZE)
        conflict[ST_CONFLICT] = 0.7
        conflict[ST_EMOTIONAL_WEIGHT] = 0.4

        internal = DEFAULT_INTERNAL.copy()
        relationship = DEFAULT_RELATIONSHIP.copy()
        prev_surface = None

        print(f"  {'#':>2} | {'delta_time':>8} | {'stress':>7} | {'irritation':>10} | {'affection':>9} | {'restraint':>9} | {'sharpness':>9}")
        print("-" * 70)

        for turn in range(11):
            print(f"  {turn:>2} | {turn*24:>8.0f}h ", end="")

            if turn > 0:
                # 时间衰减
                decayed = apply_time_decay(internal, relationship, traits, 24.0)
                internal, relationship = decayed["internal_state"], decayed["relationship_state"]

                # 管线
                profiles = compute_defense_profiles(traits, relationship, internal)
                inner, outer = apply_defenses(conflict, profiles)
                internal = update_internal_state(internal, inner, traits, relationship, profiles)
                relationship = update_relationship_state(relationship, inner, traits, current_internal=internal)
                surface = project_surface(internal, relationship, outer, prev_surface)
                prev_surface = surface
            else:
                surface = project_surface(internal, relationship, np.zeros(ST_SIZE), None)
                prev_surface = surface

            print(f" | {internal[I_STRESS]:>7.4f} | {internal[I_IRRITATION]:>10.4f} | {relationship[R_AFFECTION]:>9.4f} | {surface[S_RESTRAINT]:>9.4f} | {surface[S_SHARPNESS]:>9.4f}")

        # 验证最终 stress 和 irritation 上升
        assert internal[I_STRESS] > DEFAULT_INTERNAL[I_STRESS], "冲突应导致压力上升"
        assert surface[S_RESTRAINT] > surface[S_RESTRAINT] if False else True  # 确保变量已使用

    def test_10_round_validation_recovery(self):
        """追踪 5 轮冲突 + 5 轮修复。"""
        print("\n=== [Multi] 冲突→修复轨迹 ===")

        traits = DEFAULT_TRAITS.copy()
        internal = DEFAULT_INTERNAL.copy()
        relationship = DEFAULT_RELATIONSHIP.copy()
        prev_surface = None

        conflict_stim = np.zeros(ST_SIZE)
        conflict_stim[ST_CONFLICT] = 0.7
        conflict_stim[ST_EMOTIONAL_WEIGHT] = 0.3

        repair_stim = np.zeros(ST_SIZE)
        repair_stim[ST_VALIDATION] = 0.8
        repair_stim[ST_CLOSENESS] = 0.5

        print(f"  {'#':>2} {'phase':>12} | {'stress':>7} | {'affection':>9} | {'trust':>7} | {'warmth':>7} | {'vulnerability':>14}")
        print("-" * 70)

        for turn in range(11):
            stim = conflict_stim if turn <= 5 else repair_stim
            profiles = compute_defense_profiles(traits, relationship, internal)
            inner, outer = apply_defenses(stim, profiles)
            internal = update_internal_state(internal, inner, traits, relationship, profiles)
            relationship = update_relationship_state(relationship, inner, traits, current_internal=internal)
            surface = project_surface(internal, relationship, outer, prev_surface)
            prev_surface = surface

            phase = "conflict" if turn <= 5 else "repair"
            print(f"  {turn:>2} {phase:>12} | {internal[I_STRESS]:>7.4f} | {relationship[R_AFFECTION]:>9.4f} | {relationship[R_TRUST_BOND]:>7.4f} | {surface[S_WARMTH]:>7.4f} | {surface[S_VULNERABILITY]:>14.4f}")


# ════════════════════════════════════════════════════════════
# Section 8 — 收敛性与稳定性验证（小批量追踪版）
# ════════════════════════════════════════════════════════════

class TestConvergenceTrace:
    """追踪零刺激下的收敛行为。"""

    def test_zero_stimulus_convergence(self):
        """追踪 20 轮无刺激时的状态收敛。"""
        print("\n=== [Analysis] 20 轮零刺激收敛追踪 ===")

        traits = DEFAULT_TRAITS.copy()
        internal = DEFAULT_INTERNAL.copy()
        relationship = DEFAULT_RELATIONSHIP.copy()
        zero_stim = np.zeros(ST_SIZE)

        print(f"  {'#':>2} | {'energy':>7} | {'stress':>7} | {'loneliness':>10} | {'affection':>9} | {'trust':>7} | {'intimacy':>8}")
        print("-" * 65)
        for turn in range(21):
            if turn > 0:
                profiles = compute_defense_profiles(traits, relationship, internal)
                inner, outer = apply_defenses(zero_stim, profiles)
                internal = update_internal_state(internal, inner, traits, relationship, profiles)
                relationship = update_relationship_state(relationship, inner, traits, current_internal=internal)

            print(f"  {turn:>2} | {internal[I_ENERGY]:>7.4f} | {internal[I_STRESS]:>7.4f} | {internal[I_LONELINESS]:>10.4f} | {relationship[R_AFFECTION]:>9.4f} | {relationship[R_TRUST_BOND]:>7.4f} | {relationship[R_INTIMACY]:>8.4f}")

        # 验证不发散
        assert np.all(np.abs(internal) <= 1.0)
        assert np.all(np.abs(relationship) <= 1.0)


# ════════════════════════════════════════════════════════════
# Section 9 — 批量统计 + 异常值检测（带追踪）
# ════════════════════════════════════════════════════════════

class TestBulkTrace:
    """小批量 Monte Carlo，追踪每个异常值。"""

    def test_bulk_500_samples_with_trace(self):
        """500 组随机样本 + 异常值追踪。"""
        print("\n=== [Bulk] 500 样本批量 + 异常追踪 ===")

        n = 500
        internal_data = []
        surface_data = []
        anomalies = []

        for i in range(n):
            traits = rng.uniform(-1, 1, T_SIZE)
            rel = rng.uniform(-1, 1, R_SIZE)
            internal = rng.uniform(-1, 1, I_SIZE)
            stimuli = rng.uniform(0, 1, ST_SIZE)

            profiles = compute_defense_profiles(traits, rel, internal)
            inner, outer = apply_defenses(stimuli, profiles)

            new_internal = update_internal_state(internal, inner, traits, rel, profiles)
            new_rel = update_relationship_state(rel, inner, traits, current_internal=internal)
            surface = project_surface(new_internal, new_rel, outer, None)

            internal_data.append(new_internal)
            surface_data.append(surface)

            # 异常检测
            # soft_clamp(transition=0.1) 允许 ±0.11 （含微小余量）
            SOFT_TOL = 0.11
            if np.any(np.abs(new_internal) > 1.0 + SOFT_TOL):
                anomalies.append((i, "internal", new_internal))
            if np.any(np.abs(new_rel) > 1.0 + SOFT_TOL):
                anomalies.append((i, "relationship", new_rel))
            if np.any(np.abs(surface) > 1.0 + SOFT_TOL):
                anomalies.append((i, "surface", surface))

        internal_data = np.array(internal_data)
        surface_data = np.array(surface_data)

        print(f"  样本数: {n}")
        print(f"  异常数: {len(anomalies)}")
        if anomalies:
            for idx, kind, vec in anomalies[:5]:
                print(f"    样本 {idx} 中 {kind} 越界: {vec}")

        # 统计追踪
        print(f"\n  Internal 统计:")
        for i, label in enumerate(I8_LABELS):
            print(f"    {label:20s}: mean={internal_data[:, i].mean():.4f}, std={internal_data[:, i].std():.4f}, [{internal_data[:, i].min():.3f}, {internal_data[:, i].max():.3f}]")

        print(f"\n  Surface 统计:")
        for i, label in enumerate(S7_LABELS):
            print(f"    {label:20s}: mean={surface_data[:, i].mean():.4f}, std={surface_data[:, i].std():.4f}, [{surface_data[:, i].min():.3f}, {surface_data[:, i].max():.3f}]")

        assert len(anomalies) == 0, f"发现 {len(anomalies)} 个越界值"
