"""异常探测 —— 专门寻找隐藏的异常现象。

不追求"通过"，而是暴露潜在的数值/设计问题。
"""

import numpy as np
import pytest
from state import (
    DEFAULT_TRAITS, DEFAULT_INTERNAL, DEFAULT_RELATIONSHIP,
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE, I_SIZE,
    R_AFFECTION, R_TRUST, R_FAMILIARITY, R_DEPENDENCY,
    R_EMOTIONAL_SAFETY, R_ROMANTIC_TENSION, R_SIZE,
    S_EXPRESSIVENESS, S_WARMTH, S_SHARPNESS, S_SOFTNESS,
    S_ENTHUSIASM, S_RESTRAINT, S_VULNERABILITY, S_SIZE,
    ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT,
    ST_DEPENDENCY, ST_TEASING, ST_EMOTIONAL_WEIGHT, ST_SIZE,
    I_LABELS, R_LABELS, S_LABELS, ST_LABELS,
    T_EMOTIONAL_STABILITY, T_OPTIMISM, T_ANXIETY_PRONENESS,
    T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE,
)
from state_engine import update_all, initialize_all
from state_engine._defenses import compute_defense_profiles, apply_defenses
from state_engine._dynamics import (
    compute_setpoint, compute_rel_setpoint,
    update_internal_state, update_relationship_state,
)
from state_engine._surface import project_surface


class TestAnomalySingleRoundResponsiveness:
    """探测：单轮刺激对状态的实际影响力。"""

    def test_max_deviation_from_default(self, rng):
        """100,000 组刺激中，单轮内部状态最大偏离 default 多少？

        这是衡量引擎"响应灵敏度"的关键指标。
        """
        n = 100_000
        internal = DEFAULT_INTERNAL.copy()
        rel = DEFAULT_RELATIONSHIP.copy()
        traits = DEFAULT_TRAITS.copy()

        stimuli_batch = rng.beta(0.3, 0.3, size=(n, ST_SIZE))  # 极端分布

        max_deviations = np.zeros(I_SIZE)
        min_deviations = np.zeros(I_SIZE)
        max_abs_delta = np.zeros(I_SIZE)

        for i in range(n):
            result = update_all(internal, rel, traits, stimuli_batch[i])
            delta = result["internal_state"] - internal
            max_deviations = np.maximum(max_deviations, delta)
            min_deviations = np.minimum(min_deviations, delta)
            max_abs_delta = np.maximum(max_abs_delta, np.abs(delta))

        print(f"\n  === 单轮响应灵敏度 (n={n:,}, 极端刺激分布) ===")
        print(f"  {'维度':>18s} | {'最大上升':>10s} | {'最大下降':>10s} | {'最大绝对变化':>12s}")
        print(f"  {'-'*18} | {'-'*10} | {'-'*10} | {'-'*12}")
        for dim in range(I_SIZE):
            print(f"  {I_LABELS[dim]:>18s} | {max_deviations[dim]:+10.4f} | {min_deviations[dim]:+10.4f} | {max_abs_delta[dim]:12.4f}")

        # 关键发现：如果某维度最大绝对变化 < 0.02，说明该维度几乎不受刺激影响
        stuck_dims = [I_LABELS[d] for d in range(I_SIZE) if max_abs_delta[d] < 0.02]
        if stuck_dims:
            print(f"\n  ⚠️ 低响应维度 (|Δmax| < 0.02): {stuck_dims}")
            print(f"     这些维度在极端刺激下几乎不变")

    def test_relationship_responsiveness(self, rng):
        """关系状态对刺激的响应灵敏度。

        已知问题：关系状态变化极慢（γ_rel ~ 0.005-0.01）。
        这里量化单轮最大响应。
        """
        n = 100_000
        internal = DEFAULT_INTERNAL.copy()
        rel = DEFAULT_RELATIONSHIP.copy()
        traits = DEFAULT_TRAITS.copy()

        stimuli_batch = rng.beta(0.3, 0.3, size=(n, ST_SIZE))

        max_abs_delta = np.zeros(R_SIZE)

        for i in range(n):
            result = update_all(internal, rel, traits, stimuli_batch[i])
            delta = result["relationship_state"] - rel
            max_abs_delta = np.maximum(max_abs_delta, np.abs(delta))

        print(f"\n  === 关系状态响应灵敏度 (n={n:,}) ===")
        for dim in range(R_SIZE):
            print(f"  {R_LABELS[dim]:>18s}: |Δmax| = {max_abs_delta[dim]:.6f}")

        stuck_dims = [R_LABELS[d] for d in range(R_SIZE) if max_abs_delta[d] < 0.005]
        if stuck_dims:
            print(f"\n  ⚠️ 几乎不响应的关系维度 (|Δmax| < 0.005): {stuck_dims}")

    def test_surface_amplification(self, rng):
        """表面状态的变化幅度 vs 内部状态变化幅度。

        表面应该是内部状态的"放大版"——变化幅度应 ≥ 内部。
        """
        n = 20_000
        internal = DEFAULT_INTERNAL.copy()
        rel = DEFAULT_RELATIONSHIP.copy()
        traits = DEFAULT_TRAITS.copy()

        stimuli_batch = rng.beta(2, 2, size=(n, ST_SIZE))

        internal_ranges = np.zeros((n, I_SIZE))
        surface_ranges = np.zeros((n, S_SIZE))

        for i in range(n):
            result = update_all(internal, rel, traits, stimuli_batch[i])
            internal_ranges[i] = np.abs(result["internal_state"] - internal)
            surface_ranges[i] = np.abs(result["surface_state"] - result["surface_state"])  # always 0...

        # 改用 sigma 比较
        all_internal = np.zeros((n, I_SIZE))
        all_surface = np.zeros((n, S_SIZE))
        for i in range(n):
            result = update_all(internal, rel, traits, stimuli_batch[i])
            all_internal[i] = result["internal_state"]
            all_surface[i] = result["surface_state"]

        print(f"\n  === 内部 vs 表面 变化幅度比较 ===")
        for dim_i in range(I_SIZE):
            i_std = all_internal[:, dim_i].std()
            print(f"  {I_LABELS[dim_i]:>18s}: σ_internal={i_std:.4f}")

        for dim_s in range(S_SIZE):
            s_std = all_surface[:, dim_s].std()
            print(f"  {S_LABELS[dim_s]:>18s}: σ_surface={s_std:.4f}")


class TestAnomalyRateParameters:
    """探测：α/β 速率参数的分布和边界（γ 已弃用，转移到 _decay.py）。"""

    def test_beta_vs_alpha_balance(self, rng):
        """α 和 β 的相对大小决定刺激 vs 耦合的权重。

        探测在随机参数下 α/β 的分布。
        """
        from state_engine._utils import soft_clamp

        n = 20_000
        traits = rng.uniform(-0.999, 0.999, size=(n, 10))
        rel = rng.uniform(-0.999, 0.999, size=(n, 6))

        alphas = np.empty(n)
        betas = np.empty(n)
        gammas = np.empty(n)

        for i in range(n):
            t = traits[i]
            r = rel[i]

            alpha = t[3] * 0.30 + (1.0 - t[3]) * 0.15 + r[1] * 0.12  # T_EMOTIONAL_STABILITY=3, R_TRUST=1
            alpha = soft_clamp(alpha, 0.02, 0.35)

            beta = 0.10
            beta = soft_clamp(beta, 0.01, 0.35)  # without profiles, beta stays at 0.10

            gamma = 0.08 + t[3] * 0.10 + t[4] * 0.06 - t[5] * 0.06
            gamma = soft_clamp(gamma, 0.01, 0.25)

            alphas[i] = alpha
            betas[i] = beta
            gammas[i] = gamma

        print(f"\n  === 速率参数分布 (n={n:,}) ===")
        print(f"  α (耦合): μ={alphas.mean():.3f} σ={alphas.std():.3f} [{alphas.min():.3f}, {alphas.max():.3f}]")
        print(f"  β (刺激): μ={betas.mean():.3f} σ={betas.std():.3f} [{betas.min():.3f}, {betas.max():.3f}]")
        print(f"  γ (恢复): μ={gammas.mean():.3f} σ={gammas.std():.3f} [{gammas.min():.3f}, {gammas.max():.3f}]")

        # α/β 比率
        ratio = alphas / (betas + 1e-10)
        print(f"  α/β 比率: μ={ratio.mean():.2f} σ={ratio.std():.2f} [{ratio.min():.2f}, {ratio.max():.2f}]")

        # 多少情况下 α > β（耦合主导）?
        alpha_dominates = (alphas > betas).mean()
        print(f"  α > β 的比例: {alpha_dominates*100:.1f}%")
        if alpha_dominates > 0.5:
            print(f"  ⚠️ 大多数情况下耦合效应强于刺激效应")

    def test_defense_profile_extremes(self, rng):
        """探测防御剖面在何种参数组合下达到极端值（接近 0 或 1）。"""
        n = 30_000
        traits = rng.beta(0.2, 0.2, size=(n, 10)) * 2 - 1
        rel = rng.beta(0.2, 0.2, size=(n, 6)) * 2 - 1
        internal = rng.beta(0.2, 0.2, size=(n, 8)) * 2 - 1

        deact_means = np.empty(n)
        hyper_means = np.empty(n)
        deact_maxs = np.empty(n)
        hyper_maxs = np.empty(n)

        for i in range(n):
            p = compute_defense_profiles(traits[i], rel[i], internal[i])
            deact_means[i] = p[0].mean()
            hyper_means[i] = p[1].mean()
            deact_maxs[i] = p[0].max()
            hyper_maxs[i] = p[1].max()

        print(f"\n  === 防御剖面极值分析 (n={n:,}, 极端人格/状态) ===")
        print(f"  deact mean: [{deact_means.min():.4f}, {deact_means.max():.4f}]")
        print(f"  deact max:  [{deact_maxs.min():.4f}, {deact_maxs.max():.4f}]")
        print(f"  hyper mean: [{hyper_means.min():.4f}, {hyper_means.max():.4f}]")
        print(f"  hyper max:  [{hyper_maxs.min():.4f}, {hyper_maxs.max():.4f}]")

        # 全零剖面？
        near_zero = (deact_means < 0.05) & (hyper_means < 0.05)
        print(f"  双零剖面 (both < 0.05): {near_zero.sum()} ({near_zero.sum()/n*100:.1f}%)")

        # 全一剖面？
        near_one = (deact_means > 0.90) & (hyper_means > 0.90)
        print(f"  双一剖面 (both > 0.90): {near_one.sum()} ({near_one.sum()/n*100:.1f}%)")


class TestAnomalySaturation:
    """探测：状态在极端条件下是否过早饱和。"""

    def test_all_stimuli_max_saturation(self, default_traits, default_internal, default_relationship):
        """全部刺激 max(1.0) 持续施加 100 轮，观察是否卡在 0 或 1。"""
        s = np.ones(ST_SIZE)
        current_internal = default_internal.copy()
        current_rel = default_relationship.copy()

        saturated_dims = set()

        for round_idx in range(100):
            result = update_all(current_internal, current_rel, default_traits, s)
            current_internal = result["internal_state"]
            current_rel = result["relationship_state"]

            for dim in range(I_SIZE):
                if current_internal[dim] <= -0.999 or current_internal[dim] >= 0.999:
                    saturated_dims.add(f"internal.{I_LABELS[dim]}")
            for dim in range(R_SIZE):
                if current_rel[dim] <= -0.999 or current_rel[dim] >= 0.999:
                    saturated_dims.add(f"rel.{R_LABELS[dim]}")

        print(f"\n  === 全刺激 max 100 轮后的饱和检测 ===")
        print(f"  内部状态: {[f'{v:.4f}' for v in current_internal]}")
        print(f"  关系状态: {[f'{v:.4f}' for v in current_rel]}")
        if saturated_dims:
            print(f"  ⚠️ 饱和维度: {sorted(saturated_dims)}")
        else:
            print(f"  ✅ 无饱和（所有维度在 (-0.999, 0.999) 内）")

    def test_zero_stimuli_floor(self, default_traits, default_internal, default_relationship):
        """零刺激持续施压 500 轮，观察是否卡在 setpoint 以下无法回升。"""
        s = np.zeros(ST_SIZE)
        current_internal = default_internal.copy()
        current_rel = default_relationship.copy()

        sp_internal = compute_setpoint(default_traits)
        sp_rel = compute_rel_setpoint(default_traits)

        print(f"\n  === 零刺激 500 轮后的稳态偏差 ===")
        print(f"  {'维度':>18s} | {'当前':>8s} | {'setpoint':>8s} | {'偏差':>8s}")
        print(f"  {'-'*18} | {'-'*8} | {'-'*8} | {'-'*8}")

        for _ in range(500):
            result = update_all(current_internal, current_rel, default_traits, s)
            current_internal = result["internal_state"]
            current_rel = result["relationship_state"]

        for dim in range(I_SIZE):
            dev = abs(current_internal[dim] - sp_internal[dim])
            marker = " ⚠️" if dev > 0.05 else ""
            print(f"  {I_LABELS[dim]:>18s} | {current_internal[dim]:8.4f} | {sp_internal[dim]:8.4f} | {dev:8.4f}{marker}")

        for dim in range(R_SIZE):
            dev = abs(current_rel[dim] - sp_rel[dim])
            marker = " ⚠️" if dev > 0.05 else ""
            print(f"  {R_LABELS[dim]:>18s} | {current_rel[dim]:8.4f} | {sp_rel[dim]:8.4f} | {dev:8.4f}{marker}")


class TestAnomalyDefenseCollapse:
    """探测：防御剖面是否会失效（deact 和 hyper 同时趋近 0 或同时趋近 1）。"""

    def test_profile_independence(self, rng):
        """deact 和 hyper 的相关性 —— 如果高度相关则失去了独立防御维度的意义。"""
        n = 30_000
        traits = rng.uniform(-1, 1, size=(n, 10))
        rel = rng.uniform(-1, 1, size=(n, 6))
        internal = rng.uniform(-1, 1, size=(n, 8))

        deact_arr = np.empty(n)
        hyper_arr = np.empty(n)

        for i in range(n):
            p = compute_defense_profiles(traits[i], rel[i], internal[i])
            deact_arr[i] = p[0].mean()
            hyper_arr[i] = p[1].mean()

        corr = np.corrcoef(deact_arr, hyper_arr)[0, 1]
        print(f"\n  === 防御剖面独立性 ===")
        print(f"  deact × hyper 相关系数: r = {corr:.4f}")
        if abs(corr) > 0.7:
            print(f"  ⚠️ 高度相关 —— deact 和 hyper 不是独立维度")
        elif abs(corr) > 0.3:
            print(f"  ℹ️ 中度相关 —— 部分共享方差")
        else:
            print(f"  ✅ 低度相关 —— 两个防御维度基本独立")

        # 同时高或同时低的比例
        both_high = (deact_arr > 0.6) & (hyper_arr > 0.6)
        both_low = (deact_arr < 0.3) & (hyper_arr < 0.3)
        print(f"  同时高 (>0.6): {both_high.sum()/n*100:.1f}%")
        print(f"  同时低 (<0.3): {both_low.sum()/n*100:.1f}%")
        print(f"  铁壁 (高deact+低hyper): {((deact_arr > 0.5) & (hyper_arr < 0.3)).sum()/n*100:.1f}%")
        print(f"  玻璃心 (低deact+高hyper): {((deact_arr < 0.3) & (hyper_arr > 0.5)).sum()/n*100:.1f}%")


class TestAnomalyTraitSensitivity:
    """探测：默认特质值的微小扰动是否会引起剧烈行为变化。"""

    def test_trait_gradient(self, rng):
        """对每个特质维度做 ±0.1 扰动，测状态变化幅度。

        如果某个特质维度的微小变化导致巨大的行为差异，
        说明系统对该特质过于敏感。
        """
        stimuli = np.ones(ST_SIZE) * 0.5
        internal = DEFAULT_INTERNAL.copy()
        rel = DEFAULT_RELATIONSHIP.copy()

        base_result = update_all(internal, rel, DEFAULT_TRAITS, stimuli)
        base_internal = base_result["internal_state"]

        print(f"\n  === 特质敏感度 (ΔTrait = ±0.1) ===")
        print(f"  {'特质':>22s} | {'-0.1 Δ':>10s} | {'+0.1 Δ':>10s}")
        print(f"  {'-'*22} | {'-'*10} | {'-'*10}")

        from state import T_LABELS
        for t_dim in range(10):
            t_lo = DEFAULT_TRAITS.copy()
            t_lo[t_dim] = max(-0.95, t_lo[t_dim] - 0.1)
            t_hi = DEFAULT_TRAITS.copy()
            t_hi[t_dim] = min(0.95, t_hi[t_dim] + 0.1)

            r_lo = update_all(internal, rel, t_lo, stimuli)
            r_hi = update_all(internal, rel, t_hi, stimuli)

            delta_lo = np.linalg.norm(r_lo["internal_state"] - base_internal)
            delta_hi = np.linalg.norm(r_hi["internal_state"] - base_internal)

            print(f"  {T_LABELS[t_dim]:>22s} | {delta_lo:10.4f} | {delta_hi:10.4f}")


class TestAnomalyMultiRoundDrift:
    """探测：多轮迭代中状态是否发生不可逆漂移。"""

    def test_round_trip_hysteresis(self, default_traits, default_internal, default_relationship):
        """施加正向刺激 50 轮，再施加反向刺激 50 轮，是否回到原位？

        如果回不去，说明存在迟滞效应——系统在两种刺激下不对称。
        """
        pos_stim = np.zeros(ST_SIZE)
        pos_stim[ST_VALIDATION] = 0.8
        pos_stim[ST_CLOSENESS] = 0.6

        neg_stim = np.zeros(ST_SIZE)
        neg_stim[ST_CONFLICT] = 0.8
        neg_stim[ST_ABANDONMENT] = 0.6

        internal_start = default_internal.copy()
        rel_start = default_relationship.copy()

        current_i = internal_start.copy()
        current_r = rel_start.copy()

        # 正向 50 轮
        for _ in range(50):
            result = update_all(current_i, current_r, default_traits, pos_stim)
            current_i = result["internal_state"]
            current_r = result["relationship_state"]

        after_pos = current_i.copy()

        # 反向 50 轮
        for _ in range(50):
            result = update_all(current_i, current_r, default_traits, neg_stim)
            current_i = result["internal_state"]
            current_r = result["relationship_state"]

        after_neg = current_i.copy()

        # 偏离起点的距离
        pos_dev = np.linalg.norm(after_pos - internal_start)
        neg_dev = np.linalg.norm(after_neg - internal_start)
        trip_dev = np.linalg.norm(after_neg - after_pos)

        print(f"\n  === 往返迟滞测试 ===")
        print(f"  起点 → 正刺激50轮: 偏差 = {pos_dev:.4f}")
        print(f"  起点 → 正50轮 → 负50轮: 偏差 = {neg_dev:.4f}")
        print(f"  正50轮 vs 负50轮: 距离 = {trip_dev:.4f}")

        # 如果往返后离起点比单程还近，说明负刺激在修复
        if neg_dev < pos_dev:
            print(f"  ✅ 负刺激部分修复了正刺激的影响")
        else:
            print(f"  ⚠️ 往返后未回到原位（迟滞或不对称）")


class TestAnomalySurfaceDegeneracy:
    """探测：表面投影是否存在退化（多个内部状态映射到同一个表面）。"""

    def test_surface_saturation_rate(self, rng):
        """在随机参数空间中 surface 各维度达到 [-1.0, -0.99] 或 [0.99, 1.0] 的频率。"""
        n = 50_000
        internal = rng.uniform(-1, 1, size=(n, 8))
        relationship = rng.uniform(-1, 1, size=(n, 6))
        traits = rng.uniform(-1, 1, size=(n, 10))
        outer = rng.uniform(-1, 1, size=(n, 7))

        floor_counts = np.zeros(S_SIZE)
        ceil_counts = np.zeros(S_SIZE)

        for i in range(n):
            s = project_surface(internal[i], relationship[i], traits[i], outer[i])
            floor_counts += (s < -0.99).astype(int)
            ceil_counts += (s > 0.99).astype(int)

        print(f"\n  === 表面维度饱和率 (n={n:,}) ===")
        print(f"  {'维度':>18s} | {'触底率':>8s} | {'触顶率':>8s}")
        print(f"  {'-'*18} | {'-'*8} | {'-'*8}")
        for dim in range(S_SIZE):
            print(f"  {S_LABELS[dim]:>18s} | {floor_counts[dim]/n*100:7.2f}% | {ceil_counts[dim]/n*100:7.2f}%")
            if floor_counts[dim] / n > 0.1:
                print(f"                     ⚠️ 触底率 > 10% — 该维度容易退化到 -1")
            if ceil_counts[dim] / n > 0.1:
                print(f"                     ⚠️ 触顶率 > 10% — 该维度容易退化到 1")


class TestAnomalyBetaModulation:
    """探测：β 是否随防御剖面有意义地变化。"""

    def test_beta_range_after_fix(self, rng):
        """β 在不同防御配置下应有可观测的变动范围。

        修复后（2026-06-18）：sigmoid 缩放 + β 公式重写，
        预期 β 有效范围 > 0.10（旧设计仅 0.042）。
        """
        from state_engine._dynamics import update_internal_state
        n = 50_000
        traits_batch = rng.uniform(-1, 1, size=(n, 10))
        rel_batch = rng.uniform(-1, 1, size=(n, 6))
        internal_batch = rng.uniform(-1, 1, size=(n, 8))
        zero_stim = np.zeros(7)

        betas = np.empty(n)
        for i in range(n):
            profiles = compute_defense_profiles(
                traits_batch[i], rel_batch[i], internal_batch[i],
            )
            deact, hyper = profiles[0].mean(), profiles[1].mean()
            beta = max(0.01, min(0.35, 0.05 + hyper * 0.35 - deact * 0.15))
            betas[i] = beta

        low_group = betas < betas.mean() - betas.std()
        high_group = betas > betas.mean() + betas.std()
        effective_range = betas.max() - betas.min()
        coef_var = betas.std() / betas.mean()

        print(f"\n  === β 调制范围检验 (n={n:,}) ===")
        print(f"  β 范围: [{betas.min():.4f}, {betas.max():.4f}]")
        print(f"  β 均值: {betas.mean():.4f} ± {betas.std():.4f}")
        print(f"  β 有效变动: {effective_range:.4f}")
        print(f"  β 变异系数: {coef_var:.3f}")
        print(f"  低 β 组占比: {low_group.mean()*100:.1f}%  "
              f"(mean={betas[low_group].mean():.4f})")
        print(f"  高 β 组占比: {high_group.mean()*100:.1f}%  "
              f"(mean={betas[high_group].mean():.4f})")

        if effective_range < 0.05:
            print(f"  🔴 β 有效变动 {effective_range:.4f} < 0.05 — 调制基本失效")
        elif effective_range < 0.15:
            print(f"  🟡 β 有效变动 {effective_range:.4f} — 有一定范围但偏窄")
        else:
            print(f"  ✅ β 有效变动 {effective_range:.4f} — 调制效果良好")


class TestAnomalyMatrixNoise:
    """探测：映射矩阵噪声是否引入系统性偏差。"""

    def test_matrix_noise_no_systematic_drift(self, rng):
        """矩阵噪声每轮 ε ~ N(0, 0.03²) 叠加到 W 上，
        长期运行时不应导致状态均值系统性偏离无噪声基线。
        """
        from tests.test_adversarial_engine import (
            CoupledAgents, SurfaceToStimuliMapping, detect_anomalies,
        )
        from state import DEFAULT_TRAITS

        n_steps = 1000

        # 无噪声基线
        clean = CoupledAgents(
            traits_a=DEFAULT_TRAITS.copy(),
            traits_b=DEFAULT_TRAITS.copy(),
            mapping_a2b=SurfaceToStimuliMapping(),
            mapping_b2a=SurfaceToStimuliMapping(),
        )
        clean.run(n_steps)
        baseline_a = clean.history.get_agent_trajectory("A", "internal").mean(axis=0)

        # 有噪声
        noisy = CoupledAgents(
            traits_a=DEFAULT_TRAITS.copy(),
            traits_b=DEFAULT_TRAITS.copy(),
            mapping_a2b=SurfaceToStimuliMapping(matrix_noise_std=0.03),
            mapping_b2a=SurfaceToStimuliMapping(matrix_noise_std=0.03),
            rng=rng,
        )
        noisy.run(n_steps)
        noisy_mean_a = noisy.history.get_agent_trajectory("A", "internal").mean(axis=0)

        drift = np.abs(noisy_mean_a - baseline_a)
        max_drift = drift.max()
        avg_drift = drift.mean()

        print(f"\n  === 矩阵噪声系统性漂移检验 (n={n_steps} 轮) ===")
        print(f"  {'维度':>18s} | {'无噪声均值':>10s} | {'有噪声均值':>10s} | {'漂移':>8s}")
        print(f"  {'-'*18} | {'-'*10} | {'-'*10} | {'-'*8}")
        for dim in range(len(baseline_a)):
            print(f"  {I_LABELS[dim]:>18s} | {baseline_a[dim]:10.4f} | "
                  f"{noisy_mean_a[dim]:10.4f} | {drift[dim]:8.4f}")
        print(f"  {'-'*50}")
        print(f"  最大维度漂移: {max_drift:.4f}")
        print(f"  平均维度漂移: {avg_drift:.4f}")

        # 噪声应不导致系统性偏差（如果噪声是零均值且对称的）
        if max_drift > 0.05:
            print(f"  ⚠️ 最大漂移 {max_drift:.4f} > 0.05 — 可能存在系统性偏差")
        else:
            print(f"  ✅ 漂移在可接受范围内")

        # 检查是否有边界违规
        reports = detect_anomalies(noisy.history, DEFAULT_TRAITS, DEFAULT_TRAITS)
        boundary = [r for r in reports if r.anomaly_type == "boundary"]
        if boundary:
            print(f"  ⚠️ 噪声运行中出现 {len(boundary)} 个边界违规")
        else:
            print(f"  ✅ 无边界违规")
