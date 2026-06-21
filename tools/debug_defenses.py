"""防御剖面方差分解 + 调制增强实验。

研究问题:
  1. 固定 traits 时, 各信号源贡献多少方差?
  2. 现有调制系数是否太小? 放大后能否提升维度独立性?
  3. 能否从系数设计中找到根本问题?
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from state import (
    DEFAULT_TRAITS, ST_LABELS,
    T_PRIDE, T_JEALOUSY_SENSITIVITY, T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE,
    T_EMOTIONAL_STABILITY, T_EMOTIONAL_OPENNESS, T_SENSITIVITY, T_ANGER_REACTIVITY,
    I_STRESS, I_INSECURITY, I_LONGING,
    R_AFFECTION, R_TRUST_BOND, R_INTIMACY,
    ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT,
    ST_DEPENDENCY, ST_TEASING, ST_EMOTIONAL_WEIGHT, ST_SIZE,
)
from state_engine._defenses import (
    compute_defense_profiles, _apply_additive, _apply_multiplicative,
    STABILITY_DEACT_A, OPENNESS_DEACT_A, AVOIDANCE_DEACT_A,
    STRESS_DEACT_A, INSECURITY_DEACT_A,
    TRUST_BOND_DEACT_M,
    SENSITIVITY_HYPER_A, AVOIDANCE_HYPER_A,
    AFFECTION_HYPER_M_NEW, INTIMACY_HYPER_M,
    INSECURITY_HYPER_A, LONGING_HYPER_A,
)
from state_engine._utils import _sigmoid

RNG = np.random.default_rng(42)
N = 100_000


def var_decomp(data, labels, label):
    """方差分解: 总方差 + 每维方差"""
    var_per_dim = data.var(axis=0)
    total_var = var_per_dim.sum()
    print(f"\n--- {label} ---")
    print(f"  总方差: {total_var:.6f}")
    for d in range(7):
        pct = var_per_dim[d] / total_var * 100
        print(f"    {labels[d]:20s} var={var_per_dim[d]:.6f} ({pct:.1f}%)")


def run_pca(data, label, dim_labels):
    n, d = data.shape
    c = data - data.mean(0)
    _, s, _ = np.linalg.svd(c, full_matrices=False)
    ev = s**2 / (n-1)
    ratio = ev / ev.sum()
    cum = np.cumsum(ratio)
    eff_d = int(np.searchsorted(cum, 0.95) + 1)
    eff_r = np.exp(-np.sum((ratio + 1e-30) * np.log(ratio + 1e-30)))
    corr = np.corrcoef(data.T)
    np.fill_diagonal(corr, 0)
    mx = np.max(np.abs(corr))
    np.fill_diagonal(corr, 1)

    indep = np.zeros(d)
    for dim in range(d):
        other = [j for j in range(d) if j != dim]
        X = np.column_stack([np.ones(n), data[:, other]])
        beta, rss, rank, _ = np.linalg.lstsq(X, data[:, dim], rcond=None)
        # rss 可能形状为 (0,) 当秩不足; 退化为 rss=0
        rss_val = rss[0] if len(rss) > 0 else 0.0
        tss = np.sum((data[:, dim] - data[:, dim].mean())**2)
        indep[dim] = rss_val / tss if tss > 1e-15 else 0.0

    print(f"\n── {label} ──")
    print(f"  PC1={ratio[0]*100:.1f}%, PC2={ratio[1]*100:.1f}%, PC3={ratio[2]*100:.1f}%")
    print(f"  95%有效维={eff_d}/{d}, 有效秩={eff_r:.2f}, max|r|={mx:.4f}")
    print(f"  独立方差均值={indep.mean()*100:.1f}%")
    for dim in range(d):
        print(f"    {dim_labels[dim]:20s} {indep[dim]*100:5.1f}%")
    return {"eff_d": eff_d, "eff_r": eff_r, "pc1": ratio[0], "mx": mx, "indep_mean": indep.mean(), "indep": indep}


# ═══════════════════════════════════════════════════════════
# 实验 A: 逐个信号源开关，看谁贡献方差
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("实验 A: 方差来源分解")
print("=" * 70)

traits = np.tile(DEFAULT_TRAITS, (N, 1))
internal = RNG.uniform(-1, 1, size=(N, 8))
rel = RNG.uniform(-1, 1, size=(N, 6))

# A0: 基线 — 全随机
deact_full = np.empty((N, 7))
hyper_full = np.empty((N, 7))
for i in range(N):
    p = compute_defense_profiles(traits[i], rel[i], internal[i])
    deact_full[i] = p[0]
    hyper_full[i] = p[1]

print("\nA0: 全随机 (固定 traits, 随机 internal+rel)")
r_full = run_pca(deact_full, "去激活 (全随机)", ST_LABELS)

# A1: 仅变化 internal, 固定 relationship=0
rel_zero = np.zeros((N, 6))
deact_int = np.empty((N, 7))
hyper_int = np.empty((N, 7))
for i in range(N):
    p = compute_defense_profiles(traits[i], rel_zero[i], internal[i])
    deact_int[i] = p[0]
    hyper_int[i] = p[1]
print("\nA1: 仅 internal 变化 (rel=0)")
r_int = run_pca(deact_int, "去激活 (仅internal)", ST_LABELS)

# A2: 仅变化 relationship, 固定 internal=0
int_zero = np.zeros((N, 8))
deact_rel = np.empty((N, 7))
hyper_rel = np.empty((N, 7))
for i in range(N):
    p = compute_defense_profiles(traits[i], rel[i], int_zero[i])
    deact_rel[i] = p[0]
    hyper_rel[i] = p[1]
print("\nA2: 仅 relationship 变化 (internal=0)")
r_rel = run_pca(deact_rel, "去激活 (仅rel)", ST_LABELS)

print("\n" + "=" * 70)
print(f"方差源对比 (独立方差均值):")
print(f"  全随机:     deact={r_full['indep_mean']*100:.1f}%, hyper=...")
print(f"  仅 internal: deact={r_int['indep_mean']*100:.1f}%")
print(f"  仅 rel:      deact={r_rel['indep_mean']*100:.1f}%")


# ═══════════════════════════════════════════════════════════
# 实验 B: 放大调制强度
# ═══════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("实验 B: 调制强度放大 3×")
print("=" * 70)

# 复制 compute_defense_profiles, 但将状态调制系数放大 3 倍
def compute_profiles_amplified(traits, rel, internal, deact_scale=1.0, hyper_scale=1.0):
    profiles = np.zeros((2, 7))
    deact = np.zeros(7)
    deact[ST_ABANDONMENT] = 0.30 + traits[T_PRIDE] * 0.225 + traits[T_JEALOUSY_SENSITIVITY] * 0.09
    deact[ST_VALIDATION]  = 0.25 + traits[T_PRIDE] * 0.20
    deact[ST_DEPENDENCY]  = 0.28 + traits[T_PRIDE] * 0.19 + traits[T_ATTACHMENT_AVOIDANCE] * 0.10
    deact[ST_CLOSENESS]   = 0.15 + traits[T_PRIDE] * 0.09 + traits[T_ATTACHMENT_AVOIDANCE] * 0.075
    deact[ST_CONFLICT]    = 0.20 + traits[T_PRIDE] * 0.14 + traits[T_ANGER_REACTIVITY] * 0.11
    deact[ST_TEASING]     = 0.20 + traits[T_PRIDE] * 0.16
    deact[ST_EMOTIONAL_WEIGHT] = 0.25 + traits[T_PRIDE] * 0.14

    _apply_additive(deact, traits[T_EMOTIONAL_STABILITY]   * deact_scale, STABILITY_DEACT_A)
    _apply_additive(deact, traits[T_EMOTIONAL_OPENNESS]    * deact_scale, OPENNESS_DEACT_A)
    _apply_additive(deact, traits[T_ATTACHMENT_AVOIDANCE]  * deact_scale, AVOIDANCE_DEACT_A)
    _apply_multiplicative(deact, rel[R_TRUST_BOND]         * deact_scale, TRUST_BOND_DEACT_M)
    _apply_additive(deact, internal[I_STRESS]              * deact_scale, STRESS_DEACT_A)
    _apply_additive(deact, internal[I_INSECURITY]          * deact_scale, INSECURITY_DEACT_A)
    profiles[0] = _sigmoid((deact - 0.35) * 5.0)

    hyper = np.zeros(7)
    hyper[ST_ABANDONMENT] = 0.45 + traits[T_ATTACHMENT_ANXIETY] * 0.275 + traits[T_JEALOUSY_SENSITIVITY] * 0.15
    hyper[ST_CLOSENESS]   = 0.30 + traits[T_ATTACHMENT_ANXIETY] * 0.25
    hyper[ST_DEPENDENCY]  = 0.35 + traits[T_ATTACHMENT_ANXIETY] * 0.20
    hyper[ST_VALIDATION]  = 0.15 + traits[T_ATTACHMENT_ANXIETY] * 0.10
    hyper[ST_CONFLICT]    = 0.15 + traits[T_ATTACHMENT_ANXIETY] * 0.15
    hyper[ST_TEASING]     = 0.10 + traits[T_JEALOUSY_SENSITIVITY] * 0.10
    hyper[ST_EMOTIONAL_WEIGHT] = 0.20 + traits[T_ATTACHMENT_ANXIETY] * 0.15

    _apply_additive(hyper, traits[T_SENSITIVITY]           * hyper_scale, SENSITIVITY_HYPER_A)
    _apply_additive(hyper, traits[T_ATTACHMENT_AVOIDANCE]  * hyper_scale, AVOIDANCE_HYPER_A)
    _apply_multiplicative(hyper, rel[R_AFFECTION]          * hyper_scale, AFFECTION_HYPER_M_NEW)
    _apply_multiplicative(hyper, rel[R_INTIMACY]           * hyper_scale, INTIMACY_HYPER_M)
    _apply_additive(hyper, internal[I_INSECURITY]          * hyper_scale, INSECURITY_HYPER_A)
    _apply_additive(hyper, internal[I_LONGING]             * hyper_scale, LONGING_HYPER_A)
    profiles[1] = _sigmoid((hyper - 0.38) * 5.0)

    return profiles


traits_fixed = np.tile(DEFAULT_TRAITS, (N, 1))
for scale in [3.0, 5.0, 10.0]:
    deact_amp = np.empty((N, 7))
    hyper_amp = np.empty((N, 7))
    for i in range(N):
        p = compute_profiles_amplified(traits_fixed[i], rel[i], internal[i], deact_scale=scale, hyper_scale=scale)
        deact_amp[i] = p[0]
        hyper_amp[i] = p[1]
    rd = run_pca(deact_amp, f"去激活 (调制 ×{scale})", ST_LABELS)
    rh = run_pca(hyper_amp, f"过度激活 (调制 ×{scale})", ST_LABELS)
    print(f"  ×{scale}: deact独立方差={rd['indep_mean']*100:.1f}%, hyper独立方差={rh['indep_mean']*100:.1f}%")


# ═══════════════════════════════════════════════════════════
# 实验 C: 去掉 sigmoid, 在原始线性空间做分析
# ═══════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("实验 C: sigmoid 前的线性空间分析")
print("=" * 70)

deact_pre_flat = np.empty((N, 7))
for i in range(N):
    p = compute_profiles_amplified(traits_fixed[i], rel[i], internal[i], deact_scale=1.0, hyper_scale=1.0)
    deact_pre_flat[i] = p[0]  # Already sigmoided

# Re-run on pre-sigmoid values
deact_pre = np.empty((N, 7))
hyper_pre = np.empty((N, 7))
for i in range(N):
    deact = np.zeros(7)
    deact[ST_ABANDONMENT] = 0.30 + DEFAULT_TRAITS[T_PRIDE]*0.225 + DEFAULT_TRAITS[T_JEALOUSY_SENSITIVITY]*0.09
    deact[ST_VALIDATION]  = 0.25 + DEFAULT_TRAITS[T_PRIDE]*0.20
    deact[ST_DEPENDENCY]  = 0.28 + DEFAULT_TRAITS[T_PRIDE]*0.19 + DEFAULT_TRAITS[T_ATTACHMENT_AVOIDANCE]*0.10
    deact[ST_CLOSENESS]   = 0.15 + DEFAULT_TRAITS[T_PRIDE]*0.09 + DEFAULT_TRAITS[T_ATTACHMENT_AVOIDANCE]*0.075
    deact[ST_CONFLICT]    = 0.20 + DEFAULT_TRAITS[T_PRIDE]*0.14 + DEFAULT_TRAITS[T_ANGER_REACTIVITY]*0.11
    deact[ST_TEASING]     = 0.20 + DEFAULT_TRAITS[T_PRIDE]*0.16
    deact[ST_EMOTIONAL_WEIGHT] = 0.25 + DEFAULT_TRAITS[T_PRIDE]*0.14
    _apply_additive(deact, internal[i, I_STRESS], STRESS_DEACT_A)
    _apply_additive(deact, internal[i, I_INSECURITY], INSECURITY_DEACT_A)
    _apply_multiplicative(deact, rel[i, R_TRUST_BOND], TRUST_BOND_DEACT_M)
    deact_pre[i] = deact

    hyper = np.zeros(7)
    hyper[ST_ABANDONMENT] = 0.45 + DEFAULT_TRAITS[T_ATTACHMENT_ANXIETY]*0.275 + DEFAULT_TRAITS[T_JEALOUSY_SENSITIVITY]*0.15
    hyper[ST_CLOSENESS]   = 0.30 + DEFAULT_TRAITS[T_ATTACHMENT_ANXIETY]*0.25
    hyper[ST_DEPENDENCY]  = 0.35 + DEFAULT_TRAITS[T_ATTACHMENT_ANXIETY]*0.20
    hyper[ST_VALIDATION]  = 0.15 + DEFAULT_TRAITS[T_ATTACHMENT_ANXIETY]*0.10
    hyper[ST_CONFLICT]    = 0.15 + DEFAULT_TRAITS[T_ATTACHMENT_ANXIETY]*0.15
    hyper[ST_TEASING]     = 0.10 + DEFAULT_TRAITS[T_JEALOUSY_SENSITIVITY]*0.10
    hyper[ST_EMOTIONAL_WEIGHT] = 0.20 + DEFAULT_TRAITS[T_ATTACHMENT_ANXIETY]*0.15
    _apply_additive(hyper, internal[i, I_INSECURITY], INSECURITY_HYPER_A)
    _apply_additive(hyper, internal[i, I_LONGING], LONGING_HYPER_A)
    _apply_multiplicative(hyper, rel[i, R_AFFECTION], AFFECTION_HYPER_M_NEW)
    _apply_multiplicative(hyper, rel[i, R_INTIMACY], INTIMACY_HYPER_M)
    hyper_pre[i] = hyper

print("\nC: 去激活 (sigmoid 前, 线性空间)")
run_pca(deact_pre, "去激活 (线性)", ST_LABELS)
print("\nC: 过度激活 (sigmoid 前, 线性空间)")
run_pca(hyper_pre, "过度激活 (线性)", ST_LABELS)

# 打印 deact_pre 的列均值，看基线偏移
print(f"\n  去激活 sigmoid 前均值 (n={N}):")
for d in range(7):
    print(f"    {ST_LABELS[d]:20s} mean={deact_pre[:,d].mean():.4f}, std={deact_pre[:,d].std():.4f}")

# 最后检查各维度在 sigmoid 前的动态范围
print(f"\n  各维度动态范围 (pre-sigmoid deactivation):")
for d in range(7):
    lo, hi = deact_pre[:,d].min(), deact_pre[:,d].max()
    print(f"    {ST_LABELS[d]:20s} [{lo:.3f}, {hi:.3f}]  range={hi-lo:.3f}")

# 与偏移量 0.35 比较
print(f"\n  sigmoid 的翻转点: deact-0.35 | hyper-0.38")
print(f"  deact 各维偏离 0.35 的程度:")
for d in range(7):
    offset = deact_pre[:,d].mean() - 0.35
    sigmoid_val = 1.0 / (1.0 + np.exp(-offset * 5.0))
    print(f"    {ST_LABELS[d]:20s} mean-0.35={offset:.4f}, sigmoid(offset*5)={sigmoid_val:.4f}")
