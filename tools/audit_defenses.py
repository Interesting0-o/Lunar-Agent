"""防御剖面独立性审计 — 基于 SPARSE_ANTAGONIST_ANALYSIS §6.3 方法。

生成 n=100,000 随机状态轨迹，对 compute_defense_profiles 输出的
deactivation (7,) 和 hyperactivation (7,) 做 PCA + 相关矩阵分析。
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from state import ST_LABELS
from state_engine._defenses import compute_defense_profiles
from tests.conftest import generate_random_states, generate_random_traits

N_SAMPLES = 100_000
RNG = np.random.default_rng(42)

SEP = "=" * 70
SUB = "-" * 50


def pca_analysis(data: np.ndarray, label: str, dim_labels: list) -> dict:
    """全手动 PCA: 中心化 → SVD → 方差分解 + 独立方差"""
    n, d = data.shape
    print(f"\n{SUB}")
    print(f"  {label}")
    print(f"{SUB}")

    centered = data - data.mean(axis=0)
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    explained_var = s ** 2 / (n - 1)
    total_var = explained_var.sum()
    explained_ratio = explained_var / total_var
    cum_ratio = np.cumsum(explained_ratio)

    # 有效维: 累积 95% 方差
    eff_dim_95 = int(np.searchsorted(cum_ratio, 0.95) + 1)

    # 有效秩: 基于解释方差比的谱熵
    # (归一化奇异值平方和, 等价于 explained_ratio)
    ratio = explained_ratio + 1e-30  # 避免 log(0)
    eff_rank = np.exp(-np.sum(ratio * np.log(ratio)))

    # 相关矩阵
    corr = np.corrcoef(data.T)
    np.fill_diagonal(corr, 0)
    max_abs_r = np.max(np.abs(corr))
    np.fill_diagonal(corr, 1)

    # 独立方差: 每维被其他 6 维回归后剩余的独特方差
    indep_var = np.zeros(d)
    for dim in range(d):
        other_dims = [j for j in range(d) if j != dim]
        X = np.column_stack([np.ones(n), data[:, other_dims]])
        y = data[:, dim]
        beta, res_sum_sq, _, _ = np.linalg.lstsq(X, y, rcond=None)
        total_ss = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - res_sum_sq[0] / total_ss
        indep_var[dim] = 1.0 - r2

    # 打印
    print(f"  PC1 解释方差: {explained_ratio[0]*100:.1f}%")
    print(f"  PC2 解释方差: {explained_ratio[1]*100:.1f}%")
    print(f"  PC3 解释方差: {explained_ratio[2]*100:.1f}%")
    print(f"  95% 有效维:   {eff_dim_95}/{d}")
    print(f"  有效秩:        {eff_rank:.2f}")
    print(f"  维度间 max|r|: {max_abs_r:.4f}")
    print(f"  均值独立方差:  {indep_var.mean()*100:.1f}%")
    print()

    # 每维明细
    sep_str = "  " + "-"*18 + " " + "-"*7 + " " + "-"*7 + " " + "-"*10
    hdr = f"  {'刺激维度':<18s} {'均值':>7s} {'标准差':>7s} {'独立方差':>10s}"
    print(hdr)
    print(sep_str)
    for dim in range(d):
        avg = data[:, dim].mean()
        std = data[:, dim].std()
        iv = indep_var[dim] * 100
        print(f"  {dim_labels[dim]:<18s} {avg:7.4f} {std:7.4f} {iv:9.1f}%")
    print()

    # 累积方差曲线
    print("  累积方差 (前 7 PC):")
    for i in range(d):
        bar_len = int(cum_ratio[i] * 30)
        bar = chr(0x2588) * bar_len
        print(f"    PC{i+1}: {cum_ratio[i]*100:5.1f}% {bar}")

    return {
        "eff_dim_95": eff_dim_95,
        "eff_rank": eff_rank,
        "pc1_ratio": explained_ratio[0],
        "max_abs_r": max_abs_r,
        "indep_var_mean": indep_var.mean(),
        "indep_var_per_dim": indep_var,
        "cum_ratio": cum_ratio,
        "explained_ratio": explained_ratio,
    }


def cross_profile_analysis(deact_all, hyper_all, results_deact, results_hyper):
    """交叉剖面分析"""
    print(f"{SUB}")
    print(f"  交叉剖面分析")
    print(f"{SUB}")

    flat_d = deact_all.T.flatten()
    flat_h = hyper_all.T.flatten()
    cross_corr = np.corrcoef(flat_d, flat_h)[0, 1]
    print(f"  deactivation x hyperactivation r = {cross_corr:.4f}")

    dim_cross = np.array([
        np.corrcoef(deact_all[:, d], hyper_all[:, d])[0, 1]
        for d in range(7)
    ])
    print(f"  逐维 deact x hyper r:")
    for d in range(7):
        print(f"    {ST_LABELS[d]:<18s} {dim_cross[d]:.4f}")

    print(f"\n  去激活内部相关矩阵 (max|r|={results_deact['max_abs_r']:.4f})")
    print(f"  过度激活内部相关矩阵 (max|r|={results_hyper['max_abs_r']:.4f})")


def summary_table(results_deact, results_hyper):
    """汇总对比表格"""
    print(f"\n{SEP}")
    print(f"  审计汇总")
    print(f"{SEP}")
    hdr = f"  {'指标':<35s} {'去激活':>10s} {'过度激活':>10s}"
    sep3 = "  " + "-"*35 + " " + "-"*10 + " " + "-"*10
    print(hdr)
    print(sep3)

    r = results_deact
    h = results_hyper
    print(f"  {'95% 有效维':<35s} {r['eff_dim_95']:>4d}/7{'':>5s} {h['eff_dim_95']:>4d}/7")
    print(f"  {'有效秩':<35s} {r['eff_rank']:>10.2f} {h['eff_rank']:>10.2f}")
    print(f"  {'PC1 解释方差':<35s} {r['pc1_ratio']*100:>9.1f}% {h['pc1_ratio']*100:>9.1f}%")
    print(f"  {'均值独立方差':<35s} {r['indep_var_mean']*100:>9.1f}% {h['indep_var_mean']*100:>9.1f}%")
    print(f"  {'维度间 max|r|':<35s} {r['max_abs_r']:>10.4f} {h['max_abs_r']:>10.4f}")

    # 与修复前对比
    print(f"\n  与修复前对比 (来自报告 §7.3.1):")
    vsep = "  " + "-"*35 + " " + "-"*12 + " " + "-"*14 + " " + "-"*14
    vhdr = f"  {'指标':<35s} {'修复前':>12s} {'修复后(deact)':>14s} {'修复后(hyper)':>14s}"
    print(vsep)
    print(vhdr)
    print(vsep)
    print(f"  {'有效秩':<35s} {'~1.0':>12s} {r['eff_rank']:>14.2f} {h['eff_rank']:>14.2f}")
    print(f"  {'PC1':<35s} {'~99%':>12s} {r['pc1_ratio']*100:>13.1f}% {h['pc1_ratio']*100:>13.1f}%")
    print(f"  {'均值独立方差':<35s} {'0.0%':>12s} {r['indep_var_mean']*100:>13.1f}% {h['indep_var_mean']*100:>13.1f}%")
    print(f"  {'95% 有效维':<35s} {'1/7':>12s} {r['eff_dim_95']:>4d}/7{'':>8s} {h['eff_dim_95']:>4d}/7")
    print()


def main():
    print(SEP)
    print("防御剖面独立性审计")
    print(f"n = {N_SAMPLES:,} 随机状态样本")
    print(SEP)

    # Step 1: 生成随机状态
    print("\n[1/4] 生成随机状态...")
    traits = generate_random_traits(RNG, N_SAMPLES)           # (N, 10)
    internal, relationship = generate_random_states(RNG, N_SAMPLES)  # (N, 8), (N, 6)

    # Step 2: 计算防御剖面
    print("[2/4] 计算防御剖面 (N=100,000)...")
    deact_all = np.empty((N_SAMPLES, 7), dtype=np.float64)
    hyper_all = np.empty((N_SAMPLES, 7), dtype=np.float64)

    for i in range(N_SAMPLES):
        profiles = compute_defense_profiles(traits[i], relationship[i], internal[i])
        deact_all[i] = profiles[0]
        hyper_all[i] = profiles[1]

    # Step 3: PCA 分析
    print("[3/4] PCA + 独立方差分析...")
    results_deact = pca_analysis(deact_all, "去激活剖面 (Deactivation)", ST_LABELS)
    results_hyper = pca_analysis(hyper_all, "过度激活剖面 (Hyperactivation)", ST_LABELS)

    # 交叉剖面分析
    print("[4/4] 交叉剖面分析...")
    cross_profile_analysis(deact_all, hyper_all, results_deact, results_hyper)

    # 汇总
    summary_table(results_deact, results_hyper)


if __name__ == "__main__":
    main()
