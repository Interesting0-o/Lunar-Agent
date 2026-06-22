# State Engine 测试报告

> **报告日期**: 2026-06-22 | **用例数**: 249 | **通过率**: 100% (242/242, 7 skipped) | **执行时间**: 133s
>
> 本版变更：Hyperactivation 从纯秩-1 拆分为**人格基线（秩-1, traits+rel）** + **状态调制（HYPER_STATE_MODULATION）**。RELATIONSHIP_COUPLING 从 66.7% 密度压缩至 22.2%（约束⑥合规）。新增参数灵敏度分析模块（12 项测试）。**修复约束②（StimulusMetadata）和约束⑪（formatter 连续投影）**。全管线 242 项通过。

---

## 一、测试体系架构

### 1.1 分层结构

测试按引擎层纵向组织：

```
层 1 — 数值工具 (_utils.py)
  │   soft_clamp, sigmoid 的数值稳定性
  │
层 2 — B 矩阵 + 耦合矩阵 (_matrices.py)
  │   形状、方向性、谱半径、符号结构
  │
层 3 — 防御剖面 (_defenses.py)
  │   形状、范围、心理方向、调制特异性、apply_defenses 不变量
  │
层 4 — 残差动力学 (_dynamics.py)
  │   计算 setpoint → 零刺激收敛 → 刺激方向 → 速率调制 → 
  │   变化率上限 → 强度单调性 → 交互边界
  │
层 5 — 表面投影 (_surface.py)
  │   边界、方向性、外刺激效应、惯性混合、表面→内部反馈
  │
层 6 — 时间衰减 (_decay.py)
  │   基本衰减 → 不对称 → 人格调制 → 时间曲线 → 边界鲁棒性
  │
层 7 — 管线 (pipeline + 场景 + Monte Carlo)
  │   update_all 端到端 → 4 场景 → 重复刺激单调 → 500k 随机
  │
层 8 — 对抗引擎 (adversarial_engine)
  │   对称/非对称特质 → 极端特质 → 噪声 → 长程稳定性 → 10,000 轮
  │
层 9 — 异常检测 (anomalies)
  │   响应度 → 速率参数 → 饱和 → 防御塌缩 → trait 敏感度 → 迟滞
  │
层 10 — 参数灵敏度 (sensitivity) [新增]
  │   每参数扰动分析 → 冗余检测 → 边界安全验证
```

### 1.2 本版新增覆盖（06-22，12+12 项）

| 新增测试 | 类型 | 覆盖缺口 |
|---------|------|----------|
| `TestParameterSensitivity` (12 项) | 结构+验证 | 参数冗余检测、安全感上限 |
| `TestFullSensitivityReport` (1 项, 标记跳过) | 报告 | `--run-full-sensitivity` 全量报告 |

详见第四章。

---

## 二、测试用例全清单

### 2.1 数值工具 (`test_utils.py` — 23 项)

**`TestSoftClampBounds`**（5 项）— soft_clamp 边界行为：
1. `test_identity_within_bounds_single` — 单值在 [-1, 1] 内通过不变
2. `test_identity_within_bounds_massive` — 50,000 随机值在 [-1, 1] 内全部通过
3. `test_upper_suppression` — x > 1 平滑压缩到 [1.0, ~1.1]
4. `test_lower_suppression` — x < -1 平滑压缩到 [~-1.1, -1.0]
5. `test_extreme_values_no_nan` — ±1e10, ±1e100, inf 全部产生有限值

**`TestSoftClampMonotonicity`**（4 项）— 全局单调性：
6. `test_monotonic_within_bounds` — (-0.95, 0.95) 内严格单调递增
7. `test_monotonic_below_low` — x < -1 时严格单调递增
8. `test_monotonic_above_high` — x > 1 时严格单调递增
9. `test_global_monotonic` — [-1.5, 1.5] 全域单调

**`TestSoftClampCustomBounds`**（4 项）— 自定义边界：
10. `test_custom_bounds_identity` — [-0.5, 0.5] 内通过不变
11. `test_custom_bounds_symmetric` — ±2.0 压缩到 ~1.1/ -1.1
12. `test_old_default_bounds` — 旧版 [0, 1] 默认边界兼容
13. `test_wide_transition` — 宽过渡区允许更多 overshoot

**`TestSigmoid`**（7 项）— sigmoid 数值性质：
14. `test_center` — sigmoid(0) = 0.5
15. `test_symmetry` — sigmoid(-x) = 1 - sigmoid(x)
16. `test_monotonic` — 严格单调递增
17. `test_large_positive_no_overflow` — x=1000 输出 ≈ 1.0，有限
18. `test_large_negative_no_underflow` — x=-1000 输出 ≈ 0.0，有限
19. `test_range` — 所有输出 ∈ [0, 1]
20. `test_bulk_monotonic_many_distributions` — uniform/normal/beta 下单调

**`TestStressSoftClamp`**（3 项）— 极端条件稳健性：
21. `test_random_bounds_identity` — 随机 low/high，内部通过不变
22. `test_random_bounds_suppression` — 随机边界，外部压缩，无 NaN
23. `test_tiny_transition_no_nan` — transition=1e-6 不产生 NaN

### 2.2 B 矩阵 + 耦合 (`test_matrices.py` — 9 项)

**`TestMatrixShapes`**（1 项）：
1. `test_input_influence_shape` — INPUT_INFLUENCE_B 形状为 (7, 8)

**`TestInputInfluenceDirectional`**（4 项）— B 矩阵方向正确性：
2. `test_conflict_causes_stress` — B[conflict, stress] > 0
3. `test_conflict_causes_irritation` — B[conflict, irritation] > 0
4. `test_validation_reduces_insecurity` — B[validation, insecurity] < 0
5. `test_abandonment_increases_insecurity` — B[abandonment, insecurity] > 0

**`TestRelStimulusDirection`**（2 项）— 关系刺激方向性：
6. `test_rel_conflict_reduces_trust_bond` — conflict → trust_bond 下降
7. `test_rel_validation_increases_affection` — validation → affection 上升

**`TestCouplingContractivity`**（2 项）：
8. `test_internal_coupling_spectral_radius` — 耦合雅可比谱半径 ρ < 0.95（实测 0.0986）
9. `test_coupling_sign_structure` — 5 条耦合边符号符合心理学预期

### 2.3 防御剖面 (`test_defenses.py` — 26 项)

**`TestDefenseProfilesShape`**（2 项）：
1. `test_shape` — profiles shape = (2, 7)
2. `test_range_default` — 默认 profiles ∈ [0, 1]

**`TestDefenseProfilesBulk`**（3 项）：
3. `test_all_in_bounds` — 20,000 随机输入，profiles ∈ [0, 1]
4. `test_extreme_traits_bounds` — 10,000 beta(0.2,0.2) 输入，无 NaN/Inf
5. `test_profile_statistics` — 20,000 样本 deact/hyper 均值（诊断）

**`TestDefenseDirectional`**（11 项）— 心理方向验证：
6. `test_high_avoidance_high_deactivation` — 高回避 → deact ↑
7. `test_high_anxiety_high_hyperactivation` — 高依恋焦虑 → hyper ↑
8. `test_high_pride_high_deactivation` — 高自尊 → deact ↑
9. `test_trust_reduces_deactivation` — 高信任 → deact ↓
10. `test_affection_boosts_hyperactivation` — 高好感 → hyper ↑
11. `test_insecurity_increases_deactivation` — 高不安 → deact ↑
12. `test_insecurity_increases_hyperactivation` — 高不安 → hyper ↑
13. `test_high_stability_low_deactivation` — 高稳定 → deact ↓
14. `test_stress_modulation_is_dimension_specific` — stress 对 deact 逐维影响力不同
15. `test_trust_modulation_is_dimension_specific` — trust 对 deact 逐维影响力不同
16. `test_insecurity_hyperactivation_is_dimension_specific` — insecurity 只影响 abandonment（HYPER_STATE_MODULATION 维度特异性）

**`TestApplyDefenses`**（7 项）— 防御应用核心不变量：
17. `test_zero_stimuli_zero_output` — 零刺激 → inner=outer=0
18. `test_inner_ge_outer` — **核心不变量**：20k 样本 inner ≥ outer 逐元素
19. `test_inner_ge_outer_high_stimuli_edge_case` — 高刺激可轻度违反 inner≥outer，max < 0.05
20. `test_hyper_amplifies` — hyper=0.9 → inner ≥ stimuli（放大）
21. `test_deact_suppresses_outer` — deact=0.9 → outer 被压到 stimuli 以下
22. `test_both_high_independent` — hyper=deact=0.9 → inner > stimuli, outer < inner（口是心非的计算实现）
23. `test_output_bounds_bulk` — 20,000 随机输入，inner/outer ∈ [0, 1]

**`TestDefenseCornerCases`**（3 项）：
24. `test_extreme_stimuli_all_zero` — 零刺激 + 默认 profile → inner=outer=0
25. `test_extreme_stimuli_all_one` — 全 1 刺激 ∈ [0, 1.11]
26. `test_extreme_profiles` — 全 0 profile: inner=outer=0.5；全 1 profile: inner>0.5, outer<inner

### 2.4 残差动力学 (`test_dynamics.py` — 46 项)

**`TestSetpoint`**（5 项）— 人格稳态基线：
1. `test_internal_setpoint_range` — 20k 随机 traits，setpoint ∈ [-0.9, 0.9]
2. `test_rel_setpoint_range` — 20k 随机 traits，rel setpoint ∈ [-0.96, 0.96]
3. `test_high_anxiety_higher_stress_setpoint` — 高焦虑 → stress setpoint ↑
4. `test_high_avoidance_lower_trust_setpoint` — 高回避 → trust setpoint ↓
5. `test_default_setpoint_finite` — 全部有限

**`TestConvergence`**（4 项）— 零刺激收敛性：
6. `test_internal_converges_stable` — 零刺激 2000 轮，收敛到耦合平衡点
7. `test_internal_converges_from_extremes` — 不同极端起点收敛到相近 L2 区域
8. `test_relationship_converges_stable` — 关系态零刺激收敛
9. `test_zero_stimulus_no_divergence` — 2000 轮零刺激，无 NaN，状态 ∈ [-1.1, 1.1]

**`TestStimulusDirectionality`**（10 项）— B 矩阵方向管线验证：
10. `test_abandonment_increases_insecurity` — 抛弃 → insecurity↑ loneliness↑
11. `test_validation_reduces_insecurity` — 认可 → insecurity↓
12. `test_closeness_reduces_loneliness` — 靠近 → loneliness↓
13. `test_conflict_increases_stress` — 冲突 → stress↑ irritation↑ energy↓
14. `test_dependency_reduces_loneliness` — 被需要 → loneliness↓
15. `test_emotional_weight_increases_stress` — 沉重 → stress↑ mental_fatigue↑
16. `test_rel_abandonment_reduces_trust_bond` — 抛弃 → trust↓
17. `test_rel_validation_increases_affection` — 认可 → affection↑
18. `test_rel_closeness_increases_intimacy` — 靠近 → intimacy↑
19. `test_rel_conflict_reduces_trust_bond` — 冲突 → trust↓

**`TestDefenseRateModulation`**（2 项）：
20. `test_hyper_increases_stimulus_acceptance` — hyper=0.9 比 hyper=0.1 状态变化更大
21. `test_deact_reduces_stimulus_response` — deact=0.9 比 deact=0.1 状态变化更小

**`TestDynamicsBulk`**（4 项）：
22. `test_internal_update_bounds` — 20k 随机输入，输出 ∈ [-1, 1.11]，无 NaN
23. `test_relationship_update_bounds` — 20k 随机输入，输出 ∈ [-1, 1.11]，无 NaN
24. `test_time_scale_separation` — 关系态变化幅度 < 内部态变化幅度
25. `test_change_magnitudes_statistics` — 15k 样本，最大单步变化分布；> 0.5 的 ≤ 5 例

**`TestLongConvergenceStress`**（2 项）：
26. `test_thousand_rounds_no_nan` — 1000 步随机刺激，无 NaN
27. `test_alternating_scenarios_no_divergence` — 500 步交替正/负刺激，L2 < 5.0

**`TestPerTurnRateLimit`**（3 项）：
28. `test_internal_rate_limits` — 20k 随机输入，内部 8 维每轮 |Δ| ≤ 安全上限
29. `test_relationship_rate_limits` — 20k 随机输入，关系 3 维每轮 |Δ| ≤ 上限
30. `test_time_scale_separation` — 10k 样本 α/α_rel 均值比 ≥ 2.0（实测 6.1×）

**`TestStimulusIntensityMonotonicity`**（16 项）：
31-46. `test_internal_monotonic[16 条 B 直连路径]` — 每条路径 15 强度级验证单调性
47. `test_internal_monotonic_bulk` — 所有路径 10 组随机 trait 散弹验证
48. `test_zero_stimulus_no_change` — 50 轮零刺激，|Δ| < 0.15

**`TestStimulusInteraction`**（4 项）：
49. `test_validation_compensates_abandonment` — validation 缓解 abandonment 引起的不安
50. `test_abandonment_worsens_conflict_stress` — abandonment + conflict → stress 不低于单独 conflict
51. `test_validation_plus_closeness_boosts_energy` — 联合 energy 不低于单独 validation
52. `test_non_additivity_bounded` — 非加性偏差 > 0.1 的比例 < 10%

### 2.5 表面投影 (`test_surface.py` — 22 项)

**`TestSurfaceProjectionBounds`**（3 项）：
1. `test_default_output` — 默认输入 ∈ [-1, 1]
2. `test_bulk_random` — 20k 随机输入 ∈ [-1, 1.11]，无 NaN
3. `test_extreme_inputs` — ±1/0 所有组合，有限

**`TestSurfaceDirectionality`**（5 项）：
4. `test_high_energy_high_enthusiasm` — energy↑ → enthusiasm↑
5. `test_high_irritation_high_sharpness` — irritation↑ → sharpness↑
6. `test_high_affection_high_warmth` — affection↑ → warmth↑
7. `test_high_stress_low_warmth` — stress↑ → warmth↓
8. `test_high_fatigue_low_expressiveness` — fatigue↑ → expressiveness↓

**`TestOuterStimuliEffect`**（3 项）：
9. `test_validation_outer_increases_warmth` — 外刺激 validation → warmth↑
10. `test_conflict_outer_increases_sharpness` — 外刺激 conflict → sharpness↑
11. `test_outer_stimuli_different_from_inner_only` — 不同外刺激产生不同 surface

**`TestSurfaceStatistics`**（1 项）：
12. `test_surface_distribution` — 20k 样本，每维 std > 0.01（无退化维度）

**`TestSurfaceInertia`**（3 项）：
13. `test_inertia_changes_output` — 相同 raw 不同 prev 产生不同 surface
14. `test_no_prev_differs_from_zero_prev` — None 无拖拽，全零有拖拽
15. `test_high_stress_increases_inertia` — 高压力下 surface 更接近 prev（α 更低）

**`TestSurfaceAlpha`**（4 项）：
16. `test_alpha_bounds` — α ∈ [0.1, 0.9]
17. `test_alpha_default` — 默认状态 α ≈ 0.58
18. `test_alpha_stress_reduces_alpha` — 高压力→α↓
19. `test_alpha_energy_increases_alpha` — 高精力→α↑

**`TestSurfaceFeedback`**（3 项）：
20. `test_feedback_shape` — 反馈为 (8,) 向量
21. `test_feedback_magnitude` — |fb| < 0.2
22. `test_feedback_zero_with_zero_surface` — 零 surface → 零反馈

### 2.6 时间衰减 (`test_decay.py` — 33 项)

**`TestBasicDecay`**（5 项）：
1. `test_zero_delta_no_change` — dt=0 → 不变
2. `test_microscopic_delta_no_change` — dt < min_delta → 跳过
3. `test_at_setpoint_no_change` — 已达 setpoint → 衰减后不变
4. `test_convergence_toward_setpoint` — 长 dt 趋近 setpoint
5. `test_monotonicity_wrt_delta` — dt 越大越接近 setpoint

**`TestAsymmetricDecay`**（5 项）：
6. `test_negative_faster_than_positive` — 负面偏差恢复快于正面（关系态）
7. `test_asymmetry_ratio_approaches_boost` — 非对称比趋近配置值 1.8
8. `test_internal_state_not_affected` — 内部态 NOT 非对称
9. `test_config_change_boost` — boost 1→2→3 倍残余递减
10. `test_each_dimension_independent` — 每个关系维独立符合负快于正

**`TestPersonalityModulation`**（5 项）：
11. `test_internal_mod_range` — 内部人格调制 ∈ [0.3, 2.0]
12. `test_relationship_mod_range` — 关系调制 ∈ [0.3, 2.0]
13. `test_internal_mod_direction` — 稳定/乐观 → mod>1（快恢复）；焦虑/易怒 → mod<1（慢恢复）
14. `test_relationship_mod_direction` — 回避 → mod>1；焦虑 → mod<1
15. `test_personality_affects_recovery_rate` — 回避比焦虑从关系损伤中恢复更快

**`TestTimeCurve`**（3 项）：
16. `test_lambda_decreases_with_delta` — λ_eff 随 dt 单调递减
17. `test_very_long_delta_lambda_approaches_zero` — dt=1e6 时 λ_eff → 极小
18. `test_time_curve_difference` — 内部阻尼 (k=0.05) > 关系阻尼 (k=0.001)

**`TestBoundaryRobustness`**（3 项）：
19. `test_extreme_values_stay_in_bounds` — ±1.5 输入衰减后 ∈ [-1.1, 1.1]
20. `test_all_traits_extremes` — 全 traits 极端，输出 ∈ [-1, 1]
21. `test_single_dimension_decay_isolation` — 仅 perturbed 维变化，其他保持 setpoint

**`TestBulkStatistics`**（6 项）：
22. `test_no_boundary_violations` — 5000 样本无越界
23. `test_setpoint_convergence` — dt=10000 时 max 误差 < 0.6
24. `test_asymmetry_anomaly_detection` — 无 NaN/Inf/负比率
25. `test_no_nan_in_outputs` — 各种 dt + 极端态，无 NaN
26. `test_internal_no_asymmetry_in_bulk` — 内部态非对称比 ≈ 1.0
27. `test_convenience_api_alignment` — apply_time_decay 与独立调用一致

**`TestVisualization`**（6 项，CI 跳过）

### 2.7 管线 (`test_pipeline.py` — 24 项)

**`TestPipelineBasics`**（5 项）：
1. `test_first_run_uses_initialize` — None 状态 → initialize_all
2. `test_output_shapes` — internal=(8,), relationship=(3,), surface=(7,)
3. `test_all_outputs_finite` — 无 NaN/Inf
4. `test_all_outputs_in_bounds` — ∈ [-1, 1]
5. `test_zero_stimuli_small_change` — 零刺激 max 变化 < 0.1

**`TestScenarios`**（6 项）— 心理场景推理：
6. `test_abandonment_scenario` — 抛弃→ insecurity↑ loneliness↑ stress↑ trust↓
7. `test_validation_scenario` — 认可→ insecurity↓ affection↑
8. `test_conflict_scenario` — 冲突→ stress↑ irritation↑ energy↓ trust↓
9. `test_closeness_scenario` — 靠近→ loneliness↓ affection↑ trust↑（跨尺度耦合）
10. `test_teasing_scenario` — 调侃→ intimacy↑
11. `test_stimulus_specificity` — 每种刺激只影响预期维度

**`TestRepeatedSingleStimulus`**（4 项）：
12. `test_repeated_abandonment_monotonic` — 多次抛弃→ insecurity/loneliness/stress 单调↑
13. `test_repeated_validation_monotonic` — 多次认可→ insecurity 单调↓
14. `test_repeated_closeness_monotonic` — 多次靠近→ loneliness 单调↓（允许软边界 < 1e-4 浮动）
15. `test_repeated_teasing_monotonic` — 多次调侃→ intimacy 单调↑

**`TestMultiRound`**（4 项）：
16. `test_cumulative_conflict` — 10 轮冲突→ stress 单调↑, trust 单调↓
17. `test_cumulative_validation` — 10 轮认可→ insecurity 单调↓, affection 单调↑
18. `test_saturation_behavior` — 50 轮全 0.8 刺激，状态 ∈ [-1.11, 1.11]
19. `test_stimulus_cessation_stops_accumulation` — 停止刺激后不再恶化

**`TestMonteCarloMassive`**（2 项）：
20. `test_massive_random_stimuli` — 200,000 随机刺激，NaN=0，越界 < 0.5%
21. `test_monte_carlo_with_random_traits` — 50,000 随机 traits+states+stimuli，越界=0

**`TestStatisticsSummary`**（3 项）：
22. `test_internal_state_spread` — 50k 样本，各维 σ > 0.005
23. `test_surface_state_spread` — 50k 样本，surface σ 分析
24. `test_single_stimulus_impact_matrix` — 每种刺激在 1.0 时的单维冲击（诊断）

### 2.8 对抗引擎 (`test_adversarial_engine.py` — 23 项)

（无变动，与上一版一致）

**`TestBasicCoupling`**（3 项）
**`TestAsymmetricTraits`**（3 项）
**`TestExtremeTraits`**（12 项）
**`TestAdversarialMapping`**（3 项）
**`TestPerturbationInjection`**（2 项）
**`TestMonteCarlo`**（2 项）
**`TestLongRunStability`**（2 项）
**`TestDefenseProfileDynamics`**（1 项）
**`TestSetpointConvergence`**（2 项）
**`TestStressScenarios`**（2 项）

### 2.9 异常检测 (`test_anomalies.py` — 13 项)

（无变动，与上一版一致）

**`TestAnomalySingleRoundResponsiveness`**（3 项，诊断）
**`TestAnomalyRateParameters`**（2 项，诊断）
**`TestAnomalySaturation`**（2 项）
**`TestAnomalyDefenseCollapse`**（1 项）
**`TestAnomalyTraitSensitivity`**（1 项，诊断）
**`TestAnomalyMultiRoundDrift`**（1 项，诊断）
**`TestAnomalySurfaceDegeneracy`**（1 项）
**`TestAnomalyBetaModulation`**（1 项）
**`TestAnomalyMatrixNoise`**（1 项）

### 2.10 参数灵敏度 (`test_sensitivity.py` — 12 项，🆕 新增)

**`TestParameterSensitivity`**（12 项）— 每参数组独立验证：
1. `test_B_int_sensitivity` — INPUT_INFLUENCE_B 的 16 个非零元素（max_s < 0.50）
2. `test_B_rel_sensitivity` — REL_INPUT_INFLUENCE_B 的 6 个非零元素
3. `test_surface_feedback_sensitivity` — SURFACE_FEEDBACK_MATRIX 的 9 个非零元素
4. `test_internal_coupling_sensitivity` — INTERNAL_COUPLING 的 11 个非零元素
5. `test_cross_scale_coupling_sensitivity` — CROSS_SCALE_COUPLING 的 5 个非零元素
6. `test_self_decay_sensitivity` — SELF_DECAY 的 8 个值
7. `test_decay_targets_sensitivity` — DECAY_TARGETS 的 8 个值
8. `test_rel_self_decay_sensitivity` — REL_SELF_DECAY 的 3 个值
9. `test_surface_mapper_sensitivity` — SURFACE_MAPPER 的 21 个连接 + 7 偏置
10. `test_hyper_state_mod_sensitivity` — HYPER_STATE_MODULATION 的 15 个连接
11. `test_deact_intensity_sensitivity` — DEACT_INTENSITY 的 9 个连接
12. `test_hyper_intensity_sensitivity` — HYPER_INTENSITY 的 6 个连接

**`TestFullSensitivityReport`**（1 项，标记跳过）：
13. `test_full_report` — 仅 `--run-full-sensitivity` 启用，打印完整灵敏度分析报告

每项测试验证：`max_s < 0.50`（扰动 ×0.5 后输出变化不超过基线的 50%）。

---

## 三、关键实证数据

### 3.1 单轮变化率上限（n=500,000）

| 维度 | 实测 max\|Δ\| | 安全上限（+50%） | 安全上限测试 |
|------|:-----------:|:--------------:|:-----------:|
| energy | 0.1215 | **0.19** | ✅ |
| stress | 0.2356 | **0.36** | ✅ |
| loneliness | 0.2157 | **0.33** | ✅ |
| insecurity | 0.1828 | **0.28** | ✅ |
| irritation | 0.1882 | **0.29** | ✅ |
| longing | 0.1679 | **0.26** | ✅ |
| social_battery | 0.1340 | **0.21** | ✅ |
| mental_fatigue | 0.1607 | **0.25** | ✅ |
| affection | 0.0218 | **0.04** | ✅ |
| trust_bond | 0.0286 | **0.05** | ✅ |
| intimacy | 0.0216 | **0.04** | ✅ |

所有上限通过 20,000 组随机输入验证，**无越界**。

### 3.2 刺激强度单调性

**16 条 B 矩阵直连路径全部单调**。实测方法：
- 每条路径取 15 个强度点 [0.0, 0.071, ..., 1.0]
- 30 组随机 traits + internal + relationship
- 检查状态变化量的符号一致性（所有非零 diff 同号）

### 3.3 零刺激收敛

| 轮数 | 内部态 L2 | 关系态 L2 |
|:----:|:---------:|:---------:|
| 100 | 0.4137 | 0.0603 |
| 500 | 0.1091 | 0.0137 |
| 2000 | 0.1034 | 0.0072 |

不发散，收敛到耦合平衡点（非 setpoint 锚定）。有效雅可比谱半径 ρ = 0.19（远 < 1.0）。

### 3.4 双刺激交互分析

| 交互对 | 验证内容 | 结果 |
|--------|---------|:----:|
| abandonment + validation | insecurity 增量 ≤ 单独 abandonment | ✅ |
| abandonment + conflict | stress 增量 ≥ 单独 conflict | ✅ |
| validation + closeness | energy 增量 ≥ 单独 validation | ✅ |

非加性边界：交互效应主要来自 soft_clamp 非线性，max 非加性偏离约 0.07（极端边界附近）。

### 3.5 防御剖面分布（n=30,000）

| 剖面 | 均值范围 | 覆盖率 |
|------|:--------:|:------:|
| deactivation | [0.063, 0.811] | 74.8% |
| hyperactivation | [0.061, 0.723] | 66.2% |

极值接近但未完全覆盖 [0, 1] 全域——上界受限由 sigmoid 饱和特性导致。

### 3.6 表面放大系数（n=20,000）

| 指标 | 数值 |
|------|:----:|
| 内部态 σ 均值 | 0.522 |
| 表面 σ 均值 | 0.277 |
| 表面/内部 σ 比 | 0.53 |

表面表达比内部感受"更收敛"——验证了防御机制削减极端表达的预期效果。

### 3.7 耦合结构

| 属性 | 数值 | 判定 |
|------|:----:|:----:|
| 内部耦合密度 | 11/64 (17.2%) | ✅ 稀疏 |
| 关系耦合密度 | **2/9 (22.2%)** | ✅ **≤30%（已修复）** |
| 内部耦合谱半径 | 0.0986 | ✅ ≪ 1.0 |
| 有效雅可比谱半径 | 0.1897 | ✅ ≪ 1.0 |
| 时间尺度比 α/α_rel | 6.1× | ✅ 分离充分 |

### 3.8 参数灵敏度分布（🆕 新增）

30 场景 × 20 轮 × 扰动 ×0.5 的累积灵敏度：

| 级别 | 参数数 | 占比 |
|:----:|:-----:|:----:|
| 🔴 强 (>0.15) | 4 | 3% |
| 🟠 中 (0.05-0.15) | 21 | 17% |
| 🟡 弱 (0.01-0.05) | 39 | 32% |
| 🟢 忽略 (<0.01) | **60** | **48%** |

**各组件平均灵敏度**：

```
HyperI        ████  0.0775  │ 1强 4中弱 1忽略  ← 最强！回避(-2.10)敏感度0.31
SELF_DECAY    ███   0.0652  │ 0强 8中弱 0忽略
B_int         ███   0.0604  │ 3强 11中弱 2忽略
B_rel         ██    0.0450  │ 0强 6中弱 0忽略
int_coup      ██    0.0435  │ 0强 9中弱 2忽略
REL_SELF_DEC  █     0.0188  │ 0强 2中弱 1忽略
xscale        █     0.0130  │ 0强 3中弱 2忽略
SurfMap       █     0.0123  │ 0强 10中弱 18忽略 ← 18/28 可移除
surf_fb       ▏     0.0074  │ 0强 3中弱 6忽略
HState        ▏     0.0061  │ 0强 3中弱 12忽略 ← 12/15 可移除
DECAY_TARGETS ▏     0.0032  │ 0强 1中弱 7忽略   ← 仅social_battery有效
DeactI        ▏     0.0005  │ 0强 0中弱 9忽略   ← 单会话内不激活
```

---

## 四、新增覆盖率分析（06-22）

本版补充 12 项测试，覆盖参数灵敏度分析维度。

### 4.1 参数灵敏度（TestParameterSensitivity）

**背景**：173 个参数的系统需要可重复的定量验证，确保：
1. 参数扰动不导致系统发散（max_s < 0.50 安全边界）
2. 发现冗余参数以简化模型

**方案**：对每个参数组的每个非零元素，扰动 ×0.5，跑 10 场景 ×20 轮，测量归一化灵敏度 `s = mean(|Δoutput|) / mean(|output|)`。直接修改编译后的 `_weight_matrix`，绕过 SemanticWeight 的 domain 验证。

**问题发现**：~48% 的参数在多轮累积下灵敏度 < 0.01，可安全移除候选包括：
- `DECAY_TARGETS` 的 7 个零值（仅 social_battery 有效）
- `SurfMap` 的 18 个弱连接
- `HState` 的 12 条弱连接

---

## 五、异常与问题清单

### 5.1 异常值

| 类型 | 项目 | 严重度 | 说明 |
|------|------|:------:|------|
| 维度 | surface 全部维同向运动 > 96% | 🟡 | 耦合驱动非独立表达，设计约束非 bug |
| 维度 | vulnerability 触底率 5.2% | 🟡 | 基线 -0.5 偏低，高 pride 压抑脆弱 |
| 性能 | conflict 影响 0.056 vs teasing 0.019 | 🟡 | 2.9:1 不均衡，B 矩阵入边数不同导致 |
| 性能 | 往返迟滞 4.40 | 🟡 | 无 γ 后状态累积，需时间衰减周期恢复 |
| 剖面 | deact/hyper 极值未覆盖 [0,1] 全域 | 🟢 | sigmoid 饱和，上界约 0.81/0.72 |
| 精度 | magnitude 与 \|value\| 不匹配（4 处） | ℹ️ | 校准参数轻微超 range，功能无影响 |

### 5.2 已知约束违反

| 约束 | 违反位置 | 严重度 | 说明 |
|:----:|---------|:------:|------|
| ① | `_surface.py` | ✅ **已修复** | traits 已从 surface 移除（06-22 重构） |
| ② | `perception.py` | ✅ **已修复** | `StimulusMetadata` 已实现（含 confidence/source/decay_modulator/timestamp），confidence 缩放刺激 + source=3 清零 + decay_modulator 传入 _decay（06-22） |
| ④ | `_surface.py` | ⚠️ **by design** | outer_stimuli 进 surface 已由 defenses 压抑（06-22 确认） |
| ⑪ | `state_formatter.py:_desc()` | ✅ **已修复** | 5 级硬阈值替换为 9 区连续投影，锚点之间用"略偏"前缀平滑过渡（06-22） |
| ⑥ | ~~关系耦合 66.7%~~ | ✅ **已修复** | 改为级联设计（AFF→TRUST→INTIM），密度降至 22.2%（06-22） |

### 5.3 已解决问题（本版关闭）

| 问题 | 解决方案 | 日期 |
|------|---------|:----:|
| 约束⑥ 关系耦合密度 66.7% | 改为级联设计(2 连接)，密度降至 22.2% | 06-22 |
| Hyper 人格/状态方差方向不一致 | 拆分为人格基线(秩-1) + 状态调制(HYPER_STATE_MODULATION) | 06-22 |
| 无参数灵敏度守护 | 新增 test_sensitivity.py（12 项测试） | 06-22 |
| 防御剖面无交叉调制能力 | HYPER_STATE_MODULATION 提供维度特异性 | 06-22 |

---

## 六、不变式清单

| # | 不变式 | 验证方式 | 状态 |
|:-:|--------|---------|:----:|
| 1 | 所有状态向量 ∈ soft_clamp 范围 [-1.1, 1.1] | 20k~500k 批量测试 | ✅ |
| 2 | sigmoid 全域单调递增，输出 ∈ [0, 1] | 7 项 sigmoid 测试 | ✅ |
| 3 | profiles[0,1] ∈ [0, 1] | 3 项批量测试 + 30k 极端 | ✅ |
| 4 | inner ≥ outer（中等刺激） | 20k 样本逐元素比较 | ✅ |
| 5 | 零刺激收敛到耦合平衡，不发散 | 2000 轮收敛 + 10,000 轮长程 | ✅ |
| 6 | setpoint ∈ [-0.9, 0.9] | 20k 随机 traits | ✅ |
| 7 | 防御剖面独立性 | r=0.23 < 0.3 | ✅ |
| 8 | 非对称衰减负/正恢复比 ≈ 1.8 | 200+ 样本 | ✅ |
| 9 | 单轮变化率不超过安全上限 | 20k 样本 × 18 维 | ✅ |
| 10 | B 矩阵直连路径强度单调 | 16 路径 × 15 强度级 | ✅ |
| 11 | 耦合收缩 ρ(J) < 1.0 | 数值雅可比谱半径 | ✅ 实测 0.19 |
| 12 | 时间尺度分离 α/α_rel ≥ 2.0 | 10k 样本均值比 | ✅ 实测 6.1× |
| 13 | 参数灵敏度 max_s < 0.50 | 12 组 × 扰动 ×0.5 | ✅ 🆕 |
| 14 | 矩阵密度 ≤ 30%（约束⑥） | 所有 2D 矩阵 | ✅ 🆕 修复(22.2%) |

---

## 七、运行方式

```bash
# 全部测试（249 项）
uv run pytest tests/ -v

# 本版新增测试
uv run pytest tests/test_sensitivity.py -v                     # 参数灵敏度（12 项，快速）
uv run pytest tests/ --run-full-sensitivity                    # 全量灵敏度报告（30 场景）

# 单模块测试
uv run pytest tests/test_utils.py -v                           # 数值工具（23 项）
uv run pytest tests/test_matrices.py -v                        # 矩阵（9 项）
uv run pytest tests/test_defenses.py -v                        # 防御剖面（26 项）
uv run pytest tests/test_dynamics.py -v                        # 动力学（46 项）
uv run pytest tests/test_surface.py -v                         # 表面（22 项）
uv run pytest tests/test_decay.py -v                           # 衰减（33 项）
uv run pytest tests/test_pipeline.py -v                        # 管线（24 项）
uv run pytest tests/test_adversarial_engine.py -v              # 对抗引擎（23 项）
uv run pytest tests/test_anomalies.py -v -s                    # 异常检测（13 项，诊断输出）
uv run pytest tests/test_sensitivity.py -v                     # 参数灵敏度（12 项）

# 极限压力
uv run pytest tests/test_pipeline.py::TestExtremeStress -v -s
```
