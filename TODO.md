# TODO

> Lunar 项目待办事项与已知问题清单。
> 对抗检验报告见 `md/STATE_ENGINE_TEST_REPORT.md`，soft_clamp 分析见 memory。

---

## 已完成 ✅

- [x] 移除 HiddenState 层（已并入 Gate Control + Surface Projection）
- [x] 移除 SocialSignals / InteractionImpact 中间层（perception 直接输出 7 维 StimulusVector）
- [x] 状态引擎解耦为独立包 `state_engine/`
- [x] 图编排层解耦为 `graph/` 包
- [x] Prompt 数据外置为 `prompts/` 包
- [x] **Defense Profiles 合并**：(3,7) 矩阵 → (2,7) deactivation/hyperactivation，基于 Bowlby 二分法
- [x] **残差式状态更新**：`h_t = h_{t-1} + Δt·(α·Δ_coupling + β·Δ_stimulus + γ·Δ_homeostatic)`
- [x] **稳态恢复内建到动力学**：旧 `apply_decay` 废弃，由 γ·(setpoint−h) 替代
- [x] **表面投影软阈值化**：硬阈值分支改为 sigmoid 连续贡献
- [x] **时间感知衰减组件**：新建 `_decay.py`，以真实时间戳 + 指数衰减 + 人格调制驱动离线状态恢复
- [x] **死代码清理**：移除旧 `_decay.py`、`PERSONALITY_BIAS_C`、`validate_matrices`、decay 常量
- [x] **对抗式双Agent耦合检验**：新建 `tests/test_adversarial_engine.py`（34 用例，185 全通过）
- [x] **大规模异常扫描**：200 组 × 1000 轮 = 20 万轮双Agent耦合，报告写入 `md/STATE_ENGINE_TEST_REPORT.md`

---

## P0 — 阻塞性

- [ ] State Formatter 重写：当前 `_desc()` 将连续状态离散化为 5 级文本描述，破坏 State Engine 连续性。
- [x] 时间驱动衰减：~~引入真实时间戳机制~~ ✅ 已完成。
- [ ] **硬编码参数过多**：State Engine 总计 ~190 个手工数值，详见下方「P1 → 参数管理」。
- [ ] **loneliness / social_battery 稳态崩溃** 🔴：耦合矩阵对角惯性 0.85 + α/γ=3.2x → 真实稳态 ≠ setpoint。loneliness 在 93% 试验中偏离>0.2（全部偏低），social_battery 在 92% 试验中偏离>0.2（全部偏低）。根因：这两个维度在 STATE_COUPLING_A 中无跨维度入边。需提高 γ 或增加耦合入边。
- [ ] **noisy_perception 确定性崩溃** 🔴：tanh 非线性 + DEFAULT_TRAITS + 噪声 → social_battery 从 0.60 跌至 0.08。Top 10 最严重案例中 9/10 为此场景。需在 tanh 模式下增加均值校正或归一化。
- [ ] **α 永远碾压 β** 🔴：α/β 比率恒 ≥ 1.51（均值 2.79×），对话影响被结构性压制。需提高 β_base（0.10→0.15）或降低 α 基线。

## P1 — 重要

- [ ] **soft_clamp 语义决策** 🟡：三个方向待选 — A) 全局 sigmoid 映射（需全引擎重校准）；B) `high + t·tanh(...)` + `np.clip` 兜底（改动最小）；C) 拆成 `soft_clamp`（数值安全网）+ `psychometric_scale`（心理测量接口）两个函数。详见 `memory/soft-clamp-redesign.md`。
- [ ] 感知层注入状态上下文：把 `internal_state` / `relationship_state` 摘要拼入 perception prompt，避免只看到最近 4 条消息。
- [ ] 矩阵权重外部化：详见下方「参数问题详细分析」。
- [ ] 感知上下文扩展：将上下文窗口从 4 条扩展到 12~20 条，必要时结合向量检索。
- [ ] 感知输出值域校验：增加对 `user_stimuli` 结果 ∈ [0,1] 的验证，防止 LLM 越界输出。
- [ ] LLM 输出-状态对齐校验：建立评估闭环，验证回复是否真实反映 `state_description`。
- [ ] Surface 闭环增益 > 1.0：98% 试验中 Surface→Stimuli→Surface 增益 > 1.0（均值 1.14×），信号不衰减反放大。需审查默认耦合矩阵 W 的谱范数。
- [ ] 防御剖面无法极端化：deact/hyper 卡在 [0.34, 0.62]，Bowlby 四种模式均无法调出。需调整 sigmoid 偏移/权重。
- [ ] 往返迟滞：先正向再反向刺激 → 状态被推得更远而非抵消。系统无"抵消"机制。

---

**参数问题详细分析:**

State Engine 共有约 **190 个硬编码数值参数**，分布在 5 个模块:

| 模块 | 参数数 | 类型 |
|------|--------|------|
| `_matrices.py` | 38 | 耦合矩阵非零元、谱归一化阈值 |
| `_defenses.py` | 52 | deact/hyper 基线×特质系数、全局调制、sigmoid 阈值 |
| `_dynamics.py` | 41 | α/β/γ 速率系数、setpoint 偏移 |
| `_surface.py` | 33 | 内部→表面基线系数、刺激贡献、特质修饰 |
| `_decay.py` | 26 | λ_base 值、personality_mod 系数 |

**影响:**

1. **标定脆弱** — 190 个参数通过 inner→outer→dynamics→surface→LLM 的链式传递相互耦合。
2. **无法验证** — 每个系数背后是心理学假设，无实证数据支撑。
3. **单角色过拟合** — 所有参数为月下誓约调校，换角色需重调 120+/190 个参数。
4. **量级不统一** — 系数范围从 0.05 到 0.55（11× 差异）。

**解决路径:**

| 优先级 | 方案 | 效果 | 代价 |
|--------|------|------|------|
| **A (近期)** | 外置为 JSON/YAML 配置 | 换角色只需改配置 | 低 |
| **B (近期)** | 减少参数数量 | 合并冗余维度 | 中 |
| **C (中期)** | 结构约束替代手写 | 低秩映射函数 | 高 |
| **D (远期)** | LLM 驱动参数生成 | 人设自动标定 | 中 |

---

## P2 — 增强

- [ ] 记忆系统：短期情景记忆 + 长期摘要记忆 + 向量检索/关系记忆。
- [ ] 用户心理模型：独立维护用户人格、情绪和偏好，实现 Theory of Mind。
- [ ] 目标/动机系统：建模角色主动目标和行动倾向。
- [ ] 特质演化：让 `traits` 随长程互动缓慢更新。
- [ ] 刺激维度扩展：增加 `anticipation`、`guilt`、`disappointment`、`gratitude` 等。
- [ ] FastAPI 服务化：完善 `main.py`，支持会话管理、多用户隔离。
- [ ] G_LEAKAGE 清理：`GateVector` 与实际门控维度保持一致。
- [ ] test.json 更新：移除已废弃的字段。
- [ ] REL_STATE_COUPLING_A 谱半径：原始 ρ=1.0063 > 0.95，触发归一化。
- [ ] longing / romantic_tension 响应过弱：单轮最大变化仅 0.019 / 0.005。

---

*最后更新：2026-06-17*
