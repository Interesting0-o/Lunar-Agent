# TODO

> Lunar 项目待办事项与已知问题清单。
> 对抗检验报告见 `md/ROADMAP.md`「测试报告」章节，soft_clamp 分析见 memory。
> 双速 SSM 框架设计见 `md/AFFECTIVE_GEOMETRY_RESEARCH.md`「双速 SSM 深度分析」章节，有效自由度分析见下方。

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
- [x] **浪漫张力近乎冻结修复**：B 矩阵加 3 条入边（validation/dependency/emotional_weight→tension）+ 耦合 2 条入边（affection/trust→tension），单轮响应 0.0084→0.0124（+48%）
- [x] **SELF_DECAY 隐性税修复**：统一 0.15→每维度独立数组（0.10-0.12），tax 总值 internal↓22%、relationship↓28%
- [x] **跨尺度耦合缺失修复**：`update_relationship_state` 新增 `current_internal` 参数，5 条内→关耦合规则（stress→trust/safety、loneliness→tension、energy→affection、insecurity→dependency）
- [x] **β 逐维度解耦**：`hyper.mean()` 全局标量→`(7,)` 逐刺激维度向量，保留防御剖面在具体刺激类型上的选择性
- [x] **social_battery 结构性修复**：B 矩阵增 validation→+0.15、closeness→-0.10→+0.08；新增 energy→social_battery 耦合；DECAY_TARGETS[I_SOCIAL_BATTERY]=0.20
- [x] **表面层 stress 去放大**：stress 从 warmth/sharpness/restraint 分散出口，解除 4.5×放大；fatigue 系数 0.30→0.15

---

## P0 — 阻塞性

- [ ] **状态空间维度严重冗余** 🔴：PCA 分析显示 14 维状态空间有效自由度仅 6 维（95% 方差），8 维贡献 < 1% 方差。
  - **数据**：irritation×mental_fatigue r=+0.997，familiarity×romantic_tension r=+0.997，familiarity×dependency r=+0.986，前 2 个主成分解释 73% 方差
  - **根因**：耦合矩阵过密，维度间全协同无拮抗；关系维度 6 维全部正相关同步运动；pride 锁死在 Traits 不是动态状态
  - **影响**：心理表达力 ≈ 6 维而非 14 维；关系维度实际只编码 2-3 件事；longing/insecurity 独立方差 < 10%
  - **方向**：见 `md/AFFECTIVE_GEOMETRY_RESEARCH.md` 双速分解（PAD 正交基底 + 慢速依恋关系）或稀疏化耦合
- [ ] State Formatter 重写：当前 `_desc()` 将连续状态离散化为 5 级文本描述，破坏 State Engine 连续性。
- [x] 时间驱动衰减：~~引入真实时间戳机制~~ ✅ 已完成。
- [ ] **硬编码参数过多**：State Engine 总计 ~190 个手工数值，详见下方「P1 → 参数管理」。

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
- [x] G_LEAKAGE 清理：已移除。
- [x] REL_STATE_COUPLING_A 谱半径：已替换为命名耦合规则 + SELF_DECAY。
- [x] longing / romantic_tension 响应过弱：已修复（浪漫张力 0.0084→0.0124，longing 0.0206→0.0353）

---

*最后更新：2026-06-20*
