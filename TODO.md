# TODO

> Lunar 项目待办事项与已知问题清单。
> 详细研究笔记见 `STATE_ENGINE_RESEARCH_REPORT.md` 和 `TODO_LUNAR_STATE_ENGINE.md`。

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

---

## P0 — 阻塞性

- [ ] State Formatter 重写：当前 `_desc()` 将连续状态离散化为 5 级文本描述，破坏 State Engine 连续性。
- [x] 时间驱动衰减：~~引入真实时间戳机制~~ ✅ 已完成 — `_decay.py` 已实现混合指数衰减 + 人格调制。
- [ ] **硬编码参数过多**：State Engine 总计 ~190 个手工数值，详见下方「P1 → 参数管理」。

## P1 — 重要

- [ ] 感知层注入状态上下文：把 `internal_state` / `relationship_state` 摘要拼入 perception prompt，避免只看到最近 4 条消息。
- [ ] 矩阵权重外部化：详见下方「参数问题详细分析」。
- [ ] 感知上下文扩展：将上下文窗口从 4 条扩展到 12~20 条，必要时结合向量检索。
- [ ] 感知输出值域校验：增加对 `user_stimuli` 结果 ∈ [0,1] 的验证，防止 LLM 越界输出。
- [ ] LLM 输出-状态对齐校验：建立评估闭环，验证回复是否真实反映 `state_description`。
- [ ] 单元测试：补齐长期模拟、状态边界、门控范围、矩阵稳定性等不变量测试。

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

1. **标定脆弱** — 190 个参数通过 inner→outer→dynamics→surface→LLM 的链式传递相互耦合。改 `_defenses.py` 的 pride 系数会间接改变 `_surface.py` 的表面温度，中间经过 3 层非线性变换（sigmoid + soft_clamp + 矩阵乘法），无法预测最终效果。

2. **无法验证** — 每个系数背后是一个心理学假设（如"高自尊对被抛弃的去激活系数 = 0.45 / 单位偏离"），但没有任何实证数据支撑。系统行为建立在 190 个"合理猜测"之上。

3. **单角色过拟合** — 所有参数为月下誓约（高自尊、依恋焦虑、吸血鬼）调校。换角色约需重调 120+/190 个参数，几乎等于重新构建。

4. **量级不统一** — 系数范围从 0.05 到 0.55（11× 差异），没有一致的设计规范。每个模块独立选择量级，跨模块交互难以预测。

**解决路径:**

| 优先级 | 方案 | 效果 | 代价 |
|--------|------|------|------|
| **A (近期)** | 外置为 JSON/YAML 配置 | 换角色只需改配置；参数可独立审查和版本管理 | 低 — 提取 `_matrices.py`/`_defenses.py`/`_decay.py` 的常量段 |
| **B (近期)** | 减少参数数量 | 合并冗余维度，如 (3,7)→(2,7) 已做的那样 | 中 — 逐模块审查可合并的系数 |
| **C (中期)** | 结构约束替代手写 | 用低秩映射函数 `σ(W@t_dev + b)` 替代逐维手写系数；W 可以显式设计或学习得到 | 高 — 需重新设计 defense/dynamics/surface 的参数生成逻辑 |
| **D (远期)** | LLM 驱动参数生成 | 给 LLM 角色描述 → LLM 输出 JSON 配置；用角色人设自动标定所有参数 | 中 — 需设计验证 prompt 和一致性检查 |

**A 是明确的下一步** — 将 `_matrices.py` 的 A/B 矩阵、`_defenses.py` 的剖面系数、`_decay.py` 的 λ_base 提取为 JSON 配置文件，配合 JSON Schema 验证。不减少参数数量，但使多角色支持成为可能。

## P2 — 增强

- [ ] 记忆系统：短期情景记忆 + 长期摘要记忆 + 向量检索/关系记忆。
- [ ] 用户心理模型：独立维护用户人格、情绪和偏好，实现 Theory of Mind。
- [ ] 目标/动机系统：建模角色主动目标和行动倾向，而不是纯被动反应。
- [ ] 特质演化：让 `traits` 随长程互动缓慢更新，支持依恋/愤怒/失望等长期变化。
- [ ] 刺激维度扩展：增加 `anticipation`、`guilt`、`disappointment`、`gratitude` 等评价性刺激。
- [ ] FastAPI 服务化：完善 `main.py`，支持会话管理、多用户隔离。
- [ ] G_LEAKAGE 清理：`GateVector` 与实际门控维度保持一致，移除冗余项。
- [ ] test.json 更新：移除已废弃的 `user_signals` / `user_interaction_impact` 字段。

---

*最后更新：2026-06-16*
