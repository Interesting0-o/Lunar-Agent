# TODO

> Lunar 项目待办事项与已知问题清单。
> 详细研究笔记见 `STATE_ENGINE_RESEARCH_REPORT.md` 和 `TODO_LUNAR_STATE_ENGINE.md`。

---

## 已完成 ✅

- [x] 移除 HiddenState 层（已并入 Gate Control + Surface Projection）
- [x] 移除 SocialSignals / InteractionImpact 中间层（perception 直接输出 7 维 StimulusVector）
- [x] Gate Control 动态化（引入 relationship + internal 调制，不再仅依赖恒定 traits）
- [x] 状态引擎解耦为独立包 `state_engine/`（8 个模块）
- [x] 图编排层解耦为 `graph/` 包
- [x] Prompt 数据外置为 `prompts/` 包
- [x] **Trait Modulation + Relationship Modulation + Gate Control 合并为统一 Defense Profiles 层**
  - 门控从 3 个全局标量 → 3×7 逐维度敏感度剖面矩阵
  - suppression / vulnerability / attachment 每维独立作用于对应刺激类型
  - `_gates.py` → `_defenses.py`
- [x] **状态更新改为残差形式** `h_t = h_{t-1} + Δt·(α·Δ_coupling + β·Δ_stimulus + γ·Δ_homeostatic)`
  - 门控控制变化速率（α,β,γ），不控制状态比例
  - 长程对话中旧状态以 1.0 权重保留（不再被遗忘门稀释）
- [x] **稳态恢复内建到动力学**：`apply_decay` 废弃，由 `γ·(setpoint−h)` 替代
  - setpoint 由人格决定（`compute_setpoint(traits)`），不再使用固定 DEFAULT
  - `_decay.py` 不再被 pipeline 引用
- [x] **表面投影软阈值化**：`if traits > 0.6` 硬分支 → sigmoid 连续贡献

---

## P0 — 阻塞性

- [x] ~~残差式动力学更新~~ ✅ 已完成
- [x] ~~Decay baseline 人格化~~ ✅ 已完成（`compute_setpoint(traits)` + 残差动力学内建稳态恢复）
- [x] ~~Decay 修复 — 严格 < 1~~ ✅ 已完成（`apply_decay` 断言 + 废弃独立 decay 层）
- [x] ~~谱归一化~~ ✅ 已完成（A_rel 从 1.0063 → 0.95）
- [x] ~~硬阈值 → 软阈值~~ ✅ 已完成（`_surface.py` 已改，`_decay.py` 共振逻辑已废弃）
- [ ] **State Formatter 重写**：当前 `_desc()` 用 5 级阈值将连续状态重新离散化，破坏 State Engine 的连续性优势。改为连续加权语义投影

## P1 — 重要

- [ ] **感知层注入角色状态上下文**：当前 perception 只有最近 4 条消息，看不到角色当前心理状态。应把 internal/relationship 摘要拼入感知 prompt
- [ ] **矩阵权重外部化**：所有 M_trait、M_rel、A、B、gate 系数硬编码。提取为 JSON/YAML 配置，支持多角色
- [ ] **感知上下文窗口扩展**：4 → 12~20 条消息 + Chroma 向量检索
- [ ] **刺激-特质共振逻辑迁移**：当前 `_decay.py` 中的 if-else 共振（刺激 × 特质 → 恢复放缓）应迁移到动力学 A 矩阵的跨维度耦合项
- [ ] **硬阈值 → 软阈值**：`_decay.py` 和 `_surface.py` 中的 `if x > 0.3` 改为 sigmoid 连续贡献
- [ ] **感知输出值域校验**：当前只检查类型，不检查 ∈ [0,1]，LLM 可能输出越界值
- [ ] **LLM 输出-状态对齐校验**：验证 LLM 回复是否真实反映了 state_description，建立评估闭环
- [ ] **单元测试**：长时间模拟不发散、状态边界、门控范围等不变量测试

## P2 — 增强

- [ ] **记忆系统**：短期情景记忆 + 长期摘要记忆 + Chroma 向量检索
- [ ] **用户心理模型**：独立维护用户人格/情绪/偏好状态，实现 Theory of Mind
- [ ] **目标/动机系统**：角色主动行为倾向（寻求亲密、回避冲突、策划互动），不只被动反应
- [ ] **特质演化**：Traits 在长程互动中缓慢更新（依恋焦虑随安全感建立而降低）
- [ ] **刺激维度扩展**：增加 anticipation、guilt、disappointment 等维度
- [ ] **FastAPI 服务化**：`main.py` 目前是 stub，完善会话管理、多用户隔离
- [ ] **G_LEAKAGE 清理**：GateVector 声明了 4 维但只用 3 维，移除冗余
- [ ] **test.json 更新**：仍包含已废弃的 `user_signals` / `user_interaction_impact` 字段

---

*最后更新：2026-06-15*
