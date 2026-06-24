# TODO

> Lunar 项目待办事项与已知问题清单。
> 更新于 2026-06-23，已整合 06-23 修复批（#1-6 状态引擎设计缺陷）。

---

## 已完成 ✅

- [x] **约束② StimulusMetadata** — 感知节点返回 confidence/source/decay_modulator/timestamp 结构
- [x] **约束⑪ StateFormatter 连续投影** — `_desc()` 替换为 9 区连续投影，消除硬阈值离散
- [x] **decay_modulator 持久化 + 时间衰减接入管线** — `state_engine_node` 先 `apply_time_decay` 再 `update_all`
- [x] **表面惯性时间衰减** — `project_surface` 新增 `delta_hours`，`prev_surface` 向 raw 回归
- [x] **表面负值反馈** — `SURFACE_FEEDBACK_NEG` 矩阵处理压抑/伪装的代谢成本
- [x] **State Formatter 重写**：`_desc()` 连续投影替代 5 级硬阈值 ✅ 06-22
- [x] **参数集中管理**：250+ 参数通过 WeightMapper/WeightVector/LinearMapping 管理，全 provenance

### 06-23 修复批 ✅
- [x] **非对称衰减正负判断** — `deviation < 0` → `(current < 0) & (deviation < 0)`（`_decay.py`）
- [x] **α/α_rel 裁剪边界放宽** — α: [0.02,0.35]→[0.05,0.40]; α_rel: [0.005,0.06]→[0.005,0.08]
- [x] **β_stim 乘法公式** — 加性→乘法 `β = max(ε, BASE+hyper·GAIN) · (1-deact·0.5)`
- [x] **vulnerability 入边增强** — 新增 stress(+0.10) + energy(-0.05) → vulnerability
- [x] **表面→内部反馈因果延迟** — surface[t-1] 影响 internal[t] 而非同轮即时反馈
- [x] **双速 rel_buffer** — 关系态每 3 轮更新一次（`REL_BUFFER_INTERVAL=3`）
- [x] **B 矩阵秩验证** — 新增 `test_matrices.py` 中 `TestMatrixRank`（INPUT_INFLUENCE_B 秩≥6, REL 秩=3）
- [x] **关系级联双向** — 新增 intimacy→affection(+0.03) + intimacy→trust_bond(-0.01)
- [x] **感知层注入状态上下文** — internal/relationship 摘要注入 perception prompt
- [x] **深度分析测试** — 新增 `test_deep_analysis.py`（20 项追踪测试）

---

## 待检验 🔬（已知但未修复）

- [ ] **时间衰减渐近收敛** ⚠️：
  - 理论残余 exp(-λ_base/k) 最大 9.07%（longing 维度），实际影响很小（Δt>72h 已<5%）
  - if 分支在 Δt=168h 处的非平滑仅 0.13% 突变
  - **评估**：影响🟢，待大版本时用 γ<1 公式级修复

- [ ] **状态空间维度冗余（旧）** 🔴：PCA 14维有效自由度仅6维。B矩阵去相关化已完成（密度28.6%），但全局雅可比密度19.9%，仍偏高。

---

## P1 — 逻辑缺陷

- [ ] **表面→内部反馈因果方向（已修复）**
- [ ] **双速动力学是参数慢速而非结构慢速（已修复）**
- [ ] **B 矩阵 skip_rank 缺失验证（已修复）**
- [ ] **关系级联单向（已修复）**
- [ ] **感知层注入状态上下文（已修复）**
- [ ] **α/β 截断抹平 personality 差异（已修复）**
- [ ] **非对称衰减正负判断（已修复）**
- [ ] **β_stim 防负保护（已修复）**
- [ ] **vulnerability 输入不足（已修复）**

## P2 — 增强

- [ ] **权重生成化 Phase 2**：从"文档化硬编码"升级到"生成化权重（Generator 对象，修改原理参数自动重计算）"。
- [ ] **记忆系统集成**：`memory_inject_node` + `memory_summery_node` 接入 `graph/_builder.py`。
- [ ] **特质演化**：让 `traits` 随长程互动缓慢更新。
- [ ] **刺激维度扩展**：增加 `anticipation`、`guilt`、`disappointment`。
- [ ] **UserModel / Theory of Mind**：独立用户心理剖面。
- [ ] **FastAPI 服务化**：完善 `main.py`。
- [ ] **双速缓冲区（已实现）**

---

## 已归档（见 md/ROADMAP.md）

- 权重外部化 Phase 1 → 已完成（WeightMapper 替代裸数值）
- 内部驱力系统 → 见 `md/INTERNAL_DRIVE_SYSTEM.md`
