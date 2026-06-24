# Lunar 状态引擎路线图、执行计划与测试报告

> 2026-06-23 | 06-23 修复批完成（反馈延迟 / rel_buffer / 级联双向 / 感知状态上下文 / 深度分析测试）

---

## 一、问题清单

### 🔴 严重问题（影响核心功能）

#### 🔴 问题 1：特质永远不变，角色"长不大"

- 问题：`T_SENSITIVITY / T_ATTACHMENT_ANXIETY` 等 10 维特质在整个生命周期内不更新
- 后果：玩 100 轮和 1 轮性格没差别
- 学术参考：Bowlby IWM 修正、McAdams 三层人格（特质 → 适应 → 叙事）
- **状态**：待方案

#### 🔴 问题 2：刺激向量只有 7 类信号，关键情绪类别缺失

- 现有 7 维偏重"关系性"刺激，缺少 `ST_ANTICIPATION`、`ST_GUILT`、`ST_DISAPPOINTMENT`、`ST_GRATITUDE`、`ST_CURIOSITY`
- 缺少 Appraisal 层（goal_congruence, certainty, agency, coping_potential）
- 学术参考：Plutchik 情感轮盘、Ekman 6 基本情绪、Scherer CPM
- **状态**：待方案

#### 🔴 问题 3：记忆系统尚未完全集成

- MemoryStore + 三路检索 + MemoryNode 已实现，`memory_inject_node` 和 `memory_summery_node` 在 `nodes.py` 中有完整逻辑（LLM 调用 + JSON 解析 + 持久化），但**未注册到 `graph/_builder.py`**，流水线不在活跃状态
- **状态**：部分实现，待集成

#### 🔴 问题 4：无"目标/意图"系统，角色"无欲无求"

- 角色完全被动响应，没有任何"想做某事"的内部驱动
- 学术参考：BDI 模型、Schema Theory、SDT（autonomy/competence/relatedness）
- **状态**：待方案

#### 🔴 问题 5：状态空间维度严重冗余，表达力不足

- **PCA 实证**：14 维状态空间有效自由度仅 6 维（95% 方差），前 2 个主成分解释 73% 方差
- **最大冗余**：familiarity×romantic_tension r=+0.997，irritation×mental_fatigue r=+0.997
- **根因**：耦合矩阵过密导致维度全协同无拮抗；关系维度 6 维全部正相关同步运动
- **影响**：心理表达力 ≈ 6 维而非 14 维
- **方向**：语义合并关系维度 6→3（已完成），B 矩阵去相关化（已完成，B_int 28.6% + B_rel 28.6%），双速 SSM/PAD 正交基底（待远期）
- **状态**：B 矩阵已修复，全局雅可比密度从 23.9% 降至 19.9%

### 🟡 中等问题（影响真实感）

#### 🟡 问题 6：表面表达只有 7 维，无法表达细腻情感
- 7 维过于粗粒度——"温柔地笑"和"害羞地笑"无法区分
- **状态**：待方案

#### 🟡 问题 7：无"动作/行为"系统，角色"只说不做"
- 角色只能说话，无法模拟主动行为
- **状态**：待方案

#### 🟡 问题 8：无"内部独白"机制，角色"无意识流"
- 两次对话之间完全静止，没有反思
- 学术参考：Vygotsky Inner Speech、Default Mode Network
- **状态**：待方案

#### 🟡 问题 9：对"用户"无模型，角色不知道"你是谁"
- 无 UserModel，每次对话都是"陌生人的第一次见面"
- 学术参考：Bowlby IWM（对他人表征）、Theory of Mind
- **状态**：待方案

#### 🟡 问题 10：表面→内部反馈的因果方向是瞬时的（设计局限）✅ 已修复

- 2026-06-23 修复：将 feedback 从同轮即时改为 `surface[t-1] → internal[t]`（延迟版）
- `_pipeline.py` 中 feedback 计算移至动力学之前，使用 prev_surface 而非当前 surface
- **状态**：✅ 已修复

#### 🟡 问题 11：双速动力学是参数慢速而非结构慢速 ✅ 已修复

- 2026-06-23 修复：关系态增加显式低通滤波器（`rel_buffer`）
- `_pipeline.py` 新增 `REL_BUFFER_INTERVAL=3`：关系态每 3 轮更新一次，中间轮次冻结
- `State` TypedDict 新增 `rel_update_counter` 字段持久化计数
- **状态**：✅ 已修复

#### 🟡 问题 12：B 矩阵跳过秩检查（skip_rank=True） ✅ 已修复

- 2026-06-23 修复：`tests/test_matrices.py` 新增 `TestMatrixRank`
- `INPUT_INFLUENCE_B` 非零秩验证 ≥ 6，`REL_INPUT_INFLUENCE_B` 满秩 = 3
- **状态**：✅ 已修复

#### 🟡 问题 13：关系级联单向（intimacy 成"死胡同"） ✅ 已修复

- 2026-06-23 修复：`_dynamics_weights.py` 新增 intimacy→affection(+0.03) + intimacy→trust_bond(-0.01)
- intimacy 拓扑从级联终点变为有反馈回路的双向耦合
- **状态**：✅ 已修复

#### 🟡 问题 14：感知层无状态上下文 ✅ 已修复

- 2026-06-23 修复：`perception.py` 注入 internal/relationship 状态摘要到 prompt
- `nodes.py` 中 `perception_node` 传递当前状态给感知调用
- **状态**：✅ 已修复

#### 🟡 问题 15：β_stim 负值 + 截断丢失调制 ✅ 已修复

- 2026-06-23 修复：加性→乘法公式 `β = max(ε, BASE+hyper·GAIN) · (1-deact·0.5)`
- avoidant 人格的 β 从固定 0.01 恢复为保留 60% 的抑制比例
- **状态**：✅ 已修复

#### 🟡 问题 16：非对称衰减正负判断错误 ✅ 已修复

- 2026-06-23 修复：`deviation < 0` → `(current < 0) & (deviation < 0)`（双条件）
- energy/social_battery 等正向维度不再被误加速衰减
- **状态**：✅ 已修复

#### 🟡 问题 17：vulnerability 输入不足 ✅ 已修复

- 2026-06-23 修复：新增 stress→vulnerability(+0.10) + energy→vulnerability(-0.05)
- 总输入权重从 0.65 → 0.80
- **状态**：✅ 已修复

#### 🟡 问题 18：α/α_rel 裁剪边界过紧 ✅ 已修复

- 2026-06-23 修复：α 上界 0.35→0.40, α_rel 上界 0.06→0.08
- 总截断率从 29% 降至 ~15%（随机人格分布下）
- **状态**：✅ 已修复

### 🟢 锦上添花（影响沉浸感）

| 问题 | 内容 | 状态 |
|------|------|------|
| 🟢 问题 19 | 无"昼夜节律" | 待方案 |
| 🟢 问题 20 | 无"个体微习惯" | 待方案 |
| 🟢 问题 21 | 防御维度待扩展（当前仅 2 维，Vaillant 有 10+ 种） | 方法论文档已写，待扩展 |
| 🟢 问题 22 | 无"主观时间感" | 待方案 |

---

## 二、执行计划

### 当前状态与剩余工作

> 截至 2026-06-23，核心数学基础设施 + 状态引擎设计缺陷修复已完成：
> - ✅ 关系维度 6→3 语义合并（方案 A）
> - ✅ 防御剖面多维输入源（方案 B，实际合并了原多维调制而非独立新变量）
> - ✅ β 调制系数释放（方案 C）— β 有效范围 [0.005, 0.35]，乘法公式
> - ✅ 约束框架完整实现（方案 G）— WeightMapper + WeightVector + LinearMapping + ConstraintRegistry
> - ✅ 250+ 参数全量迁移（无 origin=legacy 参数）
> - ✅ 表面反馈因果延迟（#1）
> - ✅ 双速 rel_buffer（#2）
> - ✅ B 矩阵秩验证（#3）
> - ✅ 关系级联双向（#4）
> - ✅ 感知状态上下文（#5）
> - ✅ 深度分析测试（20 项）
> - ✅ α/β 修正 + β_stim 乘法公式 + vulnerability 入边

| 编码 | 方案 | 等级 | 工期 | 依赖 | 状态 |
|:---:|------|:---:|:----:|:----:|:----:|
| **H** | 记忆系统集成 | **P1** | 1d | 无 | ⏳ 代码已写，需接入 graph |
| **D** | OCC 拮抗对/刺激扩展 | P1 | 3d | 无 | ❌ 待方案 |
| **E** | Trait 演化 v1 | **P2** | 5d | D（推荐） | ❌ 待方案 |
| **F** | 双速 SSM v1 | **P2** | 2w | E | ❌ 待方案 |

### P1（短期优先 — 剩余项）

| 方案 | 目标 | 预计改动 | 工期 |
|------|------|---------|:----:|
| **H：记忆集成** | `memory_inject_node` + `memory_summery_node` 接入 `graph/_builder.py` | 只需改 `graph/_builder.py`，节点函数已就绪 | **1d** |
| **D：OCC/刺激扩展** | 不修改 ST_SIZE，在 perception.py 添加评价后处理或扩展维数 | `perception.py`, `config.py` | 3d |

### 已完成的 P1 项目（06-23）
- ✅ **表面反馈因果延迟** — `_pipeline.py` 调用顺序调整
- ✅ **双速 rel_buffer** — `state.py` + `_pipeline.py` + `nodes.py`
- ✅ **B 矩阵秩验证** — `test_matrices.py`
- ✅ **关系级联双向** — `_dynamics_weights.py`
- ✅ **感知状态上下文** — `perception.py` + `nodes.py`
- ✅ **α/α_rel 边界放宽** — `_dynamics.py`
- ✅ **β_stim 乘法公式** — `_dynamics.py`
- ✅ **非对称衰减修复** — `_decay.py`
- ✅ **vulnerability 入边增强** — `_surface_weights.py`

### P2（中期）

#### 方案 E：Trait 演化 v1

使 10 维 trait 在不超出对数漂移边界的前提下，随交互结果缓慢更新。基于 SALM 对数收敛定理（0.08log(k)+0.12），每 5 轮触发一次更新，置信度从 0.3 开始渐进增长。

```python
# 核心增量规则（示例草案）
delta[T_EMOTIONAL_STABILITY] -= internal[I_STRESS] * 0.005  # 长期压力→稳定性↓
delta[T_ATTACHMENT_AVOIDANCE] += (0.5 - relationship[R_TRUST_BOND]) * 0.01  # 低信任→回避↑
delta[T_PRIDE] += (deact_avg - 0.5) * 0.01  # 持续高去激活→骄傲↑
```

#### 方案 F：双速 SSM v1

将扁平动力学分解为快速层（PAD, 3 维, dt=1）和慢速层（关系态, 3 维, 累积更新）。详细设计见 `AFFECTIVE_GEOMETRY_RESEARCH.md`。

### P3（远期方向）

1. 非线性动力学 / PLRNN — 添加 W·φ(z) 非线性项以支持相变和多稳态
2. UserModel / UserProfile — 用户心理剖面
3. ~~State Formatter 连续化~~ ✅ **已完成（06-22）**
4. 权重生成化 Phase 2 — 从"文档化权重（WeightMapper.connect value 硬编码）"升级到"生成化权重（支持 Generator 对象，修改原理参数自动重计算）"
5. ~~延迟表面反馈~~ ✅ **已完成（06-23）**
6. ~~双速缓冲区~~ ✅ **已完成（06-23）**
7. FastAPI 服务化
8. LLM-as-a-Judge 评估闭环

### 验收标准

| 方案 | 验收标准 | 验证方法 |
|------|---------|---------|
| **A** | 有效自由度 14→≥7，PC1 < 40%（关系态） | `test_anomalies.py` PCA |
| **B** | 独立方差 deact ≥ 5%, hyper ≥ 3% | `tools/audit_defenses.py` |
| **C** | β 有效范围 ≥ [0.01, 0.20] | `test_dynamics.py` |
| **D** | OCC 映射不破坏下游 shape | `test_pipeline.py` |
| **E** | 4000 轮漂移 ≤ 0.12，稳定性 ≥ 0.87 | 新测试 `test_trait_evolution.py` |
| **F** | 快速层 τ=2-5，慢速层 τ=50-200 | `test_dynamics.py` |

### 依赖与并行性

```
周 1                   周 2                   周 3                   周 4
│                      │                      │                      │
├── A (语义合并) ──────┤                      │                      │
│         └── 降低 B 矩阵冗余 ─────────── F 的前提                │
│                      │                      │                      │
├── B (防御输入) ───┤   （独立完成）         │                      │
│                      │                      │                      │
├────────────────── C (β 释放) ────┤         │                      │
├──────────── D (OCC) ────────────┤          │                      │
│         └── 提供评价信号 ─────────── E 的信号源                   │
├────────────────── G (约束框架) ──┤         │                      │
├────────────── H (记忆) ───┤                │                      │
│                      │                      │                      │
│                      ├──────── E (Trait 演化) ────────┤          │
│                      │         └── 解除静止基线 ────── F 的前提  │
│                      │                      │                      │
│                      │                      ├── F (双速 SSM) ───│
│                      │                      │  依赖于 A + E      │
```

**关键路径：** A → F | D → E → F
**并行分支（完全独立）：** B, C, G, H

---

## 三、测试报告（当前快照）

> 2026-06-23 | 264 测试用例全部通过（06-23 修复批：反馈延迟 / rel_buffer / 级联双向 / 感知状态 / 深度分析）

### 测试概览

| 指标 | 数值 |
|------|------|
| 测试文件 | 10 个模块（+1 新增 `test_deep_analysis.py`） |
| 测试用例 | **264**（含 Surface 惯性/反馈测试 22 项 + 深度分析追踪测试 20 项 + 34 个对抗式双Agent耦合检验 + 27 个时间衰减监督测试） |
| 通过率 | 100% (264/264) |
| 执行时间 | 173s |
| 最大单测数据量 | **500,000 组** Monte Carlo |
| 对抗检验轮数 | 100 组 × 500 轮 + 5 场景 × 1000-1500 轮独立报告 |
| 总模拟步数 | 约 **200 万** 次管线调用 |
| 异常发现 | **12 个**（本版新增维度冗余 P0 问题） |

### 本版架构变更（06-23 修复批）

1. **表面→内部反馈因果延迟**：同轮即时 → `surface[t-1] → internal[t]`，先反馈再动力学
2. **双速 rel_buffer**：关系态每 3 轮更新一次（`REL_BUFFER_INTERVAL=3`），中间轮次冻结
3. **关系级联双向**：intimacy 新增→affection(+0.03) + →trust_bond(-0.01) 反馈回路
4. **感知状态上下文**：internal/relationship 状态摘要注入 perception prompt
5. **B 矩阵秩验证**：新增 TestMatrixRank 测试（INPUT_INFLUENCE_B 秩≥6, REL 满秩=3）
6. **非对称衰减**：`deviation < 0` → `(current < 0) & (deviation < 0)` 双条件
7. **β_stim 乘法公式**：加性 → 乘法 `β = max(ε, BASE+hyper·GAIN) · (1-deact·0.5)`
8. **α/α_rel 边界放宽**：α [0.02,0.35]→[0.05,0.40], α_rel [0.005,0.06]→[0.005,0.08]
9. **vulnerability 入边增强**：新增 stress(+0.10) + energy(-0.05)

### 异常值清单

| 指标 | 之前 | 现在 | 变化 |
|------|:----:|:----:|:----:|
| `longing` 最大变化 | 0.0206 | 0.0353 | +71% |
| `romantic_tension` 最大变化 | 0.0084 | 0.0109 | +30% |
| `stress` 最大变化 | 0.0987 | 0.1038 | 🟢 敏感度保持 |
| `loneliness` 最大变化 | ~0.06 | 0.1077 | +80% |
| `social_battery` 最大变化 | 0.0710 | 0.0703 | 🟢 双向可调 |

### PCA 维度冗余实证（核心发现）

**14 维名义空间，仅 6 维有效自由度。**

| 指标 | 数值 |
|------|:----:|
| 名义维度 | 14（内部 8 + 关系 6） |
| 有效维度 (95% 方差) | **6** |
| 有效维度 (99% 方差) | 9 |
| 前 2 个 PC 解释方差 | **73%** |
| 最大维度间相关 | **r=+0.997** |

**高度冗余的维度对：** familiarity×romantic_tension r=+0.997, irritation×mental_fatigue r=+0.997, dependency×romantic_tension r=+0.992

**独立方差最低的维度：** insecurity (0.094), longing (0.101), energy (0.163)

**根因：** 耦合矩阵过密 + 关系维度全正相关同步 + 缺少拮抗对 + pride 锁死在 Traits 中。

### 防御剖面表现

| 指标 | 修复前 | 修复后 | 判定 |
|------|:-----:|:-----:|:----:|
| deact 均值范围 | [0.306, 0.602] | **[0.053, 0.858]** | 🟢 |
| hyper 均值范围 | [0.320, 0.575] | **[0.058, 0.800]** | 🟢 |
| deact×hyper 相关系数 | r=-0.240 | r=-0.230 | 🟢 独立性保留 |

### 不变式清单

| 不变式 | 状态 |
|--------|:----:|
| 所有状态向量 ∈ soft_clamp 范围 | ✅ |
| sigmoid 单调性 | ✅ |
| profiles 范围 [0, 1] | ✅ |
| β 有效变动 > 0.10 | ✅ 实际 0.289 |
| setpoint ∈ [-0.9, 0.9] | ✅ |
| 零刺激收敛到耦合平衡 | ✅ |
| 长程不发散 (500-10,000 轮) | ✅ |
| 防御剖面独立性 (|r|=0.23 < 0.3) | ✅ |

### 当前问题优先级

| 优先级 | 数量 | 问题 |
|:------:|:----:|------|
| 🔴 P0 | 1 | 维度冗余：14 维仅 6 维有效自由度 |
| 🟡 P1 | 1 | 时间衰减渐近残余（影响：🟢 实际 <5%） |
| 🟢 待实施 | 5 | 记忆集成 / OCC 刺激扩展 / Trait 演化 / 权重生成化 / FastAPI |
| ℹ️ 文档化 | 2 | soft_clamp 过渡区，sigmoid 浮点饱和 |

### 运行测试

```bash
# 全部测试（264 用例）
uv run pytest tests/ -v

# 深度分析追踪测试（20 项，带 -s 查看追踪数据）
uv run pytest tests/test_deep_analysis.py -v -s

# 异常探测（含维度分析）
uv run pytest tests/test_anomalies.py -v -s

# 时间衰减（27 用例，不含可视化）
uv run pytest tests/test_decay.py -v -k "not TestVisualization"
```

---

## 四、问题溯源汇总

| 问题 | 关键词 | 推荐文献 |
|------|--------|---------|
| 1 | 特质演化 | Bowlby 依恋理论, McAdams 人生叙事, Whole Trait Theory |
| 2 | 情绪分类 + Appraisal | Plutchik, Ekman, Scherer CPM, OCC 模型 |
| 3 | 记忆集成 | 当前 `nodes.py` 节点代码完整但未注册到 `graph/_builder.py`，设计见 `MEMORY_SYSTEM.md` |
| 4 | 行为驱动 | BDI 模型, SDT, Goal-Directed Behavior |
| 5 | 维度冗余 | `SPARSE_ANTAGONIST_ANALYSIS.md`, `AFFECTIVE_GEOMETRY_RESEARCH.md` |
| 6 | 情绪表达 | FACS, Russell Circumplex Model |
| 7 | 主动行为 | Embodied AI, Action Systems |
| 8 | 内部语言 | Inner Speech, Default Mode Network |
| 9 | 用户建模 | User Modeling, Theory of Mind, IWM |
| 12 | 防御扩展 | Anna Freud, Vaillant, 见 `DEFENSE_PROFILE_METHODOLOGY.md` |

---

## 五、开源生态可借鉴部分

| Lunar 需求 | 可参考的开源项目 | 可复用的设计 |
|-----------|----------------|------------|
| 情感正交维度 | **Bhava** (Rust) | PAD 3 维 + OCC 标签投影的设计模式 |
| Trait 演化 | **Relic** | 置信度追踪 + 观察计数的渐进更新 |
| 记忆集成 | **Soul Protocol** | 5 级记忆 + 躯体标记的设计 |
| 约束框架 | **soulcuit v2** | 约束图 + 心理学关系的手工构建 |
| 连续状态机 | **Soul Engine** | MentalProcesses + WorkingMemory 模式 |
| 用户建模 | **Relic** | 纵向建模 + 分离证据流 + 置信度 |
| 情感衰减 | **Sentimo** | 人格权重的指数平滑衰减 |

---

## 六、已完成项目（归档）

| 项目 | 完成时间 |
|------|---------|
| 刺激+耦合驱动残差更新（γ 移除） | 06-18 |
| A 矩阵 → 10 条命名耦合规则 + SELF_DECAY | 06-18 |
| 防御剖面 sigmoid 缩放 + β 调制修复 | 06-18 |
| 时间感知衰减 `_decay.py` | 06-18 |
| 非对称衰减 FAB (negative_decay_boost=1.8) | 06-19 |
| soft_clamp 替代 np.clip | 已修复 |
| G_LEAKAGE 死代码清理 | 已修复 |
| 浪漫张力 B 矩阵 + 耦合入边加固 | 06-20 |
| SELF_DECAY 每维度独立数组 | 06-20 |
| REL_SELF_DECAY 每维度独立数组 | 06-20 |
| 跨尺度耦合（内→关） | 06-20 |
| PCA 维度冗余实证 | 06-20 |
| Surface 惯性混合 + 反馈 + traits 间接化 | 06-22 |
| 230 测试用例全通过 | 06-22 |
| 关系维度 6→3 语义合并 | 06-21 |
| B_int 去相关化 + 稀疏化（44.6% → 28.6%，约束⑥合规） | 06-21 |
| B_rel 去相关化 + 稀疏化（38.1% → 28.6%，全正交签名） | 06-21 |
| WeightMapper 骨架（约束⑤ — 语义映射层） | 06-21 |
| WeightVector 骨架（12 组防御权重 provenance） | 06-21 |
| ConstraintRegistry 骨架（约束⑧ — 参数审计） | 06-21 |
| JSON 权重外部化导入/导出 | 06-21 |
| 全局雅可比密度从 23.9% → 19.9% | 06-21 |
| **LinearMapping 类 + BiasWeight（表面/速率/setpoint 迁移基础）** | **06-21** |
| **表面投影 35 条线性系数 + 7 偏置 → SURFACE_MAPPER** | **06-21** |
| **动力学 α/β/setpoint/耦合/自阻尼 全部迁移** | **06-21** |
| **防御剖面 deact/hyper 基线 + sigmoid + apply 增益迁移** | **06-21** |
| **衰减 λ/时间曲线/人格调制 全部迁移** | **06-21** |

### 06-23 修复批 ✅

| 项目 | 文件 | 类型 |
|------|------|:----:|
| 非对称衰减正负判断 | `_decay.py` | 公式修复 |
| α/α_rel 边界放宽 | `_dynamics.py` | 参数修正 |
| β_stim 乘法公式 | `_dynamics.py` | 公式重构 |
| vulnerability 入边增强 | `_surface_weights.py` | 参数修正 |
| 表面→内部反馈因果延迟 | `_pipeline.py` | 结构重排 |
| 双速 rel_buffer | `_pipeline.py` + `nodes.py` + `state.py` | 新功能 |
| B 矩阵秩验证 | `test_matrices.py` | 测试增强 |
| 关系级联双向 | `_dynamics_weights.py` | 参数修正 |
| 感知状态上下文 | `perception.py` + `nodes.py` | 新功能 |
| 深度分析测试 | `test_deep_analysis.py` | 新测试模块 |
