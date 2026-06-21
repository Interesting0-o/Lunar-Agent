# Lunar 状态引擎路线图、执行计划与测试报告

> 2026-06-21 | 整合 ROADMAP.md（问题清单）+ EXECUTION_PLAN.md（排期计划）+ STATE_ENGINE_TEST_REPORT.md（测试快照）
>
> 已完成的架构修复（LSTM 三门控 → 残差动力学、decay>1 → 时间衰减、A 矩阵 → 命名规则、β 调制修复、SELF_DECAY 每维独立数组、跨尺度耦合）不再列于问题清单中，但保留在已完成记录中。

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

- MemoryStore + 三路检索 + MemoryNode 已实现，但 `memory_inject_node` 和 `memory_summery_node` 在 `nodes.py` 中为 stub
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
- **方向**：语义合并关系维度 6→3，或双速 SSM/PAD 正交基底
- **状态**：待方案（已制定 P0 执行计划）

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

### 🟢 锦上添花（影响沉浸感）

| 问题 | 内容 | 状态 |
|------|------|------|
| 🟢 问题 10 | 无"昼夜节律" | 待方案 |
| 🟢 问题 11 | 无"个体微习惯" | 待方案 |
| 🟢 问题 12 | 防御维度待扩展（当前仅 2 维，Vaillant 有 10+ 种） | 方法论文档已写，待扩展 |
| 🟢 问题 13 | 无"主观时间感" | 待方案 |

---

## 二、执行计划

### 总体排期

```
周 1        周 2        周 3        周 4
├P0-A──────┤
├P0-B──┤
            ├P1-C──────┤
            ├P1-D──────────┤
            ├P1-G──────┤
            ├P1-H──┤
                        ├P2-E──────────┤
                        ├P2-F──────────┤
                                       ├P3-远期───
```

| 编码 | 方案 | 等级 | 工期 | 依赖 | 排期 |
|:---:|------|:---:|:----:|:----:|:----:|
| **A** | 语义合并关系维度 6→3 | **P0** | 1d | 无 | 本周 |
| **B** | 增加防御剖面输入源 | **P0** | 1d | 无 | 本周 |
| **C** | β 调制系数释放 | P1 | 0.5d | A（推荐） | 下周 |
| **D** | OCC 拮抗对扩展刺激空间 | P1 | 3d | 无 | 下周 |
| **G** | 约束框架最低实现 | P1 | 3d | 无 | 下周 |
| **H** | 记忆系统集成 | P1 | 2d | 无 | 下周 |
| **E** | Trait 演化 v1 | **P2** | 5d | D | 周 3 |
| **F** | 双速 SSM v1 | **P2** | 2w | A, E | 周 3-4 |

### P0（本周）：低风险高回报

#### 方案 A：语义合并关系维度 6→3

**目标：** 将关系态从 6 维合并为 3 维，将有效自由度从 ~5 提升至 ~7-8。

**映射规则：**

| 当前 6 维 | 合并后 3 维 | 心理学框架 |
|-----------|-----------|-----------|
| `R_AFFECTION` + `R_INTIMACY` + `R_TRUST_BOND` | **R_BOND**（情感纽带） | Sternberg 亲密三成分 |
| `R_EMOTIONAL_SAFETY` + `R_FAMILIARITY` | **R_SAFETY**（安全感） | Bowlby 安全基地 |
| `R_DEPENDENCY` + `R_ROMANTIC_TENSION` | **R_INVOLVEMENT**（卷入度） | Berscheid 暧昧张力 |

**待修改文件顺序：** `state.py` → `_matrices.py` → `_dynamics.py` → `_surface.py` → `_defenses.py` → `_decay.py` → `tests/`

**核心耦合矩阵草案（3×3）：**

```
         R_BOND   R_SAFETY   R_INVOLVEMENT
R_BOND    [0.0      0.08       0.05]
R_SAFETY   [0.06     0.0        0.0]
R_INVOLVEMENT [0.0   -0.03      0.0]
```

3×3 矩阵允许自然出现拮抗对（involvement 负向影响 safety），当前 6 维做不到。

#### 方案 B：增加防御剖面输入源

**目标：** 将去激活的输入源从 3→5，过度激活从 4→6，使独立方差从 ~0% → 5-10%。

**待修改：** 仅 `_defenses.py` 一个文件。

新增权重数组：
- 去激活新增：`INTIMACY_DEACT_A`、`LONELINESS_DEACT_A`、`ENERGY_DEACT_A`
- 过度激活新增：`STRESS_HYPER_A`、`FATIGUE_HYPER_A`

验证方法：
```bash
uv run python tools/audit_defenses.py
# 预期: 去激活独立方差: ~0% → ~5-8%
#       过度激活独立方差: ~0% → ~3-6%
```

### P1（下周）：中风险中回报

| 方案 | 目标 | 改动文件 | 工期 |
|------|------|---------|:----:|
| **C：β 释放** | β 有效范围 [0.02,0.08] → [0.01,0.20] | `_dynamics.py` | 0.5d |
| **D：OCC 拮抗对** | 不修改 ST_SIZE，在 perception.py 添加评价后处理 | `perception.py`, `config.py` | 3d |
| **G：约束框架** | 实现 WeightMapper + 4 条关键约束自动检查 | 新建 `_validator.py` | 3d |
| **H：记忆集成** | 完成 nodes.py 两个 stub | `nodes.py`, `_builder.py`, `state.py` | 2d |

### P2（周 3-4）：高风险高回报

#### 方案 E：Trait 演化 v1

使 10 维 trait 在不超出对数漂移边界的前提下，随交互结果缓慢更新。基于 SALM 对数收敛定理（0.08log(k)+0.12），每 5 轮触发一次更新，置信度从 0.3 开始渐进增长。

```python
# 核心增量规则
delta[T_EMOTIONAL_STABILITY] -= internal[I_STRESS] * 0.005  # 长期压力→稳定性↓
delta[T_ATTACHMENT_AVOIDANCE] += (0.5 - relationship[R_BOND]) * 0.01  # 低信任→回避↑
delta[T_PRIDE] += (deact_avg - 0.5) * 0.01  # 持续高去激活→骄傲↑
```

#### 方案 F：双速 SSM v1

将扁平动力学分解为快速层（PAD, 3 维, dt=1）和慢速层（关系态, 6/3 维, 累积更新）。详细设计见 `AFFECTIVE_GEOMETRY_RESEARCH.md`。

### P3（远期方向）

1. 非线性动力学 / PLRNN — 添加 W·φ(z) 非线性项以支持相变和多稳态
2. UserModel / UserProfile — 用户心理剖面
3. State Formatter 连续化 — 替代当前 5 级离散 `_desc()`
4. 权重外部化 — 所有矩阵参数移至 JSON/YAML 配置
5. FastAPI 服务化
6. LLM-as-a-Judge 评估闭环

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

> 2026-06-20 | 192 用例全部通过 | 155s | 四项结构性修复

### 测试概览

| 指标 | 数值 |
|------|------|
| 测试文件 | 9 个模块 |
| 测试用例 | **192**（含 34 个对抗式双Agent耦合检验 + 27 个时间衰减监督测试） |
| 通过率 | 100% (192/192) |
| 执行时间 | 155s |
| 最大单测数据量 | **500,000 组** Monte Carlo |
| 对抗检验轮数 | 100 组 × 500 轮 + 5 场景 × 1000-1500 轮独立报告 |
| 总模拟步数 | 约 **200 万** 次管线调用 |
| 异常发现 | **12 个**（本版新增维度冗余 P0 问题） |

### 本版架构变更

1. **浪漫张力输入加固**：B 矩阵新增 3 条心理学依据的刺激→张力入边，单轮响应 +48%
2. **SELF_DECAY 隐性税修复**：统一 0.15 → 每维度独立数组，tax 总值 internal↓22%、relationship↓28%
3. **跨尺度耦合（内→关）**：`update_relationship_state` 新增 5 条跨尺度耦合规则
4. **β 逐维度解耦**：`hyper.mean()` 全局标量 → (7,) 逐刺激维度向量，β 有效变动 0.042→0.289
5. **social_battery 结构性修复**：B 矩阵增补 + 新耦合 + DECAY_TARGETS
6. **表面层 stress 去放大**：解除 4.5× 放大，fatigue 系数 0.30→0.15

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
| 🟡 P1 | 4 | 往返迟滞，震荡慢衰减，关系态渐近残余，单刺激影响不均衡 |
| 🟢 P2 | 4 | vulnerability 触底 5%，剖面可达极值有限，expressiveness 压缩，内部态渐近残余 |
| ℹ️ 文档化 | 2 | soft_clamp 过渡区，sigmoid 浮点饱和 |

### 运行测试

```bash
# 全部测试（192 用例）
uv run pytest tests/ -v

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
| 3 | 记忆集成 | 当前代码 `nodes.py` stub, 设计见 `MEMORY_SYSTEM.md` |
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
| 192 测试用例全通过 | 06-19 |
