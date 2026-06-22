# Lunar 状态引擎深度研究报告

> 2026-06-21 | 基于 md/ 全部文档 + Web of Science 前沿论文 + GitHub 开源社区综合调研

---

## 一、系统现状全景

Lunar 是一个基于 Bowlby 依恋理论的防御驱动型状态引擎，目前处于 **Phase 1 完成 → Phase 2 启动** 的过渡期。

### 1.1 已完成的工作

| 阶段 | 内容 | 状态 |
|------|------|------|
| 0→1 防御替代门控 | Bowlby 去激活/过度激活替代旧 4 门系统 | ✅ |
| 0→1 残差动力学 | `h_t = h_{t-1} + dt·(αΔ_c + βΔ_s + γΔ_h)` | ✅ |
| 0→1 逐维 SELF_DECAY | 从统一 0.15 改为每维度独立数组 | ✅ |
| 0→1 跨尺度耦合 | 关系态接入 internal[I_INSECURITY/I_LONGING] | ✅ |
| 0→1 β 解耦 | 刺激接受率从捆绑 γ 改为独立 7 维数组 | ✅ |
| 0→1 防御逐维调制 | 13 组全局调制→逐维权重（ACA §7.3.4） | ✅ |
| 0→1 时间感知衰减 | 混合指数衰减 + 人格调制 + 非对称 | ✅ |

### 1.2 待解决问题清单

源于 ROADMAP.md + SPARSE_ANTAGONIST_ANALYSIS.md + STATE_ENGINE_CONSTRAINTS.md + 新审计发现：

| # | 等级 | 问题 | 根因 | 影响范围 |
|:-:|:---:|------|------|---------|
| 1 | 🔴 | **维度冗余**（14→6 有效维） | B 矩阵密集映射 | 全状态空间 |
| 2 | 🔴 | **Trait 静态** | 从未实现演化 | 全线传导 |
| 3 | 🔴 | **防御剖面独立方差 ≈ 0%**（真实场景） | 输入源太少（3→7） | 防御系统 |
| 4 | 🔴 | **记忆系统 stub** | 未完成集成 | LangGraph 管线 |
| 5 | 🔴 | **SurfaceState 未注入 LLM** | 设计遗落 | 核心功能 |
| 6 | 🔴 | **无 Appraisal 层** | 仅 7 维刺激不够 | 感知系统 |
| 7 | 🟡 | **约束框架部分实施**（⑤✅ ⑧✅，其余仍违反） | WeightMapper/ConstraintRegistry 骨架已实现 | 架构治理 |
| 8 | 🟡 | **扁平动力学架构** | 统一 dt=1，无快慢分离 | 动力学 |
| 9 | 🟡 | **无目标/意图系统** | 纯被动 | 行为系统 |
| 10 | 🟡 | **无 UserModel** | 未开发 | 角色交互 |
| 11 | 🟡 | **StateFormatter 离散化** | 5 级阈值→重离散化 | 表现层 |
| 12 | 🟡 | **突破事件无处理器** | stubbed | 核心功能 |
| 13 | 🟢 | **权重硬编码** | 未外部化 | 工程化 |
| 14 | 🟢 | **上下文窗口仅 4 条** | 未扩展 | 感知 |
| 15 | 🟢 | **decay.py setpoint 重复** | 代码重复 | 可维护性 |

---

## 二、根因深析：三个互相交织的结构性缺陷

### 2.1 缺陷 A：信息瓶颈（B 矩阵结构性支配）

SPARSE_ANTAGONIST_ANALYSIS §7.5 的核心发现——**耦合拓扑不是维度冗余的根因，B 矩阵的密集映射才是。**

```
7 维刺激 → REL_INPUT_INFLUENCE_B (42 条映射) → 6 维同步运动
            ↑ β × B ≈ 0.0032/轮，是耦合 5-10 倍
```

验证证据：
- 删除全部耦合边 → 冗余度反而上升（PC1 87.4%→89.2%）
- 增加拮抗边、断环、结构平衡 → 有效自由度纹丝不动
- 即使每个关系维有独立的微分方程，共享输入意味着共享动态

### 2.2 缺陷 B：输入自由度不足（防御剖面对偶问题）

**DEFENSE_PROFILE_INDEPENDENCE_AUDIT.md** 的新发现——防御剖面面临和 B 矩阵**同构的问题**：

```
关系态:   7 维刺激 → B 矩阵 (42 条) → 6 维同步
防御剖面: 3 标量输入 → 逐维权重 (21 条) → 7 维同步
```

固定 traits（真实运行场景）时：

| 实验 | PC1 | 有效秩 | 独立方差 |
|------|:---:|:-----:|:-------:|
| 全随机 | 90.4% | 1.40 | 0.0% |
| 仅 internal 变化 | 90.6% | 1.37 | 0.0% |
| 仅 relationship 变化 | **100.0%** | **1.00** | 0.0% |

"仅 rel" 时 PC1=100%，因为去激活的唯一关系调制器是 `trust_bond` 一个标量。7 个输出是同一标量乘以不同系数的线性组合——完美共线。

**两条根因链的汇合点：**

```
问题 A: 刺激空间 → 关系态映射     → 信息瓶颈在 B 矩阵
问题 B: 状态空间 → 防御剖面映射 → 信息瓶颈在输入源个数
        ↑ 本质相同：输出维数远大于输入独立源数
```

### 2.3 缺陷 C：Trait 静态性跨系统传导

Trait 不变→防御剖面基线偏移不变→刺激特异性调制被静态基线淹没（~90% 方差来自 traits）→dynamics 中 trait 不更新→surface 投影的 trait 路径不动。

```
Traits 静态 → 防御剖面 (§7.3.5)
           → 动力学 setpoint 不移动
           → surface 投影的 trait 路径恒定
           → 角色"长不大"
```

修复逐维权重后，动态调制已有维度特异性，但仅占总方差的 12.6%（deact）和 8.7%（hyper）。剩余的 ~87-91% 是 traits 决定且永不移位的基线。

---

## 三、学术前沿

### 3.1 情感评价计算（OCC + PAD + Personality 三合一）

**Liu et al. (HICSS 2025)** — OCC-PAD-OCEAN 框架：用 VGG19 从视频预测 Big Five，桥接 OCC 评价结构、PAD 维度空间和 OCEAN 人格模型。验证了对尽责性、外倾性和宜人性的显著预测力。

**关键公式（AIAT 2025, MBTI→PAD 映射）：**
```
Pleasure   = 0.21E + 0.59A + 0.19N
Arousal    = 0.15O + 0.30A − 0.57N
Dominance  = 0.25O + 0.17C + 0.60E − 0.32A
```

**对 Lunar 的启示：** OCC 的 22 种情感类型（joy/distress、hope/fear、pride/shame、admiration/reproach、love/hate、gratitude/anger 等拮抗对）可以直接映射到 Lunar 的刺激空间。当前 7 维刺激向量可扩展为：

| 当前 7 维 | OCC 映射 | 建议扩展 |
|-----------|---------|---------|
| abandonment | fear/loss | ← 保留，细化 |
| validation | pride/shame | ← 扩展为 pride+shame |
| closeness | love/attachment | ← 保留 |
| conflict | anger/distress | ← 扩展为 anger+gratitude |
| dependency | need | ← 保留 |
| teasing | joy | ← 扩展为 joy+play |
| emotional_weight | distress | ← 保留 |

### 3.2 双速情感动力学

**Sentipolis (Fu et al., arXiv 2026)** — 最直接相关的论文。双速情感动力学 + 连续 PAD 表示 + 情感-记忆耦合：

- 快速层: 即时情绪反应（每轮更新）
- 慢速层: 累积情感基调（低通滤波）
- 情感真实性提升 2×（human evaluation）
- 网络诊断显示互易的、中度聚类的、时间稳定的关系结构

**对 Lunar 的启示：** 验证了 DUAL_TIMESCALE_SSM.md 的双速架构方向。Sentipolis 在 LLM agent 上的成功说明双速架构对角色真实感有直接可测量的提升。

### 3.3 非线性 SSM 替代线性动力学

**PLRNN (npj Digital Medicine, 2025)** — 分段线性 RNN 状态空间模型在情感轨迹预测上显著优于 VAR(1)、Kalman filter 和 Transformer。捕获了线性模型错过的**多稳态和相变**。

结论：情感动力学本质上是非线性的。当前 Lunar 的线性残差方程 `h_t = h_{t-1} + dt·(αΔ_c + βΔ_s)` 在数学上是稳定的，但可能错过了真实情感系统的非线性行为（阈值触发、习惯化、对立过程反弹）。

### 3.4 MECoT: 马尔可夫情感链式推理

**Wei et al. (ACL 2025)** — 双过程架构：快（马尔可夫链情感处理器）+ 慢（LLM 理性调节）。

- 12 维情绪圆周模型（valence × arousal 的细化）
- 人格加权的状态转移矩阵
- 93.3% 情感准确率

**对 Lunar 的启示：** 情感转移矩阵 + 人格权重的组合，本质上和 Lunar 的状态耦合矩阵 + trait 调制是同一思想。但 MECoT 用离散马尔可夫链 + 12 个离散状态做了简化，而 Lunar 用连续状态空间 + 微分方程做了更深层但更复杂的建模。

### 3.5 MATE: 确定性情感中间件

**Lobozov (2026)** — 纯函数 `transition(state, event) → new_state`，零 LLM 调用：
- 30 维特征系统
- 密度矩阵（量子概率）替代古典向量
- 对立过程（A-process → B-process 延迟反弹）
- 双过程习惯化

**对 Lunar 的启示：** MATE 的密度矩阵比 Lunar 的古典向量更丰富（可以建模叠加态），但 41500 行内核也说明复杂度急剧上升。对立过程和习惯化是 Lunar 完全缺失的——当前没有"迟滞反弹"机制。

### 3.6 Scherer CPM 的计算实现

**Taj (VU Amsterdam PhD, 2023)** — 时序因果网络实现完整 CPM 四阶段：

```
Relevance → Implication → Coping Potential → Normative Significance
  (新颖性)    (目标一致性)    (控制/权力/适应)    (内外标准)
```

使用微分方程：状态激活值 [0,1] 或 [-1,1]，因果连接权重 ω ∈ [-1,1]，速度因子 η ∈ [0,1] 控制变化率。

**对 Lunar 的启示：** Lunar 当前的 `user_stimuli` 实际上是一个隐式的评价输出（LLM 将用户输入映射到 7 维），但没有显式的评价过程。CPM 的 4 个 SEC 组提供了如何分解"评价→情感"过程的正式框架。

### 3.7 依恋理论计算建模

**ANEX_BayesMind (Notion, 2026)** — 贝叶斯信念系统，将 4 种依恋风格编码为先验：
- 先验与身份模块 → 观察与证据处理 → 信念更新引擎 → 认知与目标层 → 输出生成
- 34% 行为一致性提升，41% 矛盾响应减少

**Petters (CME 2017)** — 自主智能体的依恋风格涌现：初始相同的智能体通过小随机波动→正反馈放大→分叉为安全型/不安全型。

**对 Lunar 的启示：** Lunar 目前使用依恋理论作为设计的**描述性框架**（trait 中有 attachment_anxiety 和 avoidance），但没有实现**依恋系统本身的动力学**（内部工作模型的更新、依恋行为系统的激活/解除）。ANEX 的贝叶斯方法提供了一个可选的实现路径。

---

## 四、开源社区生态

### 4.1 最相关的开源项目

| 项目 | 关联度 | 核心思想 | 评估 |
|------|:-----:|---------|------|
| **Soul Protocol** | ★★★★★ | OCEAN + 5 级记忆 + 躯体标记 + .soul 文件移植 | 最接近的"完整框架"对照 |
| **Soul Engine (OpenSouls)** | ★★★★☆ | MentalProcesses 状态机 + WorkingMemory | LangGraph 结构的替代思路 |
| **EloPhanto** | ★★★★☆ | PAD 基板 + OCC 标签 + 伊戈置信度 | 与 Lunar 设计哲学最接近 |
| **Relic** | ★★★★☆ | 附着理论 + 置信度追踪 + 纵向建模 | 用户建模层可借鉴 |
| **Sentimo** | ★★★☆☆ | Big Five + 6 情感 + 双内存 + 指数平滑衰减 | 模型简单但对快速迭代有用 |
| **MECoT** | ★★★☆☆ | 12 维圆周模型 + 马尔可夫转移矩阵 | 离散情感标签的黄金标准 |
| **soulcuit v2** | ★★☆☆☆ | 约束图：特质×需求×情感×认知 | 全局雅可比稀疏的设计蓝图 |
| **GAMYGDALA** | ★★☆☆☆ | 评价式情感引擎 + Phaser | 经典但架构已过时 |
| **josephkirk/EmotionEngine** | ★★☆☆☆ | Plutchik + 弹簧物理 + Unreal Engine 5 | 弹簧物理→新的动力学范式 |
| **ReflexCore** | ★★☆☆☆ | 7 层认知 + 感知→情感分析→特质记忆 | 分层架构参照 |

### 4.2 Soul Protocol 深度分析

最值得深入研究的对照项目。关键设计：

```python
# 典型的 Soul 状态片段（从公开文档重构）
{
  "soul_id": "kiana_2025",
  "personality": {  # OCEAN
    "openness": 0.7, "conscientiousness": 0.5,
    "extraversion": 0.8, "agreeableness": 0.6,
    "neuroticism": 0.4
  },
  "emotion": {  # Damasio 躯体标记
    "valence": -0.2, "arousal": 0.6,
    "somatic_markers": [
      {"stimulus": "abandonment_trigger", "marker": -0.7, "decay": 0.95}
    ]
  },
  "memory": {  # 5 级
    "working": [...], "episodic": [...],
    "semantic": [...], "emotional": [...], "procedural": [...]
  }
}
```

与 Lunar 的差异：
- Soul 使用 OCEAN + 二维情感（valence/arousal），Lunar 使用 8+6+7+10 维（更丰富但更复杂）
- Soul 的躯体标记效应 = Lunar 的刺激 + 防御机制的合并
- Soul 的 5 级记忆 = Lunar 的 3 级记忆的超集
- Soul 的 `.soul` 文件可移植性优于 Lunar 的硬编码默认值

### 4.3 EloPhanto 深度分析

PAD 基板 + OCC 标签的设计与 Lunar 的 DUAL_TIMESCALE_SSM.md 高度一致：

```python
# EloPhanto 的 Affect 层结构（从公开文档重构）
class Affect:
    pad: np.ndarray  # (3,) — PAD 基板，每轮更新
    occ_labels: dict  # OCC 22 情感标签的强度
    
    def update(self, stimuli):
        self.pad += delta_pad(stimuli)  # 连续更新
        self.occ_labels = project(self.pad)  # PAD → OCC 投影
        self.pad *= decay  # 自然衰减（快于 trait 变化）
```

关键差异：EloPhanto 从 PAD 3 维投影到 OCC 标签，Lunar 从 7 维刺激独立计算。前者有维度优势（3 维保证正交性），后者有语义丰富度（7 维直接对应心理类别）。

---

## 五、可行解决方案

### 5.1 优先级矩阵

| 方案 | 影响 | 成本 | 风险 | 依赖 | 排期 |
|------|:---:|:----:|:----:|------|:----:|
| **A: 语义合并关系维度 6→3** | 🟢 高 | 🟢 1d | 🟢 低 | 无 | **P0 本周** |
| **B: 增加防御剖面输入源** | 🟢 高 | 🟢 1d | 🟢 低 | 无 | **P0 本周** |
| **C: β 调制系数释放** | 🟡 中 | 🟢 0.5d | 🟡 中 | A | P1 |
| **D: OCC 拮抗对扩展** | 🟢 高 | 🟡 3d | 🟡 中 | 无 | P1 |
| **E: Trait 演化 v1** | 🔴 极高 | 🟡 5d | 🔴 高 | D | P2 |
| **F: 双速 SSM v1** | 🔴 极高 | 🔴 2w | 🔴 高 | A, E | P2 远期 |
| **G: 约束框架实现** | 🟡 中 | 🟡 3d | 🟢 低 | 无 | P1 |
| **H: 记忆系统集成** | 🟢 高 | 🟡 2d | 🟡 中 | 无 | P1 |
| **I: 状态空间→PAD 混合** | 🟢 高 | 🔴 5d | 🔴 高 | F | P3 |

### 5.2 方案 A（推荐 P0）：语义合并关系维度 6→3

**依据：** SPARSE_ANTAGONIST_ANALYSIS §7.5 和 §5 — 预计有效自由度增益 30-50%。

**具体设计：**

| 6 维当前 | 3 维合并 | 心理学依据 |
|----------|---------|-----------|
| `R_AFFECTION` + `R_INTIMACY` | **R_BOND**（情感纽带） | Sternberg 亲密成分：喜欢+亲密→情感纽带 |
| `R_TRUST_BOND` + `R_EMOTIONAL_SAFETY` | **R_TRUST**（信任/安全） | Bowlby 安全基地：信任=安全+可靠 |
| `R_FAMILIARITY` + `R_DEPENDENCY` + `R_ROMANTIC_TENSION` | **R_INVOLVEMENT**（卷入度） | 熟悉+依赖+张力→综合卷入度 |

**实施步骤：**
1. 修改 `state.py`：`R_SIZE = 3`，新增 `R_BOND=0, R_TRUST=1, R_INVOLVEMENT=2`
2. 修改 `_matrices.py`：重建 `REL_STATE_COUPLING_A`（3×3）和 `REL_INPUT_INFLUENCE_B`（3×7）
3. 修改 `_dynamics.py`：`compute_rel_setpoint()` 和 `update_relationship_state()` 使用 3 维
4. 修改 `_surface.py`：surface 投影使用 3 维关系输入
5. 调整 `R_LABELS` 和相关提及
6. 重跑所有测试，更新断言

**风险缓解：**
- `_decay.py` 需同步修改（`REL_SELF_DECAY` 从 6→3 维）
- 旧 checkpoint 不兼容（但 SQLite saver 会重新初始化）
- 测试断言几乎全需要新 baseline

### 5.3 方案 B（推荐 P0）：增加防御剖面输入源

**依据：** DEFENSE_PROFILE_INDEPENDENCE_AUDIT.md §V-A — 当前仅 3 个标量输入驱动 7 维输出。

**具体设计：**

新增去激活调制器：

| 新增输入 | 来源维度 | 权重数组 | 心理学依据 |
|---------|---------|---------|-----------|
| `R_INTIMACY` | relationship | `INTIMACY_DEACT_A` (7,) | 暧昧气氛下防御变化 |
| `I_LONELINESS` | internal | `LONELINESS_DEACT_A` (7,) | 孤独时更渴望联结→降防 |
| `I_ENERGY` | internal | `ENERGY_DEACT_A` (7,) | 精力充沛时更敢于面对冲突 |

新增过度激活调制器：

| 新增输入 | 来源维度 | 权重数组 | 心理学依据 |
|---------|---------|---------|-----------|
| `I_STRESS` | internal | `STRESS_HYPER_A` (7,) | 压力→过度激活全面放大 |
| `I_MENTAL_FATIGUE` | internal | `FATIGUE_HYPER_A` (7,) | 疲劳→情感反应阈值降低 |

**实施步骤：**
1. 在 `_defenses.py` 中定义 5 组新权重数组（每组 7 维）——*10 分钟*
2. 在 `compute_defense_profiles()` 中添加对应的调制步骤——*15 分钟*
3. 验证：每组均需 PCA 独立方差 > 2%（**无需改动 `_defenses.py` 外的任何文件**）
4. 运行现有测试验证无回归

**预期效果：** 输入源从 3→5（去激活）和 4→6（过度激活），独立方差预期从 ~0% → ~5-10%。

**权重设计样例：**

```python
LONELINESS_DEACT_A: np.ndarray = np.array([
    0.05,  # ST_ABANDONMENT    — 孤独→更怕被抛弃（特敏锐）
    0.12,  # ST_VALIDATION     — 核心: 孤独→极度需要被认可
    0.10,  # ST_CLOSENESS      — 孤独→渴望亲近，降低防御
    -0.03, # ST_CONFLICT       — 孤独→不敢冲突（怕失去）
    0.08,  # ST_DEPENDENCY     — 孤独→想依赖
    0.00,  # ST_TEASING        — 不影响调侃
    0.07,  # ST_EMOTIONAL_WEIGHT — 孤独→不回避沉重（反正已经沉重了）
])
```

### 5.4 方案 C（P1）：β 调制系数释放（通用动力学增强）

**当前问题：** `_dynamics.py` 的 `update_internal_state` 中 β 系数（刺激接受率）是全局标量（0.04），即使在 2025-06-18 已解耦为 8 维独立数组，绝对大小仍受限于统一基准。

**建议：** 将 β 有效范围从 [0.02, 0.08] 扩展到 [0.01, 0.20]，使高焦虑角色的刺激接受率达到低焦虑角色的 20 倍（而不是当前仅 4 倍）。

```python
# _dynamics.py 中 β 的范围
# 当前: beta[i] = base_rate_i + defense_mod + trait_mod
#     base_rate_i ∈ [0.02, 0.08]
# 建议: base_rate_i ∈ [0.01, 0.20]
```

### 5.5 方案 D（P1）：OCC 拮抗对扩展刺激空间

**依据：** OCC 模型 + SPARSE_ANTAGONIST_ANALYSIS §3.1。当前 7 维刺激可以扩展为 OCC 结构化的 12 维空间，每对互为拮抗。

**建议扩展（+5 维）：**

| 新增刺激 | 拮抗对 | OCC 类别 | 心理学依据 |
|---------|--------|---------|-----------|
| `ST_JOY` | −0.3 × distress | joy/distress | 事件合意性 |
| `ST_HOPE` | −0.3 × fear | hope/fear | 未来事件预期 |
| `ST_PRIDE` | −0.2 × shame | pride/shame | 自我行动评价 |
| `ST_ADMIRATION` | −0.2 × reproach | admiration/reproach | 他人行动评价 |
| `ST_GRATITUDE` | −0.25 × anger | gratitude/anger | 他人意图 |

**注意：** 修改刺激维度数（`ST_SIZE`）会影响 `_matrices.py` 中的所有矩阵形状、`_defenses.py` 中的权重数组、`_dynamics.py` 中的 B 矩阵、`_surface.py` 中的投影矩阵。预计需要修改 **8+ 个文件**。所以虽然本身难度不高，但连锁反应大。

**变通方案：** 不在刺激向量中扩展，而是在 `perception.py` 层面添加一个"评价后处理"步骤，从 LLM 输出中解构出 OCC 评价维度，映射到现有 7 维刺激空间——不影响下游代码。

### 5.6 方案 E（P2）：Trait 演化 v1

**约束条件（来自 SALM 的对数收敛定理）：**
```python
||R_{t+k} - R_t|| ≤ γ·log(k) + ε   # γ = 0.08
```
保证 4000 轮后人格稳定性 ≥ 0.87。

**最小可行设计（基于 Attachment 理论 + Relic 置信度追踪）：**

```python
class TraitEvolution:
    def __init__(self, traits, decay_rate=0.001, max_drift=0.02):
        self.traits = traits.copy()
        self.confidence = np.ones(10) * 0.5  # 置信度 0-1
        self.observation_count = np.zeros(10)
    
    def update(self, internal, relationship, stimuli, defense_profiles):
        # 1. 计算期望值与实际值的偏差
        # 2. 按置信度加权更新
        # 3. 应用对数漂移边界
        # 4. 增加观察计数 / 提升置信度
```

**实施建议：** 在 `state_engine_node` 末尾添加 trait 更新步骤，每 5-10 轮触发一次（不是每轮——太频繁反而没有意义）。

### 5.7 方案 F（P2）：双速 SSM v1

直接对应 DUAL_TIMESCALE_SSM.md 的 Phase 1。

**状态空间分解：**

```
当前: internal (8,) + relationship (6,) = 14 维，统一 dt=1
分解: 
  fast (3,) — P/A/D 连续值，dt=1，τ=2-5 轮
  internal (5,) — 去除 P/A 的剩余维度，dt=1，τ=5-20 轮
  relationship (6,) — dt=每 5-50 轮累积更新，τ=50-200 轮
```

其中 P/A/D 的快速动力学（仿 Sentipolis）：
```
E_fast_{t+1} = (1 - δ_e)·E_fast_t + α_e·S_t + β_e·(R_slow_t - μ_e) + σ_e·ξ_t
```

### 5.8 方案 G（P1）：约束框架实现

STATE_ENGINE_CONSTRAINTS.md 的 9 条约束中，当前实际遵守仅 1 条（谱半径）。

**最低可行实现顺序：**
1. **约束⑤ WeightMapper**：在 `state_engine/_validator.py` 中实现（1d）。将 `_matrices.py` 中的裸数字改由 WeightMapper 管理
2. **约束⑨ 全局雅可比稀疏**：实现组合雅可比检查（1d）。验证管道级边密度 ≤ 30%
3. **约束③ 矩阵低秩**：为所有矩阵添加有效秩检查和预期秩注释（0.5d）
4. **约束①/④**：**已修复（06-22 Surface 重构）** — traits 从 surface 移除，outer_stimuli 确认为 defenses 压抑后输入

### 5.9 方案 H（P1）：记忆系统集成

MEMORY_SYSTEM.md 已提供完整设计。当前仅需在 `nodes.py` 中完成两个 stub 函数并注册到 `_builder.py`。

**关键修正（已识别于 MEMORY_SYSTEM.md）：**
- `user_stimuli` 清除责任从 `state_engine_node` 移到 `memory_formation_node`
- 在 State 中添加 `retrieved_memories: Optional[str]`

### 5.10 前沿架构参考：PLRNN + SSM 替代残差动力学

来自 npj Digital Medicine (2025) 的发现：分段线性 RNN 在情感轨迹预测上显著优于线性模型。

**PLRNN 的残差形式：**
```python
z_t = A·z_{t-1} + W·φ(z_{t-1}) + h_t + C·s_t
# 其中 φ 是分段线性函数（ReLU-like）
```

与 Lunar 当前残差方程的比较：
```
Lunar:  h_t = h_{t-1} + dt·(α·A·h_{t-1} + β·B·s_t + γ·(μ - h_{t-1}))
PLRNN:  z_t = A·z_{t-1} + W·ReLU(z_{t-1}) + C·s_t
```

差异：
- Lunar 的 coupling (A·h) ≈ PLRNN 的 A·z（线性部分）
- Lunar 的 stimulus (B·s) ≈ PLRNN 的 C·s（输入）
- Lunar **缺少** PLRNN 的 `W·φ(z)` 非线性项——这意味着 Lunar 无法产生分叉、多稳态等非线性行为

**建议：** 如果未来要实现相变（如"暧昧→冷/热"分叉），需要添加非线性耦合项。当前 Phase 1 不需要。

---

## 六、总体路线图

```
现在──┐
       │
       ├ [P0-本周]  方案 A: 语义合并 6→3
       ├ [P0-本周]  方案 B: 增加防御输入源
       │
       ├ [P1-1周]   方案 C: β 调制释放
       ├ [P1-1周]   方案 D: OCC 拮抗对扩展
       ├ [P1-1周]   方案 G: 约束框架最低实现
       ├ [P1-1周]   方案 H: 记忆系统集成
       │
       ├ [P2-2周]   方案 E: Trait 演化 v1
       │
       ├ [P2-3周]   方案 F: 双速 SSM v1
       │             (依赖于 A + E 完成)
       │
       └ [P3-远期]  非线性动力学 / PLRNN
                     UserModel / UserProfile
                     State Formatter 连续化
                     FastAPI 服务化
```

### 关键依赖

```
A (语义合并) → 降低 B 矩阵冗余 → F (双速 SSM) 的前提
B (防御输入) → 独立方差提升 → 防御系统的基线修正
D (OCC 扩展) → 提供 trait 更新的信号源 → E (Trait 演化)
E (Trait 演化) → 解除静态基线 → 全系统杠杆效应
A + E → 解决维度冗余 + 静态基线的组合根因 → 远期架构
```

---

## 七、参考文献

### 学术论文
1. Liu, F. et al. (2025). "OCC-PAD-OCEAN: A Quantitative Perceptible Modeling of Big Five Personality Based on Computational Affection". *HICSS 2025*.
2. Fu, C. et al. (2026). "Sentipolis: Emotion-Aware Agents for Social Simulations". *arXiv:2601.18027*.
3. Wei, Y. et al. (2025). "MECoT: Markov Emotional Chain-of-Thought for Personality-Consistent Role-Playing". *ACL 2025 Findings*.
4. Lobozov, A. (2026). "MATE: Mathematical Architecture for Thoughtful Entities". *Zenodo*.
5. Taj, F. (2023). "Temporal Causal Network Model of Scherer's Component Process Model". *VU Amsterdam PhD Thesis*.
6. Solomon, R.L. & Corbit, J.D. (1974). "An opponent-process theory of motivation". *Psychological Review*.
7. Ortony, A., Clore, G.L. & Collins, A. (1988). *The Cognitive Structure of Emotions*. Cambridge University Press.
8. Mehrabian, A. (1996). "Pleasure-arousal-dominance: A general framework for describing and measuring individual differences in temperament". *Current Psychology*.
9. Oravcez, Z. et al. (2009-2011). "O-U process models for affect dynamics". *Multivariate Behavioral Research*.
10. Loossens, T. et al. (2020-2024). "Affective Ising model: Nonlinear multi-attractor landscape for emotion". *Emotion*.
11. Koley, S. (2025). "SALM: Bounded personality drift theorem". *Preprint*.
12. Richardson, E., Beath, A. & Boag, S. (2023, 2025). "Defense mechanisms in attachment theory - empirical validation". *Journal of Personality Assessment*.
13. Berscheid, E. (1983). "Emotion in close relationships: A stimulation value model". *Close Relationships*.

### 开源项目
14. **Soul Protocol** — https://pypi.org/project/soul-protocol/
15. **Soul Engine (OpenSouls)** — https://github.com/opensouls/opensouls
16. **EloPhanto** — https://github.com/elophanto/EloPhanto
17. **Relic** — https://github.com/yuzushi-dev/Relic
18. **Sentimo** — https://github.com/0petru/sentimo
19. **soulcuit v2** — https://github.com/dynamder/Soulcuit
20. **JosephKirk/EmotionEngine** — https://github.com/josephkirk/EmotionEngine
21. **GAMYGDALA** — https://ii.tudelft.nl/~joostb/gamygdala/
22. **MECoT Code** — https://anonymous.4open.science/r/MECoT

### Lunar 文档
23. `SPARSE_ANTAGONIST_ANALYSIS.md` — 稀疏化耦合与拮抗对方案分析
24. `DEFENSE_PROFILE_INDEPENDENCE_AUDIT.md` — 防御剖面独立性审计
25. `DUAL_TIMESCALE_SSM.md` — 双速状态空间模型研究
26. `AFFECTIVE_GEOMETRY_RESEARCH.md` — 情感几何与信息瓶颈
27. `STATE_ENGINE_CONSTRAINTS.md` — 状态引擎约束框架宪法
28. `ARCHITECTURE.md` — 当前架构全景
29. `ROADMAP.md` — 待办与路线图
30. `STATE_ENGINE_TEST_REPORT.md` — 测试报告
31. `DEFENSE_PROFILE_METHODOLOGY.md` — 防御剖面方法论
32. `MEMORY_SYSTEM.md` — 记忆系统设计
33. `TIME_DECAY_DESIGN.md` — 时间衰减设计
