# Lunar 待办与路线图

> 2026-06-19 | 合并自 TODO_LUNAR_STATE_ENGINE.md + STATE_ENGINE_RESEARCH_REPORT.md（前瞻路线图）
>
> 已完成的架构修复（LSTM 三门控 → 残差动力学、decay>1 → 时间衰减、A 矩阵 → 命名规则、β 调制修复）不再列于此。

---

## 一、严重问题（🔴 影响核心功能）

### 🔴 问题 1：特质永远不变，角色"长不大"

- 问题：`T_SENSITIVITY / T_ATTACHMENT_ANXIETY` 等 10 维特质在整个生命周期内不更新
- 后果：玩 100 轮和 1 轮性格没差别
- 学术参考：Bowlby IWM 修正、McAdams 三层人格（特质 → 适应 → 叙事）
- **状态**：待方案

### 🔴 问题 2：刺激向量只有 7 类信号，关键情绪类别缺失

- 现有 7 维偏重"关系性"刺激，缺少 `ST_ANTICIPATION`、`ST_GUILT`、`ST_DISAPPOINTMENT`、`ST_GRATITUDE`、`ST_CURIOSITY`
- 缺少 Appraisal 层（goal_congruence, certainty, agency, coping_potential）
- 学术参考：Plutchik 情感轮盘、Ekman 6 基本情绪、Scherer CPM
- **状态**：待方案

### 🔴 问题 3：记忆系统尚未完全集成

- **当前状态**：MemoryStore + 三路检索 + MemoryNode 已实现，但 `memory_inject_node` 和 `memory_summery_node` 在 `nodes.py` 中为 stub
- 需要：两个图节点完成 → 记忆注入 → LLM 感知记忆
- **状态**：部分实现，待集成

### 🔴 问题 4：无"目标/意图"系统，角色"无欲无求"

- 角色完全被动响应，没有任何"想做某事"的内部驱动
- 学术参考：BDI 模型、Schema Theory、Goal-Directed Behavior、SDT（autonomy/competence/relatedness）
- **状态**：待方案

---

## 二、中等问题（🟡 影响真实感）

### 🟡 问题 5：表面表达只有 7 维，无法表达细腻情感

- 7 维过于粗粒度——"温柔地笑"和"害羞地笑"无法区分
- 缺少 playfulness、defiance、longing、contempt 等
- 学术参考：FACS（44 个动作单元）、Russell Circumplex（valence × arousal）
- **状态**：待方案

### 🟡 问题 6：无"动作/行为"系统，角色"只说不做"

- 角色只能说话，无法模拟"给你倒杯水""整理房间"等主动行为
- **状态**：待方案

### 🟡 问题 7：无"内部独白"机制，角色"无意识流"

- 两次对话之间完全静止，没有"我刚才是不是说错话了"的反思
- 学术参考：Vygotsky Inner Speech、Default Mode Network
- **状态**：待方案

### 🟡 问题 8：对"用户"无模型，角色不知道"你是谁"

- 无 UserModel，不知道用户人格、偏好、压力源
- 每次对话都是"陌生人的第一次见面"
- 学术参考：Bowlby IWM（对他人表征）、Theory of Mind
- **状态**：待方案

---

## 三、锦上添花（🟢 影响沉浸感）

### 🟢 问题 9：无"昼夜节律"
- **状态**：待方案

### 🟢 问题 10：无"个体微习惯"
- **状态**：待方案

### 🟢 问题 11：防御维度待扩展

- 当前只有 deactivation/hyperactivation 两维。Vaillant 分类有 10+ 种防御机制（压抑、投射、合理化、反向形成、升华等）
- 扩展需遵循 `DEFENSE_PROFILE_METHODOLOGY.md` 的标准化流程
- **状态**：方法论文档已写，待扩展

### 🟢 问题 12：无"主观时间感"
- **状态**：待方案

---

## 四、路线图

### Phase A：基础扩展（优先）
- 🔴 问题 2：刺激维度 + Appraisal 层
- 🔴 问题 3：记忆系统图节点完成
- 🟡 问题 8：用户模型 v1

### Phase B：核心能力
- 🔴 问题 1：特质演化
- 🔴 问题 4：目标/意图系统（BDI + SDT）
- 🟡 问题 7：内部独白/反刍

### Phase C：沉浸感增强
- 🟡 问题 5：表面表达扩展
- 🟡 问题 6：行为系统
- 🟢 问题 9、10、11、12

### Phase D：工程化（持续）
- 权重外部化（YAML/JSON 配置）
- State Formatter 连续化（当前 5 级阈值离散化）
- LLM-as-a-judge 评估闭环
- FastAPI 服务化

---

## 五、问题溯源汇总

| 问题 | 关键词 | 推荐文献 |
|------|--------|---------|
| 1 | 特质演化 | Bowlby 依恋理论, McAdams 人生叙事, Whole Trait Theory |
| 2 | 情绪分类 + Appraisal | Plutchik, Ekman, Scherer CPM, OCC 模型 |
| 3 | 记忆集成 | 当前代码 `nodes.py` stub, 设计见 `MEMORY_SYSTEM.md` |
| 4 | 行为驱动 | BDI 模型, SDT, Goal-Directed Behavior |
| 5 | 情绪表达 | FACS, Russell Circumplex Model |
| 6 | 主动行为 | Embodied AI, Action Systems |
| 7 | 内部语言 | Inner Speech, Default Mode Network, Reflection Loops |
| 8 | 用户建模 | User Modeling, Theory of Mind, IWM |
| 9 | 昼夜节律 | Chronopsychology |
| 10 | 个体差异 | Personality Psychology |
| 11 | 防御扩展 | Anna Freud, Vaillant, 新维听添加见 DEFENSE_PROFILE_METHODOLOGY.md |
| 12 | 时间感知 | Psychology of Time, 主观时间膨胀 |

---

## 六、📌 重要已完成（从旧 TODO 归档）

| 项目 | 完成时间 |
|------|---------|
| 刺激+耦合驱动残差更新（γ 移除） | 06-18 |
| A 矩阵 → 10 条命名耦合规则 + SELF_DECAY | 06-18 |
| 防御剖面 sigmoid 缩放 + β 调制修复 | 06-18 |
| 时间感知衰减 `_decay.py`（真实时间幂律衰减） | 06-18 |
| non对称衰减 FAB (negative_decay_boost=1.8) | 06-19 |
| soft_clamp 替代 np.clip | 已修复 |
| G_LEAKAGE 死代码清理 | 已修复 |
| 192 测试用例 + 7 面监督体系 | 06-19 |
