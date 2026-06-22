# 内部驱力系统设计 —— 让状态引擎从被动到主动

> **版本**: 初稿 v1 | **日期**: 2026-06-22 | **状态**: 设计研讨
>
> 相关文档: [ARCHITECTURE.md](ARCHITECTURE.md) | [STATE_ENGINE_CONSTRAINTS.md](STATE_ENGINE_CONSTRAINTS.md) | [DEFENSE_PROFILE_METHODOLOGY.md](DEFENSE_PROFILE_METHODOLOGY.md) | [ROADMAP.md](ROADMAP.md)

---

## 摘要

Lunar 当前是纯反应式系统：所有心理刺激唯一来源于 `perception_node` 对用户消息的提取。没有内部驱力，角色在对话间隔中处于"心理冻结"状态。

本方案在状态引擎中增加一个**内部驱力生成层（Internal Drive Layer）**，使角色能自主产生心理刺激——孤独时渴望连接、思念时不自觉靠近、独处太久开始焦虑——让状态引擎从"被动响应器"变为"自驱动情感系统"。

---

## 一、理论依据

### 1.1 Bowlby 内部控制论模型（已有基础）

Bowlby (1969/1982) 的依恋理论本身就否定了"心理是纯反应式的"这一前提。他借用 Craik (1943) 的"工作模型"概念和**控制论**框架，提出：

> 个体在头脑中进行"小型实验"（small scale experiments within the head）——模拟可能的行动及其后果，再决定是否执行。

这意味着**心理活动可以脱离外部刺激独立运行**。Bowlby 的控制论模型中包含：
- **目标校正行为（goal-corrected behavior）** ——行为由内部目标驱动，非外部刺激
- **稳态恢复（homeorhesis）** ——系统有内在的朝向稳态发展的趋势
- **探索系统（exploratory system）** ——与依恋系统互补，在安全感满足后自动激活

> **引用**: Bowlby J. (1969/1982). *Attachment and Loss, Vol. 1: Attachment*. | Petters & Waters (2014). "From Internal Working Models to Embodied Working Models". AISB50.

### 1.2 Panksepp 的 SEEKING 系统——基本驱力的神经基础

Panksepp (1998) 在情感神经科学中识别出七个原始情感系统，其中 **SEEKING** 系统是最基础的**通用探索/觅食系统**，它产生好奇心、兴趣和期待——在无外部刺激时依然活跃。

| 系统 | 功能 | 对应 Lunar 驱力 |
|------|------|----------------|
| **SEEKING** | 探索、期待、好奇 | 渴望连接、寻求认可 |
| **FEAR** | 回避威胁 | 被抛弃恐惧（已有 ST_ABANDONMENT） |
| **RAGE** | 愤怒、攻击 | 烦躁积累（已有 ST_CONFLICT） |
| **PANIC** | 分离痛苦 | 孤独、思念积累（已有 ST_ABANDONMENT） |
| **CARE** | 养育、关怀 | 被需要、亲密靠近 |
| **LUST** | 性欲、社交亲近 | 亲密张力、渴望 |
| **PLAY** | 社交游戏、欢乐 | 调侃、互动 |

关键洞察：SEEKING 系统是"永远开启"的。在计算机似中，Joseph & Levkowitz (2011) 将其建模为：

```
SEEKING += 1000/foodDistance + 1000/AGE + HUNGER + RAGE + PANIC + FEAR + PLAY + LUST
```

即使在没有外部"食物"时，SEEKING 仍然因内部积累（饥饿/孤独/无聊）而激活。

> **引用**: Panksepp J. (1998). *Affective Neuroscience*. | Joseph & Levkowitz (2011). "Patterns of Emotion Driven by Affect State and Environment". PATTERNS 2011.

### 1.3 DMN（默认模式网络）——静息态的自发认知

神经科学发现：大脑在"无事可做"时并非空闲——**默认模式网络（DMN）** 的活跃程度反而升高（Raichle et al., 2001）。DMN 负责：
- **心智游移（mind wandering）** ——自发回忆、未来模拟
- **社会认知** ——推测他人想法、自我反思
- **情绪调节** ——对过去的情绪事件重评估

计算模型（Senthilvanan et al., 2025）将 DMN-CEN-SN 三网络建模为耦合随机微分方程，其中 DMN 的活跃度与**内部情感状态**成正比——情感越强，自发认知越频繁。

这为"角色在对话间隔中内心活动不会停止"提供了神经科学依据。

> **引用**: Raichle et al. (2001). "A default mode of brain function". PNAS. | Senthilvanan et al. (2025). "Comprehensive Computational Framework for DMN-CEN-SN Dynamics".

### 1.4 驱力积累/阈值的心理学模型

Hull (1943) 的驱力还原理论（Drive Reduction Theory）将驱力定义为：

```
D = 生理需求 × 习惯强度
行为 = D × Habit
```

在情感领域，**孤独感**不是"被冷落"那一刻才产生的——它在社交隔离中逐渐积累，达到阈值后才进入意识（Cacioppo & Hawkley, 2009）。这一积累过程天然适合用**残差动力学**建模：

```
drive[t] = drive[t-1] + Δt · (accumulation_rate - decay_rate)
```

这正是 Lunar 残差动力学 `h_t = h_{t-1} + Δt · (...)` 的形式。

---

## 二、设计

### 2.1 核心思路

在现有状态引擎的 4 步管线中，增加一个**零号步骤（Step 0）**：

```
现有管线:
  ① Defense Profiles → ② Dynamics → ③ Surface → ④ Feedback

新管线:
  ⓪ Internal Drives → ① Defense Profiles → ② Dynamics → ③ Surface → ④ Feedback
                     ↑ 内驱作为额外刺激输入
```

⓪ 的职责：从当前状态向量（internal + relationship + traits）生成一个**内部驱力刺激向量（drive_stimuli, 7D）**，与外部刺激合并后进入防御剖面。

### 2.2 数学模型

驱力向量 `D(t)` 是一个 7 维刺激向量（与 `StimulusVector` 同构），由三部分组成：

```
D(t) = D_baseline(traits) + D_accumulated(t) + D_spontaneous(t)

其中:
  D_baseline(traits)      — 人格决定的"基线渴望"，静态
  D_accumulated(t)        — 状态维度的持续性积累效应，动态
  D_spontaneous(t)        — 随机游走成分，模拟心智游移
```

#### 2.2.1 D_baseline — 人格调制基线驱力

某些人格天生有更强的内在驱力方向：

| 特质组合 | 驱力方向 | 心理学依据 |
|----------|---------|-----------|
| 高依恋焦虑 + 低回避 | → ST_CLOSENESS ↑, ST_ABANDONMENT ↑ | 焦虑型依恋：即使安全也倾向寻求更多确认 |
| 高回避 | → ST_CLOSENESS ↓ | 回避型：内驱方向是拉开距离 |
| 高敏感 + 高焦虑 | → ST_EMOTIONAL_WEIGHT ↑ | 情绪放大倾向 |
| 高乐观 + 低回避 | → ST_VALIDATION ↑ | 积极寻求认可 |
| 高依恋焦虑 + 高敏感 | → ST_TEASING 敏感化 | 对模糊社交信号赋予更多意义 |

通过 `DRIVE_BASELINE_MAPPER`（`LinearMapping`，与 SURFACE_MAPPER 同构）计算：

```python
D_baseline = soft_clamp(DRIVE_BASELINE_MAPPER.compute(traits), -0.1, 0.3)
```

**约束合规**：这是一个 `LinearMapping`，所有参数通过 `connect()` 注册 provenance，服从约束⑤⑧。

#### 2.2.2 D_accumulated — 状态驱动的积累驱力

这是核心机制：**内部状态本身产生刺激**。数学上与现有动力学对称：

```python
drive_accumulated[stimulus_dim] = Σ W_drive[state_dim, stimulus_dim] · state[state_dim]
```

即：`D_acc = state @ W_drive`，其中 `state = [internal(8) + relationship(3) + surface(7)]` 共 18 维输入 → 7 维输出。

关键的心理学映射规则（**每条附带心理学依据，通过 WeightMapper `connect()` 注册**）：

| 输入状态 | 输出刺激 | 强度 | 心理学依据 |
|----------|---------|:----:|-----------|
| ↑ I_LONELINESS | → ST_CLOSENESS ↑ | 0.20 | 孤独产生靠近渴望（Cacioppo, 2009） |
| ↑ I_LONELINESS | → ST_ABANDONMENT ↑ | 0.10 | 孤独激活被抛弃恐惧 |
| ↑ I_LONGING | → ST_CLOSENESS ↑ | 0.25 | 思念产生连接冲动 |
| ↑ I_LONGING | → ST_VALIDATION ↑ | 0.10 | 思念中确认关系需求 |
| ↑ I_LONGING | → ST_EMOTIONAL_WEIGHT ↑ | 0.05 | 思念伴随情绪重量 |
| ↓ I_ENERGY | → ST_CLOSENESS ↓ | -0.10 | 疲惫降低社交意愿 |
| ↑ I_STRESS | → ST_CONFLICT ↑ | 0.15 | 高压力易触发对抗（Berkowitz, 1990） |
| ↑ I_STRESS | → ST_CLOSENESS 取决于人格 | ±0.08 | 压力下"趋向 vs 回避"人格分化 |
| ↑ I_IRRITATION | → ST_CONFLICT ↑ | 0.20 | 烦躁积累→冲突倾向 |
| ↑ I_INSECURITY | → ST_ABANDONMENT ↑ | 0.20 | 不安→被抛弃恐惧 |
| ↑ I_INSECURITY | → ST_VALIDATION ↑ | 0.15 | 不安→寻求确认 |
| ↑ I_SOCIAL_BATTERY | → ST_CLOSENESS ↑ | 0.15 | 电量足→想社交 |
| ↑ I_MENTAL_FATIGUE | → 所有刺激 ↓ | -0.05 | 疲惫降低整体反应度 |
| ↑ R_AFFECTION | → ST_VALIDATION ↑ | 0.15 | 好感→想被认可 |
| ↑ R_AFFECTION | → ST_CLOSENESS ↑ | 0.12 | 好感→想靠近 |
| ↑ R_TRUST_BOND | → ST_CLOSENESS ↑ | 0.10 | 信任→安全的靠近 |
| ↑ R_INTIMACY | → ST_CLOSENESS ↑ | 0.18 | 亲密→更想靠近 |
| ↑ R_INTIMACY | → ST_DEPENDENCY ↑ | 0.10 | 亲密→依赖倾向 |
| ↓ R_AFFECTION | → ST_CONFLICT ↑ | 0.08 | 好感下降→关系张力 |
| S_VULNERABILITY ↑ | → ST_CLOSENESS ↑ | 0.08 | 脆弱时想被抱抱 |
| S_RESTRAINT ↑ | → 所有↓ | -0.05 | 克制时掩盖驱力 |

**注意**：这些是**净积累**——不是每轮都从零开始，而是状态已经存在的偏差的持续效应。例如，如果 I_LONELINESS 已经是 0.6，驱力会持续产生 ST_CLOSENESS，直到孤独下降后自然消退。

**约束合规声明**：
- `W_drive` 是 `(18, 7)` 矩阵，通过 `WeightMapper` 构建，密度目标 ≤25%（服从约束⑥）
- 每条规则通过 `connect(source, target, value, magnitude, domain, rationale, origin, reviewed)` 注册（服从约束⑤⑧）
- 最终用户的 `D_accumulated = state @ W_drive`，与现有 `INPUT_INFLUENCE_B` 模式一致

#### 2.2.3 D_spontaneous — 心智游移模拟

当对话间隔较长（Δt 大），或角色独处时，模拟 DMN 式的心智游移。这使用一个极度**低通滤波**的 Ornstein-Uhlenbeck 过程：

```python
def _ou_step(current_drive, target, theta, sigma, dt):
    """Ornstein-Uhlenbeck process for spontaneous drive fluctuation.
    
    dD = θ·(μ - D)·dt + σ·dW
    
    where:
      θ — 回归速率（慢：0.01-0.05，心智游移变化极缓）
      μ — 目标值（人格基线）
      σ — 波动幅度（小：0.01-0.03）
      dW — Wiener 过程增量
    """
    drift = theta * (target - current_drive) * dt
    diffusion = sigma * np.random.normal(0, np.sqrt(dt))
    return current_drive + drift + diffusion
```

关键参数：
- `θ ≈ 0.02-0.05`：非常慢的回归，一天的驱力方向变化很缓
- `σ ≈ 0.01-0.03`：波动小，不会产生突兀的情绪跳跃
- `μ = D_baseline`：回归目标是人格基线
- 仅在 `Δt > 30` 分钟时激活（太短间隔不触发心智游移）
- **与所有现有约束兼容**：这是纯随机过程，不涉及参数矩阵，不在约束⑤⑧范围内

### 2.3 与现有管线的集成

```
                    ┌─────────────────────────────┐
                    │   Ø Internal Drives          │
                    │   D = D_baseline             │
                    │     + D_accumulated          │
                    │     + D_spontaneous          │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
  user_stimuli(7) ──────────► ⊕ 合并刺激
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   ① Defense Profiles        │
                    │   stim + drives → profiles  │
                    │   → inner, outer            │
                    └─────────────┬───────────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │   ② Residual Dynamics       │
                    │   (使用合并后的 inner)       │
                    └─────────────┬───────────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │   ③ Surface Projection      │
                    │   (使用合并后的 outer)       │
                    └─────────────┬───────────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │   ④ Surface Feedback        │
                    └─────────────────────────────┘
```

### 2.4 时间感知的时间调制

内驱不是恒定的——它受**自上次交互以来的时间**调制：

```
D_effective = D(t) · f(Δt)

where f(Δt) = 1 - exp(-λ_activation · Δt)
```

- `Δt` = 自上次用户交互的小时数
- `λ_activation` ≈ 0.3-1.0 /小时（1-3 小时内驱逐渐达到最大值）
- 当 `Δt → ∞`，`f → 1.0`（内驱满负荷）
- 当 `Δt → 0`，`f → 0`（正在对话时，让外部刺激主导）

**效果**：用户离开 30 分钟回来，角色的孤独感已经积累到了她会主动说"你终于回来了"的程度。但正在对话时，内驱退到后台，不干扰正常的交互驱动。

---

## 三、约束影响分析

### 3.1 新增约束评估

| # | 约束 | 是否受影响 | 处理 |
|---|------|:---------:|------|
| ① | Trait 不直接影响状态 | ⚠️ | D_baseline 使用 traits 计算驱力基线，但不参与状态更新方程主项——与 setpoint 的角色相同，"状态基线偏移"不违反约束① |
| ② | 刺激携带元属性 | ❌ 未受影响 | 内驱生成 7D 向量与 user_stimuli 同类，现有缺陷不变 |
| ③ | 矩阵低秩 | ✅ | W_drive (18×7) 通过 WeightMapper 构建，自然保证逻辑低秩 |
| ④ | 禁止跨层直接连线 | ⚠️ | surface→stimuli 的映射（如 S_VULNERABILITY → ST_CLOSENESS）是"层内自指"还是"跨层"？辩护：这发生在 Step 0（驱力生成层），不是 Step ③ 的主投影路径，且 state→stimuli 映射本身就是驱力的定义——不属于约束④范围 |
| ⑤ | 语义映射层 | ✅ | W_drive 通过 WeightMapper connect() 构建，每行带 rationale |
| ⑥ | 正交稀疏 | ✅ | W_drive 密度目标 ≤25%，即将已有的 18×7=126 参数中约 30 条非零 |
| ⑦ | 谱半径 | ✅ | W_drive 不是状态更新矩阵，不参与动力学更新方程——谱半径约束不适用 |
| ⑧ | 参数审计 | ✅ | 全部通过 connect() 注册 |
| ⑨ | 全局雅可比稀疏 | ✅ | 需要重新计算（增加 18→7 路径），但路径数增加 ≤30 条，总密度预计从 19.9% 升至 ~22% |
| ⑩ | 刺激正交性 | ✅ | 内驱不改变刺激维度含义 |
| ⑪ | 状态格式化连续性 | ❌ 未受影响 | 无关 |

### 3.2 需要关注的风险

1. **正反馈环风险**：loneliness → ST_CLOSENESS → 更新后 loneliness 下降（正交互）→ CLOSENESS 下降。这没问题，是负反馈。但 `anxiety → ST_ABANDONMENT → 更新后 anxiety 上升` 可能形成正反馈。需要**设置上限**：D_accumulated 的上界 ≤0.25（约为最大外部刺激的 1/3）。

2. **驱力与外刺激竞争**：当内驱持续产生 ST_CLOSENESS 且外部刺激也产生 ST_CLOSENESS 时，总和可能超过 1.0。解决方案：在合并时使用 `soft_clamp(total_stimuli, 0, 1)`，确保单维度不上溢。

3. **测试负担**：新增 12+ 测试用例（驱力方向性、积累时间曲线、与外刺激竞争、正反馈安全等）。

---

## 四、实现计划

### Phase 1：核心模块（新增 ~120 行，改动 ~20 行）

| 文件 | 操作 | 内容 |
|------|------|------|
| `state_engine/_drives.py` | 新增 | `compute_internal_drives()` — D_baseline + D_accumulated + D_spontaneous |
| `state_engine/_drives_weights.py` | 新增 | `W_DRIVE` (WeightMapper, 18→7)、`DRIVE_BASELINE_MAPPER` (LinearMapping, 10→7)、`DRIVE_TIME_CONSTANT` (WeightVector) |
| `state_engine/_pipeline.py` | 修改 | Step 0 调用 `compute_internal_drives()` + 合并到 stimuli |
| `state_engine/__init__.py` | 修改 | 导出 `compute_internal_drives` |

### Phase 2：时间感知集成（已部分存在）

| 文件 | 操作 | 内容 |
|------|------|------|
| `state_engine/_decay.py` | 修改 | 导出 `compute_delta_hours()` 给 _drives.py 使用 |
| `state_engine/_drives.py` | 修改 | 集成 Δt → f(Δt) 时间调制曲线 |

### Phase 3：测试（新增 ~150 行）

| 文件 | 操作 | 内容 |
|------|------|------|
| `tests/test_drives.py` | 新增 | 驱力方向性测试（12+ 条 rule 逐条验证） |
| `tests/test_dynamics.py` | 修改 | 增加内驱 + 外部刺激组合场景 |
| `tests/test_pipeline.py` | 修改 | 内驱场景：长时间间隔 → 内驱主导 |

---

## 五、后续扩展方向

### 5.1 后台守护进程（Daemon）

Phase 1 完成后，系统已经在**每次对话时**有内部驱力。要进一步实现"退出后仍在活跃"，需要后台循环：

```
while True:
    load_state()
    apply_time_decay(Δt)
    compute_internal_drives()  # 复用 Phase 1
    update_all(drives)         # 复用现有引擎
    if check_threshold():      # 阈值检测
        push_initiative()      # SSE / WebSocket
    save_state()
    sleep(interval)
```

这与 Phase 1 完全独立——daemon 只是把 Phase 1 的函数在一个循环中调用。见 [ROADMAP.md](ROADMAP.md) §后台进程。

### 5.2 主动意图表达（Initiative Layer）

内驱到达阈值后，角色不仅状态在变化，还会产生"想做什么"的意图信号。这是方案 B 的延续——见 [ROADMAP.md](ROADMAP.md) §主动表达。

### 5.3 驱动记忆自组织

当内驱积累到高强度时，角色应该更倾向于检索相关记忆（孤独时想起过去的温暖时刻/被抛弃时刻）。这需要记忆系统与驱力层耦合——见 [MEMORY_SYSTEM.md](MEMORY_SYSTEM.md)。

---

## 六、未解决的问题

1. **驱力的个性化速度**：不同人格的内驱积累速度是否应该不同（如焦虑型积累更快）？如果需要，`λ_activation` 人格调制就是新的 WeightMapper。

2. **D_spontaneous 的复现性**：随机过程使结果不可完全复现——这对测试是挑战。解决方案：`_drives.py` 接受可选的 `rng` 参数，测试时注入固定种子 `np.random.default_rng(42)`。

3. **与现有 `apply_time_decay` 的衔接**：时间衰减将状态拉向 setpoint，内驱将状态推离 setpoint。两者在长时间间隔下的互动需要数值验证（Monte Carlo 10k 步）。

4. **正反馈安全边界**：什么情况下内驱 + 外刺激 + 动力学耦合 = 震荡？需要谱半径重验 + 10,000 轮长程稳定性测试。

---

## 参考文献

1. Bowlby J. (1969/1982). *Attachment and Loss, Vol. 1: Attachment*. Hogarth Press.
2. Panksepp J. (1998). *Affective Neuroscience: The Foundations of Human and Animal Emotions*. Oxford.
3. Cacioppo J.T. & Hawkley L.C. (2009). "Perceived social isolation and cognition". *Trends in Cognitive Sciences*, 13(10).
4. Raichle M.E. et al. (2001). "A default mode of brain function". *PNAS*, 98(2).
5. Joseph P. & Levkowitz H. (2011). "Patterns of Emotion Driven by Affect State and Environment". *PATTERNS 2011*.
6. Petters D. & Waters E. (2014). "From Internal Working Models to Embodied Working Models". *AISB50*.
7. Berkowitz L. (1990). "On the formation and regulation of anger and aggression". *American Psychologist*, 45(4).
8. Hull C.L. (1943). *Principles of Behavior*. Appleton-Century.
9. Senthilvanan et al. (2025). "Comprehensive Computational Framework for DMN-CEN-SN Dynamics".
10. Craik K. (1943). *The Nature of Explanation*. Cambridge.
