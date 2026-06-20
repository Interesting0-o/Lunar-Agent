# 双速情感动力学 (Dual-Timescale SSM) 研究报告

> 2026-06-20 | 基于 Sentipolis / SALM / MATE / Affective Ising Model / OU 过程 等完整 PDF 论文的深度分析

---

## 一、痛点：扁平状态空间的维度灾难

Lunar 当前的状态引擎采用**扁平残差动力学**：

```
h_t = h_{t-1} + dt · (α·Δ_coupling + Δ_stimulus)
```

所有状态维度（内部 8 维 + 关系 6 维）在**同一时间尺度**上更新。这带来了几个无法回避的问题：

| 问题 | 表现 |
|------|------|
| **瞬时刺激 vs 长期依恋 共用同一通道** | 一句调戏和一段关系积累都通过同样的残差公式作用于 romantic_tension，无法区分"当下心跳加速"和"长期暧昧感" |
| **无天然低通滤波** | 关系状态（本应缓慢变化）对单轮刺激的响应度与内部状态在同一量级，需要通过手动调小 α_rel / β_rel 来人为压制 |
| **人格基线只通过时间衰减介入** | 每轮对话中 personality setpoint 对动力学无锚定作用（见 SELF_DECAY 隐性税报告），长周期一致性无保证 |
| **维度灾难** | 14 维扁平向量中，每维既承载快速情绪又承载慢速特质，语义边界模糊 |

**核心矛盾：** 无法在应对即时对话刺激的同时，维持一个长周期的、战略性的心理规划路径。

---

## 二、学术框架综述

### 2.1 Sentipolis — 双速情感动力学 (Fu et al., 2026)

**论文：** [Sentipolis: Emotion-Aware Agents for Social Simulations](https://arxiv.org/abs/2601.18027)

Sentipolis 是直接解决"情感健忘症"（emotional amnesia）的框架——LLM 智能体的情感如果不建模为持久状态，就只是短暂提示。其核心贡献：

**双速情感动力学（Dual-Speed Emotion Dynamics）：**
- **快速推理（Fast inference）：** 每轮对话后，对双方参与者均触发即时情感更新。基于 LLM 评估对话事件对 PAD 各维度的 delta 值
- **慢速推理（Slow inference）：** 融入 Generative Agents 的反思机制。当累积"poignancy score"超过阈值（默认 150）时触发反思，检索近期的 30 条重要记忆，由 LLM 生成高层级洞察（insight），最终计算 PAD delta 并更新**基线 PAD**
- 两层是加法耦合：快速层提供瞬时移位，慢速层在反思后调整基线

**PAD 语义富集（Semantic Enrichment）：**
- 不直接将连续 PAD 值注入 LLM 提示词
- 使用 k-NN 在真实人类 PAD 标注数据（MSP-Podcast 语料库，264 个 Plutchik 标签的 PAD 锚点）上找到最近邻
- 映射到 Plutchik 情感标签 + 人格档案 + 记忆检索 → 生成**生动的自然语言情感段落**注入 prompt
- 这相当于 Lunar 的 state_formatter 层，但更富集且基于真实人类数据

**关键发现：**
- 双速动力学使情感连续性提升超过 2 倍（believability）
- LLM 容量越大，提升越显著；小模型可能被慢速层过度驱动
- 产生了互惠的、适度聚类的、时间上稳定的关系结构

### 2.2 SALM — 有界人格漂移定理 (Koley, 2025)

**论文：** [SALM: A Multi-Agent Framework for Language Model-Driven Social Network Simulation](https://arxiv.org/abs/2505.09081)

SALM 的核心贡献是对人格一致性给出了**形式化数学保证**：

**有界人格漂移定理（Bounded Personality Drift）：**

$$ \|p_{t+k} - p_t\| \leq 0.08 \log(k) + 0.12 $$

其中：
- $p_t$ = 时刻 $t$ 的人格向量
- $\| \cdot \|$ = 欧几里得范数
- $k$ = 交互轮数
- 0.08 = 学习率（上下文敏感因子）
- 0.12 = 基线漂移容差

**含义：** 人格漂移以**对数速率**增长——k 从 1 到 4000，log(k) 仅从 0 增长到 ~8.3，保证智能体在数千轮模拟后仍保持 0.87+ 的人格稳定性。

**情感动力学公式：**

$$ E_{t+1} = (1-\delta)E_t + \alpha I_t + \beta C_t $$

其中 $E_t$ 是 PAD 情感向量，$I_t$ 是交互冲击，$C_t$ 是认知上下文，$\delta$ 是自然衰减率，$\alpha,\beta$ 是影响系数。

**其他成果：**
- 层次化提示（hierarchical prompting）→ 73% token 缩减，4000+ 轮稳定
- 注意力记忆系统 → 80% cache 命中率，95% CI [78%, 82%]，亚线性（9.5%）内存增长
- 行为一致性 0.91 ± 0.03

**有界人格漂移定理的数学证明**（来自附录 A）：

基于人格更新规则 $p_{t+1} = p_t + \eta_t \nabla L(p_t, c_t)$，其中 $\eta_t$ 是递减学习率 $\eta_t = \frac{\alpha}{t+\tau}$，$\nabla L$ 满足 Lipschitz 连续 $\|\nabla L(p,c)\| \leq L$：

$$\|p_{t+k} - p_t\| \leq \sum_{i=0}^{k-1} \eta_{t+i}L = \alpha L \sum_{i=0}^{k-1} \frac{1}{t+\tau+i} \leq \alpha L \int_{t+\tau-1}^{t+\tau+k} \frac{dx}{x} = \alpha L \log\left(\frac{t+\tau+k}{t+\tau-1}\right)$$

吸收常数项得到 $\|p_{t+k} - p_t\| \leq \alpha L \log(k) + \beta$，代入校准参数即得 0.08log(k)+0.12。

### 2.3 MATE — 确定性情感中间件 (Lobozov, 2026)

**论文：** [MATE: A Deterministic Affective Middleware for LLM-Based Companions](source/paper_v8.pdf)

MATE 是一个**纯函数式的情感内核**——`transition(state, event) → new_state`——零 LLM 调用，完全可复现。它集成 20+ 心理学理论为 8 个模块：

#### 核心模块

| 模块 | 理论来源 | 功能 |
|------|---------|------|
| **量子概率** | Busemeyer & Bruza 量子认知 | 8×8 密度矩阵，非对易情感叠加 |
| **双过程习惯化** | Thompson & Spencer | 重复刺激响应递减 |
| **30 特质性格** | Bowlby/Young/Vaillant/Bandura | 经验驱动的人格成长 |
| **内在世界** | Zajonc/Berlyne/Festinger | 品味、渴望、内部冲突 |
| **7 维记忆图** | Damasio 躯体标记 | 带情感标签的图结构记忆 |
| **5 轴觉识场** | Global Workspace Theory | 统一情境意识 |
| **稳态情绪调节** | 同态调节 | O-U 过程驱动情绪回归基线 |
| **SPARK 自创生环** | Maturana & Varela | 信念→感知→证据→信念闭环 |

#### 情感动力学

**Ornstein-Uhlenbeck 情绪回归：**
在没有交互时，情绪通过 OU 过程向人格基线回归：

$$ dX(t) = \theta(\mu - X(t))dt + \sigma dW(t) $$

同时实现 Solomon & Corbit 的**对立过程理论（Opponent Process）**：每个情感尖峰触发一个延迟的反向摆动（B-process）。Joy 下降时 sadness 从积累的 joy-B 中提升。B-process 衰减比 A-process 慢 4 倍——产生耐受性和戒断效应。

**双过程习惯化（Dual-Process Habituation）：**
重复刺激 → 响应递减。受 Thompson & Spencer 理论驱动，习惯化率由人格（开放性 O、神经质 N）调制。

#### 密度矩阵（量子概率形式化）

MATE 不使用古典概率向量，而使用 8×8 **密度矩阵** $\rho$（$\text{Tr}(\rho)=1$，正定 Hermitian）：

- **情感叠加：** 对角元 = 经典概率强度（如 joy=0.47），非对角元 = 量子相干性（如 joy↔anticipation=0.34 编码"温暖的期待"）
- **非对易顺序效应：** 事件 "温暖→敌意" 产生 PAD=(-0.07,+0.58,+0.11)，"敌意→温暖" 产生 PAD=(+0.36,+0.38,+0.22)，$\| \Delta \text{PAD} \| = 0.48$——古典字典产生 $\Delta=0$ 因为加法交换
- **退相干即决策：** Lindblad 主方程驱动退相干：
  $$ \frac{d\rho}{dt} = -i[H,\rho] + \sum_k \gamma_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\} \right) $$
  退相干率 $\gamma$ 由人格调制：$\gamma = \gamma_{\text{base}} \times (1 - N \times 0.3) \times (1 - O \times 0.3) \times (1 - r \times 0.2)$，其中 N/OCEAN 神经质，O 开放性，r 反思性

#### 跨尺度类比 Lunar

| MATE 概念 | Lunar 对应 | 差异 |
|-----------|-----------|------|
| PAD 情感状态 (3 维) | InternalState (8 维) | MATE 更紧凑，使用心理学验证的 PAD |
| 30 特质性格系统 | Traits (10 维) | MATE 更丰富，基于临床心理学 |
| 躯体标记 + 7 维图 | MemoryStore | MATE 有情感加权记忆检索 |
| O-U 情绪回归 | `_decay.py` | 同构 |
| 对立过程 | — | Lunar 未实现 |
| 密度矩阵形式化 | — | Lunar 使用古典向量 |

### 2.4 心理学建模基础

#### 2.4.1 Ornstein-Uhlenbeck 过程 (Oravecz et al., 2009-2011)

情感动力学的标准数学工具，描述均数回归（mean-reverting）随机过程：

$$ dX(t) = \theta(\mu - X(t))dt + \sigma dW(t) $$

其中 $\theta = 1/\tau$ 是回归速率，$\tau$ 是**时间常数**，$\mu$ 是**吸引子（基线）**，$\sigma$ 是波动率。

**实证发现：**
- 唤醒度（Arousal）比效价（Valence）有更快的动力学（更大的 $\theta$，更短的 $\tau$）
- 典型 $\theta$ 范围 0.1-0.3，对应 $\tau \approx 3-10$ 时间单位
- 神经质（Neuroticism）→ 更长的效价回归时间（更小的 $\theta$）
- **情感时间常数 $\tau$ 是人格差异的可量化指标**

#### 2.4.2 情感伊辛模型 — Affective Ising Model (Loossens, Vanhasbroeck et al., 2020-2024)

OU 模型只有**单一线性吸引子**（抛物面碗）。AIM 扩展为**非线性多吸引子景观**：

- **情感表面：** 每个个体特有的"丘陵地貌"，山谷 = 稳定情感状态（吸引子）
- **多稳态（Multistability）：** 状态可以在多个吸引子间跳跃（如中性→积极→消极的相变）
- **L 型 PA-NA 关系：** 积极情感和消极情感的非线性耦合
- **外部事件倾斜景观：** 刺激通过改变吸引子地貌来影响情感，而非简单的加法

AIM 使用 SDE 形式，但漂移项是**非线性**的（势函数的梯度），没有解析解，需数值方法（Euler-Maruyama）。

#### 2.4.3 双环架构 — 快速行为环 + 慢速语言环 (arXiv 2512.20629)

与情感动力学无关但构架平行：

- **行为环（快）：** Q-learning 每步更新
- **语言环（慢）：** 反思文本 → 语义嵌入 → 更新潜策略向量
  $$ \text{latent} \leftarrow \text{latent} + \eta \cdot f(\text{reflection\_embedding}, \text{reward}) $$
- 潜向量在 5-10 步内快速适应，10-40 步稳定，**关键事件时出现 >0.6 的尖峰 L2 变化**

---

## 三、快慢层解耦数学框架设计

基于上述学术框架，为 Lunar 设计的双速 SSM（State Space Model）结构：

### 3.1 状态空间分解

将当前的扁平 14 维向量拆分为两个耦合的层级：

| 层级 | 维度 | 更新频率 | 语义 | 示例 |
|------|------|---------|------|------|
| **快速层** $E_{\text{fast}}$ | 2-3 维 (P, A) | **每轮对话** | 即时情感心境 | Pleasure（愉悦-不悦）、Arousal（兴奋-平静） |
| **慢速层** $R_{\text{slow}}$ | 6 维 | **低通滤波累积** | 全局关系与依恋 | Trust（信任）、Romantic Tension（浪漫张力） |

**设计理由：**

- **Pleasure-Arousal** 作为快速层的 2 维正交基底，足以编码绝大多数情感反应（Russell 的 Circumplex 模型验证了 40+ 年）
- **关系状态**天然是慢变量——信任不会因为一句冲突就崩塌，也不会因为一句认可就完全修复
- 分离后，关系状态不再需要人工调小 α_rel / β_rel 来压制瞬时响应

### 3.2 快速层动力学

$$ E_{t+1}^{\text{fast}} = (1 - \delta_e)E_t^{\text{fast}} + \alpha_e S_t + \beta_e (R_t^{\text{slow}} - \mu_e) + \sigma_e \xi_t $$

| 项 | 含义 | 参考值 |
|----|------|--------|
| $(1 - \delta_e)E_t^{\text{fast}}$ | 自然衰减，$\delta_e$ 控制情绪消退速度（Brownian 式）| 0.3-0.5/轮 |
| $\alpha_e S_t$ | 刺激驱动（$\alpha_e$ 逐刺激维度的接受率）| 0.1-0.4 |
| $\beta_e(R_t^{\text{slow}} - \mu_e)$ | **慢速层的引力吸引子**——关系状态作为基线 | 0.05-0.1 |
| $\sigma_e \xi_t$ | 随机波动（情感固有的随机性） | $\sigma_e \approx 0.02$ |

**关键设计：** 慢速层 $R_{\text{slow}}$ 作为快速层的**强制回归基线**（attractor）。高信任的角色在冲突后愉悦度下降，但 $R_{\text{slow}}$ 将其"拉回"高位基线。这相当于 OU 过程中的 $\theta(\mu - X)$，但 $\theta$ 由慢速层状态动态调制。

### 3.3 慢速层动力学

$$ R_{t+1}^{\text{slow}} = R_t^{\text{slow}} + \eta \cdot f\!\left(\sum_{i=t-k}^{t} \Delta E_i^{\text{fast}}\right) - \gamma (R_t^{\text{slow}} - \mu_r) $$

| 项 | 含义 | 参考值 |
|----|------|--------|
| $\eta \cdot f(\sum \Delta E_i^{\text{fast}})$ | **低通滤波累积**——只有持续的情绪冲击才能推动慢速层变化 | $\eta \approx 0.01-0.05$ |
| $\gamma(R_t^{\text{slow}} - \mu_r)$ | 稳态恢复（向人格基线回归） | $\gamma \approx 0.005-0.01$/轮 |

**低通滤波函数 $f$ 的设计：**

$$ f(\Sigma) = \tanh(\lambda \cdot \|\Sigma\|) \cdot \frac{\Sigma}{\|\Sigma\|} $$

- 只有 **k 轮内的累积能量**超过阈值时，慢速层才会漂移
- $\tanh$ 提供饱和——不会因为单一极端事件导致关系状态剧烈波动

### 3.4 层间耦合

```
  刺激 S_t
    │
    ▼
┌─────────────────┐    低通滤波（k 轮累积）    ┌─────────────────┐
│  快速层 E_fast   │ ──────────────────────► │  慢速层 R_slow   │
│  (Pleasure,      │                          │  (Trust,         │
│   Arousal)       │ ◄────────────────────── │   Tension, ...)  │
└─────────────────┘    引力吸引子（基线回归）   └─────────────────┘
```

两层双向耦合：
1. **快→慢：** $E_{\text{fast}}$ 的**累积能量**（积分）推动 $R_{\text{slow}}$ 缓慢漂移
2. **慢→快：** $R_{\text{slow}}$ 作为 $E_{\text{fast}}$ 的 OU 回归基线 $\mu$

### 3.5 对数收敛约束

借鉴 SALM 的 Bounded Personality Drift 定理，对慢速层施加形式化约束：

$$ \|R_{t+k} - R_t\| \leq \gamma \log(k) + \epsilon $$

其中：
- $\gamma$ 是"漂移速率上限"（人格的可塑性系数）
- $\epsilon$ 是基线噪声容差
- $k$ 是交互轮数

**当慢速层达到边界时，系统需要触发"关键事件"才能进一步漂移**——这自然产生了叙事弧：日常对话只引起快速层的波动，只有重大事件（表白、背叛等）才能推动关系状态的实质性变化。

这与 SALM 的 0.08log(k)+0.12 约束同构，但参数化为 Lunar 的角色人格。

### 3.6 与现有架构的对应

| 现有组件 | 映射到双速框架 |
|---------|-------------|
| `InternalState (8,)` → **快速层** | 保留 stress/loneliness/insecurity/longing 等，但剥离 trait setpoint 锚定 |
| `RelationshipState (6,)` → **慢速层** | 保持 6 维结构，但更新频率降低为低通滤波累积 |
| `compute_setpoint()` | → 慢速层的 $\mu_r$（人格基线，几乎不变） |
| `compute_defense_profiles()` | → 快速层的 $\alpha_e$ 调制（防御影响情绪接受率） |
| `_decay.py` | → 慢速层的 $\gamma$ 项（时间衰减回归人格基线） |
| **SurfaceState (7,)** | → 从 $E_{\text{fast}}$ 和 $R_{\text{slow}}$ 联合投影 |

---

## 四、与 Lunar 的整合路径

### Phase 1 — 状态空间分解（最小改动）

```
当前:            双速（Phase 1）:
internal (8,)    e_fast (3,)  ← Pleasure, Arousal, Dominance
relationship (6,) internal (5,) ← 去掉 P/A（被 E_fast 吸收）
                 relationship (6,) ← 不变，但更新频率降为低通
```

快速层的 3 维从原有 internal 的前 3 维初始化，关系维度的 short-term 波动被明确划归到快速层。

### Phase 2 — 双时间常数动力学

为快速层和慢速层设置**独立的时间常数**：

| 层级 | dt (对话级别) | 典型 $\tau$ | 功能 |
|------|-------------|------------|------|
| $E_{\text{fast}}$ | 1.0（每轮更新）| 2-5 轮 | 即时情感反应 + Brownian 衰减 |
| $R_{\text{slow}}$ | 每 N 轮通过低通滤波累积 | 50-200 轮 | 依恋关系 + 人格锚定 |

### Phase 3 — 对数收敛约束实现

在慢速层更新后添加边界检查：

```python
def apply_drift_bound(R_new, R_prev, step_count, gamma=0.08, epsilon=0.12):
    """SALM-inspired bounded personality drift constraint."""
    max_drift = gamma * np.log(max(step_count, 1)) + epsilon
    drift = np.linalg.norm(R_new - R_prev)
    if drift > max_drift:
        R_new = R_prev + (R_new - R_prev) * (max_drift / drift)
    return R_new
```

### 现有工程兼容性

- `_dynamics.py` 需要重写为两个独立的更新函数（fast + slow）
- `_pipeline.py` 的 `update_all()` 需要调整调用顺序（先快后慢）
- `state.py` 需要新增 `E_fast` 维度常量
- `_matrices.py` 需要新增 fast→slow 的累积投影矩阵
- `_decay.py` 的时间衰减逻辑改为只作用于慢速层
- 现有测试需要分层适配（快速层测试 + 慢速层测试 + 耦合测试）

---

## 五、参考文献

| 框架 | 论文 | 核心贡献 | 链接 |
|------|------|---------|------|
| MATE | Lobozov, 2026 | 确定性情感中间件 + 密度矩阵 + O-U + 对立过程 | [source/paper_v8.pdf](source/paper_v8.pdf) |
| Sentipolis | Fu et al., 2026 | 双速情感动力学 + PAD + 情感-记忆耦合 | [source/2601.18027v2.pdf](source/2601.18027v2.pdf) |
| SALM | Koley, 2025 | 有界人格漂移定理 + 对数收敛约束 | [source/2505.09081v2.pdf](source/2505.09081v2.pdf) |
| OU Model | Oravecz et al., 2009-2011 | 层次化 OU 过程建模核心情感 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/36107656/) |
| Affective Ising Model | Loossens, Vanhasbroeck et al., 2020-2024 | 非线性多吸引子情感景观 | [KU Leuven](https://ppw.kuleuven.be/okp/_pdf/Vanhasbroeck2022SDACE.pdf) |
| Dual-loop RL+LLM | arXiv 2512.20629, 2025 | 快速行为环 + 慢速语言环架构 | [arXiv:2512.20629](https://arxiv.org/abs/2512.20629) |
| Dual-Speed Survey | EmergentMind, 2026 | 双速情感动力学范式综述 | [emergentmind.com](https://www.emergentmind.com/topics/dual-speed-emotion-dynamics) |
| Circumplex Model | Russell, 1980 | 情感环状模型（效价-唤醒度二维正交基底） | — |
| EMA Appraisal Theory | Marsella & Gratch, 2009 | 快速评价驱动反应 + 慢速整合反思 | — |
| Arch. Prerequisites | Broughton & Ciacciarella, 2026 | Affective Residue + Contextual Decay | [source/Architectural Prerequisites for Sustainable Relational Intelligence in Large Language Models.pdf](source/Architectural%20Prerequisites%20for%20Sustainable%20Relational%20Intelligence%20in%20Large%20Language%20Models.pdf) |

---

## 六、结论

双速情感动力学为 Lunar 的浪漫张力近乎冻结问题提供了一个**结构性解**而非参数调优解：

- **当前**：所有维度共享同一时间常数，关系维度的人为压制导致响应迟钝
- **双速**：快速层（P/A）负责即时情感反应——响应灵敏，接受刺激直接驱动；慢速层（关系 6 维）通过低通滤波累积——天然迟钝，不需要人为压制

SALM 的对数收敛约束进一步从数学上保证了长期人格一致性，这是当前 Lunar 完全没有的形式化保证。

> **建议优先级：** 若要工程化实施，推荐从 Phase 1 的状态空间分解开始——这是最小破坏性改动，同时为后续双时间常数动力学铺平道路。
