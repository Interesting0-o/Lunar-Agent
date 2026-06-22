# 情感几何学研究：软性双曲阻尼 × 信息瓶颈 × 双速度 SSM

> 2026-06-19 | 探索 Lunar 状态引擎的三个底层数学扩展方向，
> 解决高维扁平心理空间的维度灾难问题。
>
> **整合说明**：本文档合并了 AFFECTIVE_GEOMETRY_RESEARCH.md（三层数学扩展研究）+ DUAL_TIMESCALE_SSM.md（双速 SSM 深度分析报告）。DUAL_TIMESCALE_SSM 是第三扩展方向（双速度 SSM）的详细展开，作为本文的深度附录。

---

## 摘要

Lunar 的状态引擎在 8（内部）+ 6（关系）+ 7（表面）+ 10（特质）= **31 维联合状态空间**中运行。Del Giudice (2023) 证明，当心理表征空间的维度超过 ~10 时，**距离集中效应**导致所有点趋于等距——"每个人都同样与众不同"，情绪指纹失去区分度。

本文提出三层数学框架解决此问题：

1. **软性双曲阻尼**（Soft Hyperbolic Damping）：用人格调制的 `tanh` 参数化边界，替代全局 `transition`
2. **信息瓶颈（IB）情绪乘性调制**：将防御剖面重新形式化为 IB Lagrangian，`deact/hyper` 对应压缩/保真权衡
3. **快慢双速度 SSM**：显式联合 14 维状态空间模型，谱半径分离创造内在时间结构

三者构成：**IB 压低有效维度 → SSM 分离时间尺度 → 双曲阻尼保持有界流形**

---

## 第一部分：三层数学框架

### 一、软性双曲阻尼

#### 1.1 形式化定义

当前 `soft_clamp` 在 `_utils.py`：

```python
def soft_clamp(x, low=-1, high=1, transition=0.1):
    upper = high + transition * np.tanh((x - high) / transition)
    lower = low - transition * np.tanh((low - x) / transition)
```

核心参数 `transition` 控制边界刚度：
- transition → 0：趋近硬裁切 (hard clip)
- transition → ∞：趋近线性恒等 (无约束)

#### 1.2 学术背景

**CD-NODE γ**（UNSW 2022）提出用 sigmoid 函数作为神经 ODE 的 γ 约束。Lunar 的 `soft_clamp` 是同构方法，区别在于使用 `tanh`——提供对称的 ± 渐近线，更适合 [-1, 1] 值域。

**IARIA 2016** 直接用 `A(t) = -A₀·th(γ·dZ/dt)` 作为阻尼函数——这里的 `th` 不是边界约束，而是**跨系统信息流的乘性闸门**。

#### 1.3 核心性质

| 性质 | 数学表达 | 意义 |
|------|---------|------|
| 单调性 | d/dx·sc(x) > 0 | 状态大小顺序保持 |
| C¹ 连续 | 分段一阶可导 | 梯度兼容未来学习 |
| 渐近线性 | lim_{x→±∞} sc'(x) = 0 | 抑制极端离群值 |
| 对称性 | sc(-x; -τ) = -sc(x; τ) | 正负情感对称 |

#### 1.4 扩展：维度级人格调制阻尼

```python
def compute_transition(traits, dim):
    base = 0.10
    base += traits[T_ANXIETY_PRONENESS] * 0.05  # 高焦虑→边界更"硬"
    base -= traits[T_EMOTIONAL_STABILITY] * 0.03  # 高稳定→边界更"软"
    if dim in {I_STRESS, I_IRRITATION}:
        base += 0.02  # 压力维度天生更易触界
    return soft_clamp(base, 0.02, 0.30)
```

**心理学依据**：Kasdorp et al. (2023) 发现高神经质个体的情绪恢复曲线呈现"硬边界"特征。

---

### 二、信息瓶颈（IB）与情绪乘性调制

#### 2.1 IB 原理

信息瓶颈（Tishby, Pereira & Bialek, 1999）寻找一个压缩表示 Z，在最大化与任务 Y 的相关性的同时最小化与输入 X 的互信息：

$$
\mathcal{L} = I(X; Z) - \beta \cdot I(Z; Y)
$$

β 控制压缩-保真权衡：β→0 极端压缩，β→∞ 保真优先。

#### 2.2 防御剖面 = IB 实例

当前 `apply_defenses()` 已经隐含 IB 结构：

| IB 组件 | Lunar 对应 | 数学形式 |
|---------|-----------|---------|
| 输入 X | stimuli (7 维) | — |
| 瓶颈 Z | inner_stimuli (7 维) | Z = X ⊙ (1 + α_H H) |
| 压缩 | deactivation 乘性抑制 | Z_out = Z ⊙ (1 - α_D D) |
| 保真 | hyperactivation 乘性放大 | J_hyper = E[‖X ⊙ H‖²] |
| 权衡 β | profiles[0] vs profiles[1] 相对强度 | β_eff = H̄ / D̄ |

**形式化 IB Lagrangian：**

$$
\mathcal{L}_{defense}(H, D) = \underbrace{\mathbb{E}[\|X \odot (1 - D)\|^2]}_{压缩项} - \underbrace{\beta \cdot \mathbb{E}[\|X \odot (1 + H)\|^2]}_{保真项}
$$

D 最小化通过的信息量，H 最大化情感保真度。

#### 2.3 乘性调制的数学优势

| 特性 | 乘性（当前） | 加性（旧架构） |
|------|------------|--------------|
| 信息流控制 | 增益 [0, 1+α] | 偏置 [-α, +α] |
| 零输入行为 | 无刺激时无信息 | 偏置产生"幽灵信息" |
| 非线性交互 | ∂²Z/∂X∂H = 1 | ∂²Z/∂X∂W = 0 |
| AND 门特性 | 需 X 和 (1+H) 同时非零 | 加性可绕过零输入 |

#### 2.4 扩展：自适应 IB 系数

```python
alpha_hyper = 0.30 + traits[T_ATTACHMENT_ANXIETY] * 0.20   # [0.1, 0.5]
alpha_deact = 0.50 + traits[T_ATTACHMENT_AVOIDANCE] * 0.20  # [0.3, 0.7]
inner = stimuli * (1.0 + hyper * alpha_hyper)
outer = stimuli * (1.0 + hyper * alpha_hyper) * (1.0 - deact * alpha_deact)
```

这等价于每维变分 IB：不同刺激维度的 β_eff 随人格动态调整。

---

### 三、维度灾难：统一框架

#### 3.1 问题诊断

**距离集中效应**（Del Giudice, 2023）：在高维心理空间（d > 10）中，大多数点远离质心，所有点之间的欧氏距离趋近相等。

**有效状态空间的甄别：**

| 层 | 维数 | 是否独立状态？ | 理由 |
|---|:----:|:-------------:|------|
| InternalState | 8 | ✅ 主状态变量 | 每轮随刺激变化 |
| RelationshipState | 3 | ✅ 主状态变量（慢尺度） | 跨轮累积 |
| SurfaceState | 7 | **🔄 混合状态（06-22 重构）** | **有惯性 `s(t)=α·raw+(1-α)·s(t-1)`，产生反馈到 internal** |
| Traits | 10 | ❌ 会话内常量 | 不随状态变化（待演化） |
| StimulusVector | 7 | ❌ 外部输入 | 驱动而非记忆 |

**有效状态空间维度 = 14 维（8 内部 + 6 关系）**。进一步考虑双时间尺度分离，单次对话窗口内实际有效自由度 ≈ 内部 8 维。

即使在 8-14 维范围内，距离集中效应已开始显现：
- 14 维均匀分布两点间平均距离 ≈ 3.06，标准差 ≈ 0.19
- 8 维时 CV ≈ 0.045，已低于 0.10 的理想阈值

#### 3.2 三层解法

```
问题：31 维扁平空间 → 距离集中 → 情绪指纹不可区分
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
      IB 瓶颈        双速度 SSM     双曲阻尼
    压到 ~5 维     谱半径分离      边界软约束
      (压缩)       (时间结构)      (流形保持)
            │             │             │
            └─────────────┼─────────────┘
                          ▼
                   有效区分度恢复
```

**第一层：IB 瓶颈**——防御剖面的乘性调制从 31 维中提取仅 ~3-5 有效自由度。

**第二层：双速度 SSM 的时间结构**——快慢分离创造非各向同性的状态空间。由于 λ_I ≫ λ_R，联合状态的协方差矩阵具有分层的特征值谱。

**第三层：双曲阻尼的流形保持**——通过维度级的人格调制 transition，创造非均匀的边界度量，进一步破坏距离集中所需的各向同性。

#### 3.3 定量预期

| 指标 | 当前（14 维均匀） | 期望（IB+SSM+阻尼） |
|------|:---------------:|:-----------------:|
| 有效自由度 | ~14 | **~4-5** |
| 距离变异系数 (CV) | ~0.06 | **> 0.15** |
| 距离分位数比 (Q95/Q5) | ~1.10 | **> 1.6** |
| 余弦相似度方差 | ~0.002 | **> 0.02** |

---

### 四、实施路线图（第一阶段）

**Phase 1：双曲阻尼人格化（1 天）**
- `soft_clamp` 增加 transition 人格调制参数，每维独立
- 验证：边界触顶率由当前 5.21% 降至 < 1%

**Phase 2：IB 系数自适应（2 天）**
- `apply_defenses` 中 0.50/0.70 替换为人格调制变量
- 验证：β 有效变动从 0.289 → > 0.40

**Phase 3：跨尺度耦合（3 天）**
- 在 `_dynamics.py` 中增加 4-6 条跨尺度耦合规则
- 验证：最大化 Monte Carlo 距离 CV > 0.30

---

## 第二部分：双速 SSM 深度分析

> 基于 Sentipolis / SALM / MATE / Affective Ising Model / OU 过程等完整 PDF 论文的深度分析，
> 是第一部分"快慢双速度 SSM"方向的详细展开。

### 5.1 痛点：扁平状态空间的维度灾难

Lunar 当前的状态引擎采用扁平残差动力学：

$$
h_t = h_{t-1} + dt · (α·Δ_{coupling} + Δ_{stimulus})
$$

所有状态维度在同一时间尺度上更新，带来几个无法回避的问题：

| 问题 | 表现 |
|------|------|
| 瞬时刺激 vs 长期依恋 共用同一通道 | 无法区分"当下心跳加速"和"长期暧昧感" |
| 无天然低通滤波 | 关系状态对单轮刺激的响应度与内部状态在同一量级 |
| 人格基线只通过时间衰减介入 | 每轮对话中 personality setpoint 对动力学无锚定作用 |
| 维度灾难 | 14 维扁平向量中语义边界模糊 |

**核心矛盾：** 无法在应对即时对话刺激的同时，维持一个长周期的、战略性的心理规划路径。

### 5.2 学术框架综述

#### 5.2.1 Sentipolis — 双速情感动力学 (Fu et al., 2026)

**论文：** [Sentipolis: Emotion-Aware Agents for Social Simulations](https://arxiv.org/abs/2601.18027)

Sentipolis 是直接解决"情感健忘症"的框架：

**双速情感动力学：**
- **快速推理（Fast inference）：** 每轮对话后即时情感更新，基于 LLM 评估对 PAD 各维度的 delta
- **慢速推理（Slow inference）：** 融入反思机制，累积 poignancy score 超过阈值（150）时触发反思
- 两层加法耦合：快速层提供瞬时移位，慢速层在反思后调整基线

**PAD 语义富集：** 使用 k-NN 在真实人类 PAD 标注数据上找到最近邻，映射到 Plutchik 情感标签。

**关键发现：** 双速动力学使情感连续性提升超过 2 倍；LLM 容量越大提升越显著；产生了互惠的、适度聚类的、时间上稳定的关系结构。

#### 5.2.2 SALM — 有界人格漂移定理 (Koley, 2025)

**论文：** [SALM: A Multi-Agent Framework for Language Model-Driven Social Network Simulation](https://arxiv.org/abs/2505.09081)

**有界人格漂移定理（Bounded Personality Drift）：**

$$ \|p_{t+k} - p_t\| \leq 0.08 \log(k) + 0.12 $$

**含义：** 人格漂移以对数速率增长——k 从 1 到 4000，log(k) 仅从 0 到 ~8.3，保证智能体在数千轮后仍保持 0.87+ 的人格稳定性。

**情感动力学：** $E_{t+1} = (1-\delta)E_t + \alpha I_t + \beta C_t$

**理论证明：** 基于学习率 η_t = α/(t+τ) 递减和 Lipschitz 连续的梯度，通过对数积分得到 $$\|p_{t+k} - p_t\| \leq \alpha L \log(k) + \beta$$。

#### 5.2.3 MATE — 确定性情感中间件 (Lobozov, 2026)

MATE 是一个纯函数式的情感内核——`transition(state, event) → new_state`——零 LLM 调用，完全可复现。

**核心模块：**

| 模块 | 理论来源 | 功能 |
|------|---------|------|
| 量子概率 | Busemeyer & Bruza 量子认知 | 8×8 密度矩阵，非对易情感叠加 |
| 双过程习惯化 | Thompson & Spencer | 重复刺激响应递减 |
| 30 特质性格 | Bowlby/Young/Vaillant/Bandura | 经验驱动的人格成长 |
| 7 维记忆图 | Damasio 躯体标记 | 带情感标签的图结构记忆 |

**Ornstein-Uhlenbeck 情绪回归：** $dX(t) = \theta(\mu - X(t))dt + \sigma dW(t)$，同时实现对立过程理论——每个情感尖峰触发延迟的反向摆动，B-process 衰减比 A-process 慢 4 倍。

**密度矩阵形式化：** 使用 8×8 密度矩阵 ρ（Tr(ρ)=1，正定 Hermitian），非对易顺序效应产生不同的事件顺序 → 不同情感结果。

**跨尺度类比 Lunar：**

| MATE 概念 | Lunar 对应 | 差异 |
|-----------|-----------|------|
| PAD 情感状态 (3 维) | InternalState (8 维) | MATE 更紧凑 |
| 30 特质性格系统 | Traits (10 维) | MATE 更丰富 |
| O-U 情绪回归 | _decay.py | 同构 |
| 对立过程 | — | Lunar 未实现 |
| 密度矩阵形式化 | — | Lunar 使用古典向量 |

#### 5.2.4 Ornstein-Uhlenbeck 过程 (Oravecz et al., 2009-2011)

情感动力学的标准数学工具：$dX(t) = \theta(\mu - X(t))dt + \sigma dW(t)$

**实证发现：** 唤醒度比效价有更快的动力学；典型 θ 范围 0.1-0.3；神经质 → 更长的效价回归时间。**情感时间常数 τ 是人格差异的可量化指标。**

#### 5.2.5 情感伊辛模型 — Affective Ising Model (Loossens et al., 2020-2024)

OU 模型只有单一线性吸引子。AIM 扩展为非线性多吸引子景观：情感表面 = 每个个体特有的"丘陵地貌"；多稳态允许状态跳跃；外部事件倾斜景观而非简单加法。

### 5.3 快慢层解耦数学框架设计

#### 5.3.1 状态空间分解

| 层级 | 维度 | 更新频率 | 语义 | 示例 |
|------|------|---------|------|------|
| **快速层** E_fast | 2-3 维 (P, A) | **每轮对话** | 即时情感心境 | Pleasure（愉悦-不悦）、Arousal（兴奋-平静） |
| **慢速层** R_slow | 6 维 | **低通滤波累积** | 全局关系与依恋 | Trust（信任）、Romantic Tension（浪漫张力） |

#### 5.3.2 快速层动力学

$$ E_{t+1}^{\text{fast}} = (1 - \delta_e)E_t^{\text{fast}} + \alpha_e S_t + \beta_e (R_t^{\text{slow}} - \mu_e) + \sigma_e \xi_t $$

关键设计：慢速层 R_slow 作为快速层的强制回归基线（attractor）。

#### 5.3.3 慢速层动力学

$$ R_{t+1}^{\text{slow}} = R_t^{\text{slow}} + \eta \cdot f\!\left(\sum_{i=t-k}^{t} \Delta E_i^{\text{fast}}\right) - \gamma (R_t^{\text{slow}} - \mu_r) $$

低通滤波函数：$f(\Sigma) = \tanh(\lambda \cdot \|\Sigma\|) \cdot \frac{\Sigma}{\|\Sigma\|}$，只有 k 轮内的累积能量超过阈值时才会漂移。

#### 5.3.4 层间耦合

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
1. **快→慢：** E_fast 的累积能量（积分）推动 R_slow 缓慢漂移
2. **慢→快：** R_slow 作为 E_fast 的 OU 回归基线 μ

#### 5.3.5 对数收敛约束

借鉴 SALM 的 Bounded Personality Drift 定理：

$$ \|R_{t+k} - R_t\| \leq \gamma \log(k) + \epsilon $$

当慢速层达到边界时，系统需要触发"关键事件"才能进一步漂移——这自然产生了叙事弧。

#### 5.3.6 与现有架构的对应

| 现有组件 | 映射到双速框架 |
|---------|-------------|
| InternalState (8,) → **快速层** | 保留 stress/loneliness/insecurity/longing 等 |
| RelationshipState (6,) → **慢速层** | 更新频率降为低通滤波累积 |
| compute_setpoint() | → 慢速层的 μ_r（人格基线） |
| compute_defense_profiles() | → 快速层的 α_e 调制 |
| _decay.py | → 慢速层的 γ 项 |
| SurfaceState (7,) | → 从 E_fast 和 R_slow 联合投影 |

### 5.4 与 Lunar 的整合路径

**Phase 1 — 状态空间分解（最小改动）**

```
当前:            双速（Phase 1）:
internal (8,)    e_fast (3,)  ← Pleasure, Arousal, Dominance
relationship (6,) internal (5,) ← 去掉 P/A（被 E_fast 吸收）
                 relationship (6,) ← 不变，但更新频率降为低通
```

**Phase 2 — 双时间常数动力学**

| 层级 | dt (对话级别) | 典型 τ | 功能 |
|------|-------------|--------|------|
| E_fast | 1.0（每轮更新） | 2-5 轮 | 即时情感反应 + Brownian 衰减 |
| R_slow | 低通滤波累积 | 50-200 轮 | 依恋关系 + 人格锚定 |

**Phase 3 — 对数收敛约束实现**

```python
def apply_drift_bound(R_new, R_prev, step_count, gamma=0.08, epsilon=0.12):
    max_drift = gamma * np.log(max(step_count, 1)) + epsilon
    drift = np.linalg.norm(R_new - R_prev)
    if drift > max_drift:
        R_new = R_prev + (R_new - R_prev) * (max_drift / drift)
    return R_new
```

**现有工程兼容性：** `_dynamics.py` 重写为两个独立更新函数；`_pipeline.py` 调整调用顺序；`state.py` 新增 E_fast 维度常量；`_matrices.py` 新增 fast→slow 累积投影矩阵。

### 5.5 结论

双速情感动力学为 Lunar 的维度冗余问题提供了一个**结构性解**而非参数调优解：

- **当前**：所有维度共享同一时间常数，关系维度的人为压制导致响应迟钝
- **双速**：快速层（P/A）负责即时情感反应——响应灵敏；慢速层（关系 6 维）通过低通滤波累积——天然迟钝，不需要人为压制

SALM 的对数收敛约束进一步从数学上保证了长期人格一致性。

---

## 参考文献

### 第一部分参考文献

[1] CD-NODE γ — Constrained Dynamics Neural ODE for Emotion Prediction. UNSW PhD Thesis, 2022.
[2] EmotiGOV: A Computational Model of Emotional Dynamics. *International Journal on Life Science and Technologies*, 2016.
[3] Tishby, N., Pereira, F. C., & Bialek, W. (1999). The information bottleneck method. *arXiv:physics/0004057*.
[4] Mittal, T., et al. (2020). M3ER: Multiplicative Multimodal Emotion Recognition. *AAAI 2020*.
[5] Yang et al. (2026). STHMA: Decoupling Spatio-Temporal Dynamics in EEG via Hybrid State Space Modeling. *Brain Sciences*, 16(3), 267.
[6] Bringmann, L. F., et al. (2023). Characterizing affect dynamics with a damped linear oscillator model. *Psychological Methods*.
[7] Steyer, R., et al. (1999). Latent State-Trait Theory. In *Personality Psychology in Europe*, 7, 169–183.
[8] Del Giudice, M. (2023). Individual and Group Differences in multivariate domains: The Curse of Dimensionality. *Personality and Individual Differences*.
[9] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
[10] Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752).

### 第二部分参考文献

| 框架 | 论文 | 核心贡献 | 链接 |
|------|------|---------|------|
| MATE | Lobozov, 2026 | 确定性情感中间件 + 密度矩阵 + O-U + 对立过程 | — |
| Sentipolis | Fu et al., 2026 | 双速情感动力学 + PAD + 情感-记忆耦合 | [arXiv:2601.18027](https://arxiv.org/abs/2601.18027) |
| SALM | Koley, 2025 | 有界人格漂移定理 + 对数收敛约束 | [arXiv:2505.09081](https://arxiv.org/abs/2505.09081) |
| OU Model | Oravecz et al., 2009-2011 | 层次化 OU 过程建模核心情感 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/36107656/) |
| Affective Ising Model | Loossens et al., 2020-2024 | 非线性多吸引子情感景观 | [KU Leuven](https://ppw.kuleuven.be/okp/_pdf/Vanhasbroeck2022SDACE.pdf) |
| Dual-loop RL+LLM | arXiv 2512.20629, 2025 | 快速行为环 + 慢速语言环架构 | [arXiv:2512.20629](https://arxiv.org/abs/2512.20629) |
| Circumplex Model | Russell, 1980 | 情感环状模型（效价-唤醒度二维正交基底） | — |
| Arch. Prerequisites | Broughton & Ciacciarella, 2026 | Affective Residue + Contextual Decay | — |
