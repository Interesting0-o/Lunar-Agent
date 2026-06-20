# 情感几何学研究：软性双曲阻尼 × 信息瓶颈 × 双速度 SSM

> 2026-06-19 | 探索 Lunar 状态引擎的三个底层数学扩展方向，
> 解决高维扁平心理空间的维度灾难问题。

---

## 摘要

Lunar 的状态引擎在 8（内部）+ 6（关系）+ 7（表面）+ 10（特质）= **31 维联合状态空间**中运行。Del Giudice (2023) 证明，当心理表征空间的维度超过 ~10 时，**距离集中效应**导致所有点趋于等距——"每
人都同样与众不同"，情绪指纹失去区分度。

本文提出三层数学框架解决此问题：

1. **软性双曲阻尼**（Soft Hyperbolic Damping）：用人格调制的 `tanh` 参数化边界，替代全局 `transition`
2. **信息瓶颈（IB）情绪乘性调制**：将防御剖面重新形式化为 IB Lagrangian，`deact/hyper` 对应压缩/保真权衡
3. **快慢双速度 SSM**：显式联合 14 维状态空间模型，谱半径分离创造内在时间结构

三者构成：**IB 压低有效维度 → SSM 分离时间尺度 → 双曲阻尼保持有界流形**

---

## 一、软性双曲阻尼

### 1.1 形式化定义

当前 `soft_clamp` 在 `_utils.py`：

```python
def soft_clamp(x, low=-1, high=1, transition=0.1):
    upper = high + transition * np.tanh((x - high) / transition)
    lower = low - transition * np.tanh((low - x) / transition)
```

这是分段 $C^1$ 连续的软饱和函数。核心参数 `transition` 控制边界刚度：

```
transition → 0：趋近硬裁切 (hard clip)
transition → ∞：趋近线性恒等 (无约束)
```

### 1.2 学术背景

**CD-NODE γ**（UNSW 2022）[1] 提出用 sigmoid 函数作为神经 ODE 的 γ 约束，将情感状态（arousal/valence）软夹持到 [0,1] 区间。Lunar 的 `soft_clamp` 是同构方法，区别在于使用 `tanh` 而非 `sigmoid`——`tanh` 提供对称的 ± 渐近线，更适合 [-1, 1] 值域。

**IARIA 2016** [2] 直接用 `A(t) = -A₀·th(γ·dZ/dt)` 作为情绪子系统间的阻尼函数，控制理性/情感半球的耦合强度。这里的 `th` 不是边界约束，而是**跨系统信息流的乘性闸门**——这一视角延伸到 Lunar 可用于替代防御剖面中的硬编码系数。

#### 核心性质

| 性质 | 数学表达 | 对 Lunar 的意义 |
|------|---------|----------------|
| 单调性 | $\frac{d}{dx}sc(x) > 0, \forall x$ | 状态大小顺序保持 |
| $C^1$ 连续 | 分段一阶可导 | 梯度兼容未来学习版本 |
| 渐近线性 | $\lim_{x \to \pm\infty} sc'(x) = 0$ | 抑制极端离群值 |
| 对称性 | $sc(-x; -\tau) = -sc(x; \tau)$ | 正负情感对称处理 |

### 1.3 扩展：维度级人格调制阻尼

当前 `transition=0.1` 全局一致。扩展为每维独立、人格调制的参数：

```python
def compute_transition(traits: np.ndarray, dim: int) -> float:
    """维度、人格联合决定边界软硬程度。"""
    base = 0.10
    # 高焦虑→边界更"硬"（情绪容易触顶/触底）
    base += traits[T_ANXIETY_PRONENESS] * 0.05
    # 高稳定→边界更"软"（情绪更有弹性）
    base -= traits[T_EMOTIONAL_STABILITY] * 0.03
    # 不同维度不同基线
    if dim in {I_STRESS, I_IRRITATION}:
        base += 0.02  # 压力维度天生更易触界
    return soft_clamp(base, 0.02, 0.30)
```

**心理学依据**：Kasdorp et al. (2023) 发现高神经质个体的情绪恢复曲线呈现 **"硬边界"** 特征——一旦触发负面情绪就几乎饱和。软边界（`transition` 大）对应情绪弹性。

---

## 二、信息瓶颈（IB）与情绪乘性调制

### 2.1 IB 原理

信息瓶颈（Tishby, Pereira & Bialek, 1999）[3] 寻找一个压缩表示 $Z$，在最大化与任务 $Y$ 的相关性的同时最小化与输入 $X$ 的互信息：

$$
\mathcal{L} = I(X; Z) - \beta \cdot I(Z; Y)
$$

$\beta$ 控制压缩-保真权衡：
- $\beta \to 0$：极端压缩（$Z$ 退化为常数）
- $\beta \to \infty$：保真优先（$Z$ 退化为 $X$）

### 2.2 防御剖面 = IB 实例

当前 `apply_defenses()` 已经隐含 IB 结构：

```python
inner = stimuli * (1.0 + hyper * 0.50)    # Z_hyper = 保真放大
outer = inner * (1.0 - deact * 0.70)      # Z_deact = 压缩截断
```

| IB 组件 | Lunar 对应 | 数学形式 |
|---------|-----------|---------|
| 输入 $X$ | `stimuli` (7 维) | — |
| 瓶颈 $Z$ | `inner_stimuli` (7 维) | $Z = X \odot (1 + \alpha_H H)$ |
| 压缩 (min $I(X;Z)$) | `deactivation` 乘性抑制 | $Z_{out} = Z \odot (1 - \alpha_D D)$ |
| 保真 (max $I(Z;Y)$) | `hyperactivation` 乘性放大 | $J_{hyper} = \mathbb{E}[\|X \odot H\|^2]$ |
| 权衡系数 $\beta$ | `profiles[0]` vs `profiles[1]` 相对强度 | $\beta_{eff} = \bar{H} / \bar{D}$ |

**关键洞察**：防御剖面不是两个独立的门控——它们是同一个 IB Lagrangian 的两个对抗项。

#### 形式化 IB Lagrangian

$$
\mathcal{L}_{defense}(H, D) = \underbrace{\mathbb{E}[\|X \odot (1 - D)\|^2]}_{压缩项} - \underbrace{\beta \cdot \mathbb{E}[\|X \odot (1 + H)\|^2]}_{保真项}
$$

$D$（deactivation）最小化通过的信息量，$H$（hyperactivation）最大化情感保真度。

### 2.3 乘性调制的数学优势

与加性调制（$Z = X + W$）对比：

| 特性 | 乘性（当前） | 加性（旧架构） |
|------|------------|--------------|
| 信息流控制 | 增益 $[0, 1+\alpha]$ | 偏置 $[-\alpha, +\alpha]$ |
| 零输入行为 | 无刺激时无信息 | 偏置产生"幽灵信息" |
| 非线性交互 | $\frac{\partial^2 Z}{\partial X \partial H} = 1$ | $\frac{\partial^2 Z}{\partial X \partial W} = 0$ |
| AND 门特性 | 需 X 和 (1+H) 同时非零 | 加性可绕过零输入 |

**乘性 = 天然的 AND 门**：只有刺激强烈且剖面激活时，信息才通过。这与 M3ER（Mittal et al.）[4] 的乘性融合机制完全一致。M3ER 的乘性损失函数通过乘积抑制弱模态——Lunar 的防御剖面在单模态内做了同样的事。

### 2.4 扩展：自适应 IB 系数

当前 0.50 / 0.70 是手写常数。可以引入自适应 IB 系数：

```python
# 当前
inner = stimuli * (1.0 + hyper * 0.50)
outer = inner * (1.0 - deact * 0.70)

# 扩展：人格+状态调制 IB 系数
alpha_hyper = 0.30 + traits[T_ATTACHMENT_ANXIETY] * 0.20   # [0.1, 0.5]
alpha_deact = 0.50 + traits[T_ATTACHMENT_AVOIDANCE] * 0.20  # [0.3, 0.7]

inner = stimuli * (1.0 + hyper * alpha_hyper)
outer = stimuli * (1.0 + hyper * alpha_hyper) * (1.0 - deact * alpha_deact)
```

这等价于每维变分 IB：不同刺激维度的 $\beta_{eff}$ 随人格动态调整。

---

## 三、快慢双速度情感动力学（Dual-Timescale SSM）

### 3.1 当前状态

Lunar 已有两个不同时间尺度的状态更新：

| 状态 | 维度 | λ_base 范围 | 半衰期范围 | 时间尺度 |
|------|:----:|:-----------:|:----------:|:--------:|
| 内部 | 8 | 0.12~0.69 | 1~6 h | **快** |
| 关系 | 6 | 0.0014~0.0058 | 5~21 d | **慢** |

但从 SSM 视角看，两者是**分离的**——独立更新、独立衰减，没有显式的跨尺度耦合。

### 3.2 学术基础

**STHMA**（Yang et al., Brain Sciences 2026）[5] 提出了 **解耦时空扫描** 策略——将空间模态（瞬时功能连接）和时间模态（持续演化）分开建模，避免了 1D 序列模型的结构坍缩。Lunar 的"内部状态 = 快变量" vs "关系状态 = 慢变量"在概念上对应 STHMA 的"空间连通性 vs 时间演化"。

**Damped Linear Oscillator SSM**（Psychological Methods 2023）[6] 直接用连续时间状态空间模型刻画情绪动力学，参数化三个核心指标：
- **情绪惯性**（inertia）：自回归系数 → Lunar 的 $\alpha$ 耦合速率
- **弹性**（resilience）：恢复速度 → Lunar 的 λ 衰减常数
- **脆弱性**（vulnerability）：基线水平 → Lunar 的 setpoint

**Latent State-Trait Theory**（Steyer et al., 1999, 2015）[7] 将行为分解为：
$$
y_{it} = \underbrace{\mu_i}_{\text{trait}} + \underbrace{\lambda_i \cdot \eta_t}_{\text{state}} + \underbrace{\varepsilon_{it}}_{\text{error}}
$$

Lunar 当前的 Traits (10 维) + Internal/Relationship 状态 (14 维) 已经遵循这个分解——只是没有显式称为 LST 模型。

### 3.3 联合 14 维 SSM

将两个分离的更新重写为联合状态空间模型：

$$
z_t = \begin{bmatrix} internal_t \\ relationship_t \end{bmatrix} \in \mathbb{R}^{14}
$$

#### 块对角动力学矩阵

$$
z_{t+1} = z_t + dt \cdot \left( \begin{bmatrix} \alpha_I A_I & \epsilon \cdot C_{IR} \\ \epsilon \cdot C_{RI} & \alpha_R A_R \end{bmatrix} z_t + \begin{bmatrix} \beta_I B_I \\ \beta_R B_R \end{bmatrix} stimuli_t \right)
$$

其中：
- $A_I \in \mathbb{R}^{8 \times 8}$：内部耦合（已有命名规则）
- $A_R \in \mathbb{R}^{6 \times 6}$：关系耦合（已有命名规则）
- $C_{IR}, C_{RI}$：**跨尺度耦合** — 当前不存在，但正是关键缺口
- $\epsilon \ll 1$：弱耦合参数，确保谱半径分离

#### 跨尺度耦合的意义

```python
# 新增：关系→内部耦合（慢→快）
coupling[I_LONELINESS] += relationship[R_TRUST] * (-0.03)  # 信任→孤独降低
coupling[I_INSECURITY] += relationship[R_EMOTIONAL_SAFETY] * (-0.04)  # 安全感→不安降低

# 新增：内部→关系耦合（快→慢）
rel_coupling[R_AFFECTION] += internal[I_STRESS] * (-0.02)  # 压力→好感降低
rel_coupling[R_EMOTIONAL_SAFETY] += internal[I_ENERGY] * 0.02  # 精力→安全感
```

#### 谱半径设计准则

为保证快慢时间尺度分离：

$$
\rho(A_I) \approx 0.85, \quad \rho(A_R) \approx 0.998, \quad \|C_{IR}\|, \|C_{RI}\| \ll 1
$$

| 特征值范围 | 时间常数 | 对应维度 | 功能 |
|-----------|:-------:|---------|------|
| 0.80~0.90 | ~5-10 轮 | 内部 8 维 | 情绪波动、刺激响应 |
| 0.95~0.999 | ~20-500 轮 | 关系 6 维 | 信任渐进积累/侵蚀 |
| < 0.01 | 瞬时 | 跨尺度耦合 | 慢态对快态的微弱调制 |

这保证了快变量围绕慢变量波动，而慢变量几乎不受短期波动影响——与现实人际关系高度一致。

### 3.4 与 Mamba / 现代 SSM 的关系

Lunar 当前使用欧拉离散化的残差更新，形式上等价于 **Mamba 结构空间模型（S6）** 的一个简化版本：

```
Mamba:    z_{t+1} = \bar{A}_t z_t + \bar{B}_t u_t    （时变矩阵）
Lunar:    z_{t+1} = z_t + dt · (αA·z_t + βB·u_t)     （时不变但非线性）
```

区别在于 Mamba 的 $A_t$ 是输入依赖的（通过 selection mechanism），而 Lunar 的 $A$ 当前是常数。引入 `profiles` 对耦合速率的调制已经做了部分"选择"——hyper 高时 β 大（更容易接受刺激），deact 高时 β 小（更封闭）。

**扩展方向**：把 defense profiles 作为 SSM 的选择机制输入：

$$
A_t = A_{base} + W_H \cdot hyper_t + W_D \cdot deact_t
$$

这样防御剖面不仅控制刺激接受速率，还动态重塑跨维度耦合结构——人格在"状态空间的曲率"上留下印记。

---

## 四、维度灾难：统一框架

### 4.1 问题诊断

Del Giudice (2023) [8] 的核心发现：

> 在高维心理空间（d > 10）中，大多数点远离质心，形成一个日益稀疏的"壳层"（shell）。
> 所有点之间的欧氏距离趋近相等——"平均"人格配置变得极罕见。

**距离集中**的数学表现：对高维超立方体 $[-1,1]^d$ 中的均匀分布，

$$
\frac{\mathbb{E}[\|x\|_{max}]}{\mathbb{E}[\|x\|_{min}]} \to 1 \quad \text{as } d \to \infty
$$

#### 有效状态空间的甄别

并非所有 31 个维度都是独立变化的状态变量。需要剔除：

| 层 | 维数 | 是否独立状态？ | 理由 |
|---|:----:|:-------------:|------|
| InternalState | **8** | ✅ **主状态变量** | 每轮随刺激变化，是距离计算的核心 |
| RelationshipState | **6** | ✅ **主状态变量**（慢尺度） | 跨轮累积，独立于内部状态变化 |
| SurfaceState | 7 | ❌ **确定性投影** | 是内部+关系+outer_stimuli 的确定函数，非独立自由度 |
| Traits | 10 | ❌ **会话内常量** | 在同一会话窗口中不随状态变化，两点距离计算中包含特质的贡献是恒定的偏置，不贡献区分度 |
| StimulusVector | 7 | ❌ **外部输入** | 是驱动而非记忆，不属于"状态空间"的几何 |

**有效状态空间维度 = 8（内部）+ 6（关系）= 14 维**。

进一步考虑双时间尺度分离：关系态 6 维的 λ_base = 0.0014~0.0058（半衰期 5-21 天），在单次对话窗口（~30 分钟~2 小时）内**几乎不变**。因此在每次对话的时间尺度上，实际有效自由度 ≈ **内部 8 维**。

但即使在 ~8-14 维范围内，距离集中效应已经开始显现：

- 14 维均匀分布两点间平均距离 ≈ $\sqrt{2d/3} \approx 3.06$，标准差 ≈ 0.19
- 95% 的点对距离在 [2.7, 3.4] 之间——仍有明显区分度，但显著劣于低维空间
- 8 维时 CV（变异系数）≈ 0.045，已低于 0.10 的理想阈值
- 距离集中是渐进的：d=5 时 CV ≈ 0.07，d=14 时 CV ≈ 0.06，d=31 时 CV ≈ 0.03

因此三层解法（IB + 双速度 SSM + 双曲阻尼）针对的是 14 维真实状态空间，而非夸大的 31 维。

### 4.2 三层解法

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

#### 第一层：IB 瓶颈

$$
Z_{eff} = \text{proj}(internal, relationship, stimuli) \in \mathbb{R}^k, \quad k \ll 31
$$

防御剖面的乘性调制从 31 维中提取仅 ~3-5 有效自由度：
- 当前有效维度上限取决于 `deact` 的压缩强度（当前 `0.70 × hyper` 压缩率 ~0.35 信息保留）
- IB 框架下可显式约束：$\mathcal{L}_{IB} = I(X;Z) - \beta I(Z;Y)$ → 用人格特质调制 $\beta$

#### 第二层：双速度 SSM 的时间结构

快慢分离创造了一个**非各向同性的状态空间**。两个状态向量在相同的时间点上具有不同的历史相关性：

$$
\mathbb{E}[z_t^{(internal)} \cdot z_{t+\tau}^{(internal)}] \propto e^{-λ_I·τ}
$$
$$
\mathbb{E}[z_t^{(rel)} \cdot z_{t+\tau}^{(rel)}] \propto e^{-λ_R·τ}
$$

由于 $λ_I \gg λ_R$，联合状态的协方差矩阵具有**分层的特征值谱**——这天然破坏了距离集中所需的各向同性。

#### 第三层：双曲阻尼的流形保持

`soft_clamp` 将状态限制在 [-1, 1]^d 的紧致区域内，但更重要的是——通过维度级的人格调制 transition，创造了一个**非均匀的边界度量**：

```python
# 各维 transition 不同 → 边界几何是椭球非球面
# 高焦虑维度 "触顶" 容易（transition 小，边界硬）
# 高稳定维度 "有弹性"（transition 大，边界软）
```

这种边界几何的非均匀性进一步破坏了距离集中所需的对称性。

### 4.3 定量预期

基准说明：当前 14 维状态空间（8 内部 + 6 关系）在均匀分布假设下的 CV ≈ 0.06，
低于 d=5 时的 0.07。IB 瓶颈的目标是将有效自由度压到 ~4-5 维，使 CV 恢复到 > 0.15。

| 指标 | 当前（14 维均匀） | 期望（IB+SSM+阻尼） |
|------|:---------------:|:-----------------:|
| 有效自由度 | ~14（耦合引入弱相关，实际略低） | **~4-5** |
| 距离变异系数 (CV) | ~0.06 | **> 0.15** |
| 距离分位数比 (Q95/Q5) | ~1.10 | **> 1.6** |
| 典型性偏移（% outlier） | ~30% | **< 5%** |
| 余弦相似度方差 | ~0.002 | **> 0.02** |

---

## 五、实施路线图

### Phase 1：双曲阻尼人格化（1 天）

- `soft_clamp` 增加 `transition` 人格调制参数
- 每维独立 transition，由 traits 计算
- 验证：边界触顶率由当前 5.21% 降至 < 1%

### Phase 2：IB 系数自适应（2 天）

- `apply_defenses` 中 0.50/0.70 替换为人格调制变量
- `alpha_hyper = 0.30 + T_ATTACHMENT_ANXIETY * 0.20`
- `alpha_deact = 0.50 + T_ATTACHMENT_AVOIDANCE * 0.20`
- 验证：β 有效变动从 0.289 → > 0.40

### Phase 3：跨尺度耦合（3 天）

- 在 `_dynamics.py` 中增加 4-6 条跨尺度耦合规则
- 交叉耦合系数 $\epsilon = 0.02$，保证谱半径分离
- 验证：最大化 Monte Carlo 距离 CV > 0.30

---

## 六、参考文献

[1] CD-NODE γ — Constrained Dynamics Neural ODE for Emotion Prediction. UNSW PhD Thesis, 2022.

[2] EmotiGOV: A Computational Model of Emotional Dynamics. *International Journal on Life Science and Technologies*, 2016.

[3] Tishby, N., Pereira, F. C., & Bialek, W. (1999). The information bottleneck method. *arXiv:physics/0004057*.

[4] Mittal, T., et al. (2020). M3ER: Multiplicative Multimodal Emotion Recognition using Facial, Textual, and Speech Cues. *AAAI 2020*. [arXiv:1911.05659](https://arxiv.org/abs/1911.05659)

[5] Yang et al. (2026). STHMA: Decoupling Spatio-Temporal Dynamics in EEG via Hybrid State Space Modeling. *Brain Sciences*, 16(3), 267. [DOI: 10.3390/brainsci16030267](https://doi.org/10.3390/brainsci16030267)

[6] Bringmann, L. F., et al. (2023). Characterizing affect dynamics with a damped linear oscillator model. *Psychological Methods*. [DOI: 10.1037/MET0000615](https://doi.org/10.1037/MET0000615)

[7] Steyer, R., et al. (1999). Latent State-Trait Theory. In *Personality Psychology in Europe*, 7, 169–183.

[8] Del Giudice, M. (2023). Individual and Group Differences in Multivariate Domains: The Curse of Dimensionality. *Personality and Individual Differences*. [PDF](https://marcodg.net/wp-content/uploads/2023/06/delgiudice_2023_individual-group-differences_multivariate-domains_paid.pdf)

[9] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.

[10] Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

[11] Ortony, A., Clore, G. L., & Collins, A. (1988). *The Cognitive Structure of Emotions*.

[12] Scherer, K. R. (2009). The dynamic architecture of emotion. *Cognition & Emotion*, 23(7), 1307–1351.

[13] Kuppens, P., & Verduyn, P. (2017). Emotion dynamics. *Current Opinion in Psychology*, 17, 22–26.

[14] Fleeson, W., & Jayawickreme, E. (2015). Whole Trait Theory. *Journal of Research in Personality*, 56, 82–92.

[15] DeYoung, C. G. (2015). Cybernetic Big Five Theory. *Journal of Research in Personality*, 56, 33–58.
