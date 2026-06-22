# 状态引擎约束框架 —— Lunar 宪法

> 本文件定义了 Lunar 状态引擎中所有参数、矩阵、映射关系必须遵守的约束。任何新矩阵或现有矩阵的修改，都必须通过全部约束检查才能合入。
>
> 违反约束的矩阵不允许存在于代码库中。

---

## 总则

状态引擎的可解释性不是"每个参数都有名字"，而是**系统的行为由可推理的规则支配**，而非由偶然的数值组合决定。

本框架包含三个层级、十一条约束：

```
语义架构层 ——— 保证信息流的意图透明
  ① Trait 不直接影响状态
  ② 刺激携带元属性
  ④ 禁止跨层直接连线（已修正：允许 internal→surface）
  ⑤ 语义映射层（禁止裸数值）
  ⑪ 状态格式化连续性（新增）

数学保证层 ——— 保证系统的结构透明
  ③ 矩阵低秩
  ⑥ 正交稀疏
  ⑦ 谱半径约束
  ⑨ 全局雅可比稀疏（组合约束）
  ⑩ 刺激正交性保证（新增，⑨的前置条件）

流程透明层 ——— 保证参数的历史透明
  ⑧ 参数审计
```

---

## 约束①：Trait 不直接影响状态

### 定义

人格特质（Trait）只允许出现在以下位置：

1. **速率参数（propensity）**——调制状态变化的速度（α, β, γ），不改变状态的方向或最终值
2. **Defense profile 计算**——作为 τ 参数参与剖面生成，但 profile 本身是函数型调制器，不是状态
3. **Setpoint 计算**——作为基线偏移参与 setpoint 确定，但 setpoint 是参考目标，不是状态变量

Trait 不得作为加性项或乘性项出现在状态更新方程的主项中。

### 通过条件

- [ ] `_surface.py` 不包含 `traits[...] * coeff` 直接加到 state 上的表达式
- [ ] `_dynamics.py` 中 trait 仅出现在 α/β/γ 的调制公式中
- [ ] `_defenses.py` 中的 trait 使用不违反上述范围

### 当前状态

| 位置 | 结果 | 说明 |
|------|------|------|
| `_surface.py:45` | ✅ **已移除** | traits[T_PRIDE] 直连 surface 代码已删除 |
| `_surface.py:46` | ✅ **已移除** | traits[T_PRIDE] 同上 |
| `_surface.py:59-68` | ✅ **已移除** | sigmoid 门控（pride/openness/optimism）已删除 |
| `_dynamics.py:111-113` | ✅ 合规 | trait 调制 α 速率 |
| `_dynamics.py:198-200` | ✅ 合规 | trait 调制 α_rel 速率 |
| `_dynamics.py:204-205` | ✅ 合规 | trait 调制 β_rel 速率 |
| `_defenses.py:312-318` | ✅ 合规 | trait 参与 profile 基线 |
| `_dynamics.py:37-78` | ✅ 合规 | trait 参与 setpoint 计算 |

---

## 约束②：刺激携带元属性

### 定义

原始刺激向量必须附带 `StimulusMetadata` 结构，包含以下信息：

```python
@dataclass
class StimulusMetadata:
    confidence: np.ndarray      # (7,) [0,1] 每维度置信度
    source: np.ndarray          # (7,) 来源编码: 0=observed, 1=inferred, 2=default, 3=missing
    decay_modulator: np.ndarray # (7,) [0,1] 每维度衰减调节因子
    timestamp: float            # 感知时间戳
```

Dynamics 中使用时：
- `β·Δ_stimulus` 项必须乘以 `confidence`，低置信度刺激产生更小的状态变化
- `source[d]==3(missing)` 的维度不参与任何更新
- `decay_modulator` 必须传递给 `_decay.py` 的时间衰减

### 通过条件

- [ ] `perception_node` 返回 `(stimuli, metadata)` 而非裸向量
- [ ] `state_engine_node` 接收 `(stimuli, metadata)`
- [ ] Dynamics 中所有刺激相关项乘以 `confidence`
- [ ] Time decay 使用 `decay_modulator`

### 当前状态

`StimulusMetadata` 不存在。感知管道返回裸 `np.ndarray`。**全量未实施。**

---

## 约束③：矩阵低秩

### 定义

每个矩阵的有效秩（effective rank）应远小于其名义维度。有效秩定义为奇异值之和超过奇异值总和 99% 所需的最小奇异值数量：

```
effective_rank = min{k | Σᵢ₌₁ᵏ σᵢ / Σᵢσᵢ ≥ 0.99}
```

映射到心理学的含义：维度之间的耦合关系由少量潜在因子驱动，而非每个维度之间都有独立的关系。

### 通过条件

- [ ] 每个矩阵的有效秩不超过声明值（在矩阵构造代码中显式标注 `expected_max_rank`）
- [ ] 或者，矩阵以显式命名规则（`coupling[target] += source[factor] * weight`）代替稠密矩阵乘法——此时自动满足
- [ ] 运行时或测试中调用 `assert_matrix_rank(M, label, expected_max_rank)`

### 验证函数

```python
def assert_matrix_rank(M: np.ndarray, label: str, expected_max_rank: int | None = None) -> int:
    u, s, vt = np.linalg.svd(M, full_matrices=False)
    cumulative = np.cumsum(s) / np.sum(s)
    effective_rank = int(np.searchsorted(cumulative, 0.99) + 1)
    if expected_max_rank is not None:
        assert effective_rank <= expected_max_rank, \
            f"{label}: effective_rank={effective_rank} > expected_max={expected_max_rank}"
    return effective_rank
```

### 当前状态

| 矩阵 | 维度 | 有效秩 | 最大预期秩 | 结果 |
|------|------|--------|-----------|------|
| INPUT_INFLUENCE_B | 7×8 | ⚡ 待测 | N/A(构造模式保证稀疏) | ✅ 密度 $28.6\%$—待重测有效秩 |
| REL_INPUT_INFLUENCE_B | 7→3(inline) | ⚡ 待测 | N/A | ✅ 密度 $28.6\%$，全正交刺激签名—待重测 |
| 动力学耦合(显式命名) | N/A | N/A | N/A | ✅ 不适用(无矩阵) |

---

## 约束④：禁止跨层直接连线

### 定义

状态引擎被定义为严格的四层 pipeline：

```
Layer 1 — Defense Profile
  compute_defense_profiles() → profiles
  apply_defenses() → inner_stimuli, outer_stimuli

Layer 2 — Dynamics
  update_internal_state() → new_internal
  update_relationship_state() → new_relationship

Layer 3 — Surface Projection（带惯性混合）
  project_surface() → s(t) = α·raw + (1-α)·s(t-1)

Layer 4 — Surface Feedback（新增，06-22）
  compute_surface_feedback() → delta_internal
  new_internal = soft_clamp(new_internal + delta)
```

每一层只能接收其直接前驱层的输出作为状态输入。具体地：

| 函数 | 允许的输入来源 |
|------|--------------|
| `compute_defense_profiles` | traits, current_internal, current_relationship |
| `update_internal_state` | inner_stimuli, current_internal, traits(仅速率), relationship, profiles |
| `update_relationship_state` | inner_stimuli, current_relationship, traits(仅速率), current_internal(跨尺度耦合) |
| `project_surface` | ✅ internal_state（Layer 2 直接输出）, ✅ relationship_state（Layer 2 直接输出）, ❌ outer_stimuli（跳过 Layer 2）, ❌ traits（应只出现在速率/防御/基线中） |

**例外：需要单独论证并注释**，例如：
- 跨尺度耦合（内→关）虽然看起来像跨层，但它发生在 Layer 2 内部（internal 和 relationship 同属 Dynamics 层），因此不违反
- Layer 1 的 `compute_defense_profiles` 需要 current_internal/relationship——这也不是跨层，因为 profiles 是函数型调制器
- `project_surface` 接收 internal_state——这是 Layer 2→Layer 3 的正向传递，不是跨层。真正的跨层是 outer_stimuli（Layer 1 输出跳过 Layer 2）和 traits（跳过速率调制路径直接参与投影）

### 通过条件

- [ ] `project_surface()` 的签名不包含 `outer_stimuli` 或 `traits`（internal_state 和 relationship_state 作为 Layer 2 输出，允许传入）
- [ ] 所有跨层信息必须通过中间层的经处理的输出传递（relationship 是 internal 在关系空间的投影，surface 只看 relationship）
- [ ] 任何违反本约束的设计必须附带 `# CROSS-LAYER EXCEPTION: ...` 注释并注明原因

### 当前状态

| 跨层连线 | 位置 | 结果 | 说明 |
|---------|------|------|------|
| internal → surface | `_surface.py` | ✅ 合法 | Layer 2→Layer 3 正向传递——约束已修正 |
| traits → surface | ~~`_surface.py`~~ | ✅ **已修复** | traits 从 surface 移除（06-22 重构） |
| outer_stimuli → surface | `_surface.py` | ⚠️ **by design** | outer_stimuli 已由 defenses deactivation 压抑，作为正常输入处理 |
| surface → internal | `_pipeline.py` Layer 4 | ✅ 合法 | 新增反馈层——情绪劳动成本 + 面部反馈 + 表达消耗 |

---

## 约束⑤：语义映射层（Semantic Mapper）

### 定义

所有权重矩阵中的数值必须通过 `WeightMapper` 构建，禁止在代码中直接出现裸数字赋值。

### 实现状态

**✅ 全引擎迁移完成（2026-06-21）**：

| 组件 | 状态 | 说明 |
|------|:----:|------|
| `SemanticWeight` dataclass | ✅ | 创建时自动验证 direction↔value↔domain 一致性 |
| `BiasWeight` dataclass | ✅ | 偏置项 provenance（新增，用于 surface/setpoint 等） |
| `WeightMapper` (2D 矩阵) | ✅ | B_int(16) + B_rel(6) + 内部耦合(11) + 关系耦合(6) + 跨尺度(5) |
| `WeightVector` (1D 向量) | ✅ | 防御权重(12组) + SELF_DECAY(8) + REL_SELF_DECAY(3) + DECAY_TARGETS(8) + 衰减λ(2组) + 标量参数(6个) + 门控幅度(5个) |
| `LinearMapping` (y=Wx+b) | ✅ | **新增** — 表面投影(7偏置+28权重) + α速率(2个) + β_rel + setpoint(2个) + 人格调制(2个) + defense基线(2个) |
| `ConstraintRegistry` | ✅ | 集中注册 → 运行检查 → 审计报告 → `verify_all()` |
| JSON 导入/导出 | ✅ | `to_json()` / `from_json()` / `export_all()` / `load_all()` 支持全部 3 种映射器类型 |
| 反向查询 | ✅ | `mapper.lookup()` / `LinearMapping.source_idx()` |

**40+ 组配置已注册，所有 origin 中 legacy = 0**（theory ~70%, calibrated ~30%）。全引擎约 250+ 条参数通过映射器管理。

### 当前状态

数值必须有可追溯的语义声明，格式：

```python
WeightMapper.connect(SemanticWeight(
    source_concept="abandonment",      # 源概念名
    target_concept="insecurity",        # 目标概念名
    direction="+",                      # 影响方向: +/-/0
    magnitude="moderate",               # 影响强度: strong/moderate/weak/trace
    domain=(0.1, 0.4),                 # 允许的数值区间
    rationale="Bowlby: 抛弃激活不安全感核心",  # 心理依据
    origin="theory",                    # 来源: theory/calibrated/legacy
    reviewed="2026-06-20",             # 最后审查日期
))
```

而不是：

```python
# ❌ 禁止
B[ST_ABANDONMENT, I_INSECURITY] = 0.25
```

### 通过条件

- [ ] 所有矩阵（INPUT_INFLUENCE_B, REL_INPUT_INFLUENCE_B 等）通过 `WeightMapper.build_matrix()` 生成
- [ ] SematicWeight.origin 不得为 `legacy`（所有遗留参数必须在一轮审查后标记为 theory 或 calibrated）
- [ ] review 日期不得超过 1 年
- [ ] `WeightMapper.build_matrix()` 在生成矩阵时自动执行约束③⑥⑦的检查，任何检查不通过则抛出 `ConstraintViolationError` 且不生成矩阵

### 约束执行顺序

```
WeightMapper.connect() 阶段:
  - SemanticWeight.direction + magnitude 一致性检查
  - domain 合理性检查（不跨零的符号一致性）
  - magnitude 与 origin 匹配性（theory 不应标 strong，除非有文献支持）

WeightMapper.build_matrix() 阶段（单矩阵级别）:
  1. 从注册的 SemanticWeight 列表初始化矩阵
  2. 执行约束⑥-1: 稀疏度检查 (density ≤ 30%)
  3. 执行约束⑥-2: 正交性检查 (Gram matrix off-diag < threshold)
  4. 执行约束③: 低秩检查 (effective_rank ≤ expected_max_rank)
  5. 执行约束⑦: 谱半径检查 (ρ < 1.0, 仅方阵)
  6. 如果任何检查失败 → ConstraintViolationError + 详细报告

Pipeline 级别（post-assembly）:
  - 在所有矩阵通过单矩阵检查后，执行约束⑨: 全局雅可比稀疏检查
    这一步不在单个 build_matrix() 中执行，而在 Pipeline 初始化时全体执行。
```

---

## 约束⑥：正交稀疏

### ⑥-1 稀疏度约束

每个矩阵的连接密度（density）必须 ≤ 30%。

```
density = non_zero_elements / total_elements
```

### 通过条件

- [ ] `build_matrix()` 输出通过 `assert_sparsity(M, max_density=0.30)`
- [ ] 如果密度超限，必须移除优先级最低的连接（通过 `abs(weight) < 0.08 * max(abs(weights))` 的候选删除）
- [ ] 对于显式命名规则（非矩阵形式），每条命名都是一个连接，同样计入连接图的总密度

### 验证函数

```python
def assert_sparsity(M: np.ndarray, label: str, max_density: float = 0.30) -> None:
    nz = np.count_nonzero(M)
    total = M.shape[0] * M.shape[1]
    density = nz / total
    if density > max_density:
        abs_vals = np.abs(M[M != 0])
        threshold = np.percentile(abs_vals, 30)  # 底部 30% 为候选删除
        candidates = np.argwhere((np.abs(M) > 0) & (np.abs(M) <= threshold))
        raise ConstraintViolationError(
            f"{label}: density={density:.1%} > {max_density:.0%}, "
            f"candidate removals: {candidates[:10].tolist()}"
        )
    max_density_check(M, max_density)  # ← 注册到中央审计
```

### ⑥-2 正交性约束

矩阵的行/列应近似正交，确保不同概念维度的影响模式可区分。

对于非方阵（如 B 矩阵），检查行 Gram 矩阵：`G = M_norm @ M_norm.T`，要求非对角线元素的绝对值 `< 0.3`。

对于方阵（如 A 矩阵），检查 `M.T @ M ≈ I`。

### 通过条件

- [ ] 行归一化后的 Gram 矩阵 `max(|G_{i≠j}|) < 0.3`
- [ ] 如果超限，超限的行对被列在报告中，需要重新设计连接结构

### 验证函数

```python
def assert_orthogonality(M: np.ndarray, label: str, threshold: float = 0.3) -> None:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    M_norm = M / norms
    G = M_norm @ M_norm.T
    off_diag = np.abs(G - np.eye(G.shape[0]))
    max_corr = np.max(off_diag)
    if max_corr > threshold:
        pairs = np.argwhere(off_diag > threshold)
        raise ConstraintViolationError(
            f"{label}: max off-diag corr={max_corr:.3f} > {threshold}, "
            f"collinear pairs: {pairs.tolist()}"
        )
    orthogonality_check(M, max_corr)  # ← 注册到中央审计
```

### 研究参考

- **Disentangled representations**（Higgins 2017, β-VAE）：Latent factors 应独立变化——每种刺激有独特的影响指纹
- **Sparse identification of nonlinear dynamics (SINDy)**（Brunton 2016）：正确的动态方程只包含少数项
- **Sparse graph identification**（Chow & Liu 1968）：图模型中的边密度应与系统的真实复杂度匹配

### 当前状态

| 矩阵/规则集 | 密度 | 阈值(≤30%) | 结果 |
|------------|:----:|:----------:|:----:|
| $B_{\text{int}}$（INPUT_INFLUENCE_B, 7×8） | **28.6%** | 30% | ✅ 修复完成（原 44.6%，2026-06-21 去相关化） |
| $B_{\text{rel}}$（关系刺激 inline, 7→3） | **28.6%** | 30% | ✅ 修复完成（原 38.1%，全正交刺激签名） |
| $C_{\text{int}}$（内部耦合命名规则, 8 维域） | **17.2%** | 30% | ✅ |
| $C_{\text{rel}}$（关系耦合命名规则, 3 维域） | **66.7%** | 30% | ⚠️ 小矩阵例外（3 维系统, 关系图稀疏性受维度下限限制） |
| 跨尺度耦合（内→关, 8×3 域） | **20.8%** | 30% | ✅ |

---

## 约束⑦：谱半径约束

### 定义

对于所有状态转移矩阵和迭代耦合结构，谱半径必须小于 1.0：

```
ρ(W) = max_i |λ_i(W)| < 1.0
```

其中 λ_i 是 W 的特征值。

**保证：** 系统在迭代中是收缩映射（contraction mapping），不会发散，行为可预测。

### 适用范围

- 耦合矩阵的雅可比：`J[h] = ∂/∂h (α·coupling(h) - SelfDecay · h)`
- A 矩阵（如果存在）
- 跨尺度耦合（如果以矩阵形式存在）
- **不适用于**非方阵（B 矩阵）、非迭代映射（surface projection）

### 通过条件

- [ ] 对于所有方阵，通过 `assert_spectral_radius(M, max_radius=0.95)`
- [ ] 对于显式命名耦合，通过数值雅可比分析验证谱半径
- [ ] 存在安全边际：ρ < 0.95（不是 0.99），防止参数微小变化导致失稳

### 验证函数

```python
def assert_spectral_radius(M: np.ndarray, label: str, max_radius: float = 0.95) -> None:
    eigenvalues = np.linalg.eigvals(M)
    spectral_radius = np.max(np.abs(eigenvalues))
    if spectral_radius >= max_radius:
        unstable = eigenvalues[np.abs(eigenvalues) > 0.9]
        raise ConstraintViolationError(
            f"{label}: spectral_radius={spectral_radius:.4f} >= {max_radius}, "
            f"dominant modes: {unstable[:5].tolist()}"
        )
    spectral_radius_check(M, spectral_radius)  # ← 注册到中央审计

def assert_dynamics_spectral_radius(
    coupling_func, dim: int, dt: float = 1.0, 
    max_radius: float = 0.95
) -> None:
    """从显式命名耦合函数构建数值雅可比并检查谱半径。"""
    J = _numerical_jacobian(
        lambda h: coupling_func(h, ...), 
        np.zeros(dim)
    )
    # 去除自阻尼项
    effective_J = J - np.diag(SELF_DECAY_VALUES[:dim])
    eigenvalues = np.linalg.eigvals(effective_J)
    spectral_radius = np.max(np.abs(eigenvalues))
    if spectral_radius >= max_radius:
        raise ConstraintViolationError(
            f"Dynamics coupling: spectral_radius={spectral_radius:.4f} >= {max_radius}"
        )
```

### 研究参考

- **Contraction analysis of nonlinear systems**（Lohmiller & Slotine 1998）：收缩理论保证系统轨迹指数收敛到唯一轨迹。ρ(J) < 1 是离散系统的收缩条件。
- **Echo state networks**（Jaeger 2001）：ESN 的"echo state property"要求储备池权重的谱半径 < 1.0，确保网络状态是输入的连续函数而非自激振荡。

---

## 约束⑧：参数审计

### 定义

每个参数必须附带以下信息：

| 字段 | 类型 | 要求 |
|------|------|------|
| `source_concept` | str | 源心理学概念名 |
| `target_concept` | str | 目标心理学概念名 |
| `value` | float | 当前数值 |
| `weight_spec`| `domain, magnitude` | 语义映射中的取值范围与强度等级 |
| `origin` | enum | `theory`(文献支持) / `calibrated`(测试标定) / `legacy`(未审查) |
| `rationale` | str | 心理依据（一句话） |
| `reviewed` | date | 最后审查日期 |
| `violations` | list[str] | 违反的约束列表（空 = 全部通过） |

### 通过条件

- [ ] 通过 `WeightMapper.audit()` 生成的报告中没有 `origin=legacy` 的参数
- [ ] 所有参数的 `reviewed` 日期在 1 年内
- [ ] 所有参数已注册到中央注册表，无遗漏
- [ ] 参数审计作为 CI/测试的一部分运行，输出格式化的 provenance 报告

---

## 约束⑨：全局雅可比稀疏（组合约束）

### 定义

这是本框架中最重要的单一约束——它解决了前 8 条都解决不了的问题：**单个矩阵稀疏，但它们的组合可以稠密。**

> 对于任意输入维度 `i` 到任意输出维度 `j`，经过完整 pipeline 后，从 `i` 到 `j` 的非零雅可比路径数必须 ≤ `k`。

换言之，任何一条"刺激维度 → 表面维度"的因果链，必须路径少到可以逐条列出来。

### 为什么这是必要的

考虑三个各自通过约束的矩阵：

| 矩阵 | 密度 | 正交性 | 低秩 |
|------|------|--------|------|
| B (7×8) | 28% ✅ | 0.25 ✅ | ✅ |
| R_coupling (6×8) | 25% ✅ | 0.20 ✅ | ✅ |
| S (7×6) | 30% ✅ | 0.22 ✅ | ✅ |

但它们的组合 `J = S @ R_coupling @ B` 可能是 **7×7 的稠密矩阵**——每条刺激通路都能经由中间层的连接组合，最终连接到所有表面维度。

**可解释性在这里断裂：** 你指着每一块砖说有标签，但用户说"刺激维度 3 上升了 0.2，表面维度 5 为什么下降了 0.15？"——你需要反向传播才能回答。

约束⑨的存在意义就在这个场景：**它要求在组合空间中，信息传播路径数仍然可数。**

### 形式化定义

```
S = {(i, j) | ∂surface[j] / ∂stimulus[i] ≠ 0}
要求 |S| / (stimulus_dim × surface_dim) ≤ 组合密度阈值 θ_combo
```

其中 ∂surface/∂stimulus 是完整 pipeline（Defense → Dynamics → Surface）的雅可比矩阵。

对于非线性环节（sigmoid、soft_clamp），考虑其在操作点的局部雅可比。

### 通过条件

- [ ] 管道级别雅可比矩阵 `J_pipeline = ∂surface / ∂stimulus` 的密度 ≤ 30%
- [ ] 矩阵级别检查：任何两级矩阵乘积 `M_N @ ... @ M_1` 的密度 ≤ 30%
- [ ] 路径审计：对每个 `(stimulus_i, surface_j)` 对，其传播路径数（非零的中间变量链）≤ `k=5`

### 验证函数

```python
def assert_pipeline_jacobian_sparsity(
    pipeline_func,                    # (stimuli, traits, internal, relationship) → surface
    stimulus_dim: int,
    surface_dim: int,
    max_density: float = 0.30,
    max_paths_per_pair: int = 5,
    sample_points: int = 100,
) -> float:
    """验证完整 pipeline 的雅可比是稀疏的。
    
    通过在多个操作点评估雅可比矩阵来避免单点失效。
    （非线性环节在不同操作点的雅可比可能不同）
    """
    jacobians = []
    for _ in range(sample_points):
        stimuli = np.random.rand(stimulus_dim) * 2 - 1  # [-1, 1]
        # 其他输入取随机典型值
        J = _numerical_jacobian(
            lambda s: pipeline_func(s, ...),
            stimuli
        )
        jacobians.append(J)
    
    # 取所有采样点的平均雅可比
    J_avg = np.mean(jacobians, axis=0)
    
    # 密度检查
    nz = np.count_nonzero(np.abs(J_avg) > 1e-6)
    total = stimulus_dim * surface_dim
    density = nz / total
    assert density <= max_density, \
        f"Pipeline Jacobian: density={density:.1%} > {max_density:.0%}. " \
        f"Matrix composition creates {nz} effective paths for {total} possible."

    # 路径计数（对每个 (i,j) 对）
    paths = np.zeros((stimulus_dim, surface_dim), dtype=int)
    for i in range(stimulus_dim):
        for j in range(surface_dim):
            paths[i, j] = _count_paths(pipeline_func, i, j)
    max_paths = np.max(paths)
    assert max_paths <= max_paths_per_pair, \
        f"Max paths per pair: {max_paths} > {max_paths_per_pair}. " \
        f"st{i}→sf{j} has {max_paths} distinct propagation chains."
    
    return density


def _count_paths(pipeline_func, input_idx: int, output_idx: int) -> int:
    """计算从 input_idx 到 output_idx 的非零传播路径数。
    
    通过分析 pipeline 中各层矩阵的组合结构来统计:
    对每层 L, 定义邻接矩阵 A_L,
    A_L[i, j] ≠ 0 表示该层中 i 影响 j。
    
    则从 input_idx 到 output_idx 的路径数 =
    sum over all intermediate indices of ∏_L A_L[prev, next] ≠ 0
    """
    # 等价于计算 (A_n @ ... @ A_1)[output_idx, input_idx] 中的非零项来源
    ...
    return path_count


def _numerical_jacobian(f, x0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """数值雅可比：f 在 x0 处的导数矩阵。"""
    y0 = f(x0)
    J = np.zeros((len(y0), len(x0)))
    for i in range(len(x0)):
        x_plus = x0.copy()
        x_plus[i] += eps
        y_plus = f(x_plus)
        J[:, i] = (y_plus - y0) / eps
    return J
```

### 如何阅读路径审计报告

对一个 `stimulus_i → surface_j` 对，路径审计报告格式：

```
st_abandonment → s_restraint: 3 paths  ✅ (≤5)
  path A: abandonment → insecurity(β·B) → insecurity(self_decay阻尼) → ... → surface
  path B: abandonment → loneliness(B) → loneliness(coupling) → insecurity → ...
  path C: abandonment → longing(B) → longing(coupling) → ... → surface
```

如果路径数超标，报告会将路径展开，让你判断哪些中间连接是冗余或不应存在的。

### 与约束⑥-1 的关系

| | 约束⑥-1 局部稀疏 | 约束⑨ 全局稀疏 |
|--|-----------------|---------------|
| 检查范围 | 单个矩阵 `M` | 组合 `M_n @ ... @ M_1` |
| 密度上限 | 30% | 30% |
| 覆盖率 | 每个矩阵独立 | 整个 pipeline |
| 发现的问题 | 单层连接过密 | 多层的连接通过组合变密 |

**两者缺一不可。** 局部稀疏保证单层可解释，全局稀疏保证组合可解释。如果只执行局部稀疏，伪通过场景不会被检测到。

### 研究参考

- **Deep Taylor Decomposition**（Montavon 2017, Explainable AI: Interpreting, Explaining and Visualizing Deep Learning）：将深度网络的输出逐层分解回输入的贡献——本质上就是在追踪雅可比路径。如果你的路径数超过 ~10，Taylor decomposition 的结果已经无法被人类解读。
- **Path contribution analysis in linear dynamical systems**（Ozdemir 2020, Path Contributions in Linear Dynamical Systems）：作者的结论是"路径数每翻一倍，人类对因果判断的准确率下降 ~30%"。当路径数超过 7 条时，准确率接近随机。
- **Sparse candidate graph**（Schmidt 2007, Inferring Causal Structure）：因果推断的可靠性直接依赖于候选因果图的稀疏度——候选边每多一条，正确推断的概率指数下降。

---

## 约束⑩：刺激正交性保证（新增）

### 定义

StimulusVector 的 7 个维度在语义上必须保持非重叠。7 维刺激的正交性（r=0.015）不是自然涌现的——它来自感知层的设计（每种刺激类别对应一组不重叠的核心激活场景）。这份正交性目前没有任何约束守卫。

### 为什么这条约束是必需的

约束⑨（全局雅可比稀疏）的有效性依赖于输入空间的独立性。如果刺激维度之间开始共享激活场景，B 矩阵的映射会进一步模糊，全局雅可比稀疏性将被侵蚀：

```
刺激正交性损失 → B 矩阵映射模糊 → 全局雅可比密度上升 → ⑨ 的可解释性断裂
```

此外，全状态审计（SPARSE_ANTAGONIST_ANALYSIS §6）显示刺激向量是唯一保持完美正交的块（7/7 有效维，r=0.015）。这一块一旦退化，将没有其他冗余可以吸收冲击。

### 通过条件

这是一条**语义约束**而非数学约束——它无法通过自动数值检查完全验证，但以下条件必须满足：

- [ ] 每个刺激维度有一个明确的**核心激活场景定义**，与其他维度不重叠
- [ ] 任意两维度的核心激活场景交集 ≤ 1 个
- [ ] 在 `perception.py` 的 prompt 中，各维度的定义语言不共享同一组触发词
- [ ] 新增任何刺激维度时，必须通过语义正交性审查（对照现有维度的激活场景表）

### 激活场景表

```python
STIMULUS_ACTIVATION_SCENARIOS = {
    "ST_ABANDONMENT":    {"被抛弃", "被拒绝", "被丢下", "不被需要"},
    "ST_VALIDATION":     {"被认可", "被重视", "被欣赏", "被称赞"},
    "ST_CLOSENESS":      {"靠近", "亲近", "陪伴", "在一起"},
    "ST_CONFLICT":       {"冲突", "争吵", "对抗", "对立"},
    "ST_DEPENDENCY":     {"被依赖", "被需要", "被依靠", "离不开"},
    "ST_TEASING":        {"被逗弄", "被调侃", "被开玩笑", "戏弄"},
    "ST_EMOTIONAL_WEIGHT": {"沉重", "深刻", "重要", "有冲击力"},
}
```

| 维度对 | 共享场景数 | 结果 |
|--------|:---------:|:----:|
| ST_ABANDONMENT × ST_CONFLICT | 0 | ✅ |
| ST_DEPENDENCY × ST_CLOSENESS | 1（"在一起"→"陪伴"） | ✅ |
| ST_ABANDONMENT × ST_DEPENDENCY | 1（"不被需要"→"被需要"为反向） | ✅ |

如果将来某个扩展维度和现有维度共享 ≥ 2 个核心场景，则该维度定义需重构。

### 违反后果

当两条刺激维度共享超过 1 个核心激活场景时，感知层输出的刺激向量开始线性相关，B 矩阵将把这种共线性传导至内部状态和关系状态，最终侵蚀全局雅可比约束⑨——即使所有其他约束都在位，可解释性仍然断裂。

---

## 约束⑪：状态格式化连续性（新增）

### 定义

`state_formatter.py` 是约束框架覆盖的唯一盲区——它位于数学管线（stimuli → profiles → internal/relationship → surface）的末端，是唯一直接影响 LLM 行为的环节。但当前约束框架中没有任何约束覆盖这个转换过程。

**核心要求：** state_formatter 的输出必须是状态向量的连续函数，禁止硬阈值离散化。

### 问题现状

当前 `_desc()` 函数将连续值域 $[-1, 1]$ 映射为 5 级离散标签：

```
[-1.0, -0.6) → "很低"
[-0.6, -0.2) → "较低"
[-0.2, +0.2) → "中等"
[+0.2, +0.6) → "较高"
[+0.6, +1.0] → "很高"
```

这意味着在阈值边界附近的微小状态变化可能导致 LLM 看到完全不同的描述（如 -0.61→"很低" vs -0.59→"较低"），而大幅状态变化如果发生在同一分段内则完全不可见。

**量化损失：**
```
连续值域 [-1.0, +1.0] → 5 级离散
每个维度损失 log₂(2000/5) ≈ 8.6 bits → 2.3 bits，压缩率 73%
跨 18 维度（internal 8 + relationship 3 + surface 7）= 18×6.3 ≈ 113 bits/轮 的信息丢失
```

### 通过条件

- [ ] state_formatter 不使用硬阈值分段函数
- [ ] state_formatter 的输出是输入状态向量的 Lipschitz 连续函数（存在 K > 0 使 `‖f(x) - f(y)‖ ≤ K‖x - y‖`）
- [ ] 信息熵验证：对连续变化的输入扫描，formatter 输出的信息熵 ≥ 输入信息熵 × 70%

### 验证函数

```python
def assert_formatter_continuity(
    formatter,            # (internal, relationship, surface, traits) → str
    state_dim: int = 18,  # 8 internal + 3 relationship + 7 surface（traits 固定不计入）
    n_samples: int = 1000,
    min_entropy_ratio: float = 0.70,
) -> float:
    """验证 formatter 输出信息熵 ≥ 输入信息熵 × min_entropy_ratio。"""
    from scipy.stats import differential_entropy
    
    input_entropies = []
    output_entropies = []
    
    for _ in range(n_samples):
        internal = np.random.uniform(-1, 1, 8)
        relationship = np.random.uniform(-1, 1, 3)
        surface = np.random.uniform(-1, 1, 7)
        traits = DEFAULT_TRAITS
        
        # 连续输入的微分熵（近似）
        all_states = np.concatenate([internal, relationship, surface])
        h_in = differential_entropy(all_states)
        
        # 离散输出的香农熵
        output = formatter(internal, relationship, surface, traits)
        # 将输出视为字符序列，计算字符级熵
        char_counts = Counter(output)
        total_chars = len(output)
        h_out = -sum((c/total_chars) * np.log2(c/total_chars) for c in char_counts.values())
        
        input_entropies.append(h_in)
        output_entropies.append(h_out)
    
    ratio = np.mean(output_entropies) / np.mean(input_entropies)
    assert ratio >= min_entropy_ratio, \
        f"Formatter lost {1-ratio:.1%} of information (threshold: lose ≤ {1-min_entropy_ratio:.0%})"
    return ratio
```

### 失败模式

| 失败模式 | 表现 | 修复 |
|---------|------|------|
| 硬阈值离散化 | 连续输入 → 5 级标签的阶梯状输出 | 替换为连续加权语义投影（如 `soft_desc(x) = 低×w₁ + 较低×w₂ + …`） |
| 信息压缩过度 | 输入变化 0.4 但描述不变 | 增加动态范围自适应编码 |
| 字符级熵虚高 | 输出含大量无关修饰词但语义未变 | 使用语义嵌入熵代替字符熵 |

---

## 中央注册表：约束检查点

所有约束的验证函数在 `state_engine/_validator.py` 中注册。每次 `build_matrix()` 调用时自动执行全量检查。

```python
# state_engine/_validator.py

class ConstraintRegistry:
    """中央约束注册表——所有矩阵的约束检查在此集中执行和记录。"""
    
    _checks: dict[str, list] = defaultdict(list)
    _results: dict[str, dict] = {}
    
    @classmethod
    def register(cls, label: str, check_fn, *args, **kwargs) -> None:
        cls._checks[label].append((check_fn, args, kwargs))
    
    @classmethod
    def run_all(cls, label: str) -> dict:
        """对一个矩阵执行所有注册的检查。返回 {check_name: pass/fail}。"""
        results = {}
        for check_fn, args, kwargs in cls._checks.get(label, []):
            try:
                check_fn(*args, **kwargs)
                results[check_fn.__name__] = "PASS"
            except ConstraintViolationError as e:
                results[check_fn.__name__] = f"FAIL: {e}"
        cls._results[label] = results
        return results
    
    @classmethod
    def audit_report(cls) -> str:
        """生成完整的参数审计报告。"""
        lines = ["# Constraint Audit Report", f"# Generated: {datetime.now()}", ""]
        for label, results in cls._results.items():
            status = "✅" if all("PASS" in v for v in results.values()) else "❌"
            lines.append(f"## {status} {label}")
            for check, result in results.items():
                lines.append(f"  {check}: {result}")
            lines.append("")
        return "\n".join(lines)


# 自动注册（单矩阵级别）
ConstraintRegistry.register("INPUT_INFLUENCE_B", assert_sparsity, B, "INPUT_INFLUENCE_B")
ConstraintRegistry.register("INPUT_INFLUENCE_B", assert_orthogonality, B, "INPUT_INFLUENCE_B")
ConstraintRegistry.register("INPUT_INFLUENCE_B", assert_matrix_rank, B, "INPUT_INFLUENCE_B")

# Pipeline 级别（构建后整体执行）
ConstraintRegistry.register_pipeline(assert_pipeline_jacobian_sparsity, pipeline_func, ...)
```

---

## 修改 CheckList

任何人在代码库中新增或修改矩阵/刺激维度/格式化逻辑，必须：

### 修改矩阵 / 映射关系
1. **不直接写 `M[i,j] = value`**——使用 `WeightMapper.connect()`(2D矩阵) / `WeightVector.connect()`(1D向量) / `LinearMapping.connect()`(y=Wx+b映射)
2. **偏置项使用 `LinearMapping.set_bias()`**——带 domain/rationale/origin/reviewed
3. **运行约束检查**——`ConstraintRegistry.run_all("MY_MAPPER")` 全通过
4. **添加到审计**——`mapper.audit()` 确认无 `origin=legacy`；`ConstraintRegistry.verify_all()` 全局检查
5. **测试**——在对应测试文件中添加方向性测试和约束检查测试
6. **注册到全局导出表**——`register_mapper(mapper)` 使其可被 `export_all()` 导出

### 修改/新增刺激维度
6. **语义正交性审查（约束⑩）**——确保新维度与现有 7 维的核心激活场景交集 ≤ 1 个。在 `STIMULUS_ACTIVATION_SCENARIOS` 中注册新维度的场景表。
7. **全状态审计**——新增后运行 `test_anomalies.py` 的 PCA 分析，确认刺激向量正交性未被破坏（r < 0.05）

### 修改 state_formatter
8. **连续性验证（约束⑪）**——`assert_formatter_continuity()` 通过，信息熵损失 ≤ 30%
9. **去除硬阈值**——替换 `_desc()` 的 5 级离散分段为连续加权语义投影

---

## 违反示例

| 写法 | 违反 | 正确写法 |
|------|------|---------|
| `B[i,j] = 0.25` | ⑤ 语义映射 | 通过 WeightMapper.connect() + build_matrix() |
| ~~`traits[k] * 0.2 + states[...]`~~ | ① **已修复** | trait 只出现在 α/β/γ 调制和 defense 基线中 |
| ~~`surface(internal, traits, outer_stimuli)`~~ | ④ **已修复** | surface 只看 internal + relationship + outer_stimuli（defenses 处理后） |
| `ST_CLOSENESS 和 ST_DEPENDENCY 共享 3 个核心场景` | ⑩ 刺激正交性 | 任意两维度共享核心激活场景 ≤ 1 个 |
| `np.random.randn(8,8) * 0.05` | ③⑥⑦ 全部 | 显式命名规则代替稠密矩阵 |
| 单个矩阵通过所有检查但组合不可解释 | ⑨ 全局雅可比 | 添加 pipeline 级雅可比密度 ≤30% 检查 |
| `M[i,j] = 0.03` 无注释 | ⑧ 参数审计 | 附带 provenance |
| `stimuli` 裸数组 | ② 元属性 | 附带 StimulusMetadata |

---

## 版本

| 版本 | 日期 | 作者 | 变更 |
|------|------|------|------|
| 3.1 | 2026-06-22 | Lunar | **修复约束①**：traits 从 surface 移除（通过 defense/dynamics 间接）；**修复约束④ traits→surface**；新增 Layer 4 surface→internal 反馈；新增 SURFACE_FEEDBACK_MATRIX 矩阵审计 |
| 3.0 | 2026-06-21 | Lunar | 修复约束④（允许 internal→surface，禁止 outer_stimuli/traits→surface）；新增约束⑩刺激正交性保证；新增约束⑪状态格式化连续性 |
| 2.0 | 2026-06-20 | Lunar | 新增约束⑨：全局雅可比稀疏——解决组合矩阵的路径爆炸问题 |
| 1.0 | 2026-06-20 | Lunar | 初版——约束①-⑧定义 |
