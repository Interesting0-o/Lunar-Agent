# Lunar 状态引擎架构

> 2026-06-22 | Surface 重构完成（惯性更新+双向耦合+traits间接化）| 5 节点 LangGraph | defense-based 残差动力学 | 记忆系统已集成

---

## 一、系统总览

Lunar 是一个基于 LangGraph 的 AI 角色扮演引擎。用**计算心理学状态机**替代传统 prompt-based 角色扮演，核心是一个 Bowlby 依恋防御驱动的四阶段状态引擎。

### 1.1 技术栈

| 组件 | 技术 |
|------|------|
| 图编排 | LangGraph (StateGraph) |
| 主对话 LLM | DeepSeek v4 Pro |
| 感知 LLM | Ollama qwen2.5:7b |
| 嵌入模型 | Ollama qwen3-embedding:8b |
| 状态向量 | numpy ndarray |
| 持久化 | SQLite (langgraph.checkpoint) |
| 记忆持久化 | JSON 文件 |
| 包管理 | uv (Python 3.13) |

### 1.2 LangGraph 流水线

```
START
  │
  ├─ [首次运行] → inject_system_node ─→ perception_node
  │               注入角色 system prompt   感知：user_input → stimuli
  │               + 默认 traits            3 次重试，失败→error
  │               + 种子记忆
  │
  └─ [后续轮次] → perception_node ──────────────────┘
                               │
                     route_after_perception
                      error=True → END
                      │
                      ▼
              state_engine_node
                ① Defense Profiles (Bowlby)
                ② Residual Dynamics
                ③ Surface Projection (惯性混合)
                ④ Surface → Internal Feedback
                + 清除 user_stimuli
                      │
                      ▼
              state_formatter_node
              数值状态 → 中文导演描述
                      │
                      ▼
              llm_node
              state_description 注入 SystemMessage
              DeepSeek 生成回复
                      │
                      ▼
              memory_summery_node  ──→ END
              对话总结 → 嵌入 → 保存到 MemoryStore
```

---

## 二、状态向量

### 2.1 全状态空间（名义 28 维，有效 ~12 维）

| 状态层 | 维度 | 值域 | 生命周期 | 持久化 |
|--------|:----:|:----:|---------|:------:|
| StimulusVector | 7 | $[0, 1]$ | 每轮，node 内 | ❌ 消费后清除 |
| InternalState | 8 | $[-1, 1]$ | 跨轮持续 | ✅ |
| RelationshipState | 3 | $[-1, 1]$ | 跨轮持续 | ✅ |
| SurfaceState | 7 | $[-1, 1]$ | 每轮重算 | ✅ 注入 LLM（经 state_formatter） |
| Traits | 10 | $[-1, 1]$ | 固定（待演化） | ✅ |

### 2.2 StimulusVector — 心理刺激（7 维）

perception 节点从用户输入直接提取，**一步到位**（无旧版 SocialSignals / InteractionImpact 中间表示）。

$$
\begin{aligned}
s_{\text{abandonment}} &\in [0,1] \quad \text{被抛弃恐惧} \\
s_{\text{validation}} &\in [0,1] \quad \text{被认可/被重视} \\
s_{\text{closeness}} &\in [0,1] \quad \text{亲密靠近感} \\
s_{\text{conflict}} &\in [0,1] \quad \text{冲突/对抗张力} \\
s_{\text{dependency}} &\in [0,1] \quad \text{被依赖/被需要} \\
s_{\text{teasing}} &\in [0,1] \quad \text{被逗弄/被调侃} \\
s_{\text{emotional\_weight}} &\in [0,1] \quad \text{情绪冲击强度}
\end{aligned}
$$

### 2.3 InternalState — 内部情绪指标（8 维）

$$
\begin{aligned}
i_{\text{energy}} &\in [-1,1] \quad \text{精力} \\
i_{\text{stress}} &\in [-1,1] \quad \text{压力} \\
i_{\text{loneliness}} &\in [-1,1] \quad \text{孤独} \\
i_{\text{insecurity}} &\in [-1,1] \quad \text{不安} \\
i_{\text{irritation}} &\in [-1,1] \quad \text{烦躁} \\
i_{\text{longing}} &\in [-1,1] \quad \text{渴望/思念} \\
i_{\text{social\_battery}} &\in [-1,1] \quad \text{社交电量} \\
i_{\text{mental\_fatigue}} &\in [-1,1] \quad \text{精神疲劳}
\end{aligned}
$$

### 2.4 RelationshipState — 关系状态（3 维）

2026-06-21 语义合并：原 6 维（affection, trust, familiarity, dependency, emotional_safety, romantic_tension）经 PCA 验证有效自由度仅 $\sim 2/6$，合并为 3 维。

| 常量 | 含义 | 合并来源 |
|------|------|---------|
| $\text{R\_AFFECTION}$ | 好感度 | 保留原语义 |
| $\text{R\_TRUST\_BOND}$ | 信任纽带 | trust + emotional_safety |
| $\text{R\_INTIMACY}$ | 亲密张力 | familiarity + dependency + romantic_tension |

### 2.5 SurfaceState — 表面表达（7 维）

$$
\begin{aligned}
s_{\text{expressiveness}} &\in [-1,1] \quad \text{情绪外露程度} \\
s_{\text{warmth}} &\in [-1,1] \quad \text{语气温度} \\
s_{\text{sharpness}} &\in [-1,1] \quad \text{攻击性/尖锐感} \\
s_{\text{softness}} &\in [-1,1] \quad \text{柔和度} \\
s_{\text{enthusiasm}} &\in [-1,1] \quad \text{活力/热情} \\
s_{\text{restraint}} &\in [-1,1] \quad \text{克制程度} \\
s_{\text{vulnerability}} &\in [-1,1] \quad \text{脆弱感}
\end{aligned}
$$

### 2.6 Traits — 人格特质（10 维）

| 常量 | 含义 | 默认值（月下誓约） |
|------|------|:-----------------:|
| $\text{T\_SENSITIVITY}$ | 敏感度 | +0.4 |
| $\text{T\_PRIDE}$ | 自尊心 | +0.3 |
| $\text{T\_EMOTIONAL\_OPENNESS}$ | 情绪开放性 | +0.2 |
| $\text{T\_EMOTIONAL\_STABILITY}$ | 情绪稳定性 | 0.0 |
| $\text{T\_OPTIMISM}$ | 乐观倾向 | +0.1 |
| $\text{T\_ANXIETY\_PRONENESS}$ | 焦虑倾向 | +0.2 |
| $\text{T\_ANGER\_REACTIVITY}$ | 易怒倾向 | 0.0 |
| $\text{T\_JEALOUSY\_SENSITIVITY}$ | 嫉妒敏感度 | +0.4 |
| $\text{T\_ATTACHMENT\_ANXIETY}$ | 依恋焦虑 | +0.1 |
| $\text{T\_ATTACHMENT\_AVOIDANCE}$ | 依恋回避 | **-0.6** |

> Traits 当前为**固定值**（`DEFAULT_TRAITS`），不随对话更新。Trait 演化是已知 P0 缺失。

---

## 三、感知节点（perception_node）

### 3.1 流程

```
用户消息 → extract_recent_context() → 取最近 4 条 Human/AI 消息
         → call_perception_with_retry()
             ├─ 第 1 次: 基础 prompt
             ├─ 第 2 次: +"只输出 JSON"
             └─ 第 3 次: +"严重警告：只输出 JSON 对象本身"
         → json.loads() + validate_perception_result()
         → stimuli_from_dict() → 7 维 StimulusVector
```

### 3.2 关键设计

- **无中间表示：** LLM 一步输出 7 维心理刺激（旧版有 SocialSignals 9 维 + InteractionImpact 4 维两步方案，已替换）
- **3 次重试** + 逐步升级的 JSON 强调（config.py）
- 上下文窗口 `context_window=4`（最近 4 条消息）
- 感知模型：**Ollama qwen2.5:7b**（非 DeepSeek，非 aliased）
- 全部失败 → `error=True` → `route_after_perception` 路由到 END，跳过 state_engine 和 formatter

### 3.3 输出

```python
{"user_stimuli": np.ndarray(7,)}  # stimuli_from_dict() 转换
```

---

## 四、状态引擎（四阶段管线）

### 4.1 总公式

$$
h_t = h_{t-1} + \Delta t \cdot (\alpha \cdot \Delta_{\text{coupling}} + \Delta_{\text{stimulus}})
$$

每轮对话中状态仅由**刺激**和**耦合**驱动。稳态恢复（向人格基线回归）由时间衰减在对话间隔中处理。

### 4.2 阶段 ①：防御剖面

基于 Bowlby (1980) 依恋防御二分法。`profiles \in [0,1]^{2 \times 7}$，对 7 种刺激独立激活。

#### 去激活（Deactivation）— 削减外在表达

$$
\begin{aligned}
\text{deact}[d] &= \text{baseline}[d] &&\text{(人格基线)} \\
&\quad + \sum_{t} \tau_t \cdot W_{t,d}^{\text{deact},A} &&\text{(特质调制)} \\
&\quad \times \bigl(1 + r_{\text{trust}} \cdot W_{d}^{\text{trust},M}\bigr) &&\text{(关系调制, 乘法)} \\
&\quad + i_{\text{stress}} \cdot W_{d}^{\text{stress},A} + i_{\text{insec}} \cdot W_{d}^{\text{insec},A} &&\text{(急性状态, 加法)}
\end{aligned}
$$

$$
\text{profiles}[0] = \sigma\bigl(5.0 \cdot (\text{deact} - 0.35)\bigr)
$$

#### 过度激活（Hyperactivation）— 放大内心感受（含状态调制）

$$
\begin{aligned}
\text{hyper}[d] &= \underbrace{\text{baseline}[d] + \sum_{t} \tau_t \cdot W_{t,d}^{\text{hyper},A} \times \bigl(1 + r_{\text{aff}} \cdot W_{d}^{\text{aff},M} + r_{\text{int}} \cdot W_{d}^{\text{int},M}\bigr)}_{\text{人格基线（秩-1, 仅 traits+rel）}} \\
&\quad + \underbrace{\sum_{s \in \text{internal}} i_s \cdot M_{s,d}^{\text{state}}}_{\text{状态调制（稀疏维度特异性）}} &&\text{(2026-06-22 新增)}
\end{aligned}
$$

$$
\text{profiles}[1] = \sigma\bigl(5.0 \cdot (\text{hyper} - 0.38)\bigr)
$$

**2026-06-22 更新**：原有的纯秩-1 模型（internal 作为标量强度输入）被拆分为两个独立通道：

1. **人格基线**（秩-1）：traits + relationship → 标量强度 × PC1_DIR（internal 不再参与）
2. **状态调制**（新增）：internal 状态通过 `HYPER_STATE_MODULATION` 稀疏连接（每条有心理学 provenance）直接驱动特定 hyper 维度

拆分的动因：PCA 审计发现人格差异和情绪状态差异对 hyper 方差贡献的方向不一致（PC1 夹角 142°），秩-1 模型强制两者同步是错误的。新模型允许"愤怒→冲突↑且亲密↓"等交叉调制模式。

#### 防御应用

$$
\begin{aligned}
\text{inner}[s] &= \text{stimuli}[s] \cdot (1 + 0.50 \cdot \text{hyper}[s]) &&\text{(内心感受放大)} \\
\text{outer}[s] &= \text{inner}[s] \cdot (1 - 0.70 \cdot \text{deact}[s]) &&\text{(外在表达削减)}
\end{aligned}
$$

两者独立运作——可内心翻江倒海而表面波澜不惊（高 hyper + 高 deact）。

#### 逐维度权重（2026-06-20 修复）

所有权重 $W \in \mathbb{R}^{7}$ 为逐刺激维度独立数组，替代旧版全局标量。每组均值为原全局系数以保持向后兼容。

```python
# 示例：压力对去激活
STRESS_DEACT_A = [0.08, 0.03, 0.01, 0.12, 0.04, 0.00, 0.08]
#               [AB,   VA,   CL,   CO,   DE,   TE,   EW]
```

### 4.3 阶段 ②：残差动力学

#### 内部状态更新

**耦合速率 $\alpha \in [0.02, 0.35]$：**

$$
\alpha = 0.285 + 0.15 \cdot \tau_{\text{openness}} - 0.075 \cdot \tau_{\text{stability}} + 0.06 \cdot r_{\text{trust}}
$$

**刺激接受率 $\beta_{\text{stim}} \in [0.01, 0.35]$（逐刺激维度）：**

$$
\beta_{\text{stim}}[s] = 0.05 + 0.35 \cdot \text{hyper}[s] - 0.15 \cdot \text{deact}[s]
$$

**耦合项（11 条显式命名规则，替代旧 A 矩阵）：**

$$
\begin{aligned}
\Delta i_{\text{stress}} &\mathrel{+}= -0.05 \cdot i_{\text{energy}} + 0.10 \cdot i_{\text{insecurity}} \\
\Delta i_{\text{loneliness}} &\mathrel{+}= -0.05 \cdot i_{\text{energy}} + 0.08 \cdot i_{\text{stress}} \\
\Delta i_{\text{insecurity}} &\mathrel{+}= 0.12 \cdot i_{\text{loneliness}} \\
\Delta i_{\text{irritation}} &\mathrel{+}= 0.15 \cdot i_{\text{stress}} - 0.08 \cdot i_{\text{social}} \\
\Delta i_{\text{longing}} &\mathrel{+}= 0.15 \cdot i_{\text{loneliness}} \\
\Delta i_{\text{social}} &\mathrel{+}= 0.08 \cdot i_{\text{energy}} \\
\Delta i_{\text{fatigue}} &\mathrel{+}= 0.10 \cdot i_{\text{stress}} - 0.10 \cdot i_{\text{social}}
\end{aligned}
$$

**自阻尼（每维度独立，2026-06-20 修复）：**

$$
\gamma_{\text{decay}} = [0.10, 0.12, 0.12, 0.12, 0.12, 0.12, 0.10, 0.12]
$$

$$
\Delta_{\text{coupling}} = C_{\text{int}} \cdot h - \gamma_{\text{decay}} \odot (h - \mu_{\text{decay}})
$$

**刺激项：**

$$
\Delta_{\text{stimulus}} = (\beta_{\text{stim}} \odot s_{\text{inner}})^{\mathsf{T}} \cdot B_{\text{int}}
$$

$B_{\text{int}} \in \mathbb{R}^{7 \times 8}$ 密度 $28.6\%$（2026-06-21 去相关化，原 $44.6\%$）。

**完整更新：**

$$
h_t = \text{soft\_clamp}(h_{t-1} + \Delta t \cdot (\alpha \cdot \Delta_{\text{coupling}} + \Delta_{\text{stimulus}}),\; -1,\; 1)
$$

#### 关系状态更新

**耦合速率 $\alpha_{\text{rel}} \in [0.005, 0.06]$：**

$$
\alpha_{\text{rel}} = 0.045 + 0.02 \cdot \tau_{\text{openness}} + 0.015 \cdot r_{\text{trust}}
$$

**刺激接受率 $\beta_{\text{rel}} \in [0.002, 0.06]$：**

$$
\beta_{\text{rel}} = 0.0275 + 0.0075 \cdot \tau_{\text{anxiety}}
$$

**耦合（6 条，含 2 条拮抗负边）：**

$$
\begin{aligned}
\Delta r_{\text{trust}} &\mathrel{+}= 0.08 \cdot r_{\text{affection}} - 0.02 \cdot r_{\text{intimacy}} &\quad \text{(拮抗)} \\
\Delta r_{\text{affection}} &\mathrel{+}= 0.04 \cdot r_{\text{trust}} - 0.02 \cdot r_{\text{intimacy}} &\quad \text{(拮抗)} \\
\Delta r_{\text{intimacy}} &\mathrel{+}= 0.035 \cdot r_{\text{affection}} + 0.04 \cdot r_{\text{trust}}
\end{aligned}
$$

**跨尺度耦合（内 $\to$ 关）：**

$$
\begin{aligned}
\Delta r_{\text{trust}} &\mathrel{+}= -0.03 \cdot i_{\text{stress}} \\
\Delta r_{\text{intimacy}} &\mathrel{+}= 0.015 \cdot i_{\text{stress}} + 0.02 \cdot i_{\text{insecurity}} + 0.02 \cdot i_{\text{loneliness}} \\
\Delta r_{\text{affection}} &\mathrel{+}= 0.015 \cdot i_{\text{energy}}
\end{aligned}
$$

**去相关刺激 B 矩阵（2026-06-21，约束⑥/⑩/⑤ 合规）：**

每维关系态接收不重叠的刺激签名：

$$
\begin{aligned}
\Delta r_{\text{affection}} &= 0.18 \cdot s_{\text{validation}} + 0.10 \cdot s_{\text{closeness}} \\
\Delta r_{\text{trust}} &= -0.25 \cdot s_{\text{conflict}} - 0.10 \cdot s_{\text{abandonment}} \\
\Delta r_{\text{intimacy}} &= 0.15 \cdot s_{\text{dependency}} + 0.10 \cdot s_{\text{teasing}}
\end{aligned}
$$

密度 $6/21 = 28.6\%$（原 $38.1\%$），零共享刺激维度，全正交签名。

**自阻尼：**

$$
\gamma_{\text{rel}} = [0.12,\; 0.12,\; 0.10]
$$

**完整更新：**

$$
\begin{aligned}
\Delta_{\text{coupling}}^{\text{rel}} &= C_{\text{rel}} \cdot r - \gamma_{\text{rel}} \odot r \\
\Delta_{\text{stimulus}}^{\text{rel}} &= \beta_{\text{rel}} \cdot B_{\text{rel}} \cdot s_{\text{inner}} \\
r_t &= \text{soft\_clamp}(r_{t-1} + \Delta t \cdot (\alpha_{\text{rel}} \cdot \Delta_{\text{coupling}}^{\text{rel}} + \Delta_{\text{stimulus}}^{\text{rel}}),\; -1,\; 1)
\end{aligned}
$$

### 4.4 表面投影（带惯性混合）

#### 设计原则（2026-06-22 重构）

- **traits 不直接作为输入**——traits 通过 defense profiles + dynamics 间接影响 surface
- **惯性混合**：`s(t) = α·raw(t) + (1-α)·s(t-1)`，α 由内部状态动态计算
- **双向耦合**：surface → internal 反馈（情绪劳动成本 + 面部反馈 + 表达消耗）
- **outer_stimuli 已是门控产物**——deactivation 已压抑外向通道，surface 直接使用

#### 线性基线 + 惯性混合

$$
\begin{aligned}
s_{\text{expressiveness}} &= -0.3 + 0.4 \cdot i_{\text{energy}} - 0.15 \cdot i_{\text{fatigue}} \\
s_{\text{warmth}} &= -0.2 + 0.4 \cdot r_{\text{affection}} - 0.15 \cdot i_{\text{stress}} \\
&\quad + 0.30 \cdot s_{\text{validation}} + 0.10 \cdot s_{\text{dependency}} \\
s_{\text{sharpness}} &= -0.1 + 0.5 \cdot i_{\text{irritation}} + 0.15 \cdot i_{\text{stress}} \\
&\quad + 0.25 \cdot s_{\text{conflict}} + 0.10 \cdot s_{\text{teasing}} \\
s_{\text{softness}} &= -0.1 + 0.2 \cdot r_{\text{trust}} + 0.20 \cdot s_{\text{closeness}} \\
s_{\text{enthusiasm}} &= -0.2 + 0.5 \cdot i_{\text{energy}} - 0.15 \cdot i_{\text{fatigue}} + 0.15 \cdot s_{\text{validation}} \\
s_{\text{restraint}} &= -0.1 + 0.3 \cdot i_{\text{insecurity}} + 0.20 \cdot i_{\text{stress}} \\
&\quad + 0.20 \cdot s_{\text{emotional\_weight}} \\
s_{\text{vulnerability}} &= -0.5 + 0.3 \cdot i_{\text{loneliness}} + 0.2 \cdot i_{\text{longing}} \\
&\quad + 0.15 \cdot s_{\text{abandonment}}
\end{aligned}
$$

注意：以上公式中**不包含 traits**（原 pride 对 restraint/vulnerability 的直连已移除）。

#### 惯性系数 α

$$
\alpha = \text{clip}(0.5 - 0.3 \cdot \max(0, i_{\text{stress}}) + 0.2 \cdot \max(0, i_{\text{energy}}),\; 0.1,\; 0.9)
$$

- 高压力 → α 下降（表面更"僵"，惯性更强）
- 高精力 → α 上升（表面更"灵"，响应更快）
- 首帧 `prev_surface=None` → 跳过混合，纯 raw 输出

#### 表面→内部反馈（新的 Layer 4）

反馈矩阵 $\text{SURFACE\_FEEDBACK} \in \mathbb{R}^{7 \times 8}$，仅 $surface > 0$ 的维度产生反馈：

$$
\Delta_{\text{internal}} = \max(0, S) \cdot M_{\text{feedback}}
$$

三条反馈机制（trace 量级 0.03–0.06）：

| 机制 | 示例 | 系数 |
|------|------|:----:|
| ① 情绪失调成本 | 克制真实感受 → 压力上升 | +0.06 |
| ② 面部/躯体反馈 | 温暖表达 → 孤独感降低 | -0.04 |
| ③ 表达消耗成本 | 表达外露 → 精力消耗 | -0.06 |

---

## 五、状态格式化（state_formatter_node）

### 5.1 职责

将 4 层数值状态向量翻译为中文"导演描述"，以 SystemMessage 注入 LLM。

### 5.2 格式策略

- **5 级离散描述**（`_desc()` 函数）：值域 $[-1, 1]$ 映射为 5 档
- 每个维度附带极性说明（如 `-1=完全隐藏情绪  0=自然流露  +1=所有情绪都写在脸上`）
- Traits 列表：仅输出显著偏离的维度（$|\tau| > 0.1$）

### 5.3 已知问题

离散 5 级阈值重离散化连续状态空间（P0 已知）。计划重写为连续加权语义投影。

---

## 六、时间衰减

### 6.1 核心公式

$$
\text{decayed}[s] = \mu_s + (h_s - \mu_s) \cdot e^{-\lambda_{\text{eff}}[s] \cdot \Delta t}
$$

### 6.2 有效衰减率

$$
\lambda_{\text{eff}}[s] = \frac{\lambda_{\text{base}}[s] \cdot p_{\text{mod}}}{1 + k \cdot \Delta t}
$$

时间曲线 $1/(1 + k \cdot \Delta t)$ 模拟幂律尾效应。

### 6.3 基础衰减率

#### 内部状态（小时级）

| 维度 | $\lambda_{\text{base}}$ (/h) | 半衰期 |
|------|:---------------------------:|:------:|
| $\text{I\_ENERGY}$ | 0.35 | $\sim$2h |
| $\text{I\_STRESS}$ | 0.23 | $\sim$3h |
| $\text{I\_LONELINESS}$ | 0.17 | $\sim$4h |
| $\text{I\_INSECURITY}$ | 0.14 | $\sim$5h |
| $\text{I\_IRRITATION}$ | 0.69 | $\sim$1h |
| $\text{I\_LONGING}$ | 0.12 | $\sim$6h |
| $\text{I\_SOCIAL\_BATTERY}$ | 0.35 | $\sim$2h |
| $\text{I\_MENTAL\_FATIGUE}$ | 0.23 | $\sim$3h |

#### 关系状态（天级）

| 维度 | $\lambda_{\text{base}}$ (/h) | 半衰期 |
|------|:---------------------------:|:------:|
| $\text{R\_AFFECTION}$ | 0.0021 | $\sim$14 天 |
| $\text{R\_TRUST\_BOND}$ | 0.0021 | $\sim$14 天 |
| $\text{R\_INTIMACY}$ | 0.0041 | $\sim$7 天 |

### 6.4 非对称衰减

负向偏离（$h_s < \mu_s$）的 $\lambda$ 乘以 $1.8$（Fading Affect Bias）。

---

## 七、记忆系统

### 7.1 架构

三层架构：`memory_inject_node` / `memory_summery_node` 在 `nodes.py` 中已完整实现（含 LLM 调用 + JSON 解析 + MemoryStore 持久化），但**尚未注册到 `graph/_builder.py`**，当前流水线未激活记忆系统。

#### 热路径（每轮自动，待接入图）

| 节点 | 触发时机 | 行为 |
|------|---------|------|
| `memory_inject_node` | llm_node 前 | 用户消息 → embedding 检索 → Top-3 记忆 → SystemMessage 注入 |
| `memory_summery_node` | llm_node 后，END 前 | 本轮对话 → LLM 总结(title+summary) → 嵌入 → 存入 MemoryStore |

#### 冷路径（后台）

- 24 条种子记忆在首次 `inject_system_node` 时写入（`_seed_character_memories()`）
- 幂等检查：`store.count() > 0` 跳过重复写入

### 7.2 记忆结构

```python
MemoryNode:
  id: UUID v4
  title: str                    # LLM 生成的标题
  content: str                  # 总结文本
  state_checkpoint: {           # 心理状态快照
    internal_state: (8,),
    relationship_state: (3,),
    surface_state: (7,),
  }
  embedding: ndarray            # qwen3-embedding:8b 嵌入
```

### 7.3 检索

- `search_by_embedding(query, top_k=3, threshold=0.7)` — 余弦相似度
- 结果以"记忆浮现"格式注入（含相似度→熟悉感映射：$\ge 0.90$ = "几乎一模一样"）

### 7.4 未完成项

- `search_by_internal_state()`（普鲁斯特效应）可用但未集成到注入节点
- 无语义合并/遗忘机制（记忆无上限增长）

---

## 八、LLM 模型架构

| 用途 | 模型 | 提供商 | 备注 |
|------|------|--------|------|
| 主对话 | `deepseek-v4-pro` | DeepSeek | 通过 LangChain init_chat_model |
| 感知提取 | `qwen2.5:7b` | Ollama | 本地，3 次重试 |
| 记忆总结 | `qwen2.5:7b` | Ollama | 独立实例 |
| 嵌入 | `qwen3-embedding:8b` | Ollama | 768 维 |

> **更新：** 感知模型已改为独立 Ollama qwen2.5:7b 模型（非 DeepSeek，非 aliased），与主对话模型分离。

---

## 九、约束框架合规状态

### 9.1 约束全集

| # | 约束 | 状态 | 说明 |
|:-:|------|:----:|------|
| ① | Trait 不直接影响状态 | ✅ | **已修复**——traits 从 surface 移除，通过 defense/dynamics 间接 |
| ② | 刺激元属性 | ❌ | StimulusMetadata 不存在 |
| ③ | 矩阵低秩 | ✅ | 均 $\ge 55\%$ |
| ④ | 禁止跨层连线（已修正） | ⚠️ | outer_stimuli→surface 是 by design（已由 defenses 压抑）；traits→surface ✅ **已移除** |
| **⑤** | **语义映射层** | **✅** | **WeightMapper/WeightVector/LinearMapping 覆盖全引擎 250+ 参数** |
| **⑥** | **正交稀疏** | **✅** | **B_int 28.6%, B_rel 28.6%** |
| ⑦ | 谱半径 $\rho < 0.95$ | ✅ | $\rho(C_{\text{int}})=0.099$, $\rho(C_{\text{rel}})=0.058$ |
| **⑧** | **参数审计** | **✅** | **ConstraintRegistry: 14+ 组配置全部注册，新增 SURFACE_FEEDBACK_MATRIX** |
| ⑨ | 全局雅可比 | ✅ | $19.9\% \le 30\%$ |
| ⑩ | 刺激正交性保证 | ⚠️ | 当前 $r=0.015$ ✅，但无代码守卫 |
| ⑪ | 状态格式化连续性 | ❌ | 5 级离散 `_desc()` 硬阈值 |

### 9.2 矩阵审计

| 矩阵 | 形状 | ③ 有效秩比 | ⑥ 密度 | ⑦ 谱半径 | ⑤ WeightMapper |
|------|:----:|:----------:|:------:|:--------:|:----:|
| $B_{\text{int}}$（去相关） | $7 \times 8$ | $77\%$ | ✅ $28.6\%$ | — | ✅ 16 entries |
| $B_{\text{rel}}$（去相关） | $7 \times 3$ | $99\%$ | ✅ $28.6\%$ | — | ✅ 6 entries |
| $C_{\text{int}}$（显式） | $8 \times 8$ | $55\%$ | ✅ $17.2\%$ | ✅ $0.099$ | — |
| $C_{\text{rel}}$（显式） | $3 \times 3$ | $85\%$ | ✅ $66.7\%$（小矩阵例外） | ✅ $0.058$ | — |
| $\text{SURFACE\_FEEDBACK}$（新增） | $7 \times 8$ | — | ✅ trace 密度 | — | ✅ 9 entries |

---

## 十、文件映射

| 文件 | 职责 |
|------|------|
| `agent.py` | TUI 交互入口（LangGraph + SQLite checkpoint） |
| `graph/_builder.py` | 图节点注册、连线、编译 |
| `graph/_routing.py` | 条件边路由：start→inject/perception, perception→state_engine/END |
| `graph/__init__.py` | 导出 compiled_graph |
| `nodes.py` | 7 个节点函数 + 记忆工具函数 |
| `perception.py` | 感知上下文提取 + LLM 调用 + JSON 验证 + 重试 |
| `state.py` | 状态向量索引常量 + 默认基线 + State TypedDict + Pydantic 兼容 |
| `state_engine/_pipeline.py` | update_all / initialize_all 编排 |
| `state_engine/_defenses.py` | Bowlby 防御剖面（逐维权重，刺激特异性） |
| `state_engine/_dynamics.py` | 残差动力学 + setpoint 计算 |
| `state_engine/_surface.py` | 表面投影（惯性混合 + 表面→内部反馈） |
| `state_engine/_surface_weights.py` | SURFACE_MAPPER (18→7 LinearMapping) + SURFACE_FEEDBACK_MATRIX (7×8 WeightMapper) |
| `state_engine/_decay.py` | 时间衰减 + DecayConfig |
| `state_engine/_matrices.py` | $B_{\text{int}}$（$7 \times 8$）+ $B_{\text{rel}}$（$7 \times 3$），均通过 WeightMapper 构建 |
| `state_engine/_validator.py` | WeightMapper + WeightVector + ConstraintRegistry 约束框架实现 |
| `state_engine/_utils.py` | soft_clamp + sigmoid |
| `state_formatter.py` | 数值状态 → 中文导演描述 |
| `memory.py` | MemoryNode + MemoryStore + 3 种检索方法 |
| `llm.py` | 4 个 LLM 模型单例 |
| `config.py` | 感知运行时配置 |
| `prompts/character.py` | 角色 system prompt |
| `prompts/perception.py` | 感知层 system prompt |
| `prompts/memory_summery.py` | 记忆总结 system prompt |
| `prompts/character_memories.py` | 24 条种子记忆锚点 |
| `main.py` | FastAPI stub（仅 health check） |
| `tools/audit_constraints.py` | 约束合规审计脚本 |
| `tests/` | 9 个测试文件，230 测试通过（06-22：+18 surface 测试） |

---

## 十一、已知问题

| 等级 | 问题 | 原因 |
|:----:|------|------|
| 🔴 P0 | Traits 永不更新 | 无 trait 演化机制 |
| 🔴 P0 | StateFormatter 离散化 | 5 级硬阈值损失连续性（约束⑪违反） |
| 🟡 P1 | 无 UserModel | 用户心理剖面缺失 |
| 🟡 P1 | 无目标/意图系统 | 角色纯被动 |
| 🟢 P2 | `_defenses.py` 计算路径使用裸数组而非 WeightVector.values | 12 组 WeightVector 已注册，但 compute_defense_profiles() 仍引用模块级裸数组 |
| 🟢 P2 | 上下文窗口仅 4 条 | context_window=4 |
| 🟢 P2 | 记忆无遗忘机制 | 记忆无上限增长 |
