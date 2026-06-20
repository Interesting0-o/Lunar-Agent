# 时间感知衰减组件设计

> 基于情感动力学和人际关系时间衰减的学术研究。
>
> **实现状态**：✅ 全部已实现（见 `_decay.py`）。`negative_decay_boost=1.8` 非对称衰减于 06-19 新增。
> 详情见测试报告可视化和 `tests/result/`。

---

## 1. 动机

### 当前状态

State Engine 的稳态恢复机制是**恒速率**的:

```python
# _dynamics.py — 每轮固定 γ 速率，无时间感知
gamma = 0.08 + traits[STABILITY] * 0.10 + ...
delta_homeostatic = setpoint - current
delta = ... + gamma * delta_homeostatic
new_state = current + delta  # dt 恒为 1.0
```

问题: 用户离开 5 分钟和离开 3 天，状态的恢复量完全相同。

### 目标

用**真实时间戳**驱动衰减，使得:
- 秒级间隔 → 几乎无衰减
- 小时级间隔 → 明显恢复
- 天级间隔 → 大幅回归基线

---

## 2. 学术依据

### 2.1 衰减函数形式

| 模型 | 公式 | 特点 | 文献 |
|------|------|------|------|
| **指数衰减** | `f(Δt) = e^(-λ·Δt)` | 计算简单、与 RL/VAR 框架兼容、短期拟合好 | Rutledge et al. (2014), Vanhasbroeck et al. (2024) |
| **幂律衰减** | `f(Δt) = (1 + λ·Δt)^(-α)` | 长期保留更好、快速初衰减+缓慢尾衰减 | Hong & Zhang (2025), Elliott & Anderson |
| **拟双曲衰减** | `f(Δt) = β·δ^Δt + (1-β)·(1+λ·Δt)^(-α)` | 结合两者优势 | Laibson (1997), Vanhasbroeck et al. (2024) |

**选择: 混合指数衰减** — 指数形式为主，但 λ 是时间依赖的（长间隔时 λ 降低，模拟幂律尾部的缓慢衰减）:

```
decay_factor(s, Δt) = exp(-λ_effective(s, Δt) × Δt)
λ_effective(s, Δt) = λ_base(s) × transition(Δt)
transition(Δt) = 1 / (1 + 0.1 × Δt)  # 衰减速率随时间放缓
```

这保留了指数衰减的计算稳定性，同时引入了长间隔的"尾衰减放缓"效应。

### 2.2 情绪衰减率的人格差异

| 发现 | 文献 |
|------|------|
| 杏仁核恢复速度预测神经质水平——恢复越慢，神经质越高 | Schuyler et al. (2014) |
| 高神经质 → 更慢的压力恢复 | Lücke et al. (2024) |
| 高伤害回避 → 对负面刺激恢复更慢 | Mardaga et al. (2006) |
| 不同情绪有不同的半衰期 (40s ~ 319s) | EDM 2025 教育数据挖掘 |

**设计含义**: λ_base[s] 应因维度而异，并被 traits 调制。

### 2.3 人际关系的时间衰减

| 发现 | 文献 |
|------|------|
| 手机通话数据显示: 时间间隔越长，下次通话时间越长（补偿行为） | Bhattacharya et al. (2017) |
| ~70% 的人认同"小别胜新婚"，但 20-40% 经历过关系恶化 | Pellegrini (1977), Knox et al. (2002) |
| 关系衰减比情绪衰减慢 3-4 个数量级 | DER 模型 (Tanguy et al., 2007) |

**设计含义**: 关系维度的 λ 比内部情绪维度小 100-1000 倍。

### 2.4 多时间尺度架构 (DER)

Tanguy et al. (2007) 的三层时间尺度:

| 层 | 时间尺度 | Lunar 对应 |
|----|---------|-----------|
| Behavior activations | 秒 | — (LLM 生成) |
| Emotions | 分钟-小时 | InternalState (8D) |
| Moods | 小时-天 | InternalState + Traits |
| Relationships | 天-周 | RelationshipState (6D) |

---

## 3. 数学设计

### 3.1 核心公式

```
decayed[s] = baseline[s] + (current[s] - baseline[s]) × exp(-λ_eff[s] × Δt)

其中:
  baseline[s]       = 人格决定的稳态基点 (compute_setpoint / compute_rel_setpoint)
  current[s]        = 当前状态值
  Δt                = 自上次更新以来的实际时间 (单位: 小时)
  λ_eff[s]          = 有效衰减率 (/小时)
```

### 3.2 有效衰减率

```
λ_eff(s, Δt) = λ_base[s] × personality_mod[s] × time_curve(Δt)

λ_base[s]:        维度基础衰减率 (查表)
personality_mod[s]: 人格调制因子 (traits → 缩放 λ)
time_curve(Δt):    时间曲线 = 1 / (1 + k × Δt)  — 长间隔放缓衰减速率
                    其中 k = 0.05 (内部) 或 k = 0.001 (关系)
```

### 3.3 人格调制

内部状态:
```
personality_mod[s] = 1.0
  - traits[STABILITY]       × 0.30    # 稳定→衰减快 (+)
  - traits[OPTIMISM]         × 0.15    # 乐观→衰减快 (+)
  + traits[ANXIETY_PRONENESS] × 0.25   # 焦虑→衰减慢 (−)
  + traits[ANGER_REACTIVITY]  × 0.10   # 易怒→衰减慢 (−)
  - traits[OPENNESS]         × 0.10    # 开放→衰减快 (+)
```

关系状态:
```
personality_mod[s] = 1.0
  + traits[AVOIDANCE]      × 0.35    # 回避→衰减快 (不联系就疏远)
  - traits[ATTACH_ANXIETY] × 0.20    # 焦虑→衰减慢 (放不下)
  + traits[STABILITY]      × 0.10    # 稳定→衰减慢 (关系稳定)
```

### 3.4 基础衰减率 (半小时)

| InternalState | λ_base (/h) | 半衰期 (默认) | 最快半衰期 | 最慢半衰期 |
|--------------|-------------|-------------|-----------|-----------|
| ENERGY | 0.35 | ~2h | 1h | 5h |
| STRESS | 0.23 | ~3h | 1.5h | 8h |
| LONELINESS | 0.17 | ~4h | 2h | 12h |
| INSECURITY | 0.14 | ~5h | 2.5h | 16h |
| IRRITATION | 0.69 | ~1h | 30min | 3h |
| LONGING | 0.12 | ~6h | 3h | 20h |
| SOCIAL_BATTERY | 0.35 | ~2h | 1h | 5h |
| MENTAL_FATIGUE | 0.23 | ~3h | 1.5h | 8h |

| RelationshipState | λ_base (/h) | 半衰期 (默认) |
|-------------------|-------------|-------------|
| AFFECTION | 0.0021 | ~14 天 |
| TRUST | 0.0014 | ~21 天 |
| FAMILIARITY | 0.0041 | ~7 天 |
| DEPENDENCY | 0.0029 | ~10 天 |
| EMOTIONAL_SAFETY | 0.0021 | ~14 天 |
| ROMANTIC_TENSION | 0.0058 | ~5 天 |

---

## 4. 与现有 State Engine 的集成

### 4.1 修改流水线

```
Before:  state_engine_node → update_all()
After:   state_engine_node → apply_time_decay() → update_all() → save_timestamp()
```

### 4.2 State 新增字段

```python
class State(TypedDict):
    # ... 现有字段 ...
    last_active_timestamp: Optional[float]  # Unix timestamp of last state update
```

### 4.3 调用方式

```python
# nodes.py — state_engine_node
import time

current_ts = time.time()
last_ts = state.get("last_active_timestamp", current_ts)
delta_hours = (current_ts - last_ts) / 3600.0

# 先衰减，再更新
if delta_hours > 0.01:  # 超过 ~36 秒才衰减
    decayed_internal = apply_time_decay(
        current_internal, traits, "internal", delta_hours
    )
    decayed_rel = apply_time_decay(
        current_relationship, traits, "relationship", delta_hours
    )
else:
    decayed_internal = current_internal
    decayed_rel = current_relationship

result = update_all(decayed_internal, decayed_rel, traits, stimuli)
result["last_active_timestamp"] = current_ts
```

### 4.4 与残差动力学的互动

时间衰减和残差动力学各司其职:
- **时间衰减** (apply_time_decay): 处理"无交互期间"的自然恢复/退化 → 由 `Δt` 驱动
- **残差动力学** (update_all): 处理"有交互期间"的刺激响应 → 由 `stimuli` 驱动

两者不冲突——时间衰减在每轮开头"先让状态随时间自然变化"，然后残差动力学在此基础上施加本轮刺激的影响。

---

## 5. 衰减曲线可视化

实际衰减曲线已由测试套件生成，保存在 `tests/result/`：

| 图 | 文件 | 内容 |
|----|------|------|
| λ_base 对比 | `lambda_base_comparison.png` | 内部状态 8 维 λ_base 对比 |
| 非对称衰减 | `asymmetric_decay_curves.png` | 信任 vs 升温恢复曲线，半衰期标注 |
| 人格调制分布 | `personality_modulation_impact.png` | 5000 随机人格的调制因子分布 |
| 时间曲线阻尼 | `time_curve_damping.png` | 1/(1+k·Δt) 阻尼曲线 |
| Boost 参数扫描 | `boost_sweep.png` | boost=1.0~4.0 恢复曲线族 |

运行：`uv run pytest tests/test_decay.py::TestVisualization -v -s`

---

## 6. 参考文献

1. **Rutledge et al.** (2014). A computational and neural model of momentary subjective well-being. *PNAS*, 111(33), 12252–12257.
2. **Vanhasbroeck, N., Loossens, T., & Tuerlinckx, F.** (2024). Two peas in a pod: Discounting models as a special case of the VARMAX. KU Leuven.
3. **Schuyler et al.** (2014). Temporal dynamics of emotional responding: amygdala recovery predicts emotional traits. *Social Cognitive and Affective Neuroscience*.
4. **Lücke et al.** (2024). Neuroticism, emotional stress reactivity and recovery in daily life. *Journal of Research in Personality*.
5. **Hong & Zhang** (2025). Nonlinear Dynamical Model of Emotional Propagation Based on Caputo Derivative. *Mathematics*, 13, 2044.
6. **Tanguy, E., Willis, P., & Bryson, J.** (2007). A Dynamic Emotion Representation Model within a Facial Animation System. *IJCAI 2007*.
7. **Zhang, J., Zheng, J., & Magnenat-Thalmann, N.** (2015). PCMD: personality-characterized mood dynamics model. *Computer Animation and Virtual Worlds*, 26, 237–245.
8. **Bhattacharya et al.** (2017). Absence makes the heart grow fonder: social compensation when failure to interact risks weakening a relationship. *EPJ Data Science*, 6.
9. **Pellegrini, R. J.** (1977). Mate Separation and Emotional Attachment: Does Absence Make the Heart Grow Fonder? *Psychological Reports*, 41.
10. **Steephen, J. E.** (2013). HED: A Computational Model of Affective Adaptation and Emotion Dynamics. *IEEE Trans. Affective Computing*, 4(2), 197–210.
