# WeightMapper 实现报告

> 2026-06-22 | 约束⑤语义映射层 + 约束⑧参数审计 | 16+ 组配置全部注册（含 SURFACE_MAPPER + FEEDBACK）

---

## 一、架构总览

```
WeightMapper (2D矩阵)          WeightVector (1D向量)
       │                             │
       ├─ connect()                  ├─ connect()
       │   注册 SemanticWeight        │   注册轻量条目
       │                             │
       └─ build_matrix()             └─ build()
            │                             │
            ▼                             ▼
       ConstraintRegistry — 中央约束注册表
            │
            ├─ register()    — 注册检查函数
            ├─ run_all()     — 执行全部检查
            ├─ audit_report() — 文本审计报告
            └─ structured_report() — JSON 审计数据
```

### 1.1 核心数据类型

```python
@dataclass(frozen=True)
class SemanticWeight:
    source_idx: int       # 源概念索引
    target_idx: int       # 目标概念索引
    value: float          # 数值（必须 ∈ domain）
    direction: str        # "+"/"-"/"0"
    magnitude: str        # "strong"/"moderate"/"weak"/"trace"
    domain: tuple         # 允许值区间 (low, high)
    rationale: str        # 心理依据
    origin: str           # "theory"/"calibrated"/"legacy"
    reviewed: str         # "YYYY-MM-DD"
```

创建时自动验证：
- direction 与 value 符号一致
- `value ∈ domain`
- magnitude 的建议范围匹配

### 1.2 WeightMapper — 2D 矩阵映射器

用于 B_int、B_rel 等二维矩阵。链式调用 API：

```python
mapper = WeightMapper("INPUT_INFLUENCE_B", ST_LABELS, I_LABELS)
B = (mapper
    .connect(ST_ABANDONMENT, I_INSECURITY, 0.28, "strong", (0.20, 0.40),
             "抛弃激活不安全感核心 (Bowlby IWM)", "theory")
    .connect(ST_VALIDATION, I_ENERGY, 0.22, "strong", (0.15, 0.30),
             "被认可→精力充沛 (Bandura)", "theory")
    .build_matrix((7, 8), skip_rank=True, skip_orthogonality=True))
```

### 1.3 WeightVector — 1D 权重向量映射器

用于防御剖面权重、衰减率等非矩阵形式的逐维权重：

```python
vec = WeightVector("STRESS_DEACT_A", ST_LABELS)
vec.connect("stress", ST_CONFLICT, 0.12, "weak", (0.05, 0.20),
            "核心：压力→逃避冲突", "theory")
arr = vec.build()  # → np.ndarray(7,)
```

### 1.4 ConstraintRegistry — 中央约束注册表

```python
# 自动注册：build_matrix() / build() 时自动执行
ConstraintRegistry.register("MY_MATRIX", assert_sparsity, M, label)
results = ConstraintRegistry.run_all("MY_MATRIX")
print(ConstraintRegistry.audit_report())          # 文本
report = ConstraintRegistry.structured_report()   # JSON
```

---

## 二、已注册配置（16+ 组）

### 📐 矩阵（4）

| 名称 | 形状 | 条目 | 密度 | origin 分布 |
|------|:----:|:----:|:----:|:-----------:|
| `INPUT_INFLUENCE_B` | 7×8 | 16 | 28.6% | theory 13, calibrated 3 |
| `REL_INPUT_INFLUENCE_B` | 7×3 | 6 | 28.6% | theory 6 |
| `SURFACE_MAPPER`（LinearMapping） | **18→7** | 7 bias + 25 weight | — | theory/calibrated |
| `SURFACE_FEEDBACK_MATRIX` | **7×8** | 9 | trace | calibrated 9 |

### 📏 向量（12）

| 名称 | 维度 | 源变量 | 类型 | origin |
|------|:----:|--------|:----:|:------:|
| `STABILITY_DEACT_A` | 7 | emotional_stability | 加法 | theory 4, calibrated 3 |
| `OPENNESS_DEACT_A` | 7 | emotional_openness | 加法 | theory 2, calibrated 5 |
| `AVOIDANCE_DEACT_A` | 7 | attachment_avoidance | 加法 | theory 4, calibrated 3 |
| `STRESS_DEACT_A` | 7 | stress | 加法 | theory 3, calibrated 4 |
| `INSECURITY_DEACT_A` | 7 | insecurity | 加法 | theory 3, calibrated 4 |
| `TRUST_BOND_DEACT_M` | 7 | trust_bond | **乘法** | theory 6, calibrated 1 |
| `SENSITIVITY_HYPER_A` | 7 | sensitivity | 加法 | theory 6, calibrated 1 |
| `AVOIDANCE_HYPER_A` | 7 | attachment_avoidance | 加法 | theory 5, calibrated 2 |
| `INSECURITY_HYPER_A` | 7 | insecurity | 加法 | theory 3, calibrated 4 |
| `LONGING_HYPER_A` | 7 | longing | 加法 | theory 4, calibrated 3 |
| `AFFECTION_HYPER_M_NEW` | 7 | affection | **乘法** | theory 5, calibrated 2 |
| `INTIMACY_HYPER_M` | 7 | intimacy | **乘法** | theory 5, calibrated 2 |

**全局统计：** theory 75 entries (82%), calibrated 16 entries (18%), **legacy 0**.

---

## 三、JSON 权重外部化

所有 14 组配置可通过 `export_all("params/")` 导出为 JSON，通过 `load_all("params/")` 重加载。

### 导出格式示例

```json
// params/input_influence_b.json
{
  "mapper_type": "matrix",
  "label": "INPUT_INFLUENCE_B",
  "sources": ["abandonment_stimulus", "validation_stimulus", ...],
  "targets": ["energy", "stress", "loneliness", ...],
  "entries": [
    {
      "source": "abandonment_stimulus",
      "target": "insecurity",
      "value": 0.28,
      "direction": "+",
      "magnitude": "strong",
      "domain": [0.20, 0.40],
      "rationale": "抛弃直接激活不安全感核心 (Bowlby IWM, 1980)",
      "origin": "theory",
      "reviewed": "2026-06-21"
    }
  ]
}
```

### 矩阵一致性验证

原始 B_int 与 JSON 重加载后 B_int 的差值矩阵：
```python
np.array_equal(B_orig, B_reload)  # → True ✅
```

### 反向查询示例

| 查询 | 结果 |
|------|------|
| `lookup(source="abandonment_stimulus")` | 3 targets: insecurity, loneliness, longing |
| `lookup(target="insecurity")` | 2 sources: abandonment, validation |

---

## 四、约束合规情况

| 约束 | 状态 | 说明 |
|:----:|:----:|------|
| **⑤ 语义映射层** | ✅ | WeightMapper/WeightVector 骨架实现，B_int + B_rel + 12 组防御权重全部通过 |
| **⑧ 参数审计** | ✅ | ConstraintRegistry 实现，14 组配置全部注册，零 legacy |

### 仍为裸赋值的部分（待迁移）

| 文件 | 代码位置 | 规则数 | 优先级 |
|------|---------|:------:|:------:|
| `_dynamics.py` | 耦合规则 (11 条内部 + 6 条关系 + 5 条跨尺度) | 22 | P2 |
| `_dynamics.py` | setpoint 公式 (8 + 3 条) | 11 | P2 |
| ~~`_surface.py`~~ | ~~表面投影规则 (15 条)~~ | — | ✅ **已迁移至 SURFACE_MAPPER** |
| `_decay.py` | 衰减参数 (internal_lambda 8 + rel_lambda 3) | 11 | P3 |

以上显式命名规则和公式化参数，迁移优先级低于 B 矩阵和防御剖面——它们不需要去相关化的数学保证，只需 provenance 标注。
表面投影规则已迁移至 `_surface_weights.py:SURFACE_MAPPER`（06-21），外加 `SURFACE_FEEDBACK_MATRIX`（06-22）。

---

## 五、代码位置

| 文件 | 行数 | 内容 |
|------|:----:|------|
| `state_engine/_validator.py` | ~460 | SemanticWeight + WeightMapper + WeightVector + ConstraintRegistry + JSON 导入/导出 |
| `state_engine/_matrices.py` | ~220 | B_int + B_rel 通过 WeightMapper 构建（替换裸赋值） |
| `state_engine/_defenses.py` | +440 | 12 组防御剖面权重通过 WeightVector 注册（新增 `_register_defense_weights()`） |
| `state_engine/_dynamics.py` | 2 | 导入 REL_INPUT_INFLUENCE_B（替换内联 B_rel 计算） |
| `state_engine/_surface_weights.py` | ~200 | SURFACE_MAPPER (LinearMapping 18→7, 3 源组) + SURFACE_FEEDBACK_MATRIX (WeightMapper 7→8) |
| `tools/audit_constraints.py` | 5 | 导入 REL_INPUT_INFLUENCE_B（替换硬编码副本） |

---

## 六、使用方法

```bash
# 查看审计报告
uv run python -c "from state_engine._validator import ConstraintRegistry; print(ConstraintRegistry.audit_report())"

# 导出全部权重为 JSON
uv run python -c "from state_engine._validator import export_all; print(export_all('params'))"

# 从 JSON 重加载
uv run python -c "
from state_engine._validator import load_all
mappers = load_all('params')
B = mappers['INPUT_INFLUENCE_B'].build_matrix((7, 8))
"
```

### 扩展指引

添加新权重时，不应使用裸赋值：

```python
# ❌ 禁止
M[i, j] = 0.25

# ✅ 正确
mapper = WeightMapper("MY_MATRIX", source_labels, target_labels)
mapper.connect(i, j, 0.25, "moderate", (0.10, 0.35),
               rationale="心理学依据", origin="theory")
M = mapper.build_matrix(shape)
```
