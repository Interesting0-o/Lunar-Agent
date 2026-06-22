# Lunar — AI 角色扮演引擎，带有可解释的心理学状态机

<p align="center">
  <img src="https://img.shields.io/badge/version-0.2.0-blue" alt="version">
  <img src="https://img.shields.io/badge/python-3.13+-green" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-yellow" alt="license">
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="status">
</p>

**Lunar** 是一个基于 LangGraph 的 AI 角色扮演引擎。与"用 prompt 咒语驱动角色人格"的主流做法不同，Lunar 用**可计算的多层心理状态机**来连续模拟角色的内在情感变化。

角色原型为《崩坏3》「月下誓约·予爱以心」——一个带有依恋焦虑、高自尊、暗中在意但嘴上不说的吸血鬼少女。

## 核心特性

- **三层心理状态模型**：Internal State（真实感受）→ Relationship State（对用户的关系感知）→ Surface State（外显表达），三者解耦
- **"口是心非"的计算实现**：基于 Bowlby (1980) 依恋防御二分法——Deactivation（去激活）压抑外在表达，Hyperactivation（过度激活）放大内心感受。高 Pride + 高 Attachment Anxiety 的角色内心翻江倒海但表面波澜不惊
- **残差动力学驱动**：所有状态更新使用 `h_t = h_{t-1} + dt · (耦合 + 刺激 + 稳态)` 形式，没有 if-else 决策树。耦合关系通过显式命名规则定义，每条附带心理学注释
- **人格稳定但状态可变**：10 维 Traits 决定角色的"底色"，通过调制速率参数（α, β, γ）而非直接叠加来影响状态演化
- **时间感知衰减**：基于真实时间间隔（Δt 小时）的指数衰减，不同维度有独立的衰减速率和人格化目标基线
- **LLM 感知层**：自动从用户输入中提取 7 维心理刺激（被抛弃感、被认可感、亲密靠近、冲突、依赖、调侃、情感重量），3 轮重试 + JSON 验证
- **约束宪法 —— 防止黑盒化**：所有矩阵必须通过 9 条严格约束才能存在（详见下文"状态引擎约束框架"）
- **SQLite 持久化**：通过 LangGraph Checkpointer，状态跨会话保持

## 环境要求

- **Python** ≥ 3.13
- **Ollama** 运行中，已拉取 `nomic-embed-text`（用于记忆系统的嵌入检索）
- **DeepSeek API Key**（用于主对话 LLM 和感知模型）
- **uv** 包管理器（推荐）或 pip

## 安装

```bash
# 1. 克隆仓库
git clone <repo-url>
cd Lunar

# 2. 安装依赖（uv）
uv sync

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 4. 确保 Ollama 运行并拉取嵌入模型
ollama pull nomic-embed-text
```

## 快速开始

```bash
# 启动交互式对话（TUI）
uv run python agent.py
```

首次运行会自动创建 `db/lunar.db`（SQLite 状态持久化）。在 TUI 中输入消息即可与角色对话。

## 架构概览

```
START → inject_system → perception → state_engine → state_formatter → llm → END
                              │ error=True → END（跳过 state_engine 和 formatter）
```

| 节点 | 职责 | 模型 |
|------|------|------|
| `inject_system` | 首次运行时注入角色人设 + Traits（仅执行一次） | — |
| `perception` | 从用户输入提取 7 维心理刺激（3 轮重试 + 递增强调） | DeepSeek |
| `state_engine` | **3 步管道**：① Bowlby 防御剖面 → ② 残差动力学(内部+关系) → ③ 表面投影 | —（纯数学） |
| `state_formatter` | 数值状态向量 → 中文"导演笔记"（离散 5 级阈值） | —（纯规则） |
| `llm` | 注入角色人设 + 状态描述，生成回复 | DeepSeek |

## 状态引擎（3 步管道）

旧版 7 层管线（含门控系统）已重构为基于 Bowlby 依恋防御理论的 3 步管道：

```
① Defense Profiles（防御剖面）
   compute_defense_profiles() → profiles (2, 7)
   apply_defenses() → inner_stimuli, outer_stimuli
   
   基于 Bowlby (1980) 的依恋防御二分法：
   - Deactivation（去激活）：高回避→压抑外在表达
   - Hyperactivation（过度激活）：高焦虑→放大内心感受

② Residual Dynamics（残差动力学）
   update_internal_state() → new_internal (8,)
   update_relationship_state() → new_relationship (3,)
   
   公式: h_t = h_{t-1} + dt · (α·耦合 + β·刺激 + γ·稳态)
   - 耦合通过显式命名规则定义（每行附带心理学注释）
   - 防御剖面调制 β 和 γ 速率
   - 稳态恢复由时间衰减（_decay.py）负责，不参与每轮更新

③ Surface Projection（表面投影）
   project_surface() → surface_state (7,)
   
   内部状态 + 关系状态 → 可观测的 7 维表达
```

对比旧版门控系统（已被移除）：

| 旧版 | 新版 | 原因 |
|------|------|------|
| 4 门并行（suppression/vulnerability/attachment/leakage） | 2 维防御剖面（deactivation/hyperactivation） | Bowlby 理论依据 + PCA 验证有效维度从 ~1.5 提升至 ~4+ |
| 全局标量调制 | 刺激特异性逐维权重 | 避免维度同步漂移塌缩 |
| 独立门控逻辑 | 统一的 `compute_defense_profiles()` | 简化认知负担 |

## 状态引擎约束框架

> Lunar 的约束宪法——防止系统退化为不可解释的黑盒。

状态引擎的所有参数和矩阵必须通过以下约束才能存在。约束在 `state_engine/_validator.py` 的 `ConstraintRegistry` 中集中执行，每次 `build_matrix()` 调用时自动全量检查，失败则抛出 `ConstraintViolationError`。

### 语义架构层（保证信息流的意图透明）

| # | 约束 | 含义 |
|---|------|------|
| ① | **Trait 不直接影响状态** | Trait 只调制速率参数（α, β, γ），不参与状态更新方程的主项 |
| ② | **刺激携带元属性** | 每维刺激携带置信度、来源编码、衰减调节因子 |
| ④ | **禁止跨层直接连线** | Surface 只看 Relationship State（不跨层读 Internal 或 Traits） |
| ⑤ | **语义映射层** | 禁止裸数值 `B[i,j] = 0.25`——所有参数通过 `WeightMapper` 声明语义关系 |

### 数学保证层（保证系统的结构透明）

| # | 约束 | 含义 |
|---|------|------|
| ③ | **矩阵低秩** | 有效秩远小于名义维度，耦合由少量潜在因子驱动 |
| ⑥ | **正交稀疏** | 密度 ≤ 30%，行 Gram 矩阵非对角元素 < 0.3 |
| ⑦ | **谱半径 ρ < 0.95** | 系统是收缩映射，不会发散 |
| ⑨ | **全局雅可比稀疏** | 组合矩阵的传播路径数 ≤ 5/对——这是最重要的约束，防止"单个矩阵都通过，乘在一起变黑盒" |

### 流程透明层（保证参数的历史透明）

| # | 约束 | 含义 |
|---|------|------|
| ⑧ | **参数审计** | 每个参数有 provenance（来源、依据、审查日期），无 `origin=legacy` 参数 |

> 详细定义见 [`md/ARCHITECTURE.md`](md/ARCHITECTURE.md)「约束框架」章节。

## 项目结构

```
Lunar/
├── agent.py                 # TUI 交互入口
├── nodes.py                 # LangGraph 节点函数（5 个活跃 + 2 个存根）
├── perception.py            # 感知层：心理刺激提取 + JSON 验证 + 3 轮重试
├── state.py                 # 状态类型定义（向量索引常量 + TypedDict + 默认值）
├── state_formatter.py       # 状态格式化：数值 → "导演笔记"
├── llm.py                   # LLM 模型初始化（DeepSeek）
├── config.py                # 运行时配置（感知重试参数等）
├── main.py                  # FastAPI 入口（开发中）
├── graph/                   # LangGraph 图定义与路由
│   ├── _builder.py          #   图构造
│   ├── _routing.py          #   条件路由
│   └── __init__.py          #   编译后的导出
├── state_engine/            # 3 步心理状态管道
│   ├── _pipeline.py         #   管道编排（update_all/initialize_all）
│   ├── _defenses.py         #   防御剖面（Bowlby 二分法）← 替代旧门控系统
│   ├── _dynamics.py         #   内部 & 关系动力系统（残差形式）
│   ├── _decay.py            #   时间感知衰减（真实 Δt 驱动）
│   ├── _surface.py          #   表面投影（Internal + Relationship → 表达）
│   ├── _matrices.py         #   耦合矩阵工厂（将被 WeightMapper + Mapper 替代）
│   ├── _validator.py        #   [规划中] 约束检查注册表 + 全局雅可比验证
│   ├── _mapper.py           #   [规划中] 语义映射层（WeightMapper）
│   └── _utils.py            #   数值工具（soft_clamp, sigmoid）
├── prompts/                 # Prompt 数据
│   ├── character.py         #   角色人设 SYSTEM_PROMPT
│   ├── perception.py        #   感知系统 prompt
│   └── memory_summery.py    #   记忆系统 prompt（部分实现）
├── memory.py                # 记忆系统（向量/嵌入/混合检索）
├── tests/                   # 8 个测试文件 + conftest.py
├── db/                      # SQLite 持久化
└── md/                      # 设计文档
    ├── ARCHITECTURE.md              # 架构 + 时间衰减 + 约束宪法
    ├── STATE_ENGINE_RESEARCH_REPORT.md
    ├── DEFENSE_PROFILE_METHODOLOGY.md
    └── ...
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 图编排 | LangGraph |
| 主对话 + 感知模型 | DeepSeek (`deepseek-v4-pro`) |
| 嵌入模型（记忆检索） | Ollama (`nomic-embed-text`) |
| 状态向量 | NumPy `ndarray` |
| 持久化 | SQLite (`langgraph-checkpoint-sqlite`) + JSON |
| 包管理 | uv (Python 3.13) |
| 约束执行 | `ConstraintRegistry`（权重参数全生命周期审计）|

## 与常规 prompt-based 角色扮演的对比

| 维度 | 常规方案 | Lunar |
|------|---------|-------|
| 人设维持 | 依赖 system prompt + 模型 adherence | 数学模型保证（Traits + 动力学） |
| 情绪连贯性 | 高 prompt 长度下劣化 | 状态向量自然连续演化 |
| "口是心非" | 需要显式描述 | 防御机制 + Surface 投影自动产生 |
| 长期行为一致性 | 随着对话长度指数劣化 | 时间衰减 + Setpoint 约束稳定域 |
| 参数可解释性 | 黑盒（prompt 中的隐性 bias） | 9 条约束保证每个参数可追溯 |
| 可控性 | prompt 工程（试错） | 调 Traits/矩阵系数，行为变化可推理 |

## 已知限制

- P0: SurfaceState 未注入 LLM prompt（`state_description` 只涵盖 Internal + Relationship）
- P0: 记忆系统虽作为库完整实现，但未接入 LangGraph 管道
- P0: 离散 5 级阈值格式器（`_desc()`）将连续状态重离散化——计划重写为连续加权投影
- P1: 所有权重矩阵硬编码在代码中（`DecayConfig` 展示了外置化模式但未全面推广）——`WeightMapper` 实施后将解决

## 贡献

本项目处于早期开发阶段。欢迎提 Issue 或 PR。

改进方向和已知问题见 [ROADMAP.md](ROADMAP.md)。约束框架的完整定义见 [`md/ARCHITECTURE.md`](md/ARCHITECTURE.md)「约束框架」章节。

## 许可证

MIT © 2025 Lunar Dev
