# Lunar — AI 角色扮演引擎，带有计算心理学状态机

<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-blue" alt="version">
  <img src="https://img.shields.io/badge/python-3.13+-green" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-yellow" alt="license">
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="status">
</p>

**Lunar** 是一个基于 LangGraph 的 AI 角色扮演引擎。与"用 prompt 咒语驱动角色人格"的主流做法不同，Lunar 用**可计算的多层心理状态机**来连续模拟角色的内在情感变化。

角色原型为《崩坏3》「月下誓约·予爱以心」——一个带有依恋焦虑、高自尊、暗中在意但嘴上不说的吸血鬼少女。

## 解决了什么问题

LLM 角色扮演的常见困境：角色人设完全依赖 system prompt，导致人设漂移、情绪不连贯、长期记忆缺失。prompt 越长越容易被模型忽略，越短越扁平。

Lunar 的做法：**把角色的心理状态外置为独立的动力学系统**。LLM 不负责"维持人设"——它只负责把状态向量翻译成自然语言。人设的稳定性由数学模型保证，不由 prompt 保证。

## 核心特性

- **三层心理状态模型**：Internal State（真实感受）→ Relationship State（对用户的关系感知）→ Surface State（外显表达），三者解耦
- **"口是心非"的计算实现**：高 Pride 角色会压抑好感表达，高 Attachment 角色会放大被抛弃恐惧——内部感受与表面表达不一致
- **矩阵驱动的连续动力学**：所有状态转移由矩阵方程描述（`h_t = f(h_{t-1}, stimuli, traits)`），没有 if-else 决策树
- **人格稳定但状态可变**：10 维 Traits 决定角色的"底色"，8 维 Internal + 6 维 Relationship 随对话持续演化
- **门控防御机制**：Suppression、Vulnerability、Attachment 三向门控控制刺激如何进入、如何表达
- **LLM 感知层**：自动从用户输入中提取 7 维心理刺激（被抛弃感、被认可感、亲密靠近、冲突……）
- **SQLite 持久化**：通过 LangGraph Checkpointer，状态跨会话保持

## 环境要求

- **Python** ≥ 3.13
- **Ollama** 运行中，已拉取 `qwen2.5:7b`（用于感知节点）
- **DeepSeek API Key**（用于主对话 LLM）
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

# 4. 确保 Ollama 运行并拉取模型
ollama pull qwen2.5:7b
```

## 快速开始

```bash
# 启动交互式对话（TUI）
uv run python agent.py
```

首次运行会自动创建 `db/lunar.db`（SQLite 状态持久化）。在 TUI 中输入消息即可与角色对话。

## 架构概览

```
用户输入 → Perception Node → State Engine → State Formatter → LLM Node → 回复
              │                    │
        提取心理刺激          多层状态更新
        (7 维向量)           (Internal / Relationship
                             → Surface 投影)
```

| 节点 | 职责 | 模型 |
|------|------|------|
| Perception | 从用户输入提取 7 维心理刺激 | Ollama `qwen2.5:7b` |
| State Engine | 7 层纯函数管线，更新内部/关系/表面状态 | —（纯数学） |
| State Formatter | 数值状态 → 中文"导演笔记" | —（纯规则） |
| LLM | 注入角色人设 + 状态描述，生成回复 | DeepSeek `deepseek-v4-pro` |

详细架构见 [CLAUDE.md](CLAUDE.md) 和 [state_engine/](state_engine/)。

## 项目结构

```
Lunar/
├── agent.py              # TUI 交互入口
├── nodes.py              # LangGraph 节点函数（5 个）
├── perception.py         # 感知层：心理刺激提取 + 验证 + 重试
├── state.py              # 状态类型定义（向量索引常量 + TypedDict）
├── state_formatter.py    # 状态格式化：数值 → "导演笔记"
├── llm.py                # LLM 模型初始化（DeepSeek + Ollama）
├── config.py             # 运行时配置
├── main.py               # FastAPI 入口（开发中）
├── graph/                # LangGraph 图定义与路由
├── state_engine/         # 7 层心理状态管线
│   ├── _pipeline.py      #   管线编排
│   ├── _gates.py         #   门控计算与应用
│   ├── _dynamics.py      #   内部 & 关系动力系统
│   ├── _decay.py         #   动态衰减
│   ├── _surface.py       #   表面投影
│   ├── _matrices.py      #   耦合矩阵工厂
│   └── _utils.py         #   数值工具
├── prompts/              # Prompt 数据（角色人设 + 感知格式）
└── db/                   # SQLite 持久化
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 图编排 | LangGraph |
| 主对话模型 | DeepSeek (`deepseek-v4-pro`) |
| 感知模型 | Ollama (`qwen2.5:7b`) |
| 状态向量 | numpy `ndarray` |
| 持久化 | SQLite (`langgraph-checkpoint-sqlite`) |
| 包管理 | uv (Python 3.13) |

## 贡献

本项目处于早期开发阶段。欢迎提 Issue 或 PR。

改进方向和已知问题见 [TODO.md](TODO.md)。

## 许可证

MIT © 2025 Lunar Dev
