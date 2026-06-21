"""Lunar 人格状态的类型定义。

所有纯 float 参数的 TypedDict 已替换为 numpy 数组。
每个数组的维度索引由对应的命名常量定义（如 I_ENERGY、T_PRIDE 等），
取值见各维度的 LABELS 列表和 LABEL_IDX 映射。

层级划分：
- SurfaceState:       表面状态（即时可感知的表达特征）
- Traits:             核心特质（长期稳定的性格参数）
- InternalState:      内部状态（底层心理指标）
- RelationshipState:  关系状态（对用户的互动感知）
- StimulusVector:     心理刺激（perception 直接从用户输入提取的心理意义，7维）
- State:              顶层状态聚合（保留 TypedDict）

注意：隐藏状态层（HiddenState）、社交信号层（SocialSignals）、互动影响（InteractionImpact）、
门控层（GateVector）已移除。感知节点现在直接输出 StimulusVector，
防御剖面（deactivation/hyperactivation）替代了旧门控的功能。
"""

import numpy as np
from typing import TYPE_CHECKING, List, Optional, Annotated, TypedDict
from langgraph.graph.message import add_messages


# ── Pydantic schema 兼容性 ──
# langgraph dev / Studio 会使用 Pydantic 为 State TypedDict 生成 JSON Schema，
# 但 numpy.ndarray 不是 Pydantic v2 原生支持的类型。
# 这里提供一个兼容类型：类型检查器看到的是 np.ndarray，
# Pydantic 看到的是 list[float]（用于 schema 生成），运行时值为 numpy 数组。
if TYPE_CHECKING:
    _Array = np.ndarray
else:
    try:
        from pydantic_core import core_schema as _cs

        class _PydanticArray:
            """Pydantic 兼容的 np.ndarray 占位类型，用于 TypedDict 字段标注。

            告诉 Pydantic 将数组字段视为 list[float] 以生成 schema，
            运行时实际值仍然是 numpy ndarray。
            """
            __slots__ = ()
            @classmethod
            def __get_pydantic_core_schema__(cls, _source_type, _handler):
                return _cs.list_schema(_cs.float_schema())

        _Array = _PydanticArray
    except ImportError:
        _Array = np.ndarray


# ═══════════════════════════════════════════════════════════════
# SurfaceState — 表面状态（7 维）
# 对外呈现的表达特征，直接影响语言风格。
# 由 State Engine 根据内部状态与 Traits 计算得出。
# ═══════════════════════════════════════════════════════════════

S_EXPRESSIVENESS = 0   # 情绪外露程度（-1=内敛, 0=中性, +1=奔放）
S_WARMTH = 1           # 语气温度（-1=冰冷, 0=中性, +1=温暖）
S_SHARPNESS = 2        # 攻击性/尖锐感（-1=温和, 0=中性, +1=尖锐）
S_SOFTNESS = 3         # 柔和度（-1=生硬, 0=中性, +1=柔软）
S_ENTHUSIASM = 4       # 活力/热情（-1=低沉, 0=中性, +1=高涨）
S_RESTRAINT = 5        # 克制程度（-1=直白, 0=中性, +1=克制）
S_VULNERABILITY = 6    # 脆弱感（-1=坚强, 0=中性, +1=脆弱）
S_SIZE = 7

S_LABELS = [
    "expressiveness", "warmth", "sharpness",
    "softness", "enthusiasm", "restraint", "vulnerability",
]
S_LABEL_IDX = {k: i for i, k in enumerate(S_LABELS)}

# ═══════════════════════════════════════════════════════════════
# Traits — 核心特质（10 维）
# 长期稳定的性格参数，随时间缓慢演化。
# 分三组：基础性格、负面倾向、依恋模式。
# ═══════════════════════════════════════════════════════════════

T_SENSITIVITY = 0           # 敏感度（-1=迟钝, 0=中性, +1=敏感）
T_PRIDE = 1                 # 自尊心（-1=自卑, 0=中性, +1=高傲）
T_EMOTIONAL_OPENNESS = 2    # 情绪开放性（-1=封闭, 0=中性, +1=敞开）
T_EMOTIONAL_STABILITY = 3   # 情绪稳定性（-1=波动, 0=中性, +1=稳定）
T_OPTIMISM = 4              # 乐观倾向（-1=悲观, 0=中性, +1=乐观）
T_ANXIETY_PRONENESS = 5     # 焦虑倾向（-1=松弛, 0=中性, +1=易焦虑）
T_ANGER_REACTIVITY = 6      # 易怒倾向（-1=平和, 0=中性, +1=易怒）
T_JEALOUSY_SENSITIVITY = 7  # 嫉妒敏感度（-1=不在意, 0=中性, +1=易嫉妒）
T_ATTACHMENT_ANXIETY = 8    # 依恋焦虑（-1=安全, 0=中性, +1=害怕被抛弃）
T_ATTACHMENT_AVOIDANCE = 9  # 依恋回避（-1=亲近, 0=中性, +1=疏离）
T_SIZE = 10

T_LABELS = [
    "sensitivity", "pride", "emotional_openness", "emotional_stability", "optimism",
    "anxiety_proneness", "anger_reactivity", "jealousy_sensitivity",
    "attachment_anxiety", "attachment_avoidance",
]
T_LABEL_IDX = {k: i for i, k in enumerate(T_LABELS)}

# ═══════════════════════════════════════════════════════════════
# InternalState — 内部状态（8 维）
# 底层心理指标，受对话事件影响而变化。
# ═══════════════════════════════════════════════════════════════

I_ENERGY = 0            # 精力/能量（-1=耗尽, 0=正常, +1=充沛）
I_STRESS = 1            # 压力（-1=深度放松, 0=正常, +1=高压）
I_LONELINESS = 2        # 孤独感（-1=充实, 0=正常, +1=孤独）
I_INSECURITY = 3        # 不安全感（-1=自信, 0=正常, +1=不安）
I_IRRITATION = 4        # 烦躁程度（-1=平静, 0=正常, +1=烦躁）
I_LONGING = 5           # 渴望/思念（-1=淡然, 0=正常, +1=强烈渴望）
I_SOCIAL_BATTERY = 6    # 社交电量（-1=透支, 0=正常, +1=满电）
I_MENTAL_FATIGUE = 7    # 精神疲劳（-1=异常清醒, 0=正常, +1=疲惫）
I_SIZE = 8

I_LABELS = [
    "energy", "stress", "loneliness", "insecurity",
    "irritation", "longing", "social_battery", "mental_fatigue",
]
I_LABEL_IDX = {k: i for i, k in enumerate(I_LABELS)}

# ═══════════════════════════════════════════════════════════════
# RelationshipState — 关系状态（3 维，原 6 维语义合并）
# AI 与用户之间的互动累积感知。
#
# 合并说明（2026-06-21）:
#   R_TRUST_BOND = trust + emotional_safety（信任+情感安全→信任纽带）
#   R_INTIMACY   = familiarity + dependency + romantic_tension（熟悉+依赖+张力→亲密）
# 合并理由: 原 6 维中 trust×safety r≈0.6, fam×dep×tens r≈0.98+，
#          但 B 矩阵密度 52.4% 无法通过约束⑥（≤30%），改用显式命名规则后自动稀疏。
# ═══════════════════════════════════════════════════════════════

R_AFFECTION = 0            # 好感度（-1=厌恶, 0=中性, +1=喜爱）- 保留原语义
R_TRUST_BOND = 1           # 信任纽带（-1=怀疑, 0=中性, +1=安全）- trust+safety 合并
R_INTIMACY = 2             # 亲密张力（-1=疏离, 0=中性, +1=亲密）- fam+dep+tension 合并
R_SIZE = 3

R_LABELS = [
    "affection", "trust_bond", "intimacy",
]
R_LABEL_IDX = {k: i for i, k in enumerate(R_LABELS)}


# ── HiddenState 已移除，相关"里表情"职责合并至 defense 层（inner/outer 刺激分离） ──

# ═══════════════════════════════════════════════════════════════
# StimulusVector — 心理刺激（7 维）
# perception_node 的核心输出：LLM 直接从用户输入中提取的心理刺激强度。
# 替代了旧的两阶段方案（SocialSignals → Stimulus Construction 线性层），
# 由感知模型一步到位输出角色主观感受到的心理意义。
# ═══════════════════════════════════════════════════════════════

ST_ABANDONMENT = 0      # 被抛弃恐惧（被冷落/推开/遗弃的心理冲击）
ST_VALIDATION = 1       # 被认可/被重视感（被肯定/喜欢/认可的心理满足）
ST_CLOSENESS = 2        # 亲密靠近/连接感（被靠近/关注/亲昵的心理体验）
ST_CONFLICT = 3         # 冲突/对抗张力（被攻击/指责/对抗的心理压力）
ST_DEPENDENCY = 4       # 被依赖/被需要感（被求助/被需要的心理意义）
ST_TEASING = 5          # 被逗弄/被调侃（被调戏/逗弄的心理反应）
ST_EMOTIONAL_WEIGHT = 6 # 情绪冲击强度（本轮对话的情感重量/严重程度）
ST_SIZE = 7

ST_LABELS = [
    "abandonment_stimulus", "validation_stimulus", "closeness_stimulus",
    "conflict_stimulus", "dependency_stimulus", "teasing_stimulus",
    "emotional_weight_stimulus",
]
ST_LABEL_IDX = {k: i for i, k in enumerate(ST_LABELS)}

# ═══════════════════════════════════════════════════════════════
# 类型别名（保持导入兼容性）
# ═══════════════════════════════════════════════════════════════

InternalState = np.ndarray          # 8 维，用 I_* 索引
RelationshipState = np.ndarray      # 3 维，用 R_* 索引
SurfaceState = np.ndarray           # 7 维，用 S_* 索引
Traits = np.ndarray                 # 10 维，用 T_* 索引
StimulusVector = np.ndarray         # 7 维，用 ST_* 索引（perception 直接输出）

# ═══════════════════════════════════════════════════════════════
# 工具函数：键值对字典 → numpy 数组
# ═══════════════════════════════════════════════════════════════


def _strip_json_fence(text: str) -> str:
    """剥除 LLM 输出中常见的 ```json ... ``` 包裹，返回纯 JSON 文本。

    同时处理：
      - ```json\\n{...}\\n```
      - ```\\n{...}\\n```
      - 首尾空白
    已是纯 JSON 时原样返回。
    """
    import re

    s = text.strip()
    # 去开头的 ``` 及可选的语言标记
    s = re.sub(r"^```(?:json)?\s*", "", s)
    # 去末尾的 ```
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _dict_to_array(d: dict, label_idx: dict, size: int) -> np.ndarray:
    """将键值对字典转换为对应长度的 numpy 数组。

    缺失的键对应维度值为 0.0，多余的键被忽略。
    """
    arr = np.zeros(size, dtype=np.float64)
    for k, v in d.items():
        idx = label_idx.get(k)
        if idx is not None:
            arr[idx] = v
    return arr


def stimuli_from_dict(d: dict) -> np.ndarray:
    """将 user_stimuli 的键值对字典转换为 7 维 StimulusVector 数组。"""
    return _dict_to_array(d, ST_LABEL_IDX, ST_SIZE)


# ═══════════════════════════════════════════════════════════════
# State — 顶层状态聚合（保留 TypedDict）
# ═══════════════════════════════════════════════════════════════


class State(TypedDict):
    """顶层状态——聚合所有子状态，作为图节点的消息传递载体。"""
    messages: Annotated[List, add_messages]
    memory_id:str

    # ── 角色自身的内部状态（所有层级） ──
    surface_state: Optional[_Array]
    traits: _Array

    internal_state: Optional[_Array]
    relationship_state: Optional[_Array]

    # ── 感知节点输出（perception_node 写入，state_engine_node 消费后置为 None） ──
    user_stimuli: Optional[_Array]

    # ── 状态格式化输出（state_formatter_node 写入，llm_node 消费） ──
    state_description: Optional[str]

    # ── 系统标记 ──
    has_inject_system_prompt: bool
    error: bool


# ═══════════════════════════════════════════════════════════════
# 默认初始值
# ═══════════════════════════════════════════════════════════════

# ── 人格特质（10 维） ──
# 长期稳定的性格参数，目前以「月下誓约」人设为基准
# 值域 [-1, 1]：-1=低极, 0=中性/正常, +1=高极
DEFAULT_TRAITS: np.ndarray = np.array([
     0.4,    # T_SENSITIVITY — 敏感度（偏高，容易感知情绪变化）
     0.3,    # T_PRIDE — 自尊心（偏强，口是心非的资本）
     0.2,    # T_EMOTIONAL_OPENNESS — 情绪开放性
     0.0,    # T_EMOTIONAL_STABILITY — 情绪稳定性
     0.1,    # T_OPTIMISM — 乐观倾向
     0.2,    # T_ANXIETY_PRONENESS — 焦虑倾向（偏高，害怕被抛弃）
     0.0,    # T_ANGER_REACTIVITY — 易怒倾向
     0.4,    # T_JEALOUSY_SENSITIVITY — 嫉妒敏感度（偏高，独占欲强）
     0.1,    # T_ATTACHMENT_ANXIETY — 依恋焦虑
    -0.6,    # T_ATTACHMENT_AVOIDANCE — 依恋回避（低，渴望亲近）
], dtype=np.float64)

# ── 内部状态（8 维） ──
# 底层心理指标基线，值域 [-1, 1]
DEFAULT_INTERNAL: np.ndarray = np.array([
     0.4,    # I_ENERGY — 精力充沛
    -0.6,    # I_STRESS — 压力较低
    -0.4,    # I_LONELINESS — 略有孤独感
    -0.5,    # I_INSECURITY — 轻微不安
    -0.8,    # I_IRRITATION — 平静
    -0.2,    # I_LONGING — 有一定思念/渴望
     0.2,    # I_SOCIAL_BATTERY — 社交电量尚可
    -0.7,    # I_MENTAL_FATIGUE — 精神清醒
], dtype=np.float64)

# ── 关系状态（3 维） ──
# 对用户的关系感知基线，值域 [-1, 1]
DEFAULT_RELATIONSHIP: np.ndarray = np.array([
    0.15,    # R_AFFECTION — 初始好感偏中性略正
    0.10,    # R_TRUST_BOND — 初始信任安全感偏中性
    0.05,    # R_INTIMACY — 初始亲密张力中性
], dtype=np.float64)
