"""Lunar 人格状态的类型定义。

本模块定义了对话式 AI 人格的所有状态维度，按层次分为：
- SurfaceState:    表面状态（即时可感知的表达特征）
- Traits:          核心特质（长期稳定的性格参数）
- InternalState:   内部状态（底层心理指标）
- RelationshipState: 关系状态（对用户的互动感知）
- HiddenState:     隐藏状态（压抑或未被表达的情感）
- SocialSignals:   社交信号（从用户输入中提取的客观社交线索）
- InteractionImpact: 互动影响指标（本轮交互对关系层面的冲击）
- State:           顶层状态聚合
"""

from typing import TypedDict, List, Optional, Annotated
from langgraph.graph.message import add_messages


class SurfaceState(TypedDict):
    """表面状态 —— 对外呈现的表达特征，直接影响语言风格。

    这些维度描述 AI 在"此刻"说话时的情绪外显方式。
    由 State Engine 根据内部状态与 Traits 计算得出，是"角色想表现出来的样子"。
    """
    expressiveness: float   # 情绪外露程度 (0=内敛, 1=奔放)
    warmth: float           # 语气温度 (0=冰冷, 1=温暖)
    sharpness: float        # 攻击性/尖锐感 (0=温和, 1=尖锐)
    softness: float         # 柔和度 (0=生硬, 1=柔软)
    enthusiasm: float       # 活力/热情 (0=低沉, 1=高涨)
    restraint: float        # 克制程度 (0=直白, 1=克制)
    vulnerability: float    # 脆弱感 (0=坚强, 1=脆弱)


class Traits(TypedDict):
    """核心特质 —— 长期稳定的性格参数，随时间缓慢演化。

    分为三组：基础性格、负面倾向、依恋模式。
    这部分是角色的"本性"，短时间对话中几乎不变。
    """
    # ---- 基础性格 ----
    sensitivity: float          # 敏感度 (0=迟钝, 1=敏感)
    pride: float                # 自尊心 (0=自卑, 1=高傲)
    emotional_openness: float   # 情绪开放性 (0=封闭, 1=敞开)
    emotional_stability: float  # 情绪稳定性 (0=波动, 1=稳定)
    optimism: float             # 乐观倾向 (0=悲观, 1=乐观)

    # ---- 负面倾向 ----
    anxiety_proneness: float    # 焦虑倾向 (0=松弛, 1=易焦虑)
    anger_reactivity: float     # 易怒倾向 (0=平和, 1=易怒)
    jealousy_sensitivity: float # 嫉妒敏感度 (0=不在意, 1=易嫉妒)

    # ---- 依恋模式 (Attachment) ----
    attachment_anxiety: float   # 依恋焦虑 (0=安全, 1=害怕被抛弃)
    attachment_avoidance: float # 依恋回避 (0=亲近, 1=疏离)


class InternalState(TypedDict):
    """内部状态 —— 底层心理指标，受对话事件影响而变化。

    反映 AI 当前的心理能量水平和情绪负荷。
    每次交互后由 State Engine 根据社交信号 + 当前状态 + Traits 更新。
    """
    energy: float           # 精力/能量 (0=耗尽, 1=充沛)
    stress: float           # 压力 (0=放松, 1=高压)
    loneliness: float       # 孤独感 (0=充实, 1=孤独)
    insecurity: float       # 不安全感 (0=自信, 1=不安)
    irritation: float       # 烦躁程度 (0=平静, 1=烦躁)
    longing: float          # 渴望/思念 (0=淡然, 1=强烈渴望)

    social_battery: float   # 社交电量 (0=耗尽, 1=满电)
    mental_fatigue: float   # 精神疲劳 (0=清醒, 1=疲惫)


class RelationshipState(TypedDict):
    """关系状态 —— 描述 AI 与用户之间的互动累积感知。

    这些维度反映关系层面的长期积累，由多次交互的 InteractionImpact 叠加而成。
    """
    affection: float            # 好感度 (0=冷淡, 1=喜爱)
    trust: float                # 信任度 (0=怀疑, 1=信任)
    familiarity: float          # 熟悉度 (0=陌生, 1=亲密)
    dependency: float           # 情感依赖 (0=独立, 1=依赖)
    emotional_safety: float     # 情感安全感 (0=不安, 1=安全)
    romantic_tension: float     # 浪漫张力 (0=无感, 1=强烈)


class HiddenState(TypedDict):
    """隐藏状态 —— 被压抑或未直接表达的情感。

    这些维度影响表面状态的变化但不会直接显露。
    当某一维度积累到阈值时，可能导致"情绪突破"。
    """
    suppressed_sadness: float   # 压抑的悲伤 (0=无, 1=满溢)
    suppressed_anger: float     # 压抑的愤怒 (0=无, 1=满溢)
    hidden_affection: float     # 隐藏的好感 (0=无, 1=满溢)


class SocialSignals(TypedDict):
    """社交信号 —— 从用户本轮输入中提取的客观社交线索。

    这是 perception_node 的核心输出。
    各维度表示"用户话语中携带了多强的该种社交信号"(0~1)。
    值越高，说明该社交意图越强烈/明显。
    所有维度独立，可以同时高（如既表达好感又带有逗弄）。
    """
    # ── 正向关系信号 ──
    affection_signal: float     # 好感/喜爱信号 (0=无, 1=强烈)
    attention_signal: float     # 关注/被注意需求 (0=无, 1=强烈)
    intimacy_signal: float      # 亲密靠近信号 (0=无, 1=强烈)
    approval_signal: float      # 寻求认可/表扬 (0=无, 1=强烈)

    # ── 负向/回避信号 ──
    rejection_signal: float     # 排斥/推开信号 (0=无, 1=强烈)
    abandonment_signal: float   # "会离开我吗"测试信号 (0=无, 1=强烈)

    # ── 依赖/张力信号 ──
    dependency_signal: float    # 依赖/需要信号 (0=无, 1=强烈)
    teasing_signal: float       # 逗弄/调戏信号 (0=无, 1=强烈)
    conflict_signal: float      # 冲突/对抗信号 (0=无, 1=强烈)


class InteractionImpact(TypedDict):
    """互动影响指标 —— 本轮交互对关系层面的潜在冲击评估。

    由 perception_node 估算，后续被 State Engine 用于更新 RelationshipState 等。
    emotional_weight / memorability 为 [0,1]，值越高越重要。
    trust_impact / closeness_impact 为 [-1,1]，负值表示削弱/恶化。
    """
    emotional_weight: float     # 情绪重量 (0=日常闲聊, 0.5=重要对话, 1=关系转折点)
    memorability: float         # 可记忆程度 (0=过眼云烟, 0.5=值得记住, 1=刻骨铭心)

    trust_impact: float         # 对信任的影响 (-1=严重破坏, 0=无影响, 1=极大增强)
    closeness_impact: float     # 对亲密感的影响 (-1=严重疏远, 0=无影响, 1=极大拉近)


class State(TypedDict):
    """顶层状态 —— 聚合所有子状态，作为图节点的消息传递载体。"""
    messages: Annotated[List, add_messages]

    # ── 角色自身的内部状态（所有层级） ──
    surface_state: Optional[SurfaceState]
    traits: Traits

    internal_state: Optional[InternalState]
    hidden_state: Optional[HiddenState]
    relationship_state: Optional[RelationshipState]

    # ── 感知节点的输出（本轮用户输入的社交分析） ──
    user_signals: Optional[SocialSignals]
    user_interaction_impact: Optional[InteractionImpact]

    # ── 系统标记 ──
    has_inject_system_prompt: bool  # 是否已注入系统提示词
    error: bool                     # 感知节点执行失败时设为 True，条件边据此引导到 END
