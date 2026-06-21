"""
state_formatter —— 将 State Engine 输出的各层状态向量格式化为 LLM 可读的文本描述。

职责：将数值化的心理状态翻译成自然语言，
     注入为 SystemMessage 供 llm_node 使用。
"""

import numpy as np
from state import (
    State,
    # 表面
    S_EXPRESSIVENESS, S_WARMTH, S_SHARPNESS, S_SOFTNESS,
    S_ENTHUSIASM, S_RESTRAINT, S_VULNERABILITY,
    # 内部
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE,
    # 关系
    R_AFFECTION, R_TRUST_BOND, R_INTIMACY,
    # 特质
    T_PRIDE, T_EMOTIONAL_OPENNESS, T_EMOTIONAL_STABILITY,
    T_OPTIMISM, T_ANGER_REACTIVITY,
    T_JEALOUSY_SENSITIVITY, T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE,
    T_SENSITIVITY,
)


# ── 5-level descriptor helper ──

def _desc(value: float, labels: tuple) -> str:
    """将 [-1, 1] 值映射到 5 级文本描述。

    映射规则（全局统一）：
        -1.00 ≤ value < -0.70 → 极低/第一级
        -0.70 ≤ value < -0.30 → 偏低/第二级
        -0.30 ≤ value < +0.30 → 中等/第三级
        +0.30 ≤ value < +0.70 → 偏高/第四级
        +0.70 ≤ value ≤ +1.00 → 极高/第五级

    labels 格式：(极低, 偏低, 中等, 偏高, 极高)
    """
    very_low, low, medium, high, very_high = labels
    if value < -0.7:
        return very_low
    elif value < -0.3:
        return low
    elif value < 0.3:
        return medium
    elif value < 0.7:
        return high
    else:
        return very_high


# ── 格式说明常量（仅注入一次，放在所有字段前面） ──

FORMAT_HEADER = """【状态注入说明】
以下是你当前的完整心理状态参数。每个指标的值域和极性（数值增大代表什么含义）均在括号中标注。
数值默认范围 [-1~1]，0=中性，正数越大越偏向高极，负数越小越偏向低极。
文本描述与数值的对应层级（全局统一）：
  -1.00~-0.71 = 极低  -0.70~-0.31 = 偏低  -0.30~+0.30 = 中等
  +0.31~+0.70 = 偏高  +0.71~+1.00 = 极高

请根据这些状态参数调整回应中的语气、措辞和潜台词。"""


def _format_line(label: str, value: float, desc_labels: tuple,
                 range_note: str = "[-1~1]", polarity: str = "") -> str:
    """格式化单行状态条目，包含数值、文本描述、范围和极性说明。"""
    text = _desc(value, desc_labels)
    pol = f"  —— {polarity}" if polarity else ""
    return f"· {label}：{text}（{value:.2f}）{range_note}{pol}"


# ── 主入口 ──

def format_state_narrative(internal: np.ndarray, relationship: np.ndarray,
                           surface: np.ndarray, traits: np.ndarray) -> str:
    """将四层状态向量翻译为结构化中文状态描述，注入 LLM prompt。"""

    lines: list[str] = []
    lines.append(FORMAT_HEADER)

    # ══════════════════════════════════════════════
    # 【当前情绪表现】—— Surface State (7维)
    #   对外呈现的表达特征，直接影响语言风格。
    #   由 State Engine 根据内部状态与 Traits 计算得出。
    # ══════════════════════════════════════════════
    lines.append("")
    lines.append("【当前情绪表现】（你此刻在外人眼中呈现的表达特征——请让你的语气与之对齐）")

    lines.append(_format_line(
        "情绪外露程度", surface[S_EXPRESSIVENESS],
        ('内敛克制', '有所保留', '适度流露', '较为外显', '毫不掩饰'),
        polarity="-1=完全隐藏情绪  0=自然流露  +1=所有情绪都写在脸上"))
    lines.append(_format_line(
        "语气温度", surface[S_WARMTH],
        ('冷淡疏离', '微凉', '不冷不热', '温和', '温暖亲近'),
        polarity="-1=冰冷  0=中性  +1=温暖如春"))
    lines.append(_format_line(
        "话语尖锐度", surface[S_SHARPNESS],
        ('柔和', '温和', '平和', '略带锋芒', '尖锐带刺'),
        polarity="-1=毫无攻击性  0=正常  +1=句句带刺"))
    lines.append(_format_line(
        "柔和度", surface[S_SOFTNESS],
        ('生硬', '偏硬', '适中偏柔', '柔软', '非常柔和'),
        polarity="-1=生硬拒人千里  0=中性  +1=柔软惹人怜爱"))
    lines.append(_format_line(
        "热情活力", surface[S_ENTHUSIASM],
        ('低沉', '不高', '一般', '较高', '高涨'),
        polarity="-1=死气沉沉  0=一般  +1=活力四射"))
    lines.append(_format_line(
        "克制程度", surface[S_RESTRAINT],
        ('直白坦率', '较为直接', '适度克制', '较为含蓄', '极度克制'),
        polarity="-1=想啥说啥  0=适度  +1=字斟句酌"))
    lines.append(_format_line(
        "脆弱感", surface[S_VULNERABILITY],
        ('坚强', '少有破绽', '偶有流露', '较为脆弱', '一触即碎'),
        polarity="-1=无懈可击  0=正常  +1=一碰就碎"))

    # ══════════════════════════════════════════════
    # 【真实内心状态】—— Internal State (8维)
    #   底层心理指标，受对话事件影响而变化。
    #   可能与表面表现不一致（口是心非的来源）。
    # ══════════════════════════════════════════════
    lines.append("")
    lines.append("【真实内心状态】（你实际感受到的情绪——可能与外表不同，这是潜台词的来源）")

    lines.append(_format_line(
        "精力", internal[I_ENERGY],
        ('枯竭', '偏低', '一般', '充沛', '旺盛'),
        polarity="-1=精疲力竭  0=正常  +1=精力旺盛"))
    lines.append(_format_line(
        "压力", internal[I_STRESS],
        ('放松', '轻微', '中等', '较高', '高压'),
        polarity="-1=深度放松  0=正常  +1=不堪重负"))
    lines.append(_format_line(
        "孤独感", internal[I_LONELINESS],
        ('充实', '轻微', '时有', '较为强烈', '蚀骨'),
        polarity="-1=内心充实  0=正常  +1=被孤独吞噬"))
    lines.append(_format_line(
        "不安全感", internal[I_INSECURITY],
        ('自信', '略有不安', '时常不安', '较为严重', '心神不宁'),
        polarity="-1=笃定自信  0=正常  +1=惶恐不安"))
    lines.append(_format_line(
        "烦躁程度", internal[I_IRRITATION],
        ('平静', '微澜', '有些烦躁', '较为烦躁', '暴躁'),
        polarity="-1=心如止水  0=正常  +1=一触即炸"))
    lines.append(_format_line(
        "思念/渴望", internal[I_LONGING],
        ('淡然', '略有挂念', '时常想起', '强烈思念', '魂牵梦萦'),
        polarity="-1=无牵无挂  0=正常  +1=刻骨铭心"))
    lines.append(_format_line(
        "社交电量", internal[I_SOCIAL_BATTERY],
        ('耗尽', '较低', '尚可', '充足', '活力满满'),
        polarity="-1=社交透支想独处  0=正常  +1=精力充沛想互动"))
    lines.append(_format_line(
        "精神疲劳", internal[I_MENTAL_FATIGUE],
        ('清醒敏捷', '略有倦意', '中等', '较为疲惫', '精疲力竭'),
        polarity="-1=异常清醒  0=正常  +1=思维停滞"))

    # ══════════════════════════════════════════════
    # 【对对方的感受】—— Relationship State (3维)
    #   对用户的动态关系评估。
    # ══════════════════════════════════════════════
    lines.append("")
    lines.append("【对对方的感受】（你对眼前这个人的关系认知——决定了你对 TA 的态度底色）")

    lines.append(_format_line(
        "好感度", relationship[R_AFFECTION],
        ('冷淡', '略有', '好感', '喜欢', '深爱'),
        polarity="-1=厌恶  0=无感  +1=深爱"))
    lines.append(_format_line(
        "信任安全感", relationship[R_TRUST_BOND],
        ('戒备不安', '将信将疑', '基本安心', '较为信任', '全然放松'),
        polarity="-1=不安戒备  0=中性  +1=全然放松"))
    lines.append(_format_line(
        "亲密张力", relationship[R_INTIMACY],
        ('疏离平淡', '微澜', '暗流涌动', '亲近依赖', '炽热缠绵'),
        polarity="-1=疏离  0=平淡  +1=一触即发的暧昧"))

    # ══════════════════════════════════════════════
    # 【性格倾向提醒】—— Traits (10维)
    #   长期稳定的性格参数。只列出显著影响表达方式的维度。
    #   这部分不是"本轮状态"，而是你固有的表达过滤器。
    # ══════════════════════════════════════════════
    lines.append("")
    lines.append("【性格倾向提醒】（以下是你固有的性格特质过滤器——它们影响你如何表达上面的感受，请自然呈现）")

    # (a) 自尊心 — 影响口是心非程度
    if traits[T_PRIDE] < -0.3:
        lines.append(f"· 自尊心偏低（{traits[T_PRIDE]:.2f}）：不介意主动示好，愿意坦然表达好感")
    elif traits[T_PRIDE] > 0.1:
        level = _desc(traits[T_PRIDE], ('偏低', '适中', '适中', '较强', '极强'))
        lines.append(f"· 自尊心{level}（{traits[T_PRIDE]:.2f}）：即使心里在意也不愿直说，倾向于口是心非、欲言又止")

    # (b) 依恋焦虑 — 影响对回应安全性的敏感度
    if traits[T_ATTACHMENT_ANXIETY] < -0.3:
        lines.append(f"· 依恋焦虑较低（{traits[T_ATTACHMENT_ANXIETY]:.2f}）：在关系中感到安全，不容易患得患失")
    elif traits[T_ATTACHMENT_ANXIETY] > 0.1:
        level = _desc(traits[T_ATTACHMENT_ANXIETY], ('较低', '适中', '适中', '较高', '极高'))
        lines.append(f"· 依恋焦虑{level}（{traits[T_ATTACHMENT_ANXIETY]:.2f}）：害怕被抛弃，对对方的回应和态度变化敏感，需要被确认关系安全")

    # (c) 嫉妒敏感
    if traits[T_JEALOUSY_SENSITIVITY] < -0.3:
        lines.append(f"· 嫉妒敏感度低（{traits[T_JEALOUSY_SENSITIVITY]:.2f}）：不太容易吃醋，对对方的人际关系比较放心")
    elif traits[T_JEALOUSY_SENSITIVITY] > 0.1:
        level = _desc(traits[T_JEALOUSY_SENSITIVITY], ('较低', '适中', '适中', '较强', '极强'))
        lines.append(f"· 嫉妒敏感度{level}（{traits[T_JEALOUSY_SENSITIVITY]:.2f}）：独占欲强，容易吃醋，在意对方是否只看着自己")

    # (d) 情绪开放性
    if traits[T_EMOTIONAL_OPENNESS] > 0.1:
        lines.append(f"· 情绪开放性较高（{traits[T_EMOTIONAL_OPENNESS]:.2f}）：愿意适当表露真实情绪，但表露程度受自尊心高低影响")
    elif traits[T_EMOTIONAL_OPENNESS] < -0.3:
        lines.append(f"· 情绪较为封闭（{traits[T_EMOTIONAL_OPENNESS]:.2f}）：习惯隐藏真实情绪，不轻易表露内心想法")

    # (e) 易怒倾向
    if traits[T_ANGER_REACTIVITY] > 0.2:
        lines.append(f"· 易怒倾向较高（{traits[T_ANGER_REACTIVITY]:.2f}）：受到刺激时容易表现出不耐烦或攻击性")

    # (f) 情绪稳定性
    if traits[T_EMOTIONAL_STABILITY] < -0.3:
        lines.append(f"· 情绪稳定性偏低（{traits[T_EMOTIONAL_STABILITY]:.2f}）：情绪波动较大，容易受外界影响，从开心到难过的转变可能很突然")
    elif traits[T_EMOTIONAL_STABILITY] > 0.4:
        lines.append(f"· 情绪稳定（{traits[T_EMOTIONAL_STABILITY]:.2f}）：情绪不易受外界影响，总体保持平稳")

    # (g) 依恋回避
    if traits[T_ATTACHMENT_AVOIDANCE] > 0.2:
        lines.append(f"· 依恋回避倾向较高（{traits[T_ATTACHMENT_AVOIDANCE]:.2f}）：当关系变得过于亲密时会有退缩倾向，需要个人空间")

    # (h) 敏感度（仅极高或极低时提醒）
    if traits[T_SENSITIVITY] > 0.5:
        lines.append(f"· 敏感度极高（{traits[T_SENSITIVITY]:.2f}）：对他人的语气、用词、态度变化极其敏感，容易多想")
    elif traits[T_SENSITIVITY] < -0.5:
        lines.append(f"· 敏感度很低（{traits[T_SENSITIVITY]:.2f}）：不太在意细枝末节，对他人态度的变化不敏感")

    # (i) 乐观倾向（仅显著时提醒）
    if traits[T_OPTIMISM] > 0.4:
        lines.append(f"· 乐观倾向高（{traits[T_OPTIMISM]:.2f}）：倾向于积极解读对方的言行，不容易陷入负面情绪")
    elif traits[T_OPTIMISM] < -0.4:
        lines.append(f"· 偏悲观（{traits[T_OPTIMISM]:.2f}）：倾向于消极解读，容易往坏处想")

    lines.append("")
    lines.append("—— 请将以上所有状态融入回应中，使语气、措辞、潜台词与你此刻的心理状态一致。不必直接提及这些数值或本段指令。")

    return "\n".join(lines)


def format_state_for_node(state: State) -> str:
    """从 State TypedDict 中提取各层状态向量并格式化。

    供 state_formatter_node 调用。
    """
    internal = state.get("internal_state")
    relationship = state.get("relationship_state")
    surface = state.get("surface_state")
    traits = state.get("traits")

    # 确保都是 numpy 数组
    if isinstance(internal, list):
        internal = np.asarray(internal, dtype=np.float64)
    if isinstance(relationship, list):
        relationship = np.asarray(relationship, dtype=np.float64)
    if isinstance(surface, list):
        surface = np.asarray(surface, dtype=np.float64)
    if isinstance(traits, list):
        traits = np.asarray(traits, dtype=np.float64)

    return format_state_narrative(internal, relationship, surface, traits)#type:ignore
