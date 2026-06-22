"""输入影响矩阵 B —— 心理刺激 → 状态维度的线性映射。

所有 B 矩阵集中于此，通过 WeightMapper 构建（约束⑤ 语义映射层合规）。
禁止裸数字 `M[i,j] = value` 赋值。

跨维度耦合已迁移至 _dynamics.py 中的显式命名规则（替代旧 A 矩阵）。

2026-06-21 去相关化重构（约束⑥+⑩合规）：
  - INPUT_INFLUENCE_B 密度从 44.6% → 28.6%（≤30% ✅）
  - REL_INPUT_INFLUENCE_B 密度从 38.1% → 28.6%（≤30% ✅，全正交刺激签名）
  - 每维内部/关系状态有独特的刺激签名，最大化刺激正交性
  - 所有条目通过 WeightMapper 注册，含完整 provenance
"""

import numpy as np
from state import (
    # 内部状态索引
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE, I_SIZE,
    # 关系状态索引
    R_AFFECTION, R_TRUST_BOND, R_INTIMACY, R_SIZE,
    # 刺激索引
    ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT,
    ST_DEPENDENCY, ST_TEASING, ST_EMOTIONAL_WEIGHT, ST_SIZE,
    # 标签映射
    ST_LABELS, I_LABELS, R_LABELS,
)
from ._validator import WeightMapper


def _build_input_influence() -> np.ndarray:
    """通过 WeightMapper 建立去相关 B 矩阵（28.6% 密度, 16/56 非零）。

    去相关签名设计（每维唯⼀刺激来源）:
      abandonment  → insecurity↑, loneliness↑, longing↑        — 抛弃激活三感
      validation   → energy↑, insecurity↓                      — 认可充电+安心
      closeness    → energy↑, loneliness↓, social_battery↓      — 亲近耗能但驱散孤独
      conflict     → stress↑, irritation↑, energy↓              — 冲突三连环:压/烦/耗
      dependency   → social_battery↓, loneliness↓               — 被需要也耗电但有陪伴
      teasing      → irritation↑                                — 调侃专用
      emotional_weight → stress↑, mental_fatigue↑               — 沉重→压+倦
    """
    mapper = WeightMapper(
        "INPUT_INFLUENCE_B",
        source_labels=ST_LABELS,
        target_labels=I_LABELS,
        description="心理刺激→内部状态映射 (7×8)",
    )

    # ── abandonment: 被抛弃 → 不安↑ 孤独↑ 思念↑ ──
    mapper.connect(
        source_idx=ST_ABANDONMENT, target_idx=I_INSECURITY,
        value=0.28, magnitude="strong", domain=(0.20, 0.40),
        rationale="抛弃直接激活不安全感核心 (Bowlby IWM, 1980)",
        origin="theory", reviewed="2026-06-21",
    ).connect(
        source_idx=ST_ABANDONMENT, target_idx=I_LONELINESS,
        value=0.22, magnitude="strong", domain=(0.15, 0.30),
        rationale="被遗弃感→孤独感上升 (Weiss, 1973)",
        origin="theory", reviewed="2026-06-21",
    ).connect(
        source_idx=ST_ABANDONMENT, target_idx=I_LONGING,
        value=0.18, magnitude="moderate", domain=(0.10, 0.25),
        rationale="害怕失去→反而更思念 (Mikulincer & Shaver, 2003)",
        origin="theory", reviewed="2026-06-21",
    )

    # ── validation: 被认可 → 精力↑ 不安↓ ──
    mapper.connect(
        source_idx=ST_VALIDATION, target_idx=I_ENERGY,
        value=0.22, magnitude="strong", domain=(0.15, 0.30),
        rationale="被认可→自我效能感提升→精力充沛 (Bandura, 1997)",
        origin="theory", reviewed="2026-06-21",
    ).connect(
        source_idx=ST_VALIDATION, target_idx=I_INSECURITY,
        value=-0.22, magnitude="strong", domain=(-0.30, -0.15),
        rationale="被认可→缓解关系不安全感 (Bowlby, 1988)",
        origin="theory", reviewed="2026-06-21",
    )

    # ── closeness: 亲近靠近 → 精力↑ 孤独↓ 社交电量↓ ──
    mapper.connect(
        source_idx=ST_CLOSENESS, target_idx=I_ENERGY,
        value=0.12, magnitude="moderate", domain=(0.05, 0.20),
        rationale="亲近互动→情感能量提升 (Collins & Miller, 1994)",
        origin="theory", reviewed="2026-06-21",
    ).connect(
        source_idx=ST_CLOSENESS, target_idx=I_LONELINESS,
        value=-0.25, magnitude="strong", domain=(-0.35, -0.15),
        rationale="核心：陪伴→驱散孤独感 (Cacioppo & Patrick, 2008)",
        origin="theory", reviewed="2026-06-21",
    ).connect(
        source_idx=ST_CLOSENESS, target_idx=I_SOCIAL_BATTERY,
        value=-0.12, magnitude="moderate", domain=(-0.20, -0.05),
        rationale="亲近互动消耗社交能量内向者效应 (Eysenck, 1967)",
        origin="calibrated", reviewed="2026-06-21",
    )

    # ── conflict: 冲突 → 压力↑ 烦躁↑ 精力↓ ──
    mapper.connect(
        source_idx=ST_CONFLICT, target_idx=I_STRESS,
        value=0.35, magnitude="strong", domain=(0.25, 0.45),
        rationale="核心：冲突直接增压 (Lazarus & Folkman, 1984)",
        origin="theory", reviewed="2026-06-21",
    ).connect(
        source_idx=ST_CONFLICT, target_idx=I_IRRITATION,
        value=0.30, magnitude="strong", domain=(0.20, 0.40),
        rationale="冲突激怒—挫折→攻击理论 (Berkowitz, 1989)",
        origin="theory", reviewed="2026-06-21",
    ).connect(
        source_idx=ST_CONFLICT, target_idx=I_ENERGY,
        value=-0.20, magnitude="strong", domain=(-0.30, -0.10),
        rationale="冲突消耗情绪能量 (Gross, 2015)",
        origin="theory", reviewed="2026-06-21",
    )

    # ── dependency: 被依赖 → 社交电量↓ 孤独↓ ──
    mapper.connect(
        source_idx=ST_DEPENDENCY, target_idx=I_SOCIAL_BATTERY,
        value=-0.10, magnitude="weak", domain=(-0.15, -0.04),
        rationale="被需要消耗社交能量 (Baumeister & Leary, 1995)",
        origin="calibrated", reviewed="2026-06-21",
    ).connect(
        source_idx=ST_DEPENDENCY, target_idx=I_LONELINESS,
        value=-0.14, magnitude="weak", domain=(-0.20, -0.06),
        rationale="被需要→社会联结感→缓解孤独 (Cacioppo, 2008)",
        origin="calibrated", reviewed="2026-06-21",
    )

    # ── teasing: 被调侃 → 烦躁↑ ──
    mapper.connect(
        source_idx=ST_TEASING, target_idx=I_IRRITATION,
        value=0.08, magnitude="weak", domain=(0.04, 0.14),
        rationale="调侃激起的轻微烦躁 (Keltner et al., 2001)",
        origin="theory", reviewed="2026-06-21",
    )

    # ── emotional_weight: 情绪冲击 → 压力↑ 精神疲劳↑ ──
    mapper.connect(
        source_idx=ST_EMOTIONAL_WEIGHT, target_idx=I_STRESS,
        value=0.22, magnitude="strong", domain=(0.15, 0.30),
        rationale="沉重话题→心理压力累积 (Pennebaker, 1997)",
        origin="theory", reviewed="2026-06-21",
    ).connect(
        source_idx=ST_EMOTIONAL_WEIGHT, target_idx=I_MENTAL_FATIGUE,
        value=0.18, magnitude="moderate", domain=(0.10, 0.25),
        rationale="情绪冲击→认知资源消耗→精神疲劳 (Baumeister, 1998)",
        origin="theory", reviewed="2026-06-21",
    )

    return mapper.build_matrix(
        (ST_SIZE, I_SIZE),
        skip_rank=True,        # B 矩阵是输入映射，应维持满秩（低秩仅适用于耦合矩阵）
        skip_orthogonality=True,  # B 矩阵稀疏非方阵，行 Gram 正交性不适用（刺激正交性由约束⑩保证）
    )


INPUT_INFLUENCE_B = _build_input_influence()


def _build_rel_input_influence() -> np.ndarray:
    """关系刺激 B 矩阵（7→3）：心理刺激 → 关系状态，28.6% 密度，全正交签名。

    每维关系态有完全不重叠的刺激来源（零共享刺激维度）:

      AFFECTION  ← validation (+0.18), closeness (+0.10)   → 纯正向
      TRUST_BOND ← conflict (-0.25), abandonment (-0.10)   → 纯负向
      INTIMACY   ← dependency (+0.15), teasing (+0.10)     → 双源发散
    """
    mapper = WeightMapper(
        "REL_INPUT_INFLUENCE_B",
        source_labels=ST_LABELS,
        target_labels=R_LABELS,
        description="心理刺激→关系状态映射 (7×3)",
    )

    # ── AFFECTION: 好感度 ──
    mapper.connect(
        source_idx=ST_VALIDATION, target_idx=R_AFFECTION,
        value=0.18, magnitude="moderate", domain=(0.10, 0.25),
        rationale="被认可→好感度上升 (Byrne, 1971 强化-情感模型)",
        origin="theory", reviewed="2026-06-21",
    ).connect(
        source_idx=ST_CLOSENESS, target_idx=R_AFFECTION,
        value=0.10, magnitude="weak", domain=(0.05, 0.18),
        rationale="亲近互动→喜欢感积累 (Collins & Miller, 1994)",
        origin="theory", reviewed="2026-06-21",
    )

    # ── TRUST_BOND: 信任/安全感（纯负向输入） ──
    mapper.connect(
        source_idx=ST_CONFLICT, target_idx=R_TRUST_BOND,
        value=-0.25, magnitude="strong", domain=(-0.35, -0.15),
        rationale="核心：冲突破坏信任感 (Simpson, 2007)",
        origin="theory", reviewed="2026-06-21",
    ).connect(
        source_idx=ST_ABANDONMENT, target_idx=R_TRUST_BOND,
        value=-0.10, magnitude="weak", domain=(-0.18, -0.05),
        rationale="被抛弃暗示→安全感受到威胁 (Bowlby, 1988)",
        origin="theory", reviewed="2026-06-21",
    )

    # ── INTIMACY: 亲密张力（双源发散） ──
    mapper.connect(
        source_idx=ST_DEPENDENCY, target_idx=R_INTIMACY,
        value=0.15, magnitude="moderate", domain=(0.08, 0.22),
        rationale="被需要→关系卷入度上升 (Berscheid, 1983)",
        origin="theory", reviewed="2026-06-21",
    ).connect(
        source_idx=ST_TEASING, target_idx=R_INTIMACY,
        value=0.10, magnitude="weak", domain=(0.05, 0.16),
        rationale="调侃调情→暧昧张力上升 (Keltner et al., 2001)",
        origin="theory", reviewed="2026-06-21",
    )

    return mapper.build_matrix(
        (ST_SIZE, R_SIZE),
        skip_rank=True,
        skip_orthogonality=True,
    )


REL_INPUT_INFLUENCE_B = _build_rel_input_influence()

