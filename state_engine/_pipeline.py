"""State Engine Pipeline —— 4 步管线编排。

① 防御剖面 (deactivation / hyperactivation) → inner/outer 刺激
② 残差动力学 → 内部 + 关系状态更新（刺激+耦合驱动，无 per-turn 稳态恢复）
③ 表面投影 → 可观测表达（带惯性混合 + 时间衰减）
④ 表面→内部反馈 → 情绪失调成本 + 面部反馈 + 表达消耗 + 压抑成本

稳态恢复（拉到人格基线）由时间衰减（_decay.py）在对话间隔中处理，
不参与每轮动态。

防御剖面基于 Bowlby (1980) 依恋防御二分法:
  - Deactivation (去激活):  高回避 → 削减外在表达
  - Hyperactivation (过度激活): 高焦虑 → 放大内心感受
二者独立，形成 4 种防御模式: 铁壁、玻璃心、真淡定、纸墙。
"""

from typing import Optional
import numpy as np
from state import ST_SIZE, StimulusMetadata
from ._defenses import compute_defense_profiles, apply_defenses
from ._dynamics import (
    update_internal_state,
    update_relationship_state,
    compute_setpoint,
    compute_rel_setpoint,
)
from ._surface import project_surface, compute_surface_feedback
from ._utils import soft_clamp


def initialize_all(traits: np.ndarray) -> dict:
    """首次运行：用 Traits 初始化所有状态层。"""
    internal = compute_setpoint(traits)
    relationship = compute_rel_setpoint(traits)
    outer_zero = np.zeros(ST_SIZE, dtype=np.float64)
    surface = project_surface(internal, relationship, outer_zero, prev_surface=None)

    return {
        "internal_state": internal,
        "relationship_state": relationship,
        "surface_state": surface,
    }


# 关系更新缓冲期（轮数）：关系态每 N 轮更新一次，中间轮次保持冻结
# 实现真正的双速动力学：快速层（内部态）每轮更新，慢速层（关系态）N 轮一次
REL_BUFFER_INTERVAL = 3


def update_all(
    current_internal: Optional[np.ndarray],
    current_relationship: Optional[np.ndarray],
    traits: np.ndarray,
    stimuli: np.ndarray,
    prev_surface: Optional[np.ndarray] = None,
    stimulus_metadata: Optional[StimulusMetadata] = None,
    delta_hours: float = 0.0,
    rel_counter: int = 0,
) -> dict:
    """State Engine 主入口：4 步管线（反馈延迟版）。

    步骤:
      ① 防御剖面 → (inner_stimuli, outer_stimuli)
         deactivation 控制 outer 削减，hyperactivation 控制 inner 放大
      ④ 表面→内部反馈（延迟：上一轮 surface → 本轮 internal 调制）
         情绪失调成本 + 面部反馈 + 表达消耗（trace 量级）
      ② 残差动力学（内部 + 关系，使用反馈调制后的 internal）
      ③ 表面投影（带惯性混合 + 时间衰减）

    Args:
        current_internal: 当前内部状态 h_{t-1} (8,) 或 None
        current_relationship: 当前关系状态 r_{t-1} (3,) 或 None
        traits: 人格特质 (10,)
        stimuli: 原始心理刺激 (7,)
        prev_surface: 前一帧表面状态 (7,)，None 表示首帧
        stimulus_metadata: 刺激元属性（约束②），含置信度/来源/衰减因子
        delta_hours: 自上次更新以来的时间（小时），用于 surface 惯性衰减
        rel_counter: 关系更新计数器（每 REL_BUFFER_INTERVAL 轮更新一次关系态）

    Returns:
        {"internal_state": (8,), "relationship_state": (3,), "surface_state": (7,),
         "rel_counter": int}  # 递增后的计数器
    """
    if current_internal is None:
        return initialize_all(traits)

    # 约束②：应用刺激元属性 — 置信度缩放 + missing 维度清零
    if stimulus_metadata is not None:
        stimuli = stimuli * stimulus_metadata.confidence
        missing_mask = stimulus_metadata.source == 3
        stimuli[missing_mask] = 0.0

    # ① 防御剖面 → inner / outer
    profiles = compute_defense_profiles(traits, current_relationship, current_internal)
    inner_stimuli, outer_stimuli = apply_defenses(stimuli, profiles)

    # ④ 表面→内部反馈（延迟：上一轮 surface 影响本轮 internal）
    # 改为 surface[t-1] → internal[t] 而非旧版同轮即时反馈，
    # 更符合"先笑→然后感觉变好"的因果时序。
    if prev_surface is not None:
        feedback = compute_surface_feedback(prev_surface, current_internal)
        current_internal = soft_clamp(current_internal + feedback, -1.0, 1.0)

    # ② 残差动力学（使用反馈调制后的 current_internal）
    new_internal = update_internal_state(
        current_internal, inner_stimuli, traits, current_relationship, profiles,
    )

    # 双速动力学：关系态每 REL_BUFFER_INTERVAL 轮更新一次
    # 中间轮次保持冻结，模拟关系变化的"惯性"
    if rel_counter % REL_BUFFER_INTERVAL == 0:
        new_relationship = update_relationship_state(
            current_relationship, inner_stimuli, traits,
            current_internal=new_internal,
        )
    else:
        new_relationship = current_relationship.copy()

    # ③ 表面投影（带惯性混合 + 时间衰减）
    surface = project_surface(
        new_internal, new_relationship, outer_stimuli, prev_surface,
        delta_hours=delta_hours,
    )

    return {
        "internal_state": new_internal,
        "relationship_state": new_relationship,
        "surface_state": surface,
        "rel_counter": rel_counter + 1,
    }
