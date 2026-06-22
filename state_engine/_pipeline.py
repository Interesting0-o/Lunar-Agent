"""State Engine Pipeline —— 3 步管线编排。

① 防御剖面 (deactivation / hyperactivation) → inner/outer 刺激
② 残差动力学 → 内部 + 关系状态更新（刺激+耦合驱动，无 per-turn 稳态恢复）
③ 表面投影 → 可观测表达

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


def update_all(
    current_internal: Optional[np.ndarray],
    current_relationship: Optional[np.ndarray],
    traits: np.ndarray,
    stimuli: np.ndarray,
    prev_surface: Optional[np.ndarray] = None,
    stimulus_metadata: Optional[StimulusMetadata] = None,
) -> dict:
    """State Engine 主入口：4 步管线。

    步骤:
      ① 防御剖面 → (inner_stimuli, outer_stimuli)
         deactivation 控制 outer 削减，hyperactivation 控制 inner 放大
      ② 残差动力学（内部 + 关系，含内建稳态恢复）
      ③ 表面投影（带惯性混合）
      ④ 表面→内部反馈（情绪失调成本 + 面部反馈 + 表达消耗）

    Args:
        current_internal: 当前内部状态 h_{t-1} (8,) 或 None
        current_relationship: 当前关系状态 r_{t-1} (3,) 或 None
        traits: 人格特质 (10,)
        stimuli: 原始心理刺激 (7,)
        prev_surface: 前一帧表面状态 (7,)，None 表示首帧
        stimulus_metadata: 刺激元属性（约束②），含置信度/来源/衰减因子

    Returns:
        {"internal_state": (8,), "relationship_state": (3,), "surface_state": (7,)}
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

    # ② 残差动力学（刺激+耦合驱动，稳态恢复已移至 _decay.py）
    new_internal = update_internal_state(
        current_internal, inner_stimuli, traits, current_relationship, profiles,
    )
    new_relationship = update_relationship_state(
        current_relationship, inner_stimuli, traits,
        current_internal=new_internal,
    )

    # ③ 表面投影（带惯性混合）
    surface = project_surface(
        new_internal, new_relationship, outer_stimuli, prev_surface,
    )

    # ④ 表面→内部反馈（第二帧起生效，trace 量级微小调制）
    if prev_surface is not None:
        feedback = compute_surface_feedback(surface, new_internal)
        new_internal = soft_clamp(new_internal + feedback, -1.0, 1.0)

    return {
        "internal_state": new_internal,
        "relationship_state": new_relationship,
        "surface_state": surface,
    }
