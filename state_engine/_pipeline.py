"""State Engine 主入口：Pipeline 编排。

将 5 个子系统串联为完整的状态更新管线。
"""

from typing import Optional
import numpy as np
from state import ST_SIZE, DEFAULT_INTERNAL, DEFAULT_RELATIONSHIP
from ._gates import compute_gates, apply_gates
from ._dynamics import update_internal_dynamics, update_relationship_dynamics
from ._decay import compute_dynamic_decay, apply_decay
from ._surface import project_surface


def initialize_all(traits: np.ndarray) -> dict:
    """首次运行：用 Traits 初始化所有状态层，outer_stimuli 用 0 向量。"""
    internal = DEFAULT_INTERNAL.copy()
    relationship = DEFAULT_RELATIONSHIP.copy()
    outer_zero = np.zeros(ST_SIZE, dtype=np.float64)
    surface = project_surface(internal, relationship, traits, outer_zero)

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
) -> dict:
    """State Engine 主入口：4 步 Pipeline。

    步骤:
      ① 三向门控 → (inner_stimuli, outer_stimuli)
      ② 内部动力系统（LSTM 式 3 门控）+ 衰减
      ③ 关系动力系统（LTI）+ 衰减
      ④ 表面投影

    返回:
      {
        "internal_state":     np.ndarray,  # 8 维
        "relationship_state":  np.ndarray,  # 6 维
        "surface_state":       np.ndarray,  # 7 维
      }
    """
    if current_internal is None:
        return initialize_all(traits)

    # ① 门控
    gates = compute_gates(traits, current_relationship, current_internal)
    inner_stimuli, outer_stimuli = apply_gates(
        stimuli, gates, traits, current_relationship,
    )

    # ② 内部动力系统 + 衰减
    new_internal = update_internal_dynamics(
        current_internal, inner_stimuli, traits, current_relationship, gates,
    )
    internal_decay, rel_decay = compute_dynamic_decay(
        traits, current_relationship, current_internal, inner_stimuli,
    )
    new_internal = apply_decay(new_internal, internal_decay, DEFAULT_INTERNAL)

    # ③ 关系动力系统 + 衰减
    new_relationship = update_relationship_dynamics(current_relationship, inner_stimuli)
    new_relationship = apply_decay(new_relationship, rel_decay, DEFAULT_RELATIONSHIP)

    # ④ 表面投影
    surface = project_surface(new_internal, new_relationship, traits, outer_stimuli)

    return {
        "internal_state": new_internal,
        "relationship_state": new_relationship,
        "surface_state": surface,
    }
