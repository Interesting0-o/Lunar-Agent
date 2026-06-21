"""输入影响矩阵测试 — 形状、方向性。

验证 B 矩阵的形状正确且方向性符合心理学预期。
"""

import numpy as np
import pytest
from state import (
    I_SIZE, R_SIZE, ST_SIZE,
    ST_CONFLICT, ST_VALIDATION, ST_ABANDONMENT,
    I_STRESS, I_IRRITATION, I_INSECURITY,
    R_AFFECTION, R_TRUST_BOND, R_INTIMACY,
    DEFAULT_TRAITS, DEFAULT_RELATIONSHIP,
)


class TestMatrixShapes:
    """所有 B 矩阵的形状正确。"""

    def test_input_influence_shape(self):
        from state_engine._matrices import INPUT_INFLUENCE_B
        assert INPUT_INFLUENCE_B.shape == (ST_SIZE, I_SIZE), \
            f"INPUT_INFLUENCE_B shape={INPUT_INFLUENCE_B.shape}, expected ({ST_SIZE},{I_SIZE})"


class TestInputInfluenceDirectional:
    """输入影响矩阵的方向性正确性。"""

    def test_conflict_causes_stress(self):
        """冲突刺激 → 压力正向影响。"""
        from state_engine._matrices import INPUT_INFLUENCE_B
        assert INPUT_INFLUENCE_B[ST_CONFLICT, I_STRESS] > 0, \
            "冲突应对压力产生正向影响"

    def test_conflict_causes_irritation(self):
        """冲突刺激 → 烦躁正向影响。"""
        from state_engine._matrices import INPUT_INFLUENCE_B
        assert INPUT_INFLUENCE_B[ST_CONFLICT, I_IRRITATION] > 0

    def test_validation_reduces_insecurity(self):
        """被认可 → 不安全减少（负向影响）。"""
        from state_engine._matrices import INPUT_INFLUENCE_B
        assert INPUT_INFLUENCE_B[ST_VALIDATION, I_INSECURITY] < 0, \
            "被认可应对不安全产生负向影响"

    def test_abandonment_increases_insecurity(self):
        """被抛弃 → 不安全增加。"""
        from state_engine._matrices import INPUT_INFLUENCE_B
        assert INPUT_INFLUENCE_B[ST_ABANDONMENT, I_INSECURITY] > 0


class TestRelStimulusDirection:
    """关系刺激方向性 — 验证新显式命名规则（替代旧的 REL_INPUT_INFLUENCE_B 矩阵）。"""

    def test_rel_conflict_reduces_trust_bond(self):
        """冲突 → 关系信任下降。"""
        from state_engine._dynamics import update_relationship_state
        from state_engine._defenses import apply_defenses
        curr = DEFAULT_RELATIONSHIP.copy()
        stimuli = np.zeros(ST_SIZE)
        stimuli[ST_CONFLICT] = 0.8
        traits = DEFAULT_TRAITS.copy()
        inner_s, _ = apply_defenses(stimuli, np.ones((2, ST_SIZE)) * 0.3)
        result = update_relationship_state(curr, inner_s, traits)
        assert result[R_TRUST_BOND] < curr[R_TRUST_BOND], \
            "冲突应减少信任纽带"

    def test_rel_validation_increases_affection(self):
        """被认可 → 好感上升。"""
        from state_engine._dynamics import update_relationship_state
        from state_engine._defenses import apply_defenses
        curr = DEFAULT_RELATIONSHIP.copy()
        stimuli = np.zeros(ST_SIZE)
        stimuli[ST_VALIDATION] = 0.8
        traits = DEFAULT_TRAITS.copy()
        inner_s, _ = apply_defenses(stimuli, np.ones((2, ST_SIZE)) * 0.3)
        result = update_relationship_state(curr, inner_s, traits)
        assert result[R_AFFECTION] > curr[R_AFFECTION], \
            "被认可应增加好感度"
