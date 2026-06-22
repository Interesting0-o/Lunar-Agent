"""输入影响矩阵测试 — 形状、方向性。

验证 B 矩阵的形状正确且方向性符合心理学预期。
"""

import numpy as np
import pytest
from state import (
    I_SIZE, R_SIZE, ST_SIZE,
    ST_CONFLICT, ST_VALIDATION, ST_ABANDONMENT,
    I_ENERGY, I_STRESS, I_IRRITATION, I_INSECURITY, I_LONELINESS, I_LONGING,
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


class TestCouplingContractivity:
    """耦合系统是收缩映射 —— 保证不发散。"""

    def test_internal_coupling_spectral_radius(self):
        """内部耦合雅可比谱半径 ρ < 1.0（收缩性）。"""
        from state_engine._dynamics_weights import INTERNAL_COUPLING, SELF_DECAY
        J = INTERNAL_COUPLING - np.diag(SELF_DECAY)
        ev = np.linalg.eigvals(J)
        sr = np.max(np.abs(ev))
        assert sr < 0.95, f"内部耦合谱半径 ρ={sr:.4f} ≥ 0.95，系统可能发散"

    def test_coupling_sign_structure(self):
        """内部耦合的正负号结构符合心理学预期。"""
        from state_engine._dynamics_weights import INTERNAL_COUPLING
        # 精力→压力：负（精力充沛→压力降低）
        assert INTERNAL_COUPLING[I_ENERGY, I_STRESS] < 0, \
            "energy→stress 应为负"
        # 精力→孤独：负（精力→孤独降低）
        assert INTERNAL_COUPLING[I_ENERGY, I_LONELINESS] < 0, \
            "energy→loneliness 应为负"
        # 压力→烦躁：正
        assert INTERNAL_COUPLING[I_STRESS, I_IRRITATION] > 0, \
            "stress→irritation 应为正"
        # 孤独→不安：正
        assert INTERNAL_COUPLING[I_LONELINESS, I_INSECURITY] > 0, \
            "loneliness→insecurity 应为正"
        # 孤独→思念：正
        assert INTERNAL_COUPLING[I_LONELINESS, I_LONGING] > 0, \
            "loneliness→longing 应为正"
