"""Layer 2: 输入影响矩阵测试 — 形状、方向性。

验证 B 矩阵的形状正确且方向性符合心理学预期。
"""

import pytest
from state import I_SIZE, R_SIZE, ST_SIZE


class TestMatrixShapes:
    """所有 B 矩阵的形状正确。"""

    def test_input_influence_shape(self):
        from state_engine._matrices import INPUT_INFLUENCE_B
        assert INPUT_INFLUENCE_B.shape == (ST_SIZE, I_SIZE), \
            f"INPUT_INFLUENCE_B shape={INPUT_INFLUENCE_B.shape}, expected ({ST_SIZE},{I_SIZE})"

    def test_rel_input_influence_shape(self):
        from state_engine._matrices import REL_INPUT_INFLUENCE_B
        assert REL_INPUT_INFLUENCE_B.shape == (ST_SIZE, R_SIZE)


class TestInputInfluenceDirectional:
    """输入影响矩阵的方向性正确性。"""

    def test_conflict_causes_stress(self):
        """冲突刺激 → 压力正向影响。"""
        from state_engine._matrices import INPUT_INFLUENCE_B
        from state import ST_CONFLICT, I_STRESS
        assert INPUT_INFLUENCE_B[ST_CONFLICT, I_STRESS] > 0, \
            "冲突应对压力产生正向影响"

    def test_conflict_causes_irritation(self):
        """冲突刺激 → 烦躁正向影响。"""
        from state_engine._matrices import INPUT_INFLUENCE_B
        from state import ST_CONFLICT, I_IRRITATION
        assert INPUT_INFLUENCE_B[ST_CONFLICT, I_IRRITATION] > 0

    def test_validation_reduces_insecurity(self):
        """被认可 → 不安全减少（负向影响）。"""
        from state_engine._matrices import INPUT_INFLUENCE_B
        from state import ST_VALIDATION, I_INSECURITY
        assert INPUT_INFLUENCE_B[ST_VALIDATION, I_INSECURITY] < 0, \
            "被认可应对不安全产生负向影响"

    def test_abandonment_increases_insecurity(self):
        """被抛弃 → 不安全增加。"""
        from state_engine._matrices import INPUT_INFLUENCE_B
        from state import ST_ABANDONMENT, I_INSECURITY
        assert INPUT_INFLUENCE_B[ST_ABANDONMENT, I_INSECURITY] > 0

    def test_rel_conflict_reduces_trust(self):
        """冲突 → 关系信任下降。"""
        from state_engine._matrices import REL_INPUT_INFLUENCE_B
        from state import ST_CONFLICT, R_TRUST
        assert REL_INPUT_INFLUENCE_B[ST_CONFLICT, R_TRUST] < 0

    def test_rel_validation_increases_affection(self):
        """被认可 → 好感上升。"""
        from state_engine._matrices import REL_INPUT_INFLUENCE_B
        from state import ST_VALIDATION, R_AFFECTION
        assert REL_INPUT_INFLUENCE_B[ST_VALIDATION, R_AFFECTION] > 0
