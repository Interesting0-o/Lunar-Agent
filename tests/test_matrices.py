"""Layer 2: 矩阵常量测试 — 谱半径、形状、对角线占优。

验证所有耦合矩阵满足稳定性条件: ρ(A) < 0.95。
"""

import numpy as np
import pytest
from state import I_SIZE, R_SIZE, ST_SIZE


class TestMatrixShapes:
    """所有矩阵的形状正确。"""

    def test_state_coupling_shape(self):
        from state_engine._matrices import STATE_COUPLING_A
        assert STATE_COUPLING_A.shape == (I_SIZE, I_SIZE), \
            f"STATE_COUPLING_A shape={STATE_COUPLING_A.shape}, expected ({I_SIZE},{I_SIZE})"

    def test_input_influence_shape(self):
        from state_engine._matrices import INPUT_INFLUENCE_B
        assert INPUT_INFLUENCE_B.shape == (ST_SIZE, I_SIZE), \
            f"INPUT_INFLUENCE_B shape={INPUT_INFLUENCE_B.shape}, expected ({ST_SIZE},{I_SIZE})"

    def test_rel_state_coupling_shape(self):
        from state_engine._matrices import REL_STATE_COUPLING_A
        assert REL_STATE_COUPLING_A.shape == (R_SIZE, R_SIZE)

    def test_rel_input_influence_shape(self):
        from state_engine._matrices import REL_INPUT_INFLUENCE_B
        assert REL_INPUT_INFLUENCE_B.shape == (ST_SIZE, R_SIZE)


class TestSpectralRadius:
    """谱半径约束: ρ(A) < 0.95 保证稳定性。"""

    def test_state_coupling_spectral_radius(self):
        from state_engine._matrices import STATE_COUPLING_A
        eigenvalues = np.linalg.eigvals(STATE_COUPLING_A)
        rho = max(abs(ev) for ev in eigenvalues)
        assert rho < 0.95, \
            f"STATE_COUPLING_A 谱半径 ρ={rho:.6f} ≥ 0.95，系统可能不稳定"

    def test_rel_state_coupling_spectral_radius(self):
        from state_engine._matrices import REL_STATE_COUPLING_A
        eigenvalues = np.linalg.eigvals(REL_STATE_COUPLING_A)
        rho = max(abs(ev) for ev in eigenvalues)
        assert rho < 0.95, \
            f"REL_STATE_COUPLING_A 谱半径 ρ={rho:.6f} ≥ 0.95"

    def test_state_coupling_diagonal_dominance(self):
        """对角线占优: diag = 0.85，非对角线求和远小于对角线。"""
        from state_engine._matrices import STATE_COUPLING_A
        diag = np.diag(STATE_COUPLING_A)
        assert np.allclose(diag, 0.85, atol=0.02), f"对角线值异常: {diag}"
        # Gershgorin: 每行非对角线绝对值之和 < |对角线|
        for i in range(I_SIZE):
            off_diag_sum = np.sum(np.abs(STATE_COUPLING_A[i])) - abs(STATE_COUPLING_A[i, i])
            assert off_diag_sum < abs(STATE_COUPLING_A[i, i]) * 0.5, \
                f"第{i}行非对角线之和={off_diag_sum:.4f} > 对角线*0.5"

    def test_rel_state_coupling_diagonal_dominance(self):
        from state_engine._matrices import REL_STATE_COUPLING_A
        # 谱归一化后对角线为 0.85（原始 0.90 × 0.9441），不在 0.90 附近
        diag = np.diag(REL_STATE_COUPLING_A)
        assert np.all(diag > 0.80), f"对角线过小: {diag}"
        for i in range(R_SIZE):
            off_diag_sum = np.sum(np.abs(REL_STATE_COUPLING_A[i])) - abs(REL_STATE_COUPLING_A[i, i])
            assert off_diag_sum < abs(REL_STATE_COUPLING_A[i, i]) * 0.35, \
                f"第{i}行非对角线之和={off_diag_sum:.4f} > 对角线*0.35"


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


class TestSpectralNormalize:
    """_spectral_normalize 工厂函数行为。"""

    def test_normalize_reduces_spectral_radius(self):
        """谱归一化确实将谱半径降到目标以下。"""
        from state_engine._matrices import _spectral_normalize
        # 构造一个谱半径大的矩阵
        A = np.array([[0.9, 0.3, 0.2],
                      [0.4, 0.8, 0.1],
                      [0.3, 0.2, 0.95]], dtype=np.float64)
        orig_rho = max(abs(ev) for ev in np.linalg.eigvals(A))
        assert orig_rho >= 0.95, "测试矩阵本身谱半径应 ≥ 0.95"

        result = _spectral_normalize(A, "test")
        new_rho = max(abs(ev) for ev in np.linalg.eigvals(result))
        assert new_rho <= 0.95 + 1e-10, f"归一化后 ρ={new_rho:.6f}"

    def test_idempotent_on_stable_matrix(self):
        """已经稳定的矩阵不被修改。"""
        from state_engine._matrices import _spectral_normalize
        A = np.eye(5) * 0.5
        result = _spectral_normalize(A, "test")
        assert np.allclose(result, A), "稳定矩阵不应被修改"
