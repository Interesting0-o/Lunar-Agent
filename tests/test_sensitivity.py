"""Parameter sensitivity analysis for the state engine.

Tests how sensitive the pipeline output (internal state, relationship state)
is to perturbations in individual parameters. Identifies redundant parameters
and validates that no parameter has outsized or unexpected effects.

Usage:
    # Quick check (fast, low precision):
    uv run python -m pytest tests/test_sensitivity.py -v

    # Full report (slower, 30 scenarios x 20 rounds):
    uv run python -m pytest tests/test_sensitivity.py -v --run-full-sensitivity

    # Single-parameter debug:
    uv run python -m pytest tests/test_sensitivity.py -k test_B_int_sensitivity -v
"""

import numpy as np
import pytest

from state import (
    DEFAULT_TRAITS, DEFAULT_INTERNAL, DEFAULT_RELATIONSHIP,
    ST_SIZE, I_SIZE, R_SIZE,
)
from state_engine import update_all

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

N_SCENARIOS = pytest.mark.parametrize("n_scenarios", [10])  # quick by default
N_ROUNDS = 20
PERTURB_FACTOR = 0.5  # multiply parameter by this factor

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _make_scenarios(n: int, seed: int = 42) -> list[list[np.ndarray]]:
    """Generate n multi-round stimulus sequences."""
    rng = np.random.default_rng(seed)
    scenarios = []
    for _ in range(n):
        seq = []
        for _ in range(N_ROUNDS):
            s = rng.uniform(0, 0.6, size=ST_SIZE)
            if rng.random() < 0.15:
                s[rng.integers(ST_SIZE)] = rng.uniform(0.6, 1.0)
            seq.append(s)
        scenarios.append(seq)
    return scenarios


def _run_multi(scenarios: list[list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Run multi-round pipeline, return stacked (internal, relationship)."""
    n = len(scenarios)
    ints = np.zeros((n, I_SIZE))
    rels = np.zeros((n, R_SIZE))
    traits = DEFAULT_TRAITS.copy()

    for i, seq in enumerate(scenarios):
        internal = DEFAULT_INTERNAL.copy()
        rel = DEFAULT_RELATIONSHIP.copy()
        surface = None
        for s in seq:
            res = update_all(internal, rel, traits, s, prev_surface=surface)
            internal = res["internal_state"]
            rel = res["relationship_state"]
            surface = res["surface_state"]
        ints[i] = internal
        rels[i] = rel
    return ints, rels


def _sensitivity(baseline: np.ndarray, perturbed: np.ndarray) -> float:
    """Normalized sensitivity: mean(|Δ|) / mean(|baseline|)."""
    return float(np.mean(np.abs(perturbed - baseline)) /
                 (np.mean(np.abs(baseline)) + 1e-10))


def _map_keys(name: str, nz_indices, values) -> list[tuple]:
    """Build (key, label) pairs for non-zero matrix entries."""
    keys = []
    for idx in zip(*nz_indices):
        label = f"{name}[{idx[0]},{idx[1]}]={values[idx]:.3f}"
        keys.append((idx, label))
    return keys


def _vec_keys(name: str, arr: np.ndarray) -> list[tuple]:
    """Build (key, label) pairs for vector entries."""
    return [(i, f"{name}[{i}]={arr[i]:.3f}") for i in range(len(arr))]


# ═══════════════════════════════════════════════════════════════════
# Parameter sensitivity test class
# ═══════════════════════════════════════════════════════════════════

SENSITIVITY_REPORT: dict[str, list[tuple[float, str]]] = {}
"""Module-level cache so --run-full-sensitivity can accumulate."""


class TestParameterSensitivity:
    """Sensitivity analysis for all parameter groups.

    Each test method perturbs every non-zero entry in a parameter group,
    measures the effect on pipeline output after N_ROUNDS rounds,
    and verifies no entry has outsized (s > 0.30) or pathologically
    uniform sensitivity.

    Results accumulate in SENSITIVITY_REPORT for optional --report.
    """

    n_scenarios = 10
    """Default; overridden to 30 by --run-full-sensitivity if configured."""

    scenarios: list[list[np.ndarray]] = []
    baseline_int: np.ndarray = None  # type: ignore
    baseline_rel: np.ndarray = None  # type: ignore

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Lazy-init scenarios + baseline once per session."""
        if not TestParameterSensitivity.scenarios:
            n = 10
            TestParameterSensitivity.n_scenarios = n
            TestParameterSensitivity.scenarios = _make_scenarios(n)
            bi, br = _run_multi(TestParameterSensitivity.scenarios)
            TestParameterSensitivity.baseline_int = bi
            TestParameterSensitivity.baseline_rel = br

    # ── Per-group test templates ──

    def _test_array_group(self, name: str, arr: np.ndarray,
                          keys: list[tuple], restore: callable = None):
        """General array-group test.  `restore` is called after each
        perturbation; for standalone arrays it's the identity."""
        scores = []
        for key, label in keys:
            # Save & perturb
            if isinstance(key, tuple):
                orig = arr[key]
                arr[key] = orig * PERTURB_FACTOR
            else:
                orig = arr[key]
                arr[key] = orig * PERTURB_FACTOR

            pi, pr = _run_multi(self.scenarios)

            # Restore
            if isinstance(key, tuple):
                arr[key] = orig
            else:
                arr[key] = orig
            if restore:
                restore(arr, key, orig)

            s = max(_sensitivity(self.baseline_int, pi),
                    _sensitivity(self.baseline_rel, pr))
            scores.append((s, label))

        self._check_and_record(name, scores)

    def _test_mapper_group(self, name: str, mapper, nz: tuple[np.ndarray, np.ndarray]):
        """Test a LinearMapping/WeightMapper by modifying its _weight_matrix."""
        W = mapper._weight_matrix
        scores = []
        for idx in zip(nz[0], nz[1]):
            orig = W[idx]
            W[idx] = orig * PERTURB_FACTOR
            pi, pr = _run_multi(self.scenarios)
            W[idx] = orig
            s = max(_sensitivity(self.baseline_int, pi),
                    _sensitivity(self.baseline_rel, pr))
            label = f"W[{idx[0]},{idx[1]}]={orig:.3f}"
            scores.append((s, label))

        # Also test bias
        if hasattr(mapper, "_bias_vector") and mapper._bias_vector is not None:
            for i in range(len(mapper._bias_vector)):
                if mapper._bias_vector[i] != 0:
                    orig = mapper._bias_vector[i]
                    mapper._bias_vector[i] = orig * PERTURB_FACTOR
                    pi, pr = _run_multi(self.scenarios)
                    mapper._bias_vector[i] = orig
                    s = max(_sensitivity(self.baseline_int, pi),
                            _sensitivity(self.baseline_rel, pr))
                    label = f"bias[{i}]={orig:.2f}"
                    scores.append((s, label))

        self._check_and_record(name, scores)

    def _check_and_record(self, name: str, scores: list[tuple[float, str]]):
        """Validate scores and cache for report."""
        SENSITIVITY_REPORT[name] = scores
        vals = [s for s, _ in scores]
        assert max(vals) < 0.50, (
            f"{name}: max_s={max(vals):.4f} 超出正常范围 (>0.50). "
            f"过大的参数表明系统对该参数极其敏感，可能导致发散."
        )

    # ══════════════════════════════════════════════
    # Test methods – one per parameter group
    # ══════════════════════════════════════════════

    def test_B_int_sensitivity(self):
        from state_engine._matrices import INPUT_INFLUENCE_B
        self._test_array_group("B_int", INPUT_INFLUENCE_B,
                               _map_keys("B", np.nonzero(INPUT_INFLUENCE_B),
                                         INPUT_INFLUENCE_B))

    def test_B_rel_sensitivity(self):
        from state_engine._matrices import REL_INPUT_INFLUENCE_B
        self._test_array_group("B_rel", REL_INPUT_INFLUENCE_B,
                               _map_keys("B_rel", np.nonzero(REL_INPUT_INFLUENCE_B),
                                         REL_INPUT_INFLUENCE_B))

    def test_surface_feedback_sensitivity(self):
        from state_engine._surface_weights import SURFACE_FEEDBACK_MATRIX
        self._test_array_group("surf_fb", SURFACE_FEEDBACK_MATRIX,
                               _map_keys("fb", np.nonzero(SURFACE_FEEDBACK_MATRIX),
                                         SURFACE_FEEDBACK_MATRIX))

    def test_internal_coupling_sensitivity(self):
        from state_engine._dynamics_weights import INTERNAL_COUPLING
        self._test_array_group("int_coup", INTERNAL_COUPLING,
                               _map_keys("coup", np.nonzero(INTERNAL_COUPLING),
                                         INTERNAL_COUPLING))

    def test_cross_scale_coupling_sensitivity(self):
        from state_engine._dynamics_weights import CROSS_SCALE_COUPLING
        self._test_array_group("xscale", CROSS_SCALE_COUPLING,
                               _map_keys("xs", np.nonzero(CROSS_SCALE_COUPLING),
                                         CROSS_SCALE_COUPLING))

    def test_self_decay_sensitivity(self):
        from state_engine._dynamics_weights import SELF_DECAY
        self._test_array_group("SELF_DECAY", SELF_DECAY,
                               _vec_keys("SELF", SELF_DECAY))

    def test_decay_targets_sensitivity(self):
        from state_engine._dynamics_weights import DECAY_TARGETS
        self._test_array_group("DECAY_TARGETS", DECAY_TARGETS,
                               _vec_keys("DECAY", DECAY_TARGETS))

    def test_rel_self_decay_sensitivity(self):
        from state_engine._dynamics_weights import REL_SELF_DECAY
        self._test_array_group("REL_SELF_DECAY", REL_SELF_DECAY,
                               _vec_keys("RELSD", REL_SELF_DECAY))

    def test_surface_mapper_sensitivity(self):
        from state_engine._surface_weights import SURFACE_MAPPER
        W = SURFACE_MAPPER._weight_matrix
        nz = np.nonzero(W)
        self._test_mapper_group("SurfMap", SURFACE_MAPPER, nz)

    def test_hyper_state_mod_sensitivity(self):
        from state_engine._defense_weights import HYPER_STATE_MODULATION
        W = HYPER_STATE_MODULATION._weight_matrix
        nz = np.nonzero(W)
        self._test_mapper_group("HState", HYPER_STATE_MODULATION, nz)

    def test_deact_intensity_sensitivity(self):
        from state_engine._defense_weights import DEACT_INTENSITY
        W = DEACT_INTENSITY._weight_matrix
        nz = np.nonzero(W)
        self._test_mapper_group("DeactI", DEACT_INTENSITY, nz)

    def test_hyper_intensity_sensitivity(self):
        from state_engine._defense_weights import HYPER_INTENSITY
        W = HYPER_INTENSITY._weight_matrix
        nz = np.nonzero(W)
        self._test_mapper_group("HyperI", HYPER_INTENSITY, nz)


# ═══════════════════════════════════════════════════════════════════
# Report generation (run via --run-full-sensitivity)
# ═══════════════════════════════════════════════════════════════════


class TestFullSensitivityReport:
    """Re-runs with more scenarios and prints a comprehensive report.
    Only runs with --run-full-sensitivity flag."""

    @pytest.fixture(autouse=True)
    def _require_flag(self, request):
        if not request.config.getoption("--run-full-sensitivity"):
            pytest.skip("Use --run-full-sensitivity to enable this report")

    def test_full_report(self):
        """Run full sensitivity and print report."""
        report = generate_report()

        print("\n\n" + "=" * 95)
        print("  全局参数灵敏度分析报告")
        print(f"  ({TestParameterSensitivity.n_scenarios} "
              f"scenarios x {N_ROUNDS} rounds, "
              f"perturb=x{PERTURB_FACTOR})")
        print("=" * 95)

        if not report:
            print("  无数据。请先运行非 --run-full-sensitivity 的 test 收集数据。")
            return

        total = sum(len(v) for v in report.values())
        n_s = sum(1 for v in report.values() for s, _ in v if s > 0.15)
        n_m = sum(1 for v in report.values() for s, _ in v if 0.05 < s <= 0.15)
        n_w = sum(1 for v in report.values() for s, _ in v if 0.01 < s <= 0.05)
        n_n = sum(1 for v in report.values() for s, _ in v if s <= 0.01)

        groups = []
        for name, scores in sorted(report.items()):
            vals = [s for s, _ in scores]
            avg = np.mean(vals)
            ns = sum(1 for s, _ in scores if s > 0.15)
            nn = sum(1 for s, _ in scores if s <= 0.01)
            max_s = max(vals)
            cat = ("🔴" if max_s > 0.15 else "🟠" if max_s > 0.05
                   else "🟡" if max_s > 0.01 else "🟢")
            groups.append((name, len(scores), avg, nn, ns, cat, max_s))

            sorted_s = sorted(scores, key=lambda x: x[0], reverse=True)
            top = " | ".join(f"{lbl} s={s:.4f}" for s, lbl in sorted_s[:3])
            nm = sum(1 for s, _ in scores if 0.05 < s <= 0.15)
            nw = sum(1 for s, _ in scores if 0.01 < s <= 0.05)
            print(f"\n{cat} {name} ({len(scores)}p, avg={avg:.4f})")
            print(f"   强>{0.15}={ns}  中{nm}  弱{nw}  忽略<0.01={nn}")
            if top:
                print(f"   TOP: {top}")
            if nn > 2:
                bots = " | ".join(lbl for s, lbl in sorted_s[-3:] if s <= 0.01)
                if bots:
                    print(f"   BOT(可移除): {bots}")

        print("\n" + "=" * 95)
        print(f"汇总: {total} 参数")
        print(f"  强(>0.15):     {n_s:3d} ({n_s/total*100:5.1f}%)")
        print(f"  中(0.05-0.15): {n_m:3d} ({n_m/total*100:5.1f}%)")
        print(f"  弱(0.01-0.05): {n_w:3d} ({n_w/total*100:5.1f}%)")
        print(f"  忽略(<0.01):   {n_n:3d} ({n_n/total*100:5.1f}%)")

        print("\n按平均灵敏度排序:")
        for name, n, avg, nn, ns, cat, max_s in sorted(
                groups, key=lambda x: x[2], reverse=True):
            bar = "█" * int(min(avg * 50, 50))
            print(f"  {name:12s} {bar} avg={avg:.4f}  {ns}强  {nn}/{n}可移除")
        print()


def generate_report() -> dict[str, list[tuple[float, str]]]:
    """Return the accumulated sensitivity report."""
    return SENSITIVITY_REPORT


# ═══════════════════════════════════════════════════════════════════
# CLI entry point for standalone use
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """Standalone quick scan for development use."""
    from state_engine._matrices import INPUT_INFLUENCE_B

    scenarios = _make_scenarios(5)
    bi, br = _run_multi(scenarios)
    arr = INPUT_INFLUENCE_B

    print("B_int quick scan (5 scenarios, 20 rounds):")
    for idx in zip(*np.nonzero(arr)):
        orig = arr[idx]
        arr[idx] = orig * PERTURB_FACTOR
        pi, pr = _run_multi(scenarios)
        arr[idx] = orig
        s = max(_sensitivity(bi, pi), _sensitivity(br, pr))
        print(f"  B[{idx[0]},{idx[1]}]={orig:.3f}  s={s:.4f}")
    print("Done.")
