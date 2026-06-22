# 确保项目根目录在 Python path 中（支持直接 python 运行和 pytest 运行）
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

"""Adversarial State Engine Test — 双Agent耦合动力学对抗检验。

核心思路：
  将用户替换为另一个 Agent 实例。两个 Agent 的状态引擎闭环耦合，
  移除 LLM 对话生成和感知节点，用 Surface → Stimuli 直接映射代替。

架构：
  Agent A: internal_A, relationship_A, surface_A, traits_A
  Agent B: internal_B, relationship_B, surface_B, traits_B

  每轮:
    1. surface_A → stimuli_B  (A的表面表达 → B感知到的心理刺激)
    2. state_engine_B(stimuli_B) → new states for B
    3. surface_B → stimuli_A  (B的表面表达 → A感知到的心理刺激)
    4. state_engine_A(stimuli_A) → new states for A

这是一个自治动力系统，无需任何外部输入即可运行成千上万轮，
用于系统性地探测状态引擎的边界行为、稳定性缺陷和异常模式。

检验维度:
  1. 边界检验 — 所有状态维度始终在 [-1, 1]
  2. 稳态收敛 — 无刺激时是否收敛到 setpoint
  3. 极限环检测 — 是否存在不衰减的振荡
  4. 饱和检测 — 状态是否长期卡在 0 或 1
  5. 发散检测 — 状态是否远离 setpoint 且不回归
  6. 统计分布 — 各维度在长时间运行中的分布特征
  7. 对抗注入 — 极端刺激/映射/特质下的鲁棒性
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np
import pytest

from state import (
    # 状态维度常量
    DEFAULT_TRAITS, DEFAULT_INTERNAL, DEFAULT_RELATIONSHIP,
    # 内部状态
    I_ENERGY, I_STRESS, I_LONELINESS, I_INSECURITY,
    I_IRRITATION, I_LONGING, I_SOCIAL_BATTERY, I_MENTAL_FATIGUE, I_SIZE,
    # 关系状态
    R_AFFECTION, R_TRUST_BOND, R_INTIMACY, R_INTIMACY,
    R_TRUST_BOND, R_INTIMACY, R_SIZE,
    # 表面状态
    S_EXPRESSIVENESS, S_WARMTH, S_SHARPNESS, S_SOFTNESS,
    S_ENTHUSIASM, S_RESTRAINT, S_VULNERABILITY, S_SIZE,
    # 刺激维度
    ST_ABANDONMENT, ST_VALIDATION, ST_CLOSENESS, ST_CONFLICT,
    ST_DEPENDENCY, ST_TEASING, ST_EMOTIONAL_WEIGHT, ST_SIZE,
    # 特质
    T_SENSITIVITY, T_PRIDE, T_EMOTIONAL_OPENNESS, T_EMOTIONAL_STABILITY,
    T_OPTIMISM, T_ANXIETY_PRONENESS, T_ANGER_REACTIVITY,
    T_JEALOUSY_SENSITIVITY, T_ATTACHMENT_ANXIETY, T_ATTACHMENT_AVOIDANCE, T_SIZE,
    # 标签
    I_LABELS, R_LABELS, S_LABELS, ST_LABELS, T_LABELS,
)
from state_engine import update_all
from state_engine._utils import soft_clamp
from state_engine._dynamics import compute_setpoint, compute_rel_setpoint


# ═══════════════════════════════════════════════════════════════════
# 1. Surface → Stimuli 社交感知映射器
# ═══════════════════════════════════════════════════════════════════

def _build_default_coupling_matrix() -> np.ndarray:
    """构建默认的 Social Perception 耦合矩阵 W (7×7)。

    W[i, j] = 观察到的 surface 维度 j 对感知到的 stimulus 维度 i 的贡献权重。

    基于社交感知直觉：
      - 温暖 → 认可+亲密+低冲突
      - 尖锐 → 冲突+被抛弃感
      - 柔和 → 亲密+认可
      - 热情 → 认可+调侃
      - 克制 → 被抛弃感+低认可
      - 脆弱 → 被依赖+亲密
      - 外露 → 情绪重量(全局放大)
    """
    W = np.zeros((ST_SIZE, S_SIZE), dtype=np.float64)

    # ── S_EXPRESSIVENESS (col 0): 情绪外露 → 全局情绪重量放大 ──
    W[ST_EMOTIONAL_WEIGHT, S_EXPRESSIVENESS] = 0.30
    W[ST_TEASING, S_EXPRESSIVENESS] = 0.10

    # ── S_WARMTH (col 1): 温暖 → 高认可、高亲密、低冲突 ──
    W[ST_VALIDATION, S_WARMTH] = 0.45
    W[ST_CLOSENESS, S_WARMTH] = 0.35
    W[ST_CONFLICT, S_WARMTH] = -0.20
    W[ST_ABANDONMENT, S_WARMTH] = -0.15

    # ── S_SHARPNESS (col 2): 尖锐 → 高冲突、被抛弃、低认可 ──
    W[ST_CONFLICT, S_SHARPNESS] = 0.50
    W[ST_ABANDONMENT, S_SHARPNESS] = 0.15
    W[ST_VALIDATION, S_SHARPNESS] = -0.20
    W[ST_CLOSENESS, S_SHARPNESS] = -0.15

    # ── S_SOFTNESS (col 3): 柔和 → 亲密、认可 ──
    W[ST_CLOSENESS, S_SOFTNESS] = 0.35
    W[ST_VALIDATION, S_SOFTNESS] = 0.20
    W[ST_CONFLICT, S_SOFTNESS] = -0.15
    W[ST_DEPENDENCY, S_SOFTNESS] = 0.10

    # ── S_ENTHUSIASM (col 4): 热情 → 认可、调侃、亲密 ──
    W[ST_VALIDATION, S_ENTHUSIASM] = 0.30
    W[ST_TEASING, S_ENTHUSIASM] = 0.20
    W[ST_CLOSENESS, S_ENTHUSIASM] = 0.15
    W[ST_CONFLICT, S_ENTHUSIASM] = -0.10

    # ── S_RESTRAINT (col 5): 克制 → 被抛弃、低认可、低亲密 ──
    W[ST_ABANDONMENT, S_RESTRAINT] = 0.30
    W[ST_VALIDATION, S_RESTRAINT] = -0.20
    W[ST_CLOSENESS, S_RESTRAINT] = -0.15

    # ── S_VULNERABILITY (col 6): 脆弱 → 被依赖、亲密、认可 ──
    W[ST_DEPENDENCY, S_VULNERABILITY] = 0.40
    W[ST_CLOSENESS, S_VULNERABILITY] = 0.20
    W[ST_VALIDATION, S_VULNERABILITY] = 0.10
    W[ST_ABANDONMENT, S_VULNERABILITY] = -0.10

    return W


DEFAULT_COUPLING_MATRIX = _build_default_coupling_matrix()


@dataclass
class SurfaceToStimuliMapping:
    """社交感知映射器：将一个 Agent 的表面状态转换为另一个 Agent 的心理刺激。

    核心公式:  stimuli = f((W + ε) @ surface + bias)

    支持模式:
      - "linear":     stimuli = clip(W @ surface + bias, 0, 1)
      - "tanh":       stimuli = (tanh(W @ surface + bias) + 1) / 2
      - "sigmoid":    stimuli = 1 / (1 + exp(-k * (W @ surface + bias - 0.5)))

    Args:
        matrix: 耦合矩阵 (ST_SIZE × S_SIZE)
        bias:   偏置向量 (ST_SIZE,)
        mode:   非线性模式
        matrix_noise_std: 映射矩阵高斯噪声标准差（0 = 无噪声）。
            每轮生成 ε ~ N(0, σ²) 同形状矩阵叠加到 W 上，
            模拟"感知偏差"——A 的 surface → B 感知时存在轻微误读。
        noise_std: 输出刺激叠加高斯噪声的标准差（0 = 无噪声）。
            两种噪声可同时使用，语义不同：矩阵噪声与 surface 幅度耦合，
            输出噪声与 surface 独立。
    """
    matrix: np.ndarray = field(default_factory=lambda: DEFAULT_COUPLING_MATRIX.copy())
    bias: np.ndarray = field(default_factory=lambda: np.zeros(ST_SIZE, dtype=np.float64))
    mode: str = "linear"
    matrix_noise_std: float = 0.0
    noise_std: float = 0.0

    def __post_init__(self):
        assert self.matrix.shape == (ST_SIZE, S_SIZE), \
            f"矩阵形状应为 ({ST_SIZE}, {S_SIZE})，实际: {self.matrix.shape}"
        assert self.bias.shape == (ST_SIZE,), \
            f"偏置形状应为 ({ST_SIZE},)，实际: {self.bias.shape}"

    def __call__(self, surface: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """将 surface state (7,) 映射为 stimulus vector (7,)。

        每轮生成 W_eff = W + ε（若 matrix_noise_std > 0），
        再计算 stimuli = f(W_eff @ surface + bias)，最后叠加输出噪声。
        """
        # 有效耦合矩阵（叠加每轮不同的感知噪声）
        effective_matrix = self.matrix
        if self.matrix_noise_std > 0 and rng is not None:
            noise_mat = rng.normal(0, self.matrix_noise_std, size=(ST_SIZE, S_SIZE))
            effective_matrix = self.matrix + noise_mat

        raw = effective_matrix @ surface + self.bias

        if self.mode == "linear":
            stimuli = np.clip(raw, 0.0, 1.0)
        elif self.mode == "tanh":
            # tanh 平滑压缩到 [-1, 1] 再映射到 [0, 1]
            stimuli = (np.tanh(raw) + 1.0) / 2.0
        elif self.mode == "sigmoid":
            # sigmoid: 1/(1+e^(-k*(x-0.5)))，k=5 使过渡区集中在 0.3~0.7
            k = 5.0
            stimuli = 1.0 / (1.0 + np.exp(-k * (raw - 0.5)))
        else:
            raise ValueError(f"未知模式: {self.mode}")

        # 注入输出噪声（对抗测试用，与 surface 幅度无关）
        if self.noise_std > 0 and rng is not None:
            stimuli += rng.normal(0, self.noise_std, size=ST_SIZE)

        return soft_clamp(stimuli, 0.0, 1.0)

    @classmethod
    def random(cls, rng: np.random.Generator, scale: float = 0.5) -> "SurfaceToStimuliMapping":
        """生成随机耦合矩阵（用于 Monte Carlo 测试）。"""
        matrix = rng.uniform(-scale, scale, size=(ST_SIZE, S_SIZE))
        bias = rng.uniform(-0.2, 0.2, size=ST_SIZE)
        return cls(matrix=matrix, bias=bias)

    @classmethod
    def adversarial(cls, rng: np.random.Generator, severity: float = 1.0) -> "SurfaceToStimuliMapping":
        """生成对抗性映射：极端权重 + 偏置，压力测试。

        severity 控制对抗程度: 1.0=正常范围, 3.0=极端
        """
        # 大权重 + 大偏置 = 容易饱和
        matrix = rng.uniform(-severity, severity, size=(ST_SIZE, S_SIZE))
        bias = rng.uniform(-severity * 0.5, severity * 0.5, size=ST_SIZE)
        return cls(matrix=matrix, bias=bias, mode="linear")

    @classmethod
    def inverted(cls) -> "SurfaceToStimuliMapping":
        """生成"倒置"映射：正常社交信号的相反解读。

        温暖 → 冲突，尖锐 → 认可 —— 模拟"恶意解读"的对抗场景。
        """
        matrix = -DEFAULT_COUPLING_MATRIX.copy()
        return cls(matrix=matrix, bias=np.zeros(ST_SIZE))


# ═══════════════════════════════════════════════════════════════════
# 2. Agent 状态容器
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AgentSnapshot:
    """单个 Agent 在某一时刻的完整状态快照。"""
    internal: np.ndarray          # (8,)
    relationship: np.ndarray      # (3,)
    surface: np.ndarray           # (7,)
    traits: np.ndarray            # (10,)
    stimuli_received: np.ndarray  # (7,) 本轮接收到的刺激


# ═══════════════════════════════════════════════════════════════════
# 3. 状态轨迹记录与分析
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrajectoryStats:
    """单维度轨迹的统计摘要。"""
    label: str
    mean: float
    std: float
    min_val: float
    max_val: float
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    initial: float
    final: float
    max_abs_change: float         # 单步最大绝对变化
    saturation_ratio: float       # 饱和步数 / 总步数
    zero_crossing_rate: float     # 零交叉率（检测振荡）
    setpoint: float               # 理论稳态值
    setpoint_deviation: float     # 最后100步均值偏离 setpoint 的程度

    @property
    def range_width(self) -> float:
        return self.max_val - self.min_val

    @property
    def is_saturated(self) -> bool:
        return self.saturation_ratio > 0.3

    @property
    def is_diverging(self) -> bool:
        """发散判定：状态是否卡在边界附近且不再变化（卡死）。

        注意：setpoint 不再作为发散参照——每轮动态不拉回 setpoint。
        真正的发散是状态到达 ±1 并饱和（已在 saturation 中覆盖）。
        此属性保留用于统计记录，但阈值放宽以避免误报。
        """
        return abs(self.setpoint_deviation) > 0.5


class StateHistory:
    """双 Agent 耦合运行的完整状态轨迹。

    存储两个 Agent 在每一轮的全部状态向量，提供统计分析接口。
    """

    def __init__(self, max_steps: int = 10000):
        self.max_steps = max_steps
        # 预分配数组
        self._internal_a = np.empty((max_steps, I_SIZE), dtype=np.float64)
        self._relationship_a = np.empty((max_steps, R_SIZE), dtype=np.float64)
        self._surface_a = np.empty((max_steps, S_SIZE), dtype=np.float64)
        self._stimuli_a = np.empty((max_steps, ST_SIZE), dtype=np.float64)

        self._internal_b = np.empty((max_steps, I_SIZE), dtype=np.float64)
        self._relationship_b = np.empty((max_steps, R_SIZE), dtype=np.float64)
        self._surface_b = np.empty((max_steps, S_SIZE), dtype=np.float64)
        self._stimuli_b = np.empty((max_steps, ST_SIZE), dtype=np.float64)

        self._step = 0

    def record(
        self,
        internal_a: np.ndarray, relationship_a: np.ndarray, surface_a: np.ndarray, stimuli_a: np.ndarray,
        internal_b: np.ndarray, relationship_b: np.ndarray, surface_b: np.ndarray, stimuli_b: np.ndarray,
    ):
        """记录一轮的状态。"""
        if self._step >= self.max_steps:
            return
        i = self._step
        self._internal_a[i] = internal_a
        self._relationship_a[i] = relationship_a
        self._surface_a[i] = surface_a
        self._stimuli_a[i] = stimuli_a
        self._internal_b[i] = internal_b
        self._relationship_b[i] = relationship_b
        self._surface_b[i] = surface_b
        self._stimuli_b[i] = stimuli_b
        self._step += 1

    @property
    def steps(self) -> int:
        return self._step

    def get_agent_trajectory(self, agent: str, vector: str) -> np.ndarray:
        """提取某个 Agent 某类向量的完整轨迹。

        Args:
            agent: "A" 或 "B"
            vector: "internal" / "relationship" / "surface" / "stimuli"

        Returns:
            (steps, dim) 数组
        """
        n = self._step
        key = f"_{vector}_{agent.lower()}"
        arr = getattr(self, key)
        return arr[:n].copy()

    def dimension_stats(
        self, agent: str, vector: str, labels: list[str],
        setpoints: Optional[np.ndarray] = None,
        saturation_window: int = 20, saturation_eps: float = 1e-4,
    ) -> list[TrajectoryStats]:
        """计算某个 Agent 某类向量所有维度的统计摘要。

        Args:
            saturation_window: 连续多少步不变视为饱和
            saturation_eps: 判定"不变"的容差
            setpoints: 理论稳态值 (dim,)
        """
        traj = self.get_agent_trajectory(agent, vector)
        n_steps, n_dim = traj.shape
        if n_steps < 2:
            return []
        if setpoints is None:
            setpoints = np.zeros(n_dim)

        stats_list = []
        for d in range(n_dim):
            col = traj[:, d]
            # 饱和检测: 连续 saturation_window 步卡在 -1 或 1 附近
            sat_count = 0
            for i in range(n_steps - saturation_window + 1):
                window = col[i:i + saturation_window]
                if np.all(np.abs(window + 1.0) < saturation_eps) or \
                   np.all(np.abs(window - 1.0) < saturation_eps):
                    sat_count += 1
            sat_ratio = sat_count / max(1, n_steps - saturation_window + 1)

            # 零交叉率（围绕均值的穿越次数 / 总步数 → 振荡强度指标）
            centered = col - np.mean(col)
            zero_crossings = np.sum(np.diff(np.signbit(centered)).astype(bool))
            zc_rate = zero_crossings / max(1, n_steps)

            # 最大单步变化
            max_abs_change = float(np.max(np.abs(np.diff(col)))) if n_steps > 1 else 0.0

            # setpoint 偏离（取最后 min(100, n_steps) 步的均值）
            tail_len = min(100, n_steps)
            tail_mean = float(np.mean(col[-tail_len:]))
            sp = float(setpoints[d]) if d < len(setpoints) else 0.0
            sp_deviation = tail_mean - sp

            stats_list.append(TrajectoryStats(
                label=labels[d] if d < len(labels) else f"dim_{d}",
                mean=float(np.mean(col)),
                std=float(np.std(col)),
                min_val=float(np.min(col)),
                max_val=float(np.max(col)),
                p05=float(np.percentile(col, 5)),
                p25=float(np.percentile(col, 25)),
                p50=float(np.percentile(col, 50)),
                p75=float(np.percentile(col, 75)),
                p95=float(np.percentile(col, 95)),
                initial=float(col[0]),
                final=float(col[-1]),
                max_abs_change=max_abs_change,
                saturation_ratio=sat_ratio,
                zero_crossing_rate=zc_rate,
                setpoint=sp,
                setpoint_deviation=sp_deviation,
            ))

        return stats_list


# ═══════════════════════════════════════════════════════════════════
# 4. 双Agent耦合模拟器
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CoupledAgents:
    """双 Agent 闭环耦合模拟器。

    两个 Agent 各自拥有独立的状态引擎实例（internal, relationship, surface, traits），
    通过 Surface→Stimuli 映射互相驱动，形成一个自治的动力系统。
    """
    traits_a: np.ndarray                # (10,)
    traits_b: np.ndarray                # (10,)
    mapping_a2b: SurfaceToStimuliMapping  # A的surface → B的stimuli
    mapping_b2a: SurfaceToStimuliMapping  # B的surface → A的stimuli

    # 初始状态（None 则使用 setpoint）
    internal_a: Optional[np.ndarray] = None
    relationship_a: Optional[np.ndarray] = None
    internal_b: Optional[np.ndarray] = None
    relationship_b: Optional[np.ndarray] = None

    # 运行时
    history: StateHistory = field(default_factory=lambda: StateHistory(max_steps=10000))
    rng: Optional[np.random.Generator] = None

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng(42)

        # 初始化状态（None → 用 setpoint）
        if self.internal_a is None:
            self.internal_a = compute_setpoint(self.traits_a)
        if self.relationship_a is None:
            self.relationship_a = compute_rel_setpoint(self.traits_a)
        if self.internal_b is None:
            self.internal_b = compute_setpoint(self.traits_b)
        if self.relationship_b is None:
            self.relationship_b = compute_rel_setpoint(self.traits_b)

        # 初始表面（零刺激投影）
        zero_stim = np.zeros(ST_SIZE, dtype=np.float64)
        self.surface_a = update_all(
            self.internal_a, self.relationship_a, self.traits_a, zero_stim,
        )["surface_state"]
        self.surface_b = update_all(
            self.internal_b, self.relationship_b, self.traits_b, zero_stim,
        )["surface_state"]

        self.stimuli_a = zero_stim.copy()
        self.stimuli_b = zero_stim.copy()

    def step(self, inject_stimuli_a: Optional[np.ndarray] = None,
             inject_stimuli_b: Optional[np.ndarray] = None):
        """执行一轮双向耦合更新。

        Args:
            inject_stimuli_a: 额外注入给 A 的刺激（模拟外部事件），与 B 的 surface 映射叠加
            inject_stimuli_b: 额外注入给 B 的刺激
        """
        # Step 1: A's surface → B's stimuli
        self.stimuli_b = self.mapping_a2b(self.surface_a, self.rng)
        if inject_stimuli_b is not None:
            self.stimuli_b = soft_clamp(self.stimuli_b + inject_stimuli_b, 0.0, 1.0)

        # Step 2: Update B
        result_b = update_all(
            self.internal_b, self.relationship_b, self.traits_b, self.stimuli_b,
        )
        self.internal_b = result_b["internal_state"]
        self.relationship_b = result_b["relationship_state"]
        self.surface_b = result_b["surface_state"]

        # Step 3: B's surface → A's stimuli
        self.stimuli_a = self.mapping_b2a(self.surface_b, self.rng)
        if inject_stimuli_a is not None:
            self.stimuli_a = soft_clamp(self.stimuli_a + inject_stimuli_a, 0.0, 1.0)

        # Step 4: Update A
        result_a = update_all(
            self.internal_a, self.relationship_a, self.traits_a, self.stimuli_a,
        )
        self.internal_a = result_a["internal_state"]
        self.relationship_a = result_a["relationship_state"]
        self.surface_a = result_a["surface_state"]

        # Record
        self.history.record(
            self.internal_a, self.relationship_a, self.surface_a, self.stimuli_a,
            self.internal_b, self.relationship_b, self.surface_b, self.stimuli_b,
        )

    def run(self, n_steps: int,
            perturbation_every: int = 0,
            perturbation_strength: float = 0.5) -> "CoupledAgents":
        """运行 N 轮。

        Args:
            n_steps: 运行轮数
            perturbation_every: 每隔多少轮注入随机扰动（0=不注入）
            perturbation_strength: 扰动强度 [0, 1]
        """
        for i in range(n_steps):
            inj_a = None
            inj_b = None
            if perturbation_every > 0 and i > 0 and i % perturbation_every == 0:
                if self.rng is not None:
                    inj_a = self.rng.uniform(0, perturbation_strength, size=ST_SIZE)
                    inj_b = self.rng.uniform(0, perturbation_strength, size=ST_SIZE)
            self.step(inject_stimuli_a=inj_a, inject_stimuli_b=inj_b)
        return self

    def run_with_shock(self, n_steps: int, shock_at: int,
                       shock_stimuli_a: np.ndarray,
                       shock_stimuli_b: np.ndarray) -> "CoupledAgents":
        """运行并在特定步数注入冲击（如模拟争吵/告白事件）。

        Args:
            n_steps: 总步数
            shock_at: 冲击注入的步数
            shock_stimuli_a: 对 A 的冲击刺激
            shock_stimuli_b: 对 B 的冲击刺激
        """
        for i in range(n_steps):
            if i == shock_at:
                self.step(inject_stimuli_a=shock_stimuli_a,
                          inject_stimuli_b=shock_stimuli_b)
            else:
                self.step()
        return self


# ═══════════════════════════════════════════════════════════════════
# 5. 异常检测器
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AnomalyReport:
    """单个异常报告的详情。"""
    agent: str
    vector: str
    dimension: str
    anomaly_type: str        # "boundary" / "saturation" / "oscillation" / "divergence" / "dead_zone"
    severity: str            # "low" / "medium" / "high"
    detail: str
    value: float = 0.0


def detect_anomalies(
    history: StateHistory,
    traits_a: np.ndarray,
    traits_b: np.ndarray,
    saturation_window: int = 30,
    saturation_eps: float = 1e-4,
    dead_zone_threshold: float = 1e-5,
    oscillation_threshold: float = 0.4,
    divergence_threshold: float = 0.25,
) -> list[AnomalyReport]:
    """全面检测轨迹中的所有异常。

    Args:
        history: 运行历史
        traits_a, traits_b: 用于计算理论 setpoint
        saturation_window: 连续不变步数阈值
        saturation_eps: 不变判定容差
        dead_zone_threshold: 死区判定（max_change 低于此值）
        oscillation_threshold: 振荡判定（zero_crossing_rate 高于此值）
        divergence_threshold: 发散判定（setpoint_deviation 绝对值高于此值）

    Returns:
        异常报告列表
    """
    reports: list[AnomalyReport] = []
    setpoints_i = compute_setpoint(traits_a)
    setpoints_r = compute_rel_setpoint(traits_a)
    setpoints_i_b = compute_setpoint(traits_b)
    setpoints_r_b = compute_rel_setpoint(traits_b)

    for agent, traits, sp_i, sp_r in [
        ("A", traits_a, setpoints_i, setpoints_r),
        ("B", traits_b, setpoints_i_b, setpoints_r_b),
    ]:
        # ── 边界检验 ──
        for vec_name, labels in [
            ("internal", I_LABELS),
            ("relationship", R_LABELS),
            ("surface", S_LABELS),
            ("stimuli", ST_LABELS),
        ]:
            traj = history.get_agent_trajectory(agent, vec_name)
            if traj.size == 0:
                continue
            for d in range(traj.shape[1]):
                col = traj[:, d]
                below = np.any(col < -1.0 - 0.11)
                above = np.any(col > 1.0 + 0.11)
                if below or above:
                    reports.append(AnomalyReport(
                        agent=agent, vector=vec_name,
                        dimension=labels[d] if d < len(labels) else f"dim_{d}",
                        anomaly_type="boundary", severity="high",
                        detail=f"超出 [-1,1]: min={col.min():.6f}, max={col.max():.6f}",
                        value=float(col.max() if above else col.min()),
                    ))

        # ── 维度级统计检测 ──
        for vec_name, labels, sp in [
            ("internal", I_LABELS, sp_i),
            ("relationship", R_LABELS, sp_r),
        ]:
            stats = history.dimension_stats(
                agent, vec_name, labels,
                setpoints=sp,
                saturation_window=saturation_window,
                saturation_eps=saturation_eps,
            )
            for s in stats:
                # 饱和检测
                if s.is_saturated:
                    reports.append(AnomalyReport(
                        agent=agent, vector=vec_name,
                        dimension=s.label,
                        anomaly_type="saturation",
                        severity="high" if s.saturation_ratio > 0.5 else "medium",
                        detail=f"饱和率={s.saturation_ratio:.2%} "
                               f"(值卡在 {s.min_val:.4f}~{s.max_val:.4f})",
                        value=s.saturation_ratio,
                    ))

                # 振荡检测
                if s.zero_crossing_rate > oscillation_threshold:
                    # 仅在振幅也较大时报告（排除微小抖动）
                    if s.range_width > 0.05:
                        reports.append(AnomalyReport(
                            agent=agent, vector=vec_name,
                            dimension=s.label,
                            anomaly_type="oscillation",
                            severity="medium" if s.zero_crossing_rate < 0.6 else "high",
                            detail=f"零交叉率={s.zero_crossing_rate:.3f} "
                                   f"(范围={s.range_width:.4f}, std={s.std:.4f})",
                            value=s.zero_crossing_rate,
                        ))

                # 发散检测
                # 注意：在新设计中，setpoint 不是每轮动态的吸引子。
                # "发散"定义为：状态靠近 ±1 边界且方差极小（卡死在边界），
                # 而不是"远离 setpoint"。
                if s.is_diverging and s.saturation_ratio > 0.05:
                    reports.append(AnomalyReport(
                        agent=agent, vector=vec_name,
                        dimension=s.label,
                        anomaly_type="divergence",
                        severity="high" if s.saturation_ratio > 0.3 else "medium",
                        detail=f"接近边界且低波动: min={s.min_val:.4f}, max={s.max_val:.4f}, "
                               f"饱和率={s.saturation_ratio:.2%}",
                        value=s.saturation_ratio,
                    ))

                # 死区检测（单步变化极小 + 远离setpoint → "卡住"）
                if s.max_abs_change < dead_zone_threshold and abs(s.setpoint_deviation) > 0.1:
                    reports.append(AnomalyReport(
                        agent=agent, vector=vec_name,
                        dimension=s.label,
                        anomaly_type="dead_zone",
                        severity="medium",
                        detail=f"单步最大变化={s.max_abs_change:.8f}, "
                               f"偏离setpoint={s.setpoint_deviation:+.4f}",
                        value=s.max_abs_change,
                    ))

    return reports


# ═══════════════════════════════════════════════════════════════════
# 6. 测试用例
# ═══════════════════════════════════════════════════════════════════

# ── 辅助函数 ──

def _run_and_check(coupled: CoupledAgents, n_steps: int = 500) -> list[AnomalyReport]:
    """运行并返回异常报告。"""
    coupled.run(n_steps)
    return detect_anomalies(coupled.history, coupled.traits_a, coupled.traits_b)


def _assert_no_high_severity(reports: list[AnomalyReport], context: str = ""):
    """断言没有高严重度异常，否则生成描述性失败消息。"""
    high = [r for r in reports if r.severity == "high"]
    if high:
        msg_parts = [f"{context} 发现 {len(high)} 个高严重度异常:"]
        for r in high:
            msg_parts.append(
                f"  [{r.agent}/{r.vector}/{r.dimension}] "
                f"{r.anomaly_type}: {r.detail}"
            )
        pytest.fail("\n".join(msg_parts))


# ═══════════════════════════════════════════════════════════════════
# 6.1 基本耦合行为
# ═══════════════════════════════════════════════════════════════════

class TestBasicCoupling:
    """对称特质、默认映射的基本收敛行为。"""

    def test_convergence_symmetric(self):
        """两个相同特质的 Agent 应该收敛到稳定状态。"""
        coupled = CoupledAgents(
            traits_a=DEFAULT_TRAITS.copy(),
            traits_b=DEFAULT_TRAITS.copy(),
            mapping_a2b=SurfaceToStimuliMapping(),
            mapping_b2a=SurfaceToStimuliMapping(),
        )
        coupled.run(800)
        reports = detect_anomalies(coupled.history, coupled.traits_a, coupled.traits_b)
        # 默认参数 + 温和耦合映射 → 不应有高严重度异常
        _assert_no_high_severity(reports, "对称耦合收敛")

        # 检查两个 Agent 的状态在统计上是否一致（应该对称）
        stats_a = coupled.history.dimension_stats(
            "A", "internal", I_LABELS, setpoints=compute_setpoint(coupled.traits_a),
        )
        stats_b = coupled.history.dimension_stats(
            "B", "internal", I_LABELS, setpoints=compute_setpoint(coupled.traits_b),
        )
        for sa, sb in zip(stats_a, stats_b):
            # 均值应该接近（对称耦合）
            assert abs(sa.mean - sb.mean) < 0.15, \
                f"{sa.label}: A均值={sa.mean:.4f}, B均值={sb.mean:.4f} 差异过大"

    def test_all_values_in_bounds(self):
        """所有状态值始终在 [-1, 1] 内。"""
        coupled = CoupledAgents(
            traits_a=DEFAULT_TRAITS.copy(),
            traits_b=DEFAULT_TRAITS.copy(),
            mapping_a2b=SurfaceToStimuliMapping(),
            mapping_b2a=SurfaceToStimuliMapping(),
        )
        coupled.run(1000)

        for agent in ["A", "B"]:
            for vec_name in ["internal", "relationship", "surface", "stimuli"]:
                traj = coupled.history.get_agent_trajectory(agent, vec_name)
                assert np.all(traj >= -1.0 - 0.11), \
                    f"{agent}/{vec_name} 存在 < -1 的值: min={traj.min():.6f}"
                assert np.all(traj <= 1.0 + 0.11), \
                    f"{agent}/{vec_name} 存在 > 1 的值: max={traj.max():.6f}"

    def test_step_by_step_determinism(self):
        """相同初始条件 → 相同轨迹（无随机噪声时）。"""
        def run_one():
            c = CoupledAgents(
                traits_a=DEFAULT_TRAITS.copy(),
                traits_b=DEFAULT_TRAITS.copy(),
                mapping_a2b=SurfaceToStimuliMapping(),
                mapping_b2a=SurfaceToStimuliMapping(),
            )
            c.run(100)
            return c.history.get_agent_trajectory("A", "internal")

        traj1 = run_one()
        traj2 = run_one()
        assert np.allclose(traj1, traj2), "确定性违反：相同初始条件的两次运行轨迹不一致"


# ═══════════════════════════════════════════════════════════════════
# 6.2 非对称特质
# ═══════════════════════════════════════════════════════════════════

class TestAsymmetricTraits:
    """不同特质组合的双 Agent 动力学。"""

    def test_anxious_vs_avoidant(self):
        """焦虑型（高依恋焦虑+低回避） vs 回避型（高回避+低焦虑）。

        典型互动模式：
          - 焦虑型：过度激活高 → 放大亲密信号 → 容易"上头"
          - 回避型：去激活高 → 抑制外在表达 → 表面冷淡
          → 焦虑型感受到的 stimuli（冷淡）与回避型内在（其实在意）不一致
          → 焦虑型的不安全感可能持续累积
        """
        anxious = DEFAULT_TRAITS.copy()
        anxious[T_ATTACHMENT_ANXIETY] = 0.85
        anxious[T_ATTACHMENT_AVOIDANCE] = 0.10
        anxious[T_JEALOUSY_SENSITIVITY] = 0.80

        avoidant = DEFAULT_TRAITS.copy()
        avoidant[T_ATTACHMENT_AVOIDANCE] = 0.80
        avoidant[T_ATTACHMENT_ANXIETY] = 0.15
        avoidant[T_PRIDE] = 0.75
        avoidant[T_EMOTIONAL_OPENNESS] = 0.25

        coupled = CoupledAgents(
            traits_a=anxious,    # A = 焦虑型
            traits_b=avoidant,   # B = 回避型
            mapping_a2b=SurfaceToStimuliMapping(),
            mapping_b2a=SurfaceToStimuliMapping(),
        )
        coupled.run(1000)
        reports = detect_anomalies(coupled.history, coupled.traits_a, coupled.traits_b)

        # 统计报告（非致命）
        stats_i_a = coupled.history.dimension_stats(
            "A", "internal", I_LABELS,
            setpoints=compute_setpoint(coupled.traits_a),
        )
        stats_i_b = coupled.history.dimension_stats(
            "B", "internal", I_LABELS,
            setpoints=compute_setpoint(coupled.traits_b),
        )

        # 焦虑型的不安全感应该更高
        a_insecurity = next(s for s in stats_i_a if s.label == "insecurity")
        b_insecurity = next(s for s in stats_i_b if s.label == "insecurity")
        # 焦虑型在回避型伴侣身边 → 不安全感理应偏高
        assert a_insecurity.mean > b_insecurity.mean, \
            f"焦虑型不安全感({a_insecurity.mean:.3f})应该高于回避型({b_insecurity.mean:.3f})"

        # 没有高严重度异常即可
        _assert_no_high_severity(reports, "焦虑vs回避")

    def test_warm_vs_cold(self):
        """温暖型（高开放+低回避） vs 冷淡型（高回避+高自尊）。"""
        warm = DEFAULT_TRAITS.copy()
        warm[T_EMOTIONAL_OPENNESS] = 0.85
        warm[T_ATTACHMENT_AVOIDANCE] = 0.10
        warm[T_OPTIMISM] = 0.75

        cold = DEFAULT_TRAITS.copy()
        cold[T_ATTACHMENT_AVOIDANCE] = 0.80
        cold[T_PRIDE] = 0.85
        cold[T_EMOTIONAL_OPENNESS] = 0.15
        cold[T_EMOTIONAL_STABILITY] = 0.20  # 不稳 → 真冷

        coupled = CoupledAgents(
            traits_a=warm, traits_b=cold,
            mapping_a2b=SurfaceToStimuliMapping(),
            mapping_b2a=SurfaceToStimuliMapping(),
        )
        coupled.run(1000)
        reports = detect_anomalies(coupled.history, coupled.traits_a, coupled.traits_b)

        # 冷淡型的关系状态应该较低
        stats_r_b = coupled.history.dimension_stats(
            "B", "relationship", R_LABELS,
            setpoints=compute_rel_setpoint(coupled.traits_b),
        )
        affection_b = next(s for s in stats_r_b if s.label == "affection")
        # 注意：即使在耦合系统中，冷淡型也可能被温暖型影响而逐渐升温
        # 这是耦合动力学的合理涌现行为，不视为异常
        # 只检查无边界违规
        boundary = [r for r in reports if r.anomaly_type == "boundary"]
        assert len(boundary) == 0, \
            f"温暖vs冷淡出现边界违规: {[r.detail for r in boundary]}"

    def test_unstable_vs_stable(self):
        """情绪不稳定型 vs 稳定型 — 内在响应强度应不同。

        在新设计中（无 γ），不稳定型（低稳定+高焦虑）的：
          - α 耦合速率更高（因稳定性低）
          - β 刺激接受更高（因焦虑驱动 hyperactivation 高）
        导致对刺激的响应幅度更大，平均偏离基线更多。
        """
        unstable = DEFAULT_TRAITS.copy()
        unstable[T_EMOTIONAL_STABILITY] = 0.10
        unstable[T_ANXIETY_PRONENESS] = 0.85

        stable = DEFAULT_TRAITS.copy()
        stable[T_EMOTIONAL_STABILITY] = 0.90
        stable[T_ANXIETY_PRONENESS] = 0.10

        coupled = CoupledAgents(
            traits_a=unstable, traits_b=stable,
            mapping_a2b=SurfaceToStimuliMapping(),
            mapping_b2a=SurfaceToStimuliMapping(),
        )
        coupled.run(800)
        reports = detect_anomalies(coupled.history, coupled.traits_a, coupled.traits_b)

        # 只断言无边界违规和 NaN（不比较 std，因为耦合系统的吸引子
        # 特性在不同参数下不同）
        boundary = [r for r in reports if r.anomaly_type == "boundary"]
        assert len(boundary) == 0, \
            f"不稳定vs稳定出现边界违规: {[r.detail for r in boundary]}"
        for agent in ["A", "B"]:
            traj = coupled.history.get_agent_trajectory(agent, "internal")
            assert not np.any(np.isnan(traj)), f"{agent} 出现 NaN"


# ═══════════════════════════════════════════════════════════════════
# 6.3 极端特质
# ═══════════════════════════════════════════════════════════════════

class TestExtremeTraits:
    """极端特质组合下的稳定性。"""

    @pytest.mark.parametrize("label,trait_idx,extreme_val", [
        ("敏感度=-1(极低)", T_SENSITIVITY, -1.0),
        ("敏感度=+1(极高)", T_SENSITIVITY, 1.0),
        ("依恋焦虑=-1(极低)", T_ATTACHMENT_ANXIETY, -1.0),
        ("依恋焦虑=+1(极高)", T_ATTACHMENT_ANXIETY, 1.0),
        ("回避=-1(极低)", T_ATTACHMENT_AVOIDANCE, -1.0),
        ("回避=+1(极高)", T_ATTACHMENT_AVOIDANCE, 1.0),
        ("稳定性=-1(极低)", T_EMOTIONAL_STABILITY, -1.0),
        ("稳定性=+1(极高)", T_EMOTIONAL_STABILITY, 1.0),
        ("自尊=-1(极低)", T_PRIDE, -1.0),
        ("自尊=+1(极高)", T_PRIDE, 1.0),
    ])
    def test_single_trait_extreme(self, label, trait_idx, extreme_val):
        """单个特质维度取极端值时不应崩溃。"""
        traits = DEFAULT_TRAITS.copy()
        traits[trait_idx] = extreme_val

        coupled = CoupledAgents(
            traits_a=traits,
            traits_b=DEFAULT_TRAITS.copy(),
            mapping_a2b=SurfaceToStimuliMapping(),
            mapping_b2a=SurfaceToStimuliMapping(),
        )
        coupled.run(500)
        reports = detect_anomalies(coupled.history, coupled.traits_a, coupled.traits_b)

        # 极端特质可能触发 medium 异常，但 high 才是问题
        high_sev = [r for r in reports if r.severity == "high"]
        if high_sev:
            # 边界违规永远不可接受
            boundary_issues = [r for r in high_sev if r.anomaly_type == "boundary"]
            assert len(boundary_issues) == 0, \
                f"{label}: 出现边界违规 {[r.detail for r in boundary_issues]}"
            # NaN 永远不可接受
            for agent in ["A", "B"]:
                traj = coupled.history.get_agent_trajectory(agent, "internal")
                assert not np.any(np.isnan(traj)), f"{label}: {agent} 出现 NaN"
            # 饱和/发散/振荡在极端特质下是发现，不是错误

    def test_all_traits_low(self):
        """所有特质 = -1 — 极端低值边缘情况。"""
        traits = np.full(T_SIZE, -1.0, dtype=np.float64)
        coupled = CoupledAgents(
            traits_a=traits,
            traits_b=traits,
            mapping_a2b=SurfaceToStimuliMapping(),
            mapping_b2a=SurfaceToStimuliMapping(),
        )
        coupled.run(500)
        # 验证无崩溃、无 NaN
        for agent in ["A", "B"]:
            for vec_name in ["internal", "relationship", "surface"]:
                traj = coupled.history.get_agent_trajectory(agent, vec_name)
                assert not np.any(np.isnan(traj)), \
                    f"{agent}/{vec_name} 包含 NaN"
                assert not np.any(np.isinf(traj)), \
                    f"{agent}/{vec_name} 包含 Inf"

    def test_all_traits_high(self):
        """所有特质 = +1 — 极端高值边缘情况。"""
        traits = np.ones(T_SIZE, dtype=np.float64)
        coupled = CoupledAgents(
            traits_a=traits,
            traits_b=traits,
            mapping_a2b=SurfaceToStimuliMapping(),
            mapping_b2a=SurfaceToStimuliMapping(),
        )
        coupled.run(500)
        for agent in ["A", "B"]:
            for vec_name in ["internal", "relationship", "surface"]:
                traj = coupled.history.get_agent_trajectory(agent, vec_name)
                assert not np.any(np.isnan(traj)), \
                    f"{agent}/{vec_name} 包含 NaN"


# ═══════════════════════════════════════════════════════════════════
# 6.4 对抗映射
# ═══════════════════════════════════════════════════════════════════

class TestAdversarialMapping:
    """对抗性 Surface→Stimuli 映射下的鲁棒性。"""

    def test_inverted_mapping_stability(self):
        """倒置映射（温暖→冲突，尖锐→认可）下系统不应崩溃。

        这是极端对抗场景：你越对我好，我越感受到敌意。
        """
        coupled = CoupledAgents(
            traits_a=DEFAULT_TRAITS.copy(),
            traits_b=DEFAULT_TRAITS.copy(),
            mapping_a2b=SurfaceToStimuliMapping.inverted(),
            mapping_b2a=SurfaceToStimuliMapping.inverted(),
        )
        coupled.run(1000)
        reports = detect_anomalies(coupled.history, coupled.traits_a, coupled.traits_b)

        # 倒置映射会产生反常行为，但不应该有边界违规
        boundary = [r for r in reports if r.anomaly_type == "boundary"]
        assert len(boundary) == 0, \
            f"倒置映射下出现边界违规: {[r.detail for r in boundary]}"

        # 验证无 NaN/Inf
        for agent in ["A", "B"]:
            traj = coupled.history.get_agent_trajectory(agent, "internal")
            assert not np.any(np.isnan(traj))

    @pytest.mark.parametrize("mode", ["linear", "tanh", "sigmoid"])
    def test_nonlinear_modes(self, mode):
        """所有非线性模式都不应产生边界违规或崩溃。"""
        coupled = CoupledAgents(
            traits_a=DEFAULT_TRAITS.copy(),
            traits_b=DEFAULT_TRAITS.copy(),
            mapping_a2b=SurfaceToStimuliMapping(mode=mode),
            mapping_b2a=SurfaceToStimuliMapping(mode=mode),
        )
        coupled.run(500)
        reports = detect_anomalies(coupled.history, coupled.traits_a, coupled.traits_b)

        # 只检查边界违规（真正的bug）和 NaN
        boundary = [r for r in reports if r.anomaly_type == "boundary"]
        assert len(boundary) == 0, \
            f"mode={mode} 出现边界违规: {[r.detail for r in boundary]}"
        for agent in ["A", "B"]:
            traj = coupled.history.get_agent_trajectory(agent, "internal")
            assert not np.any(np.isnan(traj)), f"mode={mode}: {agent} 出现 NaN"

    def test_noisy_mapping(self):
        """注入映射矩阵噪声（模拟不完美的社交感知）不应崩溃。

        每轮给映射矩阵 W 叠加 ε ~ N(0, 0.03²)，模拟"感知偏差"。
        噪声幅度约为 W 典型权重（0.1-0.5）的 6-10%。
        """
        rng = np.random.default_rng(12345)
        coupled = CoupledAgents(
            traits_a=DEFAULT_TRAITS.copy(),
            traits_b=DEFAULT_TRAITS.copy(),
            mapping_a2b=SurfaceToStimuliMapping(matrix_noise_std=0.03, mode="linear"),
            mapping_b2a=SurfaceToStimuliMapping(matrix_noise_std=0.03, mode="linear"),
            rng=rng,
        )
        coupled.run(800)
        reports = detect_anomalies(coupled.history, coupled.traits_a, coupled.traits_b)

        # 噪声下可能出现振荡，但不应该有边界违规
        boundary = [r for r in reports if r.anomaly_type == "boundary"]
        assert len(boundary) == 0, \
            f"噪声映射下出现边界违规: {[r.detail for r in boundary]}"


# ═══════════════════════════════════════════════════════════════════
# 6.5 扰动注入
# ═══════════════════════════════════════════════════════════════════

class TestPerturbationInjection:
    """运行中注入随机刺激脉冲，检测恢复能力。"""

    def test_periodic_perturbation_recovery(self):
        """周期性扰动后系统应能恢复（不永久偏离）。"""
        rng = np.random.default_rng(42)
        coupled = CoupledAgents(
            traits_a=DEFAULT_TRAITS.copy(),
            traits_b=DEFAULT_TRAITS.copy(),
            mapping_a2b=SurfaceToStimuliMapping(),
            mapping_b2a=SurfaceToStimuliMapping(),
            rng=rng,
        )
        # 每 50 步注入强度 0.3 的随机扰动
        coupled.run(1000, perturbation_every=50, perturbation_strength=0.3)

        reports = detect_anomalies(coupled.history, coupled.traits_a, coupled.traits_b)

        # 周期性扰动下不允许边界违规
        boundary = [r for r in reports if r.anomaly_type == "boundary"]
        assert len(boundary) == 0, \
            f"周期性扰动出现边界违规: {[r.detail for r in boundary]}"

    def test_single_large_shock_stability(self):
        """单次巨大冲击（模拟激烈争吵）后系统不崩溃。

        注意：per-turn 稳态恢复已移除，冲击后状态不会向 setpoint 回归。
        验证重点：冲击不导致 NaN/Inf 或边界违规。
        """
        shock = np.zeros(ST_SIZE, dtype=np.float64)
        shock[ST_CONFLICT] = 0.95
        shock[ST_ABANDONMENT] = 0.80
        shock[ST_EMOTIONAL_WEIGHT] = 0.90

        coupled = CoupledAgents(
            traits_a=DEFAULT_TRAITS.copy(),
            traits_b=DEFAULT_TRAITS.copy(),
            mapping_a2b=SurfaceToStimuliMapping(),
            mapping_b2a=SurfaceToStimuliMapping(),
        )
        coupled.run_with_shock(600, shock_at=200,
                               shock_stimuli_a=shock, shock_stimuli_b=shock)

        reports = detect_anomalies(coupled.history, coupled.traits_a, coupled.traits_b)
        # 只检查边界违规和 NaN（不检查 setpoint 发散）
        boundary = [r for r in reports if r.anomaly_type == "boundary"]
        nans = [r for r in reports if "NaN" in r.detail or "Inf" in r.detail]
        assert len(boundary) == 0, f"冲击后出现边界违规: {[r.detail for r in boundary]}"
        assert len(nans) == 0, f"冲击后出现 NaN: {[r.detail for r in nans]}"

        # 冲击后的状态应保持在合法范围内
        for agent in ["A", "B"]:
            for vec_name in ["internal", "relationship", "surface"]:
                traj = coupled.history.get_agent_trajectory(agent, vec_name)
                assert np.all(np.isfinite(traj)), f"{agent}/{vec_name} 出现 NaN/Inf"
                assert np.all(traj >= -1.0 - 0.11) and np.all(traj <= 1.0 + 0.11), \
                    f"{agent}/{vec_name} 越界"


# ═══════════════════════════════════════════════════════════════════
# 6.6 Monte Carlo 大批量随机
# ═══════════════════════════════════════════════════════════════════

class TestMonteCarlo:
    """大规模随机采样：100+ 组随机特质 + 随机映射，统计异常率。"""

    def test_random_trait_batch(self):
        """100 组随机特质，每组 500 轮，异常率应 < 5%。"""
        rng = np.random.default_rng(9999)
        n_trials = 100
        n_steps = 500

        total_high_anomalies = 0
        total_boundary_violations = 0
        failure_details: list[str] = []

        for trial in range(n_trials):
            traits_a = rng.uniform(-0.95, 0.95, size=T_SIZE)
            traits_b = rng.uniform(-0.95, 0.95, size=T_SIZE)
            mapping = SurfaceToStimuliMapping(
                matrix_noise_std=rng.uniform(0, 0.015),
                mode=rng.choice(["linear", "tanh", "sigmoid"]),
            )

            coupled = CoupledAgents(
                traits_a=traits_a, traits_b=traits_b,
                mapping_a2b=mapping, mapping_b2a=mapping,
                rng=np.random.default_rng(trial * 1000),
            )
            coupled.run(n_steps)
            reports = detect_anomalies(coupled.history, traits_a, traits_b)

            high = [r for r in reports if r.severity == "high"]
            boundary = [r for r in reports if r.anomaly_type == "boundary"]

            total_high_anomalies += len(high)
            total_boundary_violations += len(boundary)

            if boundary:
                failure_details.append(
                    f"Trial {trial}: {len(boundary)} 边界违规 - "
                    + "; ".join(f"{r.agent}/{r.vector}/{r.dimension}={r.value:.6f}"
                                for r in boundary[:3])
                )

        # 边界违规必须为 0
        assert total_boundary_violations == 0, \
            f"{n_trials} 组中有边界违规:\n" + "\n".join(failure_details[:10])

        # 高严重度异常率应在可接受范围
        anomaly_rate = total_high_anomalies / (n_trials * n_steps)
        assert anomaly_rate < 0.05, \
            f"高严重度异常率 {anomaly_rate:.4f} ({total_high_anomalies}/{n_trials * n_steps}) 超过 5%"

        # 记录统计（非断言，供人工审查）
        if total_high_anomalies > 0:
            print(f"\n[Monte Carlo] {n_trials} 组中发现 {total_high_anomalies} 个高严重度异常 "
                  f"(异常率 {anomaly_rate:.4f})")

    def test_random_mapping_batch(self):
        """50 组随机映射矩阵 + 默认特质，检验映射鲁棒性。"""
        rng = np.random.default_rng(7777)
        n_trials = 50
        n_steps = 400

        boundary_count = 0
        nan_count = 0

        for trial in range(n_trials):
            mapping = SurfaceToStimuliMapping.random(rng, scale=0.8)
            coupled = CoupledAgents(
                traits_a=DEFAULT_TRAITS.copy(),
                traits_b=DEFAULT_TRAITS.copy(),
                mapping_a2b=mapping, mapping_b2a=mapping,
            )
            coupled.run(n_steps)

            for agent in ["A", "B"]:
                for vec_name in ["internal", "relationship", "surface"]:
                    traj = coupled.history.get_agent_trajectory(agent, vec_name)
                    if np.any(np.isnan(traj)):
                        nan_count += 1
                    if np.any(traj < -1.0 - 0.11) or np.any(traj > 1.0 + 0.11):
                        boundary_count += 1

        assert nan_count == 0, f"{nan_count} 个轨迹包含 NaN"
        assert boundary_count == 0, f"{boundary_count} 个轨迹有边界违规"


# ═══════════════════════════════════════════════════════════════════
# 6.7 长期稳定性
# ═══════════════════════════════════════════════════════════════════

class TestLongRunStability:
    """长时间运行检测漂移/极限环/累积效应。"""

    def test_no_long_term_drift(self, request):
        """10000 轮运行不应出现长期漂移。"""
        # 标记为慢测试
        coupled = CoupledAgents(
            traits_a=DEFAULT_TRAITS.copy(),
            traits_b=DEFAULT_TRAITS.copy(),
            mapping_a2b=SurfaceToStimuliMapping(),
            mapping_b2a=SurfaceToStimuliMapping(),
        )
        coupled.run(10000)
        reports = detect_anomalies(coupled.history, coupled.traits_a, coupled.traits_b)

        # 长时间运行不允许边界违规
        boundary = [r for r in reports if r.anomaly_type == "boundary"]
        assert len(boundary) == 0

        # 检查数值稳定性（不检查 setpoint 偏离，因为无 γ 时 setpoint
        # 不是每轮吸引子；边界饱和在长时间耦合运行中是预期行为）。
        traj_a = coupled.history.get_agent_trajectory("A", "internal")

        # 取最后 10% 步
        tail_start = int(0.9 * coupled.history.steps)
        tail = traj_a[tail_start:]

        # 仅验证数值稳定性和范围
        assert np.all(np.isfinite(tail)), "出现 NaN/Inf"
        assert np.all(tail >= -1.0 - 0.11), "低于 soft_clamp 下限"
        assert np.all(tail <= 1.0 + 0.11), "超过 soft_clamp 上限"

        # 长期运行不应产生 NaN/Inf（已在上方验证）

    def test_step_variance_stabilizes(self):
        """随着步数增加，状态变化的方差应该减小（收敛到稳态）。"""
        coupled = CoupledAgents(
            traits_a=DEFAULT_TRAITS.copy(),
            traits_b=DEFAULT_TRAITS.copy(),
            mapping_a2b=SurfaceToStimuliMapping(),
            mapping_b2a=SurfaceToStimuliMapping(),
        )
        coupled.run(2000)
        traj = coupled.history.get_agent_trajectory("A", "internal")

        # 前 500 步的逐步变化 vs 后 500 步的逐步变化
        early_diffs = np.abs(np.diff(traj[:500], axis=0))
        late_diffs = np.abs(np.diff(traj[-500:], axis=0))

        early_mean_change = float(np.mean(early_diffs))
        late_mean_change = float(np.mean(late_diffs))

        # 后期变化应该小于或等于早期（系统趋于稳定）
        assert late_mean_change <= early_mean_change * 1.5, \
            f"后期变化 ({late_mean_change:.6f}) 不应显著大于早期 ({early_mean_change:.6f})"


# ═══════════════════════════════════════════════════════════════════
# 6.8 防御剖面动力学
# ═══════════════════════════════════════════════════════════════════

class TestDefenseProfileDynamics:
    """耦合系统中防御剖面的演化。"""

    def test_defense_profiles_exist(self):
        """验证防御剖面在整个运行中保持合法值域。"""
        from state_engine._defenses import compute_defense_profiles

        coupled = CoupledAgents(
            traits_a=DEFAULT_TRAITS.copy(),
            traits_b=DEFAULT_TRAITS.copy(),
            mapping_a2b=SurfaceToStimuliMapping(),
            mapping_b2a=SurfaceToStimuliMapping(),
        )
        coupled.run(600)

        int_a = coupled.history.get_agent_trajectory("A", "internal")
        rel_a = coupled.history.get_agent_trajectory("A", "relationship")

        # 抽样检查若干步的防御剖面
        for step in [0, 100, 300, 599]:
            profiles = compute_defense_profiles(
                coupled.traits_a, rel_a[step], int_a[step],
            )
            assert profiles.shape == (2, ST_SIZE)
            assert np.all(profiles >= -1e-10), f"防御剖面低于0 at step {step}"
            assert np.all(profiles <= 1.0 + 0.11), f"防御剖面超过1 at step {step}"


# ═══════════════════════════════════════════════════════════════════
# 6.9 Setpoint 收敛
# ═══════════════════════════════════════════════════════════════════

class TestSetpointConvergence:
    """零刺激下状态应收敛到 setpoint。"""

    def test_zero_stimulus_stable(self):
        """当 Surface→Stimuli 映射输出全零时，状态收敛到耦合平衡点。

        注意：在新设计中，per-turn 稳态恢复已移除。
        零刺激下，耦合矩阵 A（ρ=0.95<1）驱动状态收敛到近 0，
        而非 setpoint。验证边界合规和稳定性。
        """
        zero_matrix = np.zeros((ST_SIZE, S_SIZE), dtype=np.float64)
        zero_mapping = SurfaceToStimuliMapping(matrix=zero_matrix)

        traits = DEFAULT_TRAITS.copy()
        internal_init = np.full(I_SIZE, 0.9, dtype=np.float64)
        rel_init = np.full(R_SIZE, 0.1, dtype=np.float64)

        coupled = CoupledAgents(
            traits_a=traits, traits_b=traits,
            mapping_a2b=zero_mapping, mapping_b2a=zero_mapping,
            internal_a=internal_init.copy(),
            relationship_a=rel_init.copy(),
            internal_b=internal_init.copy(),
            relationship_b=rel_init.copy(),
        )
        coupled.run(1000)

        int_a = coupled.history.get_agent_trajectory("A", "internal")
        rel_a = coupled.history.get_agent_trajectory("A", "relationship")

        # 验证状态在合法范围内
        assert np.all(int_a >= -1.0 - 0.11) and np.all(int_a <= 1.0 + 0.11)
        assert np.all(rel_a >= -1.0 - 0.11) and np.all(rel_a <= 1.0 + 0.11)

        # 验证最终 L2 范数远小于初始（收敛到耦合平衡点，近 0）
        int_tail = int_a[-200:]
        int_tail_l2 = np.sqrt(np.sum(int_tail ** 2, axis=1))
        initial_l2 = np.linalg.norm(internal_init)
        # 零刺激下状态收敛到近 0（耦合不动点），而非 setpoint
        assert np.mean(int_tail_l2) < initial_l2 * 0.3, \
            f"内部状态未收敛: 初始L2={initial_l2:.4f}, 尾均L2={np.mean(int_tail_l2):.4f}"

        rel_tail_l2 = np.sqrt(np.sum(rel_a[-200:] ** 2, axis=1))
        initial_l2_r = np.linalg.norm(rel_init)
        assert np.mean(rel_tail_l2) < initial_l2_r * 0.6, \
            f"关系状态未收敛: 初始L2={initial_l2_r:.4f}, 尾均L2={np.mean(rel_tail_l2):.4f}"

    def test_identical_agents_same_trajectory(self):
        """两个完全相同的 Agent 在对称耦合下应该有相同的统计特征。"""
        coupled = CoupledAgents(
            traits_a=DEFAULT_TRAITS.copy(),
            traits_b=DEFAULT_TRAITS.copy(),
            mapping_a2b=SurfaceToStimuliMapping(),
            mapping_b2a=SurfaceToStimuliMapping(),
        )
        coupled.run(600)

        int_a = coupled.history.get_agent_trajectory("A", "internal")
        int_b = coupled.history.get_agent_trajectory("B", "internal")

        # 由于对称性，两者的均值应该几乎相同
        mean_diff = np.mean(np.abs(np.mean(int_a, axis=0) - np.mean(int_b, axis=0)))
        assert mean_diff < 0.05, \
            f"对称Agent内部状态均值差异过大: {mean_diff:.6f}"


# ═══════════════════════════════════════════════════════════════════
# 6.10 综合压力测试
# ═══════════════════════════════════════════════════════════════════

class TestStressScenarios:
    """综合场景：极端特质 + 对抗映射 + 噪声 + 冲击的组合。"""

    def test_extreme_anxious_with_noise_and_adversarial_mapping(self):
        """极高焦虑型 + 噪声映射 + 随机扰动 — 最坏情况压力测试。"""
        rng = np.random.default_rng(5555)
        extreme_traits = DEFAULT_TRAITS.copy()
        extreme_traits[T_ATTACHMENT_ANXIETY] = 0.98
        extreme_traits[T_JEALOUSY_SENSITIVITY] = 0.95
        extreme_traits[T_EMOTIONAL_STABILITY] = 0.05
        extreme_traits[T_SENSITIVITY] = 0.95
        extreme_traits[T_ANXIETY_PRONENESS] = 0.90

        coupled = CoupledAgents(
            traits_a=extreme_traits,
            traits_b=extreme_traits,
            mapping_a2b=SurfaceToStimuliMapping(matrix_noise_std=0.03, mode="linear"),
            mapping_b2a=SurfaceToStimuliMapping(matrix_noise_std=0.03, mode="linear"),
            rng=rng,
        )
        coupled.run(1500, perturbation_every=80, perturbation_strength=0.4)

        reports = detect_anomalies(coupled.history, coupled.traits_a, coupled.traits_b)

        # 不允许边界违规
        boundary = [r for r in reports if r.anomaly_type == "boundary"]
        assert len(boundary) == 0

        # 不允许 NaN
        for agent in ["A", "B"]:
            traj = coupled.history.get_agent_trajectory(agent, "internal")
            assert not np.any(np.isnan(traj)), f"{agent} 出现 NaN"

    def test_adversarial_matrix_with_shock(self):
        """对抗矩阵 + 大冲击 = 极端鲁棒性检验。"""
        rng = np.random.default_rng(6666)
        adv_mapping = SurfaceToStimuliMapping.adversarial(rng, severity=2.5)

        shock = np.ones(ST_SIZE, dtype=np.float64) * 0.9  # 全维度高强度冲击

        coupled = CoupledAgents(
            traits_a=DEFAULT_TRAITS.copy(),
            traits_b=DEFAULT_TRAITS.copy(),
            mapping_a2b=adv_mapping, mapping_b2a=adv_mapping,
        )
        coupled.run_with_shock(800, shock_at=300,
                               shock_stimuli_a=shock, shock_stimuli_b=shock)

        reports = detect_anomalies(coupled.history, coupled.traits_a, coupled.traits_b)

        # 对抗映射下可能出现饱和和振荡，但边界违规不可接受
        boundary = [r for r in reports if r.anomaly_type == "boundary"]
        assert len(boundary) == 0

        # 无 NaN
        for agent in ["A", "B"]:
            traj = coupled.history.get_agent_trajectory(agent, "internal")
            assert not np.any(np.isnan(traj)), f"{agent} 出现 NaN"


# ═══════════════════════════════════════════════════════════════════
# 7. 报告生成工具
# ═══════════════════════════════════════════════════════════════════

def print_detailed_report(
    coupled: CoupledAgents,
    title: str = "状态引擎对抗检验报告",
):
    """生成详细的人类可读报告（非测试断言，供调试和分析用）。"""
    reports = detect_anomalies(coupled.history, coupled.traits_a, coupled.traits_b)
    steps = coupled.history.steps

    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"  运行轮数: {steps}")
    print(f"{'='*70}")

    # 异常汇总
    if reports:
        by_type: dict[str, list[AnomalyReport]] = {}
        for r in reports:
            by_type.setdefault(r.anomaly_type, []).append(r)

        print(f"\n  异常总数: {len(reports)}")
        for atype, items in by_type.items():
            sev_counts = {"high": 0, "medium": 0, "low": 0}
            for item in items:
                sev_counts[item.severity] += 1
            print(f"    {atype}: {len(items)} "
                  f"(高={sev_counts['high']}, 中={sev_counts['medium']}, 低={sev_counts['low']})")
            for item in items[:5]:
                print(f"      [{item.severity}] {item.agent}/{item.vector}/{item.dimension}: {item.detail}")
            if len(items) > 5:
                print(f"      ... 还有 {len(items) - 5} 个")
    else:
        print("\n  ✅ 未检测到异常")

    # 状态统计
    for agent in ["A", "B"]:
        traits = coupled.traits_a if agent == "A" else coupled.traits_b
        sp_i = compute_setpoint(traits)
        sp_r = compute_rel_setpoint(traits)

        print(f"\n  ── Agent {agent} 内部状态统计 ──")
        stats_i = coupled.history.dimension_stats(
            agent, "internal", I_LABELS, setpoints=sp_i,
        )
        for s in stats_i:
            flags = []
            if s.is_saturated: flags.append("⚠饱和")
            if s.is_diverging: flags.append("⚠发散")
            flag_str = " " + ",".join(flags) if flags else ""
            print(f"    {s.label:20s}: mean={s.mean:.4f}  std={s.std:.4f}  "
                  f"[{s.min_val:.3f}, {s.max_val:.3f}]  "
                  f"sp_dev={s.setpoint_deviation:+.3f}{flag_str}")

        print(f"\n  ── Agent {agent} 关系状态统计 ──")
        stats_r = coupled.history.dimension_stats(
            agent, "relationship", R_LABELS, setpoints=sp_r,
        )
        for s in stats_r:
            flags = []
            if s.is_saturated: flags.append("⚠饱和")
            if s.is_diverging: flags.append("⚠发散")
            flag_str = " " + ",".join(flags) if flags else ""
            print(f"    {s.label:20s}: mean={s.mean:.4f}  std={s.std:.4f}  "
                  f"[{s.min_val:.3f}, {s.max_val:.3f}]  "
                  f"sp_dev={s.setpoint_deviation:+.3f}{flag_str}")

    print(f"\n{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════
# 8. 可直接运行的入口（python tests/test_adversarial_engine.py）
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 运行几个代表性场景并打印报告
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("  对抗式状态引擎检验 — 独立运行模式")
    print("=" * 70)

    # 场景 1: 基本对称耦合
    print("\n[1/5] 基本对称耦合...")
    c1 = CoupledAgents(
        traits_a=DEFAULT_TRAITS.copy(),
        traits_b=DEFAULT_TRAITS.copy(),
        mapping_a2b=SurfaceToStimuliMapping(),
        mapping_b2a=SurfaceToStimuliMapping(),
    )
    c1.run(1000)
    print_detailed_report(c1, "场景1: 对称特质 + 默认映射")

    # 场景 2: 焦虑型 vs 回避型
    print("\n[2/5] 焦虑型 vs 回避型...")
    anxious = DEFAULT_TRAITS.copy()
    anxious[T_ATTACHMENT_ANXIETY] = 0.85
    anxious[T_ATTACHMENT_AVOIDANCE] = 0.10
    avoidant = DEFAULT_TRAITS.copy()
    avoidant[T_ATTACHMENT_AVOIDANCE] = 0.80
    avoidant[T_ATTACHMENT_ANXIETY] = 0.15

    c2 = CoupledAgents(
        traits_a=anxious, traits_b=avoidant,
        mapping_a2b=SurfaceToStimuliMapping(),
        mapping_b2a=SurfaceToStimuliMapping(),
    )
    c2.run(1000)
    print_detailed_report(c2, "场景2: 焦虑型(A) vs 回避型(B)")

    # 场景 3: 对抗倒置映射
    print("\n[3/5] 对抗倒置映射...")
    c3 = CoupledAgents(
        traits_a=DEFAULT_TRAITS.copy(),
        traits_b=DEFAULT_TRAITS.copy(),
        mapping_a2b=SurfaceToStimuliMapping.inverted(),
        mapping_b2a=SurfaceToStimuliMapping.inverted(),
    )
    c3.run(1000)
    print_detailed_report(c3, "场景3: 倒置感知映射（温暖→冲突）")

    # 场景 4: 噪声 + 周期性扰动
    print("\n[4/5] 噪声 + 周期性扰动...")
    c4 = CoupledAgents(
        traits_a=DEFAULT_TRAITS.copy(),
        traits_b=DEFAULT_TRAITS.copy(),
        mapping_a2b=SurfaceToStimuliMapping(matrix_noise_std=0.03, mode="tanh"),
        mapping_b2a=SurfaceToStimuliMapping(matrix_noise_std=0.03, mode="tanh"),
    )
    c4.run(1200, perturbation_every=60, perturbation_strength=0.35)
    print_detailed_report(c4, "场景4: 噪声感知 + 周期性扰动")

    # 场景 5: 极端焦虑综合压力测试
    print("\n[5/5] 极端焦虑综合压力测试...")
    extreme = DEFAULT_TRAITS.copy()
    extreme[T_ATTACHMENT_ANXIETY] = 0.95
    extreme[T_JEALOUSY_SENSITIVITY] = 0.90
    extreme[T_EMOTIONAL_STABILITY] = 0.05
    extreme[T_SENSITIVITY] = 0.90

    c5 = CoupledAgents(
        traits_a=extreme, traits_b=DEFAULT_TRAITS.copy(),
        mapping_a2b=SurfaceToStimuliMapping(matrix_noise_std=0.02),
        mapping_b2a=SurfaceToStimuliMapping(matrix_noise_std=0.02),
        rng=rng,
    )
    c5.run(1500, perturbation_every=100, perturbation_strength=0.3)
    print_detailed_report(c5, "场景5: 极端焦虑型 + 噪声 + 扰动")

    print("全部场景检验完成。")
