"""约束框架中央注册表 —— WeightMapper + ConstraintRegistry + WeightVector。

约束⑤语义映射层 + 约束⑧参数审计 的核心实现。
所有权重/矩阵的创建必须通过 WeightMapper/WeightVector，禁止裸数字 `M[i,j] = value`。

功能:
  - WeightMapper     : 2D 矩阵的语义映射（B_int, B_rel, 耦合矩阵）
  - WeightVector     : 1D 权重向量的语义映射（防御剖面权重、衰减率等）
  - ConstraintRegistry: 中央约束注册与审计
  - JSON 导入/导出   : 权重外部化（P2 路线图目标）

用法:
    from ._validator import WeightMapper, WeightVector, ConstraintRegistry

    # 2D 矩阵
    mapper = WeightMapper("INPUT_INFLUENCE_B", ST_LABELS, I_LABELS)
    mapper.connect(..., value=0.28, ...)
    B = mapper.build_matrix((ST_SIZE, I_SIZE))
    mapper.to_json("params/b_int.json")

    # 1D 权重向量
    vec = WeightVector("STRESS_DEACT_A", ST_LABELS)
    vec.connect(..., value=0.12, ...)
    # WeightVector 不 build_matrix，直接 .values 取 numpy 数组

    # 审计
    print(ConstraintRegistry.audit_report())
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
from typing import Optional, Any

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# 数据类型定义
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SemanticWeight:
    """语义权重声明 —— 每个矩阵条目的完整 provenance。

    创建时自动验证:
      - direction（方向）与 value（符号）一致性
      - value ∈ domain（值在允许区间内）
      - domain 不跨零（方向确定时）
      - magnitude（强度等级）的建议范围匹配
    """
    source_idx: int            # 源概念索引（如 ST_ABANDONMENT）
    target_idx: int            # 目标概念索引（如 I_INSECURITY）
    value: float               # 当前数值（必须 ∈ domain）
    direction: str             # 影响方向: "+"(正) / "-"(负) / "0"(零)
    magnitude: str             # 影响强度: "strong" / "moderate" / "weak" / "trace"
    domain: tuple              # 允许区间 (low, high)，如 (0.15, 0.35)
    rationale: str             # 心理依据（一句话）
    origin: str                # 来源: "theory"(文献) / "calibrated"(测试) / "legacy"(未审)
    reviewed: str              # 最后审查日期 "YYYY-MM-DD"

    def __post_init__(self):
        """创建时自动验证方向-符号一致性和 domain 约束。"""
        # 方向与符号一致性
        if self.value > 0 and self.direction not in ("+", "0"):
            raise ValueError(
                f"value={self.value} > 0 但 direction='{self.direction}'，应为 '+'"
            )
        if self.value < 0 and self.direction not in ("-", "0"):
            raise ValueError(
                f"value={self.value} < 0 但 direction='{self.direction}'，应为 '-'"
            )
        if self.value == 0 and self.direction != "0":
            raise ValueError(
                f"value=0 但 direction='{self.direction}'，应为 '0'"
            )

        # domain 合理性：不跨零（方向确定时）
        if self.direction == "+" and self.domain[1] <= 0:
            raise ValueError(f"direction='+' 但 domain={self.domain} 不包含正值")
        if self.direction == "-" and self.domain[0] >= 0:
            raise ValueError(f"direction='-' 但 domain={self.domain} 不包含负值")

        # value ∈ domain
        low, high = self.domain
        if not (low <= self.value <= high):
            raise ValueError(
                f"value={self.value} 不在 domain={self.domain} 范围内"
            )

        # magnitude 与 |value| 的指导性映射（不强制，仅 warning）
        mag_ranges = {
            "strong":   (0.20, 1.00),
            "moderate": (0.10, 0.30),
            "weak":     (0.04, 0.15),
            "trace":    (0.01, 0.06),
        }
        if self.magnitude in mag_ranges:
            lo, hi = mag_ranges[self.magnitude]
            if not (lo <= abs(self.value) <= hi):
                import warnings
                warnings.warn(
                    f"magnitude='{self.magnitude}' 但 |value|={abs(self.value):.2f} "
                    f"不在建议范围 {lo}–{hi}"
                )

    def to_dict(self) -> dict:
        """序列化为 JSON 兼容 dict。"""
        d = asdict(self)
        d["domain"] = list(d["domain"])  # tuple → list for JSON
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticWeight":
        """从 dict 反序列化。"""
        d = dict(d)
        d["domain"] = tuple(d["domain"])  # list → tuple
        return cls(**d)


# ═══════════════════════════════════════════════════════════════════
# 约束违反异常
# ═══════════════════════════════════════════════════════════════════

class ConstraintViolationError(Exception):
    """约束违反异常 —— 矩阵/参数不符合约束要求时抛出。"""
    pass


# ═══════════════════════════════════════════════════════════════════
# 约束验证函数（独立，可被 ConstraintRegistry 注册）
# ═══════════════════════════════════════════════════════════════════

def assert_sparsity(M: np.ndarray, label: str, max_density: float = 0.30) -> None:
    """⑥-1 稀疏度检查：非零元素密度 ≤ max_density。

    小矩阵例外：min(dim) ≤ 3 时上限放宽至 70%。
    """
    total = M.shape[0] * M.shape[1]
    nnz = np.count_nonzero(M)
    density = nnz / total
    max_allow = 0.70 if min(M.shape) <= 3 else max_density
    if density > max_allow:
        raise ConstraintViolationError(
            f"{label}: density={density:.1%} > {max_allow:.0%}"
        )


def assert_orthogonality(M: np.ndarray, label: str, threshold: float = 0.3) -> None:
    """⑥-2 正交性检查：归一化行 Gram 矩阵非对角元 < threshold。"""
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    M_norm = M / norms
    G = M_norm @ M_norm.T
    off_diag = np.abs(G - np.eye(G.shape[0]))
    max_corr = np.max(off_diag)
    if max_corr > threshold:
        pairs = np.argwhere(off_diag > threshold)
        raise ConstraintViolationError(
            f"{label}: max off-diag corr={max_corr:.3f} > {threshold}, "
            f"collinear pairs: {pairs[:5].tolist()}"
        )


def assert_matrix_rank(
    M: np.ndarray, label: str,
    expected_max_rank: Optional[int] = None,
) -> int:
    """③ 低秩检查：有效秩 ≤ expected_max_rank。

    Returns:
        effective_rank (int): 覆盖 99% 奇异值所需的最小奇异值数量
    """
    u, s, vt = np.linalg.svd(M, full_matrices=False)
    cumulative = np.cumsum(s) / np.sum(s)
    effective_rank = int(np.searchsorted(cumulative, 0.99) + 1)
    if expected_max_rank is not None:
        if effective_rank > expected_max_rank:
            raise ConstraintViolationError(
                f"{label}: effective_rank={effective_rank} > expected_max={expected_max_rank}"
            )
    return effective_rank


def assert_spectral_radius(
    M: np.ndarray, label: str, max_radius: float = 0.95,
) -> None:
    """⑦ 谱半径检查：ρ < max_radius（仅方阵）。"""
    if M.shape[0] != M.shape[1]:
        return  # 非方阵跳过
    eigenvalues = np.linalg.eigvals(M)
    spectral_radius = np.max(np.abs(eigenvalues))
    if spectral_radius >= max_radius:
        unstable = eigenvalues[np.abs(eigenvalues) > 0.9]
        raise ConstraintViolationError(
            f"{label}: spectral_radius={spectral_radius:.4f} >= {max_radius}, "
            f"dominant modes: {unstable[:5].tolist()}"
        )


# ═══════════════════════════════════════════════════════════════════
# WeightMapper — 2D 矩阵映射器
# ═══════════════════════════════════════════════════════════════════

class WeightMapper:
    """语义映射器 —— 管理 2D 矩阵（B_int, B_rel, 耦合矩阵）的 SemanticWeight 声明。

    职责:
      1. 收集 SemanticWeight 声明（connect）
      2. 建立矩阵并执行约束检查（build_matrix）
      3. JSON 导入/导出（to_json / from_json）
      4. 反向查询（lookup）
      5. 生成 provenance 审计报告（audit）

    用法:
        mapper = WeightMapper("INPUT_INFLUENCE_B", ST_LABELS, I_LABELS)
        mapper.connect(...)
        B = mapper.build_matrix((ST_SIZE, I_SIZE))
        mapper.to_json("params/b_int.json")
    """

    def __init__(
        self,
        label: str,
        source_labels: list[str],
        target_labels: list[str],
        description: str = "",
    ):
        self.label = label
        self._source_labels = list(source_labels)
        self._target_labels = list(target_labels)
        self.description = description
        self._weights: list[SemanticWeight] = []
        self._built_shape: Optional[tuple[int, int]] = None

    # ── 属性 ──

    @property
    def source_labels(self) -> list[str]:
        return self._source_labels

    @property
    def target_labels(self) -> list[str]:
        return self._target_labels

    @property
    def weights(self) -> list[SemanticWeight]:
        """返回已注册权重的只读副本。"""
        return list(self._weights)

    # ── 注册 ──

    def connect(
        self,
        source_idx: int,
        target_idx: int,
        value: float,
        magnitude: str,
        domain: tuple,
        rationale: str,
        origin: str,
        reviewed: str = "2026-06-21",
    ) -> "WeightMapper":
        """注册一条语义权重声明。

        direction 从 value 的符号自动推导。
        创建 SemanticWeight 时会自动验证 domain 和符号一致性。
        """
        direction = "+" if value > 0 else ("-" if value < 0 else "0")
        sw = SemanticWeight(
            source_idx=source_idx,
            target_idx=target_idx,
            value=value,
            direction=direction,
            magnitude=magnitude,
            domain=domain,
            rationale=rationale,
            origin=origin,
            reviewed=reviewed,
        )
        self._weights.append(sw)
        return self  # 链式调用支持

    def load_from_weights(self, weights: list[SemanticWeight]) -> "WeightMapper":
        """从已有的 SemanticWeight 列表批量加载（用于 JSON 反序列化）。"""
        self._weights = list(weights)
        return self

    # ── 矩阵构建 ──

    def build_matrix(
        self,
        shape: tuple[int, int],
        expected_max_rank: Optional[int] = None,
        skip_sparsity: bool = False,
        skip_orthogonality: bool = False,
        skip_rank: bool = False,
        skip_spectral: bool = False,
    ) -> np.ndarray:
        """从注册的 SemanticWeight 建立矩阵并执行约束检查。

        Args:
            shape: (rows, cols)
            expected_max_rank: 约束③的预期最大秩（仅方阵/耦合矩阵）
            skip_*: 跳过特定约束检查

        Returns:
            np.ndarray — 填充好的矩阵
        """
        if not self._weights:
            raise ValueError(f"{self.label}: 未注册任何 SemanticWeight")

        M = np.zeros(shape, dtype=np.float64)
        for sw in self._weights:
            M[sw.source_idx, sw.target_idx] = sw.value

        self._built_shape = shape

        # 向中央注册表注册所有约束检查
        label = self.label
        ConstraintRegistry.register(label, self._describe_weights)

        if not skip_sparsity:
            ConstraintRegistry.register(label, assert_sparsity, M, label)
        if not skip_orthogonality:
            ConstraintRegistry.register(label, assert_orthogonality, M, label)
        if not skip_rank:
            ConstraintRegistry.register(
                label, lambda: assert_matrix_rank(M, label, expected_max_rank)
            )
        if not skip_spectral:
            ConstraintRegistry.register(label, assert_spectral_radius, M, label)

        ConstraintRegistry.run_all(label)

        # 自动注册到全局导出表
        _exported_mappers[label] = self

        return M

    # ── 反向查询 ──

    def lookup(
        self,
        source_name: Optional[str] = None,
        target_name: Optional[str] = None,
    ) -> list[SemanticWeight]:
        """按概念名反向查询权重条目。

        Args:
            source_name: 源概念名（如 "abandonment_stimulus"），None 表示不筛选
            target_name: 目标概念名（如 "insecurity"），None 表示不筛选

        Returns:
            匹配的 SemanticWeight 列表
        """
        results = list(self._weights)
        if source_name is not None:
            results = [
                w for w in results
                if self._source_labels[w.source_idx] == source_name
            ]
        if target_name is not None:
            results = [
                w for w in results
                if self._target_labels[w.target_idx] == target_name
            ]
        return results

    def lookup_by_idx(
        self,
        source_idx: Optional[int] = None,
        target_idx: Optional[int] = None,
    ) -> list[SemanticWeight]:
        """按索引反向查询。"""
        results = list(self._weights)
        if source_idx is not None:
            results = [w for w in results if w.source_idx == source_idx]
        if target_idx is not None:
            results = [w for w in results if w.target_idx == target_idx]
        return results

    # ── JSON 序列化 ──

    def to_dict(self) -> dict:
        """导出为 JSON 兼容 dict。"""
        return {
            "mapper_type": "matrix",
            "label": self.label,
            "description": self.description,
            "sources": self._source_labels,
            "targets": self._target_labels,
            "entries": [{
                "source": self._source_labels[sw.source_idx],
                "target": self._target_labels[sw.target_idx],
                "value": sw.value,
                "direction": sw.direction,
                "magnitude": sw.magnitude,
                "domain": list(sw.domain),
                "rationale": sw.rationale,
                "origin": sw.origin,
                "reviewed": sw.reviewed,
            } for sw in self._weights],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WeightMapper":
        """从 dict 重建 WeightMapper（兼容 JSON 导出格式）。"""
        mapper = cls(
            label=d["label"],
            source_labels=d["sources"],
            target_labels=d["targets"],
            description=d.get("description", ""),
        )
        src_list = d["sources"]
        tgt_list = d["targets"]
        for entry in d["entries"]:
            # 兼容两种格式：source/source_idx, target/target_idx
            src_name = entry.get("source", "")
            if not src_name:
                src_idx = entry["source_idx"]
            else:
                src_idx = src_list.index(src_name)

            tgt_name = entry.get("target", "")
            if not tgt_name:
                tgt_idx = entry["target_idx"]
            else:
                tgt_idx = tgt_list.index(tgt_name)

            mapper._weights.append(SemanticWeight(
                source_idx=src_idx,
                target_idx=tgt_idx,
                value=entry["value"],
                direction=entry.get("direction", "+" if entry["value"] > 0 else "-"),
                magnitude=entry["magnitude"],
                domain=tuple(entry["domain"]),
                rationale=entry["rationale"],
                origin=entry["origin"],
                reviewed=entry.get("reviewed", "2026-06-21"),
            ))
        return mapper

    def to_json(self, filepath: str, indent: int = 2) -> str:
        """导出为 JSON 文件。"""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=indent)
        return filepath

    @classmethod
    def from_json(cls, filepath: str) -> "WeightMapper":
        """从 JSON 文件加载。"""
        with open(filepath, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

    # ── 审计 ──

    def _describe_weights(self) -> dict:
        """structured provenance（供 ConstraintRegistry 使用）。"""
        rows, cols = self._built_shape or (len(self._source_labels), len(self._target_labels))
        return {
            "mapper_type": "matrix",
            "label": self.label,
            "shape": f"{rows}×{cols}",
            "density": f"{len(self._weights)}/{rows * cols}",
            "origin_counts": dict(_count_origin(self._weights)),
            "entries": [
                {
                    "source": self._source_labels[sw.source_idx],
                    "target": self._target_labels[sw.target_idx],
                    "value": sw.value,
                    "direction": sw.direction,
                    "magnitude": sw.magnitude,
                    "origin": sw.origin,
                    "rationale": sw.rationale,
                    "reviewed": sw.reviewed,
                }
                for sw in self._weights
            ],
        }

    def audit(self) -> str:
        """生成可读的 provenance 审计文本。"""
        lines = [f"## {self.label} — 参数审计", ""]
        counts = defaultdict(int)
        for sw in self._weights:
            counts[sw.origin] += 1
            src = self._source_labels[sw.source_idx]
            tgt = self._target_labels[sw.target_idx]
            op = "×↓" if sw.magnitude == "multiplicative" else ""
            lines.append(
                f"  `{src} → {tgt}` = {sw.value:+.3f}  "
                f"[{sw.magnitude}, {sw.origin}]  {sw.rationale}"
            )
        lines.append("")
        lines.append(f"  来源分布: {dict(counts)}")
        if "legacy" in counts:
            lines.append("  ⚠️ 存在 legacy 参数，需审查")
        if counts.get("calibrated", 0) / len(self._weights) > 0.5:
            lines.append("  ⚠️ calibrated 参数过半，建议增加 theory 依据")
        lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# WeightVector — 1D 权重向量映射器
# ═══════════════════════════════════════════════════════════════════

class WeightVector:
    """语义权重向量 —— 管理 1D 权重向量（防御剖面权重、衰减率等）的 provenance。

    WeightVector 用于非矩阵形式的逐维权重（如 STABILITY_DEACT_A 的 7 维数组）。
    每条条目映射: source_variable → target_stimulus_dimension。

    用法:
        vec = WeightVector("STABILITY_DEACT_A", ST_LABELS)
        vec.connect("stability", ST_ABANDONMENT, -0.10, "weak", ...)
        array = vec.build()  # → np.ndarray(7,)
    """

    def __init__(
        self,
        label: str,
        target_labels: list[str],
        description: str = "",
    ):
        self.label = label
        self._target_labels = list(target_labels)
        self.description = description
        self._entries: list[dict] = []  # 轻量 dict，不用 SemanticWeight（source 是 string）

    def connect(
        self,
        source_name: str,
        target_idx: int,
        value: float,
        magnitude: str,
        domain: tuple,
        rationale: str,
        origin: str,
        reviewed: str = "2026-06-21",
    ) -> "WeightVector":
        """注册一条权重向量条目。

        Args:
            source_name: 源变量名（如 "stability", "stress", "trust_bond"）
            target_idx: 目标维度索引（如 ST_ABANDONMENT）
            value: 权重值
            magnitude: 强度等级
            domain: 允许值区间 (low, high)
            rationale: 心理依据
            origin: 来源
            reviewed: 审查日期
        """
        direction = "+" if value > 0 else ("-" if value < 0 else "0")

        # 轻量验证（与 SemanticWeight 一致）
        if value > 0 and direction not in ("+", "0"):
            raise ValueError(f"value={value}>0 但 direction='{direction}'")
        if value < 0 and direction not in ("-", "0"):
            raise ValueError(f"value={value}<0 但 direction='{direction}'")
        low, high = domain
        if not (low <= value <= high):
            raise ValueError(f"value={value} 不在 domain={domain} 内")

        self._entries.append({
            "source_name": source_name,
            "target_idx": target_idx,
            "value": value,
            "direction": direction,
            "magnitude": magnitude,
            "domain": list(domain),
            "rationale": rationale,
            "origin": origin,
            "reviewed": reviewed,
        })
        return self

    @property
    def target_labels(self) -> list[str]:
        return list(self._target_labels)

    @property
    def entries(self) -> list[dict]:
        return list(self._entries)

    @property
    def values(self) -> np.ndarray:
        """取构建后的向量值（必须先调 build()）。"""
        if not hasattr(self, '_built_values'):
            raise RuntimeError(f"{self.label}: 必须先调用 build()")
        return self._built_values

    # ── 向量构建 ──

    def build(self) -> np.ndarray:
        """构建 1D 权重向量并注册 provenance。"""
        arr = np.zeros(len(self._target_labels), dtype=np.float64)
        for e in self._entries:
            arr[e["target_idx"]] = e["value"]
        self._built_values = arr.copy()
        ConstraintRegistry.register(self.label, self._describe)
        ConstraintRegistry.run_all(self.label)

        # 自动注册到全局导出表
        _exported_mappers[self.label] = self

        return arr

    # ── JSON ──

    def to_dict(self) -> dict:
        return {
            "mapper_type": "vector",
            "label": self.label,
            "description": self.description,
            "targets": self._target_labels,
            "entries": self._entries,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WeightVector":
        vec = cls(
            label=d["label"],
            target_labels=d["targets"],
            description=d.get("description", ""),
        )
        vec._entries = d["entries"]
        return vec

    def to_json(self, filepath: str, indent: int = 2) -> str:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=indent)
        return filepath

    @classmethod
    def from_json(cls, filepath: str) -> "WeightVector":
        with open(filepath, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

    # ── 审计 ──

    def _describe(self) -> dict:
        return {
            "mapper_type": "vector",
            "label": self.label,
            "shape": f"({len(self._target_labels)},)",
            "origin_counts": dict(_count_origin(self._entries)),
            "entries": self._entries,
        }

    def audit(self) -> str:
        lines = [f"## {self.label} — 参数审计", ""]
        counts = defaultdict(int)
        for e in self._entries:
            counts[e["origin"]] += 1
            tgt = self._target_labels[e["target_idx"]]
            lines.append(
                f"  `{e['source_name']} → {tgt}` = {e['value']:+.3f}  "
                f"[{e['magnitude']}, {e['origin']}]  {e['rationale']}"
            )
        lines.append("")
        lines.append(f"  来源分布: {dict(counts)}")
        if "legacy" in counts:
            lines.append("  ⚠️ 存在 legacy 参数")
        lines.append("")
        return "\n".join(lines)

    def lookup(self, source_name: Optional[str] = None) -> list[dict]:
        if source_name is None:
            return list(self._entries)
        return [e for e in self._entries if e["source_name"] == source_name]


# ═══════════════════════════════════════════════════════════════════
# BiasWeight — 偏置项 provenance
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BiasWeight:
    """偏置项 provenance —— 线性映射 y = Wx + b 中的 b。

    每条偏置对应一个输出维度，携带完整 provenance。
    """
    target_idx: int            # 目标维度索引（如 S_WARMTH）
    value: float               # 偏置值
    domain: tuple              # 允许区间 (low, high)
    rationale: str             # 心理依据（一句话）
    origin: str                # 来源
    reviewed: str              # 审查日期

    def __post_init__(self):
        low, high = self.domain
        if not (low <= self.value <= high):
            raise ValueError(
                f"Bias value={self.value} 不在 domain={self.domain} 范围内"
            )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["domain"] = list(d["domain"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BiasWeight":
        d = dict(d)
        d["domain"] = tuple(d["domain"])
        return cls(**d)


# ═══════════════════════════════════════════════════════════════════
# LinearMapping — 线性映射 y = Wx + b（surface projection 等）
# ═══════════════════════════════════════════════════════════════════

class LinearMapping:
    """线性映射器 —— y = W @ x + b，带完整 provenance + 源组分段。

    WeightMapper 处理的是纯矩阵（stimulus→state），
    LinearMapping 处理的是多层拼接的源向量（internal+relationship+outer_stimuli+traits→surface）。
    支持源组分段（add_source_group）以便按组索引连接。

    用法:
        lm = LinearMapping("SURFACE_PROJECTION", S_LABELS, "表面表达投影")
        lm.add_source_group("internal", I_LABELS)
        lm.add_source_group("relationship", R_LABELS)
        lm.add_source_group("outer_stimuli", ST_LABELS)
        lm.add_source_group("traits", ["pride", "openness", "optimism"])

        lm.set_bias(S_EXPRESSIVENESS, -0.3, (-0.5, 0.0),
                     "默认表达基线", "calibrated", "2026-06-21")
        lm.connect("internal", "energy", S_EXPRESSIVENESS, 0.4,
                    "moderate", (0.25, 0.55), "精力→外露", "theory", "2026-06-21")

        W, b = lm.build()
        s = lm.compute(np.concatenate([internal, relationship, ...]))
    """

    def __init__(
        self,
        label: str,
        target_labels: list[str],
        description: str = "",
    ):
        self.label = label
        self._target_labels = list(target_labels)
        self.description = description
        self._source_labels: list[str] = []
        self._group_offsets: dict[str, int] = {}   # group_name → offset in source_labels
        self._group_sizes: dict[str, int] = {}      # group_name → dim count
        self._weights: list[SemanticWeight] = []
        self._biases: list[BiasWeight] = []
        self._weight_matrix: Optional[np.ndarray] = None
        self._bias_vector: Optional[np.ndarray] = None

    # ── 属性 ──

    @property
    def source_labels(self) -> list[str]:
        return list(self._source_labels)

    @property
    def target_labels(self) -> list[str]:
        return list(self._target_labels)

    @property
    def weights(self) -> list[SemanticWeight]:
        return list(self._weights)

    @property
    def biases(self) -> list[BiasWeight]:
        return list(self._biases)

    # ── 源组分段 ──

    def add_source_group(
        self,
        group_name: str,
        labels: list[str],
        offset: Optional[int] = None,
    ) -> "LinearMapping":
        """注册一组来源维度（如 internal 的 8 维）。

        Args:
            group_name: 组名（如 "internal"）
            labels: 该组的维度标签列表
            offset: 强制偏移（不指定则追加到末尾）
        """
        if offset is None:
            offset = len(self._source_labels)
        self._group_offsets[group_name] = offset
        self._group_sizes[group_name] = len(labels)

        # 扩展 source_labels 列表
        while len(self._source_labels) < offset + len(labels):
            self._source_labels.append("")
        for i, label in enumerate(labels):
            self._source_labels[offset + i] = label

        return self

    def source_idx(self, group_name: str, label: str) -> int:
        """获取 group 内指定标签的绝对索引。"""
        offset = self._group_offsets.get(group_name)
        if offset is None:
            raise KeyError(f"未知源组: {group_name}")
        group_labels = self._source_labels[offset:offset + self._group_sizes[group_name]]
        try:
            idx = group_labels.index(label)
        except ValueError:
            raise KeyError(f"组 {group_name} 中找不到标签 {label}")
        return offset + idx

    # ── 注册权重 ──

    def connect(
        self,
        source_group: str,
        source_label: str,
        target_idx: int,
        value: float,
        magnitude: str,
        domain: tuple,
        rationale: str,
        origin: str,
        reviewed: str = "2026-06-21",
    ) -> "LinearMapping":
        """注册一条 source→target 权重连接。"""
        src_idx = self.source_idx(source_group, source_label)
        direction = "+" if value > 0 else ("-" if value < 0 else "0")
        sw = SemanticWeight(
            source_idx=src_idx,
            target_idx=target_idx,
            value=value,
            direction=direction,
            magnitude=magnitude,
            domain=domain,
            rationale=rationale,
            origin=origin,
            reviewed=reviewed,
        )
        self._weights.append(sw)
        return self

    # ── 注册偏置 ──

    def set_bias(
        self,
        target_idx: int,
        value: float,
        domain: tuple,
        rationale: str,
        origin: str,
        reviewed: str = "2026-06-21",
    ) -> "LinearMapping":
        """设置一个输出维度的偏置项。"""
        bw = BiasWeight(
            target_idx=target_idx,
            value=value,
            domain=domain,
            rationale=rationale,
            origin=origin,
            reviewed=reviewed,
        )
        self._biases.append(bw)
        return self

    # ── 构建 ──

    def build(
        self,
        skip_sparsity: bool = False,
        skip_orthogonality: bool = False,
        skip_rank: bool = False,
        skip_spectral: bool = True,  # 非方阵默认跳过
    ) -> tuple[np.ndarray, np.ndarray]:
        """构建权重矩阵 (n_targets, n_sources) 和偏置向量 (n_targets,)。

        Returns:
            (W, b) — W: (n_targets, n_sources), b: (n_targets,)
        """
        n_src = len(self._source_labels)
        n_tgt = len(self._target_labels)

        if n_src == 0 or n_tgt == 0:
            raise ValueError(f"{self.label}: source 或 target 为空")

        W = np.zeros((n_tgt, n_src), dtype=np.float64)
        for sw in self._weights:
            W[sw.target_idx, sw.source_idx] = sw.value

        b = np.zeros(n_tgt, dtype=np.float64)
        for bw in self._biases:
            b[bw.target_idx] = bw.value

        self._weight_matrix = W
        self._bias_vector = b

        # 向 ConstraintRegistry 注册 provenance + 约束检查
        label = self.label
        ConstraintRegistry.register(label, self._describe)

        if not skip_sparsity:
            ConstraintRegistry.register(label, assert_sparsity, W, f"{label}_W")
        if not skip_orthogonality:
            ConstraintRegistry.register(label, assert_orthogonality, W, f"{label}_W")

        ConstraintRegistry.run_all(label)
        _exported_mappers[self.label] = self

        return W, b

    def compute(self, sources: np.ndarray) -> np.ndarray:
        """计算 y = W @ x + b。

        Args:
            sources: 源向量 (n_sources,)，各源组按注册顺序拼接

        Returns:
            目标向量 (n_targets,)
        """
        if self._weight_matrix is None or self._bias_vector is None:
            raise RuntimeError(f"{self.label}: 必须先调用 build()")
        return self._weight_matrix @ sources + self._bias_vector

    # ── JSON 序列化 ──

    def to_dict(self) -> dict:
        return {
            "mapper_type": "linear_mapping",
            "label": self.label,
            "description": self.description,
            "sources": self._source_labels,
            "group_offsets": self._group_offsets,
            "group_sizes": self._group_sizes,
            "targets": self._target_labels,
            "weight_entries": [
                {
                    "source": self._source_labels[sw.source_idx],
                    "target": self._target_labels[sw.target_idx],
                    "value": sw.value,
                    "magnitude": sw.magnitude,
                    "domain": list(sw.domain),
                    "rationale": sw.rationale,
                    "origin": sw.origin,
                    "reviewed": sw.reviewed,
                }
                for sw in self._weights
            ],
            "bias_entries": [
                {
                    "target": self._target_labels[bw.target_idx],
                    "value": bw.value,
                    "domain": list(bw.domain),
                    "rationale": bw.rationale,
                    "origin": bw.origin,
                    "reviewed": bw.reviewed,
                }
                for bw in self._biases
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LinearMapping":
        lm = cls(
            label=d["label"],
            target_labels=d["targets"],
            description=d.get("description", ""),
        )
        # 重建源组
        group_offsets = d.get("group_offsets", {})
        group_sizes = d.get("group_sizes", {})
        source_labels = d.get("sources", [])
        lm._source_labels = list(source_labels)
        lm._group_offsets = dict(group_offsets)
        lm._group_sizes = dict(group_sizes)
        # 加载权重
        tgt_list = d["targets"]
        src_list = source_labels
        for entry in d.get("weight_entries", []):
            src_idx = src_list.index(entry["source"])
            tgt_idx = tgt_list.index(entry["target"])
            lm._weights.append(SemanticWeight(
                source_idx=src_idx,
                target_idx=tgt_idx,
                value=entry["value"],
                direction=entry.get("direction", "+" if entry["value"] > 0 else "-"),
                magnitude=entry["magnitude"],
                domain=tuple(entry["domain"]),
                rationale=entry["rationale"],
                origin=entry["origin"],
                reviewed=entry.get("reviewed", "2026-06-21"),
            ))
        # 加载偏置
        for entry in d.get("bias_entries", []):
            tgt_idx = tgt_list.index(entry["target"])
            lm._biases.append(BiasWeight(
                target_idx=tgt_idx,
                value=entry["value"],
                domain=tuple(entry["domain"]),
                rationale=entry["rationale"],
                origin=entry["origin"],
                reviewed=entry.get("reviewed", "2026-06-21"),
            ))
        return lm

    def to_json(self, filepath: str, indent: int = 2) -> str:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=indent)
        return filepath

    @classmethod
    def from_json(cls, filepath: str) -> "LinearMapping":
        with open(filepath, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

    # ── 审计 ──

    def _describe(self) -> dict:
        return {
            "mapper_type": "linear_mapping",
            "label": self.label,
            "shape": f"({len(self._target_labels)}, {len(self._source_labels)})",
            "weight_count": len(self._weights),
            "bias_count": len(self._biases),
            "origin_counts_weights": dict(_count_origin(self._weights)),
            "origin_counts_biases": dict(_count_origin(self._biases)),
        }

    def audit(self) -> str:
        lines = [f"## {self.label} — 线性映射审计", ""]
        w_counts = defaultdict(int)
        b_counts = defaultdict(int)

        lines.append("### 权重连接")
        for sw in self._weights:
            w_counts[sw.origin] += 1
            src = self._source_labels[sw.source_idx]
            tgt = self._target_labels[sw.target_idx]
            lines.append(
                f"  `{src} → {tgt}` = {sw.value:+.3f}  "
                f"[{sw.magnitude}, {sw.origin}]  {sw.rationale}"
            )
        lines.append(f"  来源分布: {dict(w_counts)}")

        lines.append("")
        lines.append("### 偏置项")
        for bw in self._biases:
            b_counts[bw.origin] += 1
            tgt = self._target_labels[bw.target_idx]
            lines.append(
                f"  `{tgt}` bias = {bw.value:+.3f}  "
                f"[{bw.origin}]  {bw.rationale}"
            )
        lines.append(f"  来源分布: {dict(b_counts)}")
        lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def _count_origin(entries: list) -> dict:
    """计算 entries 中各 origin 的出现次数。"""
    counts = defaultdict(int)
    for e in entries:
        origin = e["origin"] if isinstance(e, dict) else e.origin
        counts[origin] += 1
    return dict(counts)


# ═══════════════════════════════════════════════════════════════════
# ConstraintRegistry
# ═══════════════════════════════════════════════════════════════════

class ConstraintRegistry:
    """中央约束注册表 —— 所有矩阵/向量的约束检查在此集中执行和记录。

    用法:
        ConstraintRegistry.register("MY_MATRIX", check_fn, arg1, arg2)
        results = ConstraintRegistry.run_all("MY_MATRIX")
        print(ConstraintRegistry.audit_report())
        report = ConstraintRegistry.structured_report()  # JSON 格式
    """

    _checks: dict[str, list] = defaultdict(list)
    _results: dict[str, dict] = {}

    @classmethod
    def register(cls, label: str, check_fn, *args, **kwargs) -> None:
        """为一个矩阵注册一个约束检查函数。"""
        cls._checks[label].append((check_fn, args, kwargs))

    @classmethod
    def run_all(cls, label: str) -> dict:
        """对一个矩阵执行所有注册的检查。返回 {检查名: (PASS/FAIL, detail)}。"""
        results = {}
        for check_fn, args, kwargs in cls._checks.get(label, []):
            fn_name = getattr(check_fn, "__name__", str(check_fn))
            try:
                result = check_fn(*args, **kwargs)
                results[fn_name] = ("PASS", result)
            except ConstraintViolationError as e:
                results[fn_name] = ("FAIL", str(e))
            except Exception as e:
                results[fn_name] = ("ERROR", str(e))
        cls._results[label] = results
        return results

    @classmethod
    def get_results(cls, label: str) -> dict:
        """获取已执行过的检查结果。"""
        return cls._results.get(label, {})

    @classmethod
    def audit_report(cls) -> str:
        """生成完整的参数审计报告（文本格式）。"""
        lines = [
            "# Constraint Audit Report",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]
        for label, results in cls._results.items():
            all_pass = all(v[0] == "PASS" for v in results.values())
            status = "✅" if all_pass else "❌"
            lines.append(f"## {status} {label}")
            for check, (verdict, detail) in results.items():
                icon = "✅" if verdict == "PASS" else "❌"
                detail_str = _summarize_detail(detail)
                lines.append(f"  {icon} {check}: {detail_str}")
            lines.append("")
        return "\n".join(lines)

    @classmethod
    def structured_report(cls) -> list[dict]:
        """生成结构化审计报告（JSON 兼容）。"""
        reports = []
        for label, results in cls._results.items():
            all_pass = all(v[0] == "PASS" for v in results.values())
            reports.append({
                "label": label,
                "status": "PASS" if all_pass else "FAIL",
                "checks": [
                    {"name": check, "verdict": verdict, "detail": _summarize_detail(detail)}
                    for check, (verdict, detail) in results.items()
                ],
            })
        return reports

    @classmethod
    def clear(cls) -> None:
        """清空所有注册（用于测试隔离）。"""
        cls._checks.clear()
        cls._results.clear()

    @classmethod
    def verify_all(cls) -> dict:
        """全局合规性检查：汇总所有 label 的检查结果。

        Returns:
            {"status": "PASS"/"FAIL",
             "total_labels": int,
             "pass_count": int,
             "fail_count": int,
             "details": dict[label, {"status": ..., "checks": ...}]}
        """
        pass_count = 0
        fail_count = 0
        details = {}

        for label, results in cls._results.items():
            all_pass = all(v[0] == "PASS" for v in results.values())
            if all_pass:
                pass_count += 1
            else:
                fail_count += 1
            details[label] = {
                "status": "PASS" if all_pass else "FAIL",
                "checks": {
                    check: verdict
                    for check, (verdict, detail) in results.items()
                },
            }

        overall = "PASS" if fail_count == 0 else "FAIL"
        return {
            "status": overall,
            "total_labels": len(cls._results),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "details": details,
        }


def _summarize_detail(detail: Any) -> str:
    """将 detail 截断为简短字符串。"""
    if isinstance(detail, dict) and "entries" in detail:
        n = len(detail["entries"])
        return f"{n} entries, origins: {detail.get('origin_counts', {})}"
    if isinstance(detail, str) and len(detail) > 120:
        return detail[:117] + "..."
    return str(detail) if detail is not None else ""


# ═══════════════════════════════════════════════════════════════════
# 便捷工具：批量导出所有已注册的 WeightMapper 配置到 JSON
# ═══════════════════════════════════════════════════════════════════

_exported_mappers: dict[str, "WeightMapper | WeightVector | LinearMapping"] = {}

def register_mapper(mapper: "WeightMapper | WeightVector | LinearMapping") -> None:
    """注册一个 mapper 供后续批量导出。

    用法:
        mapper = WeightMapper(...)
        mapper.connect(...)
        register_mapper(mapper)
    """
    _exported_mappers[mapper.label] = mapper

def export_all(output_dir: str = "params", indent: int = 2) -> list[str]:
    """将所有已注册的 mapper 导出为 JSON 文件。

    Returns:
        导出的文件路径列表
    """
    paths = []
    for label, mapper in _exported_mappers.items():
        safe_name = label.lower().replace(" ", "_")
        path = os.path.join(output_dir, f"{safe_name}.json")
        if isinstance(mapper, (WeightMapper, WeightVector, LinearMapping)):
            mapper.to_json(path, indent=indent)
            paths.append(path)
    return paths

def load_all(input_dir: str) -> dict[str, "WeightMapper | WeightVector | LinearMapping"]:
    """从目录加载所有 JSON 配置。"""
    result = {}
    for fname in os.listdir(input_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(input_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        mapper_type = d.get("mapper_type", "matrix")
        if mapper_type == "matrix":
            mapper = WeightMapper.from_dict(d)
        elif mapper_type == "vector":
            mapper = WeightVector.from_dict(d)
        elif mapper_type == "linear_mapping":
            mapper = LinearMapping.from_dict(d)
        else:
            continue
        result[mapper.label] = mapper
    return result
