"""Lunar 长期记忆系统。

提供记忆节点的 Pydantic 定义、JSON 文件持久化、以及基于内部状态的余弦相似度检索。

Usage:
    from memory import MemoryNode, MemoryStore, search_by_internal_state

    store = MemoryStore("db/memories.json")
    node = MemoryNode(
        title="关于一起看红月的对话",
        content="用户说下次一起去看红月，角色害羞地答应了...",
        state_checkpoint={
            "internal_state": current_internal.copy(),
            "relationship_state": current_rel.copy(),
            "surface_state": current_surface.copy(),
        },
    )
    store.add(node)

    # 按内部状态检索相似记忆
    results = store.search_by_internal_state(current_internal, top_k=3)
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_serializer, field_validator


# ═══════════════════════════════════════════════════════════════
# MemoryNode — 记忆节点
# ═══════════════════════════════════════════════════════════════

class MemoryNode(BaseModel):
    """长期记忆节点。

    每个节点代表一段有意义的互动记忆，包含对话内容、当时的心理状态快照，
    以及用于检索的元数据。

    Attributes:
        id: UUID v4 唯一标识。
        title: 记忆标题（通常为用户消息截断到 50 字）。
        content: 详细内容（用户消息 + 角色回复的完整文本）。
        created_at: ISO 8601 创建时间戳。
        user_message: 触发记忆的用户消息（截断到 300 字）。
        character_response: 角色当时的回复（截断到 300 字）。
        emotional_weight: 形成时的情感重量 [0,1]。
        significance: 综合显著性分数 [0,1]。
        state_checkpoint: 形成时的心理状态快照。
            keys: "internal_state" (8,), "relationship_state" (6,), "surface_state" (7,)
            values: numpy float64 数组
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    content: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    state_checkpoint: Dict[str, np.ndarray] = Field(default_factory=dict)

    model_config = {
        "arbitrary_types_allowed": True,
    }

    # ── numpy 序列化：list ↔ ndarray ──

    @field_validator("state_checkpoint", mode="before")
    @classmethod
    def _validate_state_checkpoint(cls, v: object) -> Dict[str, np.ndarray]:
        """反序列化时：将 checkpoint 中各 list 值转回 numpy float64 数组。"""
        if not isinstance(v, dict):
            return {}
        result: Dict[str, np.ndarray] = {}
        for key, val in v.items():
            if isinstance(val, list):
                result[key] = np.array(val, dtype=np.float64)
            elif isinstance(val, np.ndarray):
                result[key] = val.astype(np.float64)
        return result

    @field_serializer("state_checkpoint")
    def _serialize_state_checkpoint(self, v: Dict[str, np.ndarray]) -> Dict[str, list]:
        """序列化时：将 checkpoint 中各 ndarray 值转为 Python list。"""
        return {key: val.tolist() for key, val in v.items()}

    # ── 工厂方法 ──

    @classmethod
    def from_state_vectors(
        cls,
        title: str = "",
        content: str = "",
        internal_state: Optional[np.ndarray] = None,
        relationship_state: Optional[np.ndarray] = None,
        surface_state: Optional[np.ndarray] = None,
    ) -> "MemoryNode":
        """从分离的状态向量创建 MemoryNode。

        自动组装 state_checkpoint 字典（跳过 None 向量）。
        """
        checkpoint: Dict[str, np.ndarray] = {}
        if internal_state is not None:
            checkpoint["internal_state"] = internal_state.copy()
        if relationship_state is not None:
            checkpoint["relationship_state"] = relationship_state.copy()
        if surface_state is not None:
            checkpoint["surface_state"] = surface_state.copy()

        return cls(
            title=title,
            content=content,
            state_checkpoint=checkpoint,
        )


# ═══════════════════════════════════════════════════════════════
# 数学工具
# ═══════════════════════════════════════════════════════════════

def cos_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个等长向量的余弦相似度。

    Args:
        a: 向量 a。
        b: 向量 b。

    Returns:
        余弦相似度，范围通常为 [-1, 1]（非负向量时为 [0, 1]）。
        任一为零向量则返回 0.0。
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ═══════════════════════════════════════════════════════════════
# MemoryStore — 记忆持久化存储
# ═══════════════════════════════════════════════════════════════

class MemoryStore:
    """JSON 文件记忆存储。

    提供记忆的加载、保存、增删和基于内部状态的余弦相似度检索。
    使用原子写入（先写临时文件再 rename）保证已有数据不损坏。

    Usage:
        store = MemoryStore("db/memories.json")
        nodes = store.get_all()
        store.add(node)
        results = store.search_by_internal_state(current_internal, top_k=3)
    """

    def __init__(self, filepath: str):
        """初始化存储。

        Args:
            filepath: JSON 文件路径。目录不存在时会在首次 save() 时自动创建。
        """
        self.filepath = Path(filepath)

    # ── 持久化 ──

    def load(self) -> List[MemoryNode]:
        """从文件加载所有记忆节点。

        文件不存在或 JSON 损坏时返回空列表，不抛异常。

        Returns:
            记忆节点列表（可能为空）。
        """
        if not self.filepath.exists():
            return []
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

        if not isinstance(data, list):
            return []

        nodes: List[MemoryNode] = []
        for item in data:
            try:
                nodes.append(MemoryNode.model_validate(item))
            except Exception:
                # 跳过损坏的条目
                continue
        return nodes

    def save(self, nodes: List[MemoryNode]) -> None:
        """原子写入记忆列表到文件。

        先写入临时文件，成功后再 rename 到目标路径。
        确保写入过程中断电/崩溃不会损坏已有数据。

        Args:
            nodes: 要保存的记忆节点列表。
        """
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # mode="json" 触发 @field_serializer，将 ndarray 转为 list
        data = [node.model_dump(mode="json") for node in nodes]

        tmp_path = self.filepath.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.filepath)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    # ── CRUD ──

    def get_all(self) -> List[MemoryNode]:
        """获取所有记忆节点。

        Returns:
            记忆节点列表（可能为空）。
        """
        return self.load()

    def add(self, node: MemoryNode) -> None:
        """添加一条记忆并保存。

        Args:
            node: 要添加的记忆节点。
        """
        nodes = self.load()
        nodes.append(node)
        self.save(nodes)

    def add_batch(self, new_nodes: List[MemoryNode]) -> None:
        """批量添加记忆并保存。

        Args:
            new_nodes: 要添加的记忆节点列表。
        """
        nodes = self.load()
        nodes.extend(new_nodes)
        self.save(nodes)

    def delete(self, node_id: str) -> bool:
        """按 ID 删除一条记忆。

        Args:
            node_id: 要删除的记忆 ID。

        Returns:
            是否成功找到并删除。
        """
        nodes = self.load()
        new_nodes = [n for n in nodes if n.id != node_id]
        if len(new_nodes) == len(nodes):
            return False
        self.save(new_nodes)
        return True

    def count(self) -> int:
        """返回记忆总数。"""
        return len(self.load())

    # ── 检索 ──

    def search_by_internal_state(
        self,
        query_internal: np.ndarray,
        top_k: int = 3,
        threshold: float = 0.0,
    ) -> List[Tuple[MemoryNode, float]]:
        """基于 internal_state 的余弦相似度检索。

        计算当前内部状态向量与每条记忆中 state_checkpoint["internal_state"]
        的余弦相似度，返回最相似的 top_k 条记忆。

        这是"普鲁斯特效应"检索：相似的情绪状态会唤起相关的记忆。

        Args:
            query_internal: 当前内部状态向量 (8,)。
            top_k: 最多返回的结果数。
            threshold: 最低相似度阈值 [0, 1]。低于此值的记忆不会被返回。
                       设为 0.75 可只返回强烈相似的情绪记忆。

        Returns:
            (记忆节点, 相似度) 列表，按相似度降序排列。
        """
        nodes = self.load()
        if not nodes:
            return []

        scored: List[Tuple[MemoryNode, float]] = []
        for node in nodes:
            internal = node.state_checkpoint.get("internal_state")
            if internal is None or not isinstance(internal, np.ndarray):
                continue
            sim = cos_similarity(query_internal, internal)
            if sim >= threshold:
                scored.append((node, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ── 按文件路径的工厂方法 ──

    @classmethod
    def from_file(cls, filepath: str) -> "MemoryStore":
        """从文件路径创建 MemoryStore 并立即返回已加载的实例。

        等价于 MemoryStore(filepath)，但语义更明确。

        Args:
            filepath: JSON 文件路径。

        Returns:
            MemoryStore 实例。
        """
        return cls(filepath)

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return f"MemoryStore({self.filepath!r}, {len(self)} memories)"


# ═══════════════════════════════════════════════════════════════
# 便捷函数：脱离 MemoryStore 使用
# ═══════════════════════════════════════════════════════════════

def load_memories(filepath: str) -> List[MemoryNode]:
    """从文件路径加载所有记忆节点。

    便捷函数，等价于 MemoryStore(filepath).load()。

    Args:
        filepath: JSON 文件路径。

    Returns:
        记忆节点列表（可能为空）。
    """
    return MemoryStore(filepath).load()


def search_by_internal_state(
    query_internal: np.ndarray,
    nodes: List[MemoryNode],
    top_k: int = 3,
    threshold: float = 0.0,
) -> List[Tuple[MemoryNode, float]]:
    """在给定的记忆节点列表中，按 internal_state 余弦相似度检索。

    便捷函数，不依赖 MemoryStore 实例，可直接对已加载的节点列表检索。

    Args:
        query_internal: 当前内部状态向量 (8,)。
        nodes: 记忆节点列表。
        top_k: 最多返回的结果数。
        threshold: 最低相似度阈值 [0, 1]。

    Returns:
        (记忆节点, 相似度) 列表，按相似度降序排列。
    """
    if not nodes:
        return []

    scored: List[Tuple[MemoryNode, float]] = []
    for node in nodes:
        internal = node.state_checkpoint.get("internal_state")
        if internal is None or not isinstance(internal, np.ndarray):
            continue
        sim = cos_similarity(query_internal, internal)
        if sim >= threshold:
            scored.append((node, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
