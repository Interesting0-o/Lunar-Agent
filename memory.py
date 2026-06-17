"""Lunar 长期记忆系统。

提供记忆节点的 Pydantic 定义、JSON 文件持久化、以及三种检索方式：
  1. 向量查询 — 基于心理状态向量（internal_state）的余弦相似度
  2. Embedding 查询 — 基于文本语义嵌入向量的余弦相似度
  3. 混合查询 — 以上两种的加权组合

Usage:
    from memory import MemoryNode, MemoryStore, search_by_internal_state

    store = MemoryStore("memories")
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

    # ① 向量查询：按情绪状态相似度
    results = store.search_by_internal_state(current_internal, top_k=3)

    # ② Embedding 查询：按文本语义相似度
    results = store.search_by_embedding(query_text="红月下的约定", top_k=3)

    # ③ 混合查询：两者加权组合
    results = store.hybrid_search(
        query_internal=current_internal,
        query_text="红月下的约定",
        state_weight=0.4,
        embedding_weight=0.6,
        top_k=3,
    )
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from langchain_core.messages import BaseMessage
import numpy as np
from pydantic import BaseModel, Field, field_serializer, field_validator


class MemoryMessage(BaseMessage):
    type:str = "memory"


# ═══════════════════════════════════════════════════════════════
# MemoryNode — 记忆节点
# ═══════════════════════════════════════════════════════════════

class MemoryNode(BaseModel):
    """长期记忆节点。

    每个节点代表一段有意义的互动记忆，包含对话内容、当时的心理状态快照、
    文本语义嵌入向量，以及用于检索的元数据。

    Attributes:
        id: UUID v4 唯一标识。
        title: 记忆标题（通常为用户消息截断到 50 字）。
        content: 详细内容（用户消息 + 角色回复的完整文本）。
        created_at: ISO 8601 创建时间戳。
        state_checkpoint: 形成时的心理状态快照。
            keys: "internal_state" (8,), "relationship_state" (6,), "surface_state" (7,)
            values: numpy float64 数组
        embedding: 文本语义嵌入向量（由 embedding 模型生成），
                   用于与 state_checkpoint 联合做双重相似度检索。
                   维度取决于所用模型（如 nomic-embed-text: 768 维）。
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    content: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    state_checkpoint: Dict[str, np.ndarray] = Field(default_factory=dict)
    embedding: Optional[np.ndarray] = Field(default=None)

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

    # ── embedding 序列化 ──

    @field_validator("embedding", mode="before")
    @classmethod
    def _validate_embedding(cls, v: object) -> Optional[np.ndarray]:
        """反序列化时：将 embedding list 转回 numpy float64 数组。"""
        if v is None:
            return None
        if isinstance(v, list):
            return np.array(v, dtype=np.float64)
        if isinstance(v, np.ndarray):
            return v.astype(np.float64)
        return None

    @field_serializer("embedding")
    def _serialize_embedding(self, v: Optional[np.ndarray]) -> Optional[list]:
        """序列化时：将 embedding ndarray 转为 Python list。"""
        if v is None:
            return None
        return v.tolist()

    # ── 工厂方法 ──

    @classmethod
    def from_state_vectors(
        cls,
        title: str = "",
        content: str = "",
        internal_state: Optional[np.ndarray] = None,
        relationship_state: Optional[np.ndarray] = None,
        surface_state: Optional[np.ndarray] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> "MemoryNode":
        """从分离的状态向量创建 MemoryNode。

        自动组装 state_checkpoint 字典（跳过 None 向量）。

        Args:
            title: 记忆标题。
            content: 记忆内容。
            internal_state: 内部状态向量 (8,)。
            relationship_state: 关系状态向量 (6,)。
            surface_state: 表面状态向量 (7,)。
            embedding: 文本语义嵌入向量（可选）。
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
            embedding=embedding.copy() if embedding is not None else None,
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
# Embedding — 文本语义嵌入
# ═══════════════════════════════════════════════════════════════

def compute_embedding(
    text: str,
) -> Optional[np.ndarray]:
    """通过 Ollama 生成文本的语义嵌入向量。

    使用 langchain-ollama 的 OllamaEmbeddings 将文本转为向量。
    失败时返回 None（Ollama 未运行、模型未拉取等），由调用方自行决定行为。

    Args:
        text: 待编码的文本。
        model: Ollama embedding 模型名，如 "nomic-embed-text"、"bge-m3"、"mxbai-embed-large"。
        base_url: Ollama 服务地址。

    Returns:
        embedding 向量，或 None（生成失败时）。
    """
    if not text:
        return None

    try:
        from llm import embeddings
        vector = embeddings.embed_query(text)
        return np.array(vector, dtype=np.float64)
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Embedding 生成失败: %s", e,
        )
        return None


# ═══════════════════════════════════════════════════════════════
# MemoryStore — 记忆持久化存储
# ═══════════════════════════════════════════════════════════════

class MemoryStore:
    """JSON 文件记忆存储。

    提供记忆的加载、保存、增删和基于内部状态的余弦相似度检索。
    使用原子写入（先写临时文件再 rename）保证已有数据不损坏。

    Usage:
        store = MemoryStore("memories")
        nodes = store.get_all()
        store.add(node)
        results = store.search_by_internal_state(current_internal, top_k=3)
    """

    def __init__(self, memory_id: str):
        """初始化存储。

        Args:
            memory_id: 唯一记忆标识，对应文件 memory/{memory_id}.json。
                      目录不存在时会在首次 save() 时自动创建。
        """
        self.memory_id = memory_id
        self.filepath = Path(f"memory/{memory_id}.json")

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
        threshold: float = 0.8,
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

    def search_by_embedding(
        self,
        query_text: str,
        top_k: int = 3,
        threshold: float = 0.0,
    ) -> List[Tuple[MemoryNode, float]]:
        """基于文本语义嵌入的余弦相似度检索。

        将查询文本编码为 embedding 向量，与每条记忆中存储的 embedding
        计算余弦相似度，返回语义最相似的 top_k 条记忆。

        Args:
            query_text: 查询文本，用于生成语义嵌入。
            top_k: 最多返回的结果数。
            threshold: 最低相似度阈值 [0, 1]。
            embedding_model: Ollama embedding 模型名。
            embedding_base_url: Ollama 服务地址。

        Returns:
            (记忆节点, 相似度) 列表，按相似度降序排列。
        """
        if not query_text:
            return []

        query_embedding = compute_embedding(query_text)
        if query_embedding is None:
            return []

        nodes = self.load()
        if not nodes:
            return []

        scored: List[Tuple[MemoryNode, float]] = []
        for node in nodes:
            emb = node.embedding
            if emb is None or not isinstance(emb, np.ndarray):
                continue
            sim = cos_similarity(query_embedding, emb)
            if sim >= threshold:
                scored.append((node, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def hybrid_search(
        self,
        query_internal: np.ndarray,
        query_text: str,
        state_weight: float = 0.5,
        embedding_weight: float = 0.5,
        top_k: int = 3,
        threshold: float = 0.0 
    ) -> List[Tuple[MemoryNode, float]]:
        """混合检索：向量查询 + Embedding 查询的加权组合。

        对每条记忆同时计算：
          1. 向量相似度 — query_internal 与 memory.state_checkpoint["internal_state"] 的余弦相似度
          2. 文本相似度 — query_text 的嵌入向量与 memory.embedding 的余弦相似度

        最终得分 = state_weight × 向量相似度 + embedding_weight × 文本相似度。
        若某条记忆缺少某个字段，则仅用另一部分（权重自动归一化）。

        Args:
            query_internal: 当前内部状态向量 (8,)。
            query_text: 查询文本，用于生成语义嵌入。
            state_weight: 向量相似度权重。
            embedding_weight: 文本嵌入相似度权重。
            top_k: 最多返回的结果数。
            threshold: 最低综合得分阈值 [0, 1]。
            embedding_model: Ollama embedding 模型名。
            embedding_base_url: Ollama 服务地址。

        Returns:
            (记忆节点, 综合得分) 列表，按得分降序排列。
        """
        if not query_text:
            return self.search_by_internal_state(query_internal, top_k, threshold)

        query_embedding = compute_embedding(query_text)

        nodes = self.load()
        if not nodes:
            return []

        scored: List[Tuple[MemoryNode, float]] = []
        for node in nodes:
            internal = node.state_checkpoint.get("internal_state")
            emb = node.embedding

            # 至少需要一种信号
            if (internal is None or not isinstance(internal, np.ndarray)) and \
               (emb is None or not isinstance(emb, np.ndarray)):
                continue

            sim = 0.0
            total_weight = 0.0

            if isinstance(internal, np.ndarray):
                state_sim = cos_similarity(query_internal, internal)
                if state_sim > 0.0:
                    sim += state_weight * state_sim
                    total_weight += state_weight

            if isinstance(emb, np.ndarray) and query_embedding is not None:
                emb_sim = cos_similarity(query_embedding, emb)
                if emb_sim > 0.0:
                    sim += embedding_weight * emb_sim
                    total_weight += embedding_weight

            # 全部信号都为零 → 跳过
            if total_weight <= 0.0:
                continue

            sim /= total_weight  # 归一化，补偿缺失字段
            if sim >= threshold:
                scored.append((node, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ── 按文件路径的工厂方法 ──

    @classmethod
    def from_id(cls, memory_id: str) -> "MemoryStore":
        """从 memory_id 创建 MemoryStore。

        Args:
            memory_id: 唯一记忆标识。

        Returns:
            MemoryStore 实例。
        """
        return cls(memory_id)

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return f"MemoryStore({self.filepath!r}, {len(self)} memories)"


# ═══════════════════════════════════════════════════════════════
# 便捷函数：脱离 MemoryStore 使用
# ═══════════════════════════════════════════════════════════════

def load_memories(memory_id: str) -> List[MemoryNode]:
    """加载指定 memory_id 对应的所有记忆节点。

    便捷函数，等价于 MemoryStore(memory_id).load()。

    Args:
        memory_id: 唯一记忆标识。

    Returns:
        记忆节点列表（可能为空）。
    """
    return MemoryStore(memory_id).load()


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


def search_by_embedding(
    query_text: str,
    nodes: List[MemoryNode],
    top_k: int = 3,
    threshold: float = 0.0
) -> List[Tuple[MemoryNode, float]]:
    """在给定的记忆节点列表中，按文本语义嵌入余弦相似度检索。

    便捷函数，不依赖 MemoryStore 实例，可直接对已加载的节点列表检索。

    Args:
        query_text: 查询文本，用于生成语义嵌入。
        nodes: 记忆节点列表。
        top_k: 最多返回的结果数。
        threshold: 最低相似度阈值 [0, 1]。
        embedding_model: Ollama embedding 模型名。
        embedding_base_url: Ollama 服务地址。

    Returns:
        (记忆节点, 相似度) 列表，按相似度降序排列。
    """
    if not nodes or not query_text:
        return []

    query_embedding = compute_embedding(query_text)
    if query_embedding is None:
        return []

    scored: List[Tuple[MemoryNode, float]] = []
    for node in nodes:
        emb = node.embedding
        if emb is None or not isinstance(emb, np.ndarray):
            continue
        sim = cos_similarity(query_embedding, emb)
        if sim >= threshold:
            scored.append((node, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def hybrid_search(
    query_internal: np.ndarray,
    query_text: str,
    nodes: List[MemoryNode],
    state_weight: float = 0.5,
    embedding_weight: float = 0.5,
    top_k: int = 3,
    threshold: float = 0.0 
) -> List[Tuple[MemoryNode, float]]:
    """在给定的记忆节点列表中，进行向量 + Embedding 加权混合检索。

    便捷函数，不依赖 MemoryStore 实例，可直接对已加载的节点列表检索。
    组合了 search_by_internal_state 和 search_by_embedding 两种检索信号。

    Args:
        query_internal: 当前内部状态向量 (8,)。
        query_text: 查询文本，用于生成语义嵌入。
        nodes: 记忆节点列表。
        state_weight: 向量相似度权重。
        embedding_weight: 文本嵌入相似度权重。
        top_k: 最多返回的结果数。
        threshold: 最低综合得分阈值 [0, 1]。
        embedding_model: Ollama embedding 模型名。
        embedding_base_url: Ollama 服务地址。

    Returns:
        (记忆节点, 综合得分) 列表，按得分降序排列。
    """
    if not nodes:
        return []

    query_embedding = compute_embedding(query_text)

    scored: List[Tuple[MemoryNode, float]] = []
    for node in nodes:
        internal = node.state_checkpoint.get("internal_state")
        emb = node.embedding

        if (internal is None or not isinstance(internal, np.ndarray)) and \
           (emb is None or not isinstance(emb, np.ndarray)):
            continue

        sim = 0.0
        total_weight = 0.0

        if isinstance(internal, np.ndarray):
            state_sim = cos_similarity(query_internal, internal)
            if state_sim > 0.0:
                sim += state_weight * state_sim
                total_weight += state_weight

        if isinstance(emb, np.ndarray) and query_embedding is not None:
            emb_sim = cos_similarity(query_embedding, emb)
            if emb_sim > 0.0:
                sim += embedding_weight * emb_sim
                total_weight += embedding_weight

        if total_weight <= 0.0:
            continue

        sim /= total_weight
        if sim >= threshold:
            scored.append((node, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
