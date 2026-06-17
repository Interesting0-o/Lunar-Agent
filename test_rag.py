"""RAG 多模型对比测试: qwen3-embedding:8b vs bge-m3。

测量:
  1. 单次 embedding 生成的耗时
  2. 语义搜索命中率
  3. 第一人称记忆对检索的影响

bge-m3: BAAI 的多语言 embedding 模型, 1024d, 567M 参数。
支持 dense + sparse (lexical) 混合检索, 对中英文均有优化。
"""

import time
import numpy as np
from langchain_ollama import OllamaEmbeddings

from memory import MemoryNode, MemoryStore
from prompts.character_memories import SEED_MEMORIES


OLLAMA_URL = "http://172.18.96.1:11434"


def create_embeddings(model: str) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model, base_url=OLLAMA_URL)


def compute_embedding(text: str, emb: OllamaEmbeddings) -> np.ndarray | None:
    try:
        vector = emb.embed_query(text)
        return np.array(vector, dtype=np.float64)
    except Exception as e:
        print(f"    ❌ embedding 失败: {e}")
        return None


def build_and_index(emb_model: OllamaEmbeddings, store_id: str) -> tuple[MemoryStore, list[MemoryNode]]:
    """从 SEED_MEMORIES 构建 MemoryNode 并生成 embedding。"""
    nodes = []
    for item in SEED_MEMORIES:
        node = MemoryNode(title=item["title"], content=item["content"])
        nodes.append(node)

    success = 0
    for node in nodes:
        emb_vec = compute_embedding(node.content, emb_model)
        if emb_vec is not None:
            node.embedding = emb_vec
            success += 1

    store = MemoryStore(store_id)
    store.save(nodes)
    return store, nodes, success


def benchmark_embedding(emb_model: OllamaEmbeddings, warmup_text: str = "预热测试") -> tuple[float, int]:
    """测单次 embedding 耗时和维度。"""
    # warmup
    _ = compute_embedding(warmup_text, emb_model)

    times = []
    for text in ["克隆体和实验体的过去", "捉迷藏的游戏", "红月之下的约定"]:
        t0 = time.perf_counter()
        vec = compute_embedding(text, emb_model)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    dim = len(vec) if vec is not None else 0
    return float(np.mean(times)), dim


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度。"""
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    dot = float(np.dot(a, b))
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def search_by_embedding_local(
    store: MemoryStore,
    query_text: str,
    emb_model: OllamaEmbeddings,
    top_k: int = 3,
) -> list[tuple[MemoryNode, float]]:
    """本地 embedding 搜索，使用指定的 embedding 模型（绕过 memory.py 的全局 embedding）。"""
    if not query_text:
        return []

    query_vec = compute_embedding(query_text, emb_model)
    if query_vec is None:
        return []

    nodes = store.get_all()
    scored = []
    for node in nodes:
        emb = node.embedding
        if emb is None or not isinstance(emb, np.ndarray):
            continue
        if emb.shape != query_vec.shape:
            continue  # 维度不匹配，跳过
        sim = cos_sim(query_vec, emb)
        scored.append((node, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def run_search_test(store: MemoryStore, emb_model: OllamaEmbeddings,
                    queries: list[tuple[str, str]], label: str):
    """运行语义搜索命中率测试。"""
    hits = 0
    times = []
    details = []

    for query_text, expected_prefix in queries:
        t0 = time.perf_counter()
        results = search_by_embedding_local(store, query_text, emb_model, top_k=3)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        top_title = results[0][0].title if results else "(无)"
        hit = top_title.startswith(expected_prefix)
        if hit:
            hits += 1

        details.append((query_text, expected_prefix, top_title, hit, results))

    avg_time = np.mean(times)
    acc = hits / len(queries) * 100
    return acc, avg_time, details


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  RAG 模型对比: qwen3-embedding:8b vs bge-m3               ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    queries = [
        ("克隆体和实验体的过去",         "我是实验体A-872"),
        ("被制造出来的人偶，不是真正的人类", "我是实验体A-872"),
        ("姐姐教我的事情",               "捉迷藏是逃跑的演练"),
        ("捉迷藏的游戏",                 "捉迷藏是逃跑的演练"),
        ("独自去看世界却没有姐姐在身边的孤独","独自去看世界，却没有你"),
        ("那个夏天他穿越量子之海来救我",   "仲夏幻夜"),
        ("他用自己的血喂我，一点都不怕",   "仲夏幻夜"),
        ("被信任的人背叛和封印",         "被家臣和爷爷背叛"),
        ("一万年的等待和流放",           "一万年的等待让我长大了"),
        ("用自己的存在换他活下去",       "我与摆渡人的交易"),
        ("摩天轮上的重逢",               "摩天轮上的重逢"),
        ("你是我生命中的光",             "你是照进我生命里的一束光"),
        ("星星和月亮更喜欢哪一个",       "星星和月亮"),
        ("脚疼让他背我",                 "牧场奇谭"),
        ("不再是怪物了",                 "月下相续：我不再是怪物了"),
        ("平凡的幸福",                   "月下相续：我不再是怪物了"),
        ("红月之下的约定",               "你是照进我生命里的一束光"),
        ("吸血鬼的恋心",                 "你是照进我生命里的一束光"),
        ("为了重要的人可以变成任何东西",   "为姐姐提起链锯的那一刻"),
    ]

    models = [
        ("qwen3-embedding:8b", "qwen3-embedding:8b", "character_rag_qwen"),
        ("bge-m3",             "bge-m3",             "character_rag_bge"),
    ]

    results_all = {}

    for model_name, ollama_tag, store_id in models:
        print(f"\n{'─' * 60}")
        print(f"  📦 模型: {model_name}")
        print(f"{'─' * 60}")

        emb = create_embeddings(ollama_tag)

        # ── 基准: 单次 embedding 耗时 + 维度 ──
        avg_lat, dim = benchmark_embedding(emb)
        print(f"  单次 embedding 耗时: {avg_lat:.4f}s,  维度: {dim}")

        # ── 索引构建 + 全量 embedding 耗时 ──
        t0 = time.perf_counter()
        store, nodes, success = build_and_index(emb, store_id)
        index_time = time.perf_counter() - t0
        print(f"  索引构建: {success}/{len(nodes)} 成功, 耗时 {index_time:.2f}s")

        # ── 搜索测试 ──
        acc, search_avg, details = run_search_test(store, emb, queries, model_name)
        results_all[model_name] = (acc, avg_lat, search_avg, dim, details)

        print(f"  命中率: {acc:.1f}%,  单次搜索耗时: {search_avg:.4f}s")
        print(f"  一次完整 RAG 估计: {avg_lat + search_avg:.4f}s")

    # ═════════════════════════════════════════════════════════
    # 汇总对比
    # ═════════════════════════════════════════════════════════
    print(f"\n\n{'═' * 70}")
    print(f"  📊 汇总对比")
    print(f"{'═' * 70}")
    print(f"  {'指标':28s} | {'qwen3-embed:8b':20s} | {'bge-m3':20s}")
    print(f"  {'─' * 70}")
    print(f"  {'Embedding 维度':28s} | {results_all['qwen3-embedding:8b'][3]:>20d} | {results_all['bge-m3'][3]:>20d}")
    print(f"  {'单次 embedding 耗时':28s} | {results_all['qwen3-embedding:8b'][1]:>19.4f}s | {results_all['bge-m3'][1]:>19.4f}s")
    print(f"  {'搜索命中率':28s} | {results_all['qwen3-embedding:8b'][0]:>19.1f}% | {results_all['bge-m3'][0]:>19.1f}%")
    total_qwen = results_all['qwen3-embedding:8b'][1] + results_all['qwen3-embedding:8b'][2]
    total_bge = results_all['bge-m3'][1] + results_all['bge-m3'][2]
    print(f"  {'一次完整 RAG 估计':28s} | {total_qwen:>19.4f}s | {total_bge:>19.4f}s")

    # ── 逐条差异分析 ──
    qwen_details = results_all['qwen3-embedding:8b'][4]
    bge_details = results_all['bge-m3'][4]

    print(f"\n\n{'═' * 70}")
    print(f"  逐条对比 (✅=命中, ❌=未命中)")
    print(f"{'═' * 70}")
    print(f"  {'查询':30s} | {'qwen3-8b':12s} | {'bge-m3':12s}")
    print(f"  {'─' * 70}")

    both_correct = 0
    both_wrong = 0
    bge_better = 0
    qwen_better = 0

    for (q_text, q_exp, q_top, q_hit, _), (_, _, b_top, b_hit, _) in zip(qwen_details, bge_details):
        q_mark = "✅" if q_hit else "❌"
        b_mark = "✅" if b_hit else "❌"
        print(f"  {q_text:30s} | {q_mark} {q_top[:20]:20s} | {b_mark} {b_top[:20]:20s}")

        if q_hit and b_hit:
            both_correct += 1
        elif not q_hit and not b_hit:
            both_wrong += 1
        elif not q_hit and b_hit:
            bge_better += 1
        elif q_hit and not b_hit:
            qwen_better += 1

    print(f"  {'─' * 70}")
    print(f"  两个都命中: {both_correct}, 两个都未命中: {both_wrong}")
    print(f"  bge-m3 独自命中: {bge_better}, qwen 独自命中: {qwen_better}")


if __name__ == "__main__":
    main()
