"""
RAG 评测脚本 — 量化评估知识库检索质量

评测指标：
- 关键词命中率：检索结果中是否包含 expected_keywords
- 来源准确率：检索结果的 source_file 是否匹配

用法：
    HF_HUB_OFFLINE=1 python3 eval_rag.py
"""

import json
import time
from pathlib import Path

# ============================================================
# 【知识点】混合检索初始化 — 复用 tools.py 的检索逻辑
# ============================================================
from tools import _get_retriever, _get_bm25, _rrf_merge
import jieba

def eval_one(question: str, expected_keywords: list, source_file: str) -> dict:
    """评测单个 QA 对"""
    retriever = _get_retriever()
    bm25, bm25_docs = _get_bm25()

    start = time.time()

    # 向量检索
    vector_docs = retriever.invoke(question)

    # BM25 检索
    bm25_results = []
    if bm25 is not None:
        tokens = list(jieba.cut(question))
        scores = bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        bm25_results = [bm25_docs[i] for i in top_indices if scores[i] > 0]

    # RRF 融合
    merged = _rrf_merge(vector_docs, bm25_results, k=60, top_n=3)

    elapsed = time.time() - start

    # 检查关键词命中
    all_text = " ".join(merged)
    hits = [kw for kw in expected_keywords if kw in all_text]
    keyword_hit = len(hits) / len(expected_keywords) if expected_keywords else 0

    # 检查来源准确率
    source_hit = any(source_file in str(doc.metadata.get("source", "")) for doc in vector_docs)

    return {
        "question": question,
        "keyword_hit": keyword_hit,
        "hits": hits,
        "misses": [kw for kw in expected_keywords if kw not in all_text],
        "source_hit": source_hit,
        "time_ms": round(elapsed * 1000),
    }

def main():
    eval_path = Path(__file__).parent / "eval_set.json"
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    print(f"📊 开始评测（{len(eval_set)} 个 QA 对）\n")

    results = []
    for i, qa in enumerate(eval_set):
        print(f"  [{i+1}/{len(eval_set)}] {qa['question'][:40]}...", end=" ")
        result = eval_one(qa["question"], qa["expected_keywords"], qa["source_file"])
        results.append(result)
        status = "✅" if result["keyword_hit"] >= 0.5 else "❌"
        print(f"{status} 关键词:{result['keyword_hit']:.0%} 来源:{'✅' if result['source_hit'] else '❌'} {result['time_ms']}ms")

    # 汇总统计
    print(f"\n{'='*50}")
    print(f"📊 评测结果汇总")
    print(f"{'='*50}")

    avg_keyword = sum(r["keyword_hit"] for r in results) / len(results)
    source_acc = sum(1 for r in results if r["source_hit"]) / len(results)
    avg_time = sum(r["time_ms"] for r in results) / len(results)

    print(f"  关键词平均命中率: {avg_keyword:.1%}")
    print(f"  来源准确率:       {source_acc:.1%}")
    print(f"  平均检索耗时:     {avg_time:.0f}ms")
    print(f"  总评测数:         {len(results)}")

    # 输出失败的用例
    fails = [r for r in results if r["keyword_hit"] < 0.5]
    if fails:
        print(f"\n⚠️ 命中率 < 50% 的用例（{len(fails)} 个）：")
        for r in fails:
            print(f"  - {r['question']}")
            print(f"    命中: {r['hits']}, 缺失: {r['misses']}")

if __name__ == "__main__":
    main()
