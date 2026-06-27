# -*- coding: utf-8 -*-
"""
eval_es.py — P3 检索量化评测

在金融评测集上对比 向量 / BM25 / 混合(RRF) / 混合+Rerank：
  - chunk 级 Recall@1、Recall@5、MRR（gold 源块是否被召回到前列）
  - doc 级 Recall@5（gold 源文档是否被召回，更宽松）
据此判断：混合/Rerank 是否显著优于纯向量 → 决定是否需要 finance 词典 / bge-base / 微调。
"""
import os
import sys
import json

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL = os.path.join(BASE, "data", "eval", "eval_finance.jsonl")
TOPK = 10


def rank_of(items, key, gold):
    for i, (cid, src) in enumerate(items, 1):
        v = cid if key == "chunk" else src.get("doc_id")
        if v == gold:
            return i
    return 0


def metrics(r, evalset, **kw):
    c1 = c5 = mrr = d5 = 0
    for e in evalset:
        items = r.search(e["query"], k=TOPK, **kw)
        rc = rank_of(items, "chunk", e["gold_chunk_id"])
        rd = rank_of(items, "doc", e["gold_doc_id"])
        if rc == 1: c1 += 1
        if 1 <= rc <= 5: c5 += 1
        if rc: mrr += 1.0 / rc
        if 1 <= rd <= 5: d5 += 1
    n = len(evalset)
    return c1 / n, c5 / n, mrr / n, d5 / n


def main():
    from finrag.retrieval.retrieve_es import Retriever
    evalset = [json.loads(l) for l in open(EVAL, encoding="utf-8") if l.strip()]
    print(f"金融评测集 {len(evalset)} 条")
    r = Retriever()

    configs = [
        ("向量", dict(method="vector")),
        ("BM25", dict(method="bm25")),
        ("混合(RRF)", dict(method="hybrid")),
        ("混合+Rerank", dict(method="hybrid", rerank=True)),
    ]
    print(f"\n{'方法':<14}{'chunkR@1':>10}{'chunkR@5':>10}{'MRR':>9}{'docR@5':>9}")
    print("-" * 52)
    sc = {}
    for name, kw in configs:
        a, b, m, d = metrics(r, evalset, **kw)
        sc[name] = (a, b, m, d)
        print(f"{name:<14}{a*100:>9.1f}%{b*100:>9.1f}%{m:>9.3f}{d*100:>8.1f}%")
    print("-" * 52)
    base, best = sc["向量"], sc["混合+Rerank"]
    print(f"混合+Rerank vs 纯向量：chunkR@5 {base[1]*100:.1f}%→{best[1]*100:.1f}% "
          f"({(best[1]-base[1])*100:+.1f}pt) | MRR {base[2]:.3f}→{best[2]:.3f} ({best[2]-base[2]:+.3f})")
    bm = sc["BM25"]
    print(f"BM25 单路 chunkR@1 = {bm[0]*100:.1f}%（金融术语/代码多，看BM25是否给力）")


if __name__ == "__main__":
    main()
