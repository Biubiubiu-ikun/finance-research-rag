# -*- coding: utf-8 -*-
"""
eval_reranker.py — reranker 微调评测：通用 bge-reranker vs 金融微调 reranker

公平对比：对每个评测 query 先用【混合检索】取同一候选池(pool，不重排)，再分别用
  ① 不重排(混合 RRF 原序)  ② 通用 reranker  ③ 微调 reranker
重排候选池，算 chunk Recall@1/Recall@5/MRR + doc Recall@5（留出集 60 题，与训练 query 0 重叠）。
候选池天花板=gold 命中候选池的比例(reranker 只能在池内重排，救不回没召回的)。

用法：python eval_reranker.py
"""
import os
import sys
import json

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EIGHT = os.path.join(os.path.dirname(BASE), "ai面试八股rag", "models")
EVAL = os.path.join(BASE, "data", "eval", "eval_finance.jsonl")
GEN = os.path.join(BASE, "models", "bge-reranker-base")
if not os.path.exists(GEN):
    GEN = os.path.join(EIGHT, "bge-reranker-base")
FT = os.path.join(BASE, "models", "bge-reranker-finance")
POOL = 50   # 候选池大小(混合检索取前 POOL 个交给 reranker 重排)
TOPK = 10


def rank_of(items, key, gold):
    for i, (cid, src) in enumerate(items, 1):
        v = cid if key == "chunk" else src.get("doc_id")
        if v == gold:
            return i
    return 0


def score(pools, ce):
    """ce=None 表示不重排(用混合原序)；返回 chunkR@1/R@5/MRR/docR@5。"""
    c1 = c5 = mrr = d5 = 0
    for e, items in pools:
        if ce is None:
            ranked = items
        else:
            s = ce.predict([(e["query"], src["content"]) for _, src in items])
            idx = sorted(range(len(items)), key=lambda i: s[i], reverse=True)
            ranked = [items[i] for i in idx]
        rc = rank_of(ranked, "chunk", e["gold_chunk_id"])
        rd = rank_of(ranked, "doc", e["gold_doc_id"])
        if rc == 1: c1 += 1
        if 1 <= rc <= 5: c5 += 1
        if rc: mrr += 1.0 / rc
        if 1 <= rd <= 5: d5 += 1
    n = len(pools)
    return c1 / n, c5 / n, mrr / n, d5 / n


def main():
    from finrag.retrieval.retrieve_es import Retriever
    from sentence_transformers import CrossEncoder
    evalset = [json.loads(l) for l in open(EVAL, encoding="utf-8") if l.strip()]
    print(f"留出评测集 {len(evalset)} 条 | 候选池 pool={POOL}")
    r = Retriever()
    # 预取候选池(混合检索, 不重排)，三种重排共用 → 公平对比
    pools, ceil = [], 0
    for e in evalset:
        items = r.search(e["query"], k=POOL, method="hybrid", rerank=False, pool=POOL)
        pools.append((e, items))
        if rank_of(items, "chunk", e["gold_chunk_id"]):
            ceil += 1
    print(f"候选池天花板(gold 命中池内)：{ceil}/{len(evalset)} = {ceil/len(evalset)*100:.1f}%（reranker 能达到的 R@k 上限）\n")

    configs = [("混合(不重排)", None), ("通用 reranker", CrossEncoder(GEN))]
    if os.path.exists(FT):
        configs.append(("微调 reranker", CrossEncoder(FT)))
    else:
        print(f"⚠️ 未找到微调 reranker（{FT}），请先跑 finetune_reranker.py\n")

    print(f"{'重排方式':<16}{'chunkR@1':>10}{'chunkR@5':>10}{'MRR':>9}{'docR@5':>9}")
    print("-" * 54)
    sc = {}
    for name, ce in configs:
        a, b, m, d = score(pools, ce)
        sc[name] = (a, b, m, d)
        print(f"{name:<16}{a*100:>9.1f}%{b*100:>9.1f}%{m:>9.3f}{d*100:>8.1f}%")
    print("-" * 54)
    if "微调 reranker" in sc:
        g, f = sc["通用 reranker"], sc["微调 reranker"]
        print(f"微调 vs 通用 reranker：chunkR@1 {g[0]*100:.1f}%→{f[0]*100:.1f}% ({(f[0]-g[0])*100:+.1f}pt) | "
              f"chunkR@5 {g[1]*100:.1f}%→{f[1]*100:.1f}% ({(f[1]-g[1])*100:+.1f}pt) | "
              f"MRR {g[2]:.3f}→{f[2]:.3f} ({f[2]-g[2]:+.3f})")
        b0 = sc["混合(不重排)"]
        print(f"微调 reranker vs 不重排：MRR {b0[2]:.3f}→{f[2]:.3f} ({f[2]-b0[2]:+.3f})")


if __name__ == "__main__":
    main()
