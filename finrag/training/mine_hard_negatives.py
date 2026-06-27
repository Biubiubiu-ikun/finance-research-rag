# -*- coding: utf-8 -*-
"""
mine_hard_negatives.py — bge微调③ 难负样本挖掘（hard negative mining）

现有 train_pairs 只有 (query, positive)，微调靠 MNRL 的 in-batch 随机负样本——这些负样本"太easy"
(随便一个别的块，模型轻松区分)。难负样本=【和 query 词面/语义很像、但不是正确答案】的块，
逼模型学到更细的区分边界，通常能再涨 Recall。

做法：对每个 query 用混合检索取 top-k，排除其正例块(同 doc 也排，避免伪负)，取最靠前的非正例块作 hard neg。
输出 data/train/train_pairs_hn.jsonl: {query, positive, negative, chunk_id}
"""
import os
import sys
import json

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAIRS = os.path.join(BASE, "data", "train", "train_pairs.jsonl")
OUT = os.path.join(BASE, "data", "train", "train_pairs_hn.jsonl")


def main():
    from finrag.retrieval.retrieve_es import Retriever
    pairs = [json.loads(l) for l in open(PAIRS, encoding="utf-8") if l.strip()]
    print(f"训练对 {len(pairs)} 条，开始挖难负…")
    r = Retriever()
    # chunk_id → doc_id，挖负时连同源文档其它块一起排除(同篇大概率也相关→伪负)
    cid2doc = {}
    import glob
    for c in (json.loads(l) for l in open(os.path.join(BASE, "data", "chunks", "chunks.jsonl"), encoding="utf-8")):
        cid2doc[c["chunk_id"]] = c["doc_id"]

    out, n_hn = [], 0
    for i, p in enumerate(pairs, 1):
        gold_doc = cid2doc.get(p["chunk_id"])
        neg = None
        for cid, src in r.search(p["query"], k=10, method="hybrid"):
            if cid == p["chunk_id"] or src.get("doc_id") == gold_doc:
                continue  # 排除正例块及其同篇(防伪负)
            neg = src["content"]
            break
        rec = {"query": p["query"], "positive": p["positive"], "chunk_id": p["chunk_id"]}
        if neg:
            rec["negative"] = neg
            n_hn += 1
        out.append(rec)
        if i % 50 == 0:
            print(f"  {i}/{len(pairs)}，已挖到难负 {n_hn}")
    with open(OUT, "w", encoding="utf-8") as w:
        for rec in out:
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n✓ {len(out)} 条，其中 {n_hn} 条带难负 → {OUT}")


if __name__ == "__main__":
    main()
