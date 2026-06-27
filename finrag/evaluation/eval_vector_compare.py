# -*- coding: utf-8 -*-
"""
eval_vector_compare.py — 向量检索【横向对比 + 微调消融】（纯内存，不依赖 ES）

回答两件事：
  ① 横向对比：领域微调的 bge-small 能不能打过 更大/更强的【通用】模型(bge-base / gte-base-zh / bge-m3)？
  ② 微调消融：通用 bge-small → in-batch 微调 → +难负样本，各涨多少？
对每个模型分别 encode 全部 chunk + 评测 query(各模型用各自的检索指令前缀)，在留出集上算向量 R@1/R@5/MRR。
纯内存 cosine、不入 ES，所以各模型维度不同(512/768/1024)也无妨。
"""
import os
import sys
import json
import numpy as np
import torch
os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EIGHT = os.path.join(os.path.dirname(BASE), "ai面试八股rag", "models")
OTS = os.path.join(EIGHT, "bge-small-zh-v1.5")       # 通用 bge-small(off-the-shelf)
FT = os.path.join(BASE, "models", "bge-small-zh-finance")  # 微调(in-batch 负)
FT_HN = os.path.join(BASE, "models", "bge-small-zh-finance-hn")  # 微调(+难负样本)
M = os.path.join(BASE, "models")
CHUNKS = os.path.join(BASE, "data", "chunks", "chunks.jsonl")
EVAL = os.path.join(BASE, "data", "eval", "eval_finance.jsonl")
PREFIX = "为这个句子生成表示以用于检索相关文章："  # bge 检索指令前缀(gte/m3 不需要)

# (路径, 显示名, query 前缀, 参数量级标注)。前缀按各模型约定：bge 系列加检索指令；gte/bge-m3 不加。
MODELS = [
    (OTS,                          "通用 bge-small-v1.5", PREFIX, "24M"),
    (os.path.join(M, "bge-base-zh-v1.5"), "通用 bge-base-v1.5", PREFIX, "102M"),
    (os.path.join(M, "gte-base-zh"),      "通用 gte-base-zh",   "",     "102M"),
    (os.path.join(M, "bge-m3"),           "通用 bge-m3",        "",     "568M"),
    (FT,                           "微调 bge-small(in-batch)", PREFIX, "24M"),
    (FT_HN,                        "微调 bge-small(+难负)",    PREFIX, "24M"),
]


def evaluate(model_path, name, chunks, ids, evalset, prefix):
    m = SentenceTransformer(model_path)
    cemb = np.asarray(m.encode([c["content"] for c in chunks], normalize_embeddings=True,
                               batch_size=64, show_progress_bar=False), dtype=np.float32)
    id2pos = {cid: i for i, cid in enumerate(ids)}
    docs = [c.get("doc_id", "") for c in chunks]  # 按位置的 doc_id，用于 doc 级召回
    r1 = r5 = mrr = d5 = 0
    for e in evalset:
        qv = m.encode([prefix + e["query"]], normalize_embeddings=True)[0]
        sims = cemb @ qv
        order = np.argsort(-sims)
        gold = id2pos.get(e["gold_chunk_id"], -1)
        rank = int(np.where(order == gold)[0][0]) + 1 if gold >= 0 else 0
        if rank == 1: r1 += 1
        if 1 <= rank <= 5: r5 += 1
        if rank: mrr += 1.0 / rank
        top5_docs = {docs[order[j]] for j in range(min(5, len(order)))}  # 前5块所属文档(doc级召回)
        if e.get("gold_doc_id") and e["gold_doc_id"] in top5_docs:
            d5 += 1
    n = len(evalset)
    print(f"{name:<24}{r1/n*100:>8.1f}%{r5/n*100:>9.1f}%{mrr/n:>9.3f}{d5/n*100:>9.1f}%", flush=True)
    return r1 / n, r5 / n, mrr / n, d5 / n


def main():
    chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8") if l.strip()]
    ids = [c["chunk_id"] for c in chunks]
    evalset = [json.loads(l) for l in open(EVAL, encoding="utf-8") if l.strip()]
    print(f"chunk {len(chunks)} | 留出评测 {len(evalset)} 条 | 纯向量 cosine(不入 ES)\n")
    print(f"{'向量模型(参数量)':<24}{'R@1':>9}{'R@5':>10}{'MRR':>9}{'docR@5':>9}")
    print("-" * 64)
    res = {}
    for path, name, prefix, size in MODELS:
        if not os.path.exists(path):
            print(f"{name}（未找到，跳过）")
            continue
        res[name] = evaluate(path, f"{name}[{size}]", chunks, ids, evalset, prefix)
    print("-" * 56)
    # 横向对比结论：领域微调 small vs 通用大模型
    base = res.get("通用 bge-small-v1.5")
    ft = res.get("微调 bge-small(in-batch)")
    hn = res.get("微调 bge-small(+难负)")
    if base and ft:
        print(f"消融① 通用→in-batch微调(同为 bge-small)：R@1 {base[0]*100:.1f}%→{ft[0]*100:.1f}% "
              f"({(ft[0]-base[0])*100:+.1f}pt) | R@5 {base[1]*100:.1f}%→{ft[1]*100:.1f}% ({(ft[1]-base[1])*100:+.1f}pt) | "
              f"MRR {base[2]:.3f}→{ft[2]:.3f}")
    if ft and hn:
        print(f"消融② in-batch→+难负样本：R@1 {ft[0]*100:.1f}%→{hn[0]*100:.1f}% ({(hn[0]-ft[0])*100:+.1f}pt) | "
              f"R@5 {ft[1]*100:.1f}%→{hn[1]*100:.1f}% ({(hn[1]-ft[1])*100:+.1f}pt) | MRR {ft[2]:.3f}→{hn[2]:.3f}")
    # 横向：微调 small(24M) 对比最强通用大模型
    big = {k: v for k, v in res.items() if k.startswith("通用") and v}
    best_big = max(big.items(), key=lambda kv: kv[1][1]) if big else None  # 按 R@5 取最强通用
    if ft and best_big:
        bn, bv = best_big
        print(f"横向 微调 bge-small(24M) vs 最强通用「{bn}」：R@5 {bv[1]*100:.1f}%→{ft[1]*100:.1f}% "
              f"({(ft[1]-bv[1])*100:+.1f}pt) | R@1 {bv[0]*100:.1f}%→{ft[0]*100:.1f}% ({(ft[0]-bv[0])*100:+.1f}pt) | "
              f"docR@5 {bv[3]*100:.1f}%→{ft[3]*100:.1f}% ({(ft[3]-bv[3])*100:+.1f}pt) ——领域微调小模型能否打过通用大模型")


if __name__ == "__main__":
    main()
