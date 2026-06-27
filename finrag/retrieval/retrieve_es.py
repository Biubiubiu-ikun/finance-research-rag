# -*- coding: utf-8 -*-
"""
retrieve_es.py — P3 对 ES 的混合检索

  - 向量召回：ES kNN（bge query 向量）
  - 关键词召回：BM25（content_tokens，jieba 分词的 query）
  - 混合：RRF 融合两路排名
  - 元数据前置过滤：stock_code / date / industry / org / type ...
  - 精排：bge-reranker（复用八股项目的模型）

复用八股 bge / bge-reranker（通用模型；P3 评测后再决定是否上 finance 词典 / bge-base / 微调）。
"""
import os
import sys
import torch
if hasattr(os, "add_dll_directory"):  # 仅 Windows 需手动加载 torch DLL 目录；Linux 容器无此 API
    os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from finrag.retrieval.finance_tokenize import tokenize
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EIGHT = os.path.join(os.path.dirname(BASE), "ai面试八股rag", "models")
# 默认用领域微调 bge-small-zh-finance —— 当前 ES 即用它嵌入(实测 cosine=1.0)，查询向量必须同空间，
# 否则向量召回失真(通用模型与库向量 cosine 仅 0.83)。BGE_MODEL 环境变量可覆盖。
MODEL_PATH = os.path.join(BASE, "models", "bge-small-zh-finance")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(EIGHT, "bge-small-zh-v1.5")
if os.getenv("BGE_MODEL"):  # 切换到微调模型
    MODEL_PATH = os.getenv("BGE_MODEL")
RERANKER_PATH = os.path.join(BASE, "models", "bge-reranker-base")
if not os.path.exists(RERANKER_PATH):
    RERANKER_PATH = os.path.join(EIGHT, "bge-reranker-base")
# 默认用通用 reranker：与 bge 嵌入不同，reranker 不与 ES 索引强耦合(切换不会向量空间错乱)，
# 故保守默认通用。领域微调 reranker(eval_reranker 实测 chunk R@1/R@5/MRR 更优、docR@5 略让步)
# 可按需启用：RERANKER_MODEL=models/bge-reranker-finance
if os.getenv("RERANKER_MODEL"):
    RERANKER_PATH = os.getenv("RERANKER_MODEL")
INDEX = "research_reports"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
SRC = ["chunk_id", "doc_id", "stock_name", "org", "date", "section", "type", "content"]


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_PATH)
        self.es = Elasticsearch(os.getenv("ES_URL", "http://localhost:9200"), basic_auth=("elastic", "infini_rag_flow"),
                                verify_certs=False, request_timeout=60)
        self._reranker = None

    def _filters(self, f, date_range=None):
        clauses = [{"term": {k: v}} for k, v in (f or {}).items() if v]
        if date_range:  # 时间感知检索：date 为 keyword 且 ISO(YYYY-MM-DD)，字典序=时间序，range 直接可用
            start, end = date_range
            rng = {}
            if start:
                rng["gte"] = start
            if end:
                rng["lte"] = end
            if rng:
                clauses.append({"range": {"date": rng}})
        return clauses

    def _vector(self, q, n, f, date_range=None):
        qv = self.model.encode([QUERY_PREFIX + q], normalize_embeddings=True)[0]
        res = self.es.search(index=INDEX, source=SRC, size=n,
                             knn={"field": "embedding", "query_vector": [float(x) for x in qv],
                                  "k": n, "num_candidates": max(100, n * 5), "filter": self._filters(f, date_range)})
        return [(h["_id"], h["_source"]) for h in res["hits"]["hits"]]

    def _bm25(self, q, n, f, date_range=None):
        toks = " ".join(tokenize(q))
        body = {"bool": {"must": {"match": {"content_tokens": toks}}, "filter": self._filters(f, date_range)}}
        res = self.es.search(index=INDEX, source=SRC, size=n, query=body)
        return [(h["_id"], h["_source"]) for h in res["hits"]["hits"]]

    @staticmethod
    def _rrf(lists, weights=None, k=60):
        weights = weights or [1.0] * len(lists)
        fused, store = {}, {}
        for w, lst in zip(weights, lists):
            for rank, (cid, src) in enumerate(lst, 1):
                fused[cid] = fused.get(cid, 0) + w / (k + rank)
                store[cid] = src
        order = sorted(fused, key=fused.get, reverse=True)
        return [(cid, store[cid]) for cid in order]

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANKER_PATH)
        return self._reranker

    def _rerank(self, q, items):
        ce = self._get_reranker()
        scores = ce.predict([(q, s["content"]) for _, s in items])
        order = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
        return [items[i] for i in order]

    def search(self, q, k=5, method="hybrid", filters=None, rerank=False, pool=50, weights=(0.3, 1.0), date_range=None):
        n = max(k, pool) if rerank else (pool if method == "hybrid" else k)
        if method == "vector":
            items = self._vector(q, n, filters, date_range)
        elif method == "bm25":
            items = self._bm25(q, n, filters, date_range)
        else:  # 加权 RRF：BM25 在金融上更强，给更高权重
            items = self._rrf([self._vector(q, pool, filters, date_range), self._bm25(q, pool, filters, date_range)], weights=list(weights))
        if rerank and items:  # 空结果(如时间过滤后无命中)不进 rerank，避免 CrossEncoder.predict([]) 崩
            items = self._rerank(q, items[:pool])
        return items[:k]


if __name__ == "__main__":
    r = Retriever()
    for q in ["宁德时代2025年营收和净利润预测", "北方华创的投资评级和目标价", "比亚迪海外销量怎么样"]:
        print("\n" + "=" * 60 + f"\n问题: {q}")
        for cid, s in r.search(q, k=3, method="hybrid", rerank=True):
            print(f"  [{s['stock_name']}|{s['org']}|{s['section']}|{s['type']}] {s['content'][:60]}")
