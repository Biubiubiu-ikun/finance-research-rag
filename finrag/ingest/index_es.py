# -*- coding: utf-8 -*-
"""
index_es.py — P2⑤ 把子块写入 Elasticsearch（嵌入 + 元数据可过滤字段）

每个子块索引：
  - 元数据(keyword)：stock_code/stock_name/org/date/industry/rating/section/type → 支持【前置过滤】
  - content_tokens(text, jieba 分词)→ 支持 BM25 稀疏召回
  - embedding(dense_vector, bge)→ 支持向量稠密召回(ES kNN)
  - content / table_markdown → 展示用
为 P3「元数据过滤 + BM25/向量多路召回 + Rerank」打底。

注：bge-small-zh 是【通用】中文模型(非金融专精)，先做基线；P3 评测后再决定是否换 bge-base / 领域微调。
"""
import os
import sys
import json
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
from elasticsearch import Elasticsearch, helpers

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 默认用领域微调 bge-small-zh-finance（P3 评测证明优于通用，且现有 ES 即用它嵌入，须保持一致）。
# BGE_MODEL 环境变量可覆盖。⚠️重建索引务必与查询侧(retrieve_es)同模型，否则向量空间错乱。
MODEL_PATH = os.path.join(BASE, "models", "bge-small-zh-finance")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(BASE), "ai面试八股rag", "models", "bge-small-zh-v1.5")
if os.getenv("BGE_MODEL"):  # 用 BGE_MODEL=models/bge-small-zh-finance 切换到微调模型
    MODEL_PATH = os.getenv("BGE_MODEL")
CHUNKS = os.path.join(BASE, "data", "chunks", "chunks.jsonl")
INDEX = "research_reports"
DIM = 512

ES = Elasticsearch(os.getenv("ES_URL", "http://localhost:9200"), basic_auth=("elastic", "infini_rag_flow"),
                   verify_certs=False, request_timeout=60)

MAPPING = {
    "mappings": {"properties": {
        "chunk_id": {"type": "keyword"}, "doc_id": {"type": "keyword"}, "parent_id": {"type": "keyword"},
        "stock_code": {"type": "keyword"}, "stock_name": {"type": "keyword"}, "org": {"type": "keyword"},
        "date": {"type": "keyword"}, "industry": {"type": "keyword"}, "rating": {"type": "keyword"},
        "section": {"type": "keyword"}, "type": {"type": "keyword"}, "title": {"type": "text"},
        "content": {"type": "text"},
        "content_tokens": {"type": "text", "analyzer": "whitespace"},
        "table_markdown": {"type": "text", "index": False},
        "embedding": {"type": "dense_vector", "dims": DIM, "index": True, "similarity": "cosine"},
    }}
}


def main():
    print("加载 bge...")
    model = SentenceTransformer(MODEL_PATH)
    chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8") if l.strip()]
    print(f"子块 {len(chunks)} 个")

    # 嵌入（批量，GPU）
    texts = [c["content"] for c in chunks]
    emb = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)

    # 重建索引
    if ES.indices.exists(index=INDEX):
        ES.indices.delete(index=INDEX)
    ES.indices.create(index=INDEX, body=MAPPING)

    actions = []
    for c, v in zip(chunks, emb):
        actions.append({"_index": INDEX, "_id": c["chunk_id"], "_source": {
            "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "parent_id": c.get("parent_id", ""),
            "stock_code": c.get("stock_code", ""), "stock_name": c.get("stock_name", ""),
            "org": c.get("org", ""), "date": c.get("date", ""), "industry": c.get("industry", ""),
            "rating": c.get("rating", ""), "section": c.get("section", ""), "type": c.get("type", ""),
            "title": c.get("title", ""), "content": c["content"],
            "content_tokens": " ".join(tokenize(c["content"])),
            "table_markdown": c.get("table_markdown", ""),
            "embedding": [float(x) for x in v],
        }})
    helpers.bulk(ES, actions)
    ES.indices.refresh(index=INDEX)
    print(f"✓ 已写入 ES 索引 {INDEX}: {ES.count(index=INDEX)['count']} 条")

    # 自测：向量召回 + 元数据过滤
    q = "北方华创2025年营收和净利润预测是多少"
    qv = model.encode(["为这个句子生成表示以用于检索相关文章：" + q], normalize_embeddings=True)[0]
    res = ES.search(index=INDEX, knn={"field": "embedding", "query_vector": [float(x) for x in qv],
                                      "k": 5, "num_candidates": 50,
                                      "filter": [{"term": {"stock_code": "002371"}}]},
                    source=["stock_name", "org", "section", "type", "content"])
    print(f"\n=== 自测：{q}（过滤 stock=002371）===")
    for h in res["hits"]["hits"]:
        s = h["_source"]
        print(f"  [{s['org']}|{s['section']}|{s['type']}] {s['content'][:70]}")


if __name__ == "__main__":
    main()
