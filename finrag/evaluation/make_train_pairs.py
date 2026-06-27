# -*- coding: utf-8 -*-
"""
make_train_pairs.py — P3微调① 造训练对（query↔正确chunk）

从 chunks.jsonl 采样子块（【排除评测集 gold chunk，防数据泄漏】），
用 DeepSeek 每块生成 3 个不同角度、投资者真实会问、该块能回答的问题，
得到 (query, positive=chunk内容) 对，用于对比学习微调 bge。
输出 data/train/train_pairs.jsonl: {query, positive, chunk_id, type}
断点续跑：跳过已生成过的 chunk_id。
"""
import os
import sys
import re
import json
import random
import time
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNKS = os.path.join(BASE, "data", "chunks", "chunks.jsonl")
EVAL = os.path.join(BASE, "data", "eval", "eval_finance.jsonl")
OUT = os.path.join(BASE, "data", "train", "train_pairs.jsonl")
N_CHUNKS = 130
SEED = 11
SKIP_SEC = re.compile(r"标题与基本信息|市场数据|分析师|相关研究|相关报告")


def load_env():
    for p in [os.path.join(BASE, ".env"), os.path.join(os.path.dirname(BASE), "ai面试八股rag", ".env")]:
        if os.path.exists(p):
            for line in open(p, encoding="utf-8-sig"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")
            return


load_env()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
API_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions"
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

PROMPT = """为下面这段研报内容，生成 3 个【不同角度、投资者真实会问、且该内容能回答】的问题。
要求：每个问题点明公司名（%s）；问法多样（数字型/观点型/对比型均可）；只输出 JSON。
格式：{"questions":["...","...","..."]}

研报内容（%s）：
%s"""


def gen_qs(stock, sec, content):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT % (stock, sec, content[:1500])}],
            "response_format": {"type": "json_object"}, "temperature": 0.8, "stream": False}
    r = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                      json=body, timeout=60)
    r.raise_for_status()
    c = r.json()["choices"][0]["message"]["content"].strip()
    c = re.sub(r"^```(?:json)?|```$", "", c, flags=re.M).strip()
    return json.loads(c).get("questions", [])


def main():
    if not API_KEY:
        print("✗ 缺 DEEPSEEK_API_KEY"); return
    gold = set(json.loads(l)["gold_chunk_id"] for l in open(EVAL, encoding="utf-8") if l.strip())
    chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8") if l.strip()]
    usable = [c for c in chunks if len(c.get("content", "")) > 30 and c.get("stock_name")
              and not SKIP_SEC.search(c.get("section", "")) and c["chunk_id"] not in gold]
    by_type = {}
    for c in usable:
        by_type.setdefault(c["type"], []).append(c)
    random.seed(SEED)
    sample = []
    for t, q in {"table": 55, "text": 60, "summary": 15}.items():
        pool = by_type.get(t, [])
        sample += random.sample(pool, min(q, len(pool)))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        done = set(json.loads(l)["chunk_id"] for l in open(OUT, encoding="utf-8") if l.strip())
    sample = [c for c in sample if c["chunk_id"] not in done]
    print(f"待处理 {len(sample)} 块（已生成 {len(done)}）")

    n = 0
    with open(OUT, "a", encoding="utf-8") as f:
        for i, c in enumerate(sample, 1):
            try:
                for q in gen_qs(c["stock_name"], c.get("section", ""), c["content"]):
                    f.write(json.dumps({"query": q, "positive": c["content"],
                                        "chunk_id": c["chunk_id"], "type": c["type"]}, ensure_ascii=False) + "\n")
                    n += 1
                if i % 20 == 0:
                    print(f"  {i}/{len(sample)} 块，累计 {n} 对")
            except Exception as e:
                print(f"  [{i}] 失败 {repr(e)[:100]}")
            time.sleep(0.2)
    print(f"\n✓ 新增 {n} 对 → {OUT}")


if __name__ == "__main__":
    main()
