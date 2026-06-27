# -*- coding: utf-8 -*-
"""
make_finance_eval.py — P3 金融检索评测集（释义/反向生成法）

从 chunks.jsonl 抽样若干子块，用 DeepSeek 为每个子块生成一个"金融用户真实会问、且该块能回答"的问题，
gold = 该源子块。覆盖 文本块(事实/观点) 与 表格块(具体数字)。
输出 data/eval/eval_finance.jsonl: {query, gold_chunk_id, gold_doc_id, stock_code, type, section}
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
OUT = os.path.join(BASE, "data", "eval", "eval_finance.jsonl")
N = 200
SEED = 7
# 跳过这些低信息/样板章节
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

PROMPT = """你在模拟一位看研报的投资者，针对下面这段研报内容提一个【真实、具体、该内容能回答】的问题。
要求：问题里点明公司名（%s），像真人提问（可含年份/指标/术语）；只问这段内容能答的；只输出问题本身，不要解释。

研报内容（%s）：
%s"""


def gen_q(stock, sec, content):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT % (stock, sec, content[:1500])}],
            "temperature": 0.7, "stream": False}
    r = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                      json=body, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip().strip('"').strip("「」").strip()


def main():
    if not API_KEY:
        print("✗ 缺 DEEPSEEK_API_KEY"); return
    chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8") if l.strip()]
    # 无泄漏：排除 bge 微调【训练用过】的块(train_pairs 正例)，评测块 gold 与训练集不重叠
    train_ids = set()
    for tp in (os.path.join(BASE, "data", "train", "train_pairs.jsonl"),
               os.path.join(BASE, "data", "train", "train_pairs_hn.jsonl")):
        if os.path.exists(tp):
            for l in open(tp, encoding="utf-8"):
                cid = json.loads(l).get("chunk_id")
                if cid:
                    train_ids.add(cid)
    usable = [c for c in chunks if len(c.get("content", "")) > 30 and not SKIP_SEC.search(c.get("section", ""))
              and c.get("stock_name") and c["chunk_id"] not in train_ids]
    print(f"可用块 {len(usable)}（已排除 {len(train_ids)} 个 bge 训练块，无泄漏）")
    # 按类型分层抽样（放大到 200，覆盖 22 标的/8 行业）
    by_type = {}
    for c in usable:
        by_type.setdefault(c["type"], []).append(c)
    random.seed(SEED)
    sample = []
    quota = {"table": 90, "text": 90, "summary": 20}
    for t, q in quota.items():
        pool = by_type.get(t, [])
        sample += random.sample(pool, min(q, len(pool)))
    random.shuffle(sample)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for i, c in enumerate(sample, 1):
        try:
            q = gen_q(c["stock_name"], c.get("section", ""), c["content"])
            rows.append({"query": q, "gold_chunk_id": c["chunk_id"], "gold_doc_id": c["doc_id"],
                         "stock_code": c.get("stock_code", ""), "type": c["type"], "section": c.get("section", "")})
            print(f"[{i:2}/{len(sample)}|{c['type']}] {q}")
        except Exception as e:
            print(f"[{i}] 失败 {e}")
        time.sleep(0.25)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✓ 金融评测集 {len(rows)} 条 → {OUT}")


if __name__ == "__main__":
    main()
