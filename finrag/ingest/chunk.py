# -*- coding: utf-8 -*-
"""
chunk.py — P2② 父子层级分块（消费 LLM 结构化结果）

读 data/structured/*/*.json（extract_structure.py 产出的干净章节+元数据）→
  data/chunks/parents.jsonl  父块（章节级，粗召回+给上下文）
  data/chunks/chunks.jsonl   子块（段落/表格级，精召回；带 parent_id 回溯）

章节切分已由 LLM 完成（剔除样板、跨模板鲁棒），这里只做：
  - 把每个章节内容切成 ~450 字子块（按句末切，不切断句子）
  - 核心观点单独成块（高价值粗召回）
  - 表格作为子块挂"财务数据"父块
  - 每个块都带元数据（标的/机构/日期/行业/评级），供 ES 过滤检索
"""
import os
import re
import json
import glob

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STRUCT = os.path.join(BASE, "data", "structured")
OUT = os.path.join(BASE, "data", "chunks")
MAX_CHILD = 450

META_KEYS = ["stock_code", "stock_name", "org", "date", "industry", "rating", "title"]


def split_text(t, n=MAX_CHILD):
    parts = re.split(r"(?<=[。！？\n])", t)
    out, buf = [], ""
    for p in parts:
        if buf and len(buf) + len(p) > n:
            out.append(buf.strip())
            buf = ""
        buf += p
    if buf.strip():
        out.append(buf.strip())
    return out


def chunk_doc(s):
    doc_id = f"{s['stock_code']}_{os.path.splitext(s['file'])[0]}"
    meta = {k: s.get(k, "") for k in META_KEYS}
    parents, children = [], []

    def add_parent(pid, section, content):
        parents.append({"parent_id": pid, "doc_id": doc_id, "section": section, "content": content, **meta})

    def add_child(pid, section, ctype, content, ci, **extra):
        children.append({"chunk_id": f"{pid}_c{ci}", "doc_id": doc_id, "parent_id": pid,
                         "section": section, "type": ctype, "content": content, **meta, **extra})

    # 核心观点单独成块
    if s.get("core_view"):
        add_parent(f"{doc_id}_pS", "核心观点", s["core_view"])
        add_child(f"{doc_id}_pS", "核心观点", "summary", s["core_view"], 0)

    # 正文章节 → 父块 + 子块
    for si, sec in enumerate(s.get("sections", [])):
        pid = f"{doc_id}_p{si}"
        add_parent(pid, sec["heading"], sec["content"])
        for ci, piece in enumerate(split_text(sec["content"])):
            add_child(pid, sec["heading"], "text", piece, ci)

    # 表格 → 子块（挂财务数据父块）。可检索内容=LLM摘要；保留 markdown 供展示
    if s.get("tables"):
        pid = f"{doc_id}_pT"
        add_parent(pid, "财务数据/表格", "（见下属各表格）")
        for ti, t in enumerate(s["tables"]):
            if t.get("keep") is False:
                continue
            emb = (t.get("summary") or t.get("caption") or "").strip()
            if not emb:
                continue
            add_child(pid, "财务数据/表格", "table", emb, ti,
                      caption=t.get("caption", ""), table_source=t.get("source", ""),
                      table_type=t.get("table_type", ""),
                      table_markdown=t.get("clean_markdown") or t.get("html", ""))

    return parents, children


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(STRUCT, "*", "*.json")))
    print(f"已结构化研报: {len(files)} 篇")
    all_p, all_c = [], []
    for fp in files:
        s = json.load(open(fp, encoding="utf-8"))
        p, c = chunk_doc(s)
        all_p += p
        all_c += c
    with open(os.path.join(OUT, "parents.jsonl"), "w", encoding="utf-8") as f:
        for x in all_p:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for x in all_c:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    from collections import Counter
    typ = Counter(c["type"] for c in all_c)
    print(f"父块 {len(all_p)} | 子块 {len(all_c)} {dict(typ)} → {OUT}")

    if files:
        s = json.load(open(files[0], encoding="utf-8"))
        p, c = chunk_doc(s)
        print(f"\n=== 示例: {s['stock_name']} / {s['org']} ===")
        print("父块(章节):", [x["section"] for x in p])
        print("子块示例:")
        for ch in c[:5]:
            print(f"  [{ch['section']}|{ch['type']}] {ch.get('caption') or ch['content'][:46]}")


if __name__ == "__main__":
    main()
