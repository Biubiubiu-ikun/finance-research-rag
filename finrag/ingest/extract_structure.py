# -*- coding: utf-8 -*-
"""
extract_structure.py — P2③ LLM 结构化（章节切分 + 元数据抽取）

读 data/parsed/*/*.json → 用 DeepSeek 做语义级结构化 → data/structured/*/*.json：
  {
    stock_code, stock_name, org, date, industry,      # 机构/日期/标的来自 CSV(可靠)
    rating, target_price, core_view,                  # 来自 LLM
    sections: [{heading, content}],                   # LLM 切章节(剔除样板)，正文按编号从原文重组(忠实)
    tables, figures                                   # 沿用解析阶段
  }

高效+忠实:给 LLM 编号文本块，它只返回"每章起始块编号+标题+是否样板"，
正文由本程序按编号从原文重组，LLM 不重吐全文(省 token、不改写)。
"""
import os
import sys
import re
import json
import glob
import csv
import time
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARSED = os.path.join(BASE, "data", "parsed")
META_DIR = os.path.join(BASE, "data", "metadata")
OUT = os.path.join(BASE, "data", "structured")


def load_env():
    # 复用八股项目的 .env（同一个 DeepSeek key），优先本项目 .env
    for p in [os.path.join(BASE, ".env"),
              os.path.join(os.path.dirname(BASE), "ai面试八股rag", ".env")]:
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


def norm(s):
    return re.sub(r"[\s（）()：:，,。.、《》\"'\-—_]", "", s or "")


def load_csv_meta():
    """{规范化标题: {org,date,stock_name,code,industry,rating}}"""
    m = {}
    for fp in glob.glob(os.path.join(META_DIR, "*.csv")):
        for r in csv.DictReader(open(fp, encoding="utf-8-sig")):
            m[norm(r.get("title"))] = {
                "org": r.get("org_name", ""), "date": (r.get("publish_date", "") or "")[:10],
                "stock_name": r.get("stock_name", ""), "stock_code": r.get("stock_code", ""),
                "industry": r.get("industry_name", ""), "rating": r.get("rating_name", ""),
            }
    return m


PROMPT = """你是金融研报结构化专家。下面是一篇研报按顺序编号的文本块。

任务1·元数据：抽取 投资评级(rating)、目标价(target_price，无则"")、所属行业/赛道(industry)、核心观点(core_view，1~2句话概括)。
任务2·章节划分：给出每个章节的【起始块编号 start_idx】+ 简洁 heading；并判断是否为【样板内容】(免责声明/评级体系说明/分析师简介与执业证书/相关报告列表/法律声明/版权声明等每篇都雷同的内容) → drop=true；正文分析(事件/业绩/盈利预测/投资建议/风险提示等报告特有内容) → drop=false。

只输出 JSON：
{"metadata":{"rating":"","target_price":"","industry":"","core_view":""},
 "sections":[{"start_idx":0,"heading":"...","drop":false}]}

编号文本块：
%s"""


def call_llm(numbered):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": PROMPT % numbered}],
            "response_format": {"type": "json_object"}, "temperature": 0.1, "stream": False}
    r = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}",
                                        "Content-Type": "application/json"}, json=body, timeout=120)
    r.raise_for_status()
    c = r.json()["choices"][0]["message"]["content"].strip()
    c = re.sub(r"^```(?:json)?|```$", "", c, flags=re.M).strip()
    return json.loads(c)


def rebuild_sections(blocks, markers):
    """按 LLM 给的起始编号，从原文重组章节(忠实原文)，剔除样板。"""
    markers = sorted([m for m in markers if isinstance(m.get("start_idx"), int)],
                     key=lambda x: x["start_idx"])
    out = []
    # 第一个标记前的内容当作"摘要"
    first = markers[0]["start_idx"] if markers else len(blocks)
    if first > 1:
        txt = "\n".join(b["text"] for b in blocks[1:first]).strip()
        if txt:
            out.append({"heading": "摘要", "content": txt})
    for i, m in enumerate(markers):
        if m.get("drop"):
            continue
        s = max(0, m["start_idx"])
        e = markers[i + 1]["start_idx"] if i + 1 < len(markers) else len(blocks)
        txt = "\n".join(b["text"] for b in blocks[s:e]).strip()
        if txt:
            out.append({"heading": m.get("heading", "正文"), "content": txt})
    return out


def process(doc, csv_meta):
    blocks = doc["blocks"]
    numbered = "\n".join(f'{i}: {b["text"][:180]}' for i, b in enumerate(blocks))[:28000]
    res = call_llm(numbered)
    sections = rebuild_sections(blocks, res.get("sections", []))
    md = res.get("metadata", {})
    cm = csv_meta.get(norm(os.path.splitext(doc["file"])[0])) or csv_meta.get(norm(doc["title"])) or {}
    return {
        "stock_code": cm.get("stock_code") or doc["stock_code"],
        "stock_name": cm.get("stock_name", ""),
        "org": cm.get("org", ""), "date": cm.get("date", ""),
        "industry": cm.get("industry") or md.get("industry", ""),
        "rating": cm.get("rating") or md.get("rating", ""),
        "target_price": md.get("target_price", ""),
        "core_view": md.get("core_view", ""),
        "title": doc["title"], "file": doc["file"],
        "sections": sections, "tables": doc.get("tables", []), "figures": doc.get("figures", []),
    }


def main():
    if not API_KEY:
        print("✗ 缺 DEEPSEEK_API_KEY"); return
    csv_meta = load_csv_meta()
    files = sorted(glob.glob(os.path.join(PARSED, "*", "*.json")))
    print(f"待结构化: {len(files)} 篇（已加载 {len(csv_meta)} 条 CSV 元数据）\n")
    done = 0
    for fp in files:
        code = os.path.basename(os.path.dirname(fp))
        name = os.path.splitext(os.path.basename(fp))[0]
        out = os.path.join(OUT, code, name + ".json")
        if os.path.exists(out):
            continue
        try:
            doc = json.load(open(fp, encoding="utf-8"))
            s = process(doc, csv_meta)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            json.dump(s, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            done += 1
            print(f"[{done}] {s['stock_name']}({code}) | {s['org']} {s['date']} | 评级:{s['rating']} | 章节{len(s['sections'])}")
        except Exception as e:
            print(f"✗ {code}/{name[:24]}: {repr(e)[:160]}")
        time.sleep(0.3)
    print(f"\n完成 {done} 篇 → {OUT}")

    # 验证：打印第一篇结构
    sf = sorted(glob.glob(os.path.join(OUT, "*", "*.json")))
    if sf:
        s = json.load(open(sf[0], encoding="utf-8"))
        print(f"\n=== 示例: {s['stock_name']} {s['title'][:24]} ===")
        print(f"机构 {s['org']} | 日期 {s['date']} | 行业 {s['industry']} | 评级 {s['rating']} | 目标价 {s['target_price']}")
        print(f"核心观点: {s['core_view']}")
        print("保留章节:")
        for sec in s["sections"]:
            print(f"  ▸ {sec['heading']}  ({len(sec['content'])}字)")


if __name__ == "__main__":
    main()
