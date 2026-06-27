# -*- coding: utf-8 -*-
"""
table_summary.py — P2④ 表格→自然语言摘要 + 密集表列重构（DeepSeek）

为什么：HTML 表格本身没法做语义检索（"宁德营收预测"匹配不到 <table>）。
对每张表用 LLM 产出：
  - summary:        1~3句话(这张表是什么+关键数字)，作为该表子块的【可检索文本】
  - clean_markdown: 干净 markdown；pdfplumber 抽的列错位/数字被切开的在此【修复对齐】（不改数值）
  - table_type:     盈利预测/资产负债表/利润表/现金流量表/估值/市场数据/其他
  - keep:           图表OCR碎片/空表 → false（丢弃）

把结果写回 data/structured/*/*.json 的每个 table（按 summary 是否存在做断点续跑）。
"""
import os
import sys
import re
import json
import glob
import time
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STRUCT = os.path.join(BASE, "data", "structured")


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

PROMPT = """你是金融研报表格分析专家。下面是研报「%s — %s」中的一张表格（HTML，可能有 OCR 导致的列错位/数字被切开/多表并排）。

请输出 JSON：
{
 "keep": true/false,
 "table_type": "盈利预测/资产负债表/利润表/现金流量表/估值/市场数据/其他",
 "clean_markdown": "整理成干净的 markdown 表格。若列错位或数字被切开（例如 '20|25E' 实为 '2025E'，'491|75' 实为 '49175'）依据上下文修复对齐；并排的多张表请拆成多个表。【严禁编造或改动任何数值，只做对齐/拆分修复】。",
 "summary": "1~3句话：这张表是什么 + 关键数字/结论（如各年营收/净利预测及增速、目标价、PE 等），写成适合检索的自然语言。"
}
若该内容是图表OCR碎片/空表/无意义，keep=false 且其余留空。

表格 HTML：
%s"""


def summarize(stock, title, html):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT % (stock, title, html[:6000])}],
            "response_format": {"type": "json_object"}, "temperature": 0.1, "stream": False}
    r = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                      json=body, timeout=120)
    r.raise_for_status()
    c = r.json()["choices"][0]["message"]["content"].strip()
    c = re.sub(r"^```(?:json)?|```$", "", c, flags=re.M).strip()
    return json.loads(c)


def validate():
    """在 高研发 这篇上验证：盈利预测表摘要 + 资产负债表列重构。"""
    fp = os.path.join(STRUCT, "002371", "高研发与强激励共筑护城河.json")
    s = json.load(open(fp, encoding="utf-8"))
    print(f"验证: {s['stock_name']} / {s['org']}  共 {len(s['tables'])} 张表\n")
    for t in s["tables"]:
        if t["source"] == "deepdoc" and re.search(r"营业收入|净利|指标", t["html"]):
            print("【盈利预测表（DeepDOC）】")
            r = summarize(s["stock_name"], s["title"], t["html"])
            print("  type:", r.get("table_type"), "| keep:", r.get("keep"))
            print("  summary:", r.get("summary"))
            break
    for t in s["tables"]:
        if t["source"] == "pdfplumber":
            print("\n【密集财务全表（pdfplumber，列错位）→ LLM 重构】")
            r = summarize(s["stock_name"], s["title"], t["html"])
            print("  type:", r.get("table_type"), "| keep:", r.get("keep"))
            print("  summary:", r.get("summary"))
            print("  clean_markdown(前700字):\n" + (r.get("clean_markdown", "")[:700]))
            break


def main():
    files = sorted(glob.glob(os.path.join(STRUCT, "*", "*.json")))
    n = 0
    for fp in files:
        s = json.load(open(fp, encoding="utf-8"))
        changed = False
        for t in s.get("tables", []):
            if "summary" in t:
                continue
            try:
                r = summarize(s["stock_name"], s["title"], t["html"])
                t["summary"] = r.get("summary", "")
                t["clean_markdown"] = r.get("clean_markdown", "")
                t["table_type"] = r.get("table_type", "")
                t["keep"] = r.get("keep", True)
                changed = True
                n += 1
            except Exception as e:
                print(f"✗ {s['stock_name']} 表: {repr(e)[:120]}")
            time.sleep(0.2)
        if changed:
            json.dump(s, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"[{s['stock_name']}/{s['org']}] {len(s['tables'])} 表已摘要")
    print(f"\n本次新增摘要 {n} 张表")


if __name__ == "__main__":
    validate() if len(sys.argv) > 1 and sys.argv[1] == "validate" else main()
