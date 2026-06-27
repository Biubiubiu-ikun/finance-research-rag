# -*- coding: utf-8 -*-
"""
parse_report.py — P2① 研报结构化解析

把一篇研报 PDF 解析成结构化中间表示（存 data/parsed/<code>/<name>.json）：
  {
    file, stock_code, title,
    blocks:  [{type, text, page}]      有类型的内容块（title/text/...），已清洗模板标签
    tables:  [{caption, html, source}] 表格：DeepDOC 抽主表 + pdfplumber-text 抽密集全表
    figures: [{caption}]               图表：只留标题作定位（正文丢弃）
  }

表格混合策略：
  - DeepDOC 抽常规表（盈利预测/关键指标，干净）；跳过它糊在一起的"财务全表"
  - pdfplumber text 策略 抽无边框的密集财务全表（资产负债表/利润表/现金流量表）
  （列毛糙问题留给 P2④ 的 LLM 重构）
"""
import os
import sys
import re
import json
import glob
import time
import torch
os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SWXY = r"D:\代码随想录大模型\工业级rag项目\swxy\swxy"
sys.path.insert(0, os.path.join(SWXY, "backend", "app"))
os.environ.setdefault("NLTK_DATA", os.path.join(SWXY, "backend", "nltk_data"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pdfplumber
from service.core.deepdoc.parser.pdf_parser import RAGFlowPdfParser

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(BASE, "data", "parsed")
PER_PROC = 20  # 单进程解析篇数上限：达到即正常退出，由 robust_parse 重启新进程(释放内存/显存，防大批量累积 OOM)


def clean_tag(s):
    """去券商模板标签 [Table_xxx]（含 OCR 交错型），保留中文。"""
    if not s:
        return ""
    return re.sub(r"\[[^\]\n]{0,40}\]", lambda m: re.sub(r"[A-Za-z_\[\]]+", "", m.group()), s)


def dummy(prog=None, msg=""):
    pass


class Pdf(RAGFlowPdfParser):
    def parse(self, filename, to_page=100000, zoomin=3):
        self.__images__(filename, zoomin, 0, to_page, dummy)
        self._layouts_rec(zoomin)
        self._table_transformer_job(zoomin)
        self._text_merge()
        tbls = self._extract_table_figure(True, zoomin, True, True)
        self._concat_downward()
        blocks = []
        for b in self.boxes:
            t = clean_tag(b.get("text", "")).strip()
            if not t:
                continue
            blocks.append({"type": b.get("layout_type", "") or "text",
                           "text": t,
                           "page": int(b.get("page_number", 0) or 0)})
        return blocks, tbls


def split_tables_figures(tbls):
    tables, figures = [], []
    for tb in tbls:
        try:
            (img, content), pos = tb
        except Exception:
            continue
        if isinstance(content, str) and content.strip().lower().startswith("<table"):
            html = clean_tag(content)
            head = html[:160]
            # 跳过 DeepDOC 糊在一起的财务全表（交给 pdfplumber）
            if "资产负债表" in head and "利润表" in head:
                continue
            cap_m = re.search(r"<caption>(.*?)</caption>", html, re.S)
            caption = clean_tag(cap_m.group(1)).strip() if cap_m else ""
            tables.append({"caption": caption, "html": html, "source": "deepdoc"})
        else:
            cap = content if isinstance(content, str) else (" ".join(map(str, content)) if isinstance(content, list) else "")
            cap = clean_tag(cap).strip()[:80]
            if cap:
                figures.append({"caption": cap})
    return tables, figures


def rows_to_html(rows):
    h = "<table>"
    for r in rows:
        h += "<tr>" + "".join(f"<td>{(c or '').strip()}</td>" for c in r) + "</tr>"
    return h + "</table>"


def pdfplumber_statements(fp):
    """用 text 策略抽无边框的密集财务全表。"""
    ts = {"vertical_strategy": "text", "horizontal_strategy": "text", "snap_tolerance": 4}
    out = []
    with pdfplumber.open(fp) as pdf:
        for pi, page in enumerate(pdf.pages, 1):
            txt = page.extract_text() or ""
            if not re.search(r"资产负债表|利润表|现金流量表", txt):
                continue
            for rows in page.extract_tables(ts):
                if rows and len(rows) >= 5:
                    flat = " ".join(str(c) for r in rows for c in r if c)
                    if re.search(r"资产负债表|流动资产|营业收入", flat):
                        out.append({"caption": "财务报表（资产负债/利润/现金流，待LLM重构）",
                                    "html": rows_to_html(rows), "source": "pdfplumber", "page": pi})
    return out


def parse_one(fp, parser):
    code = os.path.basename(os.path.dirname(fp))
    name = os.path.splitext(os.path.basename(fp))[0]
    blocks, tbls = parser.parse(fp)
    tables, figures = split_tables_figures(tbls)
    tables += pdfplumber_statements(fp)
    title = next((b["text"] for b in blocks if b["type"] == "title"), name)
    return {"file": os.path.basename(fp), "stock_code": code, "title": title,
            "blocks": blocks, "tables": tables, "figures": figures}


def main():
    print("加载 DeepDOC 模型...")
    parser = Pdf()  # 模型只加载一次
    pdfs = sorted(glob.glob(os.path.join(BASE, "data", "reports", "*", "*.pdf")))
    print(f"共 {len(pdfs)} 篇研报\n")
    done = skip = fail = 0
    for fp in pdfs:
        code = os.path.basename(os.path.dirname(fp))
        name = os.path.splitext(os.path.basename(fp))[0]
        out = os.path.join(OUT_DIR, code, name + ".json")
        if os.path.exists(out):  # 断点续跑
            skip += 1
            continue
        try:
            t0 = time.time()
            doc = parse_one(fp, parser)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            json.dump(doc, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            done += 1
            print(f"[{done}] {code}/{name[:26]}  块{len(doc['blocks'])} 表{len(doc['tables'])} 图{len(doc['figures'])}  {time.time()-t0:.0f}s", flush=True)
            if done >= PER_PROC:  # 达单进程上限 → 正常退出，由外层 robust_parse 重启新进程(释放内存)
                print(f"达单进程上限 {PER_PROC} 篇，退出让外层重启续跑", flush=True)
                break
        except Exception as e:
            fail += 1
            print(f"✗ {code}/{name[:26]}: {repr(e)[:150]}", flush=True)
    print(f"\n完成 {done} 篇，跳过(已存在) {skip} 篇，失败 {fail} 篇 → {OUT_DIR}")


if __name__ == "__main__":
    main()
