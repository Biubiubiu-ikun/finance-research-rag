# -*- coding: utf-8 -*-
"""
ingest.py — P6 新研报增量入库自动管道（编排器）

把 data/reports/{code}/ 下【新增】PDF 一键走完全链路：
  parse_report → extract_structure → table_summary → chunk → index_es
  → 对受影响标的失效下游聚合缓存 → 重新生成该标的的 聚合分析(consensus/views/revisions)

增量靠现有脚本天然支持：parse/structure/table_summary 按 doc 断点续跑(只处理新增)，
chunk/index_es 全量重建(新块自然纳入)。新标的(如生益科技 600183)入库后，UI 数据驱动自动上架。

用法：
  把新研报 PDF 放到 data/reports/{股票代码}/ 下，并在 data/metadata/{code}_page1.csv 补一行元数据，然后：
  python ingest.py            # 自动检测并处理所有新 PDF
  python ingest.py --dry-run  # 只看检测到哪些新 PDF，不执行
"""
import os
import sys
import glob
import subprocess

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable
REPORTS = os.path.join(BASE, "data", "reports")
PARSED = os.path.join(BASE, "data", "parsed")
ANALYSIS = os.path.join(BASE, "data", "analysis")
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # finrag/ 包目录
# 被编排脚本各自所在的分层（重组后按层解析路径）
_SCRIPT_LAYER = {"robust_parse.py": "ingest", "extract_structure.py": "ingest", "table_summary.py": "ingest",
                 "chunk.py": "ingest", "index_es.py": "ingest",
                 "aggregate_forecast.py": "analysis", "aggregate_views.py": "analysis", "track_revisions.py": "analysis"}


def detect_new():
    """返回 (新PDF列表, 受影响code集合)。新 = reports 有 PDF 但 parsed 无对应 json。"""
    new, codes = [], set()
    for pdf in glob.glob(os.path.join(REPORTS, "*", "*.pdf")):
        code = os.path.basename(os.path.dirname(pdf))
        name = os.path.splitext(os.path.basename(pdf))[0]
        if not os.path.exists(os.path.join(PARSED, code, name + ".json")):
            new.append(pdf)
            codes.add(code)
    return new, codes


def invalidate(codes):
    """失效受影响标的的下游聚合缓存(老标的新增研报必须重算；新标的本就无缓存)。"""
    pats = []
    for c in codes:
        pats += [os.path.join(ANALYSIS, "extract", f"{c}.jsonl"),
                 os.path.join(ANALYSIS, "views", f"{c}.jsonl"),
                 os.path.join(ANALYSIS, "views_agg", f"{c}.json"),
                 os.path.join(ANALYSIS, "revisions", f"{c}.jsonl"),
                 os.path.join(ANALYSIS, f"{c}_consensus.md"),
                 os.path.join(ANALYSIS, f"{c}_views.md"),
                 os.path.join(ANALYSIS, f"{c}_revisions.md")]
    n = 0
    for p in pats:
        if os.path.exists(p):
            os.remove(p); n += 1
    print(f"  失效下游缓存 {n} 个")


def run(script, env, args=()):
    print(f"\n{'='*60}\n▶ {script} {' '.join(args)}")
    path = os.path.join(_PKG, _SCRIPT_LAYER[script], script)  # 按层解析脚本绝对路径
    r = subprocess.run([PY, path, *args], cwd=BASE, env=env)
    if r.returncode != 0:
        raise SystemExit(f"✗ {script} 失败(exit {r.returncode})，已中止管道")


def main(dry=False):
    new, codes = detect_new()
    if not new:
        print("没有检测到新研报 PDF（reports 下都已解析）。"); return
    print(f"检测到 {len(new)} 篇新研报，涉及标的代码 {sorted(codes)}：")
    for p in new:
        print("  +", os.path.relpath(p, BASE))
    if dry:
        print("\n--dry-run：仅检测，未执行。"); return

    env = dict(os.environ, PYTHONUTF8="1")  # 防 GBK；其余环境(NLTK_DATA/HF_HUB_OFFLINE)各脚本自管
    # 1~3 步：按 doc 续跑(只处理新增)；4~5 步：全量重建(新块自然纳入)
    run("robust_parse.py", env)       # DeepDOC 解析(重)；外层子进程级超时，卡死 PDF 自动隔离续跑
    run("extract_structure.py", env)  # LLM 切章节+合并CSV元数据
    run("table_summary.py", env)      # LLM 表格摘要
    invalidate(codes)                  # 重建分块/索引前，失效受影响标的的聚合缓存
    run("chunk.py", env)              # 父子分块(全量重写 jsonl)
    run("index_es.py", env)          # 嵌入+入ES(全量重建，默认微调模型，与查询侧一致)
    # 6 步：为受影响标的(重新)生成聚合分析，UI 速览即刻可看
    for c in sorted(codes):
        run("aggregate_forecast.py", env, [c])
        run("aggregate_views.py", env, [c])
        run("track_revisions.py", env, [c])

    print(f"\n{'='*60}\n✅ 入库完成。新标的已可在 Agent / Demo UI 查询")
    print("   Demo UI 若在运行，点侧栏「🔄 刷新数据缓存」即可看到新标的。")


if __name__ == "__main__":
    main(dry="--dry-run" in sys.argv)
