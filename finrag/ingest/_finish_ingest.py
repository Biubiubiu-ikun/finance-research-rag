# -*- coding: utf-8 -*-
"""一次性驱动：绕过 ingest.detect_new(它只看 parsed，会把"已解析未入库"误判为完成)，
强制跑完 结构化→分块→写ES→重算聚合。用于补跑老 9 标的 2024 入库。跑完可删。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import finrag.ingest.ingest as ingest
CODES = ["688981", "002371", "603501", "603986", "600183", "300750", "002594", "300274", "601012"]
env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")

ingest.run("robust_parse.py", env)        # 已解析的跳过；坏 PDF 已隔离
ingest.run("extract_structure.py", env)   # 结构化新文档(按 doc 续跑)
ingest.run("table_summary.py", env)       # 表格摘要
ingest.invalidate(set(CODES))             # 失效老 9 标的聚合缓存
ingest.run("chunk.py", env)               # 父子分块(全量重写)
ingest.run("index_es.py", env)            # 嵌入+写 ES(全量重建，2024 随之入库)
for c in CODES:                            # 重算受影响标的聚合
    ingest.run("aggregate_forecast.py", env, [c])
    ingest.run("aggregate_views.py", env, [c])
    ingest.run("track_revisions.py", env, [c])
print("\n==== FINISH_INGEST DONE ====")
