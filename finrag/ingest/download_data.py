# -*- coding: utf-8 -*-
"""
download_data.py — 研报采集脚本（封装 eastmoney CLI；补 HANDOFF 提到但此前从不存在的脚本）

按 watchlist.json 批量采集券商研报 → data/reports/{code}/*.pdf + data/metadata/{code}_page1.csv，
供 ingest.py 入库。两种模式：
  --full         首采：已有标的爬 since_existing(2026) 起；新标的【分年】爬 since_new(2024)~今(覆盖纵向)
  --incremental  增量：每个标的从 上次水位线-缓冲 到今天（定时任务用）

工业 ETL 要点：
  - 绕代理：东财是【国内】API，走代理会被转海外失败(akshare 同款坑) → 清空 HTTP(S)_PROXY + NO_PROXY=*
  - 幂等去重：每篇研报有唯一 info_code；同名 PDF 覆盖 + ingest 断点续跑只处理新增 → 可重复跑
  - 质量门：下载后按 PDF 页数 >= MIN_PAGES 过滤(滤掉短快评，保留深度报告)
  - 水位线：data/crawl_state.json 记每个 code 的 last_run / seen_info_codes，增量据此算 begin

用法：
  python download_data.py --full                          # 全量首采(23 标的)
  python download_data.py --full --codes 600519,600276    # 只采指定标的(试点)
  python download_data.py --incremental                   # 定时增量(配 schtasks)
  python download_data.py --full --codes 600519 --ingest  # 采完顺带触发 ingest 入库
  python download_data.py --full --dry-run                # 只查列表不下载(预估规模)
"""
import os
import sys
import csv
import json
import glob
import argparse
import subprocess
from datetime import date, timedelta

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(BASE, "data", "reports")
META_DIR = os.path.join(BASE, "data", "metadata")
STATE_FP = os.path.join(BASE, "data", "crawl_state.json")
WATCHLIST = os.path.join(BASE, "watchlist.json")
EM = os.path.join(os.path.dirname(sys.executable), "Scripts", "eastmoney.exe")
if not os.path.exists(EM):
    EM = "eastmoney"  # 退回 PATH

CSV_COLS = ["title", "org_name", "publish_date", "stock_name", "stock_code",
            "industry_name", "rating_name", "info_code", "url"]
MIN_PAGES = 4          # 质量门地板：少于该页数(纯快讯/评级调整通知,现有库中位5页)丢弃
PER_YEAR_QUERY = 8     # 新标的【每年】查的候选数
PER_YEAR_KEEP = 4      # 新标的【每年】保留数(段内按页数取 top)——逐年配额→保证 2024-2026 每年都覆盖
EXISTING_SIZE = 15     # 已有标的单段(2026~今)查的候选数
EXISTING_KEEP = 10     # 已有标的单段保留数
INCR_SIZE = 15         # 增量单段查的候选数
INCR_KEEP = 10         # 增量单段保留数
OVERLAP_DAYS = 7       # 增量回看缓冲(防边界漏报)


# ----------------------------------------------------------------- CLI 封装(绕代理)
def cli_env():
    env = {k: v for k, v in os.environ.items()
           if k.lower() not in ("http_proxy", "https_proxy", "all_proxy")}
    env.update(NO_PROXY="*", no_proxy="*", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    return env


def run_cli(args, timeout=300):
    last = None
    for attempt in range(2):  # 网络抖动重试一次
        try:
            return subprocess.run([EM] + args, env=cli_env(), capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.SubprocessError as e:
            last = e
    raise last


def query(code, begin, end, size, tmp):
    """查询某标的某时段研报列表(含 info_code)。返回 dict 行列表。"""
    run_cli(["q", "-t", "stock", "-c", code, "--begin", begin, "--end", end,
             "-s", str(size), "--save-csv", "-o", tmp])
    fp = os.path.join(tmp, f"{code}_page1.csv")
    if not os.path.exists(fp):
        return []
    with open(fp, encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if r.get("info_code")]


def download(code, begin, end, size):
    """下载某标的某时段研报 PDF 到 data/reports/{code}/。"""
    run_cli(["d", "-t", "stock", "-c", code, "--begin", begin, "--end", end,
             "-s", str(size), "-o", REPORTS_DIR])


def pdf_pages(path):
    """数 PDF 页数(质量门用)；打不开返回 -1(保留，交给后续 parse 处理)。"""
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return len(pdf.pages)
    except Exception:
        return -1


# ----------------------------------------------------------------- 状态/元数据
def load_json(fp, default):
    return json.load(open(fp, encoding="utf-8")) if os.path.exists(fp) else default


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FP), exist_ok=True)
    json.dump(state, open(STATE_FP, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def merge_metadata(code, new_rows):
    """把本次 query 的行并入 data/metadata/{code}_page1.csv，按 info_code 去重(新覆盖旧)。"""
    os.makedirs(META_DIR, exist_ok=True)
    fp = os.path.join(META_DIR, f"{code}_page1.csv")
    by_ic = {}
    if os.path.exists(fp):
        with open(fp, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("info_code"):
                    by_ic[r["info_code"]] = r
    for r in new_rows:
        by_ic[r["info_code"]] = {k: r.get(k, "") for k in CSV_COLS}
    with open(fp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(by_ic.values())
    return len(by_ic)


# ----------------------------------------------------------------- 采集单个标的
def segments(s, mode, state, today):
    """返回该标的要采集的 (begin, end, query_size, keep) 时段列表。
    新标的【按年分段 + 逐年配额】→ 保证 2024-2026 每年都覆盖(不会被某年长报告挤光)。"""
    since_existing = WL["since_existing"]
    since_new = WL["since_new"]
    if mode == "incremental":
        base = since_existing if s["existing"] else since_new
        prev = state.get(s["code"], {}).get("last_run")
        begin = base
        if prev:
            begin = max(base, (date.fromisoformat(prev) - timedelta(days=OVERLAP_DAYS)).isoformat())
        return [(begin, today, INCR_SIZE, INCR_KEEP)]
    # full：按年分段 + 逐年配额（existing 用 since_existing、new 用 since_new；两者同逻辑），
    # 保证 since~今年【每一年】都被覆盖，不会被某年的长报告把配额挤光（单段查只会拿到最近年份）。
    since = since_existing if s["existing"] else since_new
    segs = []
    for yr in range(int(since[:4]), int(today[:4]) + 1):
        b = max(since, f"{yr}-01-01")
        e = min(today, f"{yr}-12-31")
        segs.append((b, e, PER_YEAR_QUERY, PER_YEAR_KEEP))
    return segs


def collect_one(s, mode, state, today, dry):
    code, name = s["code"], s["name"]
    rdir = os.path.join(REPORTS_DIR, code)
    os.makedirs(rdir, exist_ok=True)
    tmp = os.path.join(BASE, "data", "_meta_tmp")
    os.makedirs(tmp, exist_ok=True)
    all_rows, seen_ic = [], set()
    kept, removed = 0, 0
    for begin, end, size, keep in segments(s, mode, state, today):
        rows = query(code, begin, end, size, tmp)
        for r in rows:
            if r["info_code"] not in seen_ic:
                seen_ic.add(r["info_code"])
                all_rows.append(r)
        if dry:
            continue
        # 段内质量门：① 地板删纯快讯(<MIN_PAGES) ② 该段(年)内按页数取 top keep
        before = {f for f in os.listdir(rdir) if f.lower().endswith(".pdf")}
        download(code, begin, end, size)
        after = {f for f in os.listdir(rdir) if f.lower().endswith(".pdf")}
        scored = []
        for f in after - before:
            pp = pdf_pages(os.path.join(rdir, f))
            if 0 <= pp < MIN_PAGES:
                os.remove(os.path.join(rdir, f))
                removed += 1
            else:
                scored.append((pp if pp > 0 else 0, f))  # 打不开(-1)当0：排最后但保留
        scored.sort(key=lambda x: -x[0])                  # 段内长文档优先
        for _, f in scored[keep:]:                         # 超该年配额则删最短的
            os.remove(os.path.join(rdir, f))
            removed += 1
        kept += min(len(scored), keep)
    if dry:
        print(f"  {name}({code}): 查到 {len(all_rows)} 篇候选 (segments={len(segments(s,mode,state,today))})", flush=True)
        return {"code": code, "name": name, "queried": len(all_rows), "kept": 0, "removed": 0}
    n_meta = merge_metadata(code, all_rows)
    total_pdf = len([f for f in os.listdir(rdir) if f.lower().endswith(".pdf")])
    prev_seen = set(state.get(code, {}).get("seen_info_codes", []))
    state[code] = {"last_run": today, "name": name,
                   "seen_info_codes": sorted(prev_seen | seen_ic),
                   "total_pdf": total_pdf}
    print(f"  ✓ {name}({code}): 查{len(all_rows)} 新下{kept} 删{removed} | 库内共{total_pdf}篇 meta{n_meta}", flush=True)
    return {"code": code, "name": name, "queried": len(all_rows), "kept": kept, "removed": removed, "total_pdf": total_pdf}


# ----------------------------------------------------------------- 主流程
WL = {}


def main():
    global WL
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--full", action="store_true", help="全量首采")
    g.add_argument("--incremental", action="store_true", help="水位线增量")
    ap.add_argument("--codes", default="", help="只采指定标的(逗号分隔代码)，试点用")
    ap.add_argument("--ingest", action="store_true", help="采完触发 ingest.py 入库")
    ap.add_argument("--dry-run", action="store_true", help="只查列表不下载(预估规模)")
    args = ap.parse_args()
    mode = "incremental" if args.incremental else "full"

    WL = load_json(WATCHLIST, {})
    stocks = WL["stocks"]
    if args.codes:
        want = set(args.codes.split(","))
        stocks = [s for s in stocks if s["code"] in want]
    state = load_json(STATE_FP, {})
    today = date.today().isoformat()
    print(f"=== 采集 mode={mode} 标的{len(stocks)} 个 today={today}{' [DRY]' if args.dry_run else ''} ===", flush=True)

    summary = []
    for s in stocks:
        try:
            summary.append(collect_one(s, mode, state, today, args.dry_run))
        except Exception as e:
            print(f"  ✗ {s['name']}({s['code']}) 采集出错：{repr(e)[:160]}", flush=True)
    if not args.dry_run:
        save_state(state)
        # 清理临时元数据目录
        import shutil
        shutil.rmtree(os.path.join(BASE, "data", "_meta_tmp"), ignore_errors=True)

    tot_new = sum(x.get("kept", 0) for x in summary)
    tot_q = sum(x["queried"] for x in summary)
    print(f"\n=== 汇总：候选 {tot_q} 篇 | 本次新增入库 {tot_new} 篇 | 标的 {len(summary)} 个 ===")
    if args.ingest and not args.dry_run and tot_new:
        print("\n→ 触发 ingest.py 入库 …", flush=True)
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        subprocess.run([sys.executable, "ingest.py"], cwd=BASE, env=env)


if __name__ == "__main__":
    main()
