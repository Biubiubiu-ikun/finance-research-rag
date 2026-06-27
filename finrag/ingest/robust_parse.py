# -*- coding: utf-8 -*-
"""
robust_parse.py — DeepDOC 解析的【卡死兜底】外层调度（子进程级超时）

DeepDOC 偶对个别 PDF 在 Python 层死循环(持 GIL) → 线程级超时无效(主线程拿不到 GIL，
实测五粮液卡 3h、立讯卡 17min 都没被线程超时救回)。本 wrapper 改用【子进程】跑
parse_report.py(可被强杀)，轮询 data/parsed 是否在增长：
  连续 STALL 秒无新增 → 判定卡死 → 强杀子进程 → 隔离当前卡死 PDF → 重启续跑(已解析的 skip)。
直到全部解析完。卡死 PDF 移到 data/reports_skipped/(不入库，少量损失可接受)。

被 ingest.py 当作 parse 步骤调用(替代直接跑 parse_report.py)；也可单独 `python robust_parse.py`。
"""
import os
import sys
import time
import glob
import shutil
import subprocess

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS = os.path.join(BASE, "data", "reports")
PARSED = os.path.join(BASE, "data", "parsed")
SKIP = os.path.join(BASE, "data", "reports_skipped")
STALL = 420   # 连续无新解析超过该秒数 → 判定卡死(正常篇 1-5min，留足余量；死循环则无限)
POLL = 15


def parsed_json(f):
    code = os.path.basename(os.path.dirname(f))
    name = os.path.splitext(os.path.basename(f))[0]
    return os.path.join(PARSED, code, name + ".json")


def unparsed():
    return [f for f in sorted(glob.glob(os.path.join(REPORTS, "*", "*.pdf"))) if not os.path.exists(parsed_json(f))]


def n_parsed():
    return len(glob.glob(os.path.join(PARSED, "*", "*.json")))


def n_skipped():
    return len(glob.glob(os.path.join(SKIP, "*.pdf")))


def main():
    rnd = 0
    while unparsed():
        rnd += 1
        left = len(unparsed())
        print(f"\n=== robust_parse 第 {rnd} 轮：剩 {left} 篇，启动 parse_report 子进程 ===", flush=True)
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        p = subprocess.Popen([sys.executable, "parse_report.py"], cwd=BASE, env=env)
        last_n, last_t = n_parsed(), time.time()
        while True:
            try:
                p.wait(timeout=POLL)
                break  # 子进程自己退出(正常解析完 / 报错)
            except subprocess.TimeoutExpired:
                pass
            cur = n_parsed()
            if cur > last_n:
                last_n, last_t = cur, time.time()
            elif time.time() - last_t > STALL:  # 卡死：强杀 + 隔离当前卡死 PDF
                p.kill()
                try:
                    p.wait(timeout=15)
                except Exception:
                    pass
                u = unparsed()
                if u:
                    os.makedirs(SKIP, exist_ok=True)
                    stuck = u[0]
                    try:
                        shutil.move(stuck, os.path.join(SKIP, os.path.basename(stuck)))
                        print(f"⏱ 卡死>{STALL}s，强杀并隔离: {os.path.relpath(stuck, BASE)}", flush=True)
                    except Exception as e:
                        print(f"隔离失败({e})，删除避免死循环: {os.path.relpath(stuck, BASE)}", flush=True)
                        try:
                            os.remove(stuck)
                        except Exception:
                            pass
                break  # 重启下一轮续跑
    print(f"\n✅ robust_parse 全部解析完成；累计隔离卡死 PDF {n_skipped()} 篇 → {SKIP}", flush=True)


if __name__ == "__main__":
    main()
