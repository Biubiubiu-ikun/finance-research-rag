# -*- coding: utf-8 -*-
"""
bench.py — 服务层并发压测（打纯本地接口 /consensus，反映服务自身吞吐）

为什么压 /consensus 而非 /chat：/chat 调 DeepSeek(外部API)，瓶颈在它、测不出我的服务；
/consensus 纯本地(读落盘缓存+计算，不调 LLM/ES/torch)，能测出服务层真实 QPS / p50 / p95。

用法：python bench.py [base_url] [总请求数] [并发数]
  python bench.py http://127.0.0.1:8000 300 30
"""
import sys
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 30
# 轮换几个标的，避免只打同一路径(更接近真实)
CODES = ["300750", "002594", "601012", "002371", "600183"]
URLS = [f"{BASE}/consensus/{c}" for c in CODES]


def one(i):
    t0 = time.time()
    try:
        r = requests.get(URLS[i % len(URLS)], timeout=30)
        return (time.time() - t0) * 1000, r.status_code == 200
    except Exception:
        return (time.time() - t0) * 1000, False


def pctl(xs, p):
    return sorted(xs)[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main():
    # 预热(让缓存/模块就绪)
    try:
        requests.get(URLS[0], timeout=30)
    except Exception:
        print(f"✗ 服务不可达：{BASE}（先 docker compose up 或 uvicorn）"); return
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        res = list(ex.map(one, range(N)))
    wall = time.time() - t0
    lat = [r[0] for r in res]
    ok = sum(1 for r in res if r[1])
    print(f"=== 并发压测 /consensus ===")
    print(f"目标 {BASE} | 总请求 {N} | 并发 {CONC}")
    print(f"成功率   : {ok}/{N} = {ok/N*100:.1f}%")
    print(f"吞吐 QPS : {N/wall:.1f}（{wall:.2f}s 完成 {N} 请求）")
    print(f"延迟 p50 : {statistics.median(lat):.0f} ms")
    print(f"延迟 p95 : {pctl(lat, 95):.0f} ms")
    print(f"延迟 max : {max(lat):.0f} ms")


if __name__ == "__main__":
    main()
