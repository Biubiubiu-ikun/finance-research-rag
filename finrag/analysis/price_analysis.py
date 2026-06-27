# -*- coding: utf-8 -*-
"""
price_analysis.py — 研报 vs 实际股价对比（把卖方观点接到真实市场，B 方向）

做什么：用 akshare 拉 A 股实际日线，对库内标的做两类分析——
  ① 研报有效性回测：每篇研报发布日 → 之后 30/60 个自然日的实际涨跌；
     因库内研报清一色看多，看"看多后平均涨多少 / 上涨胜率"，并按机构看谁的研报含金量高。
  ② 目标价 upside：现价 vs 各券商目标价，还有多少空间 / 是否已兑现。

gotcha：环境配了代理(127.0.0.1:7890)走不通东财 → 拉数据时【临时清代理】，拉完恢复(不影响 DeepSeek)。
缓存：股价存 data/analysis/prices/{code}.csv，避免重复联网拉。
用法：python price_analysis.py 宁德时代 | 300750 | all
"""
import os
import sys
import json
from datetime import date, timedelta

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from finrag.analysis.aggregate_forecast import load_docs, all_stocks, OUT, parse_tp, rating_direction

PRICE_DIR = os.path.join(OUT, "prices")
PROXY_KEYS = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]


def fetch_price(code, start="20250101", end=None):
    """拉 A 股前复权日线 → [(date_str, close)] 升序。临时清代理(东财走不通代理)，带本地 csv 缓存。"""
    os.makedirs(PRICE_DIR, exist_ok=True)
    fp = os.path.join(PRICE_DIR, f"{code}.csv")
    if os.path.exists(fp):
        rows = [l.strip().split(",") for l in open(fp, encoding="utf-8") if l.strip()]
        return [(d, float(c)) for d, c in rows]
    end = end or date.today().strftime("%Y%m%d")
    import time
    saved = {k: os.environ.pop(k, None) for k in PROXY_KEYS}  # 临时绕代理
    os.environ["no_proxy"] = "*"                              # 强制绕过系统代理(Windows IE设置)
    try:
        import akshare as ak
        series = None
        for attempt in range(3):                              # 东财偶发断连(限流)→重试
            try:
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
                if df is not None and len(df):
                    series = [(str(d)[:10], float(c)) for d, c in zip(df["日期"], df["收盘"])]
                    break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        if not series:                                        # fallback 新浪源
            pre = "sh" if code.startswith("6") else "sz"
            sdf = ak.stock_zh_a_daily(symbol=pre + code, start_date=start, end_date=end, adjust="qfq")
            series = [(str(d)[:10], float(c)) for d, c in zip(sdf["date"], sdf["close"])]
    finally:
        os.environ.pop("no_proxy", None)
        os.environ.update({k: v for k, v in saved.items() if v})  # 恢复(不影响 DeepSeek)
    with open(fp, "w", encoding="utf-8") as w:
        for d, c in series:
            w.write(f"{d},{c}\n")
    return series


def close_on_or_after(series, target):
    """取 >= target 日期的第一个交易日收盘(研报发布当日可能停牌/非交易日)。"""
    for d, c in series:
        if d >= target:
            return c
    return None


def _plus_days(date_str, n):
    return (date.fromisoformat(date_str[:10]) + timedelta(days=n)).isoformat()


def analyze(stock):
    code, name, docs = load_docs(stock)
    if not docs:
        return None
    series = fetch_price(code)
    if not series:
        return {"stock": name, "code": code, "error": "未取到股价"}
    cur_date, cur_price = series[-1]
    # ① 研报有效性回测
    eff, r30s, r60s, win = [], [], [], 0
    for d in sorted(docs, key=lambda x: x.get("date", "")):
        rd = d.get("date", "")
        p0 = close_on_or_after(series, rd)
        if not p0:
            continue
        p30 = close_on_or_after(series, _plus_days(rd, 30))
        p60 = close_on_or_after(series, _plus_days(rd, 60))
        ret30 = round((p30 / p0 - 1) * 100, 1) if p30 else None
        ret60 = round((p60 / p0 - 1) * 100, 1) if p60 else None
        if ret30 is not None:
            r30s.append(ret30); win += (ret30 > 0)
        if ret60 is not None:
            r60s.append(ret60)
        eff.append({"org": d["org"], "date": rd, "dir": rating_direction(d.get("rating", "")),
                    "p0": round(p0, 2), "ret30": ret30, "ret60": ret60})
    # ② 目标价 upside
    ups = []
    for d in docs:
        tp, cur = parse_tp(d.get("target_price", ""))
        if tp and cur == "元":
            ups.append({"org": d["org"], "target": tp, "upside_pct": round((tp / cur_price - 1) * 100, 1)})
    med = lambda xs: round(sorted(xs)[len(xs) // 2], 1) if xs else None
    return {"stock": name, "code": code, "current_price": cur_price, "current_date": cur_date,
            "n_reports": len(eff), "efficacy": eff,
            "avg_ret30": round(sum(r30s) / len(r30s), 1) if r30s else None,
            "avg_ret60": round(sum(r60s) / len(r60s), 1) if r60s else None,
            "win_rate_30": round(win / len(r30s) * 100, 0) if r30s else None,
            "target_upside": ups, "median_upside_pct": med([u["upside_pct"] for u in ups]),
            "_note": "看多研报发布后实际涨跌(绝对收益,未剔大盘)；upside=目标价相对现价空间"}


def render(a):
    if not a or a.get("error"):
        return f"# {a.get('stock') if a else stock}：{a.get('error','无数据') if a else '未找到'}"
    L = [f"# 研报 vs 实际股价 · {a['stock']}（{a['code']}）", "",
         f"- 现价 **{a['current_price']}** 元（{a['current_date']}）｜覆盖研报 {a['n_reports']} 篇",
         f"- **看多研报发布后**：平均 30 天 **{a['avg_ret30']}%**、60 天 **{a['avg_ret60']}%**，30 天上涨胜率 **{a['win_rate_30']}%**"]
    if a["target_upside"]:
        L.append(f"- **目标价空间**：中位 upside **{a['median_upside_pct']}%**（" +
                 "，".join(f"{u['org']} {u['target']}元/{u['upside_pct']:+.0f}%" for u in a["target_upside"]) + "）")
    L += ["", "## 各研报发布后实际表现", "| 机构 | 发布日 | 方向 | 发布价 | +30天 | +60天 |", "|---|---|---|---|---|---|"]
    for e in a["efficacy"]:
        f = lambda x: f"{x:+.1f}%" if isinstance(x, (int, float)) else "—"
        L.append(f"| {e['org']} | {e['date']} | {e['dir']} | {e['p0']} | {f(e['ret30'])} | {f(e['ret60'])} |")
    L += ["", "> 解读：库内研报全为看多；发布后为**绝对收益**(进阶可减沪深300算 **alpha**)；",
          "> 目标价 upside = 前复权现价 vs 名义目标价，**除权标的口径略有偏差**(应统一口径)。"]
    return "\n".join(L)


def run_one(stock):
    a = analyze(stock)
    if not a:
        print(f"未找到标的：{stock}"); return
    md = render(a)
    open(os.path.join(OUT, f"{a['code']}_price.md"), "w", encoding="utf-8").write(md)
    print(md)
    print(f"\n→ 已写出 {os.path.join(OUT, a['code'] + '_price.md')}")


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    target = args[0] if args else "宁德时代"
    if target == "all":
        for code, _ in all_stocks():
            run_one(code)
    else:
        run_one(target)
