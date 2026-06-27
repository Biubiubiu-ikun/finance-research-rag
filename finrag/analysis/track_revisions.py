# -*- coding: utf-8 -*-
"""
track_revisions.py — 盈利预测修正追踪（观点/预测的时间演化，金融差异化亮点）

研报常自带【纵向信号】：投资建议里直接写"上调/下调盈利预测"+本次值与【原值/前值】，
如比亚迪东吴"下调 25-27 归母净利 350/509/664（原预测 450/589/710）"。
这就是真实卖方研究的 earnings revision（盈利预测修正）——比"静态一致预期"多了【方向与动量】。

做什么：每篇研报 LLM 抽 {调整方向, 本次vs原值净利, 评级动作, 依据句} → 聚合到标的：
  近期 上调X/下调Y/维持Z家 + 各年调整幅度 + 时间线。修正数字做溯源校验(原文逐字命中)。

缓存 data/analysis/revisions/{code}.jsonl 续跑。
用法：python track_revisions.py 比亚迪 | 002594 | all [--refresh]
"""
import os
import sys
import re
import json
import time
import collections

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from finrag.analysis.aggregate_forecast import (load_docs, all_stocks, build_extractions, unit_anomalies,
                                OUT, _age_days, STALE_DAYS)
from finrag.analysis.aggregate_views import call_llm, load_opinion_blocks, clean

REV_DIR = os.path.join(OUT, "revisions")

PROMPT = """你是盈利预测修正分析师。下面是研报「%s — %s（%s）」的观点/盈利预测段落。
判断该机构【本次相对其上一次预测】对盈利预测的调整，严格输出 JSON：
{
 "direction": "上调/下调/维持/首次/未提",
 "revisions": [{"year": 2025, "new": 65.1, "old": 73.4}],
 "rating_action": "原文评级动作，如 维持买入/首次买入/上调目标价至500元/下调评级 等；没有就空字符串",
 "evidence": "最能体现调整的原文句(≤50字)"
}
判定线索：
- 出现"上调/下调…盈利预测""前值/原值/原预测"→ 据此定 direction，并把【归母净利润】本次 new 与 原值 old(亿元)成对抽出(只抽明确给了原值的年份)。
- 只写"维持盈利预测/维持评级"未改数字 → direction=维持, revisions=[]。
- "首次覆盖/首次评级" → direction=首次。
- 完全没提预测调整 → direction=未提, revisions=[]。
规则：new/old 必须是【原文逐字出现】的归母净利数字(亿元口径)，不得编造或换算；拿不准就 direction=未提。

段落：
%s"""


def norm(s):
    return re.sub(r"[,\s，]", "", str(s))


def rev_pct(new, old):
    """本次vs原值的调整。跨零/负值不能用普通百分比(由盈转亏会显示成-181%) → 给语义标签。
    返回 (pct 或 None, label 或 None)。"""
    if not isinstance(new, (int, float)) or not isinstance(old, (int, float)) or old == 0:
        return None, None
    if old > 0 and new > 0:
        return round((new / old - 1) * 100, 1), None
    if old > 0 and new <= 0:
        return None, "由盈转亏"
    if old < 0 and new > 0:
        return None, "由亏转盈"
    return None, ("亏损扩大" if new < old else "亏损收窄")  # 同负


def extract_doc(name, org, date, text):
    out = call_llm(PROMPT % (name, org, date, text[:5000]), max_tokens=1200)
    src = norm(text)
    revs = []
    for r in out.get("revisions", []):
        try:
            y = int(r["year"])
        except (TypeError, ValueError, KeyError):
            continue
        new, old = r.get("new"), r.get("old")
        # 溯源：new/old 须在原文逐字命中(防编造)
        g_new = new is not None and norm(new) in src
        g_old = old is not None and norm(old) in src
        pct, label = rev_pct(new, old)
        revs.append({"year": y, "new": new, "old": old, "pct": pct, "label": label,
                     "grounded": bool(g_new and (old is None or g_old))})
    return {"org": org, "date": date, "direction": out.get("direction", "未提"),
            "rating_action": out.get("rating_action", ""), "evidence": out.get("evidence", ""), "revisions": revs}


def build(code, refresh=False):
    os.makedirs(REV_DIR, exist_ok=True)
    fp = os.path.join(REV_DIR, f"{code}.jsonl")
    done = {}
    if os.path.exists(fp) and not refresh:
        for l in open(fp, encoding="utf-8"):
            r = json.loads(l)
            done[r["doc_id"]] = r
    _, name, docs = load_docs(code)
    byorg = load_opinion_blocks(code)  # {(org,date,doc_id):{section:[...]}}
    recs, new = [], 0
    for (org, date, doc_id), secs in byorg.items():
        if doc_id in done:
            recs.append(done[doc_id]); continue
        text = "\n".join(f"【{s}】{' '.join(v)}" for s, v in secs.items())
        try:
            r = extract_doc(name, org, date, text)
            r["doc_id"] = doc_id
        except Exception as e:
            print(f"  ✗ {org}: {repr(e)[:100]}"); continue
        recs.append(r); new += 1
        time.sleep(0.2)
    with open(fp, "w", encoding="utf-8") as w:
        for r in recs:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    if new:
        print(f"  新抽取 {new} 篇（{fp}）")
    return name, recs


def summarize(recs):
    """标的级：方向分布 + 各年净利调整幅度(中位)。"""
    dist = collections.Counter(r["direction"] for r in recs)
    by_year = collections.defaultdict(list)
    for r in recs:
        for rv in r["revisions"]:
            p, _ = rev_pct(rv.get("new"), rv.get("old"))  # 现算，旧缓存也兼容
            if p is not None and rv["grounded"]:
                by_year[rv["year"]].append(p)
    import statistics
    yr_med = {y: round(statistics.median(v), 1) for y, v in sorted(by_year.items()) if v}
    return dist, yr_med


def cross_time(code):
    """同机构多篇纵向对比：同一家券商不同时间两篇的 归母净利预测 + 评级 变化。
    复用 P4① extract 缓存(已结构化各年净利+评级+日期)，无需新 LLM；跳过量纲异常篇。"""
    _, _, docs = load_docs(code)
    recs = build_extractions(code, docs)
    anom = unit_anomalies(recs)
    byorg = collections.defaultdict(list)
    for r in recs:
        if (r["org"], r.get("date", "")) not in anom:
            byorg[r["org"]].append(r)
    out = []
    for org, rs in byorg.items():
        rs = sorted(rs, key=lambda x: x.get("date", ""))
        if len(rs) < 2 or rs[0].get("date") == rs[-1].get("date"):
            continue
        early, late = rs[0], rs[-1]
        npm = lambda r: {f["year"]: f.get("net_profit_yi") for f in r["forecasts"]
                         if f.get("period") != "A" and isinstance(f.get("net_profit_yi"), (int, float))}
        em, lm = npm(early), npm(late)
        changes = []
        for y in sorted(set(em) & set(lm)):
            p, lab = rev_pct(lm[y], em[y])  # 早→晚
            changes.append(f"{y}E {em[y]}→{lm[y]}（{lab or f'{p:+.0f}%'}）")
        a_r, l_r = early.get("rating_raw", ""), late.get("rating_raw", "")
        out.append({"org": org, "early": early.get("date", ""), "late": late.get("date", ""),
                    "rating_chg": (f"{a_r} → {l_r}" if a_r != l_r else (a_r or "—")),
                    "changes": changes})
    return out


def render(code, name, recs):
    latest = max((r["date"] for r in recs if r.get("date")), default="")
    dist, yr_med = summarize(recs)
    L = [f"# 盈利预测修正追踪 · {name}（{code}）", "",
         f"- 覆盖 **{len(recs)}** 篇 ｜ 最新 {latest}",
         f"- **近期调整方向**：" + "，".join(f"{k} {v} 家" for k, v in dist.most_common()),
         f"- **归母净利预测调整幅度（中位数）**：" + ("，".join(f"{y}E {p:+.1f}%" for y, p in yr_med.items()) if yr_med else "（少有明确原值对比）"), ""]
    # 解读
    up, down = dist.get("上调", 0), dist.get("下调", 0)
    if up or down:
        tone = "上修为主(卖方情绪转暖)" if up > down else ("下修为主(盈利承压/预期降温)" if down > up else "上下修分化")
        L.append(f"> **信号**：{tone}。盈利预测修正方向是卖方动量的领先指标，比静态评级更敏感。")
        L.append("")
    L.append("## 各机构盈利预测调整（按报告日）")
    L.append("| 机构 | 日期 | 方向 | 净利修正(本次←原值, 亿元) | 评级动作 | 依据 |")
    L.append("|---|---|---|---|---|---|")
    for r in sorted(recs, key=lambda x: x.get("date", ""), reverse=True):
        old = " ⚠️旧" if r.get("date") and _age_days(r["date"], latest) > STALE_DAYS else ""
        parts = []
        for rv in r["revisions"]:
            if rv.get("new") is None or rv.get("old") is None:
                continue
            p, lab = rev_pct(rv.get("new"), rv.get("old"))
            tag = f"{p:+.0f}%" if p is not None else (lab or "")
            parts.append(f"{rv['year']}E {rv['new']}←{rv['old']}({tag})")
        revtxt = "；".join(parts) or "—"
        ev = (r.get("evidence", "") or "").replace("|", "/")[:40]
        L.append(f"| {r['org']}{old} | {r.get('date','')} | {r['direction']} | {revtxt} | {r.get('rating_action','')[:18]} | {ev} |")
    # 同机构跨时间预测对比（纵向：同一家不同时间两篇）
    ct = cross_time(code)
    if ct:
        L.append("## 同机构跨时间预测对比（纵向）")
        L.append("| 机构 | 报告日(早→晚) | 评级变化 | 归母净利预测变化(早→晚) |")
        L.append("|---|---|---|---|")
        for r in ct:
            L.append(f"| {r['org']} | {r['early']} → {r['late']} | {r['rating_chg']} | {'；'.join(r['changes']) or '—'} |")
        L.append("")

    # 溯源
    allrev = [rv for r in recs for rv in r["revisions"]]
    g = sum(1 for rv in allrev if rv["grounded"])
    L += ["", f"## 溯源：修正数字 {g}/{len(allrev)} 逐字命中原文" + ("（全部可溯源）" if g == len(allrev) and allrev else "")]
    return "\n".join(L)


def run_one(stock, refresh=False):
    code, name, docs = load_docs(stock)
    if not docs:
        print(f"未找到标的：{stock}"); return
    print(f"\n{'='*66}\n■ {name}（{code}）盈利预测修正追踪")
    name, recs = build(code, refresh=refresh)
    md = render(code, name, recs)
    open(os.path.join(OUT, f"{code}_revisions.md"), "w", encoding="utf-8").write(md)
    print(md)
    print(f"\n→ 已写出 {os.path.join(OUT, f'{code}_revisions.md')}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    refresh = "--refresh" in sys.argv
    target = args[0] if args else "比亚迪"
    if target == "all":
        for code, _ in all_stocks():
            run_one(code, refresh=refresh)
    else:
        run_one(target, refresh=refresh)
