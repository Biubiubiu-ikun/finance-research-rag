# -*- coding: utf-8 -*-
"""
aggregate_forecast.py — P4① 多机构盈利预测 / 目标价聚合（生成分析层）

做什么：给定一个标的（名称或代码），跨所有券商把「盈利预测」聚合成一张对比表：
  - 抽取：每篇研报用 DeepSeek 从清洗后的盈利预测表(clean_markdown)抽出
          {评级, 目标价, 各年 营收/归母净利/EPS/PE}，统一换算到【亿元】。
  - 溯源(防幻觉)：抽取的每个数字都要带【原文逐字串 *_src】，回表核对：
          src 必须一字不差出现在原表里，且 *_yi 与 src 数值自洽(允许÷100换算)。
          → 数字级 grounding，比句级 entailment 对财务数据更硬。
  - 聚合：按预测年份对齐，算一致预期(中位数/均值)、区间(min~max)、分歧度(CV)、同比增速。
  - 时间一致性：标注各报告日期，优先最新；比最新报告旧 >STALE_DAYS 天的打 ⚠️。
  - 输出：控制台 + data/analysis/{code}_consensus.md。

数据源：data/structured/*/*.json（与 ES 同源，但带 clean_markdown 精确数字，且覆盖全部机构，
        聚合需要"全量"而非检索 top-k，故直接用结构化源为准；检索层证明这些块可被召回另见 retrieve_es）。

缓存/续跑：抽取结果写 data/analysis/extract/{code}.jsonl，按 doc 文件名跳过已抽；--refresh 强制重抽。

用法：
  python aggregate_forecast.py 宁德时代         # 单个标的
  python aggregate_forecast.py 300750
  python aggregate_forecast.py all               # 全部 8 个标的
  python aggregate_forecast.py 宁德时代 --refresh # 忽略缓存重抽
"""
import os
import sys
import re
import json
import glob
import time
import statistics as stats
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STRUCT = os.path.join(BASE, "data", "structured")
OUT = os.path.join(BASE, "data", "analysis")
EXTRACT_DIR = os.path.join(OUT, "extract")
STALE_DAYS = 45  # 比最新报告旧超过这么多天 → 标记为偏旧

# 抽取用于聚合的表类型（盈利预测主表 + 估值表里可能单独放 EPS/PE）
FORECAST_TABLE_TYPES = {"盈利预测", "估值"}


# ---------------------------------------------------------------- DeepSeek
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

PROMPT = """你是金融研报盈利预测抽取专家。下面是研报「%s — %s — %s」中与盈利预测/估值相关的表格（已清洗成 markdown）。

请抽取该机构对【%s】的盈利预测，严格按下面 JSON 输出：
{
 "rating_raw": "原文投资评级，如 买入/增持/推荐/优于大市；没有就空字符串",
 "target_price_raw": "原文目标价(含币种)，如 500.00元 / 220港元 / RMB220；没有就空字符串",
 "forecasts": [
   {
     "year": 2025,                 // 预测年份(整数)
     "period": "E",                // 年份后缀: A=已实现, E=预测; 无后缀填 "E"
     "revenue_yi": 4147.16,        // 营业(总)收入，单位【亿元】
     "revenue_src": "414715.59",   // ★该数字在上方表格中【一字不差】出现的原始数字串(换算前)
     "net_profit_yi": 685.82,      // 归母净利润，单位【亿元】
     "net_profit_src": "68582.20",
     "eps": 15.03,                 // 每股收益 EPS(元)
     "eps_src": "15.03",
     "pe": 24.38,                  // 市盈率 PE(倍)
     "pe_src": "24.38"
   }
 ]
}

规则：
- 单位换算：表格若以【百万元】计，营收/归母净利的 *_yi 需 ÷100 换成【亿元】；若已是【亿元】则不变。EPS、PE 不换算。
- *_src 必须是表格里【逐字出现】的数字串(换算前的原始数字)，用于回表核对防编造；某项缺失则该字段填 null。
- 归母净利润要取【归母/归属母公司】口径，不要拿"净利润(含少数股东)"。
- 只抽该公司主体真实出现的预测年份(一般 2024A~2027E)，不要编造不存在的年份或数字。

表格 markdown：
%s"""


def llm_extract(stock, org, title, source_md):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT % (stock, org, title, stock, source_md[:7000])}],
            "response_format": {"type": "json_object"}, "temperature": 0.0, "stream": False}
    r = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                      json=body, timeout=120)
    r.raise_for_status()
    c = r.json()["choices"][0]["message"]["content"].strip()
    c = re.sub(r"^```(?:json)?|```$", "", c, flags=re.M).strip()
    return json.loads(c)


# ---------------------------------------------------------------- 工具
def norm_num(s):
    """归一化数字串：去逗号/空格/全角，便于子串核对。"""
    return re.sub(r"[,\s，]", "", str(s)).replace("（", "(").replace("）", ")")


def rating_direction(raw):
    """各券商评级体系不一(买入/推荐/优于大市都是各自最高档)，统一成方向。"""
    r = re.sub(r"[（(].*?[)）]|[-\s]", "", str(raw or ""))
    if any(k in r for k in ["买入", "买进", "推荐", "增持", "优于大市", "优大于市", "看好", "强烈"]):
        return "看多"
    if any(k in r for k in ["中性", "持有", "标配", "同步"]):
        return "中性"
    if any(k in r for k in ["减持", "卖出", "跑输", "回避", "低配"]):
        return "看空"
    return "未评级" if not r else "看多"  # 本数据集均为买方推荐，默认看多


def normalize_industry(s):
    """把各券商五花八门的行业字段(光伏/光伏储能/太阳能/电力设备·电池…)归一成几个大类。
    顺序敏感：先半导体、再汽车(新能源汽车归汽车)、再新能源/电力设备。"""
    s = s or ""
    if any(k in s for k in ["半导体", "晶圆", "存储", "DRAM", "芯片", "电子", "覆铜板", "科技", "通信", "信息技术"]):
        return "半导体/电子"
    if "汽车" in s:
        return "汽车"
    if any(k in s for k in ["光伏", "储能", "太阳能", "锂电", "电池", "电力设备", "新能源", "电新", "逆变器", "AIDC"]):
        return "新能源/电力设备"
    return s or "其他"


def parse_tp(raw):
    """从目标价原文里解析 (数值, 币种)；A+H 双重上市优先取人民币(元)价。解析不出返回 (None, '')。"""
    if not raw:
        return None, ""
    # 先剔除股票代码(如 002594.CH / 1211.HK)，否则会被误当成目标价数字
    s = re.sub(r"\d{3,6}\.?[A-Z]{2}", "", str(raw)).replace(",", "")
    cands = []
    for m in re.finditer(r"\d+(?:\.\d+)?", s):
        ctx = s[max(0, m.start() - 8):m.end() + 3]  # 数字附近窗口判币种
        cur = "港元" if ("港" in ctx or "HK" in ctx.upper()) else "元"
        cands.append((float(m.group()), cur))
    if not cands:
        return None, ""
    cny = [v for v, c in cands if c != "港元"]
    return (cny[0], "元") if cny else cands[0]


def ground(val_yi, src, source_norm, allow_div100):
    """数字级溯源核对，返回 (status, src)。
    ok=原文有且数值自洽 / ungrounded=原文找不到(疑似编造) / mismatch=找到但数值对不上 / missing=未抽到。"""
    if src is None or src == "":
        return "missing", None
    s = norm_num(src)
    if s not in source_norm:
        return "ungrounded", s
    try:
        # 会计口径用括号表示负数，如 (4484) = -4484；先取绝对值再定符号
        neg = bool(re.match(r"[(（\-]", s.strip()))
        f = float(re.sub(r"[^0-9.]", "", s))
        if neg:
            f = -f
    except ValueError:
        return "ok", s
    if val_yi is None:
        return "ok", s
    tol = max(abs(f) * 0.02, 0.05)
    if abs(val_yi - f) <= tol or (allow_div100 and abs(val_yi - f / 100) <= max(abs(f / 100) * 0.02, 0.05)):
        return "ok", s
    return "mismatch", s


# ---------------------------------------------------------------- 抽取(带溯源核对)
def gather_source_md(doc):
    parts = []
    for t in doc.get("tables", []):
        if not t.get("keep", True):
            continue
        if t.get("table_type") in FORECAST_TABLE_TYPES:
            cm = t.get("clean_markdown") or t.get("summary") or ""
            if cm:
                parts.append(cm)
    return "\n\n".join(parts)


def extract_doc(doc):
    """对一篇研报抽取并溯源核对，返回一条记录(含 checks)。"""
    src_md = gather_source_md(doc)
    rec = {"org": doc["org"], "date": doc.get("date", ""), "title": doc.get("title", ""),
           "file": doc.get("file", ""), "rating_raw": doc.get("rating", ""),
           "target_price_raw": doc.get("target_price", ""), "forecasts": [], "checks": []}
    if not src_md.strip():
        rec["note"] = "无盈利预测表"
        return rec
    out = llm_extract(doc["stock_name"], doc["org"], doc.get("title", ""), src_md)
    rec["rating_raw"] = out.get("rating_raw") or rec["rating_raw"]
    rec["target_price_raw"] = out.get("target_price_raw") or rec["target_price_raw"]
    source_norm = norm_num(src_md)
    for f in out.get("forecasts", []):
        try:
            year = int(f.get("year"))
        except (TypeError, ValueError):
            continue
        row = {"year": year, "period": f.get("period", "E"),
               "revenue_yi": f.get("revenue_yi"), "net_profit_yi": f.get("net_profit_yi"),
               "eps": f.get("eps"), "pe": f.get("pe")}
        # 逐字段溯源核对
        for field, src_key, div100 in [("revenue_yi", "revenue_src", True), ("net_profit_yi", "net_profit_src", True),
                                       ("eps", "eps_src", False), ("pe", "pe_src", False)]:
            st, s = ground(f.get(field), f.get(src_key), source_norm, div100)
            rec["checks"].append({"year": year, "field": field, "status": st, "src": s})
        rec["forecasts"].append(row)
    return rec


# ---------------------------------------------------------------- 标的解析 + 缓存
def load_docs(stock):
    """stock 可为名称或代码；返回 (code, name, [doc...])。"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(STRUCT, "*", "*.json"))):
        s = json.load(open(fp, encoding="utf-8"))
        if stock in (s.get("stock_code"), s.get("stock_name")):
            docs.append(s)
    if not docs:
        return None, None, []
    return docs[0]["stock_code"], docs[0]["stock_name"], docs


def all_stocks():
    seen = {}
    for fp in glob.glob(os.path.join(STRUCT, "*", "*.json")):
        s = json.load(open(fp, encoding="utf-8"))
        seen[s["stock_code"]] = s["stock_name"]
    return sorted(seen.items())


def build_extractions(code, docs, refresh=False):
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    cache_fp = os.path.join(EXTRACT_DIR, f"{code}.jsonl")
    done = {}
    if os.path.exists(cache_fp) and not refresh:
        for line in open(cache_fp, encoding="utf-8"):
            r = json.loads(line)
            done[r.get("file") or r["title"]] = r
    recs = []
    new = 0
    for d in docs:
        key = d.get("file") or d.get("title")
        if key in done:
            recs.append(done[key])
            continue
        try:
            r = extract_doc(d)
        except Exception as e:
            print(f"  ✗ 抽取失败 [{d['org']}]: {repr(e)[:120]}")
            continue
        recs.append(r)
        new += 1
        time.sleep(0.2)
    # 重写缓存(含本次新抽)
    with open(cache_fp, "w", encoding="utf-8") as w:
        for r in recs:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    if new:
        print(f"  新抽取 {new} 篇（缓存 {cache_fp}）")
    return recs


# ---------------------------------------------------------------- 聚合 + 报告
def fmt(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def unit_anomalies(recs):
    """跨机构量纲一致性：以营收(恒正且大)为锚，某篇研报较同年中位数偏离 >10x，判为量纲异常。
    成因：部分表无单位标注，LLM 误把【亿元】当【百万】÷100（如华安隆基把676亿抽成6.76）。
    数字级溯源查不出(数字逐字命中且自洽)，需靠同业横比兜底。返回异常的 {(org,date)}。"""
    by_year = {}
    for r in recs:
        for f in r["forecasts"]:
            v = f.get("revenue_yi")
            if isinstance(v, (int, float)) and v > 0:
                by_year.setdefault(f["year"], []).append(v)
    med = {y: stats.median(v) for y, v in by_year.items() if v}
    bad = set()
    for r in recs:
        for f in r["forecasts"]:
            v, y = f.get("revenue_yi"), f["year"]
            if isinstance(v, (int, float)) and v > 0 and med.get(y, 0) > 0:
                if v / med[y] < 0.1 or v / med[y] > 10:
                    bad.add((r["org"], r.get("date", "")))
                    break
    return bad


def consensus_by_year(recs, field, exclude=None):
    """按预测年份收集 {year: [(value, org, date, stale), ...]}（只取 period=E 的预测年）。"""
    exclude = exclude or set()
    latest = max((r["date"] for r in recs if r.get("date")), default="")
    by_year = {}
    for r in recs:
        if (r["org"], r.get("date", "")) in exclude:
            continue
        stale = _age_days(r.get("date"), latest) > STALE_DAYS
        for f in r["forecasts"]:
            if f.get("period") == "A":
                continue
            v = f.get(field)
            if isinstance(v, (int, float)):
                by_year.setdefault(f["year"], []).append((v, r["org"], r.get("date", ""), stale))
    return by_year, latest


def _age_days(d, latest):
    try:
        from datetime import date
        a = date.fromisoformat(d[:10])
        b = date.fromisoformat(latest[:10])
        return (b - a).days
    except Exception:
        return 0


def stat_line(vals):
    xs = [v for v, *_ in vals]
    n = len(xs)
    if n == 0:
        return None
    md = stats.median(xs)
    mn, mx = min(xs), max(xs)
    mean = sum(xs) / n
    # 值跨越盈亏(min<0<max)时，均值接近 0，CV 失真 → 不报 CV，提示"盈亏分歧"
    straddle = mn < 0 < mx
    cv = None if (straddle or not mean) else (stats.pstdev(xs) / abs(mean) * 100 if n > 1 else 0.0)
    return {"n": n, "median": md, "mean": mean, "min": mn, "max": mx, "cv": cv, "straddle": straddle}


def render(code, name, recs):
    docs_ok = [r for r in recs if r["forecasts"]]
    latest = max((r["date"] for r in recs if r.get("date")), default="")
    industry = ""
    L = []
    L.append(f"# 多机构盈利预测一致预期 · {name}（{code}）")
    L.append("")
    L.append(f"- 覆盖机构：**{len(docs_ok)}** 家（共 {len(recs)} 篇研报）｜最新报告日：**{latest}**")
    # 评级方向
    dirs = {}
    for r in recs:
        dirs[rating_direction(r["rating_raw"])] = dirs.get(rating_direction(r["rating_raw"]), 0) + 1
    L.append(f"- 评级方向：" + "，".join(f"{k} {v}" for k, v in sorted(dirs.items(), key=lambda x: -x[1])))
    # 目标价
    tps = []
    for r in recs:
        v, cur = parse_tp(r["target_price_raw"])
        if v:
            tps.append((v, cur, r["org"], r.get("date", "")))
    if tps:
        cn = [v for v, c, *_ in tps if c != "港元"]
        line = "；".join(f"{o} {fmt(v)}{c}" for v, c, o, _ in sorted(tps))
        extra = f"（A股 {len(cn)} 家中位数 {fmt(stats.median(cn))}元）" if len(cn) >= 2 else ""
        L.append(f"- 目标价：{line} {extra}")
    else:
        L.append("- 目标价：本批研报均未给出明确目标价（A股研报常见，仅给评级）")
    L.append("")

    # 量纲异常机构(同业横比兜底无单位标注导致的÷100错抽)：从亿元口径统计中剔除
    anom = unit_anomalies(recs)

    # 归母净利润 / 营收 / EPS 对比表(核心)
    for field, label, unit in [("net_profit_yi", "归母净利润", "亿元"), ("revenue_yi", "营业收入", "亿元"), ("eps", "EPS", "元")]:
        excl = anom if unit == "亿元" else set()  # EPS 不受 ÷100 量纲问题影响
        by_year, _ = consensus_by_year(recs, field, exclude=excl)
        years = sorted(by_year)
        if not years:
            continue
        L.append(f"## {label}预测对比（{unit}）")
        L.append("")
        L.append("| 机构 | 报告日 | " + " | ".join(f"{y}E" for y in years) + " |")
        L.append("|" + "---|" * (len(years) + 2))
        # 每家机构一行(异常/偏旧的仍列出，但打标并不计入一致预期)
        org_rows = {}
        for r in recs:
            cells = {f["year"]: f[field] for f in r["forecasts"]
                     if f.get("period") != "A" and isinstance(f.get(field), (int, float))}
            if cells:
                org_rows[(r["org"], r.get("date", ""))] = cells
        for (org, date), cells in sorted(org_rows.items(), key=lambda x: x[0][1], reverse=True):
            tag = (" ⚠️量纲" if (org, date) in excl else "") + (" ⚠️旧" if _age_days(date, latest) > STALE_DAYS else "")
            L.append(f"| {org}{tag} | {date} | " + " | ".join(fmt(cells.get(y)) for y in years) + " |")
        # 一致预期统计(已剔除量纲异常机构)
        sl = {y: stat_line(by_year[y]) for y in years}
        L.append("| **一致预期(中位数)** | | " + " | ".join(fmt(sl[y]["median"]) for y in years) + " |")
        L.append("| 区间(min–max) | | " + " | ".join(f"{fmt(sl[y]['min'])}–{fmt(sl[y]['max'])}" for y in years) + " |")
        L.append("| 分歧度(CV) | | " + " | ".join("盈亏分歧" if sl[y]["cv"] is None else f"{sl[y]['cv']:.1f}%" for y in years) + " |")
        # 同比增速(中位数口径，处理扭亏/转亏)
        if field == "net_profit_yi" and len(years) >= 2:
            def g(prev, cur):
                if prev <= 0 and cur > 0:
                    return "扭亏"
                if prev > 0 and cur <= 0:
                    return "转亏"
                if prev <= 0 and cur <= 0:
                    return "亏损变动"
                return f"{(cur/prev-1)*100:+.1f}%"
            growth = ["—"] + [g(sl[years[i - 1]]["median"], sl[years[i]]["median"]) for i in range(1, len(years))]
            L.append("| 同比增速(中位数) | | " + " | ".join(growth) + " |")
        L.append("")

    # 时间一致性
    stale_orgs = [(r["org"], r["date"]) for r in recs if r.get("date") and _age_days(r["date"], latest) > STALE_DAYS and r["forecasts"]]
    L.append("## 时间一致性")
    if stale_orgs:
        L.append(f"- 最新口径 **{latest}**；以下 {len(stale_orgs)} 篇比最新报告旧 >{STALE_DAYS} 天，已在表中标 ⚠️旧，一致预期解读时建议降权：")
        for o, d in stale_orgs:
            L.append(f"  - {o}（{d}）")
    else:
        L.append(f"- 全部研报均在最新报告日 **{latest}** 前 {STALE_DAYS} 天内，时点一致，可直接横向对比。")
    L.append("")

    # 防幻觉(数字级溯源)
    checks = [c for r in recs for c in r.get("checks", [])]
    ok = sum(1 for c in checks if c["status"] == "ok")
    bad = [c for c in checks if c["status"] in ("ungrounded", "mismatch")]
    miss = sum(1 for c in checks if c["status"] == "missing")
    total_eff = len(checks) - miss
    L.append("## 事实一致性 / 防幻觉（数字级溯源）")
    rate = (ok / total_eff * 100) if total_eff else 100.0
    L.append(f"- 抽取数字 {total_eff} 个（另 {miss} 个该机构未披露）；可溯源核对通过 **{ok}（{rate:.1f}%）**：每个数字都在原表中逐字命中且换算自洽。")
    if bad:
        L.append(f"- ⚠️ {len(bad)} 个数字未通过核对（疑似抽取错误/幻觉，已可在缓存中复查）：")
        for c in bad[:10]:
            L.append(f"  - {c['year']} {c['field']}: {c['status']} (src={c['src']})")
    else:
        L.append("- ✅ 0 处幻觉：聚合表中所有财务数字均可回溯到原始研报表格。")
    # 量纲一致性兜底（数字级溯源查不出的÷100错抽，靠同业横比识别）
    if anom:
        L.append(f"- 量纲一致性：检出 {len(anom)} 篇研报数值较同业偏离 >10x（多因原表无单位标注被误÷100），"
                 f"已标 ⚠️量纲 并从亿元口径一致预期中剔除：" + "，".join(f"{o}（{d}）" for o, d in sorted(anom)))
    L.append("")
    return "\n".join(L), industry


# ---------------------------------------------------------------- 跨标的一页纸索引
def build_index():
    """汇总 8 个标的的一致预期到 data/analysis/_index.md（从缓存读，不调用 LLM）。"""
    rows = []
    for code, name in all_stocks():
        _, _, docs = load_docs(code)
        if not docs:
            continue
        recs = build_extractions(code, docs)  # 命中缓存
        industry = docs[0].get("industry", "")
        anom = unit_anomalies(recs)
        npy, _ = consensus_by_year(recs, "net_profit_yi", exclude=anom)
        years = sorted(npy)
        np25 = stat_line(npy[2025])["median"] if 2025 in npy else None
        g2526 = g2627 = "—"
        if 2025 in npy and 2026 in npy:
            a, b = stat_line(npy[2025])["median"], stat_line(npy[2026])["median"]
            g2526 = "扭亏" if a <= 0 < b else (f"{(b/a-1)*100:+.0f}%" if a > 0 else "—")
        if 2026 in npy and 2027 in npy:
            a, b = stat_line(npy[2026])["median"], stat_line(npy[2027])["median"]
            g2627 = f"{(b/a-1)*100:+.0f}%" if a > 0 else "—"
        cv26 = stat_line(npy[2026])["cv"] if 2026 in npy else None
        dirs = {}
        for r in recs:
            dirs[rating_direction(r["rating_raw"])] = dirs.get(rating_direction(r["rating_raw"]), 0) + 1
        rating_str = "/".join(f"{k}{v}" for k, v in sorted(dirs.items(), key=lambda x: -x[1]))
        tps = [parse_tp(r["target_price_raw"])[0] for r in recs if parse_tp(r["target_price_raw"])[1] == "元"]
        tps = [v for v in tps if v]
        tp_str = fmt(stats.median(tps)) + "元" if tps else "—"
        checks = [c for r in recs for c in r.get("checks", []) if c["status"] != "missing"]
        ok = sum(1 for c in checks if c["status"] == "ok")
        rate = f"{ok/len(checks)*100:.0f}%" if checks else "—"
        rows.append((name, industry, len([r for r in recs if r["forecasts"]]), rating_str,
                     fmt(np25), g2526, g2627, "盈亏分歧" if cv26 is None else f"{cv26:.1f}%", tp_str, rate))
    L = ["# 多机构盈利预测一致预期 · 跨标的总览（8 标的）", "",
         "> 由 `aggregate_forecast.py` 自动生成；各标的明细见 `{code}_consensus.md`。归母净利单位【亿元】。", "",
         "| 标的 | 行业 | 机构数 | 评级方向 | 25E净利(中位) | 25→26 | 26→27 | 分歧(26E·CV) | 目标价(中位) | 溯源通过率 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
    L += ["", "注：评级方向已跨券商体系归一(买入/推荐/优于大市等→看多)；目标价仅 A 股口径(元)，多数 A 股研报只给评级不给目标价；",
          "分歧度=各机构 2026E 归母净利的变异系数(CV)；溯源通过率=抽取数字逐字命中原表且换算自洽的比例；量纲异常机构已剔除一致预期。"]
    fp = os.path.join(OUT, "_index.md")
    open(fp, "w", encoding="utf-8").write("\n".join(L))
    print("\n" + "\n".join(L))
    print(f"\n→ 已写出 {fp}")


# ---------------------------------------------------------------- main
def run_one(stock, refresh=False):
    code, name, docs = load_docs(stock)
    if not docs:
        print(f"未找到标的：{stock}")
        return
    print(f"\n{'='*70}\n■ {name}（{code}）：{len(docs)} 篇研报")
    recs = build_extractions(code, docs, refresh=refresh)
    md, _ = render(code, name, recs)
    os.makedirs(OUT, exist_ok=True)
    out_fp = os.path.join(OUT, f"{code}_consensus.md")
    open(out_fp, "w", encoding="utf-8").write(md)
    print(md)
    print(f"\n→ 已写出 {out_fp}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    refresh = "--refresh" in sys.argv
    target = args[0] if args else "宁德时代"
    if target == "all":
        for code, name in all_stocks():
            run_one(code, refresh=refresh)
        build_index()
    elif target == "index":
        build_index()
    else:
        run_one(target, refresh=refresh)
