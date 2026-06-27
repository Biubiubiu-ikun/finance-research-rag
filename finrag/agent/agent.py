# -*- coding: utf-8 -*-
"""
agent.py — P5 金融研报分析型 Agent（多工具自主调度 + 复杂查询拆解 + 事实自检）

设计核心：Agent 不直接读原文"裸生成"，而是调用【已过 P4 防幻觉校验】的聚合工具——
  幻觉风险被锁在工具内部(数字已逐字溯源、观点已 entailment)，Agent 只在可信数据上做
  路由 / 拆解 / 横向对比 / 汇总。把"生成的发散"和"事实的严谨"解耦。

工具(把前面各阶段能力封装成 function calling 的 tool)：
  - list_stocks()                 ：标的全集(code/名称/行业/机构数)，帮 Agent 把"动力电池龙头"映射到标的
  - forecast_consensus(stock)     ：P4① 盈利预测一致预期(净利/营收/EPS 中位数·区间·分歧度CV·同比·评级·目标价)
  - view_consensus(stock)         ：P4② 观点共识/分歧(看多逻辑/风险/分歧，每条带机构)
  - retrieve(query, stock, org)   ：P3 原文混合检索(兜底回答细节，惰性加载 torch/ES)
  - compute(expression)           ：确定性数值计算(LLM 算术不可靠 → 增速差/对比交给它)

用法：
  python agent.py "对比宁德时代和隆基绿能 2026 年归母净利增速，谁的卖方分歧更大？"
  python agent.py --demo        # 跑内置 demo 问题集
"""
import os
import sys
import re
import ast
import json
import time
import operator
import statistics
import subprocess
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from finrag.analysis.aggregate_forecast import (load_docs, all_stocks, build_extractions, consensus_by_year,
                                stat_line, unit_anomalies, rating_direction, parse_tp, OUT,
                                API_KEY, API_URL, MODEL)
import finrag.analysis.aggregate_views as av

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIEWS_AGG = os.path.join(OUT, "views_agg")


# ============================================================ 工具实现
CACHE_STATS = {"hits": 0, "misses": 0}  # 内存缓存命中统计(命中=进程内已算过，免去全量扫描+重算)
_MEM = {}  # 进程内聚合结果缓存(高并发提速；新数据入库后经 /cache_clear 失效)


def _memoize(tool):
    """聚合工具结果按 stock 记忆化：首次算并存，之后直接返回(避免每次请求重扫 structured + 重算)。"""
    def deco(fn):
        def wrap(stock):
            c = _MEM.get((tool, stock))
            if c is not None:
                CACHE_STATS["hits"] += 1
                return c
            CACHE_STATS["misses"] += 1
            r = fn(stock)
            if isinstance(r, dict) and "error" not in r:
                _MEM[(tool, stock)] = r
            return r
        return wrap
    return deco


def _not_found(stock):
    """库外标的统一返回：明确未覆盖 + 可查标的，供 Agent 引导用户(不要去硬检索/编造)。"""
    return {"error": f"标的'{stock}'不在覆盖范围（库内为 8 大行业 22 标的，见 available）",
            "available": [n for _, n in all_stocks()]}


def t_list_stocks():
    out = []
    for code, name in all_stocks():
        _, _, docs = load_docs(code)
        out.append({"code": code, "name": name,
                    "industry": docs[0].get("industry", "") if docs else "",
                    "n_reports": len(docs)})
    return {"stocks": out}


def _yearstats(recs, field, exclude):
    by_year, _ = consensus_by_year(recs, field, exclude=exclude)
    res = {}
    for y in sorted(by_year):
        s = stat_line(by_year[y])
        res[str(y)] = {"median": round(s["median"], 2), "min": round(s["min"], 2),
                       "max": round(s["max"], 2), "cv_pct": None if s["cv"] is None else round(s["cv"], 1),
                       "n": s["n"]}
    return res


@_memoize("fc")
def t_forecast_consensus(stock):
    code, name, docs = load_docs(stock)
    if not docs:
        return _not_found(stock)
    recs = build_extractions(code, docs)
    anom = unit_anomalies(recs)
    latest = max((r["date"] for r in recs if r.get("date")), default="")
    np_ = _yearstats(recs, "net_profit_yi", anom)
    # 净利同比(中位数口径)
    yrs = sorted(np_)
    yoy = {}
    for i in range(1, len(yrs)):
        a, b = np_[yrs[i - 1]]["median"], np_[yrs[i]]["median"]
        yoy[yrs[i]] = ("扭亏" if a <= 0 < b else ("转亏" if b <= 0 < a else
                       ("亏损变动" if a <= 0 and b <= 0 else f"{(b/a-1)*100:+.1f}%")))
    rating = {}
    for r in recs:
        d = rating_direction(r["rating_raw"])
        rating[d] = rating.get(d, 0) + 1
    # 目标价：工具直接算好"几家/中位数/区间"(人民币口径)，别让 Agent 自己数或把均值当中位数
    tp_items = []
    for r in recs:
        v, cur = parse_tp(r["target_price_raw"])
        if v:
            tp_items.append({"org": r["org"], "val": v, "cur": cur})
    cny = sorted(t["val"] for t in tp_items if t["cur"] != "港元")
    tp = {"n_given": len(tp_items), "n_given_cny": len(cny),
          "median_cny": round(statistics.median(cny), 2) if cny else None,
          "range_cny": [min(cny), max(cny)] if cny else None, "items": tp_items}
    return {"stock": name, "code": code, "latest_report": latest, "n_orgs": len([r for r in recs if r["forecasts"]]),
            "rating_dist": rating, "target_prices": tp,
            "net_profit_yi": np_, "net_profit_yoy_median": yoy,
            "revenue_yi_median": {y: v["median"] for y, v in _yearstats(recs, "revenue_yi", anom).items()},
            "eps_median": {y: v["median"] for y, v in _yearstats(recs, "eps", set()).items()},
            "unit_anomaly_orgs": [o for o, _ in anom],
            "_note": "单位亿元；已剔除量纲异常机构；数字均经逐字溯源核对；cv_pct=各机构变异系数(分歧度)"}


@_memoize("vc")
def t_view_consensus(stock):
    code, name, docs = load_docs(stock)
    if not docs:
        return _not_found(stock)
    os.makedirs(VIEWS_AGG, exist_ok=True)
    fp = os.path.join(VIEWS_AGG, f"{code}.json")
    if os.path.exists(fp):
        return json.load(open(fp, encoding="utf-8"))
    name, recs = av.build_views(code)
    agg = av.aggregate(name, recs)
    res = {"stock": name, "code": code, "overall": agg.get("overall", ""),
           "consensus_bull": agg.get("consensus_bull", []), "consensus_risk": agg.get("consensus_risk", []),
           "divergence": agg.get("divergence", []),
           "_note": "观点经跨机构聚合；每条带提及机构；明细已过句级 entailment 核查"}
    json.dump(res, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return res


@_memoize("fr")
def t_forecast_revisions(stock):
    code, name, docs = load_docs(stock)
    if not docs:
        return _not_found(stock)
    import finrag.analysis.track_revisions as tr
    name, recs = tr.build(code)  # 命中缓存
    dist, yr_med = tr.summarize(recs)

    def _rev(rv):
        p, lab = tr.rev_pct(rv.get("new"), rv.get("old"))  # 跨零/负给标签，避免"由盈转亏"显示成-181%
        return {"year": rv["year"], "new": rv["new"], "old": rv["old"], "pct": p, "label": lab}
    items = [{"org": r["org"], "date": r["date"], "direction": r["direction"],
              "rating_action": r.get("rating_action", ""),
              "revisions": [_rev(rv) for rv in r["revisions"] if rv.get("new") is not None]}
             for r in recs]
    return {"stock": name, "code": code, "direction_dist": dict(dist),
            "net_profit_revision_pct_median": yr_med, "by_org": items,
            "_note": "盈利预测修正(本次vs原值)；direction=上调/下调/维持/首次/未提；pct=归母净利调整幅度%；修正数字已溯源"}


@_memoize("price")
def t_stock_price(stock):
    code, name, docs = load_docs(stock)
    if not docs:
        return _not_found(stock)
    import finrag.analysis.price_analysis as pa
    a = pa.analyze(stock)
    if not a or a.get("error"):
        return {"error": (a or {}).get("error", "未取到股价")}
    return {"stock": a["stock"], "code": a["code"], "current_price": a["current_price"], "current_date": a["current_date"],
            "ret30_after_report_avg": a["avg_ret30"], "ret60_after_report_avg": a["avg_ret60"], "win_rate_30_pct": a["win_rate_30"],
            "target_upside_median_pct": a["median_upside_pct"], "target_upside": a["target_upside"],
            "_note": "ret*_after_report=看多研报发布后实际涨跌%(绝对,未剔大盘)；upside=目标价相对现价空间%"}


_retriever = None


def t_retrieve(query, stock=None, org=None, start_date=None, end_date=None, k=5):
    global _retriever
    if _retriever is None:
        from finrag.retrieval.retrieve_es import Retriever  # 惰性：用到才加载 torch/ES
        _retriever = Retriever()
    filters = {}
    if stock:
        code, _, _ = load_docs(stock)
        if not code:  # 库外标的：直接报未覆盖，别用空 filter 全库检索误导
            return _not_found(stock)
        filters["stock_code"] = code
    if org:
        filters["org"] = org
    dr = (start_date, end_date) if (start_date or end_date) else None   # 时间感知检索
    hits = _retriever.search(query, k=k, method="hybrid", filters=filters or None, rerank=True, date_range=dr)
    results = [{"org": s["org"], "date": s["date"], "section": s["section"],
                "content": s["content"][:240]} for _, s in hits]
    out = {"results": results}
    if dr and not results:  # 时间过滤后为空：明确告知该区间无数据，别让模型拿其他年份硬答/编造
        out["note"] = f"该标的在 {start_date or '不限'} ~ {end_date or '不限'} 区间内【无研报】（库内该标的数据可能晚于此区间，如实告知用户、勿用其他年份数据冒充）"
    return out


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}
_FUNCS = {"min": min, "max": max, "abs": abs, "round": round}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_safe_eval(a) for a in node.args])
    raise ValueError("仅支持数字四则运算与 min/max/abs/round")


def t_compute(expression):
    try:
        v = _safe_eval(ast.parse(expression, mode="eval").body)
        return {"expression": expression, "result": round(v, 4)}
    except Exception as e:
        return {"expression": expression, "error": repr(e)[:100]}


def t_ingest_report(pdf_path, code, name="", org="", date="", industry=""):
    """对话式入库：把一篇新研报 PDF 复制就位 + 写元数据 + 跑 ingest 全链路(解析→结构化→入ES→聚合)。重型,约数分钟。"""
    import shutil
    import csv
    if not os.path.exists(pdf_path):
        return {"error": f"找不到文件：{pdf_path}"}
    if not code:
        return {"error": "需提供股票代码 code（决定入库目录）"}
    rd = os.path.join(BASE_DIR, "data", "reports", code)
    os.makedirs(rd, exist_ok=True)
    fname = os.path.basename(pdf_path)
    shutil.copy(pdf_path, os.path.join(rd, fname))
    # 写/追加 metadata CSV（title 须等于 PDF 文件名以匹配元数据）
    title = os.path.splitext(fname)[0]
    mdir = os.path.join(BASE_DIR, "data", "metadata")
    os.makedirs(mdir, exist_ok=True)
    mfp = os.path.join(mdir, f"{code}_page1.csv")
    is_new = not os.path.exists(mfp)
    with open(mfp, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["title", "org_name", "publish_date", "stock_name", "stock_code",
                        "industry_name", "rating_name", "info_code", "url"])
        w.writerow([title, org, date, name, code, industry, "", "", ""])
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    try:
        r = subprocess.run([sys.executable, "ingest.py"], cwd=BASE_DIR, env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    except subprocess.TimeoutExpired:
        return {"error": "入库超时(>15 分钟)"}
    if r.returncode != 0:
        return {"error": "入库失败", "log": (r.stdout or "")[-400:]}
    no_new = "没有检测到新研报" in (r.stdout or "")
    return {"status": "已是最新(该PDF此前已入库)" if no_new else "入库完成",
            "stock": name or code, "code": code,
            "note": "该标的现已可用 forecast_consensus / view_consensus / forecast_revisions 等查询"}


TOOLS_IMPL = {"list_stocks": t_list_stocks, "forecast_consensus": t_forecast_consensus,
              "view_consensus": t_view_consensus, "forecast_revisions": t_forecast_revisions,
              "retrieve": t_retrieve, "compute": t_compute, "ingest_report": t_ingest_report,
              "stock_price": t_stock_price}

# ============================================================ 工具 schema (OpenAI/DeepSeek function calling)
TOOLS = [
    {"type": "function", "function": {"name": "list_stocks",
        "description": "列出全部可分析标的(代码/名称/行业/研报数)。当用户用行业或泛称(如'动力电池龙头')提问、需要先确定具体标的时调用。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "forecast_consensus",
        "description": "获取某标的跨券商【盈利预测一致预期】：各年归母净利/营收/EPS 的中位数·区间·分歧度CV·同比增速，以及评级分布、目标价。数字已逐字溯源核对、已剔除量纲异常机构。回答盈利预测/增速/目标价/分歧度类问题用它。",
        "parameters": {"type": "object", "properties": {"stock": {"type": "string", "description": "标的名称或代码，如 宁德时代 或 300750"}}, "required": ["stock"]}}},
    {"type": "function", "function": {"name": "view_consensus",
        "description": "获取某标的跨券商【定性观点聚合】：共识看多逻辑、共识风险、分歧/独家观点(每条带提及机构)。明细已过句级 entailment 核查。回答看多逻辑/风险/分歧/为什么看好类问题用它。",
        "parameters": {"type": "object", "properties": {"stock": {"type": "string", "description": "标的名称或代码"}}, "required": ["stock"]}}},
    {"type": "function", "function": {"name": "forecast_revisions",
        "description": "获取某标的各券商近期【盈利预测修正】：本次相对上次是上调/下调/维持/首次，及归母净利调整幅度(%)。回答'最近卖方上调还是下调了盈利预测/谁下修了/预测动量/情绪转向'类问题用它(比静态一致预期多了方向与动量)。",
        "parameters": {"type": "object", "properties": {"stock": {"type": "string", "description": "标的名称或代码"}}, "required": ["stock"]}}},
    {"type": "function", "function": {"name": "stock_price",
        "description": "获取某标的【实际股价】与【研报有效性】：现价、看多研报发布后30/60天实际涨跌与上涨胜率、目标价相对现价的空间(upside)。回答'现在股价多少/距目标价还有多少空间/券商看多后实际涨没涨/研报准不准'类问题用它(接入真实市场数据)。",
        "parameters": {"type": "object", "properties": {"stock": {"type": "string", "description": "标的名称或代码"}}, "required": ["stock"]}}},
    {"type": "function", "function": {"name": "retrieve",
        "description": "在研报原文里做混合检索，返回最相关的若干原文片段(带机构/日期/章节)。当聚合工具回答不了的细节性问题(具体事件、某句表述、某项业务数据)时用。【按年份/时间提问时必须用 start_date/end_date 按研报发布日期过滤】(如问2024年：start_date=2024-01-01, end_date=2024-12-31)；若返回 note 说该区间无研报，就如实告知用户、不要拿其他年份数据冒充。",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "stock": {"type": "string", "description": "可选，限定标的"}, "org": {"type": "string", "description": "可选，限定机构"}, "start_date": {"type": "string", "description": "可选，起始发布日期 YYYY-MM-DD，按年份提问时设该年起点"}, "end_date": {"type": "string", "description": "可选，结束发布日期 YYYY-MM-DD"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "compute",
        "description": "确定性数值计算(支持数字四则运算、乘方，以及 min/max/abs/round)。涉及增速差、比值、百分点对比、取最值等必须用它而非心算，避免算错。",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "如 (864.06/692.85-1)*100"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "ingest_report",
        "description": "把一篇【新研报 PDF】入库到知识库(解析→结构化→分块→入ES→聚合，约数分钟)。仅当用户【明确要求】把某个本地 PDF 文件加入/入库时才调用；需要 pdf_path 和股票代码 code。",
        "parameters": {"type": "object", "properties": {
            "pdf_path": {"type": "string", "description": "PDF 文件的本地绝对路径"},
            "code": {"type": "string", "description": "股票代码，如 600183"},
            "name": {"type": "string", "description": "股票名称(可选)"},
            "org": {"type": "string", "description": "券商机构(可选)"},
            "date": {"type": "string", "description": "报告日期 YYYY-MM-DD(可选)"},
            "industry": {"type": "string", "description": "行业(可选)"}},
         "required": ["pdf_path", "code"]}}},
]

SYSTEM = """你是金融研报分析助手，只服务于库内【8 大行业 22 个标的（半导体/新能源/医药/食品饮料/银行/军工/家电/通信算力）】的卖方研报问答（盈利预测/目标价/评级/观点/风险/预测修正/原文检索等）。不确定某标的是否在库时调 list_stocks 确认。

铁律：
1. 【职责边界】问题超出上述范围时不要硬答：
   - 所问标的/行业不在库（可调 list_stocks 确认）→ 说明未覆盖、列出可查标的并引导；
   - 与金融研报分析无关（写作/闲聊/通用知识/编程/数学题等）→ 礼貌说明你的职责范围、引导回研报问题，【不要执行该无关任务】；
   - 要求【预测涨跌/荐股/买卖建议】，或问【大盘指数、汇率、宏观经济等非个股研报数据】→ 直接拒答（只提供库内研报数据、不预测涨跌、不做投资建议），【绝不调用任何工具硬凑】。
2. 【防幻觉】只能基于工具返回的数据作答，严禁编造数字/机构名/观点；工具没给的就说"数据未覆盖"。
3. 涉及数值计算（增速/差值/比值/取最值）必须调用 compute，不要心算。
4. 复杂问题先拆解：先想清楚要哪些标的/维度，再分别调工具拿数据，最后综合。
5. 【可溯源】关键数字/结论后【内联标注来源】，格式 "…692.85亿元（来源：国信证券·2025-07-21）"；回答末尾再汇总一行「数据来源」(工具 + 机构/日期)。
6. 数字带单位（亿元/元/%）；区分"一致预期(中位数)"与"个别机构"。
7. 用户明确要求把某个本地研报 PDF 加入/入库时，调用 ingest_report(需 pdf_path 与股票代码 code)——这是扩充知识库的合法操作，入库后该标的即可查询。

用中文、条理清晰作答。"""


# ============================================================ Agent 主循环
RUN_STATS = {}  # 最近一次 run 的统计(LLM 调用数 / token / 工具调用数)，供 API 暴露成本指标


def chat(messages, tools=None):
    body = {"model": MODEL, "messages": messages, "tools": tools or TOOLS, "tool_choice": "auto",
            "temperature": 0.2, "stream": False}
    last = None
    for attempt in range(4):  # DeepSeek API 偶发 SSL/网络抖动 → 指数退避重试，别让单次故障搞挂整个 Agent
        try:
            r = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                              json=body, timeout=150)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"], data.get("usage", {})
        except requests.exceptions.RequestException as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def _loop(messages, trace, stats, max_turns, tools=None):
    """跑 function-calling 多轮，直到模型不再调工具，返回最终回答文本。"""
    for _ in range(max_turns):
        msg, usage = chat(messages, tools=tools)
        stats["llm_calls"] += 1
        stats["total_tokens"] += usage.get("total_tokens", 0)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg.get("content", "")
        for tc in calls:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            if stats.get("verbose"):
                print(f"  🔧 调用 {fn}({', '.join(f'{k}={v}' for k, v in args.items())})")
            try:
                result = TOOLS_IMPL[fn](**args)
            except Exception as e:
                result = {"error": repr(e)[:150]}
            trace.append({"tool": fn, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(result, ensure_ascii=False)})
    return "（已达最大调用轮数，未能完成；请缩小问题范围）"


REFLECT_PROMPT = """你是金融问答质检员。判断【回答】是否【充分、准确】地回答了【问题】，且关键结论都能在【已获取数据】里找到支撑。
只输出 JSON：{"sufficient": true/false, "issues": ["具体不足点；sufficient=true 时为空数组[]"]}
判为不足(sufficient=false)的情形：答非所问 / 遗漏问题的某个子部分 / 要求对比却只答了一个标的 / 给了结论但数据里查不到 / 关键口径单位不清。
【问题】%s
【回答】%s
【已获取数据(工具返回)】%s"""


def reflect_check(question, answer, trace):
    import finrag.analysis.aggregate_views as av
    ev = json.dumps([{"tool": t["tool"], "result": t["result"]} for t in trace], ensure_ascii=False)[:6000]
    try:
        return av.call_llm(REFLECT_PROMPT % (question, answer, ev), max_tokens=600)
    except Exception as e:
        return {"sufficient": True, "issues": [], "_error": repr(e)[:80]}


def run(question, max_turns=8, verbose=True, reflect=False, history=None, tools=None, system=None):
    """返回 (answer, trace)。history=[{role,content}] 传入多轮对话上下文(支持追问/指代,如先问宁德再问'那隆基呢')。
    reflect=True 时答完自检：不充分则反馈续跑补调工具再答(Reflexion 式)。统计写入 RUN_STATS。
    tools/system 可覆盖工具子集与系统提示(供多Agent编排里把 Worker 限定到单一维度的工具与分工)。"""
    messages = [{"role": "system", "content": system or SYSTEM}]
    if history:
        messages += history
    messages.append({"role": "user", "content": question})
    trace = []
    stats = {"llm_calls": 0, "total_tokens": 0, "verbose": verbose}
    reflection = None
    try:
        answer = _loop(messages, trace, stats, max_turns, tools=tools)
        if reflect and trace:  # 有调工具(非越界拒答)才值得自检
            reflection = reflect_check(question, answer, trace)
            if not reflection.get("sufficient", True):
                if verbose:
                    print(f"  🔁 自检不通过 → 补充：{reflection.get('issues')}")
                messages.append({"role": "user",
                                 "content": "自检发现以下不足：" + "；".join(reflection.get("issues", [])) +
                                            "。请按需【补充调用工具】获取数据后，给出更完整准确的回答。"})
                answer = _loop(messages, trace, stats, max_turns, tools=tools)
        return answer, trace
    finally:
        RUN_STATS.clear()
        RUN_STATS.update(llm_calls=stats["llm_calls"], total_tokens=stats["total_tokens"],
                         tool_calls=len(trace), reflected=bool(reflect and trace), reflection=reflection)


DEMO = [
    "对比宁德时代和隆基绿能 2026 年归母净利增速，谁的卖方分歧更大？",
    "阳光电源有哪些券商共识看多逻辑？主要分歧在哪？",
    "半导体设备和锂电池里，哪个标的的盈利预测分歧度最高？",
    "比亚迪的目标价券商怎么看？给了目标价的有几家？",
]

if __name__ == "__main__":
    reflect = "--reflect" in sys.argv
    if "--demo" in sys.argv:
        for q in DEMO:
            print("\n" + "=" * 72 + f"\n❓ {q}\n")
            print(run(q, reflect=reflect)[0])
    else:
        q = " ".join(a for a in sys.argv[1:] if not a.startswith("--")) or DEMO[0]
        print("=" * 72 + f"\n❓ {q}\n")
        print(run(q, reflect=reflect)[0])
