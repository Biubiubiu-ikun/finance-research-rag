# -*- coding: utf-8 -*-
"""
orchestrator.py — 多Agent编排：Planner + 并行 Workers + Aggregator
                  （在 P5 单 Agent 之上做"规划-并行执行-汇总"，面向复杂多维任务）

为什么要它（vs agent.py 的单循环）：
  复杂查询（如"出一份宁德时代的多维投研简报"）要同时覆盖 盈利预测/观点/修正/股价 等多维度——
    · 单 Agent 串行调多个重工具 → 慢；上下文越堆越长 → 易顾此失彼、答不全。
  本模块拆成三段式：
    1) Planner（1 次 LLM）：把任务拆成若干【聚焦子任务】，每个绑定一个维度 + 该维度的【工具子集】(分工)，输出 JSON 计划；
    2) Workers（并行）：每个 Worker 是一个【受限 Agent】——只拿到自己维度的工具子集与聚焦子问题，独立作答；
       子任务彼此无依赖 → ThreadPoolExecutor 并行（DeepSeek/ES 均 I/O 密集，等待时释放 GIL，并行真有效）；
    3) Aggregator（1 次 LLM）：汇总各 Worker 回答 → 结构化投研简报；只重组提炼、【不新增数字】
       （沿用项目铁律：幻觉锁在工具内、生成层不裸编）。

卖点：并行（wall-clock 相比串行明显缩短，见 timing）+ 分工（每个 Worker 上下文聚焦 → 更可靠/可溯源）。

用法：
  python orchestrator.py "出一份宁德时代的多维投研简报"
  python orchestrator.py "对比宁德和隆基的盈利预测与股价表现" --seq   # 串行(测速对照)
"""
import os
import sys
import time
import json
import requests
from concurrent.futures import ThreadPoolExecutor

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from finrag.agent import agent  # 复用 8 工具 + chat/_loop/run（已支持 tools/system 子集覆盖）
import finrag.analysis.aggregate_views as av  # 复用 call_llm（返回 JSON，用于 Planner）
from finrag.analysis.aggregate_forecast import all_stocks, API_KEY, API_URL, MODEL


# ============================================================ 能力清单（Planner 的"分工菜单"）
# focus → (该维度允许调用的工具子集, 维度说明)。Worker 被限定到对应子集，强制分工。
CAPABILITIES = {
    "盈利预测与目标价": (["forecast_consensus", "compute"],
        "跨券商盈利预测一致预期(归母净利/营收/EPS 中位数·区间·分歧度CV·同比增速)、评级分布、目标价"),
    "观点逻辑与风险": (["view_consensus"],
        "共识看多逻辑、共识风险、分歧/独家观点(每条带提及机构)"),
    "盈利预测修正动量": (["forecast_revisions"],
        "近期卖方上调/下调/维持盈利预测的方向与幅度(卖方动量/情绪转向)"),
    "股价表现与研报有效性": (["stock_price"],
        "现价、看多研报发布后30/60天实际涨跌与上涨胜率、目标价相对现价的 upside"),
    "原文细节": (["retrieve", "compute"],
        "聚合工具覆盖不到的具体事件/业务数据/某句表述 → 在研报原文做混合检索"),
}


def _llm_text(prompt, max_tokens=2600, temperature=0.3):
    """自由文本 LLM 调用(给 Aggregator 出 Markdown 简报；带网络抖动重试)。"""
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "stream": False, "max_tokens": max_tokens}
    last = None
    for attempt in range(4):
        try:
            r = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}",
                              "Content-Type": "application/json"}, json=body, timeout=180)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


# ============================================================ 1) Planner
PLANNER_PROMPT = """你是金融研报分析的【任务规划器】。把用户的复杂问题拆解成若干【聚焦子任务】，
每个子任务只负责一个维度，交给对应的子分析师并行处理。

可用维度(focus 必须从下列里选，不要自创)：
%s

可分析的标的(库内仅这些；标的不在库内时 stocks 留空并照常拆维度，由子分析师负责告知未覆盖)：
%s

规则：
- 只拆解为【确实需要】的维度：综合性"投研简报/全面分析"→覆盖前4个核心维度；具体窄问题→只取相关维度(可只1个)。
- 每个子任务写一个【聚焦、自包含】的子问题(含标的名)，让子分析师无需额外上下文即可作答。
- 同一维度不要重复拆成多个子任务；最多 5 个子任务。
- "原文细节"维度仅在前4个维度覆盖不到时才用。

只输出 JSON：
{"stock": "主标的名称(多标的或无则写概括，如'宁德vs隆基')",
 "subtasks": [{"id": 1, "focus": "<上面菜单里的维度名>", "question": "聚焦子问题(含标的)"}]}

用户问题：%s"""


def make_plan(question):
    menu = "\n".join(f"  - {k}：{desc}" for k, (_, desc) in CAPABILITIES.items())
    stocks = "、".join(f"{n}({c})" for c, n in all_stocks())
    plan = av.call_llm(PLANNER_PROMPT % (menu, stocks, question), max_tokens=1200)
    subs = plan.get("subtasks") or []
    # 规整 + 兜底：未知 focus 归入"原文细节"(全工具兜底由 worker 处理)
    clean = []
    seen = set()
    for i, s in enumerate(subs[:5], 1):
        focus = s.get("focus", "").strip()
        q = (s.get("question") or "").strip()
        if not q or focus in seen:
            continue
        seen.add(focus)
        clean.append({"id": i, "focus": focus, "question": q})
    plan["subtasks"] = clean
    return plan


# ============================================================ 2) Workers（并行）
WORKER_SYS_SUFFIX = """

【你的分工】你只负责「%s」这一个维度（%s），不要越界回答其他维度。
请简明给出该维度的要点（要点式即可，不必长篇），关键数字/结论后【内联标注来源】。
若该标的不在库内，按职责边界说明未覆盖即可。"""


def run_worker(subtask):
    focus = subtask["focus"]
    cap = CAPABILITIES.get(focus)
    if cap:
        names, desc = cap
        tool_subset = [t for t in agent.TOOLS if t["function"]["name"] in names]
        sys_prompt = agent.SYSTEM + WORKER_SYS_SUFFIX % (focus, desc)
    else:  # 未知维度 → 给全工具兜底
        tool_subset, sys_prompt = None, agent.SYSTEM
    t0 = time.time()
    try:
        answer, trace = agent.run(subtask["question"], verbose=False,
                                  tools=tool_subset, system=sys_prompt, max_turns=6)
    except Exception as e:
        answer, trace = f"（该维度执行出错：{repr(e)[:120]}）", []
    return {"id": subtask["id"], "focus": focus, "question": subtask["question"],
            "answer": answer, "trace": trace,
            "tools_used": [t["tool"] for t in trace], "secs": round(time.time() - t0, 1)}


def run_workers(subtasks, parallel=True, verbose=True):
    if parallel and len(subtasks) > 1:
        with ThreadPoolExecutor(max_workers=min(len(subtasks), 5)) as ex:
            results = list(ex.map(run_worker, subtasks))
    else:
        results = [run_worker(s) for s in subtasks]
    results.sort(key=lambda r: r["id"])
    if verbose:
        for r in results:
            print(f"  ✅ [{r['focus']}] {r['secs']}s  工具={r['tools_used']}")
    return results


# ============================================================ 3) Aggregator
AGG_PROMPT = """你是金融研报【首席分析师】。下面是多位子分析师就同一任务做的【分维度调研结果】。
请汇总成一份结构清晰的【投研简报】(Markdown)：
1. 开头【核心观点】：3-5 句结论性概述，先给整体判断；
2. 按维度分小节（用 ## 小标题）呈现要点，【完整保留各子分析师回答里的内联来源标注】(如 (来源：xx证券·2025-07-21))；
3. 若各维度间存在相互【印证或矛盾】(典型：卖方一致看多/上调，但实际股价表现/有效性偏弱)，专设一节「## 交叉印证」点出，这是简报的洞察价值所在；
4. 末尾保留一行「**数据来源**：」汇总涉及的工具/机构。

铁律：只能基于下方子分析师的回答做【重组与提炼】，【严禁新增或改动任何数字、机构名、目标价或结论】；
子分析师标注"未覆盖/数据缺失"的，如实说明，不要脑补。

【原始任务】%s

【各维度调研结果】
%s"""


def aggregate(question, workers):
    parts = []
    for w in workers:
        parts.append(f"### 维度：{w['focus']}\n（子问题：{w['question']}）\n{w['answer']}")
    return _llm_text(AGG_PROMPT % (question, "\n\n".join(parts)))


# ============================================================ 编排主流程
def brief(question, parallel=True, verbose=True):
    """规划→(并行)执行→汇总。返回 dict：plan / workers / report / timing。"""
    t0 = time.time()
    if verbose:
        print(f"🧭 Planner 规划中…")
    try:
        plan = make_plan(question)
    except Exception as e:
        plan = {"stock": "", "subtasks": []}
        if verbose:
            print(f"  ⚠️ 规划失败({repr(e)[:80]})，回退为单 Worker 直答")
    subtasks = plan.get("subtasks") or []
    if not subtasks:  # 兜底：规划不出来 → 单 Worker 跑原问题(全工具)
        subtasks = [{"id": 1, "focus": "原文细节", "question": question}]
    t_plan = time.time()
    if verbose:
        print(f"  计划：{plan.get('stock','')} → {len(subtasks)} 个子任务")
        for s in subtasks:
            print(f"    #{s['id']} [{s['focus']}] {s['question']}")
        print(f"⚙️ {'并行' if parallel else '串行'}执行 {len(subtasks)} 个 Worker…")
    workers = run_workers(subtasks, parallel=parallel, verbose=verbose)
    t_work = time.time()
    if verbose:
        print("📝 Aggregator 汇总中…")
    report = aggregate(question, workers)
    t_agg = time.time()
    workers_sum = round(sum(w["secs"] for w in workers), 1)  # 串行将耗时≈各 Worker 之和
    workers_wall = round(t_work - t_plan, 1)                 # 并行实际墙钟
    timing = {"plan_secs": round(t_plan - t0, 1),
              "workers_wall_secs": workers_wall, "workers_sum_secs": workers_sum,
              "parallel_speedup": round(workers_sum / workers_wall, 2) if workers_wall else 1.0,
              "agg_secs": round(t_agg - t_work, 1), "total_secs": round(t_agg - t0, 1),
              "parallel": parallel}
    return {"question": question, "stock": plan.get("stock", ""),
            "plan": subtasks, "workers": workers, "report": report, "timing": timing}


if __name__ == "__main__":
    seq = "--seq" in sys.argv
    q = " ".join(a for a in sys.argv[1:] if not a.startswith("--")) or "出一份宁德时代的多维投研简报"
    print("=" * 78 + f"\n❓ {q}\n" + "=" * 78)
    res = brief(q, parallel=not seq)
    print("\n" + "=" * 78 + "\n📊 投研简报\n" + "=" * 78 + "\n")
    print(res["report"])
    t = res["timing"]
    print("\n" + "-" * 78)
    print(f"⏱ 用时：规划 {t['plan_secs']}s | Worker {'串行' if seq else '并行'}墙钟 {t['workers_wall_secs']}s"
          f"(各 Worker 之和 {t['workers_sum_secs']}s，并行加速 {t['parallel_speedup']}x) | 汇总 {t['agg_secs']}s"
          f" | 合计 {t['total_secs']}s")
