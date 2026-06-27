# -*- coding: utf-8 -*-
"""
eval_orchestrator.py — 多 Agent 编排【质量】量化（单 Agent vs 多 Agent，A/B 对照）

把"多 Agent 编排"从只有【速度】卖点(实测 3.3x)，补上【质量】数字：
同一批【多维复杂问题】分别走 单 Agent(agent.run) 与 多 Agent(orchestrator.brief)，
用 LLM-judge 按【维度覆盖率 / 完整性 / faithfulness】打分，输出 质量 delta + 墙钟 delta。

为什么多 Agent 该更好：Planner 强制把任务拆成聚焦子任务、各 Worker 限定工具子集独立调研，
不易像单 Agent 那样在一个长上下文里漏掉某一维 → 预期【维度覆盖率】更高。

诚实前提：LLM-judge 有偏差、题量小；这是【同 judge 同题】的相对对比，量的是"编排是否提升覆盖"，
不是绝对质量真值。最严谨应再加人工评分。

用法：
  python eval_orchestrator.py          # 全部对照题
  python eval_orchestrator.py -n 2     # 只跑前 2 题(省 API/时间)
  python eval_orchestrator.py --judge-only   # 仅自检 judge 是否能解析(1 次 API，省 GPU/ES)
"""
import os
import sys
import time
import argparse

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import finrag.analysis.aggregate_views as av  # 复用 call_llm(DeepSeek, JSON 输出)

# 多维复杂问题：天然该覆盖 盈利预测 / 观点 / 修正 / 股价 四维（最能体现编排价值）
QUESTIONS = [
    "出一份宁德时代的多维投研简报",
    "全面分析比亚迪：盈利预测一致预期、卖方观点共识与分歧、近期盈利预测修正、研报与实际股价表现",
    "恒瑞医药值得关注吗？请从盈利预测、机构观点、预测修正、股价表现多角度分析",
    "给贵州茅台做一份综合投研分析（盈利预测 / 观点 / 修正 / 股价 都要覆盖）",
]
DIMS = ["盈利预测/一致预期", "卖方观点(共识/分歧)", "盈利预测修正(上调/下调)", "股价/研报有效性"]

JUDGE_PROMPT = """你是严格的投研简报质检员。针对【问题】，评估下面这份【回答】的质量。
该类问题应覆盖以下维度：
%s

只输出 JSON：
{
 "covered": [实际覆盖到的维度序号(从1开始的整数列表)],
 "completeness": 0-100,   // 各维度内容是否充实、有数据支撑
 "faithfulness": 0-100,   // 是否仅基于给出的数据、无明显编造(仅就文本判断)
 "missed": ["漏掉或明显薄弱的维度名"]
}

【问题】%s

【回答】
%s"""


def judge(q, answer):
    dim_lines = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(DIMS))
    try:
        out = av.call_llm(JUDGE_PROMPT % (dim_lines, q, (answer or "")[:6000]), max_tokens=600)
    except Exception as e:
        return {"covered": [], "completeness": 0, "faithfulness": 0, "missed": DIMS, "coverage_pct": 0.0, "_err": repr(e)[:80]}
    cov = [c for c in (out.get("covered") or []) if isinstance(c, int)]
    out["coverage_pct"] = round(len(set(cov)) / len(DIMS) * 100, 1)
    return out


def run_single(q):
    from finrag.agent import agent  # 惰性导入：--judge-only 自检时不加载 torch/ES
    t = time.time()
    ans, _ = agent.run(q, verbose=False)
    return ans, round(time.time() - t, 1)


def run_multi(q):
    import finrag.agent.orchestrator as orchestrator
    t = time.time()
    res = orchestrator.brief(q, parallel=True, verbose=False)
    return res["report"], round(time.time() - t, 1), res.get("timing", {})


def _avg(rows, idx, key):
    vals = [(r[idx].get(key, 0) or 0) for r in rows]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=len(QUESTIONS), help="只跑前 N 题")
    ap.add_argument("--judge-only", action="store_true", help="仅用一条样例自检 judge(省 GPU/ES)")
    args = ap.parse_args()

    if args.judge_only:  # 自检：judge 能否正常解析 JSON、算覆盖率
        demo = "宁德时代盈利预测：13家券商一致预期归母净利中位数XX亿；卖方共识看多全球龙头地位，分歧在储能；近期阳光等上调盈利预测；研报发布后30天实际涨跌+X%。"
        print("judge 自检：", judge(QUESTIONS[0], demo))
        return

    qs = QUESTIONS[:args.n]
    rows = []
    for i, q in enumerate(qs, 1):
        print(f"\n[{i}/{len(qs)}] {q}", flush=True)
        s_ans, s_t = run_single(q)
        s_j = judge(q, s_ans)
        print(f"  单Agent: 覆盖{s_j['coverage_pct']}% 完整{s_j.get('completeness')} 忠实{s_j.get('faithfulness')} ⏱{s_t}s", flush=True)
        m_ans, m_t, m_tm = run_multi(q)
        m_j = judge(q, m_ans)
        print(f"  多Agent: 覆盖{m_j['coverage_pct']}% 完整{m_j.get('completeness')} 忠实{m_j.get('faithfulness')} ⏱{m_t}s (内部加速{m_tm.get('parallel_speedup', '?')}x)", flush=True)
        rows.append((s_j, s_t, m_j, m_t))

    print("\n" + "=" * 58)
    print(f"{'指标':<16}{'单Agent':>13}{'多Agent':>13}{'Δ':>10}")
    print("-" * 58)
    for label, key in [("维度覆盖率%", "coverage_pct"), ("完整性", "completeness"), ("Faithfulness", "faithfulness")]:
        s, m = _avg(rows, 0, key), _avg(rows, 2, key)
        print(f"{label:<16}{s:>13}{m:>13}{m - s:>+10.1f}")
    st = round(sum(r[1] for r in rows) / len(rows), 1)
    mt = round(sum(r[3] for r in rows) / len(rows), 1)
    print(f"{'平均墙钟s':<16}{st:>13}{mt:>13}")
    print(f"\n题量 {len(qs)}；judge=DeepSeek。诚实注：LLM-judge 有偏差、题量小，量的是【相对覆盖提升】非绝对真值。")


if __name__ == "__main__":
    main()
