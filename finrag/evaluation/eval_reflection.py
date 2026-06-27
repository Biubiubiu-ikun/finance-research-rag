# -*- coding: utf-8 -*-
"""
eval_reflection.py — Agent 自检(reflection)判定质量评测：精确率 / 召回率 / 准确率

reflect_check 的价值取决于它判得准不准：
  - 漏报(该触发却判充分,FN)→ 遗漏没补，自检失效；
  - 误报(不该触发却判不足,FP)→ 无谓重跑，浪费开销。
故构造 10 个 (问题, 回答, 已获数据, 期望是否触发) case，直接喂 reflect_check，
统计 精确率/召回率/准确率。不跑 Agent，仅 10 次 LLM judge，快。

用法：python eval_reflection.py
"""
import os
import sys
import finrag.agent.agent as agent
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT_FP = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "eval", "reflection_report.md")

# trigger=True 表示"回答残缺、应被自检判为不足(sufficient=False)"；False 表示"回答充分、应判 sufficient=True"
CASES = [
    # —— 充分(不该触发) ——
    {"q": "宁德时代2026年归母净利增速大概多少？", "trigger": False,
     "a": "宁德时代2026年归母净利中位约864亿元，同比+24.7%（来源：7家机构一致预期）。",
     "trace": [{"tool": "forecast_consensus", "result": {"stock": "宁德时代", "net_profit_yoy_median": {"2026": "+24.7%"}, "net_profit_yi": {"2026": {"median": 864.06}}}}]},
    {"q": "阳光电源的共识看多逻辑有哪些？", "trigger": False,
     "a": "阳光电源共识看多：①储能业务高增长 ②AIDC打开新增长曲线 ③业绩超预期。",
     "trace": [{"tool": "view_consensus", "result": {"stock": "阳光电源", "consensus_bull": [{"point": "储能高增长", "orgs": ["华龙"]}, {"point": "AIDC新曲线", "orgs": ["东吴"]}, {"point": "业绩超预期", "orgs": ["民生"]}]}}]},
    {"q": "比亚迪有几家券商给了目标价？", "trigger": False,
     "a": "共4家券商给出目标价（人民币口径），中位数126.75元。",
     "trace": [{"tool": "forecast_consensus", "result": {"stock": "比亚迪", "target_prices": {"n_given_cny": 4, "median_cny": 126.75, "range_cny": [117, 140]}}}]},
    {"q": "隆基绿能2025年是盈利还是亏损？", "trigger": False,
     "a": "隆基绿能2025年卖方一致预期为亏损，中位约-40亿元。",
     "trace": [{"tool": "forecast_consensus", "result": {"stock": "隆基绿能", "net_profit_yi": {"2025": {"median": -40.19}}}}]},
    {"q": "宁德时代主要风险有哪些？", "trigger": False,
     "a": "主要风险：下游需求不及预期、市场竞争加剧、原材料价格波动。",
     "trace": [{"tool": "view_consensus", "result": {"stock": "宁德时代", "consensus_risk": [{"point": "需求不及预期", "orgs": ["a"]}, {"point": "竞争加剧", "orgs": ["b"]}, {"point": "原材料价格波动", "orgs": ["c"]}]}}]},
    # —— 残缺(该触发) ——
    {"q": "对比宁德时代和隆基绿能2026年净利增速谁更高？", "trigger": True,
     "a": "宁德时代2026年归母净利增速约+24.7%。",  # 漏隆基、没对比
     "trace": [{"tool": "forecast_consensus", "result": {"stock": "宁德时代", "net_profit_yoy_median": {"2026": "+24.7%"}}}]},
    {"q": "阳光电源的看多逻辑和主要风险分别是什么？", "trigger": True,
     "a": "看多逻辑：储能高增长、AIDC新曲线。",  # 漏风险(数据里有)
     "trace": [{"tool": "view_consensus", "result": {"stock": "阳光电源", "consensus_bull": [{"point": "储能高增长", "orgs": ["x"]}], "consensus_risk": [{"point": "市场竞争加剧", "orgs": ["y"]}]}}]},
    {"q": "宁德时代2025到2027年归母净利预测分别多少？", "trigger": True,
     "a": "宁德时代2025年约693亿元。",  # 漏26/27
     "trace": [{"tool": "forecast_consensus", "result": {"stock": "宁德时代", "net_profit_yi": {"2025": {"median": 692.85}, "2026": {"median": 864.06}, "2027": {"median": 1064.94}}}}]},
    {"q": "比亚迪目标价有几家、区间多少？", "trigger": True,
     "a": "有4家券商给了目标价。",  # 漏区间(问了)
     "trace": [{"tool": "forecast_consensus", "result": {"stock": "比亚迪", "target_prices": {"n_given_cny": 4, "median_cny": 126.75, "range_cny": [117, 140]}}}]},
    {"q": "阳光电源2025年归母净利预测是多少？", "trigger": True,
     "a": "阳光电源是全球储能龙头，前景看好。",  # 答非所问(问数字答观点)
     "trace": [{"tool": "view_consensus", "result": {"stock": "阳光电源", "overall": "8家看多"}}]},
]


def main():
    tp = fp = fn = tn = 0
    lines = []
    for i, c in enumerate(CASES, 1):
        crit = agent.reflect_check(c["q"], c["a"], c["trace"])
        flagged = not crit.get("sufficient", True)  # 判不足 = 触发补充
        if c["trigger"] and flagged:
            tp += 1; tag = "TP✓(正确识别遗漏)"
        elif c["trigger"] and not flagged:
            fn += 1; tag = "FN✗(漏报:该补没补)"
        elif not c["trigger"] and flagged:
            fp += 1; tag = "FP✗(误报:无谓重跑)"
        else:
            tn += 1; tag = "TN✓(正确放行)"
        lines.append(f"| {i} | {'残缺' if c['trigger'] else '充分'} | {'判不足' if flagged else '判充分'} | {tag} | {'/'.join(crit.get('issues', []))[:40]} |")
        print(f"  case{i}: {tag}")

    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    acc = (tp + tn) / len(CASES)
    L = ["# Agent 自检(reflection)判定质量评测", "",
         f"- 构造 {len(CASES)} 个 case（{sum(c['trigger'] for c in CASES)} 残缺应触发 / {sum(not c['trigger'] for c in CASES)} 充分应放行）",
         f"- **精确率 Precision = {tp}/{tp+fp} = {prec*100:.0f}%**（判不足的里有多少真该补，越高越少无谓重跑）",
         f"- **召回率 Recall = {tp}/{tp+fn} = {rec*100:.0f}%**（真该补的里抓到多少，越高越少漏报）",
         f"- **准确率 = {tp+tn}/{len(CASES)} = {acc*100:.0f}%**", "",
         "| # | 回答 | 自检判定 | 结果 | 识别到的缺口 |",
         "|---|---|---|---|---|", *lines]
    open(OUT_FP, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    print(f"\nPrecision {prec*100:.0f}% / Recall {rec*100:.0f}% / Acc {acc*100:.0f}% → {OUT_FP}")


if __name__ == "__main__":
    main()
