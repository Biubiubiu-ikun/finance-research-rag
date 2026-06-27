# -*- coding: utf-8 -*-
"""
eval_generation.py — 生成层(P4/P5)端到端评测：RAGAS 式 Faithfulness + Correctness

为什么：P4①②已有强【内部】防幻觉指标(数字逐字溯源100% / 句级entailment 0幻觉)，
但 P5 Agent 的自然语言回答缺【外部】质量量化。本脚本用 DeepSeek 做 LLM-judge，量化：
  - Faithfulness(忠实度/无幻觉)：把回答拆成原子陈述，逐条判断是否被该次【工具返回数据】支持。
                                 = 衡量 Agent 有没有"超出工具数据去编"。
  - Correctness(正确率)：对照 gold 关键点(来自已校验的确定性聚合结果)，判断回答是否正确覆盖。

评测集 gold 取自前面已验证的聚合事实(数字溯源/entailment 都过)，故 gold 本身可信。
续跑：Agent 回答(贵)缓存 data/eval/gen_runs/{id}.json；judge 每次重算(便于调)。
用法：python eval_generation.py            # 全量
      python eval_generation.py 1 2 3      # 只跑指定题号(冒烟)
"""
import os
import sys
import json
import finrag.agent.agent as agent
from finrag.analysis.aggregate_views import call_llm

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN_DIR = os.path.join(BASE, "data", "eval", "gen_runs")
OUT_FP = os.path.join(BASE, "data", "eval", "generation_report.md")

# 评测集：gold 为"语义关键点"(judge 判语义覆盖,非字面)。覆盖 数字/对比/观点/目标价/风险/检索 多类。
EVAL = [
    {"id": 1, "q": "宁德时代2026年归母净利润预测增速（中位数口径）大概是多少？",
     "gold": ["宁德时代2026年归母净利预计同比较快增长（两位数增速）"]},
    {"id": 2, "q": "对比宁德时代和隆基绿能2026年归母净利的卖方分歧，谁更大？",
     "gold": ["隆基绿能的卖方分歧明显大于宁德时代"]},
    {"id": 3, "q": "隆基绿能2025年的卖方一致预期是盈利还是亏损？2026年呢？",
     "gold": ["隆基绿能2025年一致预期盈利承压（亏损或微利）", "2026年盈利明显改善 / 扭亏为盈"]},
    {"id": 4, "q": "阳光电源的券商共识看多逻辑主要有哪些？",
     "gold": ["储能业务高增长", "AIDC/数据中心带来新增长曲线", "业绩超预期/盈利能力提升"]},
    {"id": 5, "q": "比亚迪有几家券商给出了明确目标价？大致区间或中位数？",
     "gold": ["多家券商给出明确目标价", "比亚迪目标价中位数处于百元级（约百余元）"]},
    {"id": 6, "q": "北方华创2025年归母净利预测的机构分歧度（CV）大概多少？",
     "gold": ["北方华创2025年归母净利预测存在一定机构分歧（CV 为个位数百分比量级）"]},
    {"id": 7, "q": "宁德时代被券商提到的主要风险有哪些（共识）？",
     "gold": ["下游需求不及预期", "市场竞争加剧", "原材料价格波动"]},
    {"id": 8, "q": "库里这些标的整体的券商评级方向是怎样的？",
     "gold": ["几乎清一色看多(买入/推荐/增持等)"]},
    {"id": 9, "q": "兆易创新的看多逻辑是什么？",
     "gold": ["存储周期上行/产品涨价", "端侧AI与国产替代", "NOR Flash或利基DRAM量价齐升"]},
    {"id": 10, "q": "宁德时代的海外产能布局在哪些国家？",
     "gold": ["德国", "匈牙利", "西班牙"]},
    # —— 扩充：覆盖新行业标的（gold 取稳健的定性/结构化要点；数值型 gold 待 ingest 后对齐聚合值再补）——
    {"id": 11, "q": "恒瑞医药的核心看多逻辑是什么？",
     "gold": ["创新药收入占比提升", "国际化/海外BD授权合作", "研发管线进入收获期"]},
    {"id": 12, "q": "这个知识库覆盖了哪些行业？",
     "gold": ["半导体", "新能源", "医药", "食品饮料", "银行", "军工", "家电", "通信/算力"]},
    {"id": 13, "q": "库里医药板块有哪些标的？",
     "gold": ["恒瑞医药", "迈瑞医疗", "药明康德"]},
    {"id": 14, "q": "宁德时代的储能业务有什么看点？",
     "gold": ["储能电池出货量快速增长", "储能成为第二增长曲线"]},
    {"id": 15, "q": "比亚迪的海外/出海情况如何？",
     "gold": ["海外销量增长", "出海/全球化布局推进"]},
    {"id": 16, "q": "中芯国际被券商看多的逻辑是什么？",
     "gold": ["国产替代/自主可控", "晶圆代工产能扩张", "先进制程推进"]},
    {"id": 17, "q": "阳光电源的主营和看点是什么？",
     "gold": ["光伏逆变器全球领先", "储能业务高增长"]},
    {"id": 18, "q": "恒瑞医药2026年盈利预计是增长还是下滑？",
     "gold": ["2026年归母净利预计同比增长"]},
    {"id": 19, "q": "比亚迪和宁德时代属于哪个行业？",
     "gold": ["新能源(动力电池/新能源车)"]},
    {"id": 20, "q": "药明康德属于哪个板块？主营是什么？",
     "gold": ["医药(CXO/医药外包研发生产服务)"]},
]

JUDGE_FAITH = """你是严格的事实核查员。下面是一次问答的【回答】和该次回答【可用的证据】(分析工具实际返回的数据)。
请把【回答】拆成若干条原子事实陈述，逐条判断它是否被【证据】支持(数值/事实能在证据中找到或推出)。
输出 JSON：{"claims":[{"claim":"...","supported":true/false}],"n_claims":N,"n_supported":M}
注意：只看证据是否支持，不依赖你自己的外部知识；措辞性/总结性的话若与证据一致也算支持。

【回答】
%s

【证据(工具返回数据 JSON)】
%s"""

JUDGE_CORRECT = """你是评测员。判断【回答】是否正确覆盖了每条【参考要点】(语义覆盖即可,不要求字面一致)。
输出 JSON：{"points":[{"point":"...","covered":true/false}],"n_points":N,"n_hit":M,"comment":"≤30字"}

【问题】%s
【参考要点】
%s
【回答】
%s"""


def get_run(item):
    os.makedirs(RUN_DIR, exist_ok=True)
    fp = os.path.join(RUN_DIR, f"{item['id']}.json")
    if os.path.exists(fp):
        return json.load(open(fp, encoding="utf-8"))
    print(f"  跑 Agent Q{item['id']}…")
    answer, trace = agent.run(item["q"], verbose=False)
    rec = {"id": item["id"], "q": item["q"], "answer": answer, "trace": trace}
    json.dump(rec, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return rec


def judge(item, rec):
    evidence = json.dumps([{"tool": t["tool"], "result": t["result"]} for t in rec["trace"]], ensure_ascii=False)[:9000]
    f = call_llm(JUDGE_FAITH % (rec["answer"], evidence), max_tokens=3000)
    c = call_llm(JUDGE_CORRECT % (item["q"], "\n".join(f"- {g}" for g in item["gold"]), rec["answer"]), max_tokens=1200)
    nf, sf = f.get("n_claims", 0), f.get("n_supported", 0)
    ng, hg = c.get("n_points", len(item["gold"])), c.get("n_hit", 0)
    return {"id": item["id"], "n_claims": nf, "n_supported": sf,
            "faith": (sf / nf if nf else 1.0), "n_gold": ng, "n_hit": hg,
            "correct": (hg / ng if ng else 0.0), "comment": c.get("comment", ""),
            "unsupported": [x["claim"] for x in f.get("claims", []) if not x.get("supported")],
            "tools": [t["tool"] for t in rec["trace"]]}


def main(ids=None):
    items = [e for e in EVAL if not ids or e["id"] in ids]
    rows = []
    for it in items:
        rec = get_run(it)
        try:
            r = judge(it, rec)
        except Exception as e:  # judge 偶发 JSON 截断等：跳过该题，不拖垮整轮
            print(f"  ✗ Q{it['id']} judge 失败(跳过)：{repr(e)[:80]}")
            continue
        rows.append(r)
        print(f"  Q{r['id']}: faith {r['n_supported']}/{r['n_claims']}={r['faith']*100:.0f}%  "
              f"correct {r['n_hit']}/{r['n_gold']}={r['correct']*100:.0f}%  tools={r['tools']}")
    # 汇总
    tc = sum(r["n_claims"] for r in rows); ts = sum(r["n_supported"] for r in rows)
    tg = sum(r["n_gold"] for r in rows); th = sum(r["n_hit"] for r in rows)
    faith = ts / tc * 100 if tc else 100
    corr = th / tg * 100 if tg else 0
    L = ["# 生成层(P4/P5)端到端评测报告", "",
         f"- 评测题 **{len(rows)}** 道（覆盖盈利预测/对比/观点/目标价/风险/检索）",
         f"- **Faithfulness(无幻觉率)：{ts}/{tc} = {faith:.1f}%**（回答陈述被工具返回数据支持的比例）",
         f"- **Correctness(正确率)：{th}/{tg} = {corr:.1f}%**（对照已校验聚合 gold 的关键点覆盖率）", "",
         "| # | 问题 | Faith | Correct | 调用工具 | 备注 |",
         "|---|---|---|---|---|---|"]
    for r, it in zip(rows, items):
        L.append(f"| {r['id']} | {it['q'][:22]}… | {r['n_supported']}/{r['n_claims']} | "
                 f"{r['n_hit']}/{r['n_gold']} | {','.join(dict.fromkeys(r['tools']))} | {r['comment']} |")
    bad = [r for r in rows if r["unsupported"]]
    if bad:
        L += ["", "## 未被证据支持的陈述（潜在幻觉，需核查）"]
        for r in bad:
            for u in r["unsupported"]:
                L.append(f"- Q{r['id']}: {u}")
    else:
        L += ["", "✅ 所有回答陈述均被工具返回数据支持，无幻觉。"]
    open(OUT_FP, "w", encoding="utf-8").write("\n".join(L))
    print(f"\nFaithfulness {faith:.1f}% | Correctness {corr:.1f}% → {OUT_FP}")


if __name__ == "__main__":
    ids = set(int(a) for a in sys.argv[1:] if a.isdigit()) or None
    main(ids)
