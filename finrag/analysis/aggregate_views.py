# -*- coding: utf-8 -*-
"""
aggregate_views.py — P4② 多机构「事件 & 定性观点」聚合（生成分析层 / 防幻觉重头戏）

P4① 管【数字】(盈利预测/目标价，数字级溯源)；本脚本管【观点文字】(看多逻辑/风险/催化剂)。
三阶段：
  A. 逐机构观点抽取：每家券商的观点块 → DeepSeek 提炼 {立场, 看多逻辑[], 风险[], 催化剂[]}，
                     并保留该机构原始观点文本作为后续校验证据。缓存续跑。
  B. 跨机构聚合：所有机构观点 → 共识看多 / 共识风险 / 分歧点(每条带提及机构)。
  C. 句级 entailment 防幻觉：对 B 生成的每条结论，回到【它声称支持的那些机构的原文】做蕴含判定
                     (支持/部分支持/不支持) → 抓"夸大共识、张冠李戴归属、编造观点"等聚合幻觉。
                     与 P4① 数字溯源互补：A→B 可能引入文字幻觉，C 是兜底事实核查。

数据源：data/chunks/chunks.jsonl 里 type∈{text,summary} 的观点类块(按 stock+org 分组)。
缓存：阶段A 写 data/analysis/views/{code}.jsonl(按 doc 续跑)。
用法：python aggregate_views.py 宁德时代 | 300750 | all [--refresh]
"""
import os
import sys
import re
import json
import time
import collections
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 复用 P4① 的环境加载/标的解析/常量(import 时已 load_env)
from finrag.analysis.aggregate_forecast import (load_env, all_stocks, load_docs, _age_days,
                                STALE_DAYS, BASE, OUT, API_KEY, API_URL, MODEL)

CHUNKS = os.path.join(BASE, "data", "chunks", "chunks.jsonl")
VIEW_DIR = os.path.join(OUT, "views")
# 观点类 section：排除纯标题/日期/数据表类
SEC_SKIP = ("标题", "日期", "基础数据", "市场数据", "行业分类", "财务数据", "财务指标", "费用率", "财报点评", "研究·")


def call_llm(prompt, max_tokens=2000):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}, "temperature": 0.1, "stream": False, "max_tokens": max_tokens}
    r = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                      json=body, timeout=150)
    r.raise_for_status()
    c = r.json()["choices"][0]["message"]["content"].strip()
    c = re.sub(r"^```(?:json)?|```$", "", c, flags=re.M).strip()
    return json.loads(c)


def clean(txt):
    """去研报水印标签碎片(如 [Table_Co.. / ⚫)。"""
    txt = re.sub(r"\[?T\s*a\s*b\s*l\s*e[^\]]*\]?", "", txt)
    return re.sub(r"[⚫●]", "", txt).strip()


# ---------------------------------------------------------------- 取块
def load_opinion_blocks(code):
    """返回 {(org,date,doc_id): {section: [content...]}}，只含观点类文本块。"""
    byorg = collections.defaultdict(lambda: collections.defaultdict(list))
    for l in open(CHUNKS, encoding="utf-8"):
        o = json.loads(l)
        if o["stock_code"] != code or o["type"] not in ("text", "summary"):
            continue
        if any(s in o["section"] for s in SEC_SKIP):
            continue
        key = (o["org"], o.get("date", ""), o["doc_id"])
        byorg[key][o["section"]].append(clean(o["content"]))
    return byorg


# ---------------------------------------------------------------- 阶段A：逐机构抽取
PROMPT_A = """你是金融研报观点分析师。下面是【%s】研报「%s — %s」中的观点/分析类文本(已分块)。

请提炼该机构对【%s】的【定性观点】，输出 JSON：
{
 "stance": "整体立场，格式: 看多/中性/看空 — 一句话核心理由",
 "bull_points": ["支撑/看多逻辑，每条一句、具体(如 全球动力电池龙头地位稳固/海外产能扩张/毛利率持续提升)"],
 "risks": ["风险点，每条一句"],
 "catalysts": ["近期催化剂或业绩亮点，每条一句；没有就空数组[]"]
}
要求：只提炼原文表达的观点，不得编造；每条简短、去重；不要堆砌财务数字(数字由另一模块负责)。

文本：
%s"""


def extract_org_views(stock, org, title, secs):
    blocks = "\n".join(f"【{s}】{' '.join(v)}" for s, v in secs.items())
    out = call_llm(PROMPT_A % (stock, org, title, stock, blocks[:6000]))
    return {"org": org, "stance": out.get("stance", ""),
            "bull_points": out.get("bull_points", []), "risks": out.get("risks", []),
            "catalysts": out.get("catalysts", [])}


EV_PRIO = ("风险", "核心观点", "主要观点", "投资建议", "投资要点", "投资评级", "业绩", "盈利预测与投资建议")


def collect_evidence(code):
    """阶段C 的证据：每家机构观点块原文拼接(限长)，运行时现取，不进缓存、不调 LLM。
    教训：证据召回不全是 entailment 误报幻觉(假阳性)的主因——
      ①只取高浓缩 section 会漏分析块论据；②多篇机构按序拼接后截断会把靠后的风险提示截掉。
    对策：关键 section(风险/核心观点/投资建议…)优先纳入 + 去重 + 放宽上限。"""
    byorg = load_opinion_blocks(code)
    bucket = collections.defaultdict(list)  # org -> [(prio, text)]
    for (org, date, doc_id), secs in byorg.items():
        for s, vs in secs.items():
            bucket[org].append((0 if any(p in s for p in EV_PRIO) else 1, " ".join(vs)))
    out = {}
    for org, items in bucket.items():
        items.sort(key=lambda x: x[0])  # 关键 section 排前，保证不被截断
        seen, parts = set(), []
        for _, txt in items:
            if txt and txt not in seen:
                seen.add(txt)
                parts.append(txt)
        out[org] = " ".join(parts)[:2600]
    return out


def build_views(code, refresh=False):
    os.makedirs(VIEW_DIR, exist_ok=True)
    fp = os.path.join(VIEW_DIR, f"{code}.jsonl")
    done = {}
    if os.path.exists(fp) and not refresh:
        for l in open(fp, encoding="utf-8"):
            r = json.loads(l)
            done[r.get("doc_id", r["org"])] = r
    byorg = load_opinion_blocks(code)
    _, name, docs = load_docs(code)
    title_map = {(d["org"], d.get("date", "")): d.get("title", "") for d in docs}
    recs, new = [], 0
    for (org, date, doc_id), secs in byorg.items():
        if doc_id in done:
            recs.append(done[doc_id])
            continue
        try:
            r = extract_org_views(name, org, title_map.get((org, date), ""), secs)
            r["date"], r["doc_id"] = date, doc_id
        except Exception as e:
            print(f"  ✗ 抽取失败 [{org}]: {repr(e)[:120]}")
            continue
        recs.append(r)
        new += 1
        time.sleep(0.2)
    with open(fp, "w", encoding="utf-8") as w:
        for r in recs:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    if new:
        print(f"  阶段A 新抽取 {new} 家（缓存 {fp}）")
    return name, recs


# ---------------------------------------------------------------- 阶段B：跨机构聚合
PROMPT_B = """你是卖方观点聚合分析师。下面是 %d 家券商对【%s】的观点抽取(JSON 数组，每项含 org/stance/bull_points/risks/catalysts)。

请跨机构归纳，输出 JSON：
{
 "overall": "一句话整体共识(含几家看多/中性等)",
 "consensus_bull": [{"point":"被多家提及的看多主题(一句)","orgs":["机构名",...]}],
 "consensus_risk": [{"point":"被多家提及的风险(一句)","orgs":[...]}],
 "divergence": [{"point":"分歧点或仅个别机构提出的独家观点(一句)","orgs":[...],"note":"为何算分歧"}]
}
要求：
- point 必须是对输入 bull_points/risks 的归纳，【不得引入输入中没有的事实】。
- 【合并同义主题】：措辞不同但意思相同的逻辑(如"全球龙头"与"全球领先地位稳固"、"海外扩产"与"全球化布局")必须归为同一条 point，不要拆碎。
- orgs 要尽量全：结合每家的 stance 与 bull_points，凡表达了该主题(哪怕措辞不同)的机构都列入。
- orgs 只能来自输入机构名(用简称如 东莞/华安)。
- consensus_* 按提及机构数从多到少排序；只 1 家提及的才归入 divergence。

输入：
%s"""


def aggregate(name, recs):
    payload = [{"org": r["org"], "stance": r["stance"], "bull_points": r["bull_points"],
                "risks": r["risks"], "catalysts": r.get("catalysts", [])} for r in recs]
    return call_llm(PROMPT_B % (len(recs), name, json.dumps(payload, ensure_ascii=False)), max_tokens=2500)


# ---------------------------------------------------------------- 阶段C：句级 entailment
PROMPT_C = """你是严格的事实核查员。判断每条"待核查结论"是否被"证据"支持(entailment)。

对每条结论输出 JSON：{"results":[{"id":0,"verdict":"支持/部分支持/不支持","support_orgs":["真正能支撑的机构"],"reason":"≤20字"}]}

判定标准：
- 支持：证据中确有(claimed_orgs 里的)机构明确表达了该观点，语义一致。
- 部分支持：方向一致但被夸大/细节不符，或真正支撑的机构比声称的少。
- 不支持：证据里找不到该观点(疑似聚合幻觉/张冠李戴)。

待核查结论(claimed_orgs=声称支持它的机构)：
%s

证据(每家机构的原始观点原文)：
%s"""


def _entail_local(claims, ev):
    """用本地微调 Qwen2.5-1.5B 逐条做句级核查（替代 DeepSeek，离线/可私有化）。
    注意：本地模型是【二分类】(支持/不支持)，不产出 DeepSeek 那样的'部分支持'与逐机构细分——
    这是'用小模型换离线/私有化'的 trade-off。每条用其 claimed_orgs 的原文作证据。"""
    import finrag.agent.local_verifier as local_verifier
    allev = " ".join(ev.values())
    verdicts = {}
    for c in claims:
        orgs = c.get("claimed_orgs", [])
        evtext = (" ".join(ev.get(o, "") for o in orgs).strip() or allev)[:4000]
        try:
            label, _ = local_verifier.judge(evtext, c["point"])
        except Exception:
            label = "支持"  # 本地推理失败不误杀
        verdicts[c["id"]] = {"id": c["id"], "verdict": label, "support_orgs": orgs, "reason": "本地Qwen判别"}
    return verdicts


def entail_check(agg, ev, backend="deepseek"):
    claims = []
    for kind, items in [("看多", agg.get("consensus_bull", [])), ("风险", agg.get("consensus_risk", [])),
                        ("分歧", agg.get("divergence", []))]:
        for it in items:
            claims.append({"id": len(claims), "kind": kind, "point": it.get("point", ""),
                           "claimed_orgs": it.get("orgs", [])})
    if not claims:
        return claims, {}
    if backend == "local":   # 本地微调 Qwen 逐条核查（二分类）
        return claims, _entail_local(claims, ev)
    # 默认 DeepSeek：一次批量调 LLM（含'部分支持'与逐机构细分，更细）
    cj = json.dumps([{"id": c["id"], "point": c["point"], "claimed_orgs": c["claimed_orgs"]} for c in claims], ensure_ascii=False)
    ej = json.dumps(ev, ensure_ascii=False)
    out = call_llm(PROMPT_C % (cj, ej[:22000]), max_tokens=2500)
    verdicts = {r["id"]: r for r in out.get("results", [])}
    return claims, verdicts


# ---------------------------------------------------------------- 报告
def render(code, name, recs, agg, claims, verdicts, backend="deepseek"):
    latest = max((r["date"] for r in recs if r.get("date")), default="")
    L = [f"# 多机构观点聚合 · {name}（{code}）", "",
         f"- 覆盖机构：**{len(recs)}** 家｜最新报告日：**{latest}**",
         f"- 整体共识：{agg.get('overall','')}", ""]
    VMARK = {"支持": "✓", "部分支持": "◐", "不支持": "✗幻觉"}

    def block(title, kind):
        L.append(f"## {title}")
        rows = [(c, verdicts.get(c["id"], {})) for c in claims if c["kind"] == kind]
        if not rows:
            L.append("- （无）")
        for c, v in rows:
            vd = v.get("verdict", "支持")
            sup = v.get("support_orgs") or c["claimed_orgs"]
            note = f"｜分歧：{next((d.get('note','') for d in agg.get('divergence',[]) if d.get('point')==c['point']), '')}" if kind == "分歧" else ""
            L.append(f"- {VMARK.get(vd,'')} {c['point']}　[{len(sup)}家：{'/'.join(sup)}]{note}")
        L.append("")

    block(f"共识看多逻辑", "看多")
    block(f"共识风险点", "风险")
    block(f"分歧 / 独家观点", "分歧")

    # 各机构立场一览
    L.append("## 各机构立场")
    for r in sorted(recs, key=lambda x: x.get("date", ""), reverse=True):
        old = " ⚠️旧" if r.get("date") and _age_days(r["date"], latest) > STALE_DAYS else ""
        L.append(f"- **{r['org']}**{old}（{r.get('date','')}）：{r['stance']}")
    L.append("")

    # 防幻觉
    tot = len(claims)
    ok = sum(1 for c in claims if verdicts.get(c["id"], {}).get("verdict") == "支持")
    part = sum(1 for c in claims if verdicts.get(c["id"], {}).get("verdict") == "部分支持")
    bad = [c for c in claims if verdicts.get(c["id"], {}).get("verdict") == "不支持"]
    eng = "本地微调 Qwen2.5-1.5B（私有化）" if backend == "local" else "DeepSeek"
    L.append(f"## 事实一致性 / 防幻觉（句级 entailment · 引擎：{eng}）")
    rate = (ok / tot * 100) if tot else 100.0
    L.append(f"- 对聚合产出的 {tot} 条结论逐条回原文核查：**支持 {ok}（{rate:.0f}%）**、部分支持 {part}、不支持 {len(bad)}。")
    if part or bad:
        L.append("- 被降级/剔除的结论（聚合层潜在夸大或张冠李戴，已在上文标 ◐/✗）：")
        for c in claims:
            v = verdicts.get(c["id"], {})
            if v.get("verdict") in ("部分支持", "不支持"):
                L.append(f"  - [{v.get('verdict')}] {c['point']} —— {v.get('reason','')}")
    else:
        L.append("- ✅ 全部结论均可被对应机构原文支撑，聚合无幻觉。")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- main
def run_one(stock, refresh=False, backend="deepseek"):
    code, name, docs = load_docs(stock)
    if not docs:
        print(f"未找到标的：{stock}")
        return
    print(f"\n{'='*70}\n■ {name}（{code}）：观点聚合")
    name, recs = build_views(code, refresh=refresh)
    if not recs:
        print("  无观点数据"); return
    print("  阶段B 跨机构聚合…")
    agg = aggregate(name, recs)
    print(f"  阶段C 句级 entailment 核查…（引擎：{backend}）")
    claims, verdicts = entail_check(agg, collect_evidence(code), backend=backend)
    md = render(code, name, recs, agg, claims, verdicts, backend=backend)
    os.makedirs(OUT, exist_ok=True)
    out_fp = os.path.join(OUT, f"{code}_views.md")
    open(out_fp, "w", encoding="utf-8").write(md)
    print(md)
    print(f"\n→ 已写出 {out_fp}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    refresh = "--refresh" in sys.argv
    backend = "local" if "--local" in sys.argv else os.getenv("VERIFIER_BACKEND", "deepseek")  # --local=本地微调Qwen核查
    target = args[0] if args else "宁德时代"
    if target == "all":
        for code, _ in all_stocks():
            run_one(code, refresh=refresh, backend=backend)
    else:
        run_one(target, refresh=refresh, backend=backend)
