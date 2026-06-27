# -*- coding: utf-8 -*-
"""
app.py — 金融研报智能分析 · Streamlit Demo UI

把前面所有"看不见"的工作变成可演示的界面：
  - 💬 智能问答(Agent)：自由提问 → 分析型 Agent 多工具调度 → 回答 + 【工具调度轨迹(可溯源)】
                        每个数字/结论取自哪个工具、哪家机构，一眼可查。
  - 📊 标的速览(聚合成果)：选标的 → 直接渲染盈利预测一致预期 + 观点共识/分歧，
                        自带 ✓/◐/✗ 防幻觉标记(数字级溯源 + 句级 entailment)。

启动：streamlit run app.py
"""
import os
import sys
import glob
import json
import csv
import subprocess
import streamlit as st
import finrag.agent.agent as agent
import finrag.agent.orchestrator as orchestrator
import finrag.agent.local_verifier as local_verifier
from finrag.agent import harness  # 自研 Agent 运行时门面（智能问答/多维简报都经它进入）
from finrag.analysis.aggregate_forecast import all_stocks, OUT, normalize_industry

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(BASE_DIR, "data", "reports")
META_DIR = os.path.join(BASE_DIR, "data", "metadata")
STRUCT_DIR = os.path.join(BASE_DIR, "data", "structured")

st.set_page_config(page_title="金融研报智能分析 RAG", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def stocks():
    return all_stocks()


@st.cache_data(show_spinner=False)
def read_md(path):
    return open(path, encoding="utf-8").read() if os.path.exists(path) else ""


@st.cache_data(show_spinner=False)
def code2industry():
    """股票代码 → 大类行业（单一数据源 watchlist.json）。研报里的 industry 字段各家券商写法五花八门
    (Auto/China Auto/汽车、医药/医药生物/化学制剂/医疗器械…)，按【代码】归类比按字符串归一稳得多。"""
    fp = os.path.join(BASE_DIR, "watchlist.json")
    if not os.path.exists(fp):
        return {}
    return {s["code"]: s["industry"] for s in json.load(open(fp, encoding="utf-8")).get("stocks", [])}


@st.cache_data(show_spinner=False)
def corpus_stats():
    """从 structured 目录实时统计 标的数/研报篇数/行业集合（数据驱动，随入库自动更新）。"""
    files = glob.glob(os.path.join(STRUCT_DIR, "*", "*.json"))
    c2i = code2industry()
    codes, inds = set(), set()
    for fp in files:
        try:
            s = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        code = s.get("stock_code")
        if code:
            codes.add(code)
            ind = c2i.get(code) or normalize_industry(s.get("industry", ""))  # 先按代码归大类，库外标的兜底再按字段归一
            if ind:
                inds.add(ind)
    names = [n for _, n in all_stocks()]
    return len(codes), len(files), sorted(inds), names


def save_upload_and_ingest(pdf_file, code, name, org, date, industry):
    """保存上传 PDF + 写 metadata 行 + 跑 ingest 全链路。返回 (ok, 日志)。"""
    os.makedirs(os.path.join(REPORTS_DIR, code), exist_ok=True)
    open(os.path.join(REPORTS_DIR, code, pdf_file.name), "wb").write(pdf_file.getbuffer())
    # 写/追加 metadata CSV（title 必须等于 PDF 文件名，extract_structure 据此匹配元数据）
    title = os.path.splitext(pdf_file.name)[0]
    os.makedirs(META_DIR, exist_ok=True)
    mfp = os.path.join(META_DIR, f"{code}_page1.csv")
    is_new = not os.path.exists(mfp)
    with open(mfp, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["title", "org_name", "publish_date", "stock_name", "stock_code",
                        "industry_name", "rating_name", "info_code", "url"])
        w.writerow([title, org, date, name, code, industry, "", "", ""])
    # 独立进程跑 ingest（隔离 torch/DeepDOC，不污染 Streamlit 进程）
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "ingest.py"], cwd=BASE_DIR, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.returncode and r.stderr else "")
    return r.returncode == 0, log


# ----------------------------------------------------- 引用来源提取(inline 溯源)
def collect_citations(trace):
    """从工具调度轨迹结构化提取「引用来源」，供回答下方展示(可点开看依据)。"""
    cites = []
    for s in trace:
        fn, res = s["tool"], s["result"]
        if not isinstance(res, dict) or res.get("error"):
            continue
        if fn == "retrieve":
            for r in res.get("results", []):
                cites.append((f"{r['org']}·{r['date']}·{r['section']}", r["content"]))
        elif fn == "forecast_consensus":
            cites.append((f"{res.get('stock')}·盈利预测聚合",
                          f"{res.get('n_orgs','?')} 家机构一致预期，最新报告 {res.get('latest_report','')}；数字均逐字溯源"))
        elif fn == "forecast_revisions":
            cites.append((f"{res.get('stock')}·盈利预测修正",
                          f"修正方向分布 {res.get('direction_dist', {})}"))
        elif fn == "view_consensus":
            cites.append((f"{res.get('stock')}·观点聚合(句级entailment)", res.get("overall", "")))
    return cites


# ----------------------------------------------------- 工具轨迹渲染(可溯源)
def render_tool(step):
    fn, res = step["tool"], step["result"]
    if not isinstance(res, dict):
        st.write(res); return
    if res.get("error"):
        st.error(res["error"]); return
    if fn == "forecast_consensus":
        st.caption(f"{res.get('stock')} · {res.get('n_orgs')}家机构 · 最新 {res.get('latest_report')} · 数字已逐字溯源")
        npd = res.get("net_profit_yi", {})
        if npd:
            rows = [{"年份": f"{y}E", "净利中位数(亿)": v["median"], "区间(亿)": f"{v['min']}~{v['max']}",
                     "分歧CV": "盈亏分歧" if v["cv_pct"] is None else f"{v['cv_pct']}%", "机构数": v["n"]}
                    for y, v in npd.items()]
            st.markdown("**归母净利润一致预期**")
            st.table(rows)
        if res.get("net_profit_yoy_median"):
            st.write("净利同比(中位数)：", res["net_profit_yoy_median"])
        if res.get("rating_dist"):
            st.write("评级方向：", res["rating_dist"])
        tp = res.get("target_prices")
        if tp and tp.get("items"):
            st.write(f"目标价（人民币口径）：{tp['n_given_cny']} 家给出，中位数 {tp['median_cny']} 元，区间 {tp['range_cny']}")
            st.write({t["org"]: f"{t['val']}{t['cur']}" for t in tp["items"]})
        if res.get("unit_anomaly_orgs"):
            st.warning("量纲异常已剔除：" + "、".join(res["unit_anomaly_orgs"]))
    elif fn == "view_consensus":
        st.caption(f"{res.get('stock')} · 观点已过句级 entailment 核查")
        st.write("**整体共识**：", res.get("overall", ""))
        for label, key in [("✅ 共识看多", "consensus_bull"), ("⚠️ 共识风险", "consensus_risk"), ("🔀 分歧/独家", "divergence")]:
            items = res.get(key, [])
            if items:
                st.markdown(f"**{label}**")
                for it in items:
                    orgs = it.get("orgs", [])
                    st.markdown(f"- {it.get('point','')}　`[{len(orgs)}家: {'/'.join(orgs)}]`")
    elif fn == "retrieve":
        for r in res.get("results", []):
            st.markdown(f"> **{r['org']}**（{r['date']}·{r['section']}）：{r['content']}")
    elif fn == "compute":
        st.code(f"{res.get('expression')} = {res.get('result')}")
    elif fn == "list_stocks":
        st.table(res.get("stocks", []))
    else:
        st.json(res)


# ----------------------------------------------------- 侧边栏
with st.sidebar:
    st.header("📈 金融研报智能分析 RAG")
    _ns, _nr, _inds, _names = corpus_stats()
    st.markdown(
        f"券商研报问答与生成分析。\n\n"
        f"**覆盖 {_ns} 标的 / {_nr} 篇研报**　行业：{'、'.join(_inds)}\n\n"
        f"标的：{'、'.join(_names)}\n\n"
        "**链路**：DeepDOC 解析 → 父子分块 → ES 混合检索(BM25+领域微调bge+加权RRF+rerank) "
        "→ 跨机构聚合 → 分析型 Agent。\n\n"
        "**双防幻觉**：数字级溯源（每个数字回原表逐字核对）+ 句级 entailment（每条观点回原文蕴含判定）。"
    )
    st.divider()
    if st.button("🔄 刷新数据缓存", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("新研报入库后点此刷新；新标的会自动出现在下拉中（UI 数据驱动，无需改代码）。")
    st.caption("数据截至 2025-12；DeepSeek 生成。仅供技术演示，非投资建议。")

st.title("金融研报智能分析")

tab1, tab_brief, tab2, tab_verify, tab3 = st.tabs(
    ["💬 智能问答 (Agent)", "🧭 多维简报 (多Agent)", "📊 标的速览 (聚合成果)",
     "🔬 事实核查 (本地蒸馏)", "➕ 上传研报入库"])

# ===================================================== Tab1 智能问答（多轮对话）
with tab1:
    st.caption("分析型 Agent 自主调度 检索/聚合/计算/入库 工具；**支持多轮追问**。每轮答案下可展开工具调度轨迹溯源。")
    if "chat" not in st.session_state:
        st.session_state.chat = []   # [{role,content}] 多轮上下文(传给 agent.run history)
    cc1, cc2 = st.columns([3, 1])
    reflect = cc1.checkbox("🔁 开启自检（reflection）")
    if cc2.button("🗑 清空对话", use_container_width=True):
        st.session_state.chat = []
        st.rerun()
    ask = None
    with st.expander("💡 示例问题（点击直接提问）", expanded=not st.session_state.chat):
        for i, ex in enumerate(agent.DEMO):
            if st.button(ex, key=f"ex{i}", use_container_width=True):
                ask = ex
    # 渲染历史对话
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    typed = st.chat_input("问点什么（支持追问）…")
    if typed:
        ask = typed
    if ask:
        st.session_state.chat.append({"role": "user", "content": ask})
        with st.chat_message("user"):
            st.markdown(ask)
        with st.chat_message("assistant"):
            with st.spinner("Agent 调度中…"):
                try:
                    history = st.session_state.chat[:-1][-6:]  # 最近 3 轮上下文(防 token 膨胀)
                    answer, trace = harness.default.run(ask, reflect=reflect, history=history)  # 经自研 Agent 运行时
                    refl = harness.default.stats.get("reflection")
                except Exception as e:
                    answer, trace, refl = f"出错了：{e}", [], None
            st.markdown(answer)
            if refl is not None:
                st.caption("🔁 自检：回答充分" if refl.get("sufficient", True)
                           else "🔁 自检发现不足并已补充：" + "；".join(refl.get("issues", [])))
            cites = collect_citations(trace)
            if cites:
                with st.expander(f"📎 引用来源（{len(cites)}）"):
                    for i, (src, detail) in enumerate(cites, 1):
                        st.markdown(f"**[{i}] {src}**")
                        st.caption(detail)
            if trace:
                with st.expander(f"🔍 工具调度轨迹（{len(trace)} 步）"):
                    for i, step in enumerate(trace, 1):
                        a = "，".join(f"{k}={v}" for k, v in step["args"].items())
                        st.markdown(f"**步骤 {i}：{step['tool']}（{a}）**")
                        render_tool(step)
        st.session_state.chat.append({"role": "assistant", "content": answer})

# ===================================================== Tab 多维简报（多Agent编排）
with tab_brief:
    st.caption("**多Agent编排**：Planner 把复杂任务拆成多个【聚焦子任务】→ 多个 Worker【并行】分维度调研"
               "(盈利预测/观点/修正/股价) → Aggregator 汇总成结构化投研简报。**卖点：并行提速 + 分工聚焦**。")
    bq = st.text_input("复杂任务（多维度问题最能体现编排价值）",
                       value="出一份宁德时代的多维投研简报", key="brief_q")
    cb1, cb2 = st.columns([1, 3])
    par = cb1.checkbox("并行执行 Worker", value=True, key="brief_par",
                       help="关闭则串行，可对照看并行加速比")
    go = cb2.button("🧭 生成投研简报", type="primary", key="brief_go", disabled=not bq.strip())
    if go and bq.strip():
        with st.spinner("Planner 规划 → Workers 并行调研 → Aggregator 汇总…"):
            try:
                res = harness.default.run_parallel(bq.strip(), parallel=par)  # 多 Agent 编排经运行时
            except Exception as e:
                res, _err = None, e
                st.error(f"出错了：{e}")
        if res:
            t = res["timing"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("子任务数", len(res["plan"]))
            m2.metric("Worker 墙钟", f"{t['workers_wall_secs']}s")
            m3.metric("并行加速", f"{t['parallel_speedup']}x",
                      help=f"各 Worker 用时之和 {t['workers_sum_secs']}s（串行约需这么久）")
            m4.metric("总用时", f"{t['total_secs']}s")
            with st.expander(f"🧭 执行计划与各 Worker 调研（{len(res['workers'])} 个 · 可溯源）"):
                for w in res["workers"]:
                    st.markdown(f"**#{w['id']} 〔{w['focus']}〕**　⏱{w['secs']}s　🔧{w['tools_used']}")
                    st.caption(f"子问题：{w['question']}")
                    st.markdown(w["answer"])
                    st.divider()
            st.markdown(res["report"])
    st.caption("单标的多维任务（投研简报/全面分析）最能体现并行+分工；窄问题 Planner 会自动只拆 1 个子任务。")

# ===================================================== Tab2 标的速览
with tab2:
    opts = {f"{name}（{code}）": code for code, name in stocks()}
    label = st.selectbox("选择标的", list(opts.keys()))
    code = opts[label]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💰 盈利预测一致预期")
        md = read_md(os.path.join(OUT, f"{code}_consensus.md"))
        st.markdown(md or "_暂无，请先运行 aggregate_forecast.py_")
    with c2:
        st.subheader("🗣️ 多机构观点聚合")
        vc1, vc2 = st.columns([2, 1])
        veng = vc1.radio("句级核查引擎", ["DeepSeek（默认）", "本地微调 Qwen"], horizontal=True,
                         key=f"veng_{code}", help="切换阶段C句级 entailment 核查用哪个模型；本地=可私有化")
        if vc2.button("↻ 重新核查", key=f"vrun_{code}", help="用所选引擎重跑句级核查（约1分钟；本地引擎首次会加载模型）"):
            be_local = "Qwen" in veng
            with st.spinner(f"用「{veng}」重新核查观点…"):
                env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
                cmd = [sys.executable, "aggregate_views.py", code] + (["--local"] if be_local else [])
                try:
                    r = subprocess.run(cmd, cwd=BASE_DIR, env=env, capture_output=True,
                                       text=True, encoding="utf-8", errors="replace", timeout=420)
                    ok, errlog = r.returncode == 0, (r.stderr or r.stdout or "")
                except subprocess.TimeoutExpired:
                    ok, errlog = False, "超时（>7分钟）"
            if ok:
                st.cache_data.clear()
                st.success(f"已用「{veng}」重新核查，下方已刷新。")
            else:
                st.error("重核失败：" + errlog[-500:])
        md = read_md(os.path.join(OUT, f"{code}_views.md"))
        st.markdown(md or "_暂无，请先运行 aggregate_views.py_")
    st.divider()
    st.subheader("📈 盈利预测修正追踪（时间演化）")
    md = read_md(os.path.join(OUT, f"{code}_revisions.md"))
    st.markdown(md or "_暂无，请先运行 track_revisions.py_")
    st.divider()
    st.subheader("📉 研报 vs 实际股价（有效性回测）")
    md = read_md(os.path.join(OUT, f"{code}_price.md"))
    st.markdown(md or "_暂无，请先运行 price_analysis.py_")

# ===================================================== Tab 事实核查（本地蒸馏小模型）
with tab_verify:
    st.caption("**句级事实核查的『可私有化』版本**：原本调 DeepSeek 的「结论是否被研报片段支持」判别，"
               "用 **DeepSeek-R1 思维链蒸馏 + QLoRA 微调的 Qwen2.5-1.5B**(本地离线)替代。"
               "输出不只给『支持/不支持』，还给**为什么**的推理链（可解释）。准确率 base 68% → 蒸馏后 92%，逼近老师 95.5%。")
    EXAMPLES = {
        "① 忠实结论（应判：支持）": (
            "公司2025年前三季度归母净利润同比增长32%，创新药收入占比提升至58%，海外授权(BD)合作密集落地。",
            "公司2025年前三季度归母净利润实现同比增长，创新药收入占比提升至58%。"),
        "② 夸大/编造（应判：不支持）": (
            "公司2025年前三季度归母净利润同比增长32%，创新药收入占比提升至58%，海外授权(BD)合作密集落地。",
            "公司2025年净利润同比下滑，创新药业务收缩。"),
        "③ 张冠李戴的数字（应判：不支持）": (
            "宁德时代2024年动力电池全球市占率约37%，储能电池出货量同比增长显著。",
            "宁德时代2024年动力电池全球市占率已超过50%，稳居第一。"),
    }
    pick = st.selectbox("载入示例（也可自行修改）", list(EXAMPLES.keys()), key="vf_ex")
    ev0, cc0 = EXAMPLES[pick]
    ev = st.text_area("研报片段（依据）", value=ev0, height=110, key="vf_ev")
    cc = st.text_area("待核查结论", value=cc0, height=70, key="vf_cc")
    engine = st.radio("核查引擎（自行选择用哪个）",
                      ["本地微调 Qwen（离线）", "DeepSeek（云端）", "两者对照"],
                      horizontal=True, key="vf_engine",
                      help="本地=蒸馏的 Qwen2.5-1.5B(可私有化)；DeepSeek=原项目所用云端核查；对照=并排看学生 vs 老师")
    run_vf = st.button("🔬 核查", type="primary", key="vf_go", disabled=not (ev.strip() and cc.strip()))
    if engine != "DeepSeek（云端）" and not local_verifier.available():
        st.warning("⏳ 本地 1.5B base 权重还没就位（正在下载 ~3GB）；可先选「DeepSeek（云端）」核查。")
    if run_vf and ev.strip() and cc.strip():
        use_local = engine in ("本地微调 Qwen（离线）", "两者对照")
        use_ds = engine in ("DeepSeek（云端）", "两者对照")
        cols = st.columns(2 if engine == "两者对照" else 1)
        slot, local_label = 0, None
        if use_local:
            with cols[slot]:
                with st.spinner("本地 1.5B+LoRA 推理中（首次加载约十几秒）…"):
                    try:
                        local_label, reasoning = local_verifier.judge(ev.strip(), cc.strip())
                    except Exception as e:
                        local_label, reasoning = None, None
                        st.error(f"本地推理出错：{e}")
                if local_label:
                    c = {"支持": "green", "不支持": "red"}.get(local_label, "orange")
                    st.markdown(f"**本地学生（Qwen2.5-1.5B + LoRA）**：:{c}[**{local_label}**]")
                    if reasoning:
                        with st.expander("🧠 推理链（蒸馏自 DeepSeek-R1 的思维过程）", expanded=True):
                            st.write(reasoning)
            slot += 1
        if use_ds:
            with cols[slot]:
                with st.spinner("DeepSeek 云端判别中…"):
                    try:
                        ds_label = local_verifier.teacher_judge(ev.strip(), cc.strip())
                        c = {"支持": "green", "不支持": "red"}.get(ds_label, "orange")
                        st.markdown(f"**DeepSeek（云端）**：:{c}[**{ds_label}**]")
                        if engine == "两者对照" and local_label:
                            st.caption("✅ 本地与云端一致" if ds_label == local_label else "⚠️ 本地与云端不一致")
                    except Exception as e:
                        st.error(f"DeepSeek 调用失败：{e}")
    st.caption("本质：把 RAG 的事实核查从『调闭源 API』换成『可私有化、可解释的本地小模型』。"
               "训练/评测见 lora_finetune/（R1 思维链蒸馏 + QLoRA，师生四方对照独立 gold）。")

# ===================================================== Tab3 上传研报入库
with tab3:
    st.caption("上传新研报 PDF → 一键走完 解析/结构化/分块/入ES/聚合 全链路 → 新标的自动上架（侧栏统计、下拉、Agent 同步更新，无需改代码）。"
               "含 DeepDOC 解析，约数分钟，请勿关闭页面。")
    up = st.file_uploader("研报 PDF 文件", type=["pdf"])
    c1, c2, c3 = st.columns(3)
    in_code = c1.text_input("股票代码 *", placeholder="如 600183")
    in_name = c2.text_input("股票名称", placeholder="如 生益科技")
    in_ind = c3.text_input("行业", placeholder="如 覆铜板")
    c4, c5 = st.columns(2)
    in_org = c4.text_input("机构", placeholder="如 国信证券")
    in_date = c5.text_input("报告日期", placeholder="2025-07-21")
    ready = bool(up and in_code.strip())
    if st.button("🚀 入库", type="primary", disabled=not ready):
        with st.status("入库中：解析 → 结构化 → 分块 → 入 ES → 聚合（约数分钟）…", expanded=True) as status:
            try:
                ok, log = save_upload_and_ingest(up, in_code.strip(), in_name.strip(),
                                                 in_org.strip(), in_date.strip(), in_ind.strip())
            except Exception as e:
                ok, log = False, repr(e)
            st.code((log or "")[-3000:])
            if ok:
                status.update(label="✅ 入库完成", state="complete")
                st.cache_data.clear()
                st.success(f"「{in_name or in_code}」入库完成！数据已刷新——切到「标的速览」查看，或在「智能问答」直接提问。")
            else:
                status.update(label="✗ 入库失败（见上方日志）", state="error")
    if not ready:
        st.info("请先上传 PDF 并填写股票代码（必填）；名称/机构/日期/行业可留空（缺的由 LLM 从研报抽取）。")
