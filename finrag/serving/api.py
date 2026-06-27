# -*- coding: utf-8 -*-
"""
api.py — 金融研报智能分析 RAG 的 HTTP 服务（FastAPI）

把分析型 Agent + 聚合能力 + 增量入库 暴露成 REST 接口，供前端/下游系统调用(而非只有 Streamlit demo)。
  GET  /health                  健康检查
  GET  /stocks                  覆盖标的列表
  POST /chat   {question}        分析型 Agent 问答 → 回答 + 工具调度轨迹 + 【延迟/调用数/token/粗估成本】
  GET  /consensus/{stock}        盈利预测一致预期(结构化)
  GET  /views/{stock}            观点共识/分歧(结构化)
  GET  /revisions/{stock}        盈利预测修正(结构化)
  POST /ingest (multipart)       上传研报 PDF + 元数据 → 后台触发 ingest 全链路入库

启动：python -m uvicorn api:app --host 127.0.0.1 --port 8000
交互文档：http://127.0.0.1:8000/docs
"""
import os
import sys
import csv
import time
import subprocess
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from pydantic import BaseModel
import finrag.agent.agent as agent
import finrag.agent.orchestrator as orchestrator
from finrag.analysis.aggregate_forecast import all_stocks

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# DeepSeek deepseek-chat 粗估单价(元/百万token，输入输出混合估)，仅供量级参考，实际以账单为准
COST_PER_MTOKEN = 3.0

app = FastAPI(title="金融研报智能分析 RAG API", version="1.0",
              description="券商研报混合检索 + 跨机构聚合(双防幻觉) + 分析型 Agent 的 HTTP 服务")


class ChatReq(BaseModel):
    question: str
    reflect: bool = False  # 开启 Agent 自检(reflection)：答完自检不足则补调工具再答
    history: list = []     # 多轮对话上下文 [{role:'user'/'assistant', content:...}]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stocks")
def stocks():
    lst = [{"code": c, "name": n} for c, n in all_stocks()]
    return {"n_stocks": len(lst), "stocks": lst}


@app.post("/chat")
def chat(req: ChatReq):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")
    t0 = time.time()
    answer, trace = agent.run(req.question, verbose=False, reflect=req.reflect, history=req.history)
    toks = agent.RUN_STATS.get("total_tokens", 0)
    return {
        "answer": answer,
        "tools_used": [s["tool"] for s in trace],
        "trace": trace,
        "reflection": agent.RUN_STATS.get("reflection"),
        "metrics": {
            "latency_ms": int((time.time() - t0) * 1000),
            "llm_calls": agent.RUN_STATS.get("llm_calls", 0),
            "tool_calls": agent.RUN_STATS.get("tool_calls", 0),
            "total_tokens": toks,
            "est_cost_cny": round(toks / 1e6 * COST_PER_MTOKEN, 4),
            "reflected": agent.RUN_STATS.get("reflected", False),
        },
    }


class BriefReq(BaseModel):
    question: str
    parallel: bool = True  # 多Agent：并行执行 Workers(False=串行，用于测速对照)


@app.post("/brief")
def brief(req: BriefReq):
    """多Agent编排：Planner 拆子任务 → 并行 Workers → Aggregator 汇总成投研简报。
    返回 计划/各 Worker(维度·用时·工具·回答)/最终简报/timing(含并行加速比)。"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")
    res = orchestrator.brief(req.question, parallel=req.parallel, verbose=False)
    return {
        "question": res["question"], "stock": res["stock"],
        "plan": res["plan"], "report": res["report"], "timing": res["timing"],
        "workers": [{"id": w["id"], "focus": w["focus"], "question": w["question"],
                     "answer": w["answer"], "tools_used": w["tools_used"], "secs": w["secs"]}
                    for w in res["workers"]],
    }


@app.get("/cache_stats")
def cache_stats():
    h, m = agent.CACHE_STATS["hits"], agent.CACHE_STATS["misses"]
    tot = h + m
    return {"hits": h, "misses": m, "total": tot,
            "hit_rate": round(h / tot * 100, 1) if tot else None}


@app.post("/cache_clear")
def cache_clear():
    n = len(agent._MEM)
    agent._MEM.clear()
    agent.CACHE_STATS.update(hits=0, misses=0)
    return {"cleared": n, "status": "ok"}


@app.get("/consensus/{stock}")
def consensus(stock: str):
    return agent.t_forecast_consensus(stock)


@app.get("/views/{stock}")
def views(stock: str):
    return agent.t_view_consensus(stock)


@app.get("/revisions/{stock}")
def revisions(stock: str):
    return agent.t_forecast_revisions(stock)


def _run_ingest():
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    subprocess.run([sys.executable, "ingest.py"], cwd=BASE_DIR, env=env)


@app.post("/ingest")
async def ingest(background: BackgroundTasks, file: UploadFile = File(...),
                 code: str = Form(...), name: str = Form(""), org: str = Form(""),
                 date: str = Form(""), industry: str = Form("")):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF")
    rd = os.path.join(BASE_DIR, "data", "reports", code)
    os.makedirs(rd, exist_ok=True)
    open(os.path.join(rd, file.filename), "wb").write(await file.read())
    # 写/追加 metadata CSV（title 须等于 PDF 文件名以匹配元数据）
    title = os.path.splitext(file.filename)[0]
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
    background.add_task(_run_ingest)  # 入库重(含DeepDOC)，后台跑，立即返回
    return {"status": "started", "file": file.filename, "code": code,
            "message": "PDF 已保存、元数据已写入，入库管道后台运行中（约数分钟，完成后该标的自动可查）。"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
