# -*- coding: utf-8 -*-
"""
make_sft_data.py — 用 DeepSeek【蒸馏】造 entailment(蕴含判别) LoRA SFT 数据

任务设计：给定【研报片段(证据)】+【结论】，判断结论是否被证据支持(支持/不支持)。
  这正是本项目 P4② 里 DeepSeek 做的"句级 entailment 防幻觉"核查——把它【蒸馏】到一个小开源
  模型(Qwen2.5-1.5B)，训出来即可【替代 DeepSeek】做该子任务，也证明"能训 LLM"(算法岗门票)。

造数据：对每个研报观点块，让 DeepSeek 生成
  - faithful  : 忠实于片段的结论  → label 支持
  - unfaithful: 看似相关但矛盾/夸大/编造的结论 → label 不支持
每块产出 2 条样本(正/负)，输出 Qwen chat messages 格式 jsonl，8:2 切 train/val。

本地跑(我这边)生成数据 → 连同 train_lora.py 打包给云端；用户云端无需重跑本脚本。
用法：python make_sft_data.py [采样块数，默认180]
"""
import os
import sys
import re
import json
import glob
import time
import random
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CHUNKS = os.path.join(ROOT, "data", "chunks", "chunks.jsonl")
OUT_DIR = os.path.join(HERE, "data")


def load_env():
    for p in [os.path.join(ROOT, ".env"), os.path.join(os.path.dirname(ROOT), "ai面试八股rag", ".env")]:
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

INSTR = "你是金融研报事实核查员。判断【结论】是否被【研报片段】支持，只回答两个字：支持 或 不支持。"
GEN_PROMPT = """下面是一段研报内容。请生成两个针对它的一句话结论：
1) faithful：完全忠实于原文(方向/数字/事实都正确)的结论；
2) unfaithful：看似相关但【与原文矛盾、或夸大程度、或编造原文没有的数字/事实】的结论(要像真的、有迷惑性)。
只输出 JSON：{"faithful":"...","unfaithful":"..."}

研报内容：
%s"""

OPINION_KW = ("观点", "建议", "分析", "风险", "业绩", "评级", "亮点", "逻辑", "布局", "预测")


def call(prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}, "temperature": 0.7, "stream": False}
    r = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                      json=body, timeout=90)
    r.raise_for_status()
    c = r.json()["choices"][0]["message"]["content"].strip()
    return json.loads(re.sub(r"^```(?:json)?|```$", "", c, flags=re.M).strip())


def sample(text, concl, label):
    return {"messages": [
        {"role": "user", "content": f"{INSTR}\n【研报片段】{text}\n【结论】{concl}"},
        {"role": "assistant", "content": label}]}


def main(n=180):
    os.makedirs(OUT_DIR, exist_ok=True)
    blocks = []
    for l in open(CHUNKS, encoding="utf-8"):
        o = json.loads(l)
        if o["type"] in ("text", "summary") and len(o["content"]) > 50 and any(k in o["section"] for k in OPINION_KW):
            blocks.append(o["content"][:400])
    random.seed(42)
    random.shuffle(blocks)
    blocks = blocks[:n]
    print(f"采样 {len(blocks)} 个观点块 → 目标 {len(blocks)*2} 条样本")
    rows = []
    for i, text in enumerate(blocks):
        try:
            g = call(GEN_PROMPT % text)
            if g.get("faithful"):
                rows.append(sample(text, g["faithful"], "支持"))
            if g.get("unfaithful"):
                rows.append(sample(text, g["unfaithful"], "不支持"))
        except Exception as e:
            print(f"  ✗ {i}: {repr(e)[:80]}")
        if (i + 1) % 30 == 0:
            print(f"  进度 {i+1}/{len(blocks)}，已生成 {len(rows)} 条")
        time.sleep(0.15)
    random.shuffle(rows)
    k = int(len(rows) * 0.8)
    for name, part in [("train", rows[:k]), ("val", rows[k:])]:
        fp = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(fp, "w", encoding="utf-8") as w:
            for r in part:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(part)} 条 → {fp}")
    pos = sum(1 for r in rows if r["messages"][1]["content"] == "支持")
    print(f"共 {len(rows)} 条（支持 {pos} / 不支持 {len(rows)-pos}）")


def _used_evidence():
    """读现有 train/val 已用过的【研报片段】文本，增量时排除，避免重复。"""
    used = set()
    for fn in ("train.jsonl", "val.jsonl"):
        fp = os.path.join(OUT_DIR, fn)
        if not os.path.exists(fp):
            continue
        for l in open(fp, encoding="utf-8"):
            c = json.loads(l)["messages"][0]["content"]
            m = re.search(r"【研报片段】(.*?)\n【结论】", c, re.S)
            if m:
                used.add(m.group(1).strip())
    return used


def append_new(n, only_new_stocks=True):
    """增量追加：从【未用过的】观点块(默认优先 watchlist 里 existing=false 的新标的) 造新样本，
    8:2 追加到 train.jsonl/val.jsonl(不覆盖旧数据)。之后跑 make_sft_data_cot.py 增量补 R1 思维链。"""
    used = _used_evidence()
    new_codes = set()
    wl = os.path.join(ROOT, "watchlist.json")
    if only_new_stocks and os.path.exists(wl):
        new_codes = {s["code"] for s in json.load(open(wl, encoding="utf-8"))["stocks"] if not s.get("existing")}
    blocks = []
    for l in open(CHUNKS, encoding="utf-8"):
        o = json.loads(l)
        if o["type"] not in ("text", "summary") or len(o["content"]) <= 50:
            continue
        if not any(k in o["section"] for k in OPINION_KW):
            continue
        if new_codes and o.get("stock_code") not in new_codes:
            continue
        t = o["content"][:400]
        if t not in used:
            blocks.append(t)
    random.seed(123)
    random.shuffle(blocks)
    blocks = blocks[:n]
    print(f"增量采样 {len(blocks)} 个新观点块(排除已用{'，限新标的' if new_codes else ''}) → 目标 +{len(blocks)*2} 条")
    rows = []
    for i, text in enumerate(blocks):
        try:
            g = call(GEN_PROMPT % text)
            if g.get("faithful"):
                rows.append(sample(text, g["faithful"], "支持"))
            if g.get("unfaithful"):
                rows.append(sample(text, g["unfaithful"], "不支持"))
        except Exception as e:
            print(f"  ✗ {i}: {repr(e)[:80]}")
        if (i + 1) % 30 == 0:
            print(f"  进度 {i+1}/{len(blocks)}，已生成 {len(rows)} 条")
        time.sleep(0.15)
    random.shuffle(rows)
    k = int(len(rows) * 0.8)
    for name, part in [("train", rows[:k]), ("val", rows[k:])]:
        fp = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(fp, "a", encoding="utf-8") as w:  # 追加，不覆盖
            for r in part:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: 追加 {len(part)} 条 → {fp}")
    pos = sum(1 for r in rows if r["messages"][1]["content"] == "支持")
    print(f"本次新增 {len(rows)} 条（支持 {pos} / 不支持 {len(rows)-pos}），已追加")


if __name__ == "__main__":
    if "--add" in sys.argv:  # 增量：python make_sft_data.py --add 320  (320块→+640条，追加)
        i = sys.argv.index("--add")
        append_new(int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 320)
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 180)
