# -*- coding: utf-8 -*-
"""
make_sft_data_cot.py — R1 思维链蒸馏数据（deepseek-reasoner / DeepSeek-R1）

把 make_sft_data.py 造好的 (研报片段+结论+标签) 升级成【推理过程蒸馏】：
对每条调 DeepSeek-R1，取它的 reasoning_content(真实思维链) + 结论，
训练目标 output = "<think>\n{思维链}\n</think>\n\n{支持/不支持}"。
→ 小模型学的不只是答案，而是 R1 的【推理过程】(= 官方 R1-Distill-Qwen 的思路)。

R1 慢且贵 → 默认采样 120 条；断点续跑(中断/被回收可重跑，已生成的跳过)。
产物 data/train_cot.jsonl / val_cot.jsonl（train_lora/infer 会优先用 _cot 版）。
用法：python make_sft_data_cot.py [N]
"""
import os
import sys
import json
import time
import random
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")


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
KEY = os.getenv("DEEPSEEK_API_KEY")
URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions"


def call_r1(user):
    """调 DeepSeek-R1，返回 (思维链 reasoning_content, 最终答案 content)。R1 不支持 JSON mode/温度等。"""
    body = {"model": "deepseek-reasoner", "messages": [{"role": "user", "content": user}], "stream": False}
    r = requests.post(URL, headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                      json=body, timeout=300)
    r.raise_for_status()
    m = r.json()["choices"][0]["message"]
    return (m.get("reasoning_content") or "").strip(), (m.get("content") or "").strip()


def norm(ans):
    return "不支持" if "不支持" in ans else ("支持" if "支持" in ans else None)


def process(in_name, out_name, n):
    src = [json.loads(l) for l in open(os.path.join(DATA, in_name), encoding="utf-8") if l.strip()]
    random.seed(7)
    random.shuffle(src)
    src = src[:n]
    out_fp = os.path.join(DATA, out_name)
    done = set()
    if os.path.exists(out_fp):  # 断点续跑
        done = set(json.loads(l)["messages"][0]["content"] for l in open(out_fp, encoding="utf-8") if l.strip())
    new = 0
    with open(out_fp, "a", encoding="utf-8") as w:
        for i, s in enumerate(src, 1):
            user = s["messages"][0]["content"]
            if user in done:
                continue
            try:
                reasoning, ans = call_r1(user)
            except Exception as e:
                print(f"  ✗ {i}: {repr(e)[:80]}"); continue
            lab = norm(ans)
            if not lab or not reasoning:
                print(f"  ⚠ {i}: 无思维链或无明确结论，跳过"); continue
            out = f"<think>\n{reasoning}\n</think>\n\n{lab}"
            w.write(json.dumps({"messages": [{"role": "user", "content": user},
                                             {"role": "assistant", "content": out}]}, ensure_ascii=False) + "\n")
            w.flush()
            new += 1
            if new % 10 == 0:
                print(f"  {out_name}: +{new}（思维链均长 ~{len(reasoning)}字）")
            time.sleep(0.1)
    print(f"{out_name}: 本次新增 {new}")


def main(n=120):
    n_train = int(n * 0.8)
    process("train.jsonl", "train_cot.jsonl", n_train)
    process("val.jsonl", "val_cot.jsonl", n - n_train)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
