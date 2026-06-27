# -*- coding: utf-8 -*-
"""
eval_distill.py — 蒸馏效果【师生对比】(对照独立 ground truth，避免自证)

关键：测试集用 val.jsonl 的【构造标签】当 gold——它是 make_sft_data 造数据时按生成意图定的
(faithful→支持 / unfaithful→不支持)，比 _cot 里"R1 判断出的标签"客观，不是用老师自己的判断当 gold。
四方都在同一批输入上"判断"，对照构造标签算准确率：
  base Qwen(未微调) / 学生 LoRA / 老师 deepseek-chat / 老师 deepseek-reasoner(R1)
→ 说明 学生逼近/追上老师、远超 base。

用法：
  python eval_distill.py --teacher            # 只测老师 deepseek-chat(本地可跑,免GPU)
  python eval_distill.py --teacher --r1       # 加测老师 R1(慢)
  python eval_distill.py --students           # 测 base + LoRA(需 GPU/transformers/adapter，云端)
  python eval_distill.py --all                # 四方全测(云端)
"""
import os
import sys
import json
import re
import time
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VAL = os.path.join(HERE, "data", "val.jsonl")  # 构造标签当 gold


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


def norm(t):
    return "不支持" if "不支持" in t else ("支持" if "支持" in t else None)


def deepseek_judge(user, model):
    body = {"model": model, "messages": [{"role": "user", "content": user}], "stream": False}
    if model != "deepseek-reasoner":
        body["temperature"] = 0
    r = requests.post(URL, headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                      json=body, timeout=300)
    r.raise_for_status()
    return norm(r.json()["choices"][0]["message"]["content"] or "")


def acc_teacher(rows, model):
    print(f"  调用老师 {model} 判 {len(rows)} 条(走 API、无 GPU；R1 较慢)...", flush=True)
    ok = n = 0
    for i, r in enumerate(rows):
        user, gold = r["messages"][0]["content"], norm(r["messages"][1]["content"])
        try:
            pred = deepseek_judge(user, model)
        except Exception as e:
            print(f"  ✗ {repr(e)[:60]}"); continue
        n += 1
        ok += (pred == gold)
        if (i + 1) % 20 == 0:
            print(f"  {model}: {i+1}/{len(rows)}", flush=True)
        time.sleep(0.1)
    return ok, n


def acc_students(rows):
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    USE_4BIT = os.getenv("LOAD_4BIT", "0") == "1"   # 默认 bf16(避开 4bit+LoRA generate 卡顿)；设 1 = 4bit
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    def run(use_lora):
        if USE_4BIT:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            m = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True)
        else:
            m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        if use_lora:
            m = PeftModel.from_pretrained(m, os.path.join(HERE, "out", "adapter"))
            if not USE_4BIT:
                m = m.merge_and_unload()   # bf16 下把 LoRA 合并进 base，绕开 peft+generate 卡顿(4bit 不能 merge)
        m.eval()
        ok = 0
        for i, r in enumerate(rows):
            user, gold = r["messages"][0]["content"], norm(r["messages"][1]["content"])
            p = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
            ids = tok(p, return_tensors="pt").to(m.device)
            out = m.generate(**ids, max_new_tokens=640, do_sample=False)
            pred = norm(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).split("</think>")[-1])
            ok += (pred == gold)
            if (i + 1) % 20 == 0:
                print(f"  {'学生LoRA' if use_lora else 'base'}: {i+1}/{len(rows)}", flush=True)
        del m
        gc.collect()
        torch.cuda.empty_cache()   # 释放显存，base 与 LoRA 两次加载之间
        return ok
    return run(False), run(True), len(rows)


def main():
    rows = [json.loads(l) for l in open(VAL, encoding="utf-8") if l.strip()]
    print(f"测试集 val.jsonl {len(rows)} 条（gold=构造标签，独立于老师判断）\n")
    res = {}
    if "--teacher" in sys.argv or "--all" in sys.argv:
        ok, n = acc_teacher(rows, "deepseek-chat")
        res["老师 deepseek-chat"] = (ok, n)
        if "--r1" in sys.argv or "--all" in sys.argv:
            ok, n = acc_teacher(rows, "deepseek-reasoner")
            res["老师 deepseek-reasoner(R1)"] = (ok, n)
    if "--students" in sys.argv or "--all" in sys.argv:
        b, l, n = acc_students(rows)
        mn = os.path.basename(os.getenv("BASE_MODEL", "Qwen2.5-1.5B-Instruct").rstrip("/\\"))  # 实际 base 名，别写死 7B
        res[f"base {mn}(未微调)"] = (b, n)
        res[f"学生 LoRA({mn})"] = (l, n)
    print(f"{'选手':<26}{'准确率':>10}")
    print("-" * 38)
    for k, (ok, n) in res.items():
        print(f"{k:<26}{ok/n*100:>9.1f}%  ({ok}/{n})")
    print("\n注：gold=构造标签(proxy)，最严谨应人工标黄金集；R1 未参与造 val 故最独立。")


if __name__ == "__main__":
    main()
