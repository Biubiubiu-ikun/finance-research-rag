# -*- coding: utf-8 -*-
"""
local_verifier.py — 本地【蒸馏小模型】事实核查器（把 LoRA 微调的 Qwen2.5-1.5B 接入项目）

加载 本地 Qwen2.5-1.5B-Instruct(base) + out/adapter(R1 思维链蒸馏的 LoRA) → bf16 merge，
judge(研报片段, 结论) → (支持/不支持, <think> 推理链)。
本质是 P4② 句级 entailment 防幻觉核查的【可私有化】版本：把原本调 DeepSeek 的核查，
换成离线的本地蒸馏小模型(省 API、可私有化、还带推理过程)。

惰性加载(首次调用才载模型，避免常驻显存)；bf16 + merge_and_unload 避开 4bit+LoRA 卡顿。
用法(命令行自测)：python local_verifier.py "研报片段" "待核查结论"
"""
import os
import threading
import torch

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.getenv("VERIFIER_BASE", os.path.join(HERE, "Qwen2.5-1.5B-Instruct"))  # 本地 1.5B base
ADAPTER = os.getenv("VERIFIER_ADAPTER", os.path.join(HERE, "out", "adapter"))   # LoRA adapter
INSTR = "你是金融研报事实核查员。判断【结论】是否被【研报片段】支持，只回答两个字：支持 或 不支持。"

_model = None
_tok = None
_lock = threading.Lock()


def available():
    """接入 UI 前先判断：base 权重和 adapter 是否就位。"""
    return os.path.exists(os.path.join(BASE, "config.json")) and \
        os.path.exists(os.path.join(ADAPTER, "adapter_config.json"))


def _load():
    global _model, _tok
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        if hasattr(os, "add_dll_directory"):  # 仅 Windows 需手动加载 torch DLL
            os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))
        tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=True)
        m = PeftModel.from_pretrained(m, ADAPTER)   # adapter 叠到【本地】base 上（不读 adapter_config 里的云端路径）
        m = m.merge_and_unload()                    # 合并 LoRA → 普通模型，generate 更快、绕开 peft 卡顿
        _model, _tok = m.eval(), tok


def judge(evidence, conclusion, max_new_tokens=512):
    """返回 (label, reasoning)：label ∈ {支持, 不支持, 不确定}；reasoning = <think> 里的推理链。"""
    _load()
    user = f"{INSTR}\n【研报片段】{evidence}\n【结论】{conclusion}"
    prompt = _tok.apply_chat_template([{"role": "user", "content": user}],
                                      tokenize=False, add_generation_prompt=True)
    ids = _tok(prompt, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        out = _model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    text = _tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    reasoning, final = "", text.strip()
    if "</think>" in text:  # 拆 <think>推理</think> 与最终结论
        head, final = text.split("</think>", 1)
        reasoning = head.replace("<think>", "").strip()
        final = final.strip()
    label = "不支持" if "不支持" in final else ("支持" if "支持" in final else "不确定")
    return label, reasoning


def _deepseek_cfg():
    """从 .env 读 DeepSeek key/url（与项目其它脚本一致）。"""
    for p in [os.path.join(HERE, ".env"), os.path.join(os.path.dirname(HERE), "ai面试八股rag", ".env")]:
        if os.path.exists(p):
            for line in open(p, encoding="utf-8-sig"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            break
    key = os.getenv("DEEPSEEK_API_KEY")
    url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions"
    return key, url


def teacher_judge(evidence, conclusion):
    """对照：调 DeepSeek 老师做同样判别（证明本地蒸馏学生逼近老师）。返回 '支持'/'不支持'/'不确定'。"""
    import requests
    key, url = _deepseek_cfg()
    user = f"{INSTR}\n【研报片段】{evidence}\n【结论】{conclusion}"
    body = {"model": "deepseek-chat", "messages": [{"role": "user", "content": user}],
            "temperature": 0, "stream": False}
    r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                      json=body, timeout=60)
    r.raise_for_status()
    t = r.json()["choices"][0]["message"]["content"] or ""
    return "不支持" if "不支持" in t else ("支持" if "支持" in t else "不确定")


if __name__ == "__main__":
    import sys
    ev = sys.argv[1] if len(sys.argv) > 1 else "公司2025年营收同比增长20%，创新药收入占比提升至60%，海外BD合作频繁。"
    cc = sys.argv[2] if len(sys.argv) > 2 else "公司2025年营收同比下降，创新药占比走低。"
    print(f"【片段】{ev}\n【结论】{cc}\n")
    lab, rsn = judge(ev, cc)
    print(f"判定：{lab}\n推理链：{rsn}")
