# -*- coding: utf-8 -*-
"""
infer.py — 加载 base(+LoRA adapter) 推理，在 val 集上算准确率，验证微调效果

对比 base 与 base+LoRA 在 entailment 判别上的准确率，看 LoRA 提升多少。
默认 4bit 加载(与训练一致，省显存)；LOAD_4BIT=0 走 bf16。
用法：python infer.py            # base+LoRA
      python infer.py --base     # 仅 base(对照微调前)
"""
import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
USE_4BIT = os.getenv("LOAD_4BIT", "0") == "1"   # 默认 bf16(1.5B 推理才 ~3GB，且避开 4bit+LoRA 卡顿)
HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(HERE, "out", "adapter")
_cot = os.path.join(HERE, "data", "val_cot.jsonl")  # R1 思维链版优先
VAL = _cot if os.path.exists(_cot) else os.path.join(HERE, "data", "val.jsonl")


def load(use_lora):
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if USE_4BIT:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                     device_map="auto", trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                     device_map="auto", trust_remote_code=True)
    if use_lora:
        model = PeftModel.from_pretrained(model, ADAPTER)
        if not USE_4BIT:
            model = model.merge_and_unload()   # bf16 下把 LoRA 合并进 base，绕开 4bit+peft generate 卡顿(4bit 不能 merge)
    return tok, model.eval()


def main():
    use_lora = "--base" not in sys.argv
    tok, model = load(use_lora)
    rows = [json.loads(l) for l in open(VAL, encoding="utf-8") if l.strip()]
    ok = 0
    for i, r in enumerate(rows):
        prompt = tok.apply_chat_template(r["messages"][:1], tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=640, do_sample=False)  # 思维链版要够长
        pred = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        gold = r["messages"][1]["content"]
        pf = pred.split("</think>")[-1]   # 取思维链之后的最终结论(避免思考里的"不支持"干扰)
        gf = gold.split("</think>")[-1]
        ok += ("不支持" in pf) == ("不支持" in gf)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)   # 进度(避免长时间无输出像卡死)
    print(f"{'base+LoRA' if use_lora else 'base(微调前)'} 在 {len(rows)} 条 val 上准确率：{ok/len(rows)*100:.1f}%")


if __name__ == "__main__":
    main()
