# -*- coding: utf-8 -*-
"""
train_lora.py — LoRA/QLoRA SFT 微调 Qwen2.5-1.5B-Instruct 做 entailment(蕴含判别)

把 DeepSeek 蒸馏的 (研报片段, 结论)→支持/不支持 数据，用 LoRA 监督微调到开源模型。
LoRA：冻结 base 权重，只训各层注入的低秩矩阵(r=16)，可训参数 <1%；产物 adapter 仅几十 MB。
默认 bf16(1.5B 仅 ~4-6GB，大多数卡都够，且避开 4bit+LoRA 推理卡顿)；LOAD_4BIT=1 可改回 QLoRA 4bit(~2GB)。

云端跑：见 README.md。
用法：python train_lora.py
"""
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")   # 国内拉慢：设 BASE_MODEL=本地 ModelScope 路径
USE_4BIT = os.getenv("LOAD_4BIT", "0") == "1"                  # 默认 bf16(1.5B ~4-6GB，省心不踩 4bit+LoRA 坑)；设 1 = QLoRA 4bit(~2GB)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")


def main():
    def pick(name):  # R1 思维链蒸馏版(_cot) 优先，回退普通答案版
        cot = os.path.join(DATA, f"{name}_cot.jsonl")
        return cot if os.path.exists(cot) else os.path.join(DATA, f"{name}.jsonl")
    ds = load_dataset("json", data_files={"train": pick("train"), "validation": pick("val")})
    print(f"train {len(ds['train'])} / val {len(ds['validation'])} | {os.path.basename(pick('train'))} | model={MODEL} | 4bit={USE_4BIT}")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if USE_4BIT:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                     device_map="auto", trust_remote_code=True)
        model = prepare_model_for_kbit_training(model)
        print("QLoRA：4bit 量化加载")
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                     device_map="auto", trust_remote_code=True)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(
        output_dir=OUT, num_train_epochs=3,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,  # 等效batch16；显存紧再调小batch/调大accum
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
        bf16=True, max_length=1024, report_to="none",  # 思维链长→序列加长
        gradient_checkpointing=True,        # 省显存(可选，1.5B 关掉也行)
        completion_only_loss=True,          # 只在"支持/不支持"答案上算 loss
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"],
                         eval_dataset=ds["validation"], peft_config=lora, processing_class=tok)
    trainer.train()
    trainer.save_model(os.path.join(OUT, "adapter"))
    tok.save_pretrained(os.path.join(OUT, "adapter"))
    print(f"\n✓ LoRA adapter 已保存 → {os.path.join(OUT, 'adapter')}（推理时叠加 base）")


if __name__ == "__main__":
    main()
