# -*- coding: utf-8 -*-
"""
finetune_bge.py — P3微调② 用金融 (query↔chunk) 对微调 bge-small

对比学习 MultipleNegativesRankingLoss（batch 内互为负样本）。
query 加 bge 检索指令前缀（与推理一致），passage 不加。
微调后存到 models/bge-small-zh-finance，供 index_es/retrieve_es 通过 BGE_MODEL 环境变量切换。
"""
import os
import sys
import json
import torch
os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EIGHT = os.path.join(os.path.dirname(BASE), "ai面试八股rag", "models")
SRC_MODEL = os.path.join(BASE, "models", "bge-small-zh-v1.5")
if not os.path.exists(SRC_MODEL):
    SRC_MODEL = os.path.join(EIGHT, "bge-small-zh-v1.5")
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
EPOCHS = 3
BATCH = 32
HARD = "--hard" in sys.argv  # 难负样本模式：用挖到的 hard negative 做对比学习消融
PAIRS = os.path.join(BASE, "data", "train", "train_pairs_hn.jsonl" if HARD else "train_pairs.jsonl")
OUT_MODEL = os.path.join(BASE, "models", "bge-small-zh-finance-hn" if HARD else "bge-small-zh-finance")


def main():
    pairs = [json.loads(l) for l in open(PAIRS, encoding="utf-8") if l.strip()]
    if HARD:
        examples = [InputExample(texts=[QUERY_PREFIX + p["query"], p["positive"], p["negative"]])
                    for p in pairs if p.get("negative")]
        print(f"【难负样本模式】{len(examples)} 条 (query, 正例, 难负)；MNRL 用 in-batch + 该 hard neg")
    else:
        examples = [InputExample(texts=[QUERY_PREFIX + p["query"], p["positive"]]) for p in pairs]
        print(f"【in-batch 模式】{len(examples)} 条 (query, 正例)")

    model = SentenceTransformer(SRC_MODEL)
    print("设备:", model.device)
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = int(0.1 * len(loader) * EPOCHS)
    print(f"开始微调：epochs={EPOCHS} batch={BATCH} steps/epoch={len(loader)}")
    model.fit(train_objectives=[(loader, loss)], epochs=EPOCHS,
              warmup_steps=warmup, show_progress_bar=True)
    os.makedirs(OUT_MODEL, exist_ok=True)
    model.save(OUT_MODEL)
    print(f"\n✓ 微调模型已保存 → {OUT_MODEL}")


if __name__ == "__main__":
    main()
