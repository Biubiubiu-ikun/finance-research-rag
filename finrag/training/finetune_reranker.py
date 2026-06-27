# -*- coding: utf-8 -*-
"""
finetune_reranker.py — bge微调④ 用金融难负样本微调 cross-encoder reranker

向量召回(bi-encoder)解决"粗排"，reranker(cross-encoder)解决"精排"——query 与候选 passage
拼一起过 BERT 算相关性分，比向量点积更准但更慢，只对粗排候选池跑。
本脚本用挖好的难负样本(query, 正例, 难负)把【通用 bge-reranker-base】微调成【金融领域 reranker】：
  - 正样本 (query, positive) → label 1
  - 难负   (query, negative) → label 0   （难负=同主题但数字口径错乱/张冠李戴的块，逼模型学细粒度区分）
BCEWithLogitsLoss(num_labels=1)。微调后存 models/bge-reranker-finance，由 eval_reranker.py 对比通用 reranker。

用法：python finetune_reranker.py
"""
import os
import sys
import json
import torch
if hasattr(os, "add_dll_directory"):  # 仅 Windows 需手动加载 torch DLL；Linux 无此 API
    os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EIGHT = os.path.join(os.path.dirname(BASE), "ai面试八股rag", "models")
SRC = os.path.join(BASE, "models", "bge-reranker-base")
if not os.path.exists(SRC):
    SRC = os.path.join(EIGHT, "bge-reranker-base")  # 复用八股项目的通用 reranker
PAIRS = os.path.join(BASE, "data", "train", "train_pairs_hn.jsonl")
OUT = os.path.join(BASE, "models", "bge-reranker-finance")
EPOCHS = 2
BATCH = 16


def main():
    pairs = [json.loads(l) for l in open(PAIRS, encoding="utf-8") if l.strip()]
    examples = []
    n_pos = n_neg = 0
    for p in pairs:
        examples.append(InputExample(texts=[p["query"], p["positive"]], label=1.0))
        n_pos += 1
        if p.get("negative"):  # 难负样本作负例(reranker 学到的细粒度区分主要来自这里)
            examples.append(InputExample(texts=[p["query"], p["negative"]], label=0.0))
            n_neg += 1
    print(f"训练样本 {len(examples)} 条：正例 {n_pos} + 难负 {n_neg}")

    model = CrossEncoder(SRC, num_labels=1, max_length=512)
    print("基座:", SRC, "| 设备:", model._target_device if hasattr(model, "_target_device") else "?")
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH)
    warmup = int(0.1 * len(loader) * EPOCHS)
    print(f"开始微调：epochs={EPOCHS} batch={BATCH} steps/epoch={len(loader)} warmup={warmup}")
    model.fit(train_dataloader=loader, epochs=EPOCHS, warmup_steps=warmup, show_progress_bar=True)
    os.makedirs(OUT, exist_ok=True)
    model.save(OUT)
    print(f"\n✓ 微调 reranker 已保存 → {OUT}")


if __name__ == "__main__":
    main()
