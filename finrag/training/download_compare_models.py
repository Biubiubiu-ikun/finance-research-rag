# -*- coding: utf-8 -*-
"""
download_compare_models.py — 下载横向对比用的嵌入模型（bge-base / gte-base-zh / bge-m3）

eval_vector_compare.py 是纯内存 cosine 对比，不入 ES，所以各模型维度不同也无妨，只要拿到模型文件。
本机代理(127.0.0.1:7890)对 huggingface.co 可达 → 直接走 HF 官方下载。
失败的模型跳过(不影响其它/已下好的)；bge-m3 较大(忽略 onnx/colbert 等附属文件)。

用法：python download_compare_models.py
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 不要设 HF_HUB_OFFLINE（要联网下载）
from huggingface_hub import snapshot_download

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = {
    "bge-base-zh-v1.5": ("BAAI/bge-base-zh-v1.5", None),                 # bge 同源更大规模(对照"微调收益 vs 单纯加大模型")
    "gte-base-zh":      ("thenlper/gte-base-zh", None),                  # 另一主流中文嵌入
    "bge-m3":           ("BAAI/bge-m3", ["onnx/*", "*.onnx", "colbert*", # 多语言 SOTA(只下稠密向量需要的文件)
                                          "sparse*", "imgs/*", "*.h5", "tf_model.*", "flax_model.*", "*.ot"]),
}


def main():
    for local, (repo, ignore) in MODELS.items():
        dest = os.path.join(BASE, "models", local)
        if os.path.exists(os.path.join(dest, "config.json")):
            print(f"⏭  已存在，跳过：{local}")
            continue
        print(f"⬇  下载 {repo} → models/{local} …", flush=True)
        try:
            snapshot_download(repo_id=repo, local_dir=dest, ignore_patterns=ignore)
            print(f"✓ OK：{local}", flush=True)
        except Exception as e:
            print(f"✗ FAIL：{local} | {repr(e)[:240]}", flush=True)


if __name__ == "__main__":
    main()
