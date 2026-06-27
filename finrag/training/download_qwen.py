# -*- coding: utf-8 -*-
"""
download_qwen.py — 从 HuggingFace 下载 Qwen2.5-7B-Instruct(QLoRA 基座) 到项目目录

本机代理(127.0.0.1:7890)对 huggingface.co 可达 → 直连 HF 官方下载(海外源走代理反而通)。
约 15GB(4 个 safetensors 分片)，断点续传。下到 项目根/Qwen2.5-7B-Instruct。

用法：python download_qwen.py
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 不设 HF_HUB_OFFLINE（要联网）；HF 走系统代理
from huggingface_hub import snapshot_download

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEST = os.path.join(BASE, "Qwen2.5-7B-Instruct")


def main():
    print(f"⬇  Qwen/Qwen2.5-7B-Instruct → {DEST}", flush=True)
    path = snapshot_download(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        local_dir=DEST,
        ignore_patterns=["*.gguf", "*.onnx", "*.h5", "tf_model.*", "flax_model.*"],  # 只要 PyTorch 权重
    )
    print(f"✓ 下载完成：{path}", flush=True)


if __name__ == "__main__":
    main()
