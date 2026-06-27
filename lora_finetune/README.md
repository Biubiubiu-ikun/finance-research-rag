# LoRA/QLoRA SFT：把 DeepSeek 的「研报蕴含判别」蒸馏到 Qwen2.5-1.5B

金融研报 RAG 项目子模块。项目 P4② 用 DeepSeek 做"句级 entailment 防幻觉"（判断一条结论是否被研报原文支持）。
这里把该能力**蒸馏**到开源模型 **Qwen2.5-1.5B-Instruct**：DeepSeek 批量造带标签数据 → (Q)LoRA 监督微调
→ 训出的模型可**离线替代 DeepSeek** 做这个判别子任务（省 API、可私有化）。

## 任务
- 输入：`【研报片段】… 【结论】…`，输出：`支持` / `不支持`
- 数据：`data/train.jsonl`(**800**) / `data/val.jsonl`(**200**)；思维链版 `train_cot.jsonl`(**799**)/`val_cot.jsonl`(**199**)；Qwen chat `messages` 格式，由 `make_sft_data.py` 用 DeepSeek 蒸馏(正=忠实结论，负=矛盾/夸大/编造)，**覆盖 8 行业 22 标的**(原 9 标的 360 条 + 增量 `make_sft_data.py --add` 从新标的观点块造 640 条)

## 方法
- **R1 思维链蒸馏(推理过程蒸馏)**：训练数据由 `make_sft_data_cot.py` 用 **DeepSeek-R1(deepseek-reasoner)** 生成——取 R1 的 `reasoning_content`(真实思维链)，目标 output = `<think>思维链</think> + 支持/不支持`。让 **1.5B 小模型**学 **R1 的推理过程**(=官方 R1-Distill-Qwen 思路)，而非只学最终标签 → 难样本更稳、可解释。产 `data/train_cot.jsonl`/`val_cot.jsonl`(脚本会优先用)。
- **(Q)LoRA**：冻结 base、只训低秩矩阵(r=16)，可训参数 <1%。**1.5B bf16 单卡 ~4-6GB**(默认)；`LOAD_4BIT=1` 走 4bit 可压到 ~2GB。
- **SFT**：`trl.SFTTrainer`，`completion_only_loss=True`，`gradient_checkpointing` 省显存；思维链长→`max_length=1024`、`infer.py` 生成 `max_new_tokens=640` 并取 `</think>` 后的结论。

## 云端怎么跑（1.5B 很省，~8GB 卡就够：3060/3090/4090 均可，几元/小时）
```bash
# 1. 环境（Python 3.10+，CUDA 12.x）
pip install -r requirements.txt

# 2. 训练（默认 QLoRA 4bit，自动从 HF 下 Qwen2.5-1.5B-Instruct；国内见下方“拉模型慢”）
python train_lora.py
#   显存：默认 bf16 ~4-6GB(1.5B)；LOAD_4BIT=1 走 4bit ~2GB
#   产物：out/adapter/

# 3. 验证微调效果
python infer.py            # base + LoRA 在 val_cot 上准确率
python infer.py --base     # 仅 base（微调前对照）

# 4. ★师生对比(对照独立构造标签 val.jsonl，避免"用老师判断当gold"的自证)——证明蒸馏价值
python eval_distill.py --all    # 四方 vs 构造gold：base / 学生LoRA / 老师deepseek-chat / 老师R1
#   本地已测 老师 deepseek-chat = 94.4%(68/72,非100%→证明gold独立)；云端补 base/LoRA 看学生逼近老师、超base 多少
```

## 在 AutoDL 上跑（一步步照做，总共几块钱）

> **本地能不能跑**：换 1.5B 后门槛低多了——**bf16 推理才 ~3GB，6GB 显存的本地卡也能跑**(`LOAD_4BIT=0`，脚本会 merge LoRA 进 base、避开 4bit 卡顿)；训练 bf16 ~4-6GB，显存够也可本地训。不过云端(AutoDL)更快省事，`out/adapter/`(几十 MB)拉回本地存档即可。

**① 租实例**
- [autodl.com](https://www.autodl.com) 注册充值 → 算力市场选 **RTX 3060/3090/4090 均可**：1.5B bf16 只需 ~4-6GB、便宜卡就够，约 **几元/小时**；镜像选 **PyTorch 2.x + CUDA 12.1** → 开机进 JupyterLab。

**② 传代码 + 云端下基座**（Qwen2.5-1.5B 才 ~3GB，云端下几十秒）
```bash
# JupyterLab 把整个 lora_finetune/（代码 + data/，才几 MB）拖上去
pip install modelscope
modelscope download --model Qwen/Qwen2.5-1.5B-Instruct --local_dir /root/autodl-tmp/Qwen2.5-1.5B-Instruct
```

**③ 装依赖 + 训练 + 验证**
```bash
cd lora_finetune
pip install -r requirements.txt
export BASE_MODEL=/root/autodl-tmp/Qwen2.5-1.5B-Instruct
export DEEPSEEK_API_KEY=你的key            # eval_distill 测老师要用
python train_lora.py          # QLoRA 4bit，3 epochs，1000 条约 10-30 分钟 → out/adapter/
python infer.py --base        # 微调前 base 对照
python infer.py               # base+LoRA，看提升多少
python eval_distill.py --all  # 师生四方对比：base / 学生LoRA / 老师chat / 老师R1
```

**④ 收尾**
- `out/adapter/`（几十 MB）下载到本地存档；
- **务必"关机 / 停止实例"**——按小时计费，跑完不关会一直扣；数据盘 `/root/autodl-tmp` 关机不丢，下次开机还在。

**两个坑**：① 要连 HuggingFace 先 `source /etc/network_turbo`（AutoDL 学术加速），但用 ModelScope 下 Qwen 就不需要；② 传数据 / 装环境时开"无卡模式"省 GPU 钱。

## 关键超参（train_lora.py 可调）
| 参数 | 值 | 说明 |
|---|---|---|
| LOAD_4BIT | 0(默认) | 0=bf16(1.5B ~4-6GB，推荐)；1=4bit(~2GB) |
| r / lora_alpha | 16 / 32 | LoRA 秩与缩放 |
| epochs | 3 | 数据少 2~3 够，过多易过拟合 |
| batch × grad_accum | 4 × 4 | 等效 16；OOM 就调小 batch、调大 accum |
| lr | 2e-4 | LoRA 常用 1e-4~3e-4 |
| max_length | **1024** | 思维链版需更长(实测 998 条:中位 355/p95 566 token,仅 0.6% 超 1024;512 会截 ~10% 思维链尾部) |

## 基座模型（已下到本地，无需重拉）
**默认 base = `Qwen2.5-1.5B-Instruct`**(小模型蒸馏提升更明显，见文末「选 base 经验」)。训练/推理/评测前设好 `BASE_MODEL`：
```bash
# Linux/云端：  export BASE_MODEL=/path/to/Qwen2.5-1.5B-Instruct
# Windows 本地：set BASE_MODEL=D:\代码随想录大模型\金融研报rag\Qwen2.5-1.5B-Instruct
# train_lora.py / infer.py / eval_distill.py 都读这个环境变量；不设则默认从 HF 拉 Qwen/Qwen2.5-1.5B-Instruct
```
云端下载：`modelscope download --model Qwen/Qwen2.5-1.5B-Instruct --local_dir ./Qwen2.5-1.5B-Instruct`(国内快)，或 `huggingface-cli download`(HF 直连)。(本机项目根另存有 7B 权重 15GB；想对照试 7B base，把 `BASE_MODEL` 指向它即可。)

## 版本提示
脚本按 `trl>=0.12`（`SFTConfig`/`processing_class`/`max_length`/`completion_only_loss`）。
若报参数不识别：`pip install -U trl`，或把 `max_length`→`max_seq_length`、`processing_class`→`tokenizer`、去掉 `completion_only_loss`。

## 推理 / 评测卡住？（4bit + LoRA 的坑，实测踩过）
`infer.py` / `eval_distill.py` 在 **4bit + LoRA** 下做 generate 会**极慢甚至卡死**（量化层与 peft adapter 反复交互、GPU 空转，util 只有 20%+；纯 base 无 adapter 不触发）。**大卡（A100/3090/4090）直接 `export LOAD_4BIT=0` 走 bf16**——脚本会自动在 bf16 下 `merge_and_unload()` 把 LoRA 合并进 base、绕开此坑，1.5B bf16 才占 ~3GB。两脚本都已加进度打印，长时间无输出 ≠ 卡死。

> **选 base 的经验（重要）**：学生 base 别用太大。**7B-Instruct 做这个二分类天花板就 91%、蒸馏没空间**；换 **1.5B 做学生**，base 仅 ~70%，经 R1 思维链蒸馏后能冲到 **~92%**（实测，超 7B base、逼近老师 94.4%）——这才是"小模型推理蒸馏"该有的提升曲线，也是 R1-Distill 的正确姿势。

## 怎么讲（面试）
**"用 DeepSeek-R1 的思维链做【推理过程蒸馏】(R1-Distill 思路) + QLoRA 微调 Qwen2.5-1.5B，把 RAG 的事实核查子任务从调闭源 API 换成可私有化、且可解释的本地模型"**
—— 覆盖：造数据、**推理过程蒸馏(非仅答案蒸馏)**、(Q)LoRA/SFT/量化、能独立训练+评测 LLM(1.5B，方法可直接扩到 7B)。
亮点：微调后的小模型不只给"不支持"，还能给出**为什么不支持的推理**(可解释，难样本更稳)。
预期：`infer.py` 对照 base vs LoRA 的 val 准确率，量化提升。
数据可复现(需 DeepSeek key)：`python make_sft_data.py`(造正负结论) → `python make_sft_data_cot.py`(R1 生成思维链，慢、可断点续跑)。
