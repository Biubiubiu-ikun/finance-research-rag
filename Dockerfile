# 金融研报 RAG · API 服务镜像（问答/聚合/检索）
# 注：不含 DeepDOC 布局解析重依赖；"新研报入库"链路如需在容器内跑，另装 requirements.txt 文末依赖。
FROM python:3.12-slim

WORKDIR /app

# CPU 版 torch（GPU 部署请换 nvidia/cuda 基础镜像 + 对应 +cuXXX 轮子）
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# torch 已单独装，安装其余依赖
RUN grep -v '^torch==' requirements.txt > /tmp/req.txt && pip install --no-cache-dir -r /tmp/req.txt

COPY . .

ENV PYTHONUTF8=1 HF_HUB_OFFLINE=1
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
