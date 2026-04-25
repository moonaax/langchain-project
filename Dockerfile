# ============================================================
# 【知识点】第六阶段 — Docker 容器化
# ============================================================
# 只容器化 FastAPI 后端，Electron 前端不容器化（桌面应用）
# FAISS 索引 + BM25 语料通过 volume 挂载，不打进镜像
# HuggingFace 模型缓存通过 volume 挂载，避免重复下载

FROM python:3.12-slim

WORKDIR /app

# 安装依赖（利用 Docker 缓存层，依赖不变时不重新安装）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY server.py tools.py graph_agent.py build_index.py eval_rag.py ./
COPY eval_set.json ./

# HuggingFace 离线模式（镜像内不联网下载模型）
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_CACHE=/app/hf_cache

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
