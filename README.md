# DeepSeek Chat — Electron 桌面客户端

基于 LangChain + DeepSeek 的桌面聊天客户端，支持多轮对话和流式输出。

## 架构

```
Electron (前端界面)  ←→  FastAPI (后端 API)  ←→  LangChain + DeepSeek
     index.html              server.py              ChatOpenAI
```

## 环境要求

- Python 3.12
- Node.js（Electron 前端）
- PyTorch 只支持到 2.2.x（Python 3.12 官方安装包限制）

关键依赖版本约束：

| 包 | 版本 |
|---|---|
| langchain-classic | 1.0.x |
| torch | 2.2.x |
| numpy | <2 |
| transformers | <4.51 |
| sentence-transformers | <4 |

## 安装

```bash
# 1. Python 依赖
pip3 install -r requirements.txt
pip3 install "numpy<2" "transformers<4.51" "sentence-transformers<4" langchain-classic

# 2. Electron 依赖
cd electron && npm install && cd ..

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env 填入 DeepSeek API Key
```

## 运行

```bash
# 一键启动（后端 + 前端）
./start.sh
```

或者分开启动：

```bash
# 终端 1：启动后端
uvicorn server:app --port 8000

# 终端 2：启动 Electron
cd electron && npx electron .
```

## 功能

- 🤖 接入 DeepSeek 大模型
- 💬 多轮对话记忆
- ⚡ 流式输出（逐字显示）
- 🗑️ 清空对话历史
- 🌙 暗色主题
