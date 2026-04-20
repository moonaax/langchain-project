# DeepSeek Chat — Electron 桌面客户端

基于 LangChain + DeepSeek 的桌面聊天客户端，支持多轮对话和流式输出。

## 架构

```
Electron (前端界面)  ←→  FastAPI (后端 API)  ←→  LangChain + DeepSeek
     index.html              server.py              ChatOpenAI
```

## 安装

```bash
# 1. Python 依赖
pip3 install -r requirements.txt

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
