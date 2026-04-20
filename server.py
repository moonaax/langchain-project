"""
FastAPI 后端：LangChain + DeepSeek 流式聊天 API

AI Agent 知识点索引：
- 生产化部署: FastAPI 集成、SSE 流式推送、CORS 跨域
- LCEL: 异步流式调用 (astream)、Runnable 协议的异步接口
- Memory: 多会话隔离、session_id 机制
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# ============================================================
# 【知识点】生产化部署 — FastAPI 集成
# ============================================================
# 将 LangChain Chain 部署为 REST API 的标准方式
# LangChain 官方推荐 LangServe，但手写 FastAPI 更灵活
# 核心思路：HTTP 请求 → 调用 Chain → 返回结果
app = FastAPI()

# ============================================================
# 【知识点】生产化部署 — CORS 跨域
# ============================================================
# Electron 加载本地 file:// HTML 请求 http://localhost:8000
# 浏览器会拦截跨域请求，必须在后端允许
# allow_origins=["*"] 开发阶段用，生产环境应限制为具体域名
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 【知识点】Model I/O — 同 chat.py，模型初始化
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7,
)

# 【知识点】Model I/O — 同 chat.py，Prompt Template + MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的 AI 助手，回答简洁准确。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 【知识点】LCEL — 管道符编排
chain = prompt | llm

# 【知识点】Memory — 会话存储 + 多会话隔离
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 【知识点】Memory — RunnableWithMessageHistory
chain_with_memory = RunnableWithMessageHistory(
    chain, get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Pydantic 模型：自动校验请求体 JSON 格式
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

# ============================================================
# 【知识点】生产化部署 — SSE 流式推送
# ============================================================
# Server-Sent Events (SSE) 是 HTTP 流式推送的标准协议
# 格式：每条消息以 "data: 内容\n\n" 发送，最后发 "data: [DONE]\n\n"
# 前端用 fetch + ReadableStream 逐块读取
# 对比 WebSocket：SSE 是单向的（服务端→客户端），更简单，适合 AI 流式输出
#
# 【知识点】LCEL — 异步流式调用 (astream)
# astream() 是 stream() 的异步版本，返回 AsyncGenerator
# 在 FastAPI 的 async 函数中必须用 astream 而不是 stream
# Runnable 协议的四种调用方式：
#   invoke()  / ainvoke()  — 同步/异步，等待完整结果
#   stream()  / astream()  — 同步/异步，逐块流式返回
#   batch()   / abatch()   — 同步/异步，批量处理
@app.post("/chat")
async def chat(req: ChatRequest):
    config = {"configurable": {"session_id": req.session_id}}

    async def generate():
        async for chunk in chain_with_memory.astream({"input": req.message}, config=config):
            if chunk.content:
                yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# 清空会话历史
@app.post("/clear")
async def clear(req: ChatRequest):
    store.pop(req.session_id, None)
    return {"status": "ok"}
