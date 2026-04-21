"""
FastAPI 后端：LangChain + DeepSeek 工具调用 Agent API

第二阶段：在流式聊天基础上添加 Agent 能力

AI Agent 知识点索引：
- Agent API: 将 AgentExecutor 部署为 REST API
- astream_events: 细粒度异步事件流，捕获工具调用过程
- SSE 协议扩展: 自定义事件类型（tool_start / tool_end / token / done）
"""

import os
import json
import math
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── 模型初始化（同 chat.py）──
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7,
)

# ============================================================
# 【知识点】Tool — 工具定义（与 chat.py 完全一致）
# ============================================================
# 生产项目中应将工具定义抽取到独立模块 tools.py
# 这里为了学习清晰度，保持与 chat.py 同步

@tool
def calculator(expression: str) -> str:
    """安全计算数学表达式并返回结果。

    当用户需要进行数学计算时使用，包括加减乘除、幂运算、开方等。
    输入应为合法的 Python 数学表达式，如 "2 + 3 * 4" 或 "math.sqrt(16)"。
    """
    allowed = {"math": math, "abs": abs, "round": round, "pow": pow}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}，请检查表达式格式"

@tool
def get_current_time() -> str:
    """获取当前日期和时间。当用户询问现在几点、今天日期时使用。"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

@tool
def search_weather(city: str) -> str:
    """查询指定城市的天气信息。输入城市名称，返回当前天气。"""
    weather_data = {
        "北京": "晴天，25°C，湿度 40%",
        "上海": "多云，22°C，湿度 65%",
        "广州": "小雨，28°C，湿度 80%",
        "深圳": "阴天，27°C，湿度 75%",
        "杭州": "晴天，23°C，湿度 55%",
    }
    return weather_data.get(city, f"未找到 {city} 的天气信息，支持的城市：{', '.join(weather_data.keys())}")

tools = [calculator, get_current_time, search_weather]

# ── Agent 构建（同 chat.py）──
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个有帮助的 AI 助手，可以使用工具来回答问题。

你可以使用以下工具：
- calculator: 数学计算
- get_current_time: 获取当前时间
- search_weather: 查询天气

如果用户的问题可以直接回答，就直接回答，不需要使用工具。
如果需要使用工具，请使用后基于结果给出完整的回答。"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools,
    max_iterations=10,
    handle_parsing_errors=True,
)

# ── Memory（同 chat.py）──
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

agent_with_memory = RunnableWithMessageHistory(
    agent_executor, get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

# ============================================================
# 【知识点】Agent API — astream_events 细粒度事件流
# ============================================================
# 第一阶段用 astream() 逐 token 推送，但 Agent 有工具调用过程，
# 需要更细粒度的事件流来区分：LLM 生成 vs 工具调用 vs 最终回答
#
# astream_events(version="v2") 返回结构化事件，每个事件包含：
#   - event: 事件类型（on_chat_model_stream / on_tool_start / on_tool_end 等）
#   - name: 组件名称（如工具名 "calculator"）
#   - data: 事件数据（如 chunk.content / tool output）
#
# SSE 协议扩展：
#   标准 SSE 只有 data 字段，但可以加 event 字段区分消息类型
#   格式：event: tool_start\ndata: {...}\n\n
#   前端用 EventSource 或手动解析来分别处理不同事件类型
@app.post("/chat")
async def chat(req: ChatRequest):
    config = {"configurable": {"session_id": req.session_id}}

    async def generate():
        # ============================================================
        # 【知识点】astream_events — 事件类型说明
        # ============================================================
        # on_tool_start:      工具开始执行（含工具名和输入参数）
        # on_tool_end:        工具执行完成（含返回结果）
        # on_chat_model_stream: LLM 生成的每个 token
        #
        # 过滤策略：
        #   Agent 内部有多次 LLM 调用（决策 + 最终回答），
        #   我们只关心最终回答的 token 流，通过检查 event["name"]
        #   或 event 的层级来过滤中间决策过程的 token
        async for event in agent_with_memory.astream_events(
            {"input": req.message}, config=config, version="v2"
        ):
            kind = event["event"]

            if kind == "on_tool_start":
                # 工具开始调用 — 推送工具名称和输入参数
                payload = json.dumps({
                    "tool": event["name"],
                    "input": event["data"].get("input", {}),
                }, ensure_ascii=False)
                yield f"event: tool_start\ndata: {payload}\n\n"

            elif kind == "on_tool_end":
                # 工具执行完成 — 推送工具返回结果
                payload = json.dumps({
                    "tool": event["name"],
                    "output": event["data"].get("output", ""),
                }, ensure_ascii=False)
                yield f"event: tool_end\ndata: {payload}\n\n"

            elif kind == "on_chat_model_stream":
                # LLM 生成的 token — 只推送最终回答的内容
                chunk = event["data"].get("chunk")
                if chunk and chunk.content and isinstance(chunk.content, str):
                    yield f"data: {chunk.content}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/clear")
async def clear(req: ChatRequest):
    store.pop(req.session_id, None)
    return {"status": "ok"}
