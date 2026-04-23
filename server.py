"""
FastAPI 后端：LangChain + DeepSeek Agent API

第三阶段：新增知识库 RAG 检索工具
第四阶段：新增 LangGraph ReAct Agent 端点

AI Agent 知识点索引：
- Agent API: AgentExecutor + astream_events 流式推送（旧）
- LangGraph: StateGraph + ToolNode + 条件路由（新）
- RAG Tool: knowledge_search 工具，Agent 自主决定是否检索
- SSE 协议: tool_start / tool_end / token / done 事件类型
"""

import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from tools import all_tools

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个有帮助的 AI 助手，可以使用工具来回答问题。

你可以使用以下工具：
- calculator: 数学计算
- get_current_time: 获取当前时间
- search_weather: 查询天气
- knowledge_search: 从知识库检索技术文档（LangChain、Agent、RAG、Android 等）

当用户询问技术知识时，优先使用 knowledge_search 从知识库中检索，基于检索结果回答。
如果知识库中没有相关信息，再用你自己的知识回答。
如果用户的问题可以直接回答，不需要使用工具。

格式要求：
- 输出 Markdown 时，每个标题（#）前必须有空行
- 表格每行必须独占一行，表头、分隔行、数据行之间不能有空行
- 代码块必须用 ``` 包裹，不要省略"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, all_tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=all_tools,
    max_iterations=10, handle_parsing_errors=True,
)

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

agent_with_memory = RunnableWithMessageHistory(
    agent_executor, get_session_history,
    input_messages_key="input", history_messages_key="history",
)

# ============================================================
# 【知识点】第四阶段 — LangGraph ReAct Agent
# ============================================================
# 用 LangGraph 手动构建 ReAct 图，替代 AgentExecutor 黑盒模式
# 图结构：START → agent → ┬── 有 tool_calls → tools → agent（循环）
#                         └── 无 tool_calls → END

llm_with_tools = llm.bind_tools(all_tools)
tool_node = ToolNode(all_tools)

def graph_agent_node(state: MessagesState) -> dict:
    """LLM 推理节点"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: MessagesState) -> str:
    """条件路由：有 tool_calls → tools，否则 END"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END

graph = StateGraph(MessagesState)
graph.add_node("agent", graph_agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

graph_memory = MemorySaver()
graph_app = graph.compile(checkpointer=graph_memory)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ClearRequest(BaseModel):
    session_id: str = "default"

# ============================================================
# 【知识点】astream_events — 流式推送 Agent 推理过程（同第二阶段）
# ============================================================
@app.post("/chat")
async def chat(req: ChatRequest):
    config = {"configurable": {"session_id": req.session_id}}

    async def generate():
        in_tool = False  # 标记是否在工具调用过程中
        async for event in agent_with_memory.astream_events(
            {"input": req.message}, config=config, version="v2"
        ):
            kind = event["event"]

            if kind == "on_tool_start":
                in_tool = True
                payload = json.dumps({
                    "tool": event["name"],
                    "input": event["data"].get("input", {}),
                }, ensure_ascii=False)
                yield f"event: tool_start\ndata: {payload}\n\n"

            elif kind == "on_tool_end":
                in_tool = False
                output = event["data"].get("output", "")
                if not isinstance(output, str):
                    output = str(output)
                payload = json.dumps({
                    "tool": event["name"],
                    "output": output,
                }, ensure_ascii=False)
                yield f"event: tool_end\ndata: {payload}\n\n"

            elif kind == "on_chat_model_stream" and not in_tool:
                # 只推送非工具调用阶段的 token（即最终回答）
                chunk = event["data"].get("chunk")
                if chunk and chunk.content and isinstance(chunk.content, str):
                    yield f"data: {chunk.content}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# ============================================================
# 【知识点】第四阶段 — LangGraph 流式端点
# ============================================================
# 用 graph_app.astream_events 推流，与 AgentExecutor 端点协议一致
@app.post("/graph_chat")
async def graph_chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.session_id}}

    async def generate():
        in_tool = False
        async for event in graph_app.astream_events(
            {"messages": [HumanMessage(content=req.message)]},
            config=config, version="v2",
        ):
            kind = event["event"]

            if kind == "on_tool_start":
                in_tool = True
                payload = json.dumps({
                    "tool": event["name"],
                    "input": event["data"].get("input", {}),
                }, ensure_ascii=False)
                yield f"event: tool_start\ndata: {payload}\n\n"

            elif kind == "on_tool_end":
                in_tool = False
                output = event["data"].get("output", "")
                if not isinstance(output, str):
                    output = str(output)
                payload = json.dumps({
                    "tool": event["name"],
                    "output": output,
                }, ensure_ascii=False)
                yield f"event: tool_end\ndata: {payload}\n\n"

            elif kind == "on_chat_model_stream" and not in_tool:
                chunk = event["data"].get("chunk")
                if chunk and chunk.content and isinstance(chunk.content, str):
                    yield f"data: {chunk.content}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/clear")
async def clear(req: ClearRequest):
    store.pop(req.session_id, None)
    graph_memory.delete_thread(req.session_id)
    return {"status": "ok"}
