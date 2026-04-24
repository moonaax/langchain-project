"""
FastAPI 后端：LangChain + DeepSeek Agent API

第三阶段：新增知识库 RAG 检索工具
第四阶段：新增 LangGraph ReAct Agent 端点
第四阶段进阶：
  - 自纠错循环 — 工具失败时自动换策略重试
  - Plan-and-Execute — 先规划再执行，提高复杂任务成功率

AI Agent 知识点索引：
- Agent API: AgentExecutor + astream_events 流式推送（旧）
- LangGraph: StateGraph + ToolNode + 条件路由（新）
- 自纠错: AgentState + corrector 节点 + 条件路由
- Plan-and-Execute: planner → executor → replanner 图结构
- RAG Tool: knowledge_search 工具，Agent 自主决定是否检索
- SSE 协议: tool_start / tool_end / tool_retry / plan_step / token / done
"""

import os
import json
from pathlib import Path
from typing import TypedDict, Annotated
from operator import add
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite
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
- 表格必须严格按以下格式，每行独占一行，行与行之间不能合并到同一行：
  | 列1 | 列2 |
  |------|------|
  | 值1 | 值2 |
  错误示范：| 列1 | 列2 | 值1 | 值2 | （所有内容不能在同一行）
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
# 【知识点】第四阶段 — LangGraph ReAct Agent + 自纠错
# ============================================================
# 图结构（带自纠错）：
#   START → agent → ┬── 有 tool_calls → tools → ┬── 失败 → corrector → agent（纠错循环）
#                   │                           │
#                   │                           └── 成功 → agent（正常循环）
#                   │
#                   └── 无 tool_calls → END

llm_with_tools = llm.bind_tools(all_tools)
tool_node = ToolNode(all_tools)

MAX_RETRIES = 3

class AgentState(TypedDict):
    """扩展状态：追踪重试次数和错误信息"""
    messages: Annotated[list[BaseMessage], add]
    retry_count: int
    last_error: str

def _clean_tool_calls(messages):
    """清理不完整的 tool_calls 消息

    OpenAI API 要求 assistant 的 tool_calls 后面必须跟对应的 ToolMessage。
    如果 graph 中途崩溃或被中断，checkpointer 保存的状态可能只有 tool_calls
    没有 ToolMessage，导致下次调用 400 错误。
    """
    from langchain_core.messages import AIMessage, ToolMessage

    # 收集所有已响应的 tool_call_id
    responded_ids = set()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            responded_ids.add(msg.tool_call_id)

    # 过滤：跳过没有对应 ToolMessage 的 assistant tool_calls
    cleaned = []
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            missing = [tc for tc in msg.tool_calls if tc["id"] not in responded_ids]
            if missing:
                # 这条 assistant 消息有未响应的 tool_calls，跳过
                continue
        cleaned.append(msg)
    return cleaned

def graph_agent_node(state: AgentState) -> dict:
    """LLM 推理节点"""
    messages = _clean_tool_calls(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """条件路由：有 tool_calls → tools，否则 END"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END

def check_tool_result(state: AgentState) -> str:
    """工具执行后路由：失败 → corrector，成功 → agent"""
    last_msg = state["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else ""
    retry_count = state.get("retry_count", 0)
    is_error = any(kw in content for kw in ["错误", "error", "未找到", "失败", "无法"])
    if is_error and retry_count < MAX_RETRIES:
        return "corrector"
    return "agent"

def corrector_node(state: AgentState) -> dict:
    """自纠错：分析失败原因，注入提示引导 LLM 换策略"""
    last_error = state["messages"][-1].content
    retry_count = state.get("retry_count", 0) + 1
    correction_prompt = f"""⚠️ 工具调用失败（第 {retry_count} 次重试）：
{last_error}

请分析失败原因并换一种方式尝试：
- 如果是参数错误，请修正参数后重试
- 如果是工具选择错误，请换一个更合适的工具
- 如果所有工具都无法解决，请直接用你的知识回答，并说明工具不可用"""
    return {
        "messages": [SystemMessage(content=correction_prompt)],
        "retry_count": retry_count,
        "last_error": last_error,
    }

graph = StateGraph(AgentState)
graph.add_node("agent", graph_agent_node)
graph.add_node("tools", tool_node)
graph.add_node("corrector", corrector_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_conditional_edges("tools", check_tool_result, {"corrector": "corrector", "agent": "agent"})
graph.add_edge("corrector", "agent")

# ============================================================
# 【知识点】AsyncSqliteSaver — SQLite 持久化检查点（重启不丢对话）
# ============================================================
# 替换 MemorySaver（纯内存），对话历史写入 checkpoints.db
# 4 个模式共用同一个 SQLite 文件，用 thread_id 区分会话
# FastAPI 是异步框架，必须用 AsyncSqliteSaver（不能用同步版 SqliteSaver）
# 需要在启动事件中初始化（aiosqlite 需要事件循环）
_db_path = str(Path(__file__).parent / "checkpoints.db")
_checkpointer = None  # 启动时初始化

@app.on_event("startup")
async def init_checkpointer():
    global _checkpointer, graph_app, plan_app, human_app
    conn = await aiosqlite.connect(_db_path)
    _checkpointer = AsyncSqliteSaver(conn=conn)
    await _checkpointer.setup()
    # 编译图（需要 checkpointer 就绪后才能编译）
    graph_app = graph.compile(checkpointer=_checkpointer)
    plan_app = plan_graph.compile(checkpointer=_checkpointer)
    human_app = human_graph.compile(
        checkpointer=_checkpointer,
        interrupt_before=["tools"],
    )

# 占位，启动时替换
graph_app = None
plan_app = None
human_app = None

# ============================================================
# 【知识点】第四阶段进阶 — Plan-and-Execute 图
# ============================================================
# 图结构：
#   START → planner → executor → replanner ──继续──→ executor（循环）
#                                  │
#                                  └──完成──→ finish → END

class PlanExecuteState(TypedDict):
    """Plan-and-Execute 状态：追踪计划、执行进度、结果"""
    messages: Annotated[list[BaseMessage], add]
    plan: list[str]
    current_step: int
    past_steps: list[dict]
    response: str

PLANNER_PROMPT = """你是一个任务规划专家。请把用户的问题拆解为清晰的执行步骤。

可用工具：
- calculator: 数学计算
- get_current_time: 获取当前时间
- search_weather: 查询天气
- knowledge_search: 从知识库检索技术文档

规则：
1. 每个步骤应该是独立可执行的
2. 步骤之间有逻辑顺序
3. 需要检索知识的步骤，明确写出检索关键词
4. 最后一步应该是整合信息并给出最终回答
5. 步骤数量控制在 2-5 步

请用 JSON 数组格式输出步骤列表，例如：
["步骤1: 查询xxx", "步骤2: 计算xxx", "步骤3: 整合回答"]

只输出 JSON 数组，不要输出其他内容。"""

def plan_planner_node(state: PlanExecuteState) -> dict:
    """规划节点：根据用户问题生成执行计划"""
    user_query = state["messages"][0].content
    response = llm.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"用户问题：{user_query}"),
    ])
    content = response.content
    try:
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        plan = json.loads(content.strip())
    except (json.JSONDecodeError, IndexError):
        plan = [f"直接回答用户的问题：{user_query}"]
    return {"plan": plan, "current_step": 0, "past_steps": [], "response": ""}

EXECUTOR_PROMPT = """你是一个任务执行专家。请执行当前步骤。

已执行的步骤和结果：
{past_steps}

当前要执行的步骤：{current_step}

可用工具：
- calculator: 数学计算
- get_current_time: 获取当前时间
- search_weather: 查询天气
- knowledge_search: 从知识库检索技术文档

请直接执行这个步骤。如果需要调用工具就调用工具，不需要就直接输出结果。"""

def plan_executor_node(state: PlanExecuteState) -> dict:
    """执行节点：执行当前步骤，支持工具调用"""
    plan = state["plan"]
    current_idx = state["current_step"]
    if current_idx >= len(plan):
        return {"response": "所有步骤已执行完成。"}

    current_step = plan[current_idx]
    past_steps_text = ""
    if state["past_steps"]:
        for i, ps in enumerate(state["past_steps"]):
            past_steps_text += f"步骤{i+1}: {ps['step']}\n结果: {ps['result']}\n\n"

    prompt = EXECUTOR_PROMPT.format(
        past_steps=past_steps_text or "（这是第一个步骤）",
        current_step=current_step,
    )

    response = llm_with_tools.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"请执行：{current_step}"),
    ])

    result = response.content
    if hasattr(response, "tool_calls") and response.tool_calls:
        from langchain_core.messages import ToolMessage
        tool_results = []
        for tc in response.tool_calls:
            for tool in all_tools:
                if tool.name == tc["name"]:
                    try:
                        tool_output = tool.invoke(tc["args"])
                        tool_results.append(f"[{tc['name']}] {tool_output}")
                    except Exception as e:
                        tool_results.append(f"[{tc['name']}] 错误: {e}")
                    break
        result = "\n".join(tool_results)

    past_steps = state["past_steps"] + [{"step": current_step, "result": result}]
    return {
        "past_steps": past_steps,
        "current_step": current_idx + 1,
        "messages": [SystemMessage(content=f"步骤 {current_idx + 1} 执行结果：{result}")],
    }

REPLANNER_PROMPT = """你是一个任务评估专家。请根据已执行的步骤结果，决定下一步。

用户原始问题：{original_query}

已执行的步骤：
{past_steps}

剩余计划：
{remaining_plan}

请判断：
1. 如果剩余计划仍然合理，输出 {{"action": "continue"}}
2. 如果需要调整计划，输出 {{"action": "replan", "new_plan": ["步骤1", "步骤2", ...]}}
3. 如果已经可以给出最终回答，输出 {{"action": "finish", "response": "最终回答内容"}}

只输出 JSON，不要输出其他内容。"""

def plan_replanner_node(state: PlanExecuteState) -> dict:
    """重新规划节点：根据执行结果动态调整计划"""
    original_query = state["messages"][0].content
    past_steps = state["past_steps"]
    plan = state["plan"]
    current_step = state["current_step"]

    past_steps_text = ""
    for i, ps in enumerate(past_steps):
        past_steps_text += f"步骤{i+1}: {ps['step']}\n结果: {ps['result']}\n\n"

    remaining = plan[current_step:]
    remaining_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(remaining)) if remaining else "（无）"

    prompt = REPLANNER_PROMPT.format(
        original_query=original_query,
        past_steps=past_steps_text or "（无）",
        remaining_plan=remaining_text,
    )

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="请评估并决定下一步。"),
    ])

    try:
        content = response.content
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        decision = json.loads(content.strip())
    except (json.JSONDecodeError, IndexError):
        decision = {"action": "continue"}

    action = decision.get("action", "continue")
    if action == "finish":
        return {"response": decision.get("response", "任务完成。")}
    elif action == "replan":
        new_plan = decision.get("new_plan", plan[current_step:])
        return {"plan": new_plan, "current_step": 0, "past_steps": past_steps}
    return {}

def plan_should_continue(state: PlanExecuteState) -> str:
    """判断计划是否执行完成"""
    if state.get("response"):
        return "finish"
    if state.get("current_step", 0) >= len(state.get("plan", [])):
        return "finish"
    return "continue"

def plan_finish_node(state: PlanExecuteState) -> dict:
    """将最终回答添加到消息历史"""
    return {"messages": [AIMessage(content=state["response"])]}

plan_graph = StateGraph(PlanExecuteState)
plan_graph.add_node("planner", plan_planner_node)
plan_graph.add_node("executor", plan_executor_node)
plan_graph.add_node("replanner", plan_replanner_node)
plan_graph.add_node("finish", plan_finish_node)
plan_graph.add_edge(START, "planner")
plan_graph.add_edge("planner", "executor")
plan_graph.add_edge("executor", "replanner")
plan_graph.add_conditional_edges("replanner", plan_should_continue, {
    "continue": "executor",
    "finish": "finish",
})
plan_graph.add_edge("finish", END)

# ============================================================
# 【知识点】第四阶段进阶 — 人机协作图（interrupt_before）
# ============================================================
# 图结构与 ReAct 相同，但在 tools 节点前暂停，等用户确认后才执行
# 关键 API：
#   interrupt_before=["tools"]  — 编译时声明暂停点
#   app.invoke(None, config)    — resume 继续执行
#   app.get_state(config)       — 获取当前暂停状态（含待执行的 tool_calls）

human_graph = StateGraph(AgentState)
human_graph.add_node("agent", graph_agent_node)
human_graph.add_node("tools", tool_node)
human_graph.add_node("corrector", corrector_node)
human_graph.add_edge(START, "agent")
human_graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
human_graph.add_conditional_edges("tools", check_tool_result, {"corrector": "corrector", "agent": "agent"})
human_graph.add_edge("corrector", "agent")

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
# 【知识点】第四阶段 — LangGraph 流式端点（带自纠错）
# ============================================================
# 用 graph_app.astream_events 推流，与 AgentExecutor 端点协议一致
# 新增：corrector 节点触发时推送 tool_retry 事件
@app.post("/graph_chat")
async def graph_chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.session_id}}

    async def generate():
        in_tool = False
        async for event in graph_app.astream_events(
            {"messages": [HumanMessage(content=req.message)], "retry_count": 0, "last_error": ""},
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

            elif kind == "on_chain_start" and event.get("name") == "corrector":
                # 自纠错节点触发时通知前端
                yield f"event: tool_retry\ndata: {{\"message\": \"工具调用失败，正在自动重试...\"}}\n\n"

            elif kind == "on_chat_model_stream" and not in_tool:
                chunk = event["data"].get("chunk")
                if chunk and chunk.content and isinstance(chunk.content, str):
                    yield f"data: {chunk.content}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# ============================================================
# 【知识点】第四阶段进阶 — Plan-and-Execute 流式端点
# ============================================================
# 用 plan_app.astream_events 推流，新增 plan_step 事件通知前端当前执行步骤
@app.post("/plan_chat")
async def plan_chat(req: ChatRequest):
    config = {"configurable": {"thread_id": f"plan_{req.session_id}"}}

    async def generate():
        async for event in plan_app.astream_events(
            {"messages": [HumanMessage(content=req.message)], "plan": [], "current_step": 0, "past_steps": [], "response": ""},
            config=config, version="v2",
        ):
            kind = event["event"]

            # 规划节点完成时，推送计划步骤
            if kind == "on_chain_end" and event.get("name") == "planner":
                output = event["data"].get("output", {})
                plan = output.get("plan", [])
                if plan:
                    payload = json.dumps({"plan": plan}, ensure_ascii=False)
                    yield f"event: plan_start\ndata: {payload}\n\n"

            # 执行节点完成时，推送当前步骤结果
            elif kind == "on_chain_end" and event.get("name") == "executor":
                output = event["data"].get("output", {})
                current_step = output.get("current_step", 0)
                past_steps = output.get("past_steps", [])
                if past_steps:
                    last_step = past_steps[-1]
                    payload = json.dumps({
                        "step": current_step,
                        "description": last_step["step"],
                        "result": last_step["result"][:500],
                    }, ensure_ascii=False)
                    yield f"event: plan_step\ndata: {payload}\n\n"

            # 工具调用事件
            elif kind == "on_tool_start":
                payload = json.dumps({
                    "tool": event["name"],
                    "input": event["data"].get("input", {}),
                }, ensure_ascii=False)
                yield f"event: tool_start\ndata: {payload}\n\n"

            elif kind == "on_tool_end":
                output = event["data"].get("output", "")
                if not isinstance(output, str):
                    output = str(output)
                payload = json.dumps({
                    "tool": event["name"],
                    "output": output,
                }, ensure_ascii=False)
                yield f"event: tool_end\ndata: {payload}\n\n"

            # LLM 流式输出（最终回答）
            elif kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and chunk.content and isinstance(chunk.content, str):
                    yield f"data: {chunk.content}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# ============================================================
# 【知识点】第四阶段进阶 — 人机协作端点
# ============================================================
# 人机协作流程：
#   1. /human_chat: 发送消息，Agent 推理后可能暂停在 tools 节点
#   2. 如果暂停：返回 confirm_required 事件，包含待执行的 tool_calls
#   3. /human_confirm: 用户确认或取消
#      - confirm=true: resume 继续执行工具
#      - confirm=false: 注入拒绝消息，Agent 换策略
class ConfirmRequest(BaseModel):
    session_id: str = "default"
    confirm: bool

@app.post("/human_chat")
async def human_chat(req: ChatRequest):
    config = {"configurable": {"thread_id": f"human_{req.session_id}"}}

    # 第一次运行，可能在 tools 节点前暂停
    result = human_app.invoke(
        {"messages": [HumanMessage(content=req.message)], "retry_count": 0, "last_error": ""},
        config=config,
    )

    # 检查是否暂停在 tools 节点
    snapshot = human_app.get_state(config)
    if snapshot.next and "tools" in snapshot.next:
        # 提取待执行的 tool_calls
        last_msg = snapshot.values["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            tool_calls = [{
                "id": tc["id"],
                "name": tc["name"],
                "args": tc["args"],
            } for tc in last_msg.tool_calls]
            return {
                "status": "confirm_required",
                "tool_calls": tool_calls,
                "session_id": req.session_id,
            }

    # 没有暂停，直接返回最终回答
    ai_msg = result["messages"][-1]
    return {"status": "done", "content": ai_msg.content}

@app.post("/human_confirm")
async def human_confirm(req: ConfirmRequest):
    config = {"configurable": {"thread_id": f"human_{req.session_id}"}}

    if req.confirm:
        # 确认：resume 继续执行工具
        result = human_app.invoke(None, config=config)
    else:
        # 取消：注入拒绝消息，让 agent 换策略
        snapshot = human_app.get_state(config)
        last_msg = snapshot.values["messages"][-1]
        from langchain_core.messages import ToolMessage
        rejection_msgs = []
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                rejection_msgs.append(ToolMessage(
                    content="用户拒绝了此工具调用。请直接用你的知识回答用户的问题，不要尝试调用任何工具。",
                    tool_call_id=tc["id"],
                ))
        human_app.update_state(config, {"messages": rejection_msgs})
        result = human_app.invoke(None, config=config)

    ai_msg = result["messages"][-1]
    return {"status": "done", "content": ai_msg.content}

@app.post("/clear")
async def clear(req: ClearRequest):
    store.pop(req.session_id, None)
    await _checkpointer.delete_thread(req.session_id)
    await _checkpointer.delete_thread(f"plan_{req.session_id}")
    await _checkpointer.delete_thread(f"human_{req.session_id}")
    return {"status": "ok"}
