"""
LangChain + DeepSeek Agent（终端版）

第三阶段：在 Agent 基础上添加知识库 RAG 检索工具
第四阶段：新增 LangGraph ReAct 模式

AI Agent 知识点索引：
- Tool:  工具模块化，从 tools.py 统一导入
- Agent: create_tool_calling_agent + AgentExecutor（旧模式）
- LangGraph: StateGraph + ToolNode + 条件路由（新模式）
- RAG:   knowledge_search 工具，Agent 自主决定是否检索知识库
- Memory: RunnableWithMessageHistory / MemorySaver 多轮对话记忆
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor

# ============================================================
# 【知识点】工具模块化 — 从 tools.py 统一导入
# ============================================================
from tools import all_tools

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7,
)

# ============================================================
# 【知识点】Agent Prompt — 新增知识库检索工具说明
# ============================================================
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个有帮助的 AI 助手，可以使用工具来回答问题。

你可以使用以下工具：
- calculator: 数学计算
- get_current_time: 获取当前时间
- search_weather: 查询天气
- knowledge_search: 从知识库检索技术文档（LangChain、Agent、RAG、Android 等）

当用户询问技术知识时，优先使用 knowledge_search 从知识库中检索，基于检索结果回答。
如果知识库中没有相关信息，再用你自己的知识回答。
如果用户的问题可以直接回答，不需要使用工具。"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, all_tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=all_tools,
    verbose=True, max_iterations=10, handle_parsing_errors=True,
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

def run_agent_executor():
    """AgentExecutor 模式（第三阶段）"""
    session_id = "default"
    config = {"configurable": {"session_id": session_id}}

    print("🤖 DeepSeek Agent — AgentExecutor 模式（输入 quit 退出，输入 clear 清空历史）")
    print("🔧 可用工具: 数学计算 | 当前时间 | 天气查询 | 📚 知识库检索\n")

    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("再见！")
            break
        if user_input.lower() == "clear":
            store.pop(session_id, None)
            print("✅ 对话历史已清空\n")
            continue

        try:
            result = agent_with_memory.invoke(
                {"input": user_input}, config=config
            )
            print(f"\nAI: {result['output']}\n")
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


def run_langgraph_agent():
    """LangGraph ReAct + 自纠错模式（第四阶段进阶）"""
    from typing import TypedDict, Annotated
    from operator import add
    from langgraph.graph import StateGraph, START, END
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.messages import SystemMessage, BaseMessage

    llm_with_tools = llm.bind_tools(all_tools)
    tool_node = ToolNode(all_tools)
    MAX_RETRIES = 3

    # ============================================================
    # 【知识点】自纠错 State — 扩展 MessagesState，追踪重试次数
    # ============================================================
    # 默认 MessagesState 只有 messages 字段
    # 自纠错需要额外追踪：retry_count（重试计数）、last_error（上次错误）
    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add]  # 消息历史（追加模式）
        retry_count: int                               # 当前重试次数（覆盖模式）
        last_error: str                                # 上次工具失败的错误信息

    # ============================================================
    # 【知识点】agent_node — LLM 推理节点
    # ============================================================
    def agent_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # ============================================================
    # 【知识点】should_continue — 条件路由（LLM 输出后）
    # ============================================================
    def should_continue(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return END

    # ============================================================
    # 【知识点】check_tool_result — 工具执行后检查是否失败
    # ============================================================
    # 工具执行后路由：失败 → corrector，成功 → agent
    # 判断逻辑：最后一条 ToolMessage 的 content 是否包含错误关键词
    def check_tool_result(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        content = last_msg.content if hasattr(last_msg, "content") else ""
        retry_count = state.get("retry_count", 0)
        is_error = any(kw in content for kw in ["错误", "error", "未找到", "失败", "无法"])
        if is_error and retry_count < MAX_RETRIES:
            return "corrector"  # 失败且未超限 → 自纠错
        return "agent"          # 成功或已超限 → 正常回到 agent

    # ============================================================
    # 【知识点】corrector_node — 自纠错节点（核心）
    # ============================================================
    # 自纠错的核心：把错误信息转化为结构化的重试提示
    # 1. 读取上次错误信息
    # 2. 构造纠错提示（告诉 LLM 哪里失败、建议换什么策略）
    # 3. 递增重试计数
    def corrector_node(state: AgentState) -> dict:
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

    # ============================================================
    # 【知识点】StateGraph 构建 — 带自纠错的 ReAct 图
    # ============================================================
    # 图结构：
    #   START → agent ──有 tool_calls──→ tools ──失败──→ corrector → agent（纠错循环）
    #                │                   │
    #                │                   └──成功──→ agent（正常循环）
    #                │
    #                └──无 tool_calls──→ END
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("corrector", corrector_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", check_tool_result, {"corrector": "corrector", "agent": "agent"})
    graph.add_edge("corrector", "agent")

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    session_id = "langgraph"
    config = {"configurable": {"thread_id": session_id}}

    print("🤖 DeepSeek Agent — LangGraph ReAct + 自纠错模式（输入 quit 退出，输入 clear 清空历史）")
    print("🔧 可用工具: 数学计算 | 当前时间 | 天气查询 | 📚 知识库检索")
    print("📊 自纠错: 工具失败时自动重试，最多 3 次\n")

    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("再见！")
            break
        if user_input.lower() == "clear":
            memory.delete_thread(session_id)
            print("✅ 对话历史已清空\n")
            continue

        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)], "retry_count": 0, "last_error": ""},
                config=config,
            )
            ai_msg = result["messages"][-1]
            print(f"\nAI: {ai_msg.content}\n")
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


def main():
    print("🤖 DeepSeek Agent 终端版\n")
    print("选择执行模式：")
    print("  1. AgentExecutor（第三阶段，黑盒循环）")
    print("  2. LangGraph ReAct + 自纠错（第四阶段进阶）\n")

    choice = input("请输入 1 或 2: ").strip()
    if choice == "2":
        run_langgraph_agent()
    else:
        run_agent_executor()

if __name__ == "__main__":
    main()
