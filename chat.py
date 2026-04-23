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
    """LangGraph ReAct 模式（第四阶段）"""
    from langgraph.graph import StateGraph, START, END, MessagesState
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.memory import MemorySaver

    llm_with_tools = llm.bind_tools(all_tools)
    tool_node = ToolNode(all_tools)

    def agent_node(state: MessagesState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    session_id = "langgraph"
    config = {"configurable": {"thread_id": session_id}}

    print("🤖 DeepSeek Agent — LangGraph ReAct 模式（输入 quit 退出，输入 clear 清空历史）")
    print("🔧 可用工具: 数学计算 | 当前时间 | 天气查询 | 📚 知识库检索\n")

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
                {"messages": [HumanMessage(content=user_input)]},
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
    print("  2. LangGraph（第四阶段，手动构建图）\n")

    choice = input("请输入 1 或 2: ").strip()
    if choice == "2":
        run_langgraph_agent()
    else:
        run_agent_executor()

if __name__ == "__main__":
    main()
