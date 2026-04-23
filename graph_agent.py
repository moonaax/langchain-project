"""
LangGraph ReAct Agent（终端版）

第四阶段：用 LangGraph 手动构建 ReAct 图，替代 AgentExecutor 黑盒模式

AI Agent 知识点索引：
- LangGraph: StateGraph、Node、Edge、条件路由
- ReAct: 思考→行动→观察 循环
- ToolNode: LangGraph 内置工具执行节点
- MessagesState: LangGraph 内置消息状态
- MemorySaver: 内存检查点，支持多轮对话持久化
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from tools import all_tools

load_dotenv()

# ============================================================
# 【知识点】LLM 初始化 — 复用 DeepSeek 配置
# ============================================================
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7,
)

# 给 LLM 绑定工具（LangGraph 中 LLM 需要 bind_tools）
llm_with_tools = llm.bind_tools(all_tools)

# ============================================================
# 【知识点】ToolNode — LangGraph 内置工具执行节点
# ============================================================
# ToolNode 自动处理 AIMessage 中的 tool_calls：
# 1. 提取 tool_calls 列表
# 2. 调用对应工具
# 3. 返回 ToolMessage 列表
tool_node = ToolNode(all_tools)

# ============================================================
# 【知识点】agent_node — LLM 推理节点
# ============================================================
def agent_node(state: MessagesState) -> dict:
    """LLM 推理：接收消息历史，返回 AIMessage（可能包含 tool_calls）"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ============================================================
# 【知识点】should_continue — 条件路由
# ============================================================
# 根据最后一条 AIMessage 是否包含 tool_calls 决定走向：
#   有 tool_calls → 执行工具（tools 节点）
#   无 tool_calls → 结束（END）
def should_continue(state: MessagesState) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END

# ============================================================
# 【知识点】StateGraph 构建 — 手动搭建 ReAct 图
# ============================================================
# 图结构：
#   START → agent ──┬── 有 tool_calls → tools → agent（循环）
#                   └── 无 tool_calls → END
graph = StateGraph(MessagesState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")  # 工具执行完回到 agent，形成循环

# ============================================================
# 【知识点】MemorySaver — 内存检查点，支持多轮对话
# ============================================================
memory = MemorySaver()
app = graph.compile(checkpointer=memory)


def main():
    session_id = "default"
    config = {"configurable": {"thread_id": session_id}}

    print("🤖 DeepSeek LangGraph Agent（输入 quit 退出，输入 clear 清空历史）")
    print("🔧 可用工具: 数学计算 | 当前时间 | 天气查询 | 📚 知识库检索")
    print("📊 执行模式: LangGraph ReAct（手动构建图）\n")

    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("再见！")
            break
        if user_input.lower() == "clear":
            memory.delete_thread(config["configurable"]["thread_id"])
            print("✅ 对话历史已清空\n")
            continue

        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )
            # 最后一条消息就是 AI 的最终回答
            ai_msg = result["messages"][-1]
            print(f"\nAI: {ai_msg.content}\n")
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


if __name__ == "__main__":
    main()
