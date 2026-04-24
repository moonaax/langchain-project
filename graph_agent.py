"""
LangGraph ReAct Agent（终端版）

第四阶段：用 LangGraph 手动构建 ReAct 图，替代 AgentExecutor 黑盒模式
第四阶段进阶：自纠错循环 — 工具失败时自动注入提示，引导 LLM 换策略重试

AI Agent 知识点索引：
- LangGraph: StateGraph、Node、Edge、条件路由
- ReAct: 思考→行动→观察 循环
- ToolNode: LangGraph 内置工具执行节点
- MemorySaver: 内存检查点，支持多轮对话持久化
- 自纠错: 扩展 State 追踪重试次数，工具失败后路由到 corrector 节点注入提示
"""

import os
from typing import TypedDict, Annotated
from operator import add
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
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
tool_node = ToolNode(all_tools)

# ============================================================
# 【知识点】自纠错 State — 扩展 MessagesState，追踪重试次数
# ============================================================
# 默认 MessagesState 只有 messages 字段
# 我们额外加 retry_count（重试计数）和 last_error（上次错误信息）
MAX_RETRIES = 3

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add]  # 消息历史（追加模式）
    retry_count: int                               # 当前重试次数（覆盖模式）
    last_error: str                                # 上次工具失败的错误信息

# ============================================================
# 【知识点】agent_node — LLM 推理节点
# ============================================================
def agent_node(state: AgentState) -> dict:
    """LLM 推理：接收消息历史，返回 AIMessage（可能包含 tool_calls）"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ============================================================
# 【知识点】should_continue — 条件路由（LLM 输出后）
# ============================================================
def should_continue(state: AgentState) -> str:
    """根据 LLM 输出决定走向：有 tool_calls → tools，否则 END"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END

# ============================================================
# 【知识点】check_tool_result — 工具执行后检查是否失败
# ============================================================
def check_tool_result(state: AgentState) -> str:
    """工具执行后路由：失败 → corrector，成功 → agent

    判断逻辑：最后一条 ToolMessage 的 content 是否包含错误关键词
    """
    last_msg = state["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else ""
    retry_count = state.get("retry_count", 0)

    # 检测工具失败的常见模式
    is_error = any(kw in content for kw in ["错误", "error", "未找到", "失败", "无法"])

    if is_error and retry_count < MAX_RETRIES:
        return "corrector"  # 失败且未超限 → 自纠错
    return "agent"          # 成功或已超限 → 正常回到 agent

# ============================================================
# 【知识点】corrector_node — 自纠错节点（核心）
# ============================================================
def corrector_node(state: AgentState) -> dict:
    """自纠错：分析失败原因，注入提示引导 LLM 换策略

    这是自纠错循环的核心：
    1. 读取上次错误信息
    2. 构造纠错提示（告诉 LLM 哪里失败、建议换什么策略）
    3. 递增重试计数
    """
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
    print("📊 执行模式: LangGraph ReAct + 自纠错（最多重试 3 次）\n")

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
                {"messages": [HumanMessage(content=user_input)], "retry_count": 0, "last_error": ""},
                config=config,
            )
            # 最后一条消息就是 AI 的最终回答
            ai_msg = result["messages"][-1]
            print(f"\nAI: {ai_msg.content}\n")
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


if __name__ == "__main__":
    main()
