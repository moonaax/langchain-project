"""
LangGraph Agent（终端版）

第四阶段：用 LangGraph 手动构建 ReAct 图，替代 AgentExecutor 黑盒模式
第四阶段进阶：
  - 自纠错循环：工具失败时自动注入提示，引导 LLM 换策略重试
  - Plan-and-Execute：先规划再执行，提高复杂任务成功率
  - 人机协作：关键操作前暂停等用户确认（interrupt_before）

AI Agent 知识点索引：
- LangGraph: StateGraph、Node、Edge、条件路由
- ReAct: 思考→行动→观察 循环
- ToolNode: LangGraph 内置工具执行节点
- MemorySaver: 内存检查点，支持多轮对话持久化
- 自纠错: 扩展 State 追踪重试次数，工具失败后路由到 corrector 节点注入提示
- Plan-and-Execute: planner 规划步骤 → executor 逐步执行 → re-planner 动态调整
- 人机协作: interrupt_before 暂停图执行，等待用户确认后 resume
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
# ============================================================
#  模式一：ReAct + 自纠错
# ============================================================
# ============================================================

# ============================================================
# 【知识点】自纠错 State — 扩展 MessagesState，追踪重试次数
# ============================================================
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
    """工具执行后路由：失败 → corrector，成功 → agent"""
    last_msg = state["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else ""
    retry_count = state.get("retry_count", 0)
    is_error = any(kw in content for kw in ["错误", "error", "未找到", "失败", "无法"])
    if is_error and retry_count < MAX_RETRIES:
        return "corrector"
    return "agent"

# ============================================================
# 【知识点】corrector_node — 自纠错节点（核心）
# ============================================================
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

# ============================================================
# 【知识点】StateGraph 构建 — 带自纠错的 ReAct 图
# ============================================================
# 图结构：
#   START → agent ──有 tool_calls──→ tools ──失败──→ corrector → agent（纠错循环）
#                │                   │
#                │                   └──成功──→ agent（正常循环）
#                │
#                └──无 tool_calls──→ END
react_graph = StateGraph(AgentState)

react_graph.add_node("agent", agent_node)
react_graph.add_node("tools", tool_node)
react_graph.add_node("corrector", corrector_node)

react_graph.add_edge(START, "agent")
react_graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
react_graph.add_conditional_edges("tools", check_tool_result, {"corrector": "corrector", "agent": "agent"})
react_graph.add_edge("corrector", "agent")

react_memory = MemorySaver()
react_app = react_graph.compile(checkpointer=react_memory)


# ============================================================
# ============================================================
#  模式二：Plan-and-Execute（先规划再执行）
# ============================================================
# ============================================================

# ============================================================
# 【知识点】Plan-and-Execute State — 追踪计划、执行进度、结果
# ============================================================
# 与 ReAct 的核心区别：ReAct 是"走一步看一步"，Plan-and-Execute 是"先想好再动手"
# State 新增字段：
#   plan: 计划步骤列表，由 planner 生成
#   current_step: 当前执行到第几步
#   past_steps: 已执行步骤的结果列表（step_description, result）
#   response: 最终回答（非空时表示任务完成）

class PlanExecuteState(TypedDict):
    messages: Annotated[list[BaseMessage], add]  # 消息历史
    plan: list[str]                                # 计划步骤列表
    current_step: int                              # 当前执行到第几步
    past_steps: list[dict]                         # 已执行步骤 [{step, result}]
    response: str                                  # 最终回答

# ============================================================
# 【知识点】planner_node — 规划节点（Plan-and-Execute 的第一步）
# ============================================================
# planner 的职责：把用户问题拆解为可执行的步骤列表
# 关键：步骤要具体、可执行、有顺序依赖
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

def planner_node(state: PlanExecuteState) -> dict:
    """规划节点：根据用户问题生成执行计划"""
    user_query = state["messages"][0].content  # 第一条消息是用户输入
    response = llm.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"用户问题：{user_query}"),
    ])

    # 解析 LLM 输出的 JSON 步骤列表
    import json
    try:
        # 提取 JSON 数组（LLM 可能输出 ```json ... ``` 包裹）
        content = response.content
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        plan = json.loads(content.strip())
    except (json.JSONDecodeError, IndexError):
        # 解析失败时，用单步计划兜底
        plan = [f"直接回答用户的问题：{user_query}"]

    print(f"📋 计划（{len(plan)} 步）：")
    for i, step in enumerate(plan):
        print(f"   {i+1}. {step}")
    print()

    return {
        "plan": plan,
        "current_step": 0,
        "past_steps": [],
        "response": "",
    }

# ============================================================
# 【知识点】executor_node — 执行节点（逐步执行计划）
# ============================================================
# executor 的职责：执行当前步骤，可以调用工具
# 实现方式：用 LLM + 工具做一个 mini ReAct 循环
EXECUTOR_PROMPT = """你是一个任务执行专家。请执行当前步骤。

已执行的步骤和结果：
{past_steps}

当前要执行的步骤：{current_step}

可用工具：
- calculator: 数学计算
- get_current_time: 获取当前时间
- search_weather: 查询天气
- knowledge_search: 从知识库检索技术文档

请直接执行这个步骤。如果需要调用工具就调用工具，不需要就直接输出结果。
输出格式：先说明执行了什么，再给出结果。"""

def executor_node(state: PlanExecuteState) -> dict:
    """执行节点：执行当前步骤"""
    plan = state["plan"]
    current_idx = state["current_step"]

    if current_idx >= len(plan):
        return {"response": "所有步骤已执行完成。"}

    current_step = plan[current_idx]

    # 构造 past_steps 上下文
    past_steps_text = ""
    if state["past_steps"]:
        for i, ps in enumerate(state["past_steps"]):
            past_steps_text += f"步骤{i+1}: {ps['step']}\n结果: {ps['result']}\n\n"

    prompt = EXECUTOR_PROMPT.format(
        past_steps=past_steps_text or "（这是第一个步骤）",
        current_step=current_step,
    )

    # 用带工具的 LLM 执行（支持工具调用）
    response = llm_with_tools.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"请执行：{current_step}"),
    ])

    # 如果 LLM 要求调用工具，执行工具并获取结果
    result = response.content
    if hasattr(response, "tool_calls") and response.tool_calls:
        # 执行工具调用
        from langchain_core.messages import ToolMessage
        tool_results = []
        for tc in response.tool_calls:
            # 找到对应工具并执行
            for tool in all_tools:
                if tool.name == tc["name"]:
                    try:
                        tool_output = tool.invoke(tc["args"])
                        tool_results.append(f"[{tc['name']}] {tool_output}")
                    except Exception as e:
                        tool_results.append(f"[{tc['name']}] 错误: {e}")
                    break
        result = "\n".join(tool_results)

    print(f"▶️  执行步骤 {current_idx + 1}: {current_step}")
    print(f"   结果: {result[:200]}{'...' if len(result) > 200 else ''}\n")

    # 更新 past_steps 和 current_step
    past_steps = state["past_steps"] + [{"step": current_step, "result": result}]
    return {
        "past_steps": past_steps,
        "current_step": current_idx + 1,
        "messages": [SystemMessage(content=f"步骤 {current_idx + 1} 执行结果：{result}")],
    }

# ============================================================
# 【知识点】replanner_node — 重新规划节点（动态调整计划）
# ============================================================
# re-planner 的职责：根据已执行结果，决定下一步
# 三种走向：继续执行 / 调整计划 / 输出最终回答
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

def replanner_node(state: PlanExecuteState) -> dict:
    """重新规划节点：根据执行结果动态调整计划"""
    import json

    original_query = state["messages"][0].content
    past_steps = state["past_steps"]
    plan = state["plan"]
    current_step = state["current_step"]

    # 构造已执行步骤文本
    past_steps_text = ""
    for i, ps in enumerate(past_steps):
        past_steps_text += f"步骤{i+1}: {ps['step']}\n结果: {ps['result']}\n\n"

    # 构造剩余计划文本
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

    # 解析 LLM 输出
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
        final_response = decision.get("response", "任务完成。")
        print(f"✅ 任务完成\n")
        return {"response": final_response}
    elif action == "replan":
        new_plan = decision.get("new_plan", plan[current_step:])
        print(f"🔄 计划调整（{len(new_plan)} 步）：")
        for i, step in enumerate(new_plan):
            print(f"   {i+1}. {step}")
        print()
        return {"plan": new_plan, "current_step": 0, "past_steps": past_steps}
    else:
        # continue：继续执行下一步
        return {}

# ============================================================
# 【知识点】should_continue_plan — Plan-and-Execute 的条件路由
# ============================================================
def should_continue_plan(state: PlanExecuteState) -> str:
    """判断计划是否执行完成：response 非空 → END，否则继续执行"""
    if state.get("response"):
        return "finish"
    if state.get("current_step", 0) >= len(state.get("plan", [])):
        return "finish"
    return "continue"

# ============================================================
# 【知识点】finish_node — 完成节点，输出最终回答
# ============================================================
def finish_node(state: PlanExecuteState) -> dict:
    """将最终回答添加到消息历史"""
    return {"messages": [AIMessage(content=state["response"])]}

# ============================================================
# 【知识点】StateGraph 构建 — Plan-and-Execute 图
# ============================================================
# 图结构：
#   START → planner → executor → replanner ──继续──→ executor（循环）
#                                  │
#                                  └──完成──→ finish → END
plan_graph = StateGraph(PlanExecuteState)

plan_graph.add_node("planner", planner_node)
plan_graph.add_node("executor", executor_node)
plan_graph.add_node("replanner", replanner_node)
plan_graph.add_node("finish", finish_node)

plan_graph.add_edge(START, "planner")
plan_graph.add_edge("planner", "executor")
plan_graph.add_edge("executor", "replanner")
plan_graph.add_conditional_edges("replanner", should_continue_plan, {
    "continue": "executor",
    "finish": "finish",
})
plan_graph.add_edge("finish", END)

plan_memory = MemorySaver()
plan_app = plan_graph.compile(checkpointer=plan_memory)


# ============================================================
# ============================================================
#  模式三：人机协作（interrupt_before）
# ============================================================
# ============================================================

# ============================================================
# 【知识点】人机协作 — interrupt_before 暂停图执行
# ============================================================
# 核心思想：Agent 要调用工具时，图暂停，等用户确认后才执行
#
# 执行流程：
#   1. 用户提问 → agent 推理 → 决定调用工具
#   2. 图在 tools 节点前暂停（interrupt_before）
#   3. 返回暂停状态：包含待执行的 tool_calls
#   4. 用户确认 → resume 继续执行工具
#   5. 用户取消 → 注入拒绝消息，agent 换策略
#
# 关键 API：
#   graph.compile(interrupt_before=["tools"])  — 编译时声明暂停点
#   app.invoke(None, config)                   — resume 继续执行
#   app.get_state(config)                      — 获取当前暂停状态

# 复用 AgentState、agent_node、should_continue、corrector_node 等已有定义
human_graph = StateGraph(AgentState)

human_graph.add_node("agent", agent_node)
human_graph.add_node("tools", tool_node)
human_graph.add_node("corrector", corrector_node)

human_graph.add_edge(START, "agent")
human_graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
human_graph.add_conditional_edges("tools", check_tool_result, {"corrector": "corrector", "agent": "agent"})
human_graph.add_edge("corrector", "agent")

human_memory = MemorySaver()
# 【知识点】interrupt_before — 在 tools 节点前暂停，等待用户确认
human_app = human_graph.compile(
    checkpointer=human_memory,
    interrupt_before=["tools"],
)


# ============================================================
# ============================================================
#  终端入口
# ============================================================
# ============================================================

def run_react():
    """模式一：ReAct + 自纠错"""
    session_id = "react"
    config = {"configurable": {"thread_id": session_id}}

    print("📊 执行模式: ReAct + 自纠错（最多重试 3 次）\n")

    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            return
        if user_input.lower() == "clear":
            react_memory.delete_thread(session_id)
            print("✅ 对话历史已清空\n")
            continue

        try:
            result = react_app.invoke(
                {"messages": [HumanMessage(content=user_input)], "retry_count": 0, "last_error": ""},
                config=config,
            )
            ai_msg = result["messages"][-1]
            print(f"\nAI: {ai_msg.content}\n")
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


def run_plan_execute():
    """模式二：Plan-and-Execute"""
    session_id = "plan"
    config = {"configurable": {"thread_id": session_id}}

    print("📊 执行模式: Plan-and-Execute（先规划再执行）\n")

    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            return
        if user_input.lower() == "clear":
            plan_memory.delete_thread(session_id)
            print("✅ 对话历史已清空\n")
            continue

        try:
            result = plan_app.invoke(
                {"messages": [HumanMessage(content=user_input)], "plan": [], "current_step": 0, "past_steps": [], "response": ""},
                config=config,
            )
            ai_msg = result["messages"][-1]
            print(f"\nAI: {ai_msg.content}\n")
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


def run_human_loop():
    """模式三：人机协作（interrupt_before）"""
    session_id = "human"
    config = {"configurable": {"thread_id": session_id}}

    print("📊 执行模式: 人机协作（工具调用前暂停等你确认）\n")

    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            return
        if user_input.lower() == "clear":
            human_memory.delete_thread(session_id)
            print("✅ 对话历史已清空\n")
            continue

        try:
            # 第一次运行，可能在 tools 节点前暂停
            result = human_app.invoke(
                {"messages": [HumanMessage(content=user_input)], "retry_count": 0, "last_error": ""},
                config=config,
            )

            # 检查是否暂停在 tools 节点
            snapshot = human_app.get_state(config)
            if snapshot.next and "tools" in snapshot.next:
                # 提取待执行的 tool_calls
                last_msg = snapshot.values["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    print("\n⏸️  Agent 想调用以下工具：")
                    for tc in last_msg.tool_calls:
                        print(f"   🔧 {tc['name']}({tc['args']})")

                    confirm = input("\n确认执行？(y/n): ").strip().lower()
                    if confirm == "y":
                        print("✅ 已确认，继续执行...\n")
                        # resume：继续执行
                        result = human_app.invoke(None, config=config)
                        ai_msg = result["messages"][-1]
                        print(f"\nAI: {ai_msg.content}\n")
                    else:
                        print("❌ 已取消，Agent 将换一种方式回答...\n")
                        # 注入拒绝消息，让 agent 知道工具被拒绝了
                        from langchain_core.messages import ToolMessage
                        rejection_msgs = []
                        for tc in last_msg.tool_calls:
                            rejection_msgs.append(ToolMessage(
                                content="用户拒绝了此工具调用。请直接用你的知识回答用户的问题，不要尝试调用任何工具。",
                                tool_call_id=tc["id"],
                            ))
                        human_app.update_state(config, {"messages": rejection_msgs})
                        # resume：agent 会收到拒绝消息，换策略回答
                        result = human_app.invoke(None, config=config)
                        ai_msg = result["messages"][-1]
                        print(f"\nAI: {ai_msg.content}\n")
                else:
                    # 没有 tool_calls，直接输出
                    ai_msg = result["messages"][-1]
                    print(f"\nAI: {ai_msg.content}\n")
            else:
                # 没有暂停，直接输出
                ai_msg = result["messages"][-1]
                print(f"\nAI: {ai_msg.content}\n")

        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


def main():
    print("🤖 DeepSeek LangGraph Agent（输入 quit 退出，输入 clear 清空历史）")
    print("🔧 可用工具: 数学计算 | 当前时间 | 天气查询 | 📚 知识库检索\n")
    print("选择执行模式：")
    print("  1. ReAct + 自纠错（工具调用循环，失败自动重试）")
    print("  2. Plan-and-Execute（先规划步骤，再逐步执行）")
    print("  3. 人机协作（工具调用前暂停等你确认）\n")

    choice = input("请输入 1、2 或 3: ").strip()
    if choice == "2":
        run_plan_execute()
    elif choice == "3":
        run_human_loop()
    else:
        run_react()


if __name__ == "__main__":
    main()
