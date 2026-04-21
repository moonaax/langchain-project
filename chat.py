"""
LangChain + DeepSeek 工具调用 Agent（终端版）

第二阶段：在多轮对话基础上添加 Tool Calling 能力

AI Agent 知识点索引：
- Tool:  @tool 装饰器定义工具、工具描述的重要性
- Agent: create_tool_calling_agent（基于 Function Calling 的现代方式）
- AgentExecutor: 推理循环引擎（Thought → Action → Observation）
- Memory: 在 Agent 中保持多轮对话记忆
"""

import os
import math
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

# ============================================================
# 【知识点】Model I/O — ChatModel 初始化（同第一阶段）
# ============================================================
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7,
)

# ============================================================
# 【知识点】Tool — @tool 装饰器定义工具
# ============================================================
# @tool 是最简洁的工具定义方式，自动从函数签名和 docstring 提取：
#   - name: 函数名（LLM 通过名称引用工具）
#   - description: docstring（LLM 根据描述决定何时使用）
#   - args_schema: 从类型注解自动生成 Pydantic Schema
#
# 工具描述的质量直接决定 Agent 表现！好的描述应包含：
#   1. 用途：这个工具做什么
#   2. 时机：什么情况下应该使用
#   3. 输入：需要什么参数
#   4. 限制：不能做什么（可选）

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
    """查询指定城市的天气信息。输入城市名称，返回当前天气。

    注意：这是模拟数据，实际项目中应接入真实天气 API。
    """
    weather_data = {
        "北京": "晴天，25°C，湿度 40%",
        "上海": "多云，22°C，湿度 65%",
        "广州": "小雨，28°C，湿度 80%",
        "深圳": "阴天，27°C，湿度 75%",
        "杭州": "晴天，23°C，湿度 55%",
    }
    return weather_data.get(city, f"未找到 {city} 的天气信息，支持的城市：{', '.join(weather_data.keys())}")

# 工具列表 — Agent 可用的所有工具
# 控制在 3-8 个，太多会降低 LLM 选择准确率
tools = [calculator, get_current_time, search_weather]

# ============================================================
# 【知识点】Agent — Prompt Template（与普通 Chain 的区别）
# ============================================================
# Agent 的 Prompt 必须包含 agent_scratchpad 占位符！
# agent_scratchpad 用于存放 Agent 的中间推理过程：
#   - 每次工具调用的 Action + Observation 都会追加到这里
#   - LLM 看到之前的推理过程，决定下一步行动或给出最终答案
#
# 对比第一阶段的 Prompt：
#   第一阶段: system + history + input（简单 Chain）
#   第二阶段: system + history + input + agent_scratchpad（Agent）
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
    # Agent 推理过程的暂存区（关键！）
    # create_tool_calling_agent 会自动往这里填充中间步骤
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ============================================================
# 【知识点】Agent — create_tool_calling_agent
# ============================================================
# 基于 LLM 原生 Function Calling 能力构建 Agent
# 这是推荐的现代方式（对比 create_react_agent 的文本解析方式）
#
# Function Calling 的优势：
#   1. LLM 原生支持，参数解析更准确（不依赖正则匹配）
#   2. 支持并行工具调用（一次调用多个工具）
#   3. 不需要复杂的 Prompt 工程
#
# 返回的 agent 是一个 Runnable，但不能直接运行
# 必须用 AgentExecutor 包装才能执行推理循环
agent = create_tool_calling_agent(llm, tools, prompt)

# ============================================================
# 【知识点】Agent — AgentExecutor 推理循环引擎
# ============================================================
# AgentExecutor 是 Agent 的运行时，负责执行 "思考→行动→观察" 循环：
#
#   while True:
#       action = agent.plan(input, intermediate_steps)
#       if action == AgentFinish:  → 返回最终答案
#       observation = tool.run(action)
#       intermediate_steps.append((action, observation))
#       if iterations > max_iterations:  → 强制停止
#
# 关键参数：
#   - verbose=True: 打印推理过程（调试必备）
#   - max_iterations=10: 防止无限循环（默认 15）
#   - handle_parsing_errors=True: 自动处理 LLM 输出格式错误
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,           # 打印每一步推理过程
    max_iterations=10,      # 最大循环次数，防止无限循环
    handle_parsing_errors=True,  # 自动处理解析错误
)

# ============================================================
# 【知识点】Memory — 在 Agent 中保持多轮对话记忆
# ============================================================
# 与第一阶段相同的记忆机制，但包装的对象从 chain 变成了 agent_executor
# RunnableWithMessageHistory 会自动：
#   1. 调用前：加载历史消息到 "history" 占位符
#   2. 调用后：将本轮对话追加到历史
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def main():
    session_id = "default"
    config = {"configurable": {"session_id": session_id}}

    print("🤖 DeepSeek Agent（输入 quit 退出，输入 clear 清空历史）")
    print("🔧 可用工具: 数学计算 | 当前时间 | 天气查询\n")

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
            # ============================================================
            # 【知识点】Agent — invoke 调用
            # ============================================================
            # Agent 不适合用 stream() 逐 token 输出，因为推理过程包含多轮
            # 工具调用。invoke() 会等待整个推理循环完成后返回最终答案。
            #
            # 返回值是 dict: {"input": "...", "output": "最终答案", "history": [...]}
            # 如果设置了 return_intermediate_steps=True，还会包含中间步骤
            result = agent_with_memory.invoke(
                {"input": user_input}, config=config
            )
            print(f"\nAI: {result['output']}\n")
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")

if __name__ == "__main__":
    main()
