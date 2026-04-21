"""
LangChain + DeepSeek 工具调用 Agent（终端版）

第三阶段：在 Agent 基础上添加知识库 RAG 检索工具

AI Agent 知识点索引：
- Tool:  工具模块化，从 tools.py 统一导入
- Agent: create_tool_calling_agent + AgentExecutor
- RAG:   knowledge_search 工具，Agent 自主决定是否检索知识库
- Memory: RunnableWithMessageHistory 多轮对话记忆
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor

# ============================================================
# 【知识点】工具模块化 — 从 tools.py 统一导入
# ============================================================
# 第二阶段工具定义内联在 chat.py 和 server.py 中（重复代码）
# 第三阶段抽取到 tools.py，两端共享同一份工具定义
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

def main():
    session_id = "default"
    config = {"configurable": {"session_id": session_id}}

    print("🤖 DeepSeek Agent（输入 quit 退出，输入 clear 清空历史）")
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

if __name__ == "__main__":
    main()
