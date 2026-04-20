"""LangChain + DeepSeek 多轮对话客户端"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# 初始化 DeepSeek 模型（兼容 OpenAI 接口）
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7,
)

# Prompt 模板：系统提示 + 历史消息 + 用户输入
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的 AI 助手，回答简洁准确。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 构建 Chain：prompt → llm
chain = prompt | llm

# 会话记忆存储
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 带记忆的 Chain
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def main():
    session_id = "default"
    config = {"configurable": {"session_id": session_id}}

    print("🤖 DeepSeek 多轮对话客户端（输入 quit 退出，输入 clear 清空历史）\n")

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
            # 流式输出
            print("AI: ", end="", flush=True)
            for chunk in chain_with_memory.stream(
                {"input": user_input}, config=config
            ):
                print(chunk.content, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")

if __name__ == "__main__":
    main()
