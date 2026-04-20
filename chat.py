"""
LangChain + DeepSeek 多轮对话客户端（终端版）

AI Agent 知识点索引：
- Model I/O: ChatModel 初始化、Prompt Template、Message 类型
- LCEL: 管道符编排 (prompt | llm)、流式输出 (stream)
- Memory: RunnableWithMessageHistory、InMemoryChatMessageHistory
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# ============================================================
# 【知识点】Model I/O — ChatModel 初始化
# ============================================================
# ChatOpenAI 是 LangChain 对 OpenAI 兼容接口的封装
# DeepSeek 兼容 OpenAI 协议，只需改 base_url 和 model 即可接入
# 关键参数：
#   - model: 模型名称，决定能力和成本
#   - temperature: 0=确定性输出，1=更随机，0.7 是常用平衡值
#   - base_url: API 端点，切换不同模型提供商的关键
#   - api_key: 从环境变量读取，不硬编码（安全最佳实践）
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7,
)

# ============================================================
# 【知识点】Model I/O — Prompt Template
# ============================================================
# ChatPromptTemplate 定义对话结构，由多个 Message 组成：
#   - ("system", "..."):  SystemMessage，设定 AI 角色和行为约束
#   - ("human", "{input}"): HumanMessage，用户输入，{input} 是变量占位符
#   - MessagesPlaceholder:  动态插入位，运行时被替换为历史消息列表
#
# MessagesPlaceholder("history") 是实现多轮对话的关键：
#   它会在运行时被替换为 [HumanMessage, AIMessage, HumanMessage, ...] 的列表
#   让模型能看到之前的对话上下文
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的 AI 助手，回答简洁准确。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# ============================================================
# 【知识点】LCEL — 管道符编排
# ============================================================
# LCEL (LangChain Expression Language) 用 | 管道符将组件串联成 Chain
# prompt | llm 等价于：先用 prompt 格式化输入，再传给 llm 生成回复
# 数据流：{"input": "你好"} → prompt.invoke() → [Messages] → llm.invoke() → AIMessage
#
# 这是最简单的 Chain，后续可以扩展为：
#   prompt | llm | output_parser          （加输出解析）
#   prompt | llm | StrOutputParser()      （提取纯文本）
#   retriever | prompt | llm              （RAG 检索增强）
chain = prompt | llm

# ============================================================
# 【知识点】Memory — 会话记忆存储
# ============================================================
# InMemoryChatMessageHistory: 在内存中存储消息列表
# store 字典按 session_id 隔离不同会话的历史
# 局限：进程重启后记忆丢失
# 生产环境可替换为：
#   - RedisChatMessageHistory（Redis 持久化）
#   - SQLChatMessageHistory（数据库持久化）
#   - FileChatMessageHistory（文件持久化）
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# ============================================================
# 【知识点】Memory — RunnableWithMessageHistory
# ============================================================
# 包装原始 Chain，自动完成：
#   1. 调用前：从 get_session_history 加载历史消息，注入到 "history" 占位符
#   2. 调用后：将本轮的 HumanMessage + AIMessage 追加到历史
# 参数说明：
#   - input_messages_key="input": 告诉它用户输入在哪个字段
#   - history_messages_key="history": 告诉它历史消息注入到哪个占位符
# 使用时通过 config 传入 session_id 区分不同会话
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def main():
    session_id = "default"
    # config 通过 configurable 传递运行时参数（如 session_id）
    # 这是 LCEL 的 RunnableConfig 机制
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
            # ============================================================
            # 【知识点】LCEL — 流式输出 (stream)
            # ============================================================
            # stream() 返回一个生成器，逐 chunk 产出 AIMessageChunk
            # 每个 chunk.content 是一小段文本（通常几个字）
            # 优势：用户不用等全部生成完，首字延迟大幅降低
            # 对比：
            #   invoke()  → 等全部生成完，返回完整 AIMessage
            #   stream()  → 逐块返回 AIMessageChunk（同步）
            #   astream() → 逐块返回（异步，用于 FastAPI 等异步框架）
            #   batch()   → 批量调用，提高吞吐量
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
