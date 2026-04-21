"""
工具定义模块 — 集中管理 Agent 可用的所有工具

AI Agent 知识点索引：
- Tool: @tool 装饰器、工具描述设计
- RAG Tool: 将 FAISS 检索封装为 Agent 工具
- 工具模块化: 抽取工具定义，chat.py 和 server.py 共享

第二阶段工具: calculator, get_current_time, search_weather
第三阶段新增: knowledge_search（知识库 RAG 检索）
"""

import math
from datetime import datetime
from pathlib import Path
from langchain_core.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ============================================================
# 【知识点】RAG Tool — 加载 FAISS 索引供检索使用
# ============================================================
# 在模块加载时初始化 Retriever（只加载一次）
# FAISS.load_local 从磁盘反序列化索引，需要传入相同的 Embedding 模型
# allow_dangerous_deserialization=True: FAISS 使用 pickle，需要显式允许
INDEX_DIR = Path(__file__).parent / "faiss_index"
_retriever = None

def _get_retriever():
    """懒加载 Retriever，索引不存在时返回 None"""
    global _retriever
    if _retriever is not None:
        return _retriever
    if not INDEX_DIR.exists():
        return None
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        str(INDEX_DIR), embeddings,
        allow_dangerous_deserialization=True,
    )
    _retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return _retriever

# ============================================================
# 第二阶段工具（保持不变）
# ============================================================

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

# ============================================================
# 【知识点】RAG Tool — 将向量检索封装为 Agent 工具
# ============================================================
# 这是 RAG 与 Agent 结合的关键：
#   - 传统 RAG: 每次查询都走检索（retriever | prompt | llm）
#   - Agent RAG: 由 Agent 自主决定是否需要检索知识库
#
# Agent 会根据工具描述判断：
#   - 用户问 "LCEL 是什么" → 调用 knowledge_search
#   - 用户问 "今天几号" → 调用 get_current_time（不走 RAG）

@tool
def knowledge_search(query: str) -> str:
    """从知识库中检索相关文档。

    当用户询问 LangChain、Agent、RAG、LCEL、Android、架构设计等技术知识时使用。
    知识库包含 AI Agent 学习笔记和 Android 开发知识。
    输入为搜索关键词或问题。
    """
    retriever = _get_retriever()
    if retriever is None:
        return "知识库索引未构建，请先运行 python3 build_index.py"
    docs = retriever.invoke(query)
    if not docs:
        return f"未找到与「{query}」相关的知识"
    # 格式化检索结果：拼接内容 + 标注来源
    results = []
    for i, doc in enumerate(docs):
        source = Path(doc.metadata.get("source", "未知")).name
        results.append(f"[来源: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(results)

# ============================================================
# 工具列表 — 供 chat.py 和 server.py 导入
# ============================================================
all_tools = [calculator, get_current_time, search_weather, knowledge_search]
