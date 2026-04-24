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
import json
from datetime import datetime
from pathlib import Path
import jieba
from rank_bm25 import BM25Okapi
from langchain_core.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ============================================================
# 【知识点】RAG Tool — 加载 FAISS 索引 + BM25 语料，实现混合检索
# ============================================================
# 混合检索 = 向量检索（语义相似）+ BM25（关键词匹配）
# RRF (Reciprocal Rank Fusion) 融合两路结果的排名
# 两路检索各取 k=5，RRF 融合后取 top 3
INDEX_DIR = Path(__file__).parent / "faiss_index"
_retriever = None
_bm25 = None
_bm25_docs = None

def _get_retriever():
    """懒加载 FAISS Retriever，索引不存在时返回 None"""
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
    _retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return _retriever

def _get_bm25():
    """懒加载 BM25 索引（从构建时保存的 bm25_corpus.json 加载）"""
    global _bm25, _bm25_docs
    if _bm25 is not None:
        return _bm25, _bm25_docs
    bm25_path = INDEX_DIR / "bm25_corpus.json"
    if not bm25_path.exists():
        return None, None
    with open(bm25_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    tokenized_corpus = [doc["tokens"] for doc in corpus]
    _bm25 = BM25Okapi(tokenized_corpus)
    _bm25_docs = [(doc["content"], doc["source"]) for doc in corpus]
    return _bm25, _bm25_docs

def _rrf_merge(vector_docs, bm25_results, k=60, top_n=3):
    """RRF (Reciprocal Rank Fusion) 融合两路检索结果

    【知识点】RRF 排序公式：score = 1/(k + rank)
    两路检索各自的排名通过 RRF 融合为统一分数，k=60 是常用常数
    """
    scores = {}
    # 向量检索结果计分
    for rank, doc in enumerate(vector_docs):
        content = doc.page_content
        scores[content] = scores.get(content, 0) + 1.0 / (k + rank + 1)
    # BM25 检索结果计分
    for rank, (content, source) in enumerate(bm25_results):
        scores[content] = scores.get(content, 0) + 1.0 / (k + rank + 1)
    # 按 RRF 分数排序，取 top_n
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [content for content, _ in sorted_items]

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
    """从知识库中检索相关文档（混合检索：向量 + BM25 + RRF 融合排序）。

    当用户询问 LangChain、Agent、RAG、LCEL、Android、架构设计等技术知识时使用。
    知识库包含 AI Agent 学习笔记和 Android 开发知识。
    输入为搜索关键词或问题。
    """
    # 向量检索（语义相似）
    retriever = _get_retriever()
    if retriever is None:
        return "知识库索引未构建，请先运行 python3 build_index.py"
    vector_docs = retriever.invoke(query)

    # BM25 检索（关键词匹配）
    bm25, bm25_docs = _get_bm25()
    bm25_results = []
    if bm25 is not None:
        tokens = list(jieba.cut(query))
        scores = bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        bm25_results = [bm25_docs[i] for i in top_indices if scores[i] > 0]

    # RRF 融合排序
    merged = _rrf_merge(vector_docs, bm25_results, k=60, top_n=3)
    if not merged:
        return f"未找到与「{query}」相关的知识"

    return "\n\n---\n\n".join(merged)

# ============================================================
# 工具列表 — 供 chat.py 和 server.py 导入
# ============================================================
all_tools = [calculator, get_current_time, search_weather, knowledge_search]
