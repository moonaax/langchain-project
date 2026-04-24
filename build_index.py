"""
知识库索引构建脚本

离线阶段：加载 Markdown 文档 → 文本分块 → Embedding 向量化 → FAISS 持久化 + BM25 数据

AI Agent 知识点索引：
- Document Loader: DirectoryLoader + TextLoader 批量加载 Markdown
- Text Splitter: RecursiveCharacterTextSplitter 中文优化分块
- Embedding: HuggingFaceEmbeddings 本地向量化（无需 API Key）
- VectorStore: FAISS 高性能向量存储与持久化
- BM25: jieba 分词 + rank_bm25，与向量检索混合使用（RRF 融合排序）

用法：
    python3 build_index.py                          # 使用默认知识库路径
    python3 build_index.py /path/to/your/docs       # 指定文档目录
"""

import sys
import json
from pathlib import Path
import jieba
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ============================================================
# 【知识点】配置参数
# ============================================================
# DOCS_DIR: 知识库文档目录
# INDEX_DIR: FAISS 索引持久化目录
# EMBEDDING_MODEL: 本地 Embedding 模型
#   - bge-base-zh-v1.5: 中文效果优秀，768 维，约 400MB
#   - 首次运行会自动从 HuggingFace 下载模型
DOCS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "lib/my-knowledge-lib"
INDEX_DIR = Path(__file__).parent / "faiss_index"
EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"

def build():
    # ============================================================
    # 【知识点】Document Loader — 批量加载 Markdown 文件
    # ============================================================
    # DirectoryLoader 递归扫描目录，按 glob 模式匹配文件
    # TextLoader 以纯文本方式加载（保留 Markdown 原始格式）
    # 每个文件生成一个 Document 对象，metadata 自动包含 source 路径
    print(f"📂 加载文档: {DOCS_DIR}")
    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"   加载了 {len(docs)} 个文件")

    # ============================================================
    # 【知识点】Text Splitter — 中文优化的递归分块
    # ============================================================
    # RecursiveCharacterTextSplitter 按分隔符优先级递归分割：
    #   1. 先尝试 \n\n（段落边界）
    #   2. 再尝试 \n（行边界）
    #   3. 再尝试中文句号等标点
    #   4. 最后按字符强制切割
    #
    # chunk_size=500: 中文信息密度高，500 字符约 250-300 个汉字
    # chunk_overlap=50: 10% 重叠，保证跨块语义连续性
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️  分块后共 {len(chunks)} 个 chunk")

    # ============================================================
    # 【知识点】Embedding — 本地 HuggingFace 模型向量化
    # ============================================================
    # 使用 BGE 中文模型，无需 API Key，完全本地运行
    # normalize_embeddings=True: 归一化向量，使余弦相似度等价于内积
    # 首次运行会下载模型（约 400MB），后续使用本地缓存
    print(f"🧠 加载 Embedding 模型: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ============================================================
    # 【知识点】VectorStore — FAISS 构建与持久化
    # ============================================================
    # FAISS.from_documents: 一步完成向量化 + 索引构建
    # save_local: 将索引序列化到磁盘（index.faiss + index.pkl）
    # 后续加载用 FAISS.load_local() 即可，无需重新构建
    print(f"📦 构建 FAISS 索引...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"✅ FAISS 索引已保存到 {INDEX_DIR}（{len(chunks)} 个向量）")

    # ============================================================
    # 【知识点】BM25 语料 — jieba 分词后保存，供混合检索使用
    # ============================================================
    # 混合检索 = 向量检索（语义相似）+ BM25（关键词匹配）
    # RRF (Reciprocal Rank Fusion) 融合两路结果的排名
    # 需要在构建索引时保存分词后的语料，运行时加载即可
    print(f"🔤 构建 BM25 语料（jieba 分词）...")
    bm25_corpus = []
    for chunk in chunks:
        tokens = list(jieba.cut(chunk.page_content))
        bm25_corpus.append({
            "tokens": tokens,
            "content": chunk.page_content,
            "source": chunk.metadata.get("source", ""),
        })
    bm25_path = INDEX_DIR / "bm25_corpus.json"
    with open(bm25_path, "w", encoding="utf-8") as f:
        json.dump(bm25_corpus, f, ensure_ascii=False)
    print(f"✅ BM25 语料已保存到 {bm25_path}（{len(bm25_corpus)} 条）")

if __name__ == "__main__":
    build()
