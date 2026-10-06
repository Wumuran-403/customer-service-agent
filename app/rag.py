"""RAG：Milvus 客户端、知识库初始化与检索。"""
import os
import re
from pymilvus import MilvusClient
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    MILVUS_URI,
    DB_NAME,
    COLLECTION_NAME,
    EMBED_DIM,
    KNOWLEDGE_FILE,
)
from app.llm import get_embed_model

_client = None


def get_client() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(uri=MILVUS_URI)
        if DB_NAME not in _client.list_databases():
            _client.create_database(DB_NAME)
        _client.use_database(DB_NAME)
    return _client


def _split_into_sections(text: str):
    """按 ============================== 分隔线切出 (标题, 正文) 列表。

    文档结构：
        ==============================
        一、产品简介
        ==============================

        正文...
    """
    parts = re.split(r"={20,}", text)
    sections = []

    # parts[0] 是文档开头前言
    if parts[0].strip():
        sections.append(("文档说明", parts[0].strip()))

    # 之后交替出现: 标题行 / 正文行
    i = 1
    while i < len(parts) - 1:
        title = parts[i].strip()
        body = parts[i + 1].strip()
        if title and body:
            sections.append((title, body))
        i += 2
    return sections


def _build_chunks():
    """结构化切分：每个 chunk 带【章节：xxx】前缀，过滤过短碎片。"""
    with open(KNOWLEDGE_FILE, encoding="utf-8") as f:
        text = f.read()

    sections = _split_into_sections(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )

    chunks = []
    for title, body in sections:
        # 关键：把章节标题作为前缀拼进每个 chunk，
        # 这样向量里既包含正文，也包含"这是哪一章"的上下文
        enriched = f"【章节：{title}】\n{body}"
        for piece in splitter.split_text(enriched):
            piece = piece.strip()
            if len(piece) >= 20:  # 过滤纯标题/碎片
                chunks.append(piece)
    return chunks


def init_knowledge_base():
    """服务启动时调用。

    通过环境变量 REBUILD_VECTORDB=1 可强制重建 collection（改了切分策略后用）。
    """
    client = get_client()

    rebuild = os.getenv("REBUILD_VECTORDB", "0") == "1"
    if rebuild and client.has_collection(collection_name=COLLECTION_NAME):
        client.drop_collection(collection_name=COLLECTION_NAME)

    if not client.has_collection(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=EMBED_DIM,
            metric_type="COSINE",
        )

    stats = client.get_collection_stats(collection_name=COLLECTION_NAME)
    if stats.get("row_count", 0) > 0 and not rebuild:
        return

    chunks = _build_chunks()
    print(f"[rag] 切分出 {len(chunks)} 个 chunk，开始向量化入库...")

    embed = get_embed_model()
    vectors = embed.embed_documents(chunks)
    data = [
        {
            "id": i,
            "vector": vectors[i],
            "text": chunks[i],
            "source": "knowledge.txt",
            "chunk_id": i,
        }
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=COLLECTION_NAME, data=data)
    client.flush(collection_name=COLLECTION_NAME)
    print(f"[rag] 入库完成，共 {len(data)} 条")


def retrieve(question: str, k: int = 6):
    """向量检索，返回 top-k 命中。"""
    embed = get_embed_model()
    client = get_client()
    query_vec = embed.embed_query(question)
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        limit=k,
        output_fields=["text", "source", "chunk_id"],
    )
    return results[0]
