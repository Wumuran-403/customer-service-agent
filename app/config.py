"""全局配置：所有敏感值从 .env 读取，不硬编码。"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ---- LLM / Embedding ----
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

# ---- Milvus ----
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
DB_NAME = os.getenv("DB_NAME", "rag_tutorial")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docs")
EMBED_DIM = 1024  # BGE-M3 固定 1024 维

# ---- Postgres (长期记忆) ----
POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://postgres:postgres@localhost:5432/memory",
)

# ---- 知识库文件 ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "data", "knowledge.txt")
