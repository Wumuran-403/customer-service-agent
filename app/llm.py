"""LLM 与 Embedding 单例。"""
from langchain.embeddings import init_embeddings
from langchain.chat_models import init_chat_model

from app.config import (
    DEEPSEEK_API_URL,
    DEEPSEEK_API_KEY,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
)

_embed = None
_chat = None


def get_embed_model():
    global _embed
    if _embed is None:
        _embed = init_embeddings(
            provider="openai",
            model="BAAI/bge-m3",
            api_key=SILICONFLOW_API_KEY,
            base_url=SILICONFLOW_BASE_URL,
        )
    return _embed


def get_chat_model():
    global _chat
    if _chat is None:
        _chat = init_chat_model(
            model="deepseek-v4-flash",
            model_provider="deepseek",
            profile={"max_input_tokens": 128_000},
            base_url=DEEPSEEK_API_URL,
            api_key=DEEPSEEK_API_KEY,
            extra_body={"thinking": {"type": "disabled"}},
        )
    return _chat
