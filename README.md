# Atguigu Assistant · 客服知识库 Agent

基于 **LangChain + Milvus + PostgreSQL + DeepSeek** 的企业客服问答 Agent，支持 RAG 知识库检索与跨会话长期记忆。

## 能力

- **RAG 知识库问答**：`data/knowledge.txt`（43 个语义切片）经 BGE-M3 向量化后存入 Milvus，回答时余弦召回 Top-5 作为上下文。
- **长期记忆（Postgres）**：
  - `chat_history` 表持久化每一轮 user/assistant 消息，下次对话自动带上最近 10 轮上下文；
  - `user_facts` 表由 LLM 每轮结束后抽取用户画像事实（套餐版本、关注点等），下次对话自动注入 system prompt。
- **REST 接口**：FastAPI 自动生成 Swagger UI，启动后访问 `http://localhost:8000/docs` 可直接在网页上调试。

## 架构

```
用户请求
   │
   ▼
FastAPI (/chat)
   │
   ├──► PostgreSQL ── 加载最近 10 轮历史 + 用户已沉淀事实
   │
   ├──► Milvus ── BGE-M3 把问题向量化，余弦召回 Top-5 知识切片
   │
   ▼
DeepSeek V4 Flash ── 拼接 system + facts + history + retrieved context 生成回答
   │
   ▼
写回 PostgreSQL（本轮对话 + LLM 抽取新事实）
```

## 技术栈

| 层 | 选型 |
|---|---|
| LLM | DeepSeek V4 Flash |
| Embedding | BAAI/bge-m3（via SiliconFlow，1024 维） |
| 向量库 | Milvus 2.4 standalone |
| 长期记忆 | PostgreSQL 15 |
| API 框架 | FastAPI + Uvicorn |
| 容器 | Docker Compose |

## Quick Start

### 方式一：Docker Compose 一键启动（推荐）

```bash
cp .env.example .env
# 编辑 .env，填入你自己的 DeepSeek 和 SiliconFlow API Key

docker compose up --build
```

启动后访问：
- Swagger UI：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

### 方式二：本地开发

前置依赖：本地已跑着 Milvus（`localhost:19530`）和 Postgres（`localhost:5432`）。

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
cp .env.example .env          # 填入 API Key

uvicorn app.main:app --reload --port 8000
```

首次启动时，应用会自动检测 Milvus 集合是否为空，为空则自动导入 `data/knowledge.txt` 完成切分、向量化与入库。

## API 示例

### POST /chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u_001", "message": "基础版支持多少个外部协作者？"}'
```

返回：

```json
{
  "reply": "基础版最多允许 20 个外部协作者，且不占用正式成员名额。",
  "retrieved": [
    {"chunk_id": 19, "score": 0.78, "text": "外部协作者不占用正式成员名额..."}
  ],
  "user_facts": []
}
```

### GET /memory/{user_id}

查看某用户的全部对话历史与已沉淀的长期事实：

```bash
curl http://localhost:8000/memory/u_001
```

## 演示跨会话记忆

按这个顺序在 Swagger UI 里点几次 `/chat`：

1. `user_id=alice`，问："我买的是专业版，能开多少个知识库？"
2. `user_id=alice`，再问："那我这个版本发票怎么开？"
   - 第二次回答会自动带上"alice 是专业版用户"这个事实，不需要你重复说。
3. `GET /memory/alice` 可以看到完整对话历史和抽取出来的事实列表。

## 项目结构

```
.
├── app/
│   ├── main.py        # FastAPI 入口与 /chat、/memory 接口
│   ├── config.py      # 环境变量读取
│   ├── llm.py         # DeepSeek + BGE-M3 单例
│   ├── rag.py         # Milvus 客户端、知识库初始化与检索
│   ├── memory.py      # Postgres 长期记忆（历史 + 用户事实）
│   └── schemas.py     # Pydantic 请求/响应模型
├── data/
│   └── knowledge.txt  # 客服知识库原始文档
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## License

MIT
