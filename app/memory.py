"""长期记忆:Postgres 持久化对话历史与用户事实。

两张表：
- chat_history: 原始多轮对话，按 user_id 存取
- user_facts: 从对话中抽取的关于该用户的长期事实（如套餐版本、关注点）
"""
import psycopg2
from psycopg2.extras import RealDictCursor

from app.config import POSTGRES_URL
from app.llm import get_chat_model

_conn = None


def get_conn():
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(POSTGRES_URL)
        _conn.autocommit = True
    return _conn


def init_db():
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(64),
                role VARCHAR(16),
                content TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_facts (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(64),
                fact TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(user_id, fact)
            )
            """
        )


def load_history(user_id: str, limit: int = 10):
    conn = get_conn()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT role, content FROM chat_history WHERE user_id=%s "
            "ORDER BY id DESC LIMIT %s",
            (user_id, limit),
        )
        rows = cur.fetchall()
    return list(reversed(rows))


def save_turn(user_id: str, role: str, content: str):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat_history (user_id, role, content) VALUES (%s, %s, %s)",
            (user_id, role, content),
        )


def load_facts(user_id: str):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT fact FROM user_facts WHERE user_id=%s ORDER BY id DESC",
            (user_id,),
        )
        return [r[0] for r in cur.fetchall()]


def save_fact(user_id: str, fact: str):
    fact = (fact or "").strip().strip("。")
    if not fact or fact.upper() == "NONE":
        return
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO user_facts (user_id, fact) VALUES (%s, %s) "
            "ON CONFLICT (user_id, fact) DO NOTHING",
            (user_id, fact),
        )


def maybe_extract_fact(user_id: str, question: str, answer: str):
    """轻量事实抽取：让 LLM 判断本轮对话是否产生了值得长期记住的用户画像。"""
    llm = get_chat_model()
    resp = llm.invoke([
        {"role": "system",
         "content": "你是用户画像提取器。根据用户与客服的这轮对话，判断是否有需要长期记住的关于该用户的事实"
                    "（例如：套餐版本、关注的问题、身份角色、历史诉求）。"
                    "只输出一句话事实本身，不要寒暄、不要解释；没有就输出 NONE。"},
        {"role": "user",
         "content": f"用户问：{question}\n客服答:{answer}\n\n长期事实一句话:"},
    ])
    text = resp.content if hasattr(resp, "content") else str(resp)
    save_fact(user_id, text)
