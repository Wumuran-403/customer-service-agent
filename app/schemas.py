from pydantic import BaseModel


class ChatReq(BaseModel):
    user_id: str
    message: str


class RetrievedHit(BaseModel):
    chunk_id: int
    score: float
    text: str


class ChatResp(BaseModel):
    reply: str
    retrieved: list[RetrievedHit] = []
    user_facts: list[str] = []
