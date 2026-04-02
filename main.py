import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

# 项目根目录路径。
BASE_DIR = Path(__file__).resolve().parent


def load_dotenv_file(dotenv_path: Path) -> None:
    """加载 .env 文件到环境变量（不覆盖已有变量）。"""
    if not dotenv_path.exists():
        return

    text = dotenv_path.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        # 跳过空行和注释行。
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue

        os.environ.setdefault(key, value)


# 启动时尝试读取同目录下 .env。
load_dotenv_file(BASE_DIR / ".env")

# FastAPI 应用实例。
app = FastAPI()

# 允许前端跨域访问后端接口。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI 兼容客户端（当前接入 DeepSeek）。
# 从环境变量读取密钥，避免把密钥写死在代码里。
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("缺少 API Key，请设置环境变量 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)

# SQLite 数据库文件路径（放在后端目录下）。
DB_PATH = BASE_DIR / "chat_history.db"

# Token 预算配置（可通过环境变量覆盖）。
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "8192"))
RESERVED_OUTPUT_TOKENS = int(os.getenv("RESERVED_OUTPUT_TOKENS", "2048"))
TOKEN_CHARS_ESTIMATE = int(os.getenv("TOKEN_CHARS_ESTIMATE", "2"))
MESSAGE_OVERHEAD_TOKENS = int(os.getenv("MESSAGE_OVERHEAD_TOKENS", "6"))
SYSTEM_PROMPT_PRESETS = {
    "general": "你是一个通用 AI 助手。请用中文回答，表达清晰、简洁，优先给出可执行建议。",
    "coding": "你是一名资深编程助手。请用中文回答，先给可运行方案，再补充关键原理与边界条件，代码尽量简洁。",
    "interview": "你是一名面试助手。请用中文回答，按“思路-要点-示例”结构给出答案，突出重点并提供可复述的表达。",
    "translation": "你是一名翻译助手。请准确翻译并保留原意与语气；如有歧义，给出更自然的候选译法。",
}


class ChatRequest(BaseModel):
    """/chat/stream 接口请求体。"""

    conversation_id: int | None = None
    content: str | None = None
    messages: list[dict[str, Any]] | None = None
    system_prompt: str | None = None
    system_prompt_preset: str | None = None


def get_conn() -> sqlite3.Connection:
    """创建数据库连接，并支持按字段名访问行数据。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def estimate_text_tokens(text: str) -> int:
    """按字符数粗略估算 token 数。"""
    normalized = text or ""
    chars_per_token = max(1, TOKEN_CHARS_ESTIMATE)
    return max(1, (len(normalized) + chars_per_token - 1) // chars_per_token)


def truncate_text_to_token_budget(text: str, token_budget: int) -> str:
    """把文本截断到给定 token 预算（保留尾部）。"""
    if token_budget <= 0:
        return ""
    chars_per_token = max(1, TOKEN_CHARS_ESTIMATE)
    max_chars = token_budget * chars_per_token
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def build_context_messages(
    history: list[dict[str, Any]], system_prompt: str = ""
) -> tuple[list[dict[str, str]], int]:
    """按 token 预算裁剪历史消息，并预留输出 token。"""
    max_context = max(256, MAX_CONTEXT_TOKENS)
    reserved_output = max(1, min(RESERVED_OUTPUT_TOKENS, max_context - 1))
    input_budget = max(1, max_context - reserved_output)

    selected_reversed: list[dict[str, str]] = []
    per_message_overhead = max(1, MESSAGE_OVERHEAD_TOKENS)
    normalized_system_prompt = (system_prompt or "").strip()
    used_input_tokens = 0

    if normalized_system_prompt:
        system_cost = per_message_overhead + estimate_text_tokens(normalized_system_prompt)
        if system_cost <= input_budget:
            used_input_tokens += system_cost
        else:
            allowed = max(1, input_budget - per_message_overhead)
            normalized_system_prompt = truncate_text_to_token_budget(normalized_system_prompt, allowed)
            used_input_tokens += per_message_overhead + estimate_text_tokens(normalized_system_prompt)

    # 从最近消息往前取，优先保留最新上下文。
    for item in reversed(history):
        role = str(item.get("role", "user"))
        content = str(item.get("content", "") or "")
        message_cost = per_message_overhead + estimate_text_tokens(content)

        if used_input_tokens + message_cost <= input_budget:
            selected_reversed.append({"role": role, "content": content})
            used_input_tokens += message_cost
            continue

        # 即便单条消息超预算，也保留尾部，避免空上下文。
        if not selected_reversed:
            allowed = max(1, input_budget - per_message_overhead)
            truncated = truncate_text_to_token_budget(content, allowed)
            selected_reversed.append({"role": role, "content": truncated})
        break

    selected_messages = list(reversed(selected_reversed))
    if normalized_system_prompt:
        selected_messages.insert(0, {"role": "system", "content": normalized_system_prompt})

    return selected_messages, reserved_output


def init_db() -> None:
    """初始化数据库表，并执行轻量迁移。"""
    with get_conn() as conn:
        # 会话表。
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # 消息表。
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
            """
        )

def clear_db() -> None:
    """清空会话与消息数据，并重置自增序列。"""
    with get_conn() as conn:
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM conversations")
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('messages', 'conversations')")


def create_conversation(title: str | None = None) -> int:
    """新建会话并返回会话 ID。"""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO conversations(title) VALUES(?)",
            (title,),
        )
        return int(cur.lastrowid)


def conversation_exists(conversation_id: int) -> bool:
    """检查会话是否存在。"""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        return row is not None


def delete_conversation(conversation_id: int) -> None:
    """删除会话及其全部消息。"""
    with get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def get_messages(conversation_id: int) -> list[dict[str, Any]]:
    """读取一个会话下的全部消息（按时间正序）。"""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def save_message(conversation_id: int, role: str, content: str) -> None:
    """保存单条消息，并维护会话元信息。"""
    with get_conn() as conn:
        # 保存消息正文。
        conn.execute(
            """
            INSERT INTO messages(conversation_id, role, content)
            VALUES(?, ?, ?)
            """,
            (conversation_id, role, content),
        )
        # 更新会话最近活跃时间。
        conn.execute(
            """
            UPDATE conversations
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (conversation_id,),
        )

        # 若是用户首条消息，自动生成会话标题。
        if role == "user":
            row = conn.execute(
                "SELECT title FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            title = (row["title"] if row else None) or ""
            if not title.strip():
                normalized = " ".join(content.split()).strip()
                auto_title = normalized[:40] if normalized else f"会话 #{conversation_id}"
                conn.execute(
                    "UPDATE conversations SET title = ? WHERE id = ?",
                    (auto_title, conversation_id),
                )


def list_conversations() -> list[dict[str, Any]]:
    """返回会话列表，并附带最后一条消息摘要。"""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.title,
                c.created_at,
                c.updated_at,
                (
                    SELECT m.content
                    FROM messages m
                    WHERE m.conversation_id = c.id
                    ORDER BY m.id DESC
                    LIMIT 1
                ) AS last_message,
                (
                    SELECT m.created_at
                    FROM messages m
                    WHERE m.conversation_id = c.id
                    ORDER BY m.id DESC
                    LIMIT 1
                ) AS last_message_at
            FROM conversations c
            ORDER BY COALESCE(last_message_at, c.updated_at, c.created_at) DESC, c.id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


@app.on_event("startup")
def on_startup() -> None:
    """服务启动时初始化数据库。"""
    init_db()
    # clear_db()


@app.get("/")
def home():
    """基础健康检查接口。"""
    return {"msg": "hello ai"}


@app.post("/conversations")
def new_conversation():
    """新建空会话。"""
    conversation_id = create_conversation()
    return {"conversation_id": conversation_id, "messages": []}


@app.get("/conversations")
def conversations():
    """返回侧边栏所需会话列表。"""
    return {"conversations": list_conversations()}


@app.get("/conversations/latest")
def latest_conversation():
    """返回最新会话；若不存在则自动创建。"""
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT id
            FROM conversations
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

    # 优先使用最新会话，否则新建默认会话。
    if row is None:
        conversation_id = create_conversation()
    else:
        conversation_id = int(row["id"])

    return {
        "conversation_id": conversation_id,
        "messages": get_messages(conversation_id),
    }


@app.get("/conversations/{conversation_id}/messages")
def conversation_messages(conversation_id: int):
    """查询指定会话的完整消息。"""
    if not conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"conversation_id": conversation_id, "messages": get_messages(conversation_id)}


@app.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: int):
    """删除指定会话。"""
    if not conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    delete_conversation(conversation_id)
    return {"deleted": True, "conversation_id": conversation_id}


@app.post("/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request):
    """处理用户消息并以 SSE 流式返回助手回复。"""
    conversation_id = payload.conversation_id

    # 解析目标会话（无会话则新建）。
    if conversation_id is None:
        conversation_id = create_conversation()
    elif not conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")

    # 优先使用 content；兼容旧版 messages 字段。
    content = (payload.content or "").strip()
    if not content and payload.messages:
        user_messages = [m for m in payload.messages if m.get("role") == "user"]
        if user_messages:
            content = str(user_messages[-1].get("content", "")).strip()

    if not content:
        raise HTTPException(status_code=400, detail="empty message")

    # 先保存用户消息。
    save_message(conversation_id, "user", content)

    # 组装历史上下文发送给模型。
    system_prompt = (payload.system_prompt or "").strip()
    if not system_prompt:
        preset = (payload.system_prompt_preset or "").strip()
        if preset:
            system_prompt = SYSTEM_PROMPT_PRESETS.get(preset, "")
    history = [
        {"role": item["role"], "content": item["content"]}
        for item in get_messages(conversation_id)
    ]
    history, reserved_output_tokens = build_context_messages(history, system_prompt=system_prompt)

    async def event_stream():
        answer_parts: list[str] = []
        stream = None
        try:
            stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=history,
                max_tokens=reserved_output_tokens,
                stream=True,
            )
            for chunk in stream:
                if await request.is_disconnected():
                    break
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if not delta:
                    continue
                answer_parts.append(delta)
                yield f"data: {json.dumps({'type': 'delta', 'content': delta}, ensure_ascii=False)}\n\n"

            answer = "".join(answer_parts)
            if answer:
                save_message(conversation_id, "assistant", answer)
            if not await request.is_disconnected():
                yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            if answer_parts:
                save_message(conversation_id, "assistant", "".join(answer_parts))
            if not await request.is_disconnected():
                message = str(exc) or "stream failed"
                yield f"data: {json.dumps({'type': 'error', 'message': message}, ensure_ascii=False)}\n\n"
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
