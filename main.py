import sqlite3
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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


class ChatRequest(BaseModel):
    """/chat 接口请求体。"""

    conversation_id: int | None = None
    content: str | None = None
    messages: list[dict[str, Any]] | None = None


def get_conn() -> sqlite3.Connection:
    """创建数据库连接，并支持按字段名访问行数据。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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

        # 兼容旧库：如果没有 updated_at 字段则补上并回填。
        columns = conn.execute("PRAGMA table_info(conversations)").fetchall()
        names = {str(row["name"]) for row in columns}
        if "updated_at" not in names:
            conn.execute("ALTER TABLE conversations ADD COLUMN updated_at DATETIME")
            conn.execute(
                """
                UPDATE conversations
                SET updated_at = COALESCE(updated_at, created_at, CURRENT_TIMESTAMP)
                """
            )


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


@app.post("/chat")
async def chat(payload: ChatRequest):
    """处理用户消息、调用模型并保存双向消息。"""
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
    history = [
        {"role": item["role"], "content": item["content"]}
        for item in get_messages(conversation_id)
    ]

    # 调用模型生成回复。
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=history,
    )

    # 保存助手回复并返回前端。
    answer = resp.choices[0].message.content or ""
    save_message(conversation_id, "assistant", answer)

    return {
        "conversation_id": conversation_id,
        "answer": answer,
    }


