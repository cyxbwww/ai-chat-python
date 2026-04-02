import sqlite3
from pathlib import Path
from typing import Any

# 项目根目录与数据库文件路径。
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chat_history.db"


def get_conn() -> sqlite3.Connection:
    """创建数据库连接，并启用按字段名访问。"""
    # 每次调用返回一个新的连接，避免跨请求共享连接状态。
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """初始化会话与消息表。"""
    with get_conn() as conn:
        # 会话表：保存会话基础元信息。
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
        # 消息表：保存每条消息的角色与内容。
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
    """清空业务数据并重置自增序列。"""
    with get_conn() as conn:
        # 先删子表，再删主表。
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM conversations")
        # 同步重置 SQLite 自增计数。
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('messages', 'conversations')")


def create_conversation(title: str | None = None) -> int:
    """创建会话并返回新会话 ID。"""
    with get_conn() as conn:
        # 插入会话记录，标题可为空。
        cur = conn.execute("INSERT INTO conversations(title) VALUES(?)", (title,))
        return int(cur.lastrowid)


def conversation_exists(conversation_id: int) -> bool:
    """检查会话是否存在。"""
    with get_conn() as conn:
        # 只查询主键，减少读取负担。
        row = conn.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return row is not None


def delete_conversation(conversation_id: int) -> None:
    """删除会话及其所有消息。"""
    with get_conn() as conn:
        # 先删除会话下消息，再删除会话本身。
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def get_messages(conversation_id: int) -> list[dict[str, Any]]:
    """获取指定会话的消息列表（按创建顺序）。"""
    with get_conn() as conn:
        # 使用 id 升序确保前端按时间正序渲染。
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
    """保存一条消息并更新会话状态。"""
    with get_conn() as conn:
        # 写入消息正文。
        conn.execute(
            """
            INSERT INTO messages(conversation_id, role, content)
            VALUES(?, ?, ?)
            """,
            (conversation_id, role, content),
        )
        # 更新时间戳，便于会话列表按最新活跃排序。
        conn.execute(
            """
            UPDATE conversations
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (conversation_id,),
        )

        if role == "user":
            # 用户首条消息自动生成会话标题，提升列表可读性。
            row = conn.execute("SELECT title FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            title = (row["title"] if row else None) or ""
            if not title.strip():
                normalized = " ".join(content.split()).strip()
                auto_title = normalized[:40] if normalized else f"会话 #{conversation_id}"
                conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (auto_title, conversation_id))


def list_conversations() -> list[dict[str, Any]]:
    """返回会话列表及最后一条消息摘要信息。"""
    with get_conn() as conn:
        # 子查询提取最后一条消息内容与时间，主查询负责排序。
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

