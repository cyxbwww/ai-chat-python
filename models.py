from typing import Any

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """聊天流式接口请求体。"""

    # 会话 ID；为空时由后端新建会话。
    conversation_id: int | None = None
    # 用户本轮输入内容。
    content: str | None = None
    # 兼容旧格式的消息数组。
    messages: list[dict[str, Any]] | None = None
    # 自定义系统提示词。
    system_prompt: str | None = None
    # 系统提示词预设标识。
    system_prompt_preset: str | None = None
    # 是否从上一条 assistant 回复继续续写。
    continue_from_last: bool = False
