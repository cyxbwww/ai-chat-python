import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI

from db import (
    conversation_exists,
    create_conversation,
    delete_conversation,
    get_conn,
    get_messages,
    init_db,
    list_conversations,
    save_message,
)
from models import ChatRequest

# 项目根目录。
BASE_DIR = Path(__file__).resolve().parent


def load_dotenv_file(dotenv_path: Path) -> None:
    """读取 .env 并注入环境变量（不覆盖已有变量）。"""
    if not dotenv_path.exists():
        return

    text = dotenv_path.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        # 逐行清洗，忽略空行与注释行。
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue

        # 使用 setdefault，避免覆盖系统或容器注入值。
        os.environ.setdefault(key, value)


# 启动时加载同目录 .env。
load_dotenv_file(BASE_DIR / ".env")

# 初始化 FastAPI 应用。
app = FastAPI()

# 开放跨域，便于前端本地开发直接调用。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 读取模型 API Key（优先 DEEPSEEK）。
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing API key: set DEEPSEEK_API_KEY or OPENAI_API_KEY")

# 初始化 OpenAI 兼容客户端。
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)

# 上下文 token 预算配置。
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "8192"))
RESERVED_OUTPUT_TOKENS = int(os.getenv("RESERVED_OUTPUT_TOKENS", "2048"))
TOKEN_CHARS_ESTIMATE = int(os.getenv("TOKEN_CHARS_ESTIMATE", "2"))
MESSAGE_OVERHEAD_TOKENS = int(os.getenv("MESSAGE_OVERHEAD_TOKENS", "6"))

# 系统提示词预设。
SYSTEM_PROMPT_PRESETS = {
    "general": "你是一个通用 AI 助手。请用中文回答，表达清晰、简洁，优先给出可执行建议。",
    "coding": "你是一名资深编程助手。请用中文回答，先给可运行方案，再补充关键原理与边界条件，代码尽量简洁。",
    "interview": "你是一名面试助手。请用中文回答，按“思路-要点-示例”结构给出答案，突出重点并提供可复述的表达。",
    "translation": "你是一名翻译助手。请准确翻译并保留原意与语气；如有歧义，给出更自然的候选译法。",
}

# 约束模型默认输出格式为 Markdown。
MARKDOWN_OUTPUT_INSTRUCTION = (
    "请直接输出 Markdown，不要用 ```markdown 或 ```text 包裹整段回复；"
    "仅在需要展示代码时使用代码块。"
)

# 续写模式下追加的指令。
CONTINUATION_PROMPT = "请从你上一条回答中断的位置继续，不要重复前文内容。"

# 可视为“非代码”围栏语言集合。
NON_CODE_FENCE_LANGS = {"", "markdown", "md", "mdown", "mkd", "text", "txt", "plain", "plaintext"}


def _likely_code(text: str) -> bool:
    """判断文本是否更像代码片段。"""
    # 使用轻量关键词 + 语法符号启发式判断。
    return bool(
        re.search(
            r"(^|\n)\s{0,3}(const|let|var|function|class|import|export|if|for|while|return|def|public|private)\b|=>|[;{}()]|</?[a-z][^>]*>",
            text,
            flags=re.IGNORECASE,
        )
    )


def _should_unwrap_fence(lang: str, body: str) -> bool:
    """判断某个 Markdown 围栏是否应当解包。"""
    normalized_lang = (lang or "").strip().lower()
    if normalized_lang in NON_CODE_FENCE_LANGS:
        # 显式标注 markdown/text 时直接解包。
        if normalized_lang:
            return True
        # 未标注语言时，只有不像代码才解包。
        return not _likely_code(body)
    return False


def unwrap_pseudo_markdown_fence(text: str) -> str:
    """去掉伪 Markdown 围栏（如 ```markdown）。"""
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not normalized:
        return text or ""

    def _replace(match: re.Match[str]) -> str:
        # 提取语言标记与正文，按规则决定是否解包。
        lang = (match.group(1) or "").strip().lower()
        body = match.group(2)
        if not _should_unwrap_fence(lang, body):
            return match.group(0)
        return body.rstrip()

    return re.sub(r"```([^\n`]*)\n([\s\S]*?)\n```", _replace, normalized)


def estimate_text_tokens(text: str) -> int:
    """按字符数粗略估算 token 数。"""
    normalized = text or ""
    chars_per_token = max(1, TOKEN_CHARS_ESTIMATE)
    return max(1, (len(normalized) + chars_per_token - 1) // chars_per_token)


def truncate_text_to_token_budget(text: str, token_budget: int) -> str:
    """按 token 预算截断文本，优先保留尾部内容。"""
    if token_budget <= 0:
        return ""
    chars_per_token = max(1, TOKEN_CHARS_ESTIMATE)
    max_chars = token_budget * chars_per_token
    if len(text) <= max_chars:
        return text
    # 对续写场景更有价值的是尾部上下文。
    return text[-max_chars:]


def build_context_messages(
    history: list[dict[str, Any]], system_prompt: str = ""
) -> tuple[list[dict[str, str]], int]:
    """根据预算裁剪上下文，返回发送给模型的消息列表与输出预算。"""
    # 拆分输入预算与输出预算。
    max_context = max(256, MAX_CONTEXT_TOKENS)
    reserved_output = max(1, min(RESERVED_OUTPUT_TOKENS, max_context - 1))
    input_budget = max(1, max_context - reserved_output)
    per_message_overhead = max(1, MESSAGE_OVERHEAD_TOKENS)

    selected_reversed: list[dict[str, str]] = []
    normalized_system_prompt = (system_prompt or "").strip()
    used_input_tokens = 0

    if normalized_system_prompt:
        # 先把系统提示词放入预算。
        system_cost = per_message_overhead + estimate_text_tokens(normalized_system_prompt)
        if system_cost <= input_budget:
            used_input_tokens += system_cost
        else:
            # 系统提示词过长时按预算截断。
            allowed = max(1, input_budget - per_message_overhead)
            normalized_system_prompt = truncate_text_to_token_budget(normalized_system_prompt, allowed)
            used_input_tokens += per_message_overhead + estimate_text_tokens(normalized_system_prompt)

    # 从最新消息向前选取，优先保留近邻语境。
    for item in reversed(history):
        role = str(item.get("role", "user"))
        content = str(item.get("content", "") or "")
        message_cost = per_message_overhead + estimate_text_tokens(content)

        if used_input_tokens + message_cost <= input_budget:
            selected_reversed.append({"role": role, "content": content})
            used_input_tokens += message_cost
            continue

        # 即便超预算，也尽量保留一条裁剪后的最新消息。
        if not selected_reversed:
            allowed = max(1, input_budget - per_message_overhead)
            truncated = truncate_text_to_token_budget(content, allowed)
            selected_reversed.append({"role": role, "content": truncated})
        break

    selected_messages = list(reversed(selected_reversed))
    if normalized_system_prompt:
        # 系统提示词固定放在最前面。
        selected_messages.insert(0, {"role": "system", "content": normalized_system_prompt})
    return selected_messages, reserved_output


@app.on_event("startup")
def on_startup() -> None:
    """应用启动时初始化数据库。"""
    # 启动即建表，避免首次请求才初始化。
    init_db()


@app.get("/")
def home():
    """健康检查接口。"""
    return {"msg": "hello ai"}


@app.post("/conversations")
def new_conversation():
    """新建空会话。"""
    # 仅创建会话，不写入消息。
    conversation_id = create_conversation()
    return {"conversation_id": conversation_id, "messages": []}


@app.get("/conversations")
def conversations():
    """返回会话列表。"""
    # 侧边栏依赖该接口展示最近会话。
    return {"conversations": list_conversations()}


@app.get("/conversations/latest")
def latest_conversation():
    """获取最新会话；不存在时自动创建。"""
    with get_conn() as conn:
        # 直接读取最大 id 的会话。
        row = conn.execute(
            """
            SELECT id
            FROM conversations
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

    if row is None:
        # 首次使用时创建默认会话。
        conversation_id = create_conversation()
    else:
        conversation_id = int(row["id"])

    return {
        "conversation_id": conversation_id,
        "messages": get_messages(conversation_id),
    }


@app.get("/conversations/{conversation_id}/messages")
def conversation_messages(conversation_id: int):
    """查询指定会话完整消息。"""
    if not conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"conversation_id": conversation_id, "messages": get_messages(conversation_id)}


@app.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: int):
    """删除指定会话。"""
    if not conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    # 删除会话会级联删除其消息（由业务函数完成）。
    delete_conversation(conversation_id)
    return {"deleted": True, "conversation_id": conversation_id}


@app.post("/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request):
    """流式聊天接口，使用 SSE 返回增量结果。"""
    conversation_id = payload.conversation_id
    continue_from_last = bool(payload.continue_from_last)

    # 解析会话目标：无会话则新建；有会话但不存在则报错。
    if conversation_id is None:
        conversation_id = create_conversation()
    elif not conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")

    # 优先读取 content；兼容旧版 messages 结构。
    content = (payload.content or "").strip()
    if not content and payload.messages:
        user_messages = [m for m in payload.messages if m.get("role") == "user"]
        if user_messages:
            content = str(user_messages[-1].get("content", "")).strip()

    # 校验普通提问与续写提问的输入约束。
    if continue_from_last and content:
        raise HTTPException(status_code=400, detail="continue request should not include content")
    if not continue_from_last and not content:
        raise HTTPException(status_code=400, detail="empty message")

    # 普通提问先写入用户消息；续写请求不写入数据库。
    if not continue_from_last:
        save_message(conversation_id, "user", content)

    # 解析系统提示词：自定义优先，其次预设。
    system_prompt = (payload.system_prompt or "").strip()
    if not system_prompt:
        preset = (payload.system_prompt_preset or "").strip()
        if preset:
            system_prompt = SYSTEM_PROMPT_PRESETS.get(preset, "")

    # 统一追加 Markdown 输出约束。
    if system_prompt:
        system_prompt = f"{system_prompt}\n\n{MARKDOWN_OUTPUT_INSTRUCTION}"
    else:
        system_prompt = MARKDOWN_OUTPUT_INSTRUCTION

    # 读取历史消息并构建模型上下文。
    history = [{"role": item["role"], "content": item["content"]} for item in get_messages(conversation_id)]
    if continue_from_last:
        if not history or history[-1]["role"] != "assistant":
            raise HTTPException(status_code=400, detail="no assistant message to continue")
        # 续写通过追加一条用户指令触发模型继续输出。
        history.append({"role": "user", "content": CONTINUATION_PROMPT})
    history, reserved_output_tokens = build_context_messages(history, system_prompt=system_prompt)

    async def event_stream():
        """SSE 事件生成器。"""
        answer_parts: list[str] = []
        stream = None
        finish_reason: str | None = None
        try:
            # 发起流式请求。
            stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=history,
                max_tokens=reserved_output_tokens,
                stream=True,
            )

            for chunk in stream:
                # 客户端断开后立即结束，避免无效计算。
                if await request.is_disconnected():
                    break
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = str(choice.finish_reason)

                delta = choice.delta.content
                if not delta:
                    continue

                # 累积增量文本，并把 delta 推送给前端。
                answer_parts.append(delta)
                yield f"data: {json.dumps({'type': 'delta', 'content': delta}, ensure_ascii=False)}\n\n"

            # 流结束后清洗输出并入库。
            answer = unwrap_pseudo_markdown_fence("".join(answer_parts))
            if answer:
                save_message(conversation_id, "assistant", answer)

            # 正常结束时发送 done 事件。
            if not await request.is_disconnected():
                truncated = finish_reason == "length"
                payload_done = {
                    "type": "done",
                    "conversation_id": conversation_id,
                    "finish_reason": finish_reason,
                    "truncated": truncated,
                }
                yield f"data: {json.dumps(payload_done, ensure_ascii=False)}\n\n"

        except Exception as exc:
            # 出错时尽量保留已生成内容，减少用户损失。
            if answer_parts:
                partial_answer = unwrap_pseudo_markdown_fence("".join(answer_parts))
                save_message(conversation_id, "assistant", partial_answer)
            if not await request.is_disconnected():
                message = str(exc) or "stream failed"
                yield f"data: {json.dumps({'type': 'error', 'message': message}, ensure_ascii=False)}\n\n"
        finally:
            # 清理底层流资源，避免连接泄漏。
            close = getattr(stream, "close", None)
            if callable(close):
                close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # 禁止代理缓存，保证流式实时性。
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

