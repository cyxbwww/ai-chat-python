import json
import os
import re
import hashlib
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI

from db import (
    create_document,
    conversation_exists,
    create_conversation,
    delete_conversation,
    get_document,
    get_conn,
    get_messages,
    init_db,
    list_chunks_for_faiss,
    list_conversations,
    list_documents,
    replace_document_chunks,
    save_message,
    search_document_chunks_like,
    update_document_status,
)
from models import ChatRequest
from rag.embedding import embed_text
from rag.splitter import split_text

# 项目根目录。
BASE_DIR = Path(__file__).resolve().parent

RAG_UPLOAD_DIR = BASE_DIR / "data" / "uploads"
RAG_FAISS_DIR = BASE_DIR / "data" / "faiss"
RAG_FAISS_INDEX_FILE = RAG_FAISS_DIR / "index.faiss"
RAG_FAISS_META_FILE = RAG_FAISS_DIR / "index_meta.json"

try:
    import faiss  # type: ignore
    import numpy as np
except Exception:
    faiss = None
    np = None

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None


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
RAG_MODE_INSTRUCTION = (
    "你是企业知识库助手，请根据提供的参考资料回答问题。\n"
    "如果资料中没有相关内容，请说明“知识库中未找到相关信息”，不要编造。"
)

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


def _ensure_rag_dirs() -> None:
    """确保 RAG 目录存在。"""
    RAG_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RAG_FAISS_DIR.mkdir(parents=True, exist_ok=True)


def _extract_text_from_pdf(path: Path) -> str:
    """Extract text from a PDF file."""
    if PdfReader is None:
        raise RuntimeError("Missing dependency 'pypdf', cannot parse PDF")

    reader = PdfReader(str(path))
    page_texts: list[str] = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            page_texts.append(text)
    return "\n\n".join(page_texts).strip()


def _read_text_with_fallback(path: Path) -> str:
    """Read plain text with common UTF/CJK encodings."""
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_text_from_docx(path: Path) -> str:
    """Extract text from .docx (OpenXML Word)."""
    texts: list[str] = []
    with zipfile.ZipFile(path, "r") as zf:
        targets = [name for name in zf.namelist() if name.startswith("word/") and name.endswith(".xml")]
        for name in sorted(targets):
            data = zf.read(name)
            root = ET.fromstring(data)
            for node in root.iter():
                # Localname-safe match for <w:t>.
                if node.tag.endswith("}t") and node.text:
                    value = node.text.strip()
                    if value:
                        texts.append(value)
    return "\n".join(texts).strip()


def _extract_text_from_pptx(path: Path) -> str:
    """Extract text from .pptx (OpenXML PowerPoint)."""
    texts: list[str] = []
    with zipfile.ZipFile(path, "r") as zf:
        slides = [
            name for name in zf.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        ]
        for name in sorted(slides):
            data = zf.read(name)
            root = ET.fromstring(data)
            for node in root.iter():
                # Localname-safe match for <a:t>.
                if node.tag.endswith("}t") and node.text:
                    value = node.text.strip()
                    if value:
                        texts.append(value)
    return "\n".join(texts).strip()


def _extract_text_from_xlsx(path: Path) -> str:
    """Extract text from .xlsx (OpenXML Excel)."""
    texts: list[str] = []
    with zipfile.ZipFile(path, "r") as zf:
        # shared strings
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for node in root.iter():
                if node.tag.endswith("}t") and node.text:
                    value = node.text.strip()
                    if value:
                        texts.append(value)

        # inline strings in sheets
        sheets = [
            name for name in zf.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        ]
        for name in sorted(sheets):
            root = ET.fromstring(zf.read(name))
            for node in root.iter():
                if node.tag.endswith("}t") and node.text:
                    value = node.text.strip()
                    if value:
                        texts.append(value)
    return "\n".join(texts).strip()


def _extract_document_text(path: Path, file_ext: str) -> str:
    """Extract indexable text by file extension."""
    ext = (file_ext or path.suffix or "").lower()

    # Route by extension to avoid decoding binary files as plain text.
    if ext == ".pdf":
        return _extract_text_from_pdf(path)
    if ext == ".docx":
        return _extract_text_from_docx(path)
    if ext == ".pptx":
        return _extract_text_from_pptx(path)
    if ext == ".xlsx":
        return _extract_text_from_xlsx(path)

    # Legacy Office binary formats are explicitly rejected.
    if ext in {".doc", ".xls", ".ppt"}:
        raise RuntimeError(f"unsupported legacy office format: {ext}, please convert to .docx/.xlsx/.pptx")

    text_like_exts = {
        ".txt",
        ".md",
        ".markdown",
        ".json",
        ".csv",
        ".log",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".java",
        ".go",
        ".sql",
        ".html",
        ".htm",
        ".xml",
        ".yaml",
        ".yml",
    }
    if ext in text_like_exts or not ext:
        return _read_text_with_fallback(path)

    # Unknown extension: fallback to text read to keep backward compatibility.
    return _read_text_with_fallback(path)


def _build_faiss_index_from_chunks() -> dict[str, Any]:
    """根据 document_chunks 重建 FAISS 索引。"""
    if faiss is None or np is None:
        raise RuntimeError("faiss-cpu 未安装，无法写入 FAISS 索引")

    rows = list_chunks_for_faiss()
    if not rows:
        if RAG_FAISS_INDEX_FILE.exists():
            RAG_FAISS_INDEX_FILE.unlink()
        if RAG_FAISS_META_FILE.exists():
            RAG_FAISS_META_FILE.unlink()
        return {"vector_count": 0}

    vectors: list[list[float]] = []
    metadata: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("content", ""))
        vectors.append(embed_text(text))
        metadata.append(
            {
                "chunk_id": row.get("chunk_id"),
                "document_id": row.get("document_id"),
                "chunk_index": row.get("chunk_index"),
                "file_name": row.get("file_name"),
                "content": text,
            }
        )

    matrix = np.array(vectors, dtype="float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(RAG_FAISS_INDEX_FILE))
    RAG_FAISS_META_FILE.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    return {"vector_count": int(index.ntotal)}


def _search_faiss(query: str, top_k: int) -> list[dict[str, Any]]:
    """在 FAISS 索引中执行向量检索。"""
    if faiss is None or np is None:
        return []
    if not RAG_FAISS_INDEX_FILE.exists() or not RAG_FAISS_META_FILE.exists():
        return []
    try:
        index = faiss.read_index(str(RAG_FAISS_INDEX_FILE))
        metadata = json.loads(RAG_FAISS_META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not metadata:
        return []

    query_vec = np.array([embed_text(query)], dtype="float32")
    faiss.normalize_L2(query_vec)
    distances, ids = index.search(query_vec, max(1, top_k))
    hits: list[dict[str, Any]] = []
    for score, idx in zip(distances[0], ids[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx]
        hits.append(
            {
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "chunk_index": item.get("chunk_index"),
                "file_name": item.get("file_name"),
                "content": item.get("content"),
                "score": float(score),
            }
        )
    return hits


def _rag_search(query: str, top_k: int) -> list[dict[str, Any]]:
    """RAG 检索入口：优先 FAISS，失败回退 LIKE。"""
    q = (query or "").strip()
    if not q:
        return []
    hits = _search_faiss(q, top_k)
    if hits:
        return hits
    fallback = search_document_chunks_like(q, top_k)
    for item in fallback:
        item["score"] = 0.0
    return fallback


@app.on_event("startup")
def on_startup() -> None:
    """应用启动时初始化数据库。"""
    # 启动即建表，避免首次请求才初始化。
    init_db()


@app.on_event("startup")
def on_startup_rag_dirs() -> None:
    """额外初始化 RAG 所需目录。"""
    _ensure_rag_dirs()


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


@app.post("/rag/documents")
async def rag_upload_document(file: UploadFile = File(...)):
    """1) 上传文档：保存文件到 data/uploads，并写入 documents。"""
    # 校验文件名与内容。
    file_name = (file.filename or "").strip()
    if not file_name:
        raise HTTPException(status_code=400, detail="empty filename")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")

    # 生成落盘文件名并保存到 uploads 目录。
    ext = Path(file_name).suffix.lower()
    save_name = f"{uuid4().hex}{ext}"
    save_path = RAG_UPLOAD_DIR / save_name
    save_path.write_bytes(content)

    # 写入文档元数据到 documents。
    file_hash = hashlib.sha256(content).hexdigest()
    document_id = create_document(
        file_name=file_name,
        file_path=str(save_path),
        file_ext=ext,
        file_size=len(content),
        file_hash=file_hash,
        status="uploaded",
    )
    document = get_document(document_id)
    return {"document": document}


@app.post("/rag/documents/{document_id}/index")
def rag_build_document_index(document_id: int):
    """2) 建索引：切分文本块、写入 document_chunks，并写入 FAISS。"""
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")

    path = Path(str(doc.get("file_path", "")))
    if not path.exists():
        update_document_status(document_id, "index_failed")
        raise HTTPException(status_code=400, detail="document file not found")

    try:
        # 读取文档并切分为 chunk。
        file_ext = str(doc.get("file_ext", "") or path.suffix or "").lower()
        text = _extract_document_text(path, file_ext)
        if not text.strip():
            if file_ext == ".pdf":
                raise RuntimeError("PDF text extraction returned empty content; file may be image-only and needs OCR")
            raise RuntimeError("document text is empty after extraction")
        chunks = split_text(text)
        if not chunks:
            raise RuntimeError("document split result is empty")
        chunk_count = replace_document_chunks(document_id, chunks, token_chars_estimate=TOKEN_CHARS_ESTIMATE)
        update_document_status(document_id, "indexed")

        # 基于全量 chunk 重建 FAISS 索引文件。
        faiss_result = _build_faiss_index_from_chunks()
        return {
            "document_id": document_id,
            "chunk_count": chunk_count,
            "faiss": faiss_result,
        }
    except RuntimeError as exc:
        update_document_status(document_id, "index_failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        update_document_status(document_id, "index_failed")
        raise HTTPException(status_code=500, detail=f"build index failed: {exc}") from exc


@app.get("/rag/documents")
def rag_document_list(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
    """3) 文档列表：分页返回 documents。"""
    return {"documents": list_documents(limit=limit, offset=offset), "limit": limit, "offset": offset}


@app.post("/rag/search")
async def rag_search(request: Request):
    """4) 检索测试：优先 FAISS，失败回退 LIKE。"""
    payload = await request.json()
    query = str(payload.get("query", "")).strip()
    top_k = int(payload.get("top_k", 5) or 5)
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    top_k = max(1, min(top_k, 20))
    hits = _rag_search(query, top_k)
    return {"query": query, "top_k": top_k, "hits": hits}


@app.post("/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request):
    """流式聊天接口，使用 SSE 返回增量结果。"""
    conversation_id = payload.conversation_id
    continue_from_last = bool(payload.continue_from_last)
    rag_enabled = bool(payload.rag_enabled)
    rag_top_k = int(payload.rag_top_k or 4)
    rag_top_k = max(1, min(rag_top_k, 20))

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

    # 解析用户提供的系统提示词：仅在非 RAG 模式使用。
    user_system_prompt = ""
    if not rag_enabled:
        user_system_prompt = (payload.system_prompt or "").strip()
        if not user_system_prompt:
            preset = (payload.system_prompt_preset or "").strip()
            if preset:
                user_system_prompt = SYSTEM_PROMPT_PRESETS.get(preset, "")

        # 再兜底兼容 messages 结构里携带的 system/user（前端可显式同时传入）。
        if payload.messages and not user_system_prompt:
            system_messages = [m for m in payload.messages if m.get("role") == "system"]
            if system_messages:
                user_system_prompt = str(system_messages[-1].get("content", "")).strip()

    if rag_enabled:
        # RAG 模式下忽略前端 prompt，只保留固定 RAG 指令与输出格式约束。
        system_prompt = f"{MARKDOWN_OUTPUT_INSTRUCTION}\n\n{RAG_MODE_INSTRUCTION}"
    elif user_system_prompt:
        system_prompt = f"{user_system_prompt}\n\n{MARKDOWN_OUTPUT_INSTRUCTION}"
    else:
        system_prompt = MARKDOWN_OUTPUT_INSTRUCTION

    # 若启用 RAG，则把检索结果拼进系统提示词，增强回答依据。
    rag_references: list[dict[str, Any]] = []
    if rag_enabled and (not continue_from_last) and content:
        rag_query = content
        rag_references = _rag_search(rag_query, rag_top_k)

        if rag_references:
            context_blocks = []
            for idx, item in enumerate(rag_references, start=1):
                context_blocks.append(
                    f"[参考{idx}] 文件: {item.get('file_name')} | chunk: {item.get('chunk_index')}\n{item.get('content', '')}"
                )
            rag_context = "\n\n".join(context_blocks)
            system_prompt = (
                f"{system_prompt}\n\n"
                "以下是检索到的知识库片段，请优先据此回答；若不足请明确说明：\n"
                f"{rag_context}"
            )

    # 读取历史消息并构建模型上下文。
    history = [{"role": item["role"], "content": item["content"]} for item in get_messages(conversation_id)]
    if continue_from_last:
        if not history or history[-1]["role"] != "assistant":
            raise HTTPException(status_code=400, detail="no assistant message to continue")
        # 续写通过追加一条用户指令触发模型继续输出。
        history.append({"role": "user", "content": CONTINUATION_PROMPT})
    elif (not rag_enabled) and history and history[-1]["role"] == "user" and user_system_prompt:
        # 把“system_prompt + 用户问题”作为同一条用户消息发送给模型。
        latest_question = str(history[-1].get("content", "")).strip()
        if latest_question:
            history[-1]["content"] = (
                "请结合以下系统提示词回答用户问题：\n"
                f"{user_system_prompt}\n\n"
                "用户问题：\n"
                f"{latest_question}"
            )
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
                    "rag_references": rag_references,
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
