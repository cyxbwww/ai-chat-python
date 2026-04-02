from pathlib import Path


def load_text_documents(upload_dir: Path) -> list[dict[str, str]]:
    """加载上传目录中的文本文件。"""
    docs: list[dict[str, str]] = []
    if not upload_dir.exists():
        # 目录不存在时直接返回空列表，避免调用方额外判断。
        return docs

    for file_path in upload_dir.rglob("*"):
        # 只处理文件，跳过目录。
        if not file_path.is_file():
            continue
        try:
            # 统一按 UTF-8 读取，失败文件直接忽略。
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            continue
        docs.append({"source": str(file_path), "content": text})
    return docs

