def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """把长文本切分为固定大小的块，并保留重叠区。"""
    normalized = (text or "").strip()
    if not normalized:
        # 空文本直接返回空数组。
        return []
    if chunk_size <= 0:
        # 非法分块参数时，退化为单块输出。
        return [normalized]
    # overlap 上限控制为 chunk_size - 1，避免死循环。
    overlap = max(0, min(overlap, chunk_size - 1))

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        # 计算当前片段边界。
        end = min(len(normalized), start + chunk_size)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            # 到达末尾时结束循环。
            break
        # 下一段起点回退 overlap，形成上下文重叠。
        start = end - overlap
    return chunks

