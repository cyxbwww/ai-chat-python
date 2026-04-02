import hashlib


def embed_text(text: str, dim: int = 64) -> list[float]:
    """把文本映射为固定维度向量（占位实现）。"""
    # 先做哈希，保证相同文本得到稳定向量。
    normalized = (text or "").encode("utf-8")
    digest = hashlib.sha256(normalized).digest()
    # 扩展字节数组至目标维度。
    values = list(digest) * ((dim + len(digest) - 1) // len(digest))
    # 归一化到 [0, 1] 区间，便于后续做相似度计算。
    return [v / 255.0 for v in values[:dim]]


def embed_texts(texts: list[str], dim: int = 64) -> list[list[float]]:
    """批量计算文本向量。"""
    # 顺序映射，保持与输入文本一一对应。
    return [embed_text(item, dim=dim) for item in texts]

