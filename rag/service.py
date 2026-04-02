from pathlib import Path

from .embedding import embed_text
from .loader import load_text_documents
from .splitter import split_text
from .vector_store import JsonVectorStore


class RagService:
    """RAG 处理服务：负责入库与检索。"""

    def __init__(self, uploads_dir: Path, store_dir: Path):
        # 保存上传目录与向量库存储位置。
        self.uploads_dir = uploads_dir
        self.store = JsonVectorStore(store_dir / "vectors.json")

    def ingest(self) -> int:
        """扫描上传目录并构建向量记录。"""
        documents = load_text_documents(self.uploads_dir)
        records: list[dict] = []
        for doc in documents:
            # 逐文档切块，保留块级别元数据。
            chunks = split_text(doc["content"])
            for index, chunk in enumerate(chunks):
                records.append(
                    {
                        "source": doc["source"],
                        "chunk_index": index,
                        "text": chunk,
                        "embedding": embed_text(chunk),
                    }
                )
        # 一次性落盘，避免频繁写入。
        self.store.save(records)
        return len(records)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """按关键词命中做简易检索（占位实现）。"""
        data = self.store.load()
        if not data:
            return []

        q = (query or "").lower()
        scored = []
        for item in data:
            # 命中关键词给 1 分，否则 0 分。
            text = str(item.get("text", ""))
            score = 1.0 if q and q in text.lower() else 0.0
            scored.append((score, item))

        # 按分数倒序返回前 top_k 条。
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[: max(1, top_k)]]

