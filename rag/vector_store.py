import json
from pathlib import Path
from typing import Any


class JsonVectorStore:
    """基于 JSON 文件的轻量向量存储。"""

    def __init__(self, store_path: Path):
        # 初始化时确保父目录存在。
        self.store_path = store_path
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, items: list[dict[str, Any]]) -> None:
        """保存向量记录到磁盘。"""
        # 使用 UTF-8 + 缩进，便于人工排查内容。
        self.store_path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> list[dict[str, Any]]:
        """从磁盘加载向量记录。"""
        if not self.store_path.exists():
            # 首次运行尚未建库时返回空集合。
            return []
        try:
            return json.loads(self.store_path.read_text(encoding="utf-8"))
        except Exception:
            # 读取失败时兜底为空，避免影响主流程。
            return []

