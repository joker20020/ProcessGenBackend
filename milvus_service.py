import logging
import os
from typing import Optional
from pymilvus import MilvusClient, DataType

from config import config
from embeddings import EmbeddingService
from models import CollectionInfo

logger = logging.getLogger(__name__)


class MilvusService:
    _instance: Optional["MilvusService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self.client = MilvusClient(uri=config.milvus_uri)
            self._initialized = True

    def is_available(self) -> bool:
        try:
            self.client.list_collections()
            return True
        except Exception as e:
            logger.warning(f"Milvus 不可用: {e}")
            return False

    def init_collection(self, name: str) -> bool:
        if self.client.has_collection(collection_name=name):
            return self.client.get_load_state(collection_name=name)

        dim = EmbeddingService().get_dimension()
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="type", datatype=DataType.VARCHAR, max_length=16)
        schema.add_field(field_name="path", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="subject", datatype=DataType.VARCHAR, max_length=64)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="id", index_type="")
        index_params.add_index(field_name="embedding", index_type="", metric_type="COSINE")

        self.client.create_collection(
            collection_name=name, schema=schema, index_params=index_params
        )
        return self.client.get_load_state(collection_name=name)

    def list_collections(self) -> list[CollectionInfo]:
        result = []
        for name in self.client.list_collections():
            row_count = None
            loaded = None
            try:
                stats = self.client.get_collection_stats(collection_name=name)
                row_count = stats.get("row_count")
                loaded = self.client.get_load_state(collection_name=name)
            except Exception as e:
                logger.warning(f"读取集合 {name} 统计失败: {e}")
            result.append(CollectionInfo(name=name, row_count=row_count, loaded=loaded))
        return result

    def drop_collection(self, name: str) -> None:
        if self.client.has_collection(collection_name=name):
            self.client.drop_collection(collection_name=name)

    def insert(self, name: str, data: list[dict]) -> list:
        return self.client.insert(collection_name=name, data=data)

    def search(
        self, name: str, vector: list[float], limit: int = 10, subject: Optional[str] = None
    ) -> list[dict]:
        expr = f"subject == '{subject}'" if subject else ""
        res = self.client.search(
            collection_name=name,
            data=[vector],
            limit=limit,
            filter=expr,
            output_fields=["id", "text", "subject", "path", "type"],
        )
        items = []
        for hit in res[0]:
            entity = hit.get("entity", hit)
            item = {
                "id": entity.get("id", hit.get("id")),
                "score": float(hit.get("distance", hit.get("score", 0.0))),
                "type": entity.get("type", "text"),
                "text": entity.get("text", ""),
                "path": entity.get("path", ""),
                "subject": entity.get("subject", ""),
            }
            if item["type"] == "image":
                item["asset_path"] = _relative_to_data(item["path"])
            items.append(item)
        return items


def _relative_to_data(abs_path: str) -> Optional[str]:
    """返回相对 data/ 目录的路径，无法转换则 None。"""
    if not abs_path:
        return None
    try:
        data_dir = os.path.abspath("data")
        full = os.path.abspath(abs_path)
        if full.startswith(data_dir + os.sep):
            return os.path.relpath(full, data_dir)
    except Exception:
        pass
    return None
