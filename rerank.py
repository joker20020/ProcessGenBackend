import asyncio
from typing import Optional
from PIL import Image
from transformers import AutoModel
from config import config


class RerankService:
    _instance: Optional["RerankService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self.model = AutoModel.from_pretrained(
                config.rerank_model_name,
                torch_dtype="auto",
                trust_remote_code=True,
                device_map=config.rerank_device,
            )
            self._initialized = True

    async def compute_score(
        self, query: str, document: str, max_length: int = 1024
    ) -> float:
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self.model.compute_score(
                [[query, document]], max_length=max_length, doc_type="text"
            ),
        )
        return float(scores)

    async def compute_text_text_score(
        self, query_text: str, document_text: str, max_length: int = 1024
    ) -> float:
        return await self.compute_score(query_text, document_text, max_length)

    async def compute_text_image_score(
        self, query_text: str, document_image_path: str, max_length: int = 1024
    ) -> float:
        return await self.compute_score(query_text, document_image_path, max_length)

    async def compute_image_text_score(
        self, query_image_path: str, document_text: str, max_length: int = 1024
    ) -> float:
        return await self.compute_score(query_image_path, document_text, max_length)

    async def compute_image_image_score(
        self, query_image_path: str, document_image_path: str, max_length: int = 1024
    ) -> float:
        return await self.compute_score(
            query_image_path, document_image_path, max_length
        )

    def is_loaded(self) -> bool:
        return hasattr(self, "_initialized") and self._initialized
