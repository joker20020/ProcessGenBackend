import asyncio
from typing import Optional
from sentence_transformers import CrossEncoder
from config import config


class RerankService:
    _instance: Optional["RerankService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self.model = CrossEncoder(
                config.rerank_model_name,
                device=config.rerank_device,
            )
            self.default_prompt = config.rerank_prompt if hasattr(config, "rerank_prompt") else "Retrieve images or text relevant to the user's query."
            self._initialized = True

    async def compute_score(
        self,
        query,
        document,
        max_length: Optional[int] = None,
        prompt: Optional[str] = None,
    ) -> float:
        """
        通用分数计算：支持文本-文本、文本-图像、图像-文本、图像-图像。
        query / document 可以是：
            - 纯文本字符串
            - 图像路径字符串
            - {"text": "...", "image": "..."} 字典
        """
        loop = asyncio.get_event_loop()
        if prompt is None:
            prompt = self.default_prompt
        scores = await loop.run_in_executor(
            None,
            lambda: self.model.predict([(query, document)], prompt=prompt),
        )
        return float(scores[0])

    async def compute_text_text_score(
        self, query_text: str, document_text: str, max_length: Optional[int] = None
    ) -> float:
        return await self.compute_score(query_text, document_text, max_length)

    async def compute_text_image_score(
        self,
        query_text: str,
        document_image_path: str,
        max_length: Optional[int] = None,
    ) -> float:
        return await self.compute_score(query_text, document_image_path, max_length)

    async def compute_image_text_score(
        self,
        query_image_path: str,
        document_text: str,
        max_length: Optional[int] = None,
    ) -> float:
        return await self.compute_score(query_image_path, document_text, max_length)

    async def compute_image_image_score(
        self,
        query_image_path: str,
        document_image_path: str,
        max_length: Optional[int] = None,
    ) -> float:
        return await self.compute_score(
            query_image_path, document_image_path, max_length
        )

    def is_loaded(self) -> bool:
        return hasattr(self, "_initialized") and self._initialized