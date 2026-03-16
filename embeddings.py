import asyncio
from typing import Optional
from PIL import Image
from transformers import AutoModel
from config import config


class EmbeddingService:
    _instance: Optional["EmbeddingService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self.model = AutoModel.from_pretrained(
                config.embedding_model_name,
                torch_dtype="float32",
                device_map=config.embedding_device,
                trust_remote_code=True,
            )
            self._initialized = True

    async def get_text_embedding(self, text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self.model.get_text_embeddings(texts=[text], is_query=True)
        )
        return embeddings[0].tolist()

    async def get_image_embedding(self, image: Image.Image) -> list[float]:
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self.model.get_image_embeddings(images=[image], is_query=True)
        )
        return embeddings[0].tolist()

    async def get_fused_embedding(self, text: str, image: Image.Image) -> list[float]:
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.get_fused_embeddings(
                images=[image], texts=[text], is_query=True
            ),
        )
        return embeddings[0].tolist()

    def get_dimension(self) -> int:
        return self.model.config.hidden_size

    def is_loaded(self) -> bool:
        return hasattr(self, "_initialized") and self._initialized
