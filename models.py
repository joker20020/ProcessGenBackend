from pydantic import BaseModel, Field
from typing import Optional
from fastapi import UploadFile


class EmbeddingRequest(BaseModel):
    text: Optional[str] = Field(default=None, description="待嵌入的文本内容")
    image_file: Optional[UploadFile] = Field(
        default=None, description="待嵌入的图像文件"
    )


class EmbeddingResponse(BaseModel):
    vector: list[float] = Field(description="嵌入向量")
    dimension: int = Field(description="向量维度")
    embedding_type: str = Field(description="嵌入类型: text/image/fused")


class RerankRequest(BaseModel):
    query_type: str = Field(description="查询类型: text/image")
    query_text: Optional[str] = Field(default=None, description="查询文本")
    query_image: Optional[UploadFile] = Field(default=None, description="查询图像文件")
    document_type: str = Field(description="文档类型: text/image")
    document_text: Optional[str] = Field(default=None, description="文档文本")
    document_image: Optional[UploadFile] = Field(
        default=None, description="文档图像文件"
    )


class RerankResponse(BaseModel):
    score: float = Field(description="相似度评分 (0-1之间)")


class HealthResponse(BaseModel):
    status: str = Field(description="服务状态: healthy/unhealthy")
    embedding_model_loaded: bool = Field(description="嵌入模型是否已加载")
    rerank_model_loaded: bool = Field(description="重排模型是否已加载")
    embedding_model_name: str = Field(description="嵌入模型名称")
    rerank_model_name: str = Field(description="重排模型名称")


class ErrorResponse(BaseModel):
    error: str = Field(description="错误信息")
    detail: Optional[str] = Field(default=None, description="详细错误信息")


class ImageInfoResponse(BaseModel):
    filename: str = Field(description="图像文件名")
    size: int = Field(description="文件大小（字节）")
    format: Optional[str] = Field(default=None, description="图像格式（如JPEG、PNG等）")
    width: Optional[int] = Field(default=None, description="图像宽度（像素）")
    height: Optional[int] = Field(default=None, description="图像高度（像素）")
    created_at: Optional[str] = Field(default=None, description="文件创建时间")


class ImageListResponse(BaseModel):
    images: list[ImageInfoResponse] = Field(description="图像文件列表")
    count: int = Field(description="图像文件总数")
