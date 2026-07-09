from pydantic_settings import BaseSettings
from pydantic import Field


class AppConfig(BaseSettings):
    api_host: str = Field(default="0.0.0.0", description="API服务器监听地址")
    api_port: int = Field(default=8050, description="API服务器端口")
    log_level: str = Field(
        default="INFO", description="日志级别 (DEBUG, INFO, WARNING, ERROR)"
    )
    embedding_model_name: str = Field(
        default="Qwen/Qwen3-VL-Embedding-2B", description="嵌入模型名称或路径"
    )
    rerank_model_name: str = Field(
        default="Qwen/Qwen3-VL-Reranker-2B", description="重排模型名称或路径"
    )
    embedding_device: str = Field(
        default="cuda:1", description="嵌入模型运行设备 (cuda:0, cpu)"
    )
    rerank_device: str = Field(
        default="cuda:1", description="重排模型运行设备 (cuda:1, cpu)"
    )
    max_upload_size: int = Field(
        default=10 * 1024 * 1024, description="最大上传文件大小 (字节)"
    )

    comfyui_url: str = Field(
        default="http://127.0.0.1:8188", description="ComfyUI 服务地址"
    )
    comfyui_timeout: int = Field(default=300, description="ComfyUI 请求超时时间（秒）")

    milvus_uri: str = Field(
        default="http://localhost:19530", description="Milvus 服务器地址"
    )
    rag_subject_default: str = Field(
        default="capp", description="RAG 默认 subject 分区标签"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


config = AppConfig()
