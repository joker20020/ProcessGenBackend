from pydantic_settings import BaseSettings
from pydantic import Field


class AppConfig(BaseSettings):
    api_host: str = Field(default="0.0.0.0", description="API服务器监听地址")
    api_port: int = Field(default=8050, description="API服务器端口")
    log_level: str = Field(
        default="INFO", description="日志级别 (DEBUG, INFO, WARNING, ERROR)"
    )
    embedding_model_name: str = Field(
        default="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct", description="嵌入模型名称或路径"
    )
    rerank_model_name: str = Field(
        default="jinaai/jina-reranker-m0", description="重排模型名称或路径"
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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


config = AppConfig()
