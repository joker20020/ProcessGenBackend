import logging
import time
import os
import tempfile
from io import BytesIO
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image

from config import config
from models import (
    EmbeddingRequest,
    EmbeddingResponse,
    RerankRequest,
    RerankResponse,
    HealthResponse,
    ErrorResponse,
    ImageInfoResponse,
    ImageListResponse,
)
from embeddings import EmbeddingService
from rerank import RerankService

logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ProcessGen Model Server",
    description="多模态嵌入和重排模型服务器",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_service = EmbeddingService()
rerank_service = RerankService()

DATA_DIR = Path("data").resolve()
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}


def is_safe_path(base_path: Path, requested_path: str) -> bool:
    """
    验证请求的路径是否在允许的基础目录内，防止目录遍历攻击。

    Args:
        base_path: 允许访问的基础目录
        requested_path: 用户请求的路径

    Returns:
        bool: 路径是否安全
    """
    try:
        # 解析请求的完整路径
        requested_full_path = (base_path / requested_path).resolve()
        # 确保解析后的路径仍在基础目录内
        return requested_full_path.is_relative_to(base_path)
    except (OSError, ValueError):
        return False


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - Time: {process_time:.2f}s")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy"
        if embedding_service.is_loaded() and rerank_service.is_loaded()
        else "loading",
        embedding_model_loaded=embedding_service.is_loaded(),
        rerank_model_loaded=rerank_service.is_loaded(),
        embedding_model_name=config.embedding_model_name,
        rerank_model_name=config.rerank_model_name,
    )


@app.get("/api/v1/images", response_model=ImageListResponse)
async def list_images():
    """获取data目录下的所有图像文件列表"""
    try:
        if not DATA_DIR.exists():
            return ImageListResponse(images=[], count=0)

        image_files = []
        for file_path in DATA_DIR.iterdir():
            if (
                file_path.is_file()
                and file_path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
            ):
                try:
                    file_stat = file_path.stat()
                    image_info = ImageInfoResponse(
                        filename=file_path.name,
                        size=file_stat.st_size,
                        format=file_path.suffix[1:].upper(),
                        created_at=time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime(file_stat.st_ctime)
                        ),
                    )
                    image_files.append(image_info)
                except OSError as e:
                    logger.warning(f"Failed to read file {file_path}: {e}")
                    continue

        image_files.sort(key=lambda x: x.filename)
        return ImageListResponse(images=image_files, count=len(image_files))
    except Exception as e:
        logger.error(f"Failed to list images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/images/{filename}")
async def get_image(filename: str):
    """根据文件名获取图像文件（支持返回原始图像或元数据）"""
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if not is_safe_path(DATA_DIR, filename):
        logger.warning(f"Path traversal attempt detected: {filename}")
        raise HTTPException(status_code=403, detail="Access denied")

    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    if file_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}",
        )

    try:
        return FileResponse(
            path=str(file_path),
            media_type=f"image/{file_path.suffix[1:]}",
            filename=filename,
        )
    except Exception as e:
        logger.error(f"Failed to serve image {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/images/{filename}/info", response_model=ImageInfoResponse)
async def get_image_info(filename: str):
    """根据文件名获取图像元数据信息"""
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if not is_safe_path(DATA_DIR, filename):
        logger.warning(f"Path traversal attempt detected: {filename}")
        raise HTTPException(status_code=403, detail="Access denied")

    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    if file_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}",
        )

    try:
        file_stat = file_path.stat()
        image_format = None
        width = None
        height = None

        try:
            with Image.open(file_path) as img:
                image_format = img.format
                width, height = img.size
        except Exception as e:
            logger.warning(f"Failed to read image metadata for {filename}: {e}")

        return ImageInfoResponse(
            filename=filename,
            size=file_stat.st_size,
            format=image_format,
            width=width,
            height=height,
            created_at=time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(file_stat.st_ctime)
            ),
        )
    except Exception as e:
        logger.error(f"Failed to get image info for {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/embed", response_model=EmbeddingResponse)
async def get_embedding(
    text: Optional[str] = Form(None), image_file: Optional[UploadFile] = File(None)
):
    if not text and not image_file:
        raise HTTPException(
            status_code=400, detail="Either text or image_file must be provided"
        )

    try:
        if text and not image_file:
            vector = await embedding_service.get_text_embedding(text)
            return EmbeddingResponse(
                vector=vector,
                dimension=embedding_service.get_dimension(),
                embedding_type="text",
            )
        elif image_file and not text:
            image_data = await image_file.read()
            image = Image.open(BytesIO(image_data))
            vector = await embedding_service.get_image_embedding(image)
            return EmbeddingResponse(
                vector=vector,
                dimension=embedding_service.get_dimension(),
                embedding_type="image",
            )
        else:
            image_data = await image_file.read()
            image = Image.open(BytesIO(image_data))
            vector = await embedding_service.get_fused_embedding(text, image)
            return EmbeddingResponse(
                vector=vector,
                dimension=embedding_service.get_dimension(),
                embedding_type="fused",
            )
    except Exception as e:
        logger.error(f"Embedding error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rerank", response_model=RerankResponse)
async def rerank(
    query_type: str = Form(...),
    query_text: Optional[str] = Form(None),
    query_image: Optional[UploadFile] = File(None),
    document_type: str = Form(...),
    document_text: Optional[str] = Form(None),
    document_image: Optional[UploadFile] = File(None),
):
    if query_type not in ["text", "image"]:
        raise HTTPException(
            status_code=400, detail="query_type must be 'text' or 'image'"
        )
    if document_type not in ["text", "image"]:
        raise HTTPException(
            status_code=400, detail="document_type must be 'text' or 'image'"
        )

    if query_type == "text" and not query_text:
        raise HTTPException(
            status_code=400, detail="query_text is required when query_type is 'text'"
        )
    if query_type == "image" and not query_image:
        raise HTTPException(
            status_code=400, detail="query_image is required when query_type is 'image'"
        )
    if document_type == "text" and not document_text:
        raise HTTPException(
            status_code=400,
            detail="document_text is required when document_type is 'text'",
        )
    if document_type == "image" and not document_image:
        raise HTTPException(
            status_code=400,
            detail="document_image is required when document_type is 'image'",
        )

    try:
        query_image_path = None
        document_image_path = None

        with tempfile.TemporaryDirectory() as temp_dir:
            if query_image:
                query_image_data = await query_image.read()
                query_image_path = os.path.join(
                    temp_dir, f"query_{query_image.filename}"
                )
                with open(query_image_path, "wb") as f:
                    f.write(query_image_data)

            if document_image:
                document_image_data = await document_image.read()
                document_image_path = os.path.join(
                    temp_dir, f"doc_{document_image.filename}"
                )
                with open(document_image_path, "wb") as f:
                    f.write(document_image_data)

            if query_type == "text" and document_type == "text":
                score = await rerank_service.compute_text_text_score(
                    query_text, document_text
                )
            elif query_type == "text" and document_type == "image":
                score = await rerank_service.compute_text_image_score(
                    query_text, document_image_path
                )
            elif query_type == "image" and document_type == "text":
                score = await rerank_service.compute_image_text_score(
                    query_image_path, document_text
                )
            elif query_type == "image" and document_type == "image":
                score = await rerank_service.compute_image_image_score(
                    query_image_path, document_image_path
                )
            else:
                raise HTTPException(status_code=400, detail="Unsupported combination")

        return RerankResponse(score=score)
    except Exception as e:
        logger.error(f"Rerank error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host=config.api_host, port=config.api_port, reload=True)
