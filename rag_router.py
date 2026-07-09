import re
import time
import logging
from pathlib import Path
from io import BytesIO
from typing import Optional, List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse
from PIL import Image

from config import config
from embeddings import EmbeddingService
from milvus_service import MilvusService
from text_processor import TextProcessor
from models import (
    CreateCollectionRequest,
    CollectionInfo,
    CollectionListResponse,
    AddTextResponse,
    AddImageResponse,
    SearchResultItem,
    SearchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])

TEXT_DIR = Path("data/text").resolve()
IMAGE_DIR = Path("data/images").resolve()
ALLOWED_TEXT_EXT = {".md", ".txt", ".pdf"}
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}

_SUBJECT_RE = re.compile(r"^[A-Za-z0-9_\-一-鿿]+$")


def _validate_subject(subject: Optional[str]) -> Optional[str]:
    if subject is None:
        return None
    if not _SUBJECT_RE.fullmatch(subject):
        raise HTTPException(status_code=400, detail="subject 仅允许字母、数字、下划线、连字符及中文")
    return subject


def _ensure_milvus():
    svc = MilvusService()
    if not svc.is_available():
        raise HTTPException(status_code=503, detail="Milvus 不可用，请检查 MILVUS_URI 配置及服务状态")
    return svc


# ---- 集合管理 ----

@router.post("/collections", response_model=CollectionInfo, status_code=201)
async def create_collection(req: CreateCollectionRequest):
    if not req.collection_name or not req.collection_name.strip():
        raise HTTPException(status_code=400, detail="collection_name 不能为空")
    svc = _ensure_milvus()
    svc.init_collection(req.collection_name)
    TEXT_DIR.joinpath(req.collection_name).mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.joinpath(req.collection_name).mkdir(parents=True, exist_ok=True)
    return CollectionInfo(name=req.collection_name, loaded=True)


@router.get("/collections", response_model=CollectionListResponse)
async def list_collections():
    svc = _ensure_milvus()
    cols = svc.list_collections()
    return CollectionListResponse(collections=cols, count=len(cols))


@router.delete("/collections/{name}")
async def delete_collection(name: str):
    svc = _ensure_milvus()
    if not svc.client.has_collection(collection_name=name):
        raise HTTPException(status_code=404, detail=f"集合 {name} 不存在")
    svc.drop_collection(name)
    import shutil
    for d in (TEXT_DIR.joinpath(name), IMAGE_DIR.joinpath(name)):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    return {"status": "success", "collection_name": name, "message": f"集合 {name} 已删除"}


# ---- 添加文本 ----

@router.post("/collections/{name}/text", response_model=AddTextResponse)
async def add_text(
    name: str,
    file: UploadFile = File(...),
    subject: Optional[str] = Form(None),
):
    svc = _ensure_milvus()
    if not svc.client.has_collection(collection_name=name):
        raise HTTPException(status_code=404, detail=f"集合 {name} 不存在")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_TEXT_EXT:
        raise HTTPException(status_code=400, detail=f"仅支持 {ALLOWED_TEXT_EXT} 文件")

    raw = await file.read()
    if ext == ".pdf":
        tmp = Path(f"/tmp/_rag_{int(time.time())}.pdf")
        tmp.write_bytes(raw)
        try:
            content = TextProcessor().extract_text_from_pdf(str(tmp))
        finally:
            tmp.unlink(missing_ok=True)
    else:
        content = raw.decode("utf-8", errors="ignore")

    chunks = TextProcessor().split_markdown_for_rag(content, min_words=10, include_subsections=False)
    if not chunks:
        raise HTTPException(status_code=400, detail="文件无可用文本块")

    save_dir = TEXT_DIR.joinpath(name)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir.joinpath(f"{int(time.time())}{ext}")
    save_path.write_text(content, encoding="utf-8")

    emb = EmbeddingService()
    subj = subject or config.rag_subject_default
    data = []
    for chunk in chunks:
        vec = await emb.get_text_embedding(chunk)
        data.append({
            "embedding": vec, "type": "text", "text": chunk,
            "path": str(save_path.resolve()), "subject": subj,
        })
    svc.insert(name, data)
    return AddTextResponse(
        status="success", collection_name=name,
        chunks_inserted=len(data), saved_path=str(save_path.resolve()),
    )


# ---- 添加图像 ----

@router.post("/collections/{name}/images", response_model=AddImageResponse)
async def add_images(
    name: str,
    images: List[UploadFile] = File(...),
    descriptions: List[str] = Form(...),
    subject: Optional[str] = Form(None),
):
    svc = _ensure_milvus()
    if not svc.client.has_collection(collection_name=name):
        raise HTTPException(status_code=404, detail=f"集合 {name} 不存在")
    if len(images) != len(descriptions):
        raise HTTPException(status_code=400, detail="images 与 descriptions 数量必须一致")
    if not images:
        raise HTTPException(status_code=400, detail="至少上传一张图片")

    exts = [Path(f.filename or ".png").suffix.lower() for f in images]
    for ext in exts:
        if ext not in ALLOWED_IMAGE_EXT:
            raise HTTPException(status_code=400, detail=f"不支持图像类型 {ext}")

    save_dir = IMAGE_DIR.joinpath(name)
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    emb = EmbeddingService()
    subj = subject or config.rag_subject_default
    data = []
    for i, (img_file, desc) in enumerate(zip(images, descriptions)):
        ext = exts[i]
        raw = await img_file.read()
        save_path = save_dir.joinpath(f"{ts}_{i}{ext}")
        save_path.write_bytes(raw)
        pil = Image.open(BytesIO(raw))
        if desc and desc.strip():
            vec = await emb.get_fused_embedding(desc, pil)
        else:
            vec = await emb.get_image_embedding(pil)
        data.append({
            "embedding": vec, "type": "image", "text": desc,
            "path": str(save_path.resolve()), "subject": subj,
        })
    svc.insert(name, data)
    return AddImageResponse(
        status="success", collection_name=name, images_inserted=len(data),
    )


# ---- 检索 ----

def _build_results(raw: list[dict]) -> list[SearchResultItem]:
    items = []
    for r in raw:
        items.append(SearchResultItem(
            id=r.get("id", 0), score=r.get("score", 0.0), type=r.get("type", "text"),
            text=r.get("text", ""), path=r.get("path", ""), subject=r.get("subject", ""),
            asset_path=r.get("asset_path"),
        ))
    return items


@router.get("/collections/{name}/search", response_model=SearchResponse)
async def search_text(
    name: str,
    query: str = Query(..., description="查询文本"),
    limit: int = Query(10, ge=1, le=100),
    subject: Optional[str] = Query(None),
):
    svc = _ensure_milvus()
    if not svc.client.has_collection(collection_name=name):
        raise HTTPException(status_code=404, detail=f"集合 {name} 不存在")
    subject = _validate_subject(subject)
    emb = EmbeddingService()
    vec = await emb.get_text_embedding(query)
    raw = svc.search(name, vec, limit=limit, subject=subject)
    return SearchResponse(collection_name=name, query_type="text", results=_build_results(raw))


@router.post("/collections/{name}/search", response_model=SearchResponse)
async def search_image(
    name: str,
    image: UploadFile = File(...),
    limit: int = Form(10),
    subject: Optional[str] = Form(None),
):
    svc = _ensure_milvus()
    if not svc.client.has_collection(collection_name=name):
        raise HTTPException(status_code=404, detail=f"集合 {name} 不存在")
    subject = _validate_subject(subject)
    raw = await image.read()
    pil = Image.open(BytesIO(raw))
    emb = EmbeddingService()
    vec = await emb.get_image_embedding(pil)
    raw_results = svc.search(name, vec, limit=limit, subject=subject)
    return SearchResponse(collection_name=name, query_type="image", results=_build_results(raw_results))


# ---- 取图资源 ----

def _file_response(full: Path, ext: str):
    return FileResponse(path=str(full), media_type=f"image/{ext}", filename=full.name)


@router.get("/asset")
async def get_asset(path: str = Query(..., description="相对 data/ 的路径")):
    from api import is_safe_path, DATA_DIR
    if not is_safe_path(DATA_DIR, path):
        raise HTTPException(status_code=403, detail="Access denied")
    full = DATA_DIR / path
    if not full.exists() or not full.is_file():
        raise HTTPException(status_code=404, detail="资源不存在")
    ext = full.suffix[1:].lower()
    return _file_response(full, ext)
