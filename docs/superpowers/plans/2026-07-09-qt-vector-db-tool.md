# Qt 向量数据库操作工具 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 ProcessGen 后端新增向量库端点（基于 refrance/ 参考），并提供一个 PySide6 瘦客户端操作它们（添加文本/图像、检索、集合管理）。

**Architecture:** 后端新增聚焦的 `/api/v1/rag/*` 路由，向量生成复用已有 `EmbeddingService` 单例，Milvus 操作由新增 `MilvusService` 单例负责，`TextProcessor` 从 `refrance/` 移入主包。Qt 客户端为独立子目录 `qt_tool/`，用 `requests` + `QThreadPool` worker 调后端 HTTP，侧边导航 + 多页面布局。

**Tech Stack:** FastAPI, pymilvus, PyMuPDF, Pydantic (后端)；PySide6, requests (Qt 客户端)

## Global Constraints

- Python >=3.11
- 后端复用 `EmbeddingService` 单例生成向量，不在新代码里再加载嵌入模型
- 集合 schema 固定：`id(INT64,auto_id)`, `embedding(FLOAT_VECTOR,dim=EmbeddingService.get_dimension(),COSINE)`, `type(VARCHAR16)`, `path(VARCHAR1024)`, `text(VARCHAR65535)`, `subject(VARCHAR64)`，`enable_dynamic_field=True`
- 文件存储分集合：文本 `data/text/{collection}/{ts}.{ext}`，图片 `data/images/{collection}/{ts}_{i}.{ext}`
- Qt 客户端配置存 `~/.moyu_processgen_ui/config.json`，至少含 `backend_url`
- Qt 客户端不碰 pymilvus/模型/GPU，只通过 HTTP 调后端
- Milvus 连接失败不阻断后端启动，调到 rag 端点时才报 503
- 沿用 `api.py` 的 `HTTPException` + 全局 500 异常处理风格
- 提交信息用 `feat:`/`fix:`/`docs:`/`test:`/`chore:` 前缀

---

## File Structure

后端：
- `config.py` — 新增 `milvus_uri`, `rag_subject_default` 两个配置项
- `models.py` — 新增 rag 相关 Pydantic 模型（追加到文件末尾）
- `text_processor.py` — 新建，从 `refrance/text.py` 移入 `TextProcessor` 类（适配导入）
- `milvus_service.py` — 新建，`MilvusService` 单例
- `rag_router.py` — 新建，`/api/v1/rag/*` 路由
- `api.py` — 挂载 rag_router
- `test_api.py` — 追加 rag 端点测试函数
- `pyproject.toml` — 加 `pymilvus`, `PyMuPDF`
- `.env.example` — 加 `MILVUS_URI`, `RAG_SUBJECT_DEFAULT`
- `README.md` — 更新 API 文档

Qt 客户端（`qt_tool/`）：
- `qt_tool/requirements.txt` — PySide6, requests
- `qt_tool/config.py` — 配置读写
- `qt_tool/backend_client.py` — `BackendClient` 纯函数 HTTP 封装
- `qt_tool/workers.py` — `ApiWorker` (QRunnable + 信号)
- `qt_tool/widgets/main_window.py` — 主窗口
- `qt_tool/widgets/collection_page.py`
- `qt_tool/widgets/add_text_page.py`
- `qt_tool/widgets/add_image_page.py`
- `qt_tool/widgets/search_page.py`
- `qt_tool/widgets/settings_dialog.py`
- `qt_tool/main.py` — 入口

---

## Task 1: 移入 TextProcessor 并加依赖

**Files:**
- Create: `text_processor.py`
- Modify: `pyproject.toml`
- Modify: `.env.example`

**Interfaces:**
- Produces: `TextProcessor` 类（`extract_text_from_pdf(pdf_path) -> str`, `split_markdown_for_rag(markdown_text, min_words=50, include_subsections=True) -> list[str]`）

- [ ] **Step 1: 创建 text_processor.py**

从 `refrance/text.py` 复制 `TextProcessor` 类（`extract_text_from_pdf` + `split_markdown_for_rag`），去掉文件末尾的 `if __name__ == "__main__":` 测试块。`fitz` 导入保留（来自 PyMuPDF）。

```python
import fitz
import re
from typing import List


class TextProcessor(object):

    def __init__(self):
        pass

    def extract_text_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text

    def split_markdown_for_rag(
        self,
        markdown_text: str,
        min_words: int = 50,
        include_subsections: bool = True
    ) -> List[str]:
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        lines = [line.rstrip() for line in markdown_text.splitlines() if line.strip() or line == '\n']
        sections = []
        current_section = None
        for line in lines:
            match = heading_pattern.match(line)
            if match:
                if current_section is not None:
                    sections.append(current_section)
                level = len(match.group(1))
                heading = match.group(2).strip()
                current_section = {'level': level, 'heading': heading, 'lines': [line]}
            else:
                if current_section is None:
                    current_section = {'level': 0, 'heading': 'Introduction', 'lines': [line]}
                else:
                    current_section['lines'].append(line)
        if current_section is not None:
            sections.append(current_section)
        chunks = []
        for i, sec in enumerate(sections):
            if sec['level'] == 0 and not sec['lines']:
                continue
            path = []
            for j in range(i, -1, -1):
                if sections[j]['level'] < sec['level']:
                    path.append(sections[j]['heading'])
                elif sections[j]['level'] == sec['level'] and j < i:
                    break
            path.reverse()
            path.append(sec['heading'])
            title_path = " > ".join(path) if path else "Untitled"
            content_lines = sec['lines'][1:]
            if include_subsections:
                child_lines = []
                for j in range(i + 1, len(sections)):
                    if sections[j]['level'] <= sec['level']:
                        break
                    child_lines.extend(sections[j]['lines'])
                content_lines.extend(child_lines)
            content = '\n'.join(content_lines).strip()
            chunk_text = f"Section: {title_path}\n\n{content}".strip()
            if len(chunk_text.split()) >= min_words or i == 0:
                chunks.append(chunk_text)
            else:
                if chunks:
                    chunks[-1] += "\n\n" + chunk_text
                else:
                    chunks.append(chunk_text)
        return chunks
```

- [ ] **Step 2: 加后端依赖到 pyproject.toml**

在 `dependencies` 列表末尾（`uvicorn>=0.40.0` 后）加两行：

```toml
    "pymilvus>=2.5.0",
    "PyMuPDF>=1.24.0",
```

- [ ] **Step 3: 加环境变量到 .env.example**

在文件末尾追加：

```
# Vector Database Configuration
MILVUS_URI=http://localhost:19530

# RAG Configuration
RAG_SUBJECT_DEFAULT=capp
```

- [ ] **Step 4: 安装依赖验证导入**

Run: `cd /home/jdy/Documents/Github/ProcessGenBackend && uv sync && uv run python -c "from text_processor import TextProcessor; p=TextProcessor(); print(len(p.split_markdown_for_rag('# T\n\nhello world text here\n\n## S\n\nmore text words here', min_words=5, include_subsections=False)))"`
Expected: 输出一个数字（分块数量），无 ImportError

- [ ] **Step 5: Commit**

```bash
git add text_processor.py pyproject.toml .env.example uv.lock
git commit -m "feat: 移入 TextProcessor 并加 pymilvus/PyMuPDF 依赖"
```

---

## Task 2: 扩展 config.py 与 models.py

**Files:**
- Modify: `config.py:11-25` (在 `comfyui_timeout` 字段后加新字段)
- Modify: `models.py` (文件末尾追加)

**Interfaces:**
- Produces: `config.milvus_uri` (str), `config.rag_subject_default` (str)
- Produces: Pydantic 模型 `CreateCollectionRequest`, `CollectionInfo`, `CollectionListResponse`, `AddTextResponse`, `AddImageResponse`, `SearchResultItem`, `SearchResponse`

- [ ] **Step 1: config.py 加两个字段**

在 `config.py` 的 `comfyui_timeout` 字段后（第 30 行 `class Config:` 之前）加：

```python
    milvus_uri: str = Field(
        default="http://localhost:19530", description="Milvus 服务器地址"
    )
    rag_subject_default: str = Field(
        default="capp", description="RAG 默认 subject 分区标签"
    )
```

- [ ] **Step 2: models.py 末尾追加 rag 模型**

在 `models.py` 末尾追加：

```python
class CreateCollectionRequest(BaseModel):
    collection_name: str = Field(description="集合名称")


class CollectionInfo(BaseModel):
    name: str = Field(description="集合名称")
    row_count: Optional[int] = Field(default=None, description="实体数")
    loaded: Optional[bool] = Field(default=None, description="是否已加载")


class CollectionListResponse(BaseModel):
    collections: list[CollectionInfo] = Field(description="集合列表")
    count: int = Field(description="集合总数")


class AddTextResponse(BaseModel):
    status: str = Field(description="状态")
    collection_name: str = Field(description="集合名称")
    chunks_inserted: int = Field(description="插入的文本块数")
    saved_path: str = Field(description="保存的源文件绝对路径")


class AddImageResponse(BaseModel):
    status: str = Field(description="状态")
    collection_name: str = Field(description="集合名称")
    images_inserted: int = Field(description="插入的图像数")


class SearchResultItem(BaseModel):
    id: int = Field(description="实体ID")
    score: float = Field(description="相似度分数")
    type: str = Field(description="text / image")
    text: str = Field(description="文本块内容或图像描述")
    path: str = Field(description="源文件绝对路径")
    subject: str = Field(description="分区标签")
    asset_path: Optional[str] = Field(
        default=None, description="相对 data/ 的路径，用于取图（图片结果有值）"
    )


class SearchResponse(BaseModel):
    collection_name: str = Field(description="集合名称")
    query_type: str = Field(description="查询类型: text / image")
    results: list[SearchResultItem] = Field(description="检索结果，按相似度排序")
```

- [ ] **Step 3: 验证导入**

Run: `cd /home/jdy/Documents/Github/ProcessGenBackend && uv run python -c "from config import config; print(config.milvus_uri, config.rag_subject_default); from models import CreateCollectionRequest, CollectionListResponse, AddTextResponse, AddImageResponse, SearchResponse, SearchResultItem, CollectionInfo; print('ok')"`
Expected: 打印 `http://localhost:19530 capp` 然后 `ok`

- [ ] **Step 4: Commit**

```bash
git add config.py models.py
git commit -m "feat: config 加 milvus_uri/rag_subject_default，models 加 rag 模型"
```

---

## Task 3: MilvusService 单例

**Files:**
- Create: `milvus_service.py`
- Test: `test_milvus_service.py`

**Interfaces:**
- Consumes: `config.milvus_uri`，`EmbeddingService` (仅 `get_dimension()` 取维度)
- Produces: `MilvusService` 单例类，方法：
  - `init_collection(name: str) -> bool` — 建表并加载，已存在则返回 True
  - `list_collections() -> list[CollectionInfo]` — 返回集合信息列表
  - `drop_collection(name: str) -> None`
  - `insert(name: str, data: list[dict]) -> list` — 插入实体，返回 insert 结果
  - `search(name: str, vector: list[float], limit: int = 10, subject: str | None = None) -> list[dict]` — 检索，返回 `[{id, score, type, text, path, subject, asset_path?}]`
  - `is_available() -> bool` — 连接是否可用

- [ ] **Step 1: 写失败测试**

创建 `test_milvus_service.py`：

```python
import pytest
from unittest.mock import patch, MagicMock
from milvus_service import MilvusService


def test_singleton_returns_same_instance():
    a = MilvusService()
    b = MilvusService()
    assert a is b


def test_init_collection_calls_create_when_absent():
    svc = MilvusService.__new__(MilvusService)
    svc.client = MagicMock()
    svc.client.has_collection.return_value = False
    svc.client.create_schema.return_value = MagicMock()
    svc.client.prepare_index_params.return_value = MagicMock()
    with patch("milvus_service.EmbeddingService") as EmbMock:
        EmbMock.return_value.get_dimension.return_value = 1024
        result = svc.init_collection("capp")
    svc.client.create_collection.assert_called_once()
    assert result is True


def test_init_collection_returns_true_when_exists():
    svc = MilvusService.__new__(MilvusService)
    svc.client = MagicMock()
    svc.client.has_collection.return_value = True
    svc.client.get_load_state.return_value = True
    result = svc.init_collection("capp")
    assert result is True
    svc.client.create_collection.assert_not_called()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /home/jdy/Documents/Github/ProcessGenBackend && uv run pytest test_milvus_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'milvus_service'`

- [ ] **Step 3: 实现 milvus_service.py**

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /home/jdy/Documents/Github/ProcessGenBackend && uv run pytest test_milvus_service.py -v`
Expected: PASS（3 个测试）

- [ ] **Step 5: Commit**

```bash
git add milvus_service.py test_milvus_service.py
git commit -m "feat: MilvusService 单例，建表/列出/删除/插入/检索"
```

---

## Task 4: rag_router.py 路由

**Files:**
- Create: `rag_router.py`

**Interfaces:**
- Consumes: `MilvusService`, `EmbeddingService`, `TextProcessor`, `models.*`, `config.config`
- Produces: `router` (FastAPI APIRouter，前缀 `/api/v1/rag`)

- [ ] **Step 1: 创建 rag_router.py**

```python
import os
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
from api import is_safe_path, DATA_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])

TEXT_DIR = Path("data/text").resolve()
IMAGE_DIR = Path("data/images").resolve()
ALLOWED_TEXT_EXT = {".md", ".txt", ".pdf"}
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}


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

    save_dir = IMAGE_DIR.joinpath(name)
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    emb = EmbeddingService()
    subj = subject or config.rag_subject_default
    data = []
    for i, (img_file, desc) in enumerate(zip(images, descriptions)):
        ext = Path(img_file.filename or ".png").suffix.lower()
        if ext not in ALLOWED_IMAGE_EXT:
            raise HTTPException(status_code=400, detail=f"不支持图像类型 {ext}")
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
    if not is_safe_path(DATA_DIR, path):
        raise HTTPException(status_code=403, detail="Access denied")
    full = DATA_DIR / path
    if not full.exists() or not full.is_file():
        raise HTTPException(status_code=404, detail="资源不存在")
    ext = full.suffix[1:].lower()
    return _file_response(full, ext)
```

- [ ] **Step 2: 验证路由可导入**

Run: `cd /home/jdy/Documents/Github/ProcessGenBackend && uv run python -c "from rag_router import router; print(len(router.routes))"`
Expected: 打印一个数字（≥7），无 ImportError

- [ ] **Step 3: Commit**

```bash
git add rag_router.py
git commit -m "feat: rag_router /api/v1/rag/* 端点（集合管理/添加文本/添加图像/检索/取图）"
```

---

## Task 5: 挂载路由到 api.py

**Files:**
- Modify: `api.py` (import + app.include_router)

**Interfaces:**
- Produces: 后端 `app` 暴露 `/api/v1/rag/*`

- [ ] **Step 1: api.py 顶部加导入**

在 `api.py` 现有 `from comfyui_service import ComfyUIService` 行后加：

```python
from rag_router import router as rag_router
```

- [ ] **Step 2: app 创建后挂载路由**

在 `api.py` 的 `app = FastAPI(...)` 块之后、`app.add_middleware(...)` 之前加：

```python
app.include_router(rag_router)
```

- [ ] **Step 3: 验证 app 路由含 rag 前缀**

Run: `cd /home/jdy/Documents/Github/ProcessGenBackend && uv run python -c "from api import app; paths=[r.path for r in app.routes if hasattr(r,'path')]; print([p for p in paths if '/rag/' in p])"`
Expected: 打印含 `/api/v1/rag/collections` 等路径的列表

- [ ] **Step 4: Commit**

```bash
git add api.py
git commit -m "feat: api.py 挂载 rag_router"
```

---

## Task 6: 后端 rag 端点集成测试

**Files:**
- Modify: `test_api.py` (末尾追加测试函数)

**Interfaces:**
- Consumes: 后端运行在 `localhost:8050`，Milvus 可达（沿用 `test_api.py` 顶部 `BASE_URL`）

- [ ] **Step 1: 追加测试函数到 test_api.py**

在 `test_api.py` 末尾追加：

```python
def test_rag_collections_lifecycle():
    print("\n=== Testing RAG Collections Lifecycle ===")
    name = f"qt_test_{int(__import__('time').time())}"
    # 创建
    r = requests.post(f"{BASE_URL}/api/v1/rag/collections", json={"collection_name": name})
    assert r.status_code == 201, r.text
    # 列出
    r = requests.get(f"{BASE_URL}/api/v1/rag/collections")
    assert r.status_code == 200
    names = [c["name"] for c in r.json()["collections"]]
    assert name in names
    # 删除
    r = requests.delete(f"{BASE_URL}/api/v1/rag/collections/{name}")
    assert r.status_code == 200
    print("✓ RAG collections lifecycle passed")


def test_rag_add_text_and_search():
    print("\n=== Testing RAG Add Text + Search ===")
    name = f"qt_text_{int(__import__('time').time())}"
    requests.post(f"{BASE_URL}/api/v1/rag/collections", json={"collection_name": name})
    md = "# 工序一\n\n加工内表面螺纹孔，注意扭矩控制。\n\n更多内容文字以确保达到最小词数。"
    files = {"file": ("doc.md", md.encode("utf-8"), "text/markdown")}
    r = requests.post(f"{BASE_URL}/api/v1/rag/collections/{name}/text", files=files)
    assert r.status_code == 200, r.text
    assert r.json()["chunks_inserted"] > 0
    # 检索
    r = requests.get(f"{BASE_URL}/api/v1/rag/collections/{name}/search", params={"query": "螺纹孔", "limit": 3})
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert len(results) > 0
    assert "score" in results[0]
    requests.delete(f"{BASE_URL}/api/v1/rag/collections/{name}")
    print("✓ RAG add text + search passed")


def test_rag_add_image_and_search_and_asset():
    print("\n=== Testing RAG Add Image + Search + Asset ===")
    name = f"qt_img_{int(__import__('time').time())}"
    requests.post(f"{BASE_URL}/api/v1/rag/collections", json={"collection_name": name})
    img = Image.new("RGB", (64, 64), color="green")
    buf = BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    files = {"images": ("g.png", buf, "image/png")}
    data = {"descriptions": "一张绿色测试图", "subject": "capp"}
    r = requests.post(f"{BASE_URL}/api/v1/rag/collections/{name}/images", files=files, data=data)
    assert r.status_code == 200, r.text
    assert r.json()["images_inserted"] == 1
    # 图像检索
    buf2 = BytesIO(); img.save(buf2, format="PNG"); buf2.seek(0)
    files2 = {"image": ("g.png", buf2, "image/png")}
    r = requests.post(f"{BASE_URL}/api/v1/rag/collections/{name}/search", files=files2, data={"limit": "3"})
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert len(results) > 0
    assert results[0]["type"] == "image"
    assert results[0]["asset_path"]
    # asset 取图
    r = requests.get(f"{BASE_URL}/api/v1/rag/asset", params={"path": results[0]["asset_path"]})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")
    requests.delete(f"{BASE_URL}/api/v1/rag/collections/{name}")
    print("✓ RAG add image + search + asset passed")


if __name__ == "__main__":
    test_health()
    test_rag_collections_lifecycle()
    test_rag_add_text_and_search()
    test_rag_add_image_and_search_and_asset()
```

注：若 `test_api.py` 末尾已有 `if __name__ == "__main__":` 块，把上述三个 `test_rag_*` 函数追加到该块**之前**，并把这三个函数调用加进该 `__main__` 块里。

- [ ] **Step 2: 启动后端并跑测试**

需先确保 Milvus 服务可达（`MILVUS_URI` 指向已运行实例）。启动后端：

Run: `cd /home/jdy/Documents/Github/ProcessGenBackend && uv run python run_api.py &` （后台；等模型加载日志出现 `Uvicorn running`）

Run: `cd /home/jdy/Documents/Github/ProcessGenBackend && uv run python test_api.py`
Expected: 末尾打印三行 `✓ RAG ... passed`

- [ ] **Step 3: 停止后端**

Run: `pkill -f "run_api.py"` （或对应进程）
Expected: 后端进程结束

- [ ] **Step 4: Commit**

```bash
git add test_api.py
git commit -m "test: rag 端点集成测试（集合生命周期/文本/图像/检索/取图）"
```

---

## Task 7: Qt 客户端依赖与 config

**Files:**
- Create: `qt_tool/requirements.txt`
- Create: `qt_tool/config.py`

**Interfaces:**
- Produces: `load_config() -> dict`，`save_config(dict) -> None`，配置路径 `~/.moyu_processgen_ui/config.json`，默认 `{"backend_url": "http://localhost:8050"}`

- [ ] **Step 1: 创建 requirements.txt**

```
PySide6>=6.6.0
requests>=2.31.0
```

- [ ] **Step 2: 创建 config.py**

```python
import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".moyu_processgen_ui"
CONFIG_FILE = CONFIG_DIR / "config.json"
DEFAULT_CONFIG = {"backend_url": "http://localhost:8050"}


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    try:
        cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        cfg = dict(DEFAULT_CONFIG)
    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg)
    return merged


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **Step 3: 验证 config 读写**

Run: `cd /home/jdy/Documents/Github/ProcessGenBackend/qt_tool && python3 -c "from config import load_config, save_config; c=load_config(); print(c); save_config({'backend_url':'http://x:1'}); print(load_config()); save_config({'backend_url':'http://localhost:8050'})"`
Expected: 打印 `{'backend_url': 'http://localhost:8050'}`（首次可能写入）、`{'backend_url': 'http://x:1'}`，并恢复默认

- [ ] **Step 4: Commit**

```bash
git add qt_tool/requirements.txt qt_tool/config.py
git commit -m "feat(qt): requirements 与 config 读写"
```

---

## Task 8: BackendClient

**Files:**
- Create: `qt_tool/backend_client.py`

**Interfaces:**
- Consumes: 后端 `/api/v1/rag/*`、`/health`
- Produces: `BackendClient` 类（方法见下）

- [ ] **Step 1: 创建 backend_client.py**

```python
import os
import requests


class BackendError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{status_code}] {detail}")


class BackendClient:
    def __init__(self, base_url: str = "http://localhost:8050", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def set_base_url(self, url: str):
        self.base_url = url.rstrip("/")

    def _check(self, resp: requests.Response):
        if not resp.ok:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise BackendError(resp.status_code, str(detail))
        return resp

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        return self._check(r).json()

    def list_collections(self) -> dict:
        r = requests.get(f"{self.base_url}/api/v1/rag/collections", timeout=self.timeout)
        return self._check(r).json()

    def create_collection(self, name: str) -> dict:
        r = requests.post(f"{self.base_url}/api/v1/rag/collections",
                          json={"collection_name": name}, timeout=self.timeout)
        return self._check(r).json()

    def delete_collection(self, name: str) -> dict:
        r = requests.delete(f"{self.base_url}/api/v1/rag/collections/{name}", timeout=self.timeout)
        return self._check(r).json()

    def add_text(self, name: str, file_path: str, subject: str = "capp") -> dict:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            data = {"subject": subject}
            r = requests.post(f"{self.base_url}/api/v1/rag/collections/{name}/text",
                              files=files, data=data, timeout=self.timeout)
        return self._check(r).json()

    def add_images(self, name: str, image_paths: list[str], descriptions: list[str],
                   subject: str = "capp") -> dict:
        assert len(image_paths) == len(descriptions), "image_paths 与 descriptions 数量不一致"
        files = []
        opened = []
        try:
            for p in image_paths:
                fh = open(p, "rb")
                opened.append(fh)
                files.append(("images", (os.path.basename(p), fh)))
            data = [("descriptions", d) for d in descriptions]
            data.append(("subject", subject))
            r = requests.post(f"{self.base_url}/api/v1/rag/collections/{name}/images",
                              files=files, data=data, timeout=self.timeout)
        finally:
            for fh in opened:
                fh.close()
        return self._check(r).json()

    def search_text(self, name: str, query: str, limit: int = 10,
                    subject: str | None = None) -> dict:
        params = {"query": query, "limit": limit}
        if subject:
            params["subject"] = subject
        r = requests.get(f"{self.base_url}/api/v1/rag/collections/{name}/search",
                         params=params, timeout=self.timeout)
        return self._check(r).json()

    def search_image(self, name: str, image_path: str, limit: int = 10,
                     subject: str | None = None) -> dict:
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f)}
            data = {"limit": str(limit)}
            if subject:
                data["subject"] = subject
            r = requests.post(f"{self.base_url}/api/v1/rag/collections/{name}/search",
                              files=files, data=data, timeout=self.timeout)
        return self._check(r).json()

    def get_asset(self, asset_path: str) -> bytes:
        r = requests.get(f"{self.base_url}/api/v1/rag/asset",
                         params={"path": asset_path}, timeout=self.timeout)
        return self._check(r).content
```

- [ ] **Step 2: 验证导入**

Run: `cd /home/jdy/Documents/Github/ProcessGenBackend/qt_tool && python3 -c "from backend_client import BackendClient, BackendError; c=BackendClient(); print(c.base_url)"`
Expected: 打印 `http://localhost:8050`，无错误

- [ ] **Step 3: Commit**

```bash
git add qt_tool/backend_client.py
git commit -m "feat(qt): BackendClient HTTP 封装"
```

---

## Task 9: ApiWorker

**Files:**
- Create: `qt_tool/workers.py`

**Interfaces:**
- Consumes: `BackendClient` 方法
- Produces: `ApiWorker(QRunnable)`，信号 `finished(object)`、`failed(str)`

- [ ] **Step 1: 创建 workers.py**

```python
from PySide6.QtCore import QObject, QRunnable, Signal


class _Signals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class ApiWorker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = _Signals()

    @property
    def finished(self):
        return self.signals.finished

    @property
    def failed(self):
        return self.signals.failed

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.failed.emit(str(e))
```

- [ ] **Step 2: 验证导入（需 PySide6）**

Run: `cd /home/jdy/Documents/Github/ProcessGenBackend/qt_tool && pip install -q PySide6 requests 2>/dev/null; python3 -c "from workers import ApiWorker; w=ApiWorker(lambda: 1); print('ok', hasattr(w.finished,'emit'), hasattr(w.failed,'emit'))"`
Expected: 打印 `ok True True`（若 PySide6 未装，先 `pip install PySide6`）

- [ ] **Step 3: Commit**

```bash
git add qt_tool/workers.py
git commit -m "feat(qt): ApiWorker 线程池 worker"
```

---

## Task 10: CollectionPage

**Files:**
- Create: `qt_tool/widgets/__init__.py`
- Create: `qt_tool/widgets/collection_page.py`

**Interfaces:**
- Consumes: `BackendClient`, `ApiWorker`, `QThreadPool`
- Produces: `CollectionPage(QWidget)`，信号 `collection_changed()`（集合列表变化时通知主窗口刷新下拉）

- [ ] **Step 1: 创建 __init__.py**

`qt_tool/widgets/__init__.py` 内容为空文件（占位）。

- [ ] **Step 2: 创建 collection_page.py**

```python
from PySide6.QtCore import Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QTableWidget, QTableWidgetItem, QLabel, QMessageBox,
)
from backend_client import BackendClient, BackendError
from workers import ApiWorker


class CollectionPage(QWidget):
    collection_changed = Signal()

    def __init__(self, client: BackendClient, pool: QThreadPool, parent=None):
        super().__init__(parent)
        self.client = client
        self.pool = pool
        self._build()
        self.refresh()

    def _build(self):
        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("新建集合:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("集合名称")
        bar.addWidget(self.name_edit)
        self.create_btn = QPushButton("创建")
        self.create_btn.clicked.connect(self._on_create)
        bar.addWidget(self.create_btn)
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.refresh)
        bar.addWidget(self.refresh_btn)
        layout.addLayout(bar)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["名称", "实体数", "已加载", "操作"])
        self.table.horizontalHeader().setStretchLastSection(False)
        layout.addWidget(self.table)
        self.status = QLabel("")
        layout.addWidget(self.status)

    def refresh(self):
        worker = ApiWorker(self.client.list_collections)
        worker.finished.connect(self._on_list)
        worker.failed.connect(self._on_err)
        self.pool.start(worker)

    def _on_list(self, result):
        self.table.setRowCount(0)
        for c in result.get("collections", []):
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(c.get("name", "")))
            self.table.setItem(row, 1, QTableWidgetItem(str(c.get("row_count", ""))))
            self.table.setItem(row, 2, QTableWidgetItem(str(c.get("loaded", ""))))
            del_btn = QPushButton("删除")
            name = c.get("name", "")
            del_btn.clicked.connect(lambda _=False, n=name: self._delete(n))
            self.table.setCellWidget(row, 3, del_btn)
        self.status.setText(f"共 {result.get('count', 0)} 个集合")

    def _on_create(self):
        name = self.name_edit.text().strip()
        if not name:
            return
        worker = ApiWorker(self.client.create_collection, name)
        worker.finished.connect(lambda r: (self.name_edit.clear(), self.refresh(), self.collection_changed.emit()))
        worker.failed.connect(self._on_err)
        self.pool.start(worker)

    def _delete(self, name: str):
        if QMessageBox.question(self, "确认", f"删除集合 {name}？") != QMessageBox.Yes:
            return
        worker = ApiWorker(self.client.delete_collection, name)
        worker.finished.connect(lambda r: (self.refresh(), self.collection_changed.emit()))
        worker.failed.connect(self._on_err)
        self.pool.start(worker)

    def _on_err(self, msg):
        self.status.setText(f"错误: {msg}")
```

- [ ] **Step 3: 验证导入**

Run: `cd /home/jdy/Documents/Github/ProcessGenBackend/qt_tool && python3 -c "from widgets.collection_page import CollectionPage; print('ok')"`
Expected: 打印 `ok`

- [ ] **Step 4: Commit**

```bash
git add qt_tool/widgets/__init__.py qt_tool/widgets/collection_page.py
git commit -m "feat(qt): CollectionPage 集合管理页面"
```

---

## Task 11: AddTextPage

**Files:**
- Create: `qt_tool/widgets/add_text_page.py`

**Interfaces:**
- Consumes: `BackendClient`, `ApiWorker`
- Produces: `AddTextPage(QWidget)`

- [ ] **Step 1: 创建 add_text_page.py**

```python
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel,
    QFileDialog,
)
from backend_client import BackendClient
from workers import ApiWorker


class AddTextPage(QWidget):
    def __init__(self, client: BackendClient, pool: QThreadPool, current_collection, parent=None):
        super().__init__(parent)
        self.client = client
        self.pool = pool
        self.current_collection = current_collection  # callable 返回当前选中集合名
        self.file_path = ""
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.file_btn = QPushButton("选择文件 (.md/.txt/.pdf)")
        self.file_btn.clicked.connect(self._pick)
        self.file_label = QLabel("未选择")
        row.addWidget(self.file_btn)
        row.addWidget(self.file_label, 1)
        layout.addLayout(row)

        srow = QHBoxLayout()
        srow.addWidget(QLabel("subject:"))
        self.subject_edit = QLineEdit("capp")
        srow.addWidget(self.subject_edit, 1)
        layout.addLayout(srow)

        self.submit_btn = QPushButton("添加到集合")
        self.submit_btn.clicked.connect(self._submit)
        layout.addWidget(self.submit_btn)
        self.status = QLabel("")
        layout.addWidget(self.status)
        layout.addStretch(1)

    def _pick(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择文本文件", "",
                                            "Documents (*.md *.txt *.pdf);;All (*.*)")
        if p:
            self.file_path = p
            self.file_label.setText(p)

    def _submit(self):
        name = self.current_collection()
        if not name:
            self.status.setText("请先在左侧选择集合")
            return
        if not self.file_path:
            self.status.setText("请先选择文件")
            return
        subject = self.subject_edit.text().strip() or "capp"
        self.status.setText("上传中...")
        worker = ApiWorker(self.client.add_text, name, self.file_path, subject)
        worker.finished.connect(self._on_done)
        worker.failed.connect(self._on_err)
        self.pool.start(worker)

    def _on_done(self, r):
        self.status.setText(f"成功: 插入 {r.get('chunks_inserted')} 块 -> {r.get('saved_path')}")

    def _on_err(self, msg):
        self.status.setText(f"错误: {msg}")
```

- [ ] **Step 2: 验证导入**

Run: `cd /home/jdy/Documents/Github/ProcessGenBackend/qt_tool && python3 -c "from widgets.add_text_page import AddTextPage; print('ok')"`
Expected: 打印 `ok`

- [ ] **Step 3: Commit**

```bash
git add qt_tool/widgets/add_text_page.py
git commit -m "feat(qt): AddTextPage 添加文本页面"
```

---

## Task 12: AddImagePage

**Files:**
- Create: `qt_tool/widgets/add_image_page.py`

**Interfaces:**
- Consumes: `BackendClient`, `ApiWorker`
- Produces: `AddImagePage(QWidget)`

- [ ] **Step 1: 创建 add_image_page.py**

```python
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
)
from backend_client import BackendClient
from workers import ApiWorker


class AddImagePage(QWidget):
    def __init__(self, client: BackendClient, pool: QThreadPool, current_collection, parent=None):
        super().__init__(parent)
        self.client = client
        self.pool = pool
        self.current_collection = current_collection
        self.items: list[tuple[str, str]] = []  # (path, description)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.add_btn = QPushButton("选择图片 (可多选)")
        self.add_btn.clicked.connect(self._pick)
        bar.addWidget(self.add_btn)
        self.subject_edit = QLineEdit("capp")
        bar.addWidget(QLabel("subject:"))
        bar.addWidget(self.subject_edit)
        layout.addLayout(bar)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["描述", "路径"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table)

        self.submit_btn = QPushButton("添加到集合")
        self.submit_btn.clicked.connect(self._submit)
        layout.addWidget(self.submit_btn)
        self.status = QLabel("")
        layout.addWidget(self.status)

    def _pick(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "选择图片", "",
                                             "Images (*.png *.jpg *.jpeg *.gif *.bmp *.webp)")
        for p in ps:
            self.items.append((p, ""))
        self._refresh_table()

    def _refresh_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for path, desc in self.items:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(desc))
            self.table.setItem(row, 1, QTableWidgetItem(path))
        self.table.blockSignals(False)

    def _on_item_changed(self, item):
        row = item.row()
        if item.column() == 0 and row < len(self.items):
            path, _ = self.items[row]
            self.items[row] = (path, item.text())

    def _submit(self):
        name = self.current_collection()
        if not name:
            self.status.setText("请先在左侧选择集合")
            return
        if not self.items:
            self.status.setText("请先选择图片")
            return
        paths = [p for p, _ in self.items]
        descs = [d for _, d in self.items]
        subject = self.subject_edit.text().strip() or "capp"
        self.status.setText("上传中...")
        worker = ApiWorker(self.client.add_images, name, paths, descs, subject)
        worker.finished.connect(self._on_done)
        worker.failed.connect(self._on_err)
        self.pool.start(worker)

    def _on_done(self, r):
        n = r.get("images_inserted", 0)
        self.status.setText(f"成功: 插入 {n} 张")
        self.items.clear()
        self._refresh_table()

    def _on_err(self, msg):
        self.status.setText(f"错误: {msg}")
```

- [ ] **Step 2: 验证导入**

Run: `cd /home/jdy/Documents/Github/ProcessGenBackend/qt_tool && python3 -c "from widgets.add_image_page import AddImagePage; print('ok')"`
Expected: 打印 `ok`

- [ ] **Step 3: Commit**

```bash
git add qt_tool/widgets/add_image_page.py
git commit -m "feat(qt): AddImagePage 添加图像页面"
```

---

## Task 13: SearchPage

**Files:**
- Create: `qt_tool/widgets/search_page.py`

**Interfaces:**
- Consumes: `BackendClient`, `ApiWorker`
- Produces: `SearchPage(QWidget)`

- [ ] **Step 1: 创建 search_page.py**

```python
from PySide6.QtCore import Qt, QThreadPool, QByteArray
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel,
    QComboBox, QSpinBox, QListWidget, QListWidgetItem, QFileDialog,
)
from backend_client import BackendClient
from workers import ApiWorker


class _ResultWidget(QWidget):
    def __init__(self, item: dict, client: BackendClient, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QHBoxLayout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.thumb = QLabel()
        self.thumb.setFixedSize(64, 64)
        layout.addWidget(self.thumb)
        text = (f"[{item.get('type')}] score={item.get('score', 0):.4f}\n"
                f"{item.get('text', '')}\n来源: {item.get('path', '')}")
        layout.addWidget(QLabel(text), 1)
        if item.get("type") == "image" and item.get("asset_path"):
            worker = ApiWorker(client.get_asset, item["asset_path"])
            worker.finished.connect(self._on_bytes)
            QThreadPool.globalInstance().start(worker)
        else:
            self.thumb.setText("TXT")

    def _on_bytes(self, data: bytes):
        pix = QPixmap()
        if pix.loadFromData(QByteArray(data)):
            self.thumb.setPixmap(pix.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))


class SearchPage(QWidget):
    def __init__(self, client: BackendClient, pool: QThreadPool, current_collection, parent=None):
        super().__init__(parent)
        self.client = client
        self.pool = pool
        self.current_collection = current_collection
        self.image_path = ""
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItems(["文本", "图像"])
        self.mode.currentIndexChanged.connect(self._on_mode)
        top.addWidget(self.mode)
        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText("输入查询文本")
        top.addWidget(self.query_edit, 1)
        self.image_btn = QPushButton("选择图像")
        self.image_btn.clicked.connect(self._pick_image)
        self.image_btn.setVisible(False)
        top.addWidget(self.image_btn)
        top.addWidget(QLabel("limit:"))
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(1, 100)
        self.limit_spin.setValue(10)
        top.addWidget(self.limit_spin)
        self.search_btn = QPushButton("检索")
        self.search_btn.clicked.connect(self._search)
        top.addWidget(self.search_btn)
        layout.addLayout(top)

        self.results = QListWidget()
        layout.addWidget(self.results, 1)
        self.status = QLabel("")
        layout.addWidget(self.status)

    def _on_mode(self, idx):
        is_text = idx == 0
        self.query_edit.setVisible(is_text)
        self.image_btn.setVisible(not is_text)

    def _pick_image(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择查询图像", "",
                                           "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if p:
            self.image_path = p
            self.image_btn.setText(p)

    def _search(self):
        name = self.current_collection()
        if not name:
            self.status.setText("请先在左侧选择集合")
            return
        limit = self.limit_spin.value()
        self.results.clear()
        self.status.setText("检索中...")
        if self.mode.currentIndex() == 0:
            q = self.query_edit.text().strip()
            if not q:
                self.status.setText("请输入查询文本")
                return
            worker = ApiWorker(self.client.search_text, name, q, limit, None)
        else:
            if not self.image_path:
                self.status.setText("请选择查询图像")
                return
            worker = ApiWorker(self.client.search_image, name, self.image_path, limit, None)
        worker.finished.connect(self._on_results)
        worker.failed.connect(self._on_err)
        self.pool.start(worker)

    def _on_results(self, r):
        self.results.clear()
        items = r.get("results", [])
        for it in items:
            wi = QListWidgetItem()
            wi.setSizeHint(_ResultWidget(it, self.client).sizeHint())
            self.results.addItem(wi)
            self.results.setItemWidget(wi, _ResultWidget(it, self.client))
        self.status.setText(f"返回 {len(items)} 条")

    def _on_err(self, msg):
        self.status.setText(f"错误: {msg}")
```

- [ ] **Step 2: 验证导入**

Run: `cd /home/jdy/Documents/Github/ProcessGenBackend/qt_tool && python3 -c "from widgets.search_page import SearchPage; print('ok')"`
Expected: 打印 `ok`

- [ ] **Step 3: Commit**

```bash
git add qt_tool/widgets/search_page.py
git commit -m "feat(qt): SearchPage 检索页面（文本/图像，图像缩略图）"
```

---

## Task 14: SettingsDialog + MainWindow + main.py

**Files:**
- Create: `qt_tool/widgets/settings_dialog.py`
- Create: `qt_tool/widgets/main_window.py`
- Create: `qt_tool/main.py`

**Interfaces:**
- Consumes: 所有页面、`BackendClient`、`config`
- Produces: 可运行的应用入口 `python main.py`

- [ ] **Step 1: 创建 settings_dialog.py**

```python
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton


class SettingsDialog(QDialog):
    def __init__(self, backend_url: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("后端 URL:"))
        self.url_edit = QLineEdit(backend_url)
        layout.addWidget(self.url_edit)
        row = QHBoxLayout()
        ok = QPushButton("保存")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        row.addStretch(1)
        row.addWidget(ok)
        row.addWidget(cancel)
        layout.addLayout(row)

    def url(self) -> str:
        return self.url_edit.text().strip()
```

- [ ] **Step 2: 创建 main_window.py**

```python
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem,
    QStackedWidget, QLabel, QPushButton, QComboBox, QMessageBox,
)
from backend_client import BackendClient
from config import load_config, save_config
from widgets.settings_dialog import SettingsDialog
from widgets.collection_page import CollectionPage
from widgets.add_text_page import AddTextPage
from widgets.add_image_page import AddImagePage
from widgets.search_page import SearchPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ProcessGen 向量库工具")
        self.resize(1000, 680)
        cfg = load_config()
        self.client = BackendClient(cfg["backend_url"])
        self.pool = QThreadPool(self)

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        # 左侧
        left = QVBoxLayout()
        left.addWidget(QLabel("集合:"))
        self.collection_combo = QComboBox()
        self.collection_combo.currentTextChanged.connect(self._on_collection_changed)
        left.addWidget(self.collection_combo)
        self.nav = QListWidget()
        for label in ["添加文本", "添加图像", "检索", "集合管理"]:
            QListWidgetItem(label, self.nav)
        self.nav.setCurrentRow(0)
        left.addWidget(self.nav)
        self.health_dot = QLabel("后端状态: ?")
        left.addWidget(self.health_dot)
        self.settings_btn = QPushButton("设置")
        self.settings_btn.clicked.connect(self._open_settings)
        left.addWidget(self.settings_btn)
        root.addLayout(left, 0)

        # 右侧
        self.stack = QStackedWidget()
        self.add_text_page = AddTextPage(self.client, self.pool, self._current_collection)
        self.add_image_page = AddImagePage(self.client, self.pool, self._current_collection)
        self.search_page = SearchPage(self.client, self.pool, self._current_collection)
        self.collection_page = CollectionPage(self.client, self.pool)
        self.collection_page.collection_changed.connect(self.refresh_collections)
        self.stack.addWidget(self.add_text_page)
        self.stack.addWidget(self.add_image_page)
        self.stack.addWidget(self.search_page)
        self.stack.addWidget(self.collection_page)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        root.addWidget(self.stack, 1)

        self.setCentralWidget(central)
        self.refresh_collections()
        self._check_health()

    def _current_collection(self) -> str:
        return self.collection_combo.currentText()

    def _on_collection_changed(self, _name):
        pass

    def refresh_collections(self):
        from workers import ApiWorker
        worker = ApiWorker(self.client.list_collections)
        worker.finished.connect(self._on_collections)
        worker.failed.connect(lambda m: self.health_dot.setText(f"集合刷新错误: {m}"))
        self.pool.start(worker)

    def _on_collections(self, result):
        prev = self.collection_combo.currentText()
        self.collection_combo.blockSignals(True)
        self.collection_combo.clear()
        for c in result.get("collections", []):
            self.collection_combo.addItem(c.get("name", ""))
        if prev and self.collection_combo.findText(prev) >= 0:
            self.collection_combo.setCurrentText(prev)
        self.collection_combo.blockSignals(False)

    def _check_health(self):
        from workers import ApiWorker
        worker = ApiWorker(self.client.health)
        worker.finished.connect(lambda r: self.health_dot.setText(
            f"后端状态: ● {r.get('status', '?')}"))
        worker.failed.connect(lambda m: self.health_dot.setText(f"后端状态: ✗ {m}"))
        self.pool.start(worker)

    def _open_settings(self):
        dlg = SettingsDialog(self.client.base_url, self)
        if dlg.exec():
            url = dlg.url()
            self.client.set_base_url(url)
            save_config({"backend_url": url})
            self._check_health()
            self.refresh_collections()
```

- [ ] **Step 3: 创建 main.py**

```python
import sys
from PySide6.QtWidgets import QApplication
from widgets.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 验证应用能启动（不连后端也不应崩溃）**

Run: `cd /home/jdy/Documents/Github/ProcessGenBackend/qt_tool && QT_QPA_PLATFORM=offscreen python3 -c "from PySide6.QtWidgets import QApplication; import sys; app=QApplication(sys.argv); from widgets.main_window import MainWindow; w=MainWindow(); print('window ok', w.windowTitle())"`
Expected: 打印 `window ok ProcessGen 向量库工具`，无异常（offscreen 模式下窗口构造成功即可）

- [ ] **Step 5: Commit**

```bash
git add qt_tool/widgets/settings_dialog.py qt_tool/widgets/main_window.py qt_tool/main.py
git commit -m "feat(qt): SettingsDialog + MainWindow + 入口"
```

---

## Task 15: 更新 README 接口文档

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 在 README 的"重排服务"之后插入 rag 端点文档**

在 `README.md` 的"### 重排服务"区块之后、"### 图像生成服务 (ComfyUI)"之前，插入：

````markdown
### 向量数据库服务 (RAG)

> **注意**: 需先启动 Milvus 服务并配置 `MILVUS_URI`

#### 创建集合

```http
POST /api/v1/rag/collections
Content-Type: application/json
```

**请求体:**

```json
{"collection_name": "capp"}
```

**响应示例:**

```json
{"name": "capp", "row_count": null, "loaded": true}
```

#### 列出集合

```http
GET /api/v1/rag/collections
```

**响应示例:**

```json
{"collections": [{"name": "capp", "row_count": 12, "loaded": true}], "count": 1}
```

#### 删除集合

```http
DELETE /api/v1/rag/collections/{name}
```

删除集合及其 `data/text/{name}` 与 `data/images/{name}` 目录。

#### 添加文本

```http
POST /api/v1/rag/collections/{name}/text
Content-Type: multipart/form-data
```

**参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | file | 是 | `.md` / `.txt` / `.pdf` 文件 |
| `subject` | string | 否 | 分区标签，默认 `RAG_SUBJECT_DEFAULT` |

**响应示例:**

```json
{"status": "success", "collection_name": "capp", "chunks_inserted": 5, "saved_path": "/abs/data/text/capp/1234.md"}
```

#### 添加图像

```http
POST /api/v1/rag/collections/{name}/images
Content-Type: multipart/form-data
```

**参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `images` | file[] | 是 | 可重复多个图像文件 |
| `descriptions` | string[] | 是 | 与图片一一对应的描述 |
| `subject` | string | 否 | 分区标签 |

描述非空走融合嵌入，为空走纯图像嵌入。

**响应示例:**

```json
{"status": "success", "collection_name": "capp", "images_inserted": 2}
```

#### 检索（文本）

```http
GET /api/v1/rag/collections/{name}/search?query=文本&limit=10&subject=capp
```

#### 检索（图像）

```http
POST /api/v1/rag/collections/{name}/search
Content-Type: multipart/form-data
```

**参数:** `image`(file)、`limit`(int)、`subject`(string 可选)

**响应示例（文本/图像检索通用）:**

```json
{
  "collection_name": "capp",
  "query_type": "text",
  "results": [
    {"id": 1, "score": 0.82, "type": "text", "text": "...", "path": "/abs/...", "subject": "capp", "asset_path": null},
    {"id": 2, "score": 0.71, "type": "image", "text": "描述", "path": "/abs/...", "subject": "capp", "asset_path": "images/capp/1_0.png"}
  ]
}
```

#### 取图资源

```http
GET /api/v1/rag/asset?path=images/capp/1_0.png
```

返回图像文件（`image/<ext>`），`path` 必须为相对 `data/` 的安全路径，越界返回 403。

````

- [ ] **Step 2: 在 README "项目结构" 区块补充 rag 文件与 qt_tool 说明**

在"项目结构"代码块内，`comfyui_service.py` 行后加：

```
├── rag_router.py       # /api/v1/rag/* 向量库端点
├── milvus_service.py   # Milvus 单例服务
├── text_processor.py   # Markdown/PDF 分块
```

并在"项目结构"代码块后追加一段：

```markdown
### Qt 客户端

`qt_tool/` 为独立 PySide6 客户端，通过 HTTP 调用上述后端操作向量库。运行：

```bash
cd qt_tool
pip install -r requirements.txt
python main.py
```

首次运行会在 `~/.moyu_processgen_ui/config.json` 保存后端地址，可在界面"设置"中修改。
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README 补充 rag 端点与 qt_tool 客户端说明"
```

---

## Self-Review 记录

**1. Spec 覆盖:**
- 复用 EmbeddingService ✓（Task 3/4 用 `EmbeddingService().get_dimension()` 和各 get_*）
- 集合 schema 固定字段 ✓（Task 3 init_collection）
- 文件存储分集合 ✓（Task 4 `data/text/{name}`、`data/images/{name}`）
- Qt 配置 `~/.moyu_processgen_ui/config.json` ✓（Task 7）
- Milvus 连接失败不阻断启动、调到才报 503 ✓（Task 4 `_ensure_milvus`）
- `asset_path` + 取图端点 ✓（Task 3 search 生成、Task 4 get_asset）
- 图像检索结果缩略图 ✓（Task 13 `_ResultWidget`）
- 错误处理清单 ✓（Task 4 各 HTTPException）
- 后端配置项 + `.env.example` ✓（Task 1/2）
- 测试 ✓（Task 6）
- README 接口文档 ✓（Task 15）

**2. Placeholder 扫描:** Task 4 `get_asset` 与 `_file_response` 在 Step 1 给出完整最终代码（顶部已含 `from fastapi.responses import FileResponse`），无遗留 TBD/描述式步骤。

**3. Type 一致性:** `BackendClient` 方法名（`list_collections/create_collection/delete_collection/add_text/add_images/search_text/search_image/get_asset/health`）与各页面调用、SearchPage 缩略图取图一致；`SearchResultItem.asset_path` 在后端 search 生成、前端读取一致；`collection_changed` 信号在 CollectionPage 发出、MainWindow 接收一致。
