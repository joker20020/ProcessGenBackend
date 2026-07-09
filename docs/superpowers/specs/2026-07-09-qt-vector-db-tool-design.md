# Qt 向量数据库操作工具 — 设计文档

- 日期: 2026-07-09
- 状态: 已确认设计，待实现
- 范围: 基于 `refrance/` 参考文件，为 ProcessGen 后端新增向量库能力并提供一个 PySide6 客户端

## 1. 背景与参考

ProcessGen 后端 (`api.py`) 现为 FastAPI 模型服务，提供嵌入 (`/api/v1/embed`)、重排 (`/api/v1/rerank`)、图像管理 (`/api/v1/images*`) 与 ComfyUI 文生图/图生图。**目前没有向量数据库端点**；向量库相关逻辑只存在于 `refrance/` 参考文件中：

- `refrance/database.py` — `MoyuClient(MilvusClient)`：内嵌嵌入模型 (`gme-Qwen2-VL`)，暴露 `init_collection / insert / insert_image / search / get_*_embeddings`。集合 schema：`id, embedding, type, path, text, subject`，度量 `COSINE`，`auto_id=True`。
- `refrance/text.py` — `TextProcessor`：PyMuPDF 提取 PDF 文本 + Markdown→RAG 分块。
- `refrance/rag.py` — 参考 FastAPI 路由：`add_text / query / 集合管理` + RAG 聊天（Qwen2.5-VL + MongoDB）。

目标：在现有后端上新增聚焦的向量库端点，并写一个 PySide6 瘦客户端操作它们（添加文本/图像、检索、集合管理）。

## 2. 架构

```
┌─────────────────────────────────────────────────────────────┐
│  Qt 工具 (PySide6 客户端, qt_tool/)                          │
│  UI: MainWindow(侧边导航 + QStackedWidget) + 4 个页面         │
│  服务: BackendClient (封装 requests)                        │
│  异步: ApiWorker(QRunnable) + 信号 → 不阻塞 UI 线程          │
│  依赖: PySide6, requests                                     │
└─────────────────────── HTTP ─────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│  FastAPI 后端 (扩展 api.py)                                   │
│  现有端点(不动): /health /api/v1/embed /api/v1/rerank         │
│                  /api/v1/images* /api/v1/text-to-image ...    │
│  新增路由: /api/v1/rag/* (rag_router.py)                      │
│  新增服务: MilvusService(单例, pymilvus)                     │
│           TextProcessor (从 refrance/ 移入主包)              │
│  复用服务: EmbeddingService (单例, 已加载配置好的嵌入模型)     │
│  新增依赖: pymilvus, PyMuPDF                                  │
└─────────────────────────────────────────────────────────────┘
```

**关键决策：**

1. **向量由 `EmbeddingService` 生成，不复刻参考里"客户端内嵌模型"的做法。** 参考的 `MoyuClient` 把模型塞进客户端并硬编码 `gme-Qwen2-VL`；后端已用 `EmbeddingService` 单例加载 `config.embedding_model_name`（当前 Qwen3-VL-Embedding-2B）。复用它，模型只加载一次且与 `/api/v1/embed` 一致。代价：`EmbeddingService` 现仅暴露单条 `get_text/image/fused_embedding`，新路由批量插入时循环调用（分块/图片数量通常不大，可接受）。

2. **集合 schema 沿用参考。** `id, embedding, type, path, text, subject`，`COSINE`，`auto_id=True`，`enable_dynamic_field=True`。与参考已有数据兼容。

3. **Qt 为独立子目录 `qt_tool/`。** 纯客户端，不碰 pymilvus/模型/GPU，通过 HTTP 调后端。

4. **文件存储分集合。** 文本存 `data/text/{collection}/{ts}.{ext}`，图片存 `data/images/{collection}/{ts}_{i}.{ext}`，`path` 字段存绝对路径。

## 3. 后端数据模型与 Milvus 服务

### 3.1 集合 schema

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INT64 | 主键, `auto_id=True` |
| `embedding` | FLOAT_VECTOR, dim=`EmbeddingService.get_dimension()` | 向量 |
| `type` | VARCHAR(16) | `text` / `image` |
| `path` | VARCHAR(1024) | 源文件绝对路径 |
| `text` | VARCHAR(65535) | 文本块内容 / 图像描述 |
| `subject` | VARCHAR(64) | 分区标签，默认 `capp` |

度量 `COSINE`，`enable_dynamic_field=True`。维度运行时从 `EmbeddingService.get_dimension()` 取，不硬编码。

### 3.2 新增 Pydantic 模型 (加到 `models.py`)

```python
class CreateCollectionRequest(BaseModel):
    collection_name: str = Field(description="集合名称")

class CollectionInfo(BaseModel):
    name: str
    row_count: Optional[int] = Field(default=None, description="实体数")
    loaded: Optional[bool] = Field(default=None, description="是否已加载")

class CollectionListResponse(BaseModel):
    collections: list[CollectionInfo]
    count: int

class AddTextResponse(BaseModel):
    status: str
    collection_name: str
    chunks_inserted: int
    saved_path: str

class AddImageRequest(BaseModel):
    description: str = Field(default="", description="图像描述，参与融合嵌入")
    subject: str = Field(default="capp")

class AddImageResponse(BaseModel):
    status: str
    collection_name: str
    images_inserted: int

class SearchResultItem(BaseModel):
    id: int
    score: float
    type: str
    text: str
    path: str
    subject: str
    asset_path: Optional[str] = Field(default=None, description="相对 data/ 的路径，用于取图")

class SearchResponse(BaseModel):
    collection_name: str
    query_type: str        # text / image
    results: list[SearchResultItem]
```

### 3.3 `MilvusService` (新增 `milvus_service.py`，单例)

```python
class MilvusService:
    _instance = None
    def __new__(cls): ...          # 单例，同 EmbeddingService 写法
    def __init__(self):
        self.client = MilvusClient(uri=config.milvus_uri)
    def init_collection(self, name): ...     # 参考 database.py.init_collection
    def list_collections(self): ...          # 返回 [(name, row_count)]
    def drop_collection(self, name): ...
    def insert(self, name, data): ...
    def search(self, name, vector, limit, filter=""): ...
```

连接 URI 走 `config.milvus_uri`（默认 `http://localhost:19530`）。`init_collection` 维度用 `EmbeddingService().get_dimension()`。连接失败不阻断后端启动，调到 rag 端点时才报 503。

## 4. 后端路由端点 (新增 `rag_router.py`，前缀 `/api/v1/rag`)

### 4.1 集合管理

```
POST   /api/v1/rag/collections              创建集合
       Body(JSON): {"collection_name": "capp"}
       → 201 {"status":"success","collection_name":"capp",...}

GET    /api/v1/rag/collections              列出集合
       → 200 CollectionListResponse

DELETE /api/v1/rag/collections/{name}       删除集合(同时删 data/text/{name} 与 data/images/{name})
       → 200 {"status":"success","collection_name":"capp","message":"..."}
```

### 4.2 添加文本

```
POST /api/v1/rag/collections/{name}/text
     Content-Type: multipart/form-data
     Form: file=<.md/.pdf/.txt>, subject=capp(可选)
     → 200 AddTextResponse
```

流程（复用参考 `rag.py` 的 `add_text`，去掉模型内嵌）：
1. `.md/.txt` 直接 utf-8 解码；`.pdf` 用 `TextProcessor.extract_text_from_pdf`。
2. `TextProcessor.split_markdown_for_rag` 分块（`min_words=10, include_subsections=False`，与参考一致）。
3. 保存原文件到 `data/text/{name}/{ts}.{ext}`。
4. `EmbeddingService` 逐块 `get_text_embedding` 生成向量。
5. `MilvusService.insert` 插入 `{embedding, type:"text", text:chunk, path:abs, subject}`。

### 4.3 添加图像

```
POST /api/v1/rag/collections/{name}/images
     Content-Type: multipart/form-data
     Form: images=<file>(可重复多个), descriptions=<str>(可重复, 与图片一一对应), subject=capp(可选)
     → 200 AddImageResponse
```

流程（复用参考 `database.py` 的 `insert_image`）：
1. 接收图片列表与描述列表，数量必须相等否则 400。
2. 保存图片到 `data/images/{name}/{ts}_{i}.{ext}`。
3. `EmbeddingService` 逐张生成向量：描述非空调 `get_fused_embedding(text, image)`，描述为空调 `get_image_embedding(image)`（纯图像嵌入，仍可检索）。
4. 插入 `{embedding, type:"image", text:description, path:abs, subject}`。

### 4.4 检索

```
GET  /api/v1/rag/collections/{name}/search?query=文本&limit=10&subject=capp(可选)
POST /api/v1/rag/collections/{name}/search   (query_type=image 时, multipart)
     Form: image=<file>, limit=10, subject=capp(可选)
     → 200 SearchResponse
```

用 `EmbeddingService` 生成查询向量（文本 `get_text_embedding` / 图像 `get_image_embedding`），`MilvusService.search` 取 `limit` 条，`output_fields=["id","text","subject","path","type"]`。`subject` 作为元数据过滤（`filter="subject == '{subject}'"`，未提供 `subject` 时不过滤）。结果按相似度排序，含 `asset_path`（图片时为相对 `data/` 的路径，文本时为 None）。文本检索 GET / 图像检索 POST 同路径分方法。

### 4.5 取图资源

```
GET /api/v1/rag/asset?path=<相对 data/ 的路径>
    → 200 image/<ext>  (用 is_safe_path(DATA_DIR, path) 校验, 越界 403)
```

存储保持分集合结构（`data/images/{name}/...`），现有 `/api/v1/images/{filename}`（仅单文件名、防子目录）不动；新增 `asset` 端点按相对路径取图。Qt 用检索结果 `asset_path` 下载 bytes 画 `QPixmap`。

## 5. Qt 客户端结构

### 5.1 目录

```
qt_tool/
├── main.py                  # 入口, 启动 MainWindow
├── config.py                # 后端 URL 等, 存 ~/.moyu_processgen_ui/config.json
├── backend_client.py        # BackendClient: 封装 requests, 拼装各端点
├── workers.py               # ApiWorker(QRunnable) + 信号, 跑线程池
├── widgets/
│   ├── main_window.py       # 侧边导航 + QStackedWidget
│   ├── add_text_page.py
│   ├── add_image_page.py
│   ├── search_page.py
│   └── collection_page.py
└── requirements.txt         # PySide6, requests
```

### 5.2 `BackendClient`（纯函数无 Qt 依赖，方便单测）

```python
class BackendClient:
    def __init__(self, base_url="http://localhost:8050"): ...
    def list_collections(self) -> list[dict]
    def create_collection(self, name) -> dict
    def delete_collection(self, name) -> dict
    def add_text(self, name, file_path, subject="capp") -> dict
    def add_images(self, name, image_paths, descriptions, subject="capp") -> dict
    def search_text(self, name, query, limit=10, subject=None) -> dict
    def search_image(self, name, image_path, limit=10, subject=None) -> dict
    def health(self) -> dict
    def get_asset(self, asset_path) -> bytes
```

每方法正确拼 multipart、处理非 2xx（抛带状态码与 detail 的异常）、超时 60s（图片插入可能慢）。

### 5.3 `ApiWorker`

```python
class ApiWorker(QRunnable):
    finished = Signal(object)   # 成功结果
    failed = Signal(str)        # 错误消息
    def __init__(self, fn, *args): ...  # fn 是 BackendClient 方法
    def run(self):
        try: self.finished.emit(self.fn(*self.args))
        except Exception as e: self.failed.emit(str(e))
```

页面 `QThreadPool.globalInstance().start(worker)`，回调里更新 UI。单例 `BackendClient` + 共享 `QThreadPool`。

### 5.4 `MainWindow`（侧边导航 + 多页面）

```
┌─────────────┬──────────────────────────────────────┐
│ 集合列表    │                                       │
│ ▸ capp     │     QStackedWidget                     │
│ ▸ docs     │     (按侧边选中项切换页面)               │
│ ─────────── │                                       │
│ [+ 添加文本]│                                       │
│ [+ 添加图像]│                                       │
│ [  检索   ] │                                       │
│ [ 集合管理 ]│                                       │
│ ─────────── │                                       │
│ 后端状态 ●  │                                       │
│ [设置]      │                                       │
└─────────────┴──────────────────────────────────────┘
```

左侧：集合下拉 + 功能按钮列表；右侧：`QStackedWidget` 装 4 页按选中切换；底部：后端健康状态（启动调 `/health`，绿/红点）+ 设置按钮（改 `backend_url`，存 `~/.moyu_processgen_ui/config.json`）。

### 5.5 页面职责

| 页面 | 控件 | 调用 |
|------|------|------|
| `AddTextPage` | 文件选择、`subject` 输入、提交、进度/结果文本 | `add_text` |
| `AddImagePage` | 图片多选(可拖入缩略图)、每张配描述、`subject`、提交 | `add_images` |
| `SearchPage` | 文本/图像切换、查询框、`limit`、结果列表(文本块/缩略图+来源+分数) | `search_text` / `search_image` |
| `CollectionPage` | 集合表格(名/实体数/状态)、创建、删除 | `list/create/delete` |

**图像检索结果显示缩略图：** 拿结果 `asset_path` 调 `BackendClient.get_asset` 下载 bytes 画 `QPixmap`；先占位图标，下载完成后替换。

## 6. 错误处理、配置与测试

### 6.1 后端错误处理（沿用 `api.py` 的 `HTTPException` + 全局 500 处理）

- 集合名非法/为空 → 400
- 集合不存在（插入/检索）→ 404
- 上传文件非 `.md/.pdf/.txt` → 400
- 图片与描述数量不匹配 → 400
- Milvus 连接失败 → 503
- 向量生成失败 → 500
- `asset` 端点路径越界 → 403（复用 `is_safe_path`）

### 6.2 Qt 错误处理

`ApiWorker.failed` 信号带消息，页面在状态栏/结果区显示，不弹一堆模态框。

### 6.3 后端配置 (`config.py` 新增)

```python
milvus_uri: str = Field(default="http://localhost:19530", description="Milvus 服务器地址")
rag_subject_default: str = Field(default="capp", description="默认 subject 分区标签")
```

`.env.example` 同步加这两行。Milvus 连接失败不阻断后端启动，调到 rag 端点才报 503。

### 6.4 Qt 配置 (`~/.moyu_processgen_ui/config.json`)

```json
{"backend_url": "http://localhost:8050"}
```

首次启动无文件用默认值并写入。设置页改 `backend_url` 后保存、重测 `/health`。

### 6.5 依赖

- 后端 `pyproject.toml` 加 `pymilvus`、`PyMuPDF`。
- Qt `qt_tool/requirements.txt`：`PySide6`、`requests`。

### 6.6 后端测试 (`test_api.py`，沿用其打 `localhost:8050` 模式)

- 创建/列出/删除集合
- 上传 md 文件 → 断言 `chunks_inserted > 0`
- 上传一张图 + 描述 → 断言 `images_inserted == 1`
- 文本检索 → 断言返回结果含 `score`
- `asset` 端点取图 → 断言 200 + image/png

Qt 不写自动化测试（简单工具，手动验证为主）；`BackendClient` 设计成纯函数无 Qt 依赖，便于将来补单测。

## 7. 完成后需同步更新

- `README.md` 的 API 文档部分新增 `/api/v1/rag/*` 端点说明（集合管理、添加文本/图像、检索、取图资源）。
- `.env.example` 新增 `MILVUS_URI`、`RAG_SUBJECT_DEFAULT`。
- `pyproject.toml` 新增依赖；`qt_tool/requirements.txt` 新建。
