# ProcessGen Model Server

多模态嵌入和重排模型服务器，提供文本/图像嵌入、多模态重排评分、以及基于 ComfyUI 的图像生成服务。

## 功能特性

- **多模态嵌入**: 支持文本、图像、以及文本+图像融合嵌入
- **多模态重排**: 支持文本-文本、文本-图像、图像-文本、图像-图像的相似度评分
- **图像管理**: 提供 data 目录下图像文件的列表、获取、元数据查询
- **图像生成**: 集成 ComfyUI 实现文生图和图生图功能

## 技术栈

- **框架**: FastAPI
- **嵌入模型**: [Alibaba-NLP/gme-Qwen2-VL-2B-Instruct](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct)
- **重排模型**: [jinaai/jina-reranker-m0](https://huggingface.co/jinaai/jina-reranker-m0)
- **图像生成**: ComfyUI
- **Python**: >=3.11

## 快速开始

### 1. 环境要求

- Python 3.11+
- CUDA 设备 (用于模型推理)
- (可选) ComfyUI 服务 (用于图像生成)

### 2. 安装

```bash
# 克隆项目
git clone <repository-url>
cd ProcessGenBackend

# 安装 uv 包管理器 (推荐)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖
uv sync
```

### 3. 配置

复制环境变量模板并根据需要修改：

```bash
cp .env.example .env
```

配置项说明：

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `API_HOST` | `0.0.0.0` | API 服务器监听地址 |
| `API_PORT` | `8050` | API 服务器端口 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `EMBEDDING_MODEL_NAME` | `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct` | 嵌入模型名称或路径 |
| `RERANK_MODEL_NAME` | `jinaai/jina-reranker-m0` | 重排模型名称或路径 |
| `EMBEDDING_DEVICE` | `cuda:1` | 嵌入模型运行设备 |
| `RERANK_DEVICE` | `cuda:1` | 重排模型运行设备 |
| `MAX_UPLOAD_SIZE` | `10485760` | 最大上传文件大小 (字节) |
| `COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI 服务地址 |
| `COMFYUI_TIMEOUT` | `300` | ComfyUI 请求超时时间 (秒) |

### 4. 启动服务

```bash
# 使用 uv 运行
uv run python run_api.py

# 或者直接运行
python run_api.py
```

服务启动后访问:
- API 文档 (Swagger UI): http://localhost:8050/docs
- API 文档 (ReDoc): http://localhost:8050/redoc

## API 文档

### 健康检查

```http
GET /health
```

**响应示例:**

```json
{
  "status": "healthy",
  "embedding_model_loaded": true,
  "rerank_model_loaded": true,
  "embedding_model_name": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
  "rerank_model_name": "jinaai/jina-reranker-m0"
}
```

---

### 图像管理

#### 获取图像列表

```http
GET /api/v1/images
```

**响应示例:**

```json
{
  "images": [
    {
      "filename": "example.png",
      "size": 237188,
      "format": "PNG",
      "created_at": "2026-02-05 12:46:47"
    }
  ],
  "count": 1
}
```

#### 获取图像文件

```http
GET /api/v1/images/{filename}
```

**响应:** 返回图像文件 (Content-Type: image/png, image/jpeg 等)

#### 获取图像元数据

```http
GET /api/v1/images/{filename}/info
```

**响应示例:**

```json
{
  "filename": "example.png",
  "size": 237188,
  "format": "PNG",
  "width": 2120,
  "height": 1185,
  "created_at": "2026-02-05 12:46:47"
}
```

---

### 嵌入服务

#### 获取嵌入向量

```http
POST /api/v1/embed
Content-Type: multipart/form-data
```

**参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | 否* | 待嵌入的文本内容 |
| `image_file` | file | 否* | 待嵌入的图像文件 |

*至少提供其中一个参数，或同时提供两者获取融合嵌入

**示例 - 文本嵌入:**

```bash
curl -X POST "http://localhost:8050/api/v1/embed" \
  -F "text=这是一个测试文本"
```

**示例 - 图像嵌入:**

```bash
curl -X POST "http://localhost:8050/api/v1/embed" \
  -F "image_file=@image.png"
```

**示例 - 融合嵌入:**

```bash
curl -X POST "http://localhost:8050/api/v1/embed" \
  -F "text=这是一张图片描述" \
  -F "image_file=@image.png"
```

**响应示例:**

```json
{
  "vector": [0.123, -0.456, ...],
  "dimension": 1536,
  "embedding_type": "text"
}
```

---

### 重排服务

#### 计算相似度评分

```http
POST /api/v1/rerank
Content-Type: multipart/form-data
```

**参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query_type` | string | 是 | 查询类型: `text` 或 `image` |
| `query_text` | string | 条件 | 查询文本 (query_type=text 时必填) |
| `query_image` | file | 条件 | 查询图像 (query_type=image 时必填) |
| `document_type` | string | 是 | 文档类型: `text` 或 `image` |
| `document_text` | string | 条件 | 文档文本 (document_type=text 时必填) |
| `document_image` | file | 条件 | 文档图像 (document_type=image 时必填) |

**示例 - 文本查询 → 文档文本:**

```bash
curl -X POST "http://localhost:8050/api/v1/rerank" \
  -F "query_type=text" \
  -F "query_text=查找相关信息" \
  -F "document_type=text" \
  -F "document_text=这是一段相关文档"
```

**示例 - 文本查询 → 文档图像:**

```bash
curl -X POST "http://localhost:8050/api/v1/rerank" \
  -F "query_type=text" \
  -F "query_text=查找绿色图片" \
  -F "document_type=image" \
  -F "document_image=@doc.png"
```

**示例 - 图像查询 → 文档文本:**

```bash
curl -X POST "http://localhost:8050/api/v1/rerank" \
  -F "query_type=image" \
  -F "query_image=@query.png" \
  -F "document_type=text" \
  -F "document_text=这是一段描述"
```

**示例 - 图像查询 → 文档图像:**

```bash
curl -X POST "http://localhost:8050/api/v1/rerank" \
  -F "query_type=image" \
  -F "query_image=@query.png" \
  -F "document_type=image" \
  -F "document_image=@doc.png"
```

**响应示例:**

```json
{
  "score": 0.7261149883270264
}
```

---

### 图像生成服务 (ComfyUI)

> **注意**: 图像生成功能需要先启动 ComfyUI 服务

#### 文生图

```http
POST /api/v1/text-to-image
Content-Type: application/json
```

**请求体:**

```json
{
  "prompt": "a beautiful sunset over mountains",
  "negative_prompt": "blurry, low quality",
  "width": 512,
  "height": 512,
  "steps": 20,
  "seed": 42,
  "cfg_scale": 7.5,
  "sampler_name": "euler",
  "scheduler": "normal",
  "checkpoint": "v1-5-pruned.safetensors",
  "workflow": { ... },
  "loras": [
    {
      "name": "detail_tweaker.safetensors",
      "strength": 0.8
    }
  ]
}
```

**参数说明:**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `prompt` | string | 是 | - | 正向提示词 |
| `negative_prompt` | string | 否 | `""` | 负向提示词 |
| `width` | int | 否 | `512` | 图像宽度 (64-2048) |
| `height` | int | 否 | `512` | 图像高度 (64-2048) |
| `steps` | int | 否 | `20` | 采样步数 (1-150) |
| `seed` | int | 否 | 随机 | 随机种子 |
| `cfg_scale` | float | 否 | `7.5` | CFG 引导强度 (1.0-30.0) |
| `sampler_name` | string | 否 | - | 采样器名称 |
| `scheduler` | string | 否 | - | 调度器名称 |
| `checkpoint` | string | 是 | - | 模型检查点名称 |
| `workflow` | object | 是 | - | ComfyUI 工作流 JSON |
| `loras` | array | 否 | - | LoRA 列表 |

**响应:** PNG 图像数据

**示例:**

```bash
curl -X POST "http://localhost:8050/api/v1/text-to-image" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful landscape",
    "checkpoint": "v1-5-pruned.safetensors",
    "workflow": {"1": {"class_type": "CheckpointLoaderSimple", ...}}
  }' \
  --output generated.png
```

#### 图生图

```http
POST /api/v1/image-to-image
Content-Type: multipart/form-data
```

**参数:**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `prompt` | string | 是 | - | 正向提示词 |
| `negative_prompt` | string | 否 | `""` | 负向提示词 |
| `width` | int | 否 | `512` | 图像宽度 |
| `height` | int | 否 | `512` | 图像高度 |
| `steps` | int | 否 | `20` | 采样步数 |
| `seed` | int | 否 | 随机 | 随机种子 |
| `cfg_scale` | float | 否 | `7.5` | CFG 引导强度 |
| `sampler_name` | string | 否 | - | 采样器名称 |
| `scheduler` | string | 否 | - | 调度器名称 |
| `checkpoint` | string | 是 | - | 模型检查点名称 |
| `strength` | float | 否 | `0.75` | 图生图变换强度 (0-1) |
| `workflow` | string | 是 | - | ComfyUI 工作流 JSON 字符串 |
| `loras` | string | 否 | - | LoRA 列表 JSON 字符串 |
| `init_image` | file | 是 | - | 初始图像文件 |

**响应:** PNG 图像数据

**示例:**

```bash
curl -X POST "http://localhost:8050/api/v1/image-to-image" \
  -F "prompt=transform this image" \
  -F "checkpoint=v1-5-pruned.safetensors" \
  -F "strength=0.75" \
  -F "workflow={\"1\": {\"class_type\": \"CheckpointLoaderSimple\", ...}}" \
  -F "init_image=@input.png" \
  --output transformed.png
```

---

## 测试

项目包含完整的测试套件：

```bash
# 运行所有测试 (需要服务器运行在 localhost:8050)
uv run python test_api.py

# 或者先启动服务器，再运行测试
uv run python run_api.py &
uv run python test_api.py
```

测试覆盖:
- 健康检查
- 文本/图像/融合嵌入
- 四种重排组合 (text-text, text-image, image-text, image-image)
- 错误处理
- 图像管理端点
- ComfyUI 工作流注入
- 参数验证

---

## 项目结构

```
ProcessGenBackend/
├── api.py              # FastAPI 应用和路由定义
├── run_api.py          # 服务启动入口
├── config.py           # 配置管理 (Pydantic Settings)
├── models.py           # 请求/响应模型定义
├── embeddings.py       # 嵌入服务实现
├── rerank.py           # 重排服务实现
├── comfyui_service.py  # ComfyUI 集成服务
├── test_api.py         # 测试套件
├── pyproject.toml      # 项目依赖配置
├── .env.example        # 环境变量模板
└── data/               # 图像存储目录
    └── workflow/       # ComfyUI 工作流文件
```

---

## 常见问题

### Q: 模型加载失败怎么办?

A: 检查以下项:
1. 确保 CUDA 设备可用且内存充足
2. 检查 `EMBEDDING_DEVICE` 和 `RERANK_DEVICE` 配置
3. 如果内存不足，可以尝试使用 `cpu` 设备 (速度较慢)

### Q: ComfyUI 图像生成超时?

A: 调整 `COMFYUI_TIMEOUT` 环境变量，默认为 300 秒。复杂工作流可能需要更长时间。

### Q: 如何添加自定义工作流?

A: 将 ComfyUI 工作流 JSON 放入 `data/workflow/` 目录，然后在 API 请求中传递工作流对象。

---

## 许可证

MIT License
