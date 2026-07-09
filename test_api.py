import requests
import json
from io import BytesIO
from pathlib import Path
from PIL import Image


BASE_URL = "http://localhost:8050"

WORKFLOW_PATH = (
    Path(__file__).parent / "data" / "workflow" / "Flux-Dev-ComfyUI-Workflow.json"
)


def load_flux_workflow() -> dict:
    """Load the Flux workflow from JSON file."""
    if not WORKFLOW_PATH.exists():
        print(f"Warning: workflow file not found at {WORKFLOW_PATH}, using empty workflow")
        return {}
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


FLUX_WORKFLOW = load_flux_workflow()


def test_health():
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    result = response.json()
    assert result["embedding_model_loaded"] or result["status"] == "loading"
    assert result["rerank_model_loaded"] or result["status"] == "loading"
    print("✓ Health check passed")


def test_text_embedding():
    print("\n=== Testing Text Embedding ===")
    data = {"text": "这是一个测试文本"}
    response = requests.post(f"{BASE_URL}/api/v1/embed", data=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Dimension: {result['dimension']}")
    print(f"Embedding type: {result['embedding_type']}")
    print(f"Vector length: {len(result['vector'])}")
    assert response.status_code == 200
    assert result["embedding_type"] == "text"
    assert isinstance(result["vector"], list)
    assert len(result["vector"]) > 0
    print("✓ Text embedding passed")


def test_image_embedding():
    print("\n=== Testing Image Embedding ===")
    image = Image.new("RGB", (100, 100), color="red")
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    files = {"image_file": ("test.png", img_bytes, "image/png")}
    response = requests.post(f"{BASE_URL}/api/v1/embed", files=files)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Dimension: {result['dimension']}")
    print(f"Embedding type: {result['embedding_type']}")
    print(f"Vector length: {len(result['vector'])}")
    assert response.status_code == 200
    assert result["embedding_type"] == "image"
    assert isinstance(result["vector"], list)
    assert len(result["vector"]) > 0
    print("✓ Image embedding passed")


def test_fused_embedding():
    print("\n=== Testing Fused Embedding ===")
    image = Image.new("RGB", (100, 100), color="blue")
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    data = {"text": "这是一张蓝色的图片"}
    files = {"image_file": ("test.png", img_bytes, "image/png")}
    response = requests.post(f"{BASE_URL}/api/v1/embed", data=data, files=files)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Dimension: {result['dimension']}")
    print(f"Embedding type: {result['embedding_type']}")
    print(f"Vector length: {len(result['vector'])}")
    assert response.status_code == 200
    assert result["embedding_type"] == "fused"
    assert isinstance(result["vector"], list)
    assert len(result["vector"]) > 0
    print("✓ Fused embedding passed")


def test_rerank_text_text():
    print("\n=== Testing Rerank (Text Query -> Text Document) ===")
    data = {
        "query_type": "text",
        "query_text": "查找相关信息",
        "document_type": "text",
        "document_text": "这是一段相关文档",
    }
    response = requests.post(f"{BASE_URL}/api/v1/rerank", data=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Similarity score: {result['score']}")
    assert response.status_code == 200
    assert isinstance(result["score"], float)
    assert 0 <= result["score"] <= 1
    print("✓ Rerank (text->text) passed")


def test_rerank_text_image():
    print("\n=== Testing Rerank (Text Query -> Image Document) ===")
    image = Image.new("RGB", (100, 100), color="green")
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    data = {
        "query_type": "text",
        "query_text": "查找绿色图片",
        "document_type": "image",
    }
    files = {"document_image": ("doc.png", img_bytes, "image/png")}
    response = requests.post(f"{BASE_URL}/api/v1/rerank", data=data, files=files)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Similarity score: {result['score']}")
    assert response.status_code == 200
    assert isinstance(result["score"], float)
    assert 0 <= result["score"] <= 1
    print("✓ Rerank (text->image) passed")


def test_rerank_image_text():
    print("\n=== Testing Rerank (Image Query -> Text Document) ===")
    image = Image.new("RGB", (100, 100), color="yellow")
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    data = {
        "query_type": "image",
        "document_type": "text",
        "document_text": "这是一段关于黄色的描述",
    }
    files = {"query_image": ("query.png", img_bytes, "image/png")}
    response = requests.post(f"{BASE_URL}/api/v1/rerank", data=data, files=files)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Similarity score: {result['score']}")
    assert response.status_code == 200
    assert isinstance(result["score"], float)
    assert 0 <= result["score"] <= 1
    print("✓ Rerank (image->text) passed")


def test_rerank_image_image():
    print("\n=== Testing Rerank (Image Query -> Image Document) ===")
    query_image = Image.new("RGB", (100, 100), color="purple")
    query_img_bytes = BytesIO()
    query_image.save(query_img_bytes, format="PNG")
    query_img_bytes.seek(0)

    doc_image = Image.new("RGB", (100, 100), color="orange")
    doc_img_bytes = BytesIO()
    doc_image.save(doc_img_bytes, format="PNG")
    doc_img_bytes.seek(0)

    data = {"query_type": "image", "document_type": "image"}
    files = {
        "query_image": ("query.png", query_img_bytes, "image/png"),
        "document_image": ("doc.png", doc_img_bytes, "image/png"),
    }
    response = requests.post(f"{BASE_URL}/api/v1/rerank", data=data, files=files)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Similarity score: {result['score']}")
    assert response.status_code == 200
    assert isinstance(result["score"], float)
    assert 0 <= result["score"] <= 1
    print("✓ Rerank (image->image) passed")


def test_error_handling():
    print("\n=== Testing Error Handling ===")
    data = {}
    response = requests.post(f"{BASE_URL}/api/v1/embed", data=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 400
    print("✓ Error handling passed")


def test_text_to_image():
    print("\n=== Testing Text-to-Image with Flux Workflow ===")

    payload = {
        "prompt": "a beautiful game card design",
        "negative_prompt": "blurry, low quality",
        "width": 1024,
        "height": 1024,
        "steps": 20,
        "seed": 42,
        "cfg_scale": 3.5,
        "sampler_name": "dpmpp_2m",
        "scheduler": "simple",
        "checkpoint": "flux1-dev.safetensors",
        "workflow": FLUX_WORKFLOW,
    }

    response = requests.post(
        f"{BASE_URL}/api/v1/text-to-image",
        json=payload,
        timeout=300,
    )
    print(f"Status: {response.status_code}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0
    print(f"Image size: {len(response.content)} bytes")
    print("✓ Text-to-image with Flux workflow passed")


def test_text_to_image_with_lora():
    print("\n=== Testing Text-to-Image with LoRA and Flux Workflow ===")

    payload = {
        "prompt": "a beautiful landscape game card",
        "checkpoint": "flux1-dev.safetensors",
        "workflow": FLUX_WORKFLOW,
        "loras": [
            {"name": "detail_tweaker.safetensors", "strength": 0.8},
            {"name": "add_detail.safetensors", "strength": 1.0},
        ],
    }

    response = requests.post(
        f"{BASE_URL}/api/v1/text-to-image",
        json=payload,
        timeout=300,
    )
    print(f"Status: {response.status_code}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0
    print(f"Image size: {len(response.content)} bytes")
    print("✓ Text-to-image with LoRA and Flux workflow passed")


def test_image_to_image():
    print("\n=== Testing Image-to-Image with Flux Workflow ===")

    init_image = Image.new("RGB", (1024, 1024), color="white")
    init_img_bytes = BytesIO()
    init_image.save(init_img_bytes, format="PNG")
    init_img_bytes.seek(0)

    files = {
        "init_image": ("init.png", init_img_bytes, "image/png"),
    }
    data = {
        "prompt": "transform this image into a game card",
        "negative_prompt": "blurry",
        "width": 1024,
        "height": 1024,
        "steps": 20,
        "seed": 123,
        "cfg_scale": 3.5,
        "checkpoint": "flux1-dev.safetensors",
        "strength": 0.75,
        "workflow": json.dumps(FLUX_WORKFLOW),
    }

    response = requests.post(
        f"{BASE_URL}/api/v1/image-to-image",
        files=files,
        data=data,
        timeout=300,
    )
    print(f"Status: {response.status_code}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0
    print(f"Image size: {len(response.content)} bytes")
    print("✓ Image-to-image with Flux workflow passed")


def test_text_to_image_missing_prompt():
    print("\n=== Testing Text-to-Image Missing Prompt ===")

    payload = {
        "checkpoint": "flux1-dev.safetensors",
        "workflow": FLUX_WORKFLOW,
    }

    response = requests.post(
        f"{BASE_URL}/api/v1/text-to-image",
        json=payload,
        timeout=30,
    )
    print(f"Status: {response.status_code}")
    assert response.status_code == 422
    print("✓ Text-to-image missing prompt validation passed")


def test_text_to_image_missing_checkpoint():
    print("\n=== Testing Text-to-Image Missing Checkpoint ===")

    payload = {
        "prompt": "test prompt",
        "workflow": FLUX_WORKFLOW,
    }

    response = requests.post(
        f"{BASE_URL}/api/v1/text-to-image",
        json=payload,
        timeout=30,
    )
    print(f"Status: {response.status_code}")
    assert response.status_code == 422
    print("✓ Text-to-image missing checkpoint validation passed")


def test_text_to_image_missing_workflow():
    print("\n=== Testing Text-to-Image Missing Workflow ===")

    payload = {
        "prompt": "test prompt",
        "checkpoint": "flux1-dev.safetensors",
    }

    response = requests.post(
        f"{BASE_URL}/api/v1/text-to-image",
        json=payload,
        timeout=30,
    )
    print(f"Status: {response.status_code}")
    assert response.status_code == 422
    print("✓ Text-to-image missing workflow validation passed")


def test_image_to_image_missing_init_image():
    print("\n=== Testing Image-to-Image Missing Init Image ===")

    data = {
        "prompt": "test prompt",
        "checkpoint": "flux1-dev.safetensors",
        "workflow": json.dumps(FLUX_WORKFLOW),
    }

    response = requests.post(
        f"{BASE_URL}/api/v1/image-to-image",
        data=data,
        timeout=30,
    )
    print(f"Status: {response.status_code}")
    assert response.status_code == 422
    print("✓ Image-to-image missing init image validation passed")


def test_image_to_image_invalid_workflow_json():
    print("\n=== Testing Image-to-Image Invalid Workflow JSON ===")

    init_image = Image.new("RGB", (1024, 1024), color="white")
    init_img_bytes = BytesIO()
    init_image.save(init_img_bytes, format="PNG")
    init_img_bytes.seek(0)

    files = {
        "init_image": ("init.png", init_img_bytes, "image/png"),
    }
    data = {
        "prompt": "test prompt",
        "checkpoint": "flux1-dev.safetensors",
        "workflow": "not a valid json",
    }

    response = requests.post(
        f"{BASE_URL}/api/v1/image-to-image",
        files=files,
        data=data,
        timeout=30,
    )
    print(f"Status: {response.status_code}")
    assert response.status_code == 400
    assert "Invalid workflow JSON" in response.json()["detail"]
    print("✓ Image-to-image invalid workflow JSON validation passed")


def test_comfyui_service_workflow_injection():
    print("\n=== Testing ComfyUI Service Workflow Injection (Flux Workflow) ===")

    from comfyui_service import ComfyUIService
    from models import TextToImageRequest, LoraInfo

    service = ComfyUIService()

    loras = [
        LoraInfo(name="test_lora.safetensors", strength=0.8),
    ]

    request = TextToImageRequest(
        prompt="test prompt for game card",
        negative_prompt="test negative",
        width=768,
        height=768,
        steps=30,
        seed=42,
        cfg_scale=4.5,
        sampler_name="dpmpp_2m",
        scheduler="simple",
        checkpoint="flux1-dev.safetensors",
        loras=loras,
        workflow=FLUX_WORKFLOW.copy(),
    )

    workflow = service._inject_parameters_to_workflow(
        workflow=request.workflow,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt or "",
        width=request.width,
        height=request.height,
        steps=request.steps,
        seed=request.seed if request.seed is not None else 0,
        cfg_scale=request.cfg_scale,
        sampler_name=request.sampler_name,
        scheduler=request.scheduler,
        checkpoint=request.checkpoint,
    )

    for node_id, node_data in service._find_nodes_by_type(workflow, "KSamplerSelect"):
        assert node_data["inputs"]["sampler_name"] == "dpmpp_2m"

    for node_id, node_data in service._find_nodes_by_type(workflow, "BasicScheduler"):
        assert node_data["inputs"]["steps"] == 30
        assert node_data["inputs"]["scheduler"] == "simple"

    for node_id, node_data in service._find_nodes_by_type(workflow, "RandomNoise"):
        assert node_data["inputs"]["noise_seed"] == 42

    for node_id, node_data in service._find_nodes_by_type(workflow, "FluxGuidance"):
        assert node_data["inputs"]["guidance"] == 4.5

    clip_nodes = service._find_nodes_by_type(workflow, "CLIPTextEncode")
    assert len(clip_nodes) >= 1
    assert clip_nodes[0][1]["inputs"]["text"] == "test prompt for game card"

    for node_id, node_data in service._find_nodes_by_type(
        workflow, "EmptySD3LatentImage"
    ):
        assert node_data["inputs"]["width"] == 768
        assert node_data["inputs"]["height"] == 768

    for node_id, node_data in service._find_nodes_by_type(
        workflow, "ModelSamplingFlux"
    ):
        assert node_data["inputs"]["width"] == 768
        assert node_data["inputs"]["height"] == 768

    for node_id, node_data in service._find_nodes_by_type(workflow, "UNETLoader"):
        assert node_data["inputs"]["unet_name"] == "flux1-dev.safetensors"

    print("✓ ComfyUI service workflow injection (Flux) passed")


def test_comfyui_service_lora_injection():
    print("\n=== Testing ComfyUI Service LoRA Injection (Flux Workflow) ===")

    from comfyui_service import ComfyUIService
    from models import LoraInfo

    service = ComfyUIService()

    loras = [
        LoraInfo(name="lora1.safetensors", strength=0.8),
        LoraInfo(name="lora2.safetensors", strength=1.2),
    ]

    workflow = service._inject_loras_to_workflow(FLUX_WORKFLOW.copy(), loras)

    lora_nodes = service._find_nodes_by_type(workflow, "LoraLoader")
    assert len(lora_nodes) == 2

    first_lora = lora_nodes[0][1]
    assert first_lora["inputs"]["lora_name"] == "lora1.safetensors"
    assert first_lora["inputs"]["strength_model"] == 0.8
    assert first_lora["inputs"]["strength_clip"] == 0.8

    second_lora = lora_nodes[1][1]
    assert second_lora["inputs"]["lora_name"] == "lora2.safetensors"
    assert second_lora["inputs"]["strength_model"] == 1.2

    print("✓ ComfyUI service LoRA injection (Flux) passed")


def test_text_to_image_comfyui_not_connected():
    print("\n=== Testing Text-to-Image ComfyUI Not Connected ===")

    payload = {
        "prompt": "test prompt",
        "checkpoint": "flux1-dev.safetensors",
        "workflow": FLUX_WORKFLOW,
    }

    response = requests.post(
        f"{BASE_URL}/api/v1/text-to-image",
        json=payload,
        timeout=30,
    )
    print(f"Status: {response.status_code}")

    if response.status_code == 500:
        print("✓ Text-to-image ComfyUI not connected error passed")
    else:
        print(f"Note: Expected 500 (ComfyUI not connected), got {response.status_code}")


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


def main():
    print("=" * 60)
    print("ProcessGen Model Server API Tests")
    print("=" * 60)

    try:
        test_health()
        test_rag_collections_lifecycle()
        test_rag_add_text_and_search()
        test_rag_add_image_and_search_and_asset()
        test_text_embedding()
        test_image_embedding()
        test_fused_embedding()
        test_rerank_text_text()
        test_rerank_text_image()
        test_rerank_image_text()
        test_rerank_image_image()
        test_error_handling()
        test_text_to_image()
        # test_text_to_image_with_lora()
        test_image_to_image()
        test_text_to_image_missing_prompt()
        test_text_to_image_missing_checkpoint()
        test_text_to_image_missing_workflow()
        test_image_to_image_missing_init_image()
        test_image_to_image_invalid_workflow_json()
        test_comfyui_service_workflow_injection()
        test_comfyui_service_lora_injection()
        test_text_to_image_comfyui_not_connected()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
    except requests.exceptions.ConnectionError:
        print(
            "\n✗ Cannot connect to server. Make sure it's running on http://localhost:8050"
        )
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
