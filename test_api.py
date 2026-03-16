import requests
import time
from io import BytesIO
from PIL import Image


BASE_URL = "http://localhost:8050"


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


def main():
    print("=" * 60)
    print("ProcessGen Model Server API Tests")
    print("=" * 60)

    try:
        test_health()
        test_text_embedding()
        test_image_embedding()
        test_fused_embedding()
        test_rerank_text_text()
        test_rerank_text_image()
        test_rerank_image_text()
        test_rerank_image_image()
        test_error_handling()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
    except requests.exceptions.ConnectionError:
        print(
            "\n✗ Cannot connect to server. Make sure it's running on http://localhost:8000"
        )
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")


if __name__ == "__main__":
    main()
