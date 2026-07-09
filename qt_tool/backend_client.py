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
