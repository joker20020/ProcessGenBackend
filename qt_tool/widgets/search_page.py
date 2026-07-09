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
            widget = _ResultWidget(it, self.client)
            wi.setSizeHint(widget.sizeHint())
            self.results.addItem(wi)
            self.results.setItemWidget(wi, widget)
        self.status.setText(f"返回 {len(items)} 条")

    def _on_err(self, msg):
        self.status.setText(f"错误: {msg}")
