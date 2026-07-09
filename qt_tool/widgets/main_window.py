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
