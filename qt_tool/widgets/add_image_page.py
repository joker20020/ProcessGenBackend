from PySide6.QtCore import QThreadPool
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
