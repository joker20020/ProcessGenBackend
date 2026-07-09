from PySide6.QtCore import QThreadPool, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QTableWidget, QTableWidgetItem, QLabel, QMessageBox,
)
from backend_client import BackendClient
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
