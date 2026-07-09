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
