from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton


class SettingsDialog(QDialog):
    def __init__(self, backend_url: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("后端 URL:"))
        self.url_edit = QLineEdit(backend_url)
        layout.addWidget(self.url_edit)
        row = QHBoxLayout()
        ok = QPushButton("保存")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        row.addStretch(1)
        row.addWidget(ok)
        row.addWidget(cancel)
        layout.addLayout(row)

    def url(self) -> str:
        return self.url_edit.text().strip()
