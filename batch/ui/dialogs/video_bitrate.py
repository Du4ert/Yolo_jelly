"""Поле выбора битрейта выходного видео."""

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSpinBox, QWidget


class VideoBitrateField(QWidget):
    """Положительное значение — Мбит/с, 0 — битрейт оригинала."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.mode = QComboBox()
        self.mode.addItem("Задать вручную", False)
        self.mode.addItem("Как у исходного видео", True)
        self.mbps = QSpinBox()
        self.mbps.setRange(1, 1000)
        self.mbps.setSuffix(" Мбит/с")
        self.mbps.setValue(20)
        self.mode.currentIndexChanged.connect(self._update_mode)

        label = QLabel("Битрейт выходного видео:")
        label.setBuddy(self.mode)
        layout.addWidget(label)
        layout.addWidget(self.mode)
        layout.addWidget(self.mbps)
        layout.addStretch()

        self.setToolTip(
            "Целевой битрейт для видео с детекциями и видео с размерами.\n"
            "Разрешение и FPS не меняются. Для задания битрейта нужен FFmpeg в PATH,\n"
            "для режима «Как у исходного видео» дополнительно нужен ffprobe."
        )

    def _update_mode(self):
        self.mbps.setEnabled(not self.mode.currentData())

    def set_value(self, value: int):
        if value > 0:
            self.mbps.setValue(value)
        self.mode.setCurrentIndex(1 if value == 0 else 0)

    def value(self) -> int:
        return 0 if self.mode.currentData() else self.mbps.value()
