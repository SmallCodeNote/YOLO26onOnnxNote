from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
)

from config_manager import ConfigManager
from inference_engine import InferenceEngine
import cv2
import numpy as np


class OnnxPredictorGUI(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ONNX Predictor (YOLO Pose / DirectML)")
        self.config = ConfigManager()

        self.engine: InferenceEngine | None = None

        self._build_ui()
        self._load_config_to_controls()

    def _build_ui(self) -> None:
        # Controls
        self.textBox_onnxFilePath = QLineEdit()
        self.textBox_imageDirectoryPath = QLineEdit()
        self.textBox_resultDirectoryPath_image = QLineEdit()
        self.textBox_resultDirectoryPath_csv = QLineEdit()

        for tb in [
            self.textBox_onnxFilePath,
            self.textBox_imageDirectoryPath,
            self.textBox_resultDirectoryPath_image,
            self.textBox_resultDirectoryPath_csv,
        ]:
            tb.setMinimumWidth(200)

        self.button_browse_onnx = QPushButton("Browse...")
        self.button_browse_image_dir = QPushButton("Browse...")
        self.button_browse_result_image_dir = QPushButton("Browse...")
        self.button_browse_result_csv_dir = QPushButton("Browse...")

        self.horizontalSlider_scoreThValue = QSlider(Qt.Orientation.Horizontal)
        self.horizontalSlider_scoreThValue.setMinimum(0)
        self.horizontalSlider_scoreThValue.setMaximum(1000)
        self.horizontalSlider_scoreThValue.setSingleStep(1)
        self.horizontalSlider_scoreThValue.valueChanged.connect(
            self._on_slider_changed
        )

        self.label_scoreThValue = QLabel("0.250")
        self.label_scoreThValue.setFixedWidth(60)

        self.button_SaveConfig = QPushButton("Save Config")
        self.button_Run = QPushButton("Run")

        self.pictureBox_resultImage = QLabel()
        self.pictureBox_resultImage.setText("Result Image Preview")
        self.pictureBox_resultImage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pictureBox_resultImage.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.pictureBox_resultImage.setMinimumHeight(240)

        # Layouts
        main_layout = QVBoxLayout()

        grid = QGridLayout()
        row = 0

        # Row 0: ONNX path
        grid.addWidget(QLabel("ONNX File Path"), row, 0)
        grid.addWidget(self.textBox_onnxFilePath, row, 1)
        grid.addWidget(self.button_browse_onnx, row, 2)
        row += 1

        # Row 1: Image directory
        grid.addWidget(QLabel("Image / Video Directory"), row, 0)
        grid.addWidget(self.textBox_imageDirectoryPath, row, 1)
        grid.addWidget(self.button_browse_image_dir, row, 2)
        row += 1

        # Row 2: Result image directory
        grid.addWidget(QLabel("Result Image Directory"), row, 0)
        grid.addWidget(self.textBox_resultDirectoryPath_image, row, 1)
        grid.addWidget(self.button_browse_result_image_dir, row, 2)
        row += 1

        # Row 3: Result CSV directory
        grid.addWidget(QLabel("Result CSV Directory"), row, 0)
        grid.addWidget(self.textBox_resultDirectoryPath_csv, row, 1)
        grid.addWidget(self.button_browse_result_csv_dir, row, 2)
        row += 1

        # Row 4: Score threshold slider
        grid.addWidget(QLabel("Score Threshold"), row, 0)
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.horizontalSlider_scoreThValue)
        slider_layout.addWidget(self.label_scoreThValue)
        grid.addLayout(slider_layout, row, 1, 1, 2)
        row += 1

        main_layout.addLayout(grid)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.button_SaveConfig)
        btn_layout.addWidget(self.button_Run)
        main_layout.addLayout(btn_layout)

        # Picture box
        main_layout.addWidget(self.pictureBox_resultImage)

        self.setLayout(main_layout)

        # Connections
        self.button_browse_onnx.clicked.connect(self._browse_onnx)
        self.button_browse_image_dir.clicked.connect(self._browse_image_dir)
        self.button_browse_result_image_dir.clicked.connect(
            self._browse_result_image_dir
        )
        self.button_browse_result_csv_dir.clicked.connect(self._browse_result_csv_dir)
        self.button_SaveConfig.clicked.connect(self._save_config_from_controls)
        self.button_Run.clicked.connect(self._run_inference)

    # ---------------- Config handling ----------------

    def _load_config_to_controls(self) -> None:
        if self.config.onnx_file_path:
            self.textBox_onnxFilePath.setText(self.config.onnx_file_path)
        if self.config.image_dir_path:
            self.textBox_imageDirectoryPath.setText(self.config.image_dir_path)
        if self.config.result_dir_image:
            self.textBox_resultDirectoryPath_image.setText(
                self.config.result_dir_image
            )
        if self.config.result_dir_csv:
            self.textBox_resultDirectoryPath_csv.setText(self.config.result_dir_csv)

        # Score threshold -> slider
        score_th = self.config.score_threshold
        score_th = max(0.0, min(1.0, score_th))
        slider_val = int(round(score_th * 1000))
        self.horizontalSlider_scoreThValue.setValue(slider_val)
        self._update_score_label(score_th)

    def _save_config_from_controls(self) -> None:
        try:
            self.config.onnx_file_path = self.textBox_onnxFilePath.text().strip()
            self.config.image_dir_path = self.textBox_imageDirectoryPath.text().strip()
            self.config.result_dir_image = (
                self.textBox_resultDirectoryPath_image.text().strip()
            )
            self.config.result_dir_csv = (
                self.textBox_resultDirectoryPath_csv.text().strip()
            )
            score_th = self._get_score_threshold()
            self.config.score_threshold = score_th
            self.config.save()
            QMessageBox.information(self, "Config", "Configuration saved.")
        except Exception as e:
            print(f"[GUI] Failed to save config: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save config:\n{e}")

    # ---------------- Slider handling ----------------

    def _get_score_threshold(self) -> float:
        val = self.horizontalSlider_scoreThValue.value()
        return val / 1000.0

    def _update_score_label(self, score: float) -> None:
        self.label_scoreThValue.setText(f"{score:.3f}")

    def _on_slider_changed(self, value: int) -> None:
        score = value / 1000.0
        self._update_score_label(score)

    # ---------------- Browsers ----------------

    def _browse_onnx(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ONNX file", "", "ONNX Files (*.onnx);;All Files (*)"
        )
        if path:
            self.textBox_onnxFilePath.setText(path)

    def _browse_image_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Image/Video Directory", "")
        if path:
            self.textBox_imageDirectoryPath.setText(path)

    def _browse_result_image_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Result Image Directory", "")
        if path:
            self.textBox_resultDirectoryPath_image.setText(path)

    def _browse_result_csv_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Result CSV Directory", "")
        if path:
            self.textBox_resultDirectoryPath_csv.setText(path)

    # ---------------- Inference ----------------

    def _run_inference(self) -> None:
        try:
            onnx_path = self.textBox_onnxFilePath.text().strip()
            img_dir = self.textBox_imageDirectoryPath.text().strip()
            out_img_dir = self.textBox_resultDirectoryPath_image.text().strip()
            out_csv_dir = self.textBox_resultDirectoryPath_csv.text().strip()
            score_th = self._get_score_threshold()

            if not onnx_path or not Path(onnx_path).is_file():
                raise FileNotFoundError("ONNX file not found.")

            if not img_dir or not Path(img_dir).is_dir():
                raise FileNotFoundError("Image/Video directory not found.")

            if not out_img_dir:
                raise ValueError("Result Image Directory is empty.")
            if not out_csv_dir:
                raise ValueError("Result CSV Directory is empty.")

            img_dir_path = Path(img_dir)
            out_img_dir_path = Path(out_img_dir)
            out_csv_dir_path = Path(out_csv_dir)

            # Lazy-load engine
            if self.engine is None or self.engine.onnx_path != onnx_path:
                self.engine = InferenceEngine(onnx_path)

            # Collect files
            image_exts = {".jpg", ".jpeg", ".png"}
            video_exts = {".mp4"}

            image_files = []
            video_files = []

            for p in sorted(img_dir_path.iterdir()):
                if not p.is_file():
                    continue
                ext = p.suffix.lower()
                if ext in image_exts:
                    image_files.append(p)
                elif ext in video_exts:
                    video_files.append(p)

            if not image_files and not video_files:
                raise FileNotFoundError("No jpg/png/mp4 files found in directory.")

            # Case1: images -> single CSV "result.csv"
            if image_files:
                csv_path = out_csv_dir_path / "result.csv"
                if csv_path.is_file():
                    csv_path.unlink()
                self.engine.process_image_files(
                    image_files, out_img_dir_path, csv_path, score_th
                )
                # Show last processed image as preview
                last_draw_path = (
                    out_img_dir_path / (image_files[-1].stem + "_draw" + image_files[-1].suffix)
                )
                if last_draw_path.is_file():
                    self._show_image_in_label(last_draw_path)

            # Case2: videos -> each video has its own CSV and mp4
            for v in video_files:
                self.engine.process_video_file(
                    v, out_img_dir_path, out_csv_dir_path, score_th
                )

            QMessageBox.information(self, "Done", "Inference completed.")
        except Exception as e:
            print(f"[GUI] Error during inference: {e}")
            QMessageBox.critical(self, "Error", f"Error during inference:\n{e}")

    def _show_image_in_label(self, img_path: Path) -> None:
        img = cv2.imread(str(img_path))
        if img is None:
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        qimg = QPixmap.fromImage(
            self._numpy_to_qimage(img_rgb)
        ).scaled(
            self.pictureBox_resultImage.width(),
            self.pictureBox_resultImage.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.pictureBox_resultImage.setPixmap(qimg)

    @staticmethod
    def _numpy_to_qimage(img_rgb: np.ndarray):
        from PySide6.QtGui import QImage

        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        return QImage(
            img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        ).copy()


def main() -> None:
    app = QApplication(sys.argv)
    w = OnnxPredictorGUI()
    w.resize(800, 600)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
