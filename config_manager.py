import json
from pathlib import Path
from typing import Any, Dict

CONFIG_FILE_NAME = "setting.json"


class ConfigManager:
    def __init__(self, config_path: str | None = None) -> None:
        if config_path is None:
            self.config_path = Path(__file__).resolve().parent / CONFIG_FILE_NAME
        else:
            self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        if self.config_path.is_file():
            try:
                with self.config_path.open("r", encoding="utf-8") as f:
                    self._config = json.load(f)
            except Exception as e:
                print(f"[ConfigManager] Failed to load config: {e}")
                self._config = {}
        else:
            self._config = {}

    def save(self) -> None:
        try:
            with self.config_path.open("w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ConfigManager] Failed to save config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._config[key] = value

    # -------------------------
    # Properties used by GUI
    # -------------------------

    @property
    def onnx_file_path(self) -> str:
        return self.get("onnx_file_path", "")

    @onnx_file_path.setter
    def onnx_file_path(self, value: str) -> None:
        self.set("onnx_file_path", value)

    @property
    def image_dir_path(self) -> str:
        return self.get("image_dir_path", "")

    @image_dir_path.setter
    def image_dir_path(self, value: str) -> None:
        self.set("image_dir_path", value)

    @property
    def result_dir_image(self) -> str:
        return self.get("result_dir_image", "")

    @result_dir_image.setter
    def result_dir_image(self, value: str) -> None:
        self.set("result_dir_image", value)

    @property
    def result_dir_csv(self) -> str:
        return self.get("result_dir_csv", "")

    @result_dir_csv.setter
    def result_dir_csv(self, value: str) -> None:
        self.set("result_dir_csv", value)

    @property
    def score_threshold(self) -> float:
        return float(self.get("score_threshold", 0.25))

    @score_threshold.setter
    def score_threshold(self, value: float) -> None:
        self.set("score_threshold", float(value))
