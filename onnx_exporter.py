import sys
import json
import os
from pathlib import Path

# Import necessary external libraries
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: 'ultralytics' library not found. Please install it using: pip install ultralytics")
    sys.exit(1)

# Import PySide6 (adding QFileDialog)
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLineEdit, QComboBox, QLabel, QMessageBox, QFileDialog # Added QFileDialog
)
from PySide6.QtCore import Qt

# --- Configuration File Management Class ---

SETTINGS_FILE = "settingOnnxExporter.json"
DEFAULT_CONFIG = {
    "export_dir": "",  # Initial value is determined automatically upon setting
    "model_name": "yolov8", # Default model name
    "model_size": "n"     # Default size
}

def load_settings():
    """Loads settings from setting file. Returns default values if the file does not exist."""
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            print(f"Settings loaded from {SETTINGS_FILE}")
            return json.load(f)
    except FileNotFoundError:
        print("Setting file not found. Using default configuration.")
        return DEFAULT_CONFIG
    except json.JSONDecodeError:
        print("Error decoding setting file. Using default configuration.")
        return DEFAULT_CONFIG

def save_settings(config):
    """Saves the current settings to setting.json."""
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        print("Settings saved successfully.")
    except IOError as e:
        QMessageBox.critical(None, "Save Error", f"An error occurred while saving the configuration file: {e}")


# --- GUI Application Class ---

class ModelExporterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO ONNX Exporter")
        self.setGeometry(100, 100, 650, 320) # Adjust size slightly larger
        
        # --- Determine Initial Settings (Handling Request 3: Setting Default Directory) ---
        script_dir = Path(__file__).parent
        default_export_path = script_dir / "onnxModels"
        self.current_config = load_settings()

        # Prioritize path saved in the configuration file if it exists. Otherwise, use the default.
        saved_path = self.current_config.get("export_dir")
        if saved_path:
             # Use the saved path if it is a valid directory
            self.default_directory = Path(saved_path)
        else:
            # If not saved, use 'onnxModels' next to the script as default
            self.default_directory = default_export_path

        # Define options for model name and size
        self.model_names = ["yolov8", "yolo11", "yolo26"]
        self.model_sizes = ["n", "s", "m", "l"]

        self._setup_ui()
        self._initialize_controls()


    def _setup_ui(self):
        """Sets up the GUI layout and widgets."""
        main_layout = QVBoxLayout()

        # 1. Path Setting Area (Export Directory Path)
        path_layout = QHBoxLayout()
        self.label_dir = QLabel("Output Directory:")
        self.textBox_ExportDirectoryPath = QLineEdit(str(self.default_directory)) # Set default path
        self.button_Browse = QPushButton("Browse")
        
        # Connect to call the dialog when the button is clicked (Handling Request 1)
        self.button_Browse.clicked.connect(self.browse_directory) 

        path_layout.addWidget(self.label_dir)
        path_layout.addWidget(self.textBox_ExportDirectoryPath)
        path_layout.addWidget(self.button_Browse)
        main_layout.addLayout(path_layout)

        # 2. Model Setting Area (Model Name & Size)
        model_setting_layout = QHBoxLayout()
        
        # Model Name ComboBox
        model_name_label = QLabel("Model Name:")
        self.comboBox_ModelName = QComboBox()
        self.comboBox_ModelName.addItems(self.model_names) 
        
        # Model Size ComboBox
        model_size_label = QLabel("Size:")
        self.comboBox_ModelSize = QComboBox()
        self.comboBox_ModelSize.addItems(self.model_sizes) 

        model_setting_layout.addWidget(model_name_label)
        model_setting_layout.addWidget(self.comboBox_ModelName)
        model_setting_layout.addWidget(model_size_label)
        model_setting_layout.addWidget(self.comboBox_ModelSize)

        main_layout.addLayout(model_setting_layout)


        # 3. Run Button Area (Run Button)
        self.button_Run = QPushButton("Execute Export to ONNX Format")
        self.button_Run.setStyleSheet("font-size: 16px; padding: 10px;")
        main_layout.addWidget(self.button_Run)

        # Set the main layout
        self.setLayout(main_layout)

        # --- Connect Signals and Slots ---
        self.comboBox_ModelName.currentTextChanged.connect(lambda text: self._handle_setting_change('model_name', text))
        self.comboBox_ModelSize.currentTextChanged.connect(lambda text: self._handle_setting_change('model_size', text))
        # Use QLineEdit's textChanged to save immediately upon path change
        self.textBox_ExportDirectoryPath.textChanged.connect(lambda text: self._handle_setting_change('export_dir', text))

        # Connect the run button click event
        self.button_Run.clicked.connect(self.run_export)


    def browse_directory(self):
        """Displays a directory selection dialog using QFileDialog (Handling Request 1)."""
        # Use current text as initial directory if available, otherwise use current working directory
        current_path = self.textBox_ExportDirectoryPath.text() if self.textBox_ExportDirectoryPath.text() else os.getcwd()

        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory", current_path)
        if directory:
            self.textBox_ExportDirectoryPath.setText(directory)


    def _initialize_controls(self):
        """Initializes controls based on the contents of the settings file."""
        
        # Path box initialization is already set in __init__, but reconfirming for safety
        initial_path = self.current_config.get("export_dir", "")
        if initial_path and Path(initial_path).is_dir():
            self.textBox_ExportDirectoryPath.setText(initial_path)

        # Initialize Model Name ComboBox 
        initial_name = self.current_config.get("model_name")
        if initial_name in self.model_names:
            index = self.comboBox_ModelName.findText(initial_name) 
            self.comboBox_ModelName.setCurrentIndex(index if index != -1 else 0)

        # Initialize Size ComboBox 
        initial_size = self.current_config.get("model_size")
        if initial_size in self.model_sizes:
            index = self.comboBox_ModelSize.findText(initial_size) 
            self.comboBox_ModelSize.setCurrentIndex(index if index != -1 else 0)


    def _handle_setting_change(self, key, value):
        """Updates the settings and saves to JSON immediately when a control's value changes."""
        # Update the configuration dictionary
        if key == 'export_dir':
            self.current_config[key] = value
        else: # model_name or model_size
            self.current_config[key] = value

        # Save immediately to the JSON file 
        save_settings(self.current_config)


    def get_model_path(self):
        """Generates the full model path based on the selected name and size."""
        name = self.comboBox_ModelName.currentText()
        size = self.comboBox_ModelSize.currentText()

        # Example: "yolov8" + "n" -> Convert to format like "yolov8n-pose.pt"
        model_filename = f"{name}{size}-pose.pt"
        return model_filename

    def run_export(self):
        """Main logic to execute the export process, including directory validation (Handling Request 2)."""
        output_dir_str = self.textBox_ExportDirectoryPath.text()
        
        if not output_dir_str:
            QMessageBox.warning(self, "Error", "Please enter an output directory path.")
            return

        output_dir = Path(output_dir_str)

        # --- Directory Validation and Creation (Handling Request 2) ---
        try:
            if not output_dir.exists():
                print(f"Directory {output_dir} does not exist, attempting to create...")
                # Use exist_ok=False so an exception is raised if it exists but isn't a directory or due to permissions
                output_dir.mkdir(parents=True, exist_ok=False) 
                print("Directory created successfully.")

        except PermissionError:
            QMessageBox.critical(self, "Execution Error", f"You do not have permission to write to the specified path '{output_dir}'. Please select another directory.")
            return
        except OSError as e:
             # Other OS-level errors (e.g., invalid characters for a filename)
            QMessageBox.critical(self, "Execution Error", f"An OS error occurred during directory creation: {e}")
            return

        # ----------------------------------------------

        try:
            # Determine model path and output ONNX path
            model_file_name = self.get_model_path()
            full_model_path = output_dir / model_file_name
            onnx_output_path = output_dir / f"{Path(model_file_name).stem}.onnx"

            print("-" * 40)
            print(f"Starting execution: Exporting {model_file_name} to ONNX.")
            print(f"Output destination: {onnx_output_path}")
            print("-" * 40)


            # Load the model and export (Ultralytics processing)
            model = YOLO(full_model_path)

            # Execute export in ONNX format
            results = model.export(format='onnx', imgsz=640, verbose=False) 
            
            print("\n✅ Export successful!")
            QMessageBox.information(self, "Success", f"The ONNX file has been saved to the following path:\n{onnx_output_path}")

        except Exception as e:
            # Error handling during model loading or export
            print(f"\n❌ A critical error occurred during export: {e}")
            QMessageBox.critical(self, "Execution Error", f"An error occurred while running the export.\nDetails: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModelExporterApp()
    window.show()
    sys.exit(app.exec())
