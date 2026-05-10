from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PoseKeyPoints import PoseKeyPoints
from PoseInfo import PoseInfo, PoseInfo_ConfidenceLevel

try:
    # Attempt to import onnxruntime for model inference
    import onnxruntime as ort
except ImportError:
    ort = None


class InferenceEngine:
    """
    Handles the entire inference pipeline, including preprocessing, running the ONNX model, 
    post-processing (filtering and scaling), drawing results, and saving data.
    """
    def __init__(self, onnx_path: str) -> None:
        if ort is None:
            # Ensure that onnxruntime or its DirectML variant is installed
            raise RuntimeError("onnxruntime (or onnxruntime-directml) is required.")

        self.onnx_path = onnx_path
        # Initialize the inference session
        self.session = self._create_session(onnx_path)
        # Get input and output names from the model graph
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Check output shape: must be (1, 300, 57)
        out_shape = self.session.get_outputs()[0].shape
        if tuple(out_shape) != (1, 300, 57):
            raise ValueError(
                f"Unexpected output shape {out_shape}, expected (1, 300, 57)."
            )

    def _create_session(self, onnx_path: str):
        """
        Creates and returns an ONNX Runtime inference session.
        Prioritizes DirectML if available for GPU acceleration.
        """
        providers = []
        # Prefer DirectML if available (for Windows/DirectX accelerated hardware)
        if "DmlExecutionProvider" in (ort.get_available_providers() if ort else []):
            providers.append("DmlExecutionProvider")
        # Fallback to CPU execution provider
        providers.append("CPUExecutionProvider")
        sess_options = ort.SessionOptions()
        return ort.InferenceSession(onnx_path, sess_options, providers=providers)

    def preprocess_image(self, img_bgr: np.ndarray):
        """
        Preprocesses a BGR image for model input (resizing, padding, normalization).
        Returns the input tensor and scaling/padding parameters needed for inverse transformation.
        """
        h, w = img_bgr.shape[:2]
        target = 640  # Target dimension size

        # Calculate scale factor to fit the image into the target square (640x640)
        scale = min(target / w, target / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize the image using linear interpolation
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create a canvas of target size and pad the resized image in the center
        canvas = np.zeros((target, target, 3), dtype=np.uint8)
        pad_x = (target - new_w) // 2
        pad_y = (target - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # Convert to RGB and normalize to float [0.0, 1.0]
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        # Change layout from HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
        img_chw = np.transpose(img_float, (2, 0, 1))
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        input_tensor = np.expand_dims(img_chw, axis=0)

        # Calculate scaling factors for coordinates transformation later
        scale_x = w / new_w
        scale_y = h / new_h

        return input_tensor, scale_x, scale_y, pad_x, pad_y


    @staticmethod
    def filter_and_scale_output(output, scale_x, scale_y, pad_x, pad_y, score_th):
        """
        Post-processes the raw model output: filters detections by confidence 
        and scales coordinates back to the original image dimensions.

        Args:
            output (np.ndarray): Raw model output (1, 300, 57).
            scale_x (float): Scaling factor for X coordinates.
            scale_y (float): Scaling factor for Y coordinates.
            pad_x (int): Horizontal padding applied during preprocessing.
            pad_y (int): Vertical padding applied during preprocessing.
            score_th (float): Minimum confidence threshold to keep a detection.

        Returns: 
            np.ndarray: Filtered and scaled array of detections (N, 57).
        """
        dets = output[0]
        # Filter out detections below the score threshold (column index 4)
        dets = dets[dets[:, 4] >= score_th]

        if dets.size == 0:
            return dets

        dets_scaled = dets.copy()

        # Scale bounding box coordinates (x1, y1, x2, y2)
        # Coordinate transformation: Original_Coord = ((Model_Coord - Padding) * Scale)
        dets_scaled[:, 0] = (dets[:, 0] - pad_x) * scale_x  # x1
        dets_scaled[:, 1] = (dets[:, 1] - pad_y) * scale_y  # y1
        dets_scaled[:, 2] = (dets[:, 2] - pad_x) * scale_x  # x2
        dets_scaled[:, 3] = (dets[:, 3] - pad_y) * scale_y  # y2

        # Scale keypoint coordinates (kp1x, kp1y, ..., kp17x, kp17y)
        for kp in range(17):
            base = 6 + kp * 3 # Starting column index for the current keypoint (x, y, score)
            dets_scaled[:, base]     = (dets[:, base]     - pad_x) * scale_x    # Keypoint X
            dets_scaled[:, base + 1] = (dets[:, base + 1] - pad_y) * scale_y  # Keypoint Y
            # Score remains unchanged

        return dets_scaled

    def infer(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Runs the inference process on a single BGR image.
        Returns: raw output (1, 300, 57), scale_x, scale_y, pad_x, pad_y
        (Note: The original code returned only sx and sy in the type hint but used all four values internally).
        """
        input_tensor, sx, sy, pad_x, pad_y = self.preprocess_image(img_bgr)
        # Run the session with the input tensor
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        return outputs[0], sx, sy, pad_x, pad_y


    @staticmethod
    def format_detection_for_csv(det: np.ndarray) -> list[str]:
        """
        Converts a single detection array (57 values) into a list of strings 
        formatted for CSV output.

        Args:
            det (np.ndarray): Single detection vector (N=57).

        Returns: 
            list[str]: List containing coordinates and scores as formatted strings.
        """
        out = []

        # Bounding box (x1, y1, x2, y2) - converted to integers
        out += [str(int(det[0])), str(int(det[1])), str(int(det[2])), str(int(det[3]))]

        # Bounding box score (formatted to 3 decimal places)
        out.append(f"{det[4]:.3f}")

        # Class ID - converted to integer string
        out.append(str(int(det[5])))

        # Keypoints (x, y, score) × 17
        for kp in range(17):
            base = 6 + kp * 3 # Starting column for the current keypoint
            kx = int(det[base])
            ky = int(det[base + 1])
            ks = f"{det[base + 2]:.3f}" # Keypoint score formatted to 3 decimal places
            out += [str(kx), str(ky), ks]

        return out


    @staticmethod
    def draw_detections(img_bgr: np.ndarray, dets_scaled: np.ndarray) -> np.ndarray:
        """
        Draws bounding boxes and skeletal joints (bones) onto the original image 
        using scaled detection results.

        Args:
            img_bgr (np.ndarray): The original BGR image frame.
            dets_scaled (np.ndarray): Detections already scaled to original image coordinates (N, 57).

        Returns: 
            np.ndarray: The image with detections drawn on it.
        """
        out_img = img_bgr.copy()

        for det in dets_scaled:
            # Extract bounding box and score/class ID
            x1, y1, x2, y2 = det[0:4].astype(int)
            score = float(det[4])
            cls_id = int(det[5])

            # Draw the bounding box (Green color)
            cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Put detection label text above the bounding box
            cv2.putText(
                out_img,
                f"{cls_id}:{score:.3f}",
                (x1, max(0, y1 - 5)), # Ensure text is not drawn off-screen
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            # Draw the skeleton (bones) using PoseKeyPoints
            pose = PoseKeyPoints(det)
            pose.draw_bone_cv2(out_img, confidence_level=0.6, diameter=8)

        return out_img


    @staticmethod
    def write_csv_lines_for_image(csv_path: Path, image_name: str, dets_scaled: np.ndarray):
        """
        Writes detection data for a single image to a CSV file. 
        Appends lines if the file exists; writes header if it's new.

        Args:
            csv_path (Path): Full path to the output CSV file.
            image_name (str): Name of the processed image.
            dets_scaled (np.ndarray): Scaled detection results for the image.
        """
        # Ensure the parent directory exists
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        conf = PoseInfo_ConfidenceLevel()

        is_new_file = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8") as f:
            if is_new_file:
                # Write header if the file is being created for the first time
                f.write("ImageName," + PoseInfo.csv_header() + "\n")
            for det in dets_scaled:
                pose = PoseInfo(det, conf)
                line = pose.to_csv_line()
                # Write data line: ImageName, Data...
                f.write(f"{image_name},{line}\n")

    @staticmethod
    def write_csv_lines_for_video(csv_path: Path, frame_idx: int, dets_scaled: np.ndarray):
        """
        Writes detection data for a video frame to a CSV file. 
        Appends lines if the file exists; writes header if it's new.

        Args:
            csv_path (Path): Full path to the output CSV file.
            frame_idx (int): The index of the current frame.
            dets_scaled (np.ndarray): Scaled detection results for the frame.
        """
        # Ensure the parent directory exists
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        conf = PoseInfo_ConfidenceLevel()

        is_new_file = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8") as f:
            if is_new_file:
                # Write header if the file is being created for the first time
                f.write("Frame," + PoseInfo.csv_header() + "\n")
            for det in dets_scaled:
                pose = PoseInfo(det, conf)
                line = pose.to_csv_line()
                # Write data line: FrameIndex, Data...
                f.write(f"{frame_idx},{line}\n")


    def process_image_files(
        self,
        image_paths: List[Path],
        result_dir_image: Path,
        result_csv_path: Path,
        score_th: float,
    ) -> None:
        """
        Processes a list of image files sequentially. 
        Performs inference, scales results, draws detections, and saves output images/CSV data.
        """
        for img_path in image_paths:
            # Read the image using OpenCV
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[InferenceEngine] Failed to read image: {img_path}")
                continue

            # Run inference and get scaling/padding parameters
            out, sx, sy, pad_x, pad_y = self.infer(img)
            # Filter and scale the raw output coordinates back to original size
            dets_scaled = self.filter_and_scale_output(out, sx, sy, pad_x, pad_y, score_th)

            # Save detection data to CSV file (image mode)
            self.write_csv_lines_for_image(
                result_csv_path,
                img_path.name,
                dets_scaled,
            )

            # Draw detections on the original image and save the result
            drawn = self.draw_detections(img, dets_scaled)
            result_dir_image.mkdir(parents=True, exist_ok=True)
            out_name = img_path.stem + "_draw" + img_path.suffix
            cv2.imwrite(str(result_dir_image / out_name), drawn)

    def process_video_file(
        self,
        video_path: Path,
        result_dir_image: Path,
        result_dir_csv: Path,
        score_th: float,
    ) -> None:
        """
        Processes a video file frame by frame. 
        Performs inference on each frame and saves results to an output video and CSV log.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[InferenceEngine] Failed to open video: {video_path}")
            return

        # Get video properties (FPS, Width, Height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup output directories
        result_dir_image.mkdir(parents=True, exist_ok=True)
        result_dir_csv.mkdir(parents=True, exist_ok=True)

        out_video_path = result_dir_image / f"{video_path.stem}_draw.mp4"
        # Define video writer codec and initialize the VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

        csv_path = result_dir_csv / f"{video_path.stem}.csv"
        # Clear previous CSV content if it exists
        if csv_path.is_file():
            csv_path.unlink()

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read() # Read next frame
                if not ret:
                    break # End of video

                # Run inference on the current frame
                out, sx, sy, pad_x, pad_y = self.infer(frame)
                dets_scaled = self.filter_and_scale_output(out, sx, sy, pad_x, pad_y, score_th)

                # Save detection data to CSV file (video mode)
                self.write_csv_lines_for_video(csv_path, frame_idx, dets_scaled)

                # Draw detections on the frame and write it to the output video
                drawn = self.draw_detections(frame, dets_scaled)
                writer.write(drawn)

                frame_idx += 1
        finally:
            # Release resources cleanly
            cap.release()
            writer.release()
