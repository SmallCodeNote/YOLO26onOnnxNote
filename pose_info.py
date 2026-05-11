from __future__ import annotations
from dataclasses import dataclass
from typing import List
import math
import numpy as np

from PoseKeyPoints import PoseKeyPoints


# ---------------------------------------------------------
# PoseInfo_ConfidenceLevel
# ---------------------------------------------------------
@dataclass
class PoseInfo_ConfidenceLevel:
    Bbox: float = 0.16

    Nose: float = 0.6
    Head: float = 0.6
    Eye: float = 0.6
    Ear: float = 0.6
    Shoulder: float = 0.6
    Elbow: float = 0.6
    Wrist: float = 0.6
    Hip: float = 0.6
    Knee: float = 0.6
    Ankle: float = 0.6

    def initialize_param_from_text_lines(self, text: str) -> None:
        if not text:
            return
        lines = text.replace("\r\n", "\n").strip("\n").split("\n")
        for line in lines:
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            key, val = parts
            try:
                v = float(val)
            except ValueError:
                continue
            if hasattr(self, key):
                setattr(self, key, v)

    def set_keypoints_confidence_level(self, common: float) -> None:
        self.Nose = common
        self.Head = common
        self.Eye = common
        self.Ear = common
        self.Shoulder = common
        self.Elbow = common
        self.Wrist = common
        self.Hip = common
        self.Knee = common
        self.Ankle = common

    def param_to_text_lines(self) -> List[str]:
        return [
            f"Bbox\t{self.Bbox}",
            f"Nose\t{self.Nose}",
            f"Head\t{self.Head}",
            f"Eye\t{self.Eye}",
            f"Ear\t{self.Ear}",
            f"Shoulder\t{self.Shoulder}",
            f"Elbow\t{self.Elbow}",
            f"Wrist\t{self.Wrist}",
            f"Hip\t{self.Hip}",
            f"Knee\t{self.Knee}",
            f"Ankle\t{self.Ankle}",
        ]


# ---------------------------------------------------------
# Bbox Class
# ---------------------------------------------------------
@dataclass
class Bbox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @classmethod
    def from_output(cls, output: np.ndarray):
        return cls(
            float(output[0]),
            float(output[1]),
            float(output[2]),
            float(output[3]),
            float(output[4]),
        )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return self.x1 + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y1 + self.height / 2

    def overlap(self, other: "Bbox") -> float:
        xx1 = max(self.x1, other.x1)
        yy1 = max(self.y1, other.y1)
        xx2 = min(self.x2, other.x2)
        yy2 = min(self.y2, other.y2)

        w = max(0.0, xx2 - xx1)
        h = max(0.0, yy2 - yy1)
        inter = w * h
        union = self.width * self.height + other.width * other.height - inter
        return inter / union if union > 0 else 0.0

    def merge(self, other: "Bbox") -> None:
        if other.score > self.score:
            self.x1, self.y1, self.x2, self.y2, self.score = (
                other.x1, other.y1, other.x2, other.y2, other.score
            )


# ---------------------------------------------------------
# PoseInfo
# ---------------------------------------------------------
class PoseInfo:
    def __init__(self, det_row: np.ndarray, confidence_level: PoseInfo_ConfidenceLevel):
        """
        det_row: YOLO pose output for one person with shape (57,)
        """
        self.Bbox = Bbox.from_output(det_row)
        self.KeyPoints = PoseKeyPoints(det_row, confidence_level=confidence_level)

    # ---- CSV Header ----
    @staticmethod
    def csv_header() -> str:
        return (
            "Bbox.X,Bbox.Y,Bbox.W,Bbox.H,"
            "Head.X,Head.Y,"
            "WristLeft.X,WristLeft.Y,"
            "WristRight.X,WristRight.Y,"
            "ElbowLeftAngle,ElbowLeftLength,WristLeftLength,"
            "ElbowRightAngle,ElbowRightLength,WristRightLength,"
            "KneeLeftAngle,KneeLeftLength,AnkleLeftLength,"
            "KneeRightAngle,KneeRightLength,AnkleRightLength,"
            "EyeWidth,EarWidth,ShoulderWidth,HipWidth,"
            "TorsoLength,"
            "HeadYawAngle,"
            "TorsoSlope,"
            "ShoulderSlope,"
            "ThighLeftTorsoAngle,"
            "ThighRightTorsoAngle,"
            "ArmLeftTorsoAngle,"
            "ArmRightTorsoAngle"
        )

    # ---- CSV 1 Row ----
    def to_csv_line(self) -> str:
        kp = self.KeyPoints
        return (
            f"{self.Bbox.center_x:.0f},{self.Bbox.center_y:.0f},{self.Bbox.width:.0f},{self.Bbox.height:.0f},"
            f"{kp.Head().x:.0f},{kp.Head().y:.0f},"
            f"{kp.WristLeft.x:.0f},{kp.WristLeft.y:.0f},"
            f"{kp.WristRight.x:.0f},{kp.WristRight.y:.0f},"
            f"{kp.ElbowLeftAngle:.0f},{kp.ElbowLeftLength:.0f},{kp.WristLeftLength:.0f},"
            f"{kp.ElbowRightAngle:.0f},{kp.ElbowRightLength:.0f},{kp.WristRightLength:.0f},"
            f"{kp.KneeLeftAngle:.0f},{kp.KneeLeftLength:.0f},{kp.AnkleLeftLength:.0f},"
            f"{kp.KneeRightAngle:.0f},{kp.KneeRightLength:.0f},{kp.AnkleRightLength:.0f},"
            f"{kp.EyeWidth:.0f},{kp.EarWidth:.0f},{kp.ShoulderWidth:.0f},{kp.HipWidth:.0f},"
            f"{kp.TorsoLength:.0f},"
            f"{kp.HeadYawAngle:.0f},"
            f"{kp.TorsoSlope:.0f},"
            f"{kp.ShoulderSlope:.0f},"
            f"{kp.ThighLeftTorsoAngle:.0f},"
            f"{kp.ThighRightTorsoAngle:.0f},"
            f"{kp.ArmLeftTorsoAngle:.0f},"
            f"{kp.ArmRightTorsoAngle:.0f}"
        )
