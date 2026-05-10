# PoseKeyPoints.py
from __future__ import annotations
import math
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Iterable


# ---------------------------------------------------------
# KeyPoint Class Definition
# ---------------------------------------------------------
@dataclass
class KeyPoint:
    x: float = 0.0
    y: float = 0.0
    confidence: float = 0.0

    @classmethod
    def from_output(cls, output: np.ndarray, start_index: int, kp_index: int):
        base = start_index + 6 + kp_index * 3
        return cls(
            float(output[base]),
            float(output[base + 1]),
            float(output[base + 2]),
        )

    @property
    def position(self):
        return int(self.x), int(self.y)

    def get_rectangle(self, diameter: float):
        r = diameter / 2
        return int(self.x - r), int(self.y - r), int(diameter), int(diameter)

    def merge(self, other: "KeyPoint"):
        if other.confidence > self.confidence:
            self.x = other.x
            self.y = other.y
            self.confidence = other.confidence


# ---------------------------------------------------------
# Confidence Level Definition
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


# ---------------------------------------------------------
# PoseKeyPoints Class Definition
# ---------------------------------------------------------
class PoseKeyPoints:
    def __init__(self, output: np.ndarray, start_index: int = 0,
                 confidence_level: PoseInfo_ConfidenceLevel | None = None):

        self.confidence_level = confidence_level or PoseInfo_ConfidenceLevel()
        self._init_points(output, start_index)

    def _init_points(self, output, start_index):
        self.Nose = KeyPoint.from_output(output, start_index, 0)
        self.EyeLeft = KeyPoint.from_output(output, start_index, 1)
        self.EyeRight = KeyPoint.from_output(output, start_index, 2)
        self.EarLeft = KeyPoint.from_output(output, start_index, 3)
        self.EarRight = KeyPoint.from_output(output, start_index, 4)
        self.ShoulderLeft = KeyPoint.from_output(output, start_index, 5)
        self.ShoulderRight = KeyPoint.from_output(output, start_index, 6)
        self.ElbowLeft = KeyPoint.from_output(output, start_index, 7)
        self.ElbowRight = KeyPoint.from_output(output, start_index, 8)
        self.WristLeft = KeyPoint.from_output(output, start_index, 9)
        self.WristRight = KeyPoint.from_output(output, start_index, 10)
        self.HipLeft = KeyPoint.from_output(output, start_index, 11)
        self.HipRight = KeyPoint.from_output(output, start_index, 12)
        self.KneeLeft = KeyPoint.from_output(output, start_index, 13)
        self.KneeRight = KeyPoint.from_output(output, start_index, 14)
        self.AnkleLeft = KeyPoint.from_output(output, start_index, 15)
        self.AnkleRight = KeyPoint.from_output(output, start_index, 16)

    # ---------------------------------------------------------
    # Aggregated Points (Combining multiple keypoints into one)
    # ---------------------------------------------------------
    def _keypoint_sum(self, c, p1, p2):
        if p1.confidence >= c and p2.confidence >= c:
            return KeyPoint((p1.x + p2.x) * 0.5, (p1.y + p2.y) * 0.5, min(p1.confidence, p2.confidence))
        elif p1.confidence >= c:
            return KeyPoint(p1.x, p1.y, p1.confidence)
        elif p2.confidence >= c:
            return KeyPoint(p2.x, p2.y, p2.confidence)
        return KeyPoint()

    def _keypoint_ave(self, c, pts: Iterable[KeyPoint]):
        xs, ys, confs = [], [], []
        for p in pts:
            if p.confidence >= c:
                xs.append(p.x)
                ys.append(p.y)
                confs.append(p.confidence)
        if not xs:
            return KeyPoint()
        return KeyPoint(sum(xs)/len(xs), sum(ys)/len(ys), min(confs))

    # ---------------------------------------------------------
    # Logical Name Points (Derived points/features)
    # ---------------------------------------------------------
    def Head(self): return self._keypoint_ave(self.confidence_level.Head,
                                              [self.Nose, self.EyeLeft, self.EyeRight, self.EarLeft, self.EarRight])

    def Eye(self): return self._keypoint_sum(self.confidence_level.Eye, self.EyeLeft, self.EyeRight)
    def Ear(self): return self._keypoint_sum(self.confidence_level.Ear, self.EarLeft, self.EarRight)
    def Shoulder(self): return self._keypoint_sum(self.confidence_level.Shoulder, self.ShoulderLeft, self.ShoulderRight)
    def Hip(self): return self._keypoint_sum(self.confidence_level.Hip, self.HipLeft, self.HipRight)

    # ---------------------------------------------------------
    # Angle Calculation (KeyPointAngle)
    # ---------------------------------------------------------
    def _angle_three_points(self, p0, p1, p2, c0, c1, c2):
        if p0.confidence < c0 or p1.confidence < c1 or p2.confidence < c2:
            return -1.0
        v1x = p1.x - p0.x
        v1y = p1.y - p0.y
        v2x = p2.x - p0.x
        v2y = p2.y - p0.y
        dot = v1x*v2x + v1y*v2y
        mag1 = math.sqrt(v1x*v1x + v1y*v1y)
        mag2 = math.sqrt(v2x*v2x + v2y*v2y)
        if mag1 == 0 or mag2 == 0:
            return -1.0
        return math.degrees(math.acos(dot / (mag1 * mag2)))

    # ---------------------------------------------------------
    # Length Calculation (KeyPointLength)
    # ---------------------------------------------------------
    def _length_two_points(self, p0, p1, c0, c1):
        if p0.confidence < c0 or p1.confidence < c1:
            return -1.0
        dx = p1.x - p0.x
        dy = p1.y - p0.y
        return math.sqrt(dx*dx + dy*dy)

    # ---------------------------------------------------------
    # Width Calculation (KeyPointAngleXLength)
    # ---------------------------------------------------------
    def _angle_x_length(self, p0, p1, p2, c0, c1, c2):
        if p0.confidence < c0 or p1.confidence < c1 or p2.confidence < c2:
            return 0.0
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        length = math.sqrt(dx*dx + dy*dy)
        v1x = p1.x - p0.x
        v1y = p1.y - p0.y
        v2x = p2.x - p0.x
        v2y = p2.y - p0.y
        cross = v1x*v2y - v1y*v2x
        return length if cross >= 0 else -length

    # ---------------------------------------------------------
    # Properties (Derived measurements)
    # ---------------------------------------------------------
    @property
    def ElbowLeftAngle(self):
        return self._angle_three_points(
            self.ElbowLeft, self.ShoulderLeft, self.WristLeft,
            self.confidence_level.Elbow, self.confidence_level.Shoulder, self.confidence_level.Wrist
        )

    @property
    def ElbowRightAngle(self):
        return self._angle_three_points(
            self.ElbowRight, self.ShoulderRight, self.WristRight,
            self.confidence_level.Elbow, self.confidence_level.Shoulder, self.confidence_level.Wrist
        )

    @property
    def KneeLeftAngle(self):
        return self._angle_three_points(
            self.KneeLeft, self.HipLeft, self.AnkleLeft,
            self.confidence_level.Knee, self.confidence_level.Hip, self.confidence_level.Ankle
        )

    @property
    def KneeRightAngle(self):
        return self._angle_three_points(
            self.KneeRight, self.HipRight, self.AnkleRight,
            self.confidence_level.Knee, self.confidence_level.Hip, self.confidence_level.Ankle
        )

    @property
    def WristLeftLength(self):
        return self._length_two_points(self.ElbowLeft, self.WristLeft,
                                       self.confidence_level.Elbow, self.confidence_level.Wrist)

    @property
    def WristRightLength(self):
        return self._length_two_points(self.ElbowRight, self.WristRight,
                                       self.confidence_level.Elbow, self.confidence_level.Wrist)

    @property
    def ElbowLeftLength(self):
        return self._length_two_points(self.ShoulderLeft, self.ElbowLeft,
                                       self.confidence_level.Shoulder, self.confidence_level.Elbow)

    @property
    def ElbowRightLength(self):
        return self._length_two_points(self.ShoulderRight, self.ElbowRight,
                                       self.confidence_level.Shoulder, self.confidence_level.Elbow)

    @property
    def KneeLeftLength(self):
        return self._length_two_points(self.HipLeft, self.KneeLeft,
                                       self.confidence_level.Hip, self.confidence_level.Knee)

    @property
    def KneeRightLength(self):
        return self._length_two_points(self.HipRight, self.KneeRight,
                                       self.confidence_level.Hip, self.confidence_level.Knee)

    @property
    def AnkleLeftLength(self):
        return self._length_two_points(self.KneeLeft, self.AnkleLeft,
                                       self.confidence_level.Knee, self.confidence_level.Ankle)

    @property
    def AnkleRightLength(self):
        return self._length_two_points(self.KneeRight, self.AnkleRight,
                                       self.confidence_level.Knee, self.confidence_level.Ankle)

    @property
    def TorsoLength(self):
        return self._length_two_points(self.Shoulder(), self.Hip(),
                                       self.confidence_level.Shoulder, self.confidence_level.Hip)

    @property
    def EyeWidth(self):
        return self._angle_x_length(self.Shoulder(), self.EyeRight, self.EyeLeft,
                                    self.confidence_level.Shoulder, self.confidence_level.Eye, self.confidence_level.Eye)

    @property
    def EarWidth(self):
        return self._angle_x_length(self.Shoulder(), self.EarRight, self.EarLeft,
                                    self.confidence_level.Shoulder, self.confidence_level.Ear, self.confidence_level.Ear)

    @property
    def ShoulderWidth(self):
        return self._angle_x_length(self.Head(), self.ShoulderLeft, self.ShoulderRight,
                                    self.confidence_level.Head, self.confidence_level.Shoulder, self.confidence_level.Shoulder)

    @property
    def HipWidth(self):
        return self._angle_x_length(self.Shoulder(), self.HipLeft, self.HipRight,
                                    self.confidence_level.Shoulder, self.confidence_level.Hip, self.confidence_level.Hip)

    @property
    def HeadYawAngle(self):
        d_left = self._length_two_points(self.Nose, self.EarLeft,
                                         self.confidence_level.Nose, self.confidence_level.Ear)
        d_right = self._length_two_points(self.Nose, self.EarRight,
                                          self.confidence_level.Nose, self.confidence_level.Ear)
        s = d_left + d_right
        if s == 0:
            return -1
        ratio = (d_right - d_left) / s
        try:
            return math.degrees(math.asin(ratio))
        except ValueError:
            return -1

    @property
    def TorsoSlope(self):
        sh = self.Shoulder()
        hp = self.Hip()
        return math.degrees(math.atan2(sh.x - hp.x, sh.y - hp.y))

    @property
    def ShoulderSlope(self):
        return math.degrees(math.atan2(self.ShoulderRight.y - self.ShoulderLeft.y,
                                       self.ShoulderRight.x - self.ShoulderLeft.x))

    @property
    def ThighLeftTorsoAngle(self):
        return self._angle_three_points(
            self.HipLeft, self.KneeLeft, self.Hip(),
            self.confidence_level.Hip, self.confidence_level.Knee, self.confidence_level.Hip
        )

    @property
    def ThighRightTorsoAngle(self):
        return self._angle_three_points(
            self.HipRight, self.KneeRight, self.Hip(),
            self.confidence_level.Hip, self.confidence_level.Knee, self.confidence_level.Hip
        )

    @property
    def ArmLeftTorsoAngle(self):
        return self._angle_three_points(
            self.ShoulderLeft, self.ElbowLeft, self.Shoulder(),
            self.confidence_level.Shoulder, self.confidence_level.Elbow, self.confidence_level.Shoulder
        )

    @property
    def ArmRightTorsoAngle(self):
        return self._angle_three_points(
            self.ShoulderRight, self.ElbowRight, self.Shoulder(),
            self.confidence_level.Shoulder, self.confidence_level.Elbow, self.confidence_level.Shoulder
        )

    # ---------------------------------------------------------
    # Bone Drawing (OpenCV Visualization)
    # ---------------------------------------------------------
    def draw_bone_cv2(self, img, confidence_level=0.6, diameter=8):
        conf = confidence_level

        def line_if(a, b):
            if a.confidence >= conf and b.confidence >= conf:
                cv2.line(img, a.position, b.position, (255, 0, 0), 2)

        line_if(self.Nose, self.EyeLeft)
        line_if(self.EyeLeft, self.EarLeft)
        line_if(self.Nose, self.EyeRight)
        line_if(self.EyeRight, self.EarRight)
        line_if(self.ShoulderLeft, self.ShoulderRight)
        line_if(self.ShoulderLeft, self.ElbowLeft)
        line_if(self.ElbowLeft, self.WristLeft)
        line_if(self.ShoulderRight, self.ElbowRight)
        line_if(self.ElbowRight, self.WristRight)
        line_if(self.HipLeft, self.HipRight)
        line_if(self.HipLeft, self.KneeLeft)
        line_if(self.KneeLeft, self.AnkleLeft)
        line_if(self.HipRight, self.KneeRight)
        line_if(self.KneeRight, self.AnkleRight)

        def circle_if(p, color):
            if p.confidence >= conf:
                cv2.circle(img, p.position, diameter//2, color, -1)

        circle_if(self.Nose, (0, 255, 255))
        circle_if(self.EyeLeft, (255, 182, 193))
        circle_if(self.EyeRight, (203, 192, 255))
        circle_if(self.EarLeft, (255, 182, 193))
        circle_if(self.EarRight, (203, 192, 255))
        circle_if(self.ShoulderLeft, (255, 182, 193))
        circle_if(self.ShoulderRight, (203, 192, 255))
        circle_if(self.ElbowLeft, (255, 182, 193))
        circle_if(self.ElbowRight, (203, 192, 255))
        circle_if(self.WristLeft, (255, 182, 193))
        circle_if(self.WristRight, (203, 192, 255))
        circle_if(self.HipLeft, (255, 182, 193))
        circle_if(self.HipRight, (203, 192, 255))
        circle_if(self.KneeLeft, (255, 182, 193))
        circle_if(self.KneeRight, (203, 192, 255))
        circle_if(self.AnkleLeft, (255, 182, 193))
        circle_if(self.AnkleRight, (203, 192, 255))
