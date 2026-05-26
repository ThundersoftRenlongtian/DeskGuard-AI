import cv2
import numpy as np
import os

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
    PoseLandmarksConnections,
)
import mediapipe as mp

# 模型文件路径（与本文件同目录）
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")

# 33 个关键点的连接关系
POSE_CONNECTIONS = PoseLandmarksConnections.POSE_LANDMARKS


class _Landmark:
    """兼容旧 API 的关键点包装"""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class PoseDetector:
    """MediaPipe PoseLandmarker (Task API) 封装"""

    def __init__(self):
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self._frame_ts = 0

    def detect(self, frame):
        """检测姿态，返回关键点列表或 None"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts += 33  # ~30fps, 递增时间戳
        result = self.landmarker.detect_for_video(mp_image, self._frame_ts)
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = []
            for lm in result.pose_landmarks[0]:
                landmarks.append(_Landmark(lm.x, lm.y, lm.z, lm.visibility))
            return landmarks
        return None

    def draw(self, frame, landmarks):
        """在画面上绘制骨骼连线"""
        if landmarks is None:
            return frame
        h, w = frame.shape[:2]
        # 绘制连线
        for conn in POSE_CONNECTIONS:
            p1 = landmarks[conn.start]
            p2 = landmarks[conn.end]
            if p1.visibility > 0.5 and p2.visibility > 0.5:
                x1, y1 = int(p1.x * w), int(p1.y * h)
                x2, y2 = int(p2.x * w), int(p2.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 200), 2)
        # 绘制关键点
        for lm in landmarks:
            if lm.visibility > 0.5:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 212, 255), -1)
        return frame

    def get_point(self, landmarks, idx, frame_shape):
        """获取指定关键点的像素坐标"""
        h, w = frame_shape[:2]
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h]), lm.visibility
