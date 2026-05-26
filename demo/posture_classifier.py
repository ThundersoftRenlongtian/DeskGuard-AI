import numpy as np
import config


class PostureClassifier:
    """基于关键点角度的坐姿分类"""

    # MediaPipe Pose 关键点索引
    NOSE = 0
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    def classify(self, landmarks, frame_shape):
        """
        分类当前坐姿
        返回: (类别名, 详情字典)
        类别: 'normal', 'slouch', 'lean_forward', 'head_down', 'tilt'
        """
        if landmarks is None:
            return 'unknown', {}

        h, w = frame_shape[:2]
        points = {}
        for name, idx in [
            ('nose', self.NOSE),
            ('left_ear', self.LEFT_EAR), ('right_ear', self.RIGHT_EAR),
            ('left_shoulder', self.LEFT_SHOULDER), ('right_shoulder', self.RIGHT_SHOULDER),
            ('left_hip', self.LEFT_HIP), ('right_hip', self.RIGHT_HIP),
        ]:
            lm = landmarks[idx]
            points[name] = np.array([lm.x * w, lm.y * h])

        # 肩膀和髋部中点
        shoulder_mid = (points['left_shoulder'] + points['right_shoulder']) / 2
        hip_mid = (points['left_hip'] + points['right_hip']) / 2
        ear_mid = (points['left_ear'] + points['right_ear']) / 2

        details = {}

        # 1. 驼背检测：耳朵-肩膀-髋部角度
        angle = self._angle(ear_mid, shoulder_mid, hip_mid)
        details['shoulder_angle'] = angle

        if angle < config.SLOUCH_ANGLE_THRESHOLD:
            return 'slouch', details

        # 2. 前倾检测：肩膀相对髋部的前移
        torso_len = np.linalg.norm(shoulder_mid - hip_mid)
        if torso_len > 0:
            forward_ratio = (shoulder_mid[0] - hip_mid[0]) / torso_len
            details['forward_ratio'] = forward_ratio
            # 注意：摄像头镜像，需考虑方向
            if abs(forward_ratio) > config.LEAN_FORWARD_THRESHOLD:
                return 'lean_forward', details

        # 3. 低头检测：鼻子到肩中点的俯角
        nose_to_shoulder = points['nose'] - shoulder_mid
        head_angle = np.degrees(np.arctan2(nose_to_shoulder[1], -abs(nose_to_shoulder[0])))
        details['head_angle'] = head_angle

        if nose_to_shoulder[1] > 0:  # 鼻子低于肩膀
            # 计算相对于垂直方向的偏移
            vertical_offset = (points['nose'][1] - shoulder_mid[1]) / h
            if vertical_offset > 0.05:
                return 'head_down', details

        # 4. 歪坐检测：左右肩高度差
        shoulder_diff = abs(points['left_shoulder'][1] - points['right_shoulder'][1]) / h
        details['shoulder_tilt'] = shoulder_diff

        if shoulder_diff > config.SHOULDER_TILT_THRESHOLD:
            return 'tilt', details

        return 'normal', details

    def _angle(self, a, b, c):
        """计算三点角度 (b为顶点)"""
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.degrees(np.arccos(cos_angle))

    def classify_side(self, landmarks, frame_shape):
        """
        侧面摄像头的坐姿分类（更准确的驼背/前倾检测）
        侧面视角下 x 轴 = 前后方向，y 轴 = 上下方向
        返回: (类别名, 详情字典)
        """
        if landmarks is None:
            return 'unknown', {}

        h, w = frame_shape[:2]
        points = {}
        for name, idx in [
            ('nose', self.NOSE),
            ('left_ear', self.LEFT_EAR), ('right_ear', self.RIGHT_EAR),
            ('left_shoulder', self.LEFT_SHOULDER), ('right_shoulder', self.RIGHT_SHOULDER),
            ('left_hip', self.LEFT_HIP), ('right_hip', self.RIGHT_HIP),
        ]:
            lm = landmarks[idx]
            points[name] = np.array([lm.x * w, lm.y * h])

        # 侧面视角：取可见度高的一侧
        ear = points['left_ear']
        shoulder = points['left_shoulder']
        hip = points['left_hip']

        details = {}

        # 1. 驼背/前倾：耳-肩-髋的角度（侧面更精准）
        side_angle = self._angle(ear, shoulder, hip)
        details['side_angle'] = side_angle

        if side_angle < config.SLOUCH_ANGLE_THRESHOLD:
            return 'slouch', details

        # 2. 头部前伸：耳朵相对肩膀的水平偏移
        #    侧面视角中 x 方向就是前后
        head_forward = abs(ear[0] - shoulder[0])
        torso_len = np.linalg.norm(shoulder - hip)
        if torso_len > 0:
            head_forward_ratio = head_forward / torso_len
            details['head_forward_ratio'] = head_forward_ratio
            if head_forward_ratio > 0.4:  # 头部前伸超过躯干40%
                return 'lean_forward', details

        # 3. 低头：鼻子明显低于耳朵
        nose = points['nose']
        nose_drop = (nose[1] - ear[1]) / h
        details['nose_drop'] = nose_drop
        if nose_drop > 0.06:
            return 'head_down', details

        return 'normal', details


POSTURE_NAMES = {
    'normal': '✅ 坐姿正常',
    'slouch': '⚠️ 驼背',
    'lean_forward': '⚠️ 前倾',
    'head_down': '⚠️ 低头',
    'tilt': '⚠️ 歪坐',
    'unknown': '❓ 未检测到'
}

POSTURE_COLORS = {
    'normal': (0, 200, 0),
    'slouch': (0, 180, 255),
    'lean_forward': (0, 180, 255),
    'head_down': (0, 180, 255),
    'tilt': (0, 180, 255),
    'unknown': (128, 128, 128),
}
