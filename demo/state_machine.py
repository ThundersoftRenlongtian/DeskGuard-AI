import time
import config


class StateMachine:
    """坐姿状态机：计时、触发、冷却"""

    def __init__(self):
        self.bad_posture_start = None      # 不良坐姿开始时间
        self.sit_start = time.time()       # 开始坐的时间
        self.last_reminder_time = 0        # 上次提醒时间
        self.current_posture = 'unknown'
        self.bad_posture_duration = 0      # 当前不良坐姿持续秒数
        self.total_sit_minutes = 0         # 连续坐的分钟数

    def update(self, posture):
        """更新状态，返回触发事件或 None"""
        now = time.time()
        self.current_posture = posture
        self.total_sit_minutes = (now - self.sit_start) / 60

        # 不良坐姿计时
        if posture != 'normal' and posture != 'unknown':
            if self.bad_posture_start is None:
                self.bad_posture_start = now
            self.bad_posture_duration = now - self.bad_posture_start
        else:
            self.bad_posture_start = None
            self.bad_posture_duration = 0

        # 检查是否在冷却期
        if now - self.last_reminder_time < config.COOLDOWN_MINUTES * 60:
            return None

        # 触发条件1：不良坐姿超过阈值
        if self.bad_posture_duration >= config.POSTURE_ALERT_SECONDS:
            self.last_reminder_time = now
            self.bad_posture_start = None
            return {
                'type': 'posture_alert',
                'posture': posture,
                'duration_sec': int(self.bad_posture_duration),
                'total_sit_min': int(self.total_sit_minutes)
            }

        # 触发条件2：久坐超过阈值
        if self.total_sit_minutes >= config.SIT_DURATION_MINUTES:
            self.last_reminder_time = now
            self.sit_start = now  # 重置计时
            return {
                'type': 'sit_alert',
                'duration_min': int(self.total_sit_minutes),
                'posture': posture
            }

        return None

    def user_stood_up(self):
        """用户起身，重置久坐计时"""
        self.sit_start = time.time()
        self.total_sit_minutes = 0
