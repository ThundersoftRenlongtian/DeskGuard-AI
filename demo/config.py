# DeskGuard AI Demo 配置

# MiMo API
MIMO_API_KEY = "tp-citq4zz53l21dntvf6oyh831kauktdf558hmiw1tibzuyexo"
MIMO_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MIMO_MODEL = "MiMo-MoE-RL-2503"  # 可根据实际模型名调整

# 坐姿检测阈值
SLOUCH_ANGLE_THRESHOLD = 155        # 肩部角度 < 此值 = 驼背
HEAD_DOWN_THRESHOLD = 20            # 头部俯角 > 此值 = 低头
SHOULDER_TILT_THRESHOLD = 0.05      # 左右肩高度差比例
LEAN_FORWARD_THRESHOLD = 0.08       # 前倾偏移比例

# 提醒策略
POSTURE_ALERT_SECONDS = 60          # 不良坐姿持续60秒触发
SIT_DURATION_MINUTES = 45           # 久坐45分钟触发
COOLDOWN_MINUTES = 10               # 提醒后冷却时间（分钟）

# 摄像头
CAMERA_INDEX = 0                    # 正面摄像头编号（0=默认）
SIDE_CAMERA_INDEX = 1               # 侧面摄像头编号（-1=不使用）
