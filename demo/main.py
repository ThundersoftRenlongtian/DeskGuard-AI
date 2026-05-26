import cv2
import time
import threading
import numpy as np
from PIL import ImageFont, ImageDraw, Image

from pose_detector import PoseDetector
from posture_classifier import PostureClassifier, POSTURE_NAMES, POSTURE_COLORS
from state_machine import StateMachine
from agent import call_agent
from notifier import send_notification, set_led_color, get_led_color, LED_RGB
import config

# 加载中文字体
_FONT_PATH = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
_FONT_LARGE = ImageFont.truetype(_FONT_PATH, 28)
_FONT_SMALL = ImageFont.truetype(_FONT_PATH, 18)


def put_chinese_text(img, text, pos, font, color_bgr):
    """在 OpenCV 图像上绘制中文"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # BGR -> RGB
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(pos, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def execute_actions(actions):
    """执行 Agent 返回的动作列表"""
    for action in actions:
        func = action['function']
        args = action['args']
        if func == 'send_notification':
            send_notification(args.get('title', '提醒'), args.get('message', ''))
            print(f"  [通知] {args.get('title')}: {args.get('message')}")
        elif func == 'set_led_color':
            set_led_color(args.get('color', 'green'))
            print(f"  [LED] → {args.get('color')}")


def main():
    print("=" * 50)
    print("  DeskGuard AI Demo - 启动中...")
    print("  按 Q 退出 | 按 R 重置久坐计时")
    print("=" * 50)

    detector = PoseDetector()
    classifier = PostureClassifier()
    state = StateMachine()

    # 正面摄像头
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("[错误] 无法打开正面摄像头！请检查摄像头是否被占用。")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 侧面摄像头（可选）
    side_cap = None
    side_detector = None
    if config.SIDE_CAMERA_INDEX >= 0:
        side_cap = cv2.VideoCapture(config.SIDE_CAMERA_INDEX)
        if side_cap.isOpened():
            side_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            side_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            side_detector = PoseDetector()
            print("[双摄] 侧面摄像头已连接")
        else:
            print("[双摄] 侧面摄像头未找到，使用单摄模式")
            side_cap = None

    agent_busy = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 镜像翻转
        frame = cv2.flip(frame, 1)

        # 正面姿态检测
        landmarks = detector.detect(frame)

        # 绘制骨骼
        frame = detector.draw(frame, landmarks)

        # 正面坐姿分类
        posture, details = classifier.classify(landmarks, frame.shape)

        # 侧面摄像头增强检测
        side_frame = None
        side_posture = 'unknown'
        if side_cap is not None:
            ret_s, side_frame = side_cap.read()
            if ret_s:
                side_landmarks = side_detector.detect(side_frame)
                side_frame = side_detector.draw(side_frame, side_landmarks)
                side_posture, side_details = classifier.classify_side(side_landmarks, side_frame.shape)

                # 融合策略：如果侧面检测到问题而正面没有，以侧面为准
                if posture == 'normal' and side_posture != 'normal' and side_posture != 'unknown':
                    posture = side_posture
                    details.update(side_details)

        # 更新状态机
        event = state.update(posture)

        # 如果有触发事件，异步调用 Agent
        if event and not agent_busy:
            agent_busy = True
            print(f"\n[触发] {event['type']} - {event.get('posture', '')}")

            def _call(ev):
                nonlocal agent_busy
                actions = call_agent(ev)
                execute_actions(actions)
                agent_busy = False

            t = threading.Thread(target=_call, args=(event,), daemon=True)
            t.start()

        # 正常坐姿时灯光恢复绿色
        if posture == 'normal' and get_led_color() != 'green':
            set_led_color('green')

        # --- 绘制 UI ---
        h, w = frame.shape[:2]

        # 坐姿状态（中文）
        label = POSTURE_NAMES.get(posture, posture)
        color = POSTURE_COLORS.get(posture, (255, 255, 255))
        frame = put_chinese_text(frame, label, (20, 10), _FONT_LARGE, color)

        # 计时信息
        sit_info = f"久坐: {int(state.total_sit_minutes)}分钟"
        frame = put_chinese_text(frame, sit_info, (20, 50), _FONT_SMALL, (200, 200, 200))

        if state.bad_posture_duration > 0:
            bad_info = f"不良: {int(state.bad_posture_duration)}s / {config.POSTURE_ALERT_SECONDS}s"
            frame = put_chinese_text(frame, bad_info, (20, 75), _FONT_SMALL, (0, 180, 255))

        # 数据来源标识
        src_label = "正面+侧面" if (side_cap and side_posture != 'unknown') else "正面"
        frame = put_chinese_text(frame, src_label, (20, 100), _FONT_SMALL, (180, 180, 180))

        # LED 灯光模拟（右上角色块）
        led_color = LED_RGB.get(get_led_color(), (0, 200, 0))
        cv2.rectangle(frame, (w - 60, 10), (w - 10, 60), led_color, -1)
        cv2.rectangle(frame, (w - 60, 10), (w - 10, 60), (255, 255, 255), 1)
        cv2.putText(frame, "LED", (w - 55, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # 显示正面画面
        cv2.imshow('DeskGuard AI - Front', frame)

        # 显示侧面画面（如果有）
        if side_frame is not None:
            side_label = POSTURE_NAMES.get(side_posture, side_posture)
            side_color = POSTURE_COLORS.get(side_posture, (255, 255, 255))
            side_frame = put_chinese_text(side_frame, f"侧面: {side_label}", (20, 10), _FONT_SMALL, side_color)
            cv2.imshow('DeskGuard AI - Side', side_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            state.user_stood_up()
            set_led_color('green')
            print("[重置] 久坐计时已重置")

    cap.release()
    if side_cap:
        side_cap.release()
    cv2.destroyAllWindows()
    print("\nDeskGuard AI 已退出。")


if __name__ == '__main__':
    main()
