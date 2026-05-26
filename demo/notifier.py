import threading
try:
    from plyer import notification
    HAS_PLYER = True
except ImportError:
    HAS_PLYER = False

# Windows fallback
import ctypes


def send_notification(title, message, timeout=5):
    """发送 Windows 桌面通知"""
    if HAS_PLYER:
        try:
            notification.notify(
                title=title,
                message=message,
                app_name='DeskGuard AI',
                timeout=timeout
            )
            return
        except Exception:
            pass

    # Fallback: Windows MessageBox (在新线程中避免阻塞)
    def _show():
        ctypes.windll.user32.MessageBoxW(0, message, title, 0x40)

    t = threading.Thread(target=_show, daemon=True)
    t.start()


# LED 状态（全局，供 main.py 读取显示）
_led_color = 'green'


def set_led_color(color):
    """设置 LED 颜色状态"""
    global _led_color
    _led_color = color


def get_led_color():
    """获取当前 LED 颜色"""
    return _led_color


LED_RGB = {
    'green': (0, 200, 0),
    'yellow': (0, 200, 255),
    'red': (0, 0, 255),
}
