# DeskGuard AI — 本地实现方案书

## 一、项目目标

在 Windows 电脑上实现一个可运行的 Demo，用 PC 摄像头实时检测坐姿，结合 AI Agent 智能提醒，跑通完整功能链路。

---

## 二、功能清单

| 编号 | 功能 | 说明 | 优先级 |
|------|------|------|--------|
| F1 | 实时坐姿检测 | 摄像头+MediaPipe检测33个身体关键点 | P0 |
| F2 | 坐姿分类 | 根据关键点角度判断：正常/驼背/前倾/歪坐/低头 | P0 |
| F3 | 久坐计时 | 连续在座时间累计，超过阈值触发 | P0 |
| F4 | AI Agent 提醒 | 触发后调 DeepSeek API 生成个性化提醒内容 | P0 |
| F5 | 桌面弹窗 | Windows 系统通知显示 AI 生成的提醒 | P0 |
| F6 | LED 灯光模拟 | 在界面上用色块模拟灯光状态（绿/黄/红） | P1 |
| F7 | 数据记录 | SQLite 记录每次检测结果和提醒 | P1 |
| F8 | AI 日报 | 每天结束时调 API 生成健康总结报告 | P2 |
| F9 | 用户偏好学习 | 记录用户对提醒的反馈，调整阈值和频率 | P2 |
| F10 | 实体 LED 灯 | （选做）Arduino/ESP32 USB 串口控制 WS2812B | P3 |

---

## 三、技术方案

### 3.1 整体架构

```
┌──────────────────────────────────────────────────────┐
│                   DeskGuard AI Demo                    │
│                                                      │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │ 摄像头   │───▶│ MediaPipe   │───▶│ 坐姿分类器  │ │
│  │ (cv2)   │    │ Pose (33pt) │    │ (角度规则)  │ │
│  └─────────┘    └─────────────┘    └─────────────┘ │
│                                          │           │
│                                          ▼           │
│                                    ┌───────────┐    │
│                                    │ 状态机     │    │
│                                    │ 计时/触发  │    │
│                                    └───────────┘    │
│                                          │ 触发     │
│                                          ▼           │
│  ┌─────────────┐    ┌────────────────────────────┐ │
│  │ DeepSeek    │◀───│ Agent 调用                  │ │
│  │ API         │───▶│ (生成提醒/报告/控制指令)    │ │
│  └─────────────┘    └────────────────────────────┘ │
│                                          │           │
│                          ┌───────────────┼────────┐ │
│                          ▼               ▼        ▼ │
│                    ┌──────────┐  ┌──────────┐ ┌────┐│
│                    │ 桌面弹窗  │  │ LED模拟  │ │日志││
│                    └──────────┘  └──────────┘ └────┘│
└──────────────────────────────────────────────────────┘
```

### 3.2 坐姿分类算法

使用 MediaPipe Pose 的 33 个关键点，计算以下角度：

| 检测项 | 关键点 | 判断方法 |
|--------|--------|----------|
| 驼背 | 耳朵-肩膀-髋部 | 三点角度 < 155° |
| 前倾 | 肩膀相对于髋部的前移距离 | 水平偏移 > 阈值 |
| 低头 | 鼻子到肩中点的垂直角度 | 俯角 > 20° |
| 歪坐 | 左右肩高度差 | 差值 > 阈值 |

### 3.3 AI Agent 设计

**模型：** DeepSeek-V3 (或 DeepSeek-Chat)  
**接口：** OpenAI 兼容格式 (base_url: https://api.deepseek.com)  
**Function Calling Tools：**

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": "发送桌面弹窗通知给用户",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "通知标题"},
                    "message": {"type": "string", "description": "通知内容"},
                    "priority": {"type": "string", "enum": ["gentle", "normal", "urgent"]}
                },
                "required": ["title", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_led_color",
            "description": "设置LED灯颜色",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {"type": "string", "enum": ["green", "yellow", "red", "breathing_green"]},
                    "duration_sec": {"type": "number"}
                },
                "required": ["color"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_today_stats",
            "description": "获取今日健康统计数据",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]
```

### 3.4 文件结构

```
d:\thundersoft\demo\
├── main.py                 # 主程序入口（摄像头循环+界面）
├── pose_detector.py        # MediaPipe 姿态检测封装
├── posture_classifier.py   # 坐姿分类（角度计算+规则）
├── state_machine.py        # 状态机（计时、触发条件、冷却）
├── agent.py                # DeepSeek Agent 调用 + Function Calling
├── notifier.py             # Windows 桌面通知
├── led_simulator.py        # LED 灯光模拟（GUI色块）
├── data_store.py           # SQLite 数据记录
├── config.py               # 配置（API Key、阈值参数）
├── requirements.txt        # Python 依赖
└── README.md               # 使用说明
```

### 3.5 依赖清单

```
mediapipe>=0.10.0
opencv-python>=4.8.0
openai>=1.0.0          # DeepSeek 兼容 OpenAI SDK
plyer>=2.1.0           # 跨平台桌面通知
```

---

## 四、配置说明

```python
# config.py
DEEPSEEK_API_KEY = "sk-xxx"  # 从 platform.deepseek.com 获取
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# 坐姿阈值
SLOUCH_ANGLE_THRESHOLD = 155      # 肩部角度 < 此值判定为驼背
HEAD_DOWN_THRESHOLD = 20          # 头部俯角 > 此值判定为低头
SHOULDER_TILT_THRESHOLD = 0.05    # 左右肩高度差比例

# 提醒策略
POSTURE_ALERT_SECONDS = 60        # 不良坐姿持续60秒触发
SIT_DURATION_MINUTES = 45         # 连续久坐45分钟触发
COOLDOWN_MINUTES = 10             # 提醒冷却时间
```

---

## 五、运行方式

```bash
cd d:\thundersoft\demo
pip install -r requirements.txt
python main.py
```

启动后：
1. 摄像头画面窗口（显示骨骼点+当前坐姿状态）
2. LED 状态色块（随坐姿变化）
3. 当触发条件满足时，自动调 AI → 弹出 Windows 通知
