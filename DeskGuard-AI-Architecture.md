# DeskGuard AI — 技术方案详解

## AI Agent 桌面健康效率助手 · 完整技术实现文档

> **竞赛：** 中科创达 "AI+智能硬件" 创意征集  
> **提交人：** 田仁龙  
> **当前版本：** PC Demo（已实现并可运行）  
> **核心技术栈：** MediaPipe Pose + 小米 MiMo AI Agent (Function Calling) + 双摄像头融合

---

## 一、技术架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   DeskGuard AI 系统架构（已实现）                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │          🧠 AI Agent 决策层 — 小米 MiMo 大模型                    │   │
│  │                                                                   │   │
│  │  System Prompt (健康教练人设)                                     │   │
│  │       ↓                                                           │   │
│  │  接收事件 → 推理决策 → Function Calling 输出动作                  │   │
│  │                                                                   │   │
│  │  Tools:  send_notification(title, message)                       │   │
│  │          set_led_color(color)                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         ▲  结构化事件                    │  动作指令                     │
│         │  (坐姿类别+持续时间)            ▼                             │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐   │
│  │   👁️ 感知层 — MediaPipe Pose  │  │     📢 执行层 — 本地输出       │   │
│  │                              │  │                              │   │
│  │  正面摄像头 → 33关键点检测   │  │  Windows 桌面通知 (plyer)    │   │
│  │  侧面摄像头 → 33关键点检测   │  │  LED 灯光模拟 (OpenCV)      │   │
│  │       ↓                      │  │  终端日志输出                │   │
│  │  角度计算 → 坐姿分类         │  │                              │   │
│  │  状态机 → 触发事件           │  │                              │   │
│  └──────────────────────────────┘  └──────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  ⏱️ 状态管理层 — 本地状态机                       │   │
│  │                                                                   │   │
│  │  不良坐姿持续计时 → 60秒触发 posture_alert                       │   │
│  │  总久坐时间累计  → 45分钟触发 sit_alert                          │   │
│  │  冷却机制       → 每次提醒后 10分钟不重复                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 二、核心技术详解

### 2.1 MediaPipe PoseLandmarker（姿态检测引擎）

**是什么：** Google 开源的实时人体姿态检测模型，能从一帧图像中定位 33 个人体关键点。

**我们怎么用的：**

```python
# pose_detector.py 核心逻辑

from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

# 使用 VIDEO 模式（逐帧处理，带跟踪）
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=RunningMode.VIDEO,
    num_poses=1,                         # 只检测一人
    min_pose_detection_confidence=0.5,   # 检测置信度阈值
    min_tracking_confidence=0.5,         # 跟踪置信度阈值
)
landmarker = PoseLandmarker.create_from_options(options)

# 每帧调用
result = landmarker.detect_for_video(mp_image, timestamp_ms)
# 输出: 33 个归一化坐标点 (x, y, z, visibility)
```

**33 个关键点包括：** 鼻子、左/右眼、左/右耳、左/右肩、左/右肘、左/右手腕、左/右髋、左/右膝、左/右脚踝等。我们重点使用以下 7 个点进行坐姿分析：

| 序号 | 关键点 | 用途 |
|------|--------|------|
| 0 | 鼻子 | 低头检测 |
| 7/8 | 左/右耳 | 头部位置参考 |
| 11/12 | 左/右肩 | 驼背角度、歪坐检测 |
| 23/24 | 左/右髋 | 躯干倾斜参考 |

**模型规格：**
- 文件大小：~4MB（pose_landmarker_lite.task）
- 推理耗时：~15-30ms/帧 (CPU)
- 精度：适合坐姿场景（非极端角度）

---

### 2.2 坐姿分类算法（基于角度规则）

**是什么：** 不依赖额外模型，直接用关键点坐标计算几何关系来判断坐姿。

**我们怎么用的：**

```python
# posture_classifier.py 四种不良坐姿的判定逻辑

# ① 驼背检测：耳-肩-髋 三点角度
angle = _angle(ear_mid, shoulder_mid, hip_mid)
if angle < 155°:  → 驼背

# ② 前倾检测：肩膀相对髋部的水平偏移
forward_ratio = (shoulder_x - hip_x) / torso_length
if forward_ratio > 8%:  → 前倾

# ③ 低头检测：鼻子低于肩膀中点的垂直距离
vertical_offset = (nose_y - shoulder_y) / frame_height
if offset > 5%:  → 低头

# ④ 歪坐检测：左右肩膀高度差
shoulder_diff = |left_shoulder_y - right_shoulder_y| / frame_height
if diff > 5%:  → 歪坐
```

**为什么不用模型分类：**
- 训练数据难收集（坐姿标注成本高）
- 几何规则可解释性强，方便调参
- 延迟极低（纯数学计算，<1ms）
- 对硬件端侧部署友好

---

### 2.3 双摄像头融合（精度增强）

**为什么要双摄像头：**

单正面摄像头的致命缺陷 —— **没有深度信息**：
- 用户前倾 10cm，正面看变化很小
- 用户驼背时肩膀角度在正面投影不明显
- 侧面视角天然适合检测前后方向的姿态变化

**融合策略：**

```python
# main.py 双摄融合逻辑

# 正面摄像头 → 擅长检测：歪坐、基本驼背
posture_front = classifier.classify(front_landmarks)

# 侧面摄像头 → 擅长检测：驼背、前倾、低头
posture_side = classifier.classify_side(side_landmarks)

# 融合规则：侧面检测到问题但正面没有 → 信任侧面
if posture_front == 'normal' and posture_side != 'normal':
    final_posture = posture_side  # 侧面更准确
```

**侧面检测特殊优势：**

```python
# posture_classifier.py - classify_side()

# 侧面视角中 x 轴 = 人体前后方向
# 头部前伸比例 = 耳朵到肩膀的水平距离 / 躯干长度
head_forward_ratio = |ear_x - shoulder_x| / torso_length
if ratio > 40%:  → 头部前伸（典型的看屏幕姿态）

# 侧面角度更精确
side_angle = angle(ear, shoulder, hip)  # 侧面看驼背角度变化更明显
```

**实际摆放建议：**
- 正面摄像头：显示器顶部或笔电摄像头
- 侧面摄像头：桌面左/右侧 60-90° 位置
- 没有侧面摄像头时自动降级为单摄模式

---

### 2.4 状态机（事件触发控制）

**是什么：** 管理时间维度的逻辑——什么时候该触发提醒、什么时候该冷却。

**我们怎么用的：**

```python
# state_machine.py 核心逻辑

class StateMachine:
    def update(self, posture):
        # 每帧调用，更新计时器

        # 1. 久坐累计（无论坐姿好坏）
        self.total_sit_minutes += elapsed / 60

        # 2. 不良坐姿持续时间
        if posture != 'normal' and posture != 'unknown':
            self.bad_posture_duration += elapsed
        else:
            self.bad_posture_duration = 0  # 恢复正常就清零

        # 3. 触发条件判断
        if self.bad_posture_duration >= 60:  # 不良坐姿持续60秒
            return {'type': 'posture_alert', 'posture': posture, ...}

        if self.total_sit_minutes >= 45:  # 久坐45分钟
            return {'type': 'sit_alert', 'duration': 45, ...}

        # 4. 冷却机制：触发后10分钟不重复
        if time_since_last_alert < 600:
            return None  # 冷却中，不触发
```

**设计考虑：**
- 偶尔调整坐姿不应误触发 → 需要持续 60 秒才报警
- 反复提醒会让人烦 → 10 分钟冷却期
- 用户站起来应重置 → 手动按 R 或检测到离开

---

### 2.5 小米 MiMo AI Agent（智能决策中枢）

**是什么：** 小米发布的大语言模型，支持 OpenAI 兼容协议和 Function Calling。

**我们怎么用的：**

```python
# agent.py 核心逻辑

from openai import OpenAI

client = OpenAI(
    api_key="tp-citq4zz53l21dntvf6oyh831kauktdf558hmiw1tibzuyexo",
    base_url="https://token-plan-cn.xiaomimimo.com/v1"
)

# System Prompt — 定义 Agent 人设
SYSTEM_PROMPT = """你是 DeskGuard AI 的健康教练 Agent。
当收到坐姿异常事件时，你需要：
1. 判断严重程度
2. 用 send_notification 发送个性化的温馨提醒
3. 用 set_led_color 设置灯光状态
语气要友善鼓励，不要制造焦虑。"""

# 定义 Agent 可调用的工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": "发送桌面通知提醒用户",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "通知标题"},
                    "message": {"type": "string", "description": "通知内容"}
                },
                "required": ["title", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_led_color",
            "description": "设置LED灯颜色反映当前状态",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "enum": ["green", "yellow", "red"],
                        "description": "green=正常, yellow=警告, red=严重"
                    }
                },
                "required": ["color"]
            }
        }
    }
]

# 调用流程
response = client.chat.completions.create(
    model="MiMo-MoE-RL-2503",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"事件: {event_type}, 坐姿: {posture}, 持续: {duration}秒"}
    ],
    tools=tools,
    tool_choice="auto"  # Agent 自己决定调用哪些工具
)

# Agent 返回 Function Call → 本地执行
for tool_call in response.choices[0].message.tool_calls:
    if tool_call.function.name == "send_notification":
        # 弹出 Windows 桌面通知
        send_notification(args['title'], args['message'])
    elif tool_call.function.name == "set_led_color":
        # 改变 LED 灯颜色
        set_led_color(args['color'])
```

**为什么用 Function Calling 而不是直接模板：**

| 方案 | 缺点 | Agent + Function Calling 的优势 |
|------|------|------|
| 固定模板 "你驼背了，请调整" | 千篇一律，用户很快免疫 | 每次生成不同的提醒文案 |
| if/else 决定提醒方式 | 规则写死，扩展困难 | Agent 自己决定调什么工具 |
| 硬编码灯光颜色 | 无法感知上下文 | Agent 根据严重程度选颜色 |

**实际效果示例：**
```
输入事件: posture_alert, 坐姿: slouch, 持续: 65秒
Agent 输出:
  → send_notification("坐姿提醒", "检测到你驼背超过1分钟了，试试挺直腰背，深呼吸三次放松肩膀吧~")
  → set_led_color("yellow")
```

**Fallback 机制：** 如果 API 调用失败（网络问题），使用本地模板替代：

```python
FALLBACK_TEMPLATES = {
    'posture_alert': [
        {'function': 'send_notification', 'args': {'title': '坐姿提醒', 'message': '注意调整坐姿'}},
        {'function': 'set_led_color', 'args': {'color': 'yellow'}},
    ],
    ...
}
```

---

### 2.6 通知与输出系统

**桌面通知（plyer + ctypes）：**

```python
# notifier.py

from plyer import notification
import ctypes

def send_notification(title, message):
    try:
        # 方案1: plyer 跨平台通知
        notification.notify(title=title, message=message, timeout=8)
    except:
        # 方案2: Windows MessageBox 兜底
        ctypes.windll.user32.MessageBoxW(0, message, title, 0x40)
```

**LED 灯光模拟：**
- 在 OpenCV 窗口右上角绘制色块模拟 LED 灯
- 绿色 = 一切正常
- 黄色 = 姿态警告
- 红色 = 久坐警报
- 未来硬件版本将驱动实体 WS2812B LED 灯带

---

### 2.7 PIL 中文渲染（OpenCV 中文支持）

**问题：** OpenCV 的 `putText()` 不支持中文字符（显示为问号）。

**解决方案：** 通过 Pillow 库渲染中文到图像：

```python
from PIL import ImageFont, ImageDraw, Image

# 加载 Windows 自带微软雅黑字体
font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 28)

def put_chinese_text(img, text, pos, font, color_bgr):
    # OpenCV BGR → PIL RGB
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color_rgb)
    # PIL RGB → OpenCV BGR
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
```

---

## 三、数据流与时序

```
每帧处理流程 (~30fps):

摄像头帧 (640×480 BGR)
    │
    ▼
cv2.flip() 镜像翻转
    │
    ├──── 正面帧 ────────────────────────┐
    │                                     │
    ▼                                     ▼
MediaPipe PoseLandmarker          MediaPipe PoseLandmarker
(正面 33 关键点)                    (侧面 33 关键点)
    │                                     │
    ▼                                     ▼
PostureClassifier.classify()      PostureClassifier.classify_side()
(正面: 歪坐/基本驼背)              (侧面: 精确驼背/前倾/低头)
    │                                     │
    └────────── 融合 ─────────────────────┘
                  │
                  ▼
          最终坐姿分类结果
                  │
                  ▼
          StateMachine.update()
          (计时 + 条件判断)
                  │
            ┌─────┴──────┐
            │ 无事件     │ 有事件
            ▼            ▼
         正常渲染     异步线程调用 MiMo Agent
                         │
                         ▼
                  Agent 返回 Function Calls
                         │
                         ▼
                  执行: 通知 + LED
```

**触发 → Agent 响应延迟：** ~1-3秒（取决于网络和 API 响应速度）

---

## 四、项目文件结构

```
d:\thundersoft\demo\
├── main.py                      # 主循环：摄像头 → 检测 → 分类 → Agent → 渲染
├── config.py                    # 所有可调参数（API Key、阈值、摄像头编号）
├── pose_detector.py             # MediaPipe PoseLandmarker 封装
├── posture_classifier.py        # 坐姿分类器（正面 + 侧面两套规则）
├── state_machine.py             # 计时器 + 触发逻辑 + 冷却机制
├── agent.py                     # MiMo AI Agent 调用 + Function Calling
├── notifier.py                  # 桌面通知 + LED 状态管理
├── pose_landmarker_lite.task    # MediaPipe 模型文件 (~4MB)
└── requirements.txt             # Python 依赖
```

---

## 五、依赖技术栈

| 技术 | 版本 | 用途 | 为什么选它 |
|------|------|------|------|
| **MediaPipe** | 0.10.35 | 人体姿态检测 | Google 出品，精度高，免费，Task API 简洁 |
| **OpenCV** | 4.8+ | 摄像头读取 + UI 渲染 | 工业标准，稳定可靠 |
| **OpenAI SDK** | 1.0+ | 调用 MiMo API | MiMo 兼容 OpenAI 协议，直接复用 |
| **小米 MiMo** | MoE-RL-2503 | AI Agent 推理 | 国产大模型，支持 Function Calling |
| **Pillow** | 10+ | 中文文字渲染 | 解决 OpenCV 不支持中文的问题 |
| **plyer** | 2.1+ | 跨平台桌面通知 | 无需额外配置，一行代码弹通知 |
| **NumPy** | — | 向量/角度计算 | 数值计算标准库 |
| **Python** | 3.12 | 开发语言 | 生态丰富，原型开发快 |

---

## 六、可调参数说明

```python
# config.py 中所有可调参数

# 检测灵敏度
SLOUCH_ANGLE_THRESHOLD = 155     # 越大越敏感（160=轻微驼背就报）
HEAD_DOWN_THRESHOLD = 20         # 越小越敏感
SHOULDER_TILT_THRESHOLD = 0.05   # 越小越敏感（0.03=轻微歪就报）
LEAN_FORWARD_THRESHOLD = 0.08    # 越小越敏感

# 提醒策略
POSTURE_ALERT_SECONDS = 60       # 不良坐姿持续多久触发（可改为30快速测试）
SIT_DURATION_MINUTES = 45        # 久坐多久触发
COOLDOWN_MINUTES = 10            # 两次提醒间隔

# 摄像头
CAMERA_INDEX = 0                 # 正面摄像头（0=默认）
SIDE_CAMERA_INDEX = 1            # 侧面摄像头（-1=不使用）
```

---

## 七、隐私设计

| 措施 | 说明 |
|------|------|
| 图像不离开本地 | 摄像头帧仅在内存中处理，处理完即丢弃 |
| Agent 不接收图像 | 仅发送文字事件（"slouch, 65秒"），绝不上传视频/图片 |
| 模型在本地运行 | MediaPipe 模型完全离线，不联网 |
| 可物理断开 | 侧面摄像头随时拔除，不影响运行 |

---

## 八、从 Demo 到产品的路径

| 阶段 | 当前 Demo | 硬件原型 | 量产产品 |
|------|-----------|----------|----------|
| 感知 | PC 摄像头 + MediaPipe CPU | 树莓派4B + USB摄像头 | RV1106 NPU + GC2053 |
| AI | MiMo API (云端) | MiMo API (云端) | 端侧规则 + 云端 Agent |
| 输出 | Windows 通知 + OpenCV色块 | GPIO LED + 通知 | WS2812B LED灯带 + App |
| 形态 | 桌面窗口 | 原型板 | 显示器顶灯形态 |

---

## 九、运行方式

```bash
cd d:\thundersoft\demo
pip install -r requirements.txt
python main.py
```

- 按 **Q** 退出
- 按 **R** 重置久坐计时
- 正面摄像头 = 编号 0（默认）
- 侧面摄像头 = 编号 1（可选，`config.py` 中设 `-1` 禁用）

---

*DeskGuard AI — 看得见姿态、懂得了状态、会主动关心你的 AI 桌面健康助手。*
