import json
from openai import OpenAI
import config


client = OpenAI(
    api_key=config.MIMO_API_KEY,
    base_url=config.MIMO_BASE_URL
)

SYSTEM_PROMPT = """你是 DeskGuard AI 健康教练，运行在用户桌面。

你的职责：
1. 根据用户坐姿传感器数据，生成温和有用的提醒
2. 通过 function calling 控制灯光颜色和发送通知

决策原则：
- 提醒内容要具体（如"肩膀前倾了"比"注意坐姿"好）
- 语气温和鼓励，不制造焦虑
- 简短有用，通知内容不超过30字

当收到坐姿异常数据时，请调用 send_notification 发送提醒，同时调用 set_led_color 设置合适的灯光颜色。"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": "发送桌面弹窗通知给用户",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "通知标题，简短"},
                    "message": {"type": "string", "description": "通知内容，30字以内"},
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
                    "color": {"type": "string", "enum": ["green", "yellow", "red"]},
                },
                "required": ["color"]
            }
        }
    }
]


def call_agent(event):
    """
    调用 AI Agent 处理事件
    event: 来自状态机的触发事件字典
    返回: list of actions [{'function': name, 'args': {...}}, ...]
    """
    user_msg = f"坐姿传感器数据：{json.dumps(event, ensure_ascii=False)}"

    try:
        response = client.chat.completions.create(
            model=config.MIMO_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=200
        )

        actions = []
        msg = response.choices[0].message

        # 解析 function calls
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                actions.append({'function': func_name, 'args': func_args})

        # 如果没有 tool_calls 但有文本回复，用默认通知
        if not actions and msg.content:
            actions.append({
                'function': 'send_notification',
                'args': {'title': '健康提醒', 'message': msg.content[:50]}
            })

        return actions

    except Exception as e:
        print(f"[Agent Error] {e}")
        # 降级：使用本地模板
        return _fallback_action(event)


def _fallback_action(event):
    """API 调用失败时的本地降级方案"""
    templates = {
        'posture_alert': {
            'slouch': '肩膀有点前倾了，试试挺直一下',
            'lean_forward': '身体在往前倾，靠回椅背吧',
            'head_down': '头低得有点多，抬起来看看远处',
            'tilt': '身体有点歪，调整一下坐正',
        },
        'sit_alert': '已经连续坐了很久，站起来活动一下吧'
    }

    if event['type'] == 'posture_alert':
        msg = templates['posture_alert'].get(event['posture'], '注意调整坐姿')
    else:
        msg = templates['sit_alert']

    return [
        {'function': 'send_notification', 'args': {'title': '健康提醒', 'message': msg}},
        {'function': 'set_led_color', 'args': {'color': 'yellow'}}
    ]
