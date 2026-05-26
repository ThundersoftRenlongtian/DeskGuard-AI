# DeskGuard AI 渲染图生成 Prompt

## 用于 Midjourney / DALL-E / 通义万相 / Stable Diffusion

### Prompt 1: 产品使用场景图（主图）

**English (Midjourney/DALL-E):**
```
A modern minimalist office desk scene, a programmer sitting in front of a 27-inch monitor with correct upright posture, on top of the monitor there is a slim black aluminum LED light bar device (similar to BenQ ScreenBar shape) with a tiny camera lens in the center glowing soft cyan blue, the device projects translucent holographic cyan scan lines downward onto the person, showing detected skeleton keypoints (joints marked with glowing dots connected by lines) overlaid on the person's upper body like AR augmented reality, the desk has keyboard mouse and coffee cup, soft ambient lighting, dark modern office background, product photography style, photorealistic, 8K, studio lighting --ar 16:9 --v 6
```

**中文 (通义万相/文心一格):**
```
现代简约办公桌场景，一位程序员坐在27寸显示器前保持正确坐姿，显示器顶部安装了一个纤薄的黑色铝合金LED灯条设备（类似明基ScreenBar形态），灯条中央有一个微型摄像头镜头发出柔和的青蓝色光芒，设备向下投射半透明的全息青色扫描线条照射在人身上，显示检测到的骨骼关键点（关节用发光圆点标记并用线条连接）叠加在人体上半身像AR增强现实效果，桌面有键盘鼠标和咖啡杯，柔和环境光照明，深色现代办公室背景，产品摄影风格，照片级真实，8K画质，工作室布光
```

### Prompt 2: 产品特写图

**English:**
```
Close-up product photography of a slim minimalist black aluminum light bar device mounted on top of a computer monitor edge, the device is approximately 30cm long and 2cm thick with rounded ends, center has a small camera lens with cyan LED ring light, along the bar there are evenly spaced RGB LEDs showing gradient from green to cyan, subtle ambient glow, dark background, studio product shot, macro details visible, brushed aluminum texture, photorealistic, 8K --ar 16:9
```

### Prompt 3: AI检测效果图

**English:**
```
Split screen comparison showing AI posture detection, left side shows person with correct upright posture with green skeleton overlay and green status indicator, right side shows same person slouching forward with red skeleton overlay and warning indicator, skeleton shown as connected dots on joints (head, shoulders, elbows, hips), cyan holographic scan lines visible, dark tech background with subtle grid, futuristic UI elements, data visualization style --ar 16:9
```

### Prompt 4: 产品爆炸图/内部结构

**English:**
```
Technical exploded view diagram of a slim monitor-mounted AI health device, showing internal components separated: small camera module, LED strip PCB, main processor chip (labeled NPU), WiFi antenna, ambient light sensor, temperature sensor, aluminum housing shell, USB-C port, all floating in space with labels, dark blueprint background with cyan accent lines, technical illustration style, isometric view, clean and modern --ar 16:9
```

---

## 使用建议

1. 生成后将图片放入 `d:\thundersoft\images\` 目录
2. 在 index.html 中替换 SVG placeholder 为 `<img>` 标签
3. 建议尺寸：1920×1080 或 1200×675（16:9）
4. 建议格式：WebP（体积小）或 PNG（透明背景）
