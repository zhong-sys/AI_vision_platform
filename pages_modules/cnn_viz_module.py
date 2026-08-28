"""CNN 教学可视化模块 —— 卷积核滑动与池化交互演示

纯 Streamlit + NumPy + PIL 实现，零外部依赖。
"""

import time
from pathlib import Path
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from components.experiment_panel import (
    render_experiment_guide,
    render_learning_goals,
    render_observation,
    render_parameter_explanation,
    render_preset_controls,
)
from components.page_header import render_page_header


# ===================== 常量与配置 =====================

_CONV_KERNELS = {
    "边缘检测（Sobel X）": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "边缘检测（Sobel Y）": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
    "边缘检测（Prewitt X）": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32),
    "边缘检测（Prewitt Y）": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32),
    "边缘检测（Laplacian）": np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32),
    "边缘检测（Laplacian 8邻域）": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
    "模糊（均值）": np.ones((3, 3), dtype=np.float32) / 9.0,
    "锐化": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
}
_COMBINED_KERNEL_NAME = "组合边缘检测（Sobel X+Y）"
_FIXED_IMAGE_NAME = "assets/1.jpg"

_PALETTE_BG_DARK = (26, 26, 26)               # #1a1a1a

_POOL_CELL_SIZE = 32
_POOL_SCAN_INTERVAL_SEC = 0.18  # 池化自动播放固定间隔（较慢，便于观察）
_CONV_DISPLAY_WIDTH = 300


CNN_PRESETS = {
    "标准示例": "使用固定示例图和默认边缘核，适合先熟悉逐格卷积。",
    "边缘增强": "切换组合边缘核，观察水平与垂直响应的叠加。",
    "平滑纹理": "使用棋盘纹理和均值核，观察平滑与池化的效果。",
    "池化对照": "保留边缘核，比较最大池化和平均池化的输出差异。",
}


def _cnn_preset_values(preset_name):
    values = {
        "conv_image_source": "具体图像",
        "conv_kernel_name": "边缘检测（Laplacian 8邻域）",
        "conv_enable_gaussian": True,
        "pool_type": "最大池化",
        "conv_scan_speed": 0.01,
    }
    if preset_name == "边缘增强":
        values.update({"conv_kernel_name": _COMBINED_KERNEL_NAME, "conv_enable_gaussian": False})
    elif preset_name == "平滑纹理":
        values.update({
            "conv_image_source": "checkerboard",
            "conv_kernel_name": "模糊（均值）",
            "conv_enable_gaussian": True,
            "pool_type": "平均池化",
        })
    elif preset_name == "池化对照":
        values.update({"pool_type": "平均池化", "conv_scan_speed": 0.02})
    return values


def _apply_cnn_preset():
    preset_name = st.session_state.get("cnn_teaching_preset", "标准示例")
    for key, value in _cnn_preset_values(preset_name).items():
        st.session_state[key] = value
    st.session_state["conv_row"] = 0
    st.session_state["conv_col"] = 0
    st.session_state["pool_window_row"] = 0
    st.session_state["pool_window_col"] = 0
    st.session_state["is_scanning"] = False
    st.session_state["is_pool_scanning"] = False


def _reset_cnn_defaults():
    for key, value in _cnn_preset_values("标准示例").items():
        st.session_state[key] = value
    st.session_state["cnn_teaching_preset"] = "标准示例"
    st.session_state["conv_row"] = 0
    st.session_state["conv_col"] = 0
    st.session_state["pool_window_row"] = 0
    st.session_state["pool_window_col"] = 0
    st.session_state["is_scanning"] = False
    st.session_state["is_pool_scanning"] = False
    st.session_state["scan_position"] = (0, 0)
    st.session_state["pool_scan_position"] = (0, 0)


# ===================== 跨平台字体加载 =====================

def _load_font(size: int):
    """跨平台字体加载，优先 Linux（Noto/WQY），回退 Windows。"""
    font_paths = [
        "assets/LXGWWenKai-Regular.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    for name in ["assets/LXGWWenKai-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ===================== 图像生成与加载 =====================

def generate_demo_image(name: str, size: tuple = (128, 128)) -> Image.Image:
    """生成内置示例灰度图（skimage 不可用时使用）。"""
    h, w = size
    arr = np.zeros((h, w), dtype=np.uint8)
    if name == "camera":
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        rect = ((np.abs(x - cx) < w // 4) & (np.abs(y - cy) < h // 4)).astype(np.uint8) * 200
        grad = (255 * (x / w)).astype(np.uint8)
        arr = ((rect.astype(np.uint16) + grad) // 2).astype(np.uint8)
    elif name == "coins":
        arr.fill(60)
        centers = [(w // 4, h // 4), (3 * w // 4, h // 4),
                   (w // 2, h // 2), (w // 4, 3 * h // 4), (3 * w // 4, 3 * h // 4)]
        for cx, cy in centers:
            y, x = np.ogrid[:h, :w]
            mask = (x - cx) ** 2 + (y - cy) ** 2 < (min(h, w) // 6) ** 2
            arr[mask] = 210
    elif name == "checkerboard":
        s = max(8, min(h, w) // 8)
        for i in range(0, h, s):
            for j in range(0, w, s):
                if ((i // s) + (j // s)) % 2 == 0:
                    arr[i:i + s, j:j + s] = 220
                else:
                    arr[i:i + s, j:j + s] = 40
    elif name == "gradient":
        arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return Image.fromarray(arr, mode="L")


def load_image(source: str, upload_file=None) -> Image.Image:
    """加载并返回 RGB PIL Image（用于彩色显示）。"""
    def _to_rgb_safe(raw_img: Image.Image) -> Image.Image:
        """先处理透明通道再转 RGB，避免透明背景显示为黑底。"""
        if raw_img.mode in ("RGBA", "LA") or ("transparency" in raw_img.info):
            rgba = raw_img.convert("RGBA")
            white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            merged = Image.alpha_composite(white_bg, rgba)
            return merged.convert("RGB")
        return raw_img.convert("RGB")

    if source == "上传图片":
        if upload_file is not None:
            img = _to_rgb_safe(Image.open(upload_file))
        else:
            st.info("当前未上传图片，已回退到 camera 示例图。")
            img = generate_demo_image("camera")
    elif source == "具体图像":
        # 多路径查找 assets/1.jpg（兼容本地开发与 Streamlit Cloud）
        candidates = [
            Path(_FIXED_IMAGE_NAME),                                    # 项目根目录 (Cloud)
            Path(__file__).resolve().parent.parent / _FIXED_IMAGE_NAME,        # pages_modules/ 上级
        ]
        fixed_path = None
        for p in candidates:
            if p.exists():
                fixed_path = p
                break
        if fixed_path and fixed_path.exists():
            img = _to_rgb_safe(Image.open(fixed_path))
        else:
            st.warning("未找到 assets/1.jpg，已回退到 camera 示例图。")
            img = generate_demo_image("camera")
    elif source == "在线灰度图":
        try:
            import urllib.request
            url = "https://upload.wikimedia.org/wikipedia/commons/5/50/Vd-Orig.png"
            with urllib.request.urlopen(url, timeout=5) as resp:
                img = _to_rgb_safe(Image.open(BytesIO(resp.read())))
        except Exception:
            img = generate_demo_image("camera")
    else:
        img = generate_demo_image(source)
    img = _to_rgb_safe(img).resize((128, 128), Image.Resampling.LANCZOS)
    return img


# ===================== 卷积核 =====================

def get_kernel(kernel_name: str, custom_vals: list = None) -> np.ndarray:
    """返回 3×3 卷积核 numpy 数组。"""
    if kernel_name == "自定义" and custom_vals is not None:
        return np.array(custom_vals, dtype=np.float32).reshape(3, 3)
    return _CONV_KERNELS.get(kernel_name, _CONV_KERNELS["边缘检测（Sobel X）"]).copy()


# ===================== 绘制函数 =====================

def _upscale(img: Image.Image, factor: int) -> Image.Image:
    """最近邻放大 PIL 图像。"""
    return img.resize((img.width * factor, img.height * factor), Image.NEAREST)


def draw_receptive_field(img: Image.Image, row: int, col: int) -> Image.Image:
    """在原图上叠加半透明黄色 3×3 感受野与加粗边框，并绘制指向右侧箭头。"""
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    x0, y0 = col, row
    x1, y1 = col + 2, row + 2
    draw.rectangle([x0, y0, x1, y1], fill=(253, 224, 71, 90), outline=(255, 0, 0, 255), width=3)

    # 单条粗箭头：一根主干 + 实心三角箭头尖
    center_y = (y0 + y1) // 2
    shaft_start_x = min(base.width - 1, x1 + 1)
    shaft_end_x = min(base.width - 1, x1 + 8)
    tip_x = min(base.width - 1, x1 + 12)
    tip_half_h = 4
    draw.line([(shaft_start_x, center_y), (shaft_end_x, center_y)], fill=(255, 153, 0, 255), width=3)
    draw.polygon(
        [(tip_x, center_y), (shaft_end_x, center_y - tip_half_h), (shaft_end_x, center_y + tip_half_h)],
        fill=(255, 153, 0, 255),
    )

    return Image.alpha_composite(base, overlay).convert("RGB")


def compute_convolution(image_array: np.ndarray, kernel: np.ndarray, row: int, col: int) -> float:
    """计算单位置卷积结果（标量）。"""
    patch = image_array[row:row + 3, col:col + 3].astype(np.float32)
    return float(np.sum(patch * kernel))


def compute_combined_edge(image_array: np.ndarray, row: int, col: int) -> tuple:
    """计算 Sobel X/Y 合成边缘强度，返回 (gx, gy, magnitude)。"""
    kx = _CONV_KERNELS["边缘检测（Sobel X）"]
    ky = _CONV_KERNELS["边缘检测（Sobel Y）"]
    patch = image_array[row:row + 3, col:col + 3].astype(np.float32)
    gx = float(np.sum(patch * kx))
    gy = float(np.sum(patch * ky))
    mag = float(np.sqrt(gx * gx + gy * gy))
    return gx, gy, mag


def compute_conv_value(image_array: np.ndarray, kernel_name: str, kernel: np.ndarray, row: int, col: int) -> float:
    """统一计算单位置卷积值，支持组合边缘模式。"""
    if kernel_name == _COMBINED_KERNEL_NAME:
        _, _, mag = compute_combined_edge(image_array, row, col)
        return mag
    return compute_convolution(image_array, kernel, row, col)


def compute_full_feature_map(image_array: np.ndarray, kernel_name: str, kernel: np.ndarray) -> np.ndarray:
    """计算整张图的卷积特征图，并进行绝对值归一化（0-255）。"""
    h, w = image_array.shape
    h_out, w_out = h - 2, w - 2
    fm = np.zeros((h_out, w_out), dtype=np.float32)
    for i in range(h_out):
        for j in range(w_out):
            fm[i, j] = compute_conv_value(image_array, kernel_name, kernel, i, j)
    fm = np.abs(fm)
    fm_max = float(np.max(fm)) if fm.size > 0 else 0.0
    if fm_max > 0:
        fm = fm / fm_max * 255.0
    return np.clip(fm, 0.0, 255.0).astype(np.uint8)


def apply_gaussian_blur(image: Image.Image, enabled: bool) -> Image.Image:
    """可选高斯降噪预处理。"""
    if not enabled:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=1.0))


def _heatmap_color(val: float, vmin: float, vmax: float) -> tuple:
    """3×3 热力图配色：正值偏红、负值偏蓝、零值白灰。"""
    if abs(val) < 1e-6:
        return (220, 220, 220)
    if val > 0:
        t = min(1.0, val / vmax) if vmax > 0 else 0
        r = int(220 + (255 - 220) * t)
        g = int(220 * (1 - t * 0.85))
        b = int(220 * (1 - t * 0.85))
    else:
        t = min(1.0, val / vmin) if vmin < 0 else 0
        r = int(220 * (1 - t * 0.85))
        g = int(220 * (1 - t * 0.85))
        b = int(220 + (255 - 220) * t)
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def draw_heatmap_3x3(patch: np.ndarray, scale: int = 60) -> Image.Image:
    """绘制 3×3 像素值热力图，带数值标注。"""
    vmin, vmax = patch.min(), patch.max()
    img = Image.new("RGB", (3 * scale, 3 * scale), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _load_font(max(10, scale // 4))
    for i in range(3):
        for j in range(3):
            val = patch[i, j]
            color = _heatmap_color(val, vmin, vmax)
            x0, y0 = j * scale, i * scale
            draw.rectangle([x0, y0, x0 + scale - 1, y0 + scale - 1], fill=color, outline=(180, 180, 180))
            txt = f"{val:.0f}"
            bbox = draw.textbbox((0, 0), txt, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x0 + (scale - tw) // 2, y0 + (scale - th) // 2), txt, fill=(0, 0, 0), font=font)
    return img


def draw_kernel_matrix(kernel: np.ndarray, scale: int = 60) -> Image.Image:
    """绘制 3×3 卷积核权重矩阵图。"""
    img = Image.new("RGB", (3 * scale, 3 * scale), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    font = _load_font(max(10, scale // 4))
    abs_max = max(abs(kernel.min()), abs(kernel.max()), 1e-6)
    for i in range(3):
        for j in range(3):
            val = kernel[i, j]
            intensity = int(240 * (1 - abs(val) / abs_max))
            if val > 0:
                color = (255, intensity, intensity)
            elif val < 0:
                color = (intensity, intensity, 255)
            else:
                color = (240, 240, 240)
            x0, y0 = j * scale, i * scale
            draw.rectangle([x0, y0, x0 + scale - 1, y0 + scale - 1], fill=color, outline=(150, 150, 150))
            txt = f"{val:.1f}" if abs(val) >= 0.1 else f"{val:.2f}"
            bbox = draw.textbbox((0, 0), txt, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x0 + (scale - tw) // 2, y0 + (scale - th) // 2), txt, fill=(0, 0, 0), font=font)
    return img


def draw_combined_kernel_matrix(scale: int = 60) -> Image.Image:
    """绘制 Sobel X + Sobel Y 双核展示图。"""
    sobel_x = draw_kernel_matrix(_CONV_KERNELS["边缘检测（Sobel X）"], scale=scale)
    sobel_y = draw_kernel_matrix(_CONV_KERNELS["边缘检测（Sobel Y）"], scale=scale)
    gap = max(18, scale // 3)
    label_h = 24
    width = sobel_x.width + sobel_y.width + gap
    height = max(sobel_x.height, sobel_y.height) + label_h
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(11, scale // 4))
    canvas.paste(sobel_x, (0, label_h))
    canvas.paste(sobel_y, (sobel_x.width + gap, label_h))
    draw.text((6, 4), "Sobel X", fill=(30, 41, 59), font=font)
    draw.text((sobel_x.width + gap + 6, 4), "Sobel Y", fill=(30, 41, 59), font=font)
    return canvas


def draw_elementwise_product(patch: np.ndarray, kernel: np.ndarray, scale: int = 60) -> Image.Image:
    """绘制 3×3 逐元素乘积矩阵，并在右下角显示求和结果。"""
    product = patch * kernel
    total = float(np.sum(product))
    footer_h = max(30, scale // 2 + 12)
    img = Image.new("RGB", (3 * scale, 3 * scale + footer_h), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    font = _load_font(max(10, scale // 4))
    font_sum = _load_font(max(11, scale // 4 + 2))

    abs_max = max(abs(product.min()), abs(product.max()), 1e-6)
    vmin, vmax = -abs_max, abs_max

    for i in range(3):
        for j in range(3):
            val = float(product[i, j])
            color = _heatmap_color(val, vmin, vmax)
            x0, y0 = j * scale, i * scale
            draw.rectangle([x0, y0, x0 + scale - 1, y0 + scale - 1], fill=color, outline=(150, 150, 150))
            txt = f"{val:.1f}"
            bbox = draw.textbbox((0, 0), txt, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x0 + (scale - tw) // 2, y0 + (scale - th) // 2), txt, fill=(0, 0, 0), font=font)

    sum_txt = f"sum = {total:.2f}"
    sb = draw.textbbox((0, 0), sum_txt, font=font_sum)
    stw, sth = sb[2] - sb[0], sb[3] - sb[1]
    sx = img.width - stw - 8
    sy = 3 * scale + (footer_h - sth) // 2
    draw.rectangle([0, 3 * scale, img.width - 1, img.height - 1], fill=(238, 238, 238), outline=(200, 200, 200))
    draw.text((sx, sy), sum_txt, fill=(30, 41, 59), font=font_sum)

    return img


def draw_feature_map(feature_map: np.ndarray, current_row: int, current_col: int) -> Image.Image:
    """绘制累积特征图：黑白灰配色，未计算位置深灰，当前像素高亮。"""
    h, w = feature_map.shape
    cell = 20
    img = Image.new("RGB", (w * cell, h * cell), _PALETTE_BG_DARK)
    draw = ImageDraw.Draw(img)

    valid = ~np.isnan(feature_map)
    if valid.any():
        abs_max = float(np.max(np.abs(feature_map[valid])))
    else:
        abs_max = 1.0
    abs_max = max(abs_max, 1e-6)

    for i in range(h):
        for j in range(w):
            val = feature_map[i, j]
            if np.isnan(val):
                gray = 32
            else:
                # 使用绝对值强度做灰度映射，边缘响应越强越亮。
                t = min(1.0, abs(float(val)) / abs_max)
                gray = int(25 + 230 * t)
            color = (gray, gray, gray)
            x0, y0 = j * cell, i * cell
            draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=color)

    # 当前像素白色高亮
    cx = current_col * cell + cell // 2
    cy = current_row * cell + cell // 2
    r = max(3, cell // 3)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(245, 245, 245), outline=(180, 180, 180))

    return img


def _pool_color(val: float, vmin: float, vmax: float) -> tuple:
    """池化输入图配色。"""
    if vmin == vmax:
        return (128, 128, 128)
    t = (val - vmin) / (vmax - vmin)
    r = int(15 + (200 - 15) * t)
    g = int(23 + (220 - 23) * t)
    b = int(42 + (240 - 42) * t)
    return (r, g, b)


def draw_pooling_demo(input_map: np.ndarray, pool_type: str, window_row: int, window_col: int):
    """返回池化演示图：左侧输入带框（RGBA 叠加），右侧输出结果，中间虚线箭头连接。"""
    h, w = input_map.shape
    out_h, out_w = h // 2, w // 2
    cell = _POOL_CELL_SIZE

    # ---- 左侧输入特征图 ----
    in_w_px, in_h_px = w * cell, h * cell
    in_img = Image.new("RGB", (in_w_px, in_h_px), (30, 30, 30))
    in_draw = ImageDraw.Draw(in_img)
    vmin, vmax = input_map.min(), input_map.max()

    for i in range(h):
        for j in range(w):
            color = _pool_color(input_map[i, j], vmin, vmax)
            x0, y0 = j * cell, i * cell
            in_draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=color)
            in_draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], outline=(80, 80, 80))

    # 提取当前 2×2 窗口
    wr, wc = window_row, window_col
    patch = input_map[wr * 2:wr * 2 + 2, wc * 2:wc * 2 + 2]

    # 蓝色半透明填充窗口
    overlay = Image.new("RGBA", (in_w_px, in_h_px), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    bx0, by0 = wc * 2 * cell, wr * 2 * cell
    bx1, by1 = bx0 + 2 * cell - 1, by0 + 2 * cell - 1
    overlay_draw.rectangle([bx0, by0, bx1, by1], fill=(59, 130, 246, 51), outline=(59, 130, 246), width=2)

    # 根据池化类型调整窗口内像素亮度
    in_rgba = in_img.convert("RGBA")
    pixels = np.array(in_rgba)
    if pool_type == "最大池化":
        max_val = patch.max()
        for di in range(2):
            for dj in range(2):
                if patch[di, dj] < max_val - 1e-6:
                    px = (wc * 2 + dj) * cell
                    py = (wr * 2 + di) * cell
                    for yy in range(py, min(py + cell, in_h_px)):
                        for xx in range(px, min(px + cell, in_w_px)):
                            pixels[yy, xx, 0] = pixels[yy, xx, 0] // 2
                            pixels[yy, xx, 1] = pixels[yy, xx, 1] // 2
                            pixels[yy, xx, 2] = pixels[yy, xx, 2] // 2
    in_rgba = Image.fromarray(pixels, mode="RGBA")
    in_combined = Image.alpha_composite(in_rgba, overlay).convert("RGB")

    # ---- 右侧输出特征图 ----
    out_w_px, out_h_px = out_w * cell, out_h * cell
    out_img = Image.new("RGB", (out_w_px, out_h_px), (30, 30, 30))
    out_draw = ImageDraw.Draw(out_img)

    # 计算完整池化结果（用于显示）
    pool_result = np.full((out_h, out_w), np.nan, dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            p = input_map[i * 2:i * 2 + 2, j * 2:j * 2 + 2]
            pool_result[i, j] = float(p.max()) if pool_type == "最大池化" else float(p.mean())

    pmin, pmax = pool_result.min(), pool_result.max()
    for i in range(out_h):
        for j in range(out_w):
            color = _pool_color(pool_result[i, j], pmin, pmax)
            x0, y0 = j * cell, i * cell
            out_draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=color)
            out_draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], outline=(80, 80, 80))

    # 当前输出像素绿色高亮
    ox = wc * cell + cell // 2
    oy = wr * cell + cell // 2
    r = max(4, cell // 3)
    out_draw.ellipse([ox - r, oy - r, ox + r, oy + r], fill=(34, 197, 94), outline=(20, 83, 45), width=2)

    # ---- 合并左右 + 虚线箭头 ----
    gap = 80
    total_w = in_w_px + gap + out_w_px
    total_h = max(in_h_px, out_h_px)
    combined = Image.new("RGB", (total_w, total_h), (15, 15, 15))
    combined.paste(in_combined, (0, (total_h - in_h_px) // 2))
    combined.paste(out_img, (in_w_px + gap, (total_h - out_h_px) // 2))
    cdraw = ImageDraw.Draw(combined)

    # 画虚线箭头
    arrow_y = total_h // 2
    start_x = in_w_px + 5
    end_x = in_w_px + gap - 5
    for x in range(start_x, end_x, 10):
        cdraw.line([(x, arrow_y), (min(x + 5, end_x), arrow_y)], fill=(180, 180, 180), width=2)
    # 箭头尖端
    cdraw.polygon([(end_x, arrow_y - 6), (end_x, arrow_y + 6), (end_x + 8, arrow_y)], fill=(180, 180, 180))

    return combined


def draw_layer_chain() -> Image.Image:
    """静态展示「原图 → 卷积1 → 池化1 → 卷积2 → 池化2」尺寸变化链条。"""
    stages = [
        ("Input\n64×64", 64),
        ("Conv1\n62×62", 62),
        ("Pool1\n31×31", 31),
        ("Conv2\n29×29", 29),
        ("Pool2\n14×14", 14),
    ]
    box_size = 120
    gap = 100
    margin = 40
    label_gap = 50
    total_w = len(stages) * box_size + (len(stages) - 1) * gap + margin * 2
    total_h = box_size + margin * 2 + label_gap
    img = Image.new("RGB", (total_w, total_h), (15, 15, 15))
    draw = ImageDraw.Draw(img)
    font = _load_font(20)
    font_large = _load_font(22)

    colors = [(191, 219, 254), (187, 247, 208), (253, 230, 138), (187, 247, 208), (253, 230, 138)]
    max_stage_size = max(s for _, s in stages)
    for idx, (label, size) in enumerate(stages):
        x = margin + idx * (box_size + gap)
        y = margin
        # 小方块表示特征图尺寸（按最大阶段尺寸等比缩放）
        sq = int(size / max_stage_size * box_size)
        sq = max(16, min(sq, box_size))
        sx = x + (box_size - sq) // 2
        sy = y + (box_size - sq) // 2
        draw.rectangle([sx, sy, sx + sq - 1, sy + sq - 1], fill=colors[idx], outline=(100, 100, 100), width=2)
        # 标签
        bbox = draw.textbbox((0, 0), label, font=font_large)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x + (box_size - tw) // 2, y + box_size + 8), label, fill=(220, 220, 220), font=font_large)
        # 箭头
        if idx < len(stages) - 1:
            ax0 = x + box_size + 4
            ax1 = x + box_size + gap - 4
            ay = y + box_size // 2
            draw.line([(ax0, ay), (ax1, ay)], fill=(150, 150, 150), width=3)
            draw.polygon([(ax1, ay - 6), (ax1, ay + 6), (ax1 + 10, ay)], fill=(150, 150, 150))

    return img


# ===================== 公式格式化 =====================

def format_conv_formula(patch: np.ndarray, kernel: np.ndarray, result: float) -> str:
    """生成卷积运算展开式字符串。"""
    terms = []
    for i in range(3):
        for j in range(3):
            terms.append(f"{patch[i, j]:.1f}×({kernel[i, j]:.1f})")
    return "sum( [" + " + ".join(terms) + "] ) = " + f"{result:.1f}"


def format_combined_formula(gx: float, gy: float, result: float) -> str:
    """生成组合边缘模式的公式字符串。"""
    return f"sqrt( Gx^2 + Gy^2 ) = sqrt({gx:.1f}^2 + {gy:.1f}^2) = {result:.1f}"


# ===================== Session State 管理 =====================

def _reset_conv_state(h_out: int, w_out: int):
    st.session_state.conv_feature_map = np.full((h_out, w_out), np.nan, dtype=np.float32)
    st.session_state.conv_computed = np.zeros((h_out, w_out), dtype=bool)


def _reset_pool_state(h_out: int, w_out: int):
    st.session_state.pool_feature_map = np.full((h_out, w_out), np.nan, dtype=np.float32)
    st.session_state.pool_computed = np.zeros((h_out, w_out), dtype=bool)


def _next_conv_position(row: int, col: int, h_out: int, w_out: int) -> tuple:
    """按行优先顺序返回卷积下一位置与是否到达末尾。"""
    if h_out <= 0 or w_out <= 0:
        return 0, 0, True
    nc = col + 1
    nr = row
    if nc >= w_out:
        nc = 0
        nr = row + 1
    if nr >= h_out:
        return row, col, True
    return nr, nc, False


def _render_conv_controls(max_row: int, max_col: int, h_out: int, w_out: int, button_container=None):
    """渲染卷积扫描控制区，并返回当前位置。"""
    # 自动扫描时，在 slider 前同步当前扫描位置
    if st.session_state.is_scanning:
        sr, sc = st.session_state.scan_position
        st.session_state.conv_row = min(sr, max_row)
        st.session_state.conv_col = min(sc, max_col)

    button_col = button_container if button_container is not None else st.columns([1, 1, 1])[2]
    with button_col:
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1], gap="small")
        with btn_col1:
            if st.button("▶ 自动扫描", key="btn_scan", use_container_width=True):
                st.session_state.is_scanning = True
                st.session_state.scan_position = (st.session_state.conv_row, st.session_state.conv_col)
                st.session_state.conv_last_tick = 0.0
                st.rerun()
        with btn_col2:
            if st.button("→ 单步", key="btn_step", use_container_width=True):
                st.session_state.is_scanning = False
                nr, nc, finished = _next_conv_position(st.session_state.conv_row, st.session_state.conv_col, h_out, w_out)
                if not finished:
                    st.session_state.conv_row = nr
                    st.session_state.conv_col = nc
                    st.session_state.scan_position = (nr, nc)
                st.rerun()
        with btn_col3:
            if st.button("⏹ 停止", key="btn_stop", use_container_width=True):
                st.session_state.is_scanning = False
                st.rerun()

    slider_row1, slider_row2, slider_row3 = st.columns([1, 1, 1], gap="small")
    with slider_row1:
        row = st.slider("垂直位置", 0, max_row, key="conv_row")
    with slider_row2:
        col = st.slider("水平位置", 0, max_col, key="conv_col")
    with slider_row3:
        st.slider("扫描间隔(秒)", 0.005, 0.2, 0.01, 0.005, key="conv_scan_speed")

    return row, col


def _render_conv_visualization(
    img: Image.Image,
    img_arr: np.ndarray,
    kernel: np.ndarray,
    kernel_name: str,
    row: int,
    col: int,
    conv_feature_map: np.ndarray,
):
    """渲染五栏卷积可视化。"""
    patch = img_arr[row:row + 3, col:col + 3]
    current_val = float(conv_feature_map[row, col])
    is_combined = kernel_name == _COMBINED_KERNEL_NAME
    viz_cols = st.columns([2, 1.2, 1.2, 1.2, 2])

    with viz_cols[0]:
        st.markdown("**原图 + 感受野高亮**")
        img_marked = draw_receptive_field(img, row, col)
        ratio = _CONV_DISPLAY_WIDTH / img_marked.width
        new_h = int(img_marked.height * ratio)
        img_display = img_marked.resize((_CONV_DISPLAY_WIDTH, new_h), Image.Resampling.LANCZOS)
        st.image(img_display, use_container_width=False)

    with viz_cols[1]:
        st.markdown("**3×3 感受野热图**")
        st.image(draw_heatmap_3x3(patch, scale=66), use_container_width=False)

    with viz_cols[2]:
        st.markdown("**核权重矩阵**")
        if is_combined:
            st.image(draw_combined_kernel_matrix(scale=33), use_container_width=False)
        else:
            st.image(draw_kernel_matrix(kernel, scale=66), use_container_width=False)

    with viz_cols[3]:
        if is_combined:
            gx, gy, _ = compute_combined_edge(img_arr, row, col)
            st.markdown("**Sobel X/Y 响应合成**")
            st.code(f"Gx = {gx:.2f}\nGy = {gy:.2f}\n边缘强度 = sqrt(Gx^2+Gy^2)", language="text")
        else:
            st.markdown("**逐元素乘积 + 求和**")
            st.image(draw_elementwise_product(patch, kernel, scale=66), use_container_width=False)

    with viz_cols[4]:
        st.markdown("**累积特征图**")
        fm_img = draw_feature_map(conv_feature_map, row, col)
        fm_display = _upscale(fm_img, max(1, 300 // max(fm_img.width, 1)))
        st.image(fm_display, use_container_width=False)

    return patch, current_val


def _render_conv_scan_logic(h_out: int, w_out: int, img_arr: np.ndarray, kernel_name: str, kernel: np.ndarray):
    """渲染卷积扫描进度，并在自动扫描模式下按时间自适应推进。"""
    total_steps = max(1, h_out * w_out)
    done_steps = int(np.count_nonzero(st.session_state.conv_computed))
    progress = min(1.0, done_steps / total_steps)
    text = f"卷积扫描进度：{done_steps}/{total_steps} ({progress * 100:.1f}%)"
    st.progress(progress, text=text)

    if not st.session_state.is_scanning:
        return

    interval = st.session_state.get("conv_scan_speed", 0.03)
    last_tick = float(st.session_state.get("conv_last_tick", 0.0))
    now = time.time()
    elapsed = now - last_tick if last_tick > 0 else interval
    step_budget = max(1, int(elapsed / max(interval, 1e-6)))
    step_budget = min(step_budget, 24)

    sr, sc = st.session_state.scan_position
    st.caption(f"🔄 自动扫描中：第 {sr + 1}/{h_out} 行，第 {sc + 1}/{w_out} 列，当前帧推进 {step_budget} 步")

    finished = False
    finished_row = None
    finished_col = None
    for _ in range(step_budget):
        sr, sc = st.session_state.scan_position
        if 0 <= sr < h_out and 0 <= sc < w_out and not st.session_state.conv_computed[sr, sc]:
            val = compute_conv_value(img_arr, kernel_name, kernel, sr, sc)
            st.session_state.conv_feature_map[sr, sc] = val
            st.session_state.conv_computed[sr, sc] = True

        nr, nc, finished = _next_conv_position(sr, sc, h_out, w_out)
        if finished:
            st.session_state.is_scanning = False
            finished_row, finished_col = sr, sc
            break
        # 只更新 scan_position；下一帧在 _render_conv_controls 里、在 slider 渲染前同步到 conv_row/col
        st.session_state.scan_position = (nr, nc)

    st.session_state.conv_last_tick = now
    if finished and finished_row is not None and finished_col is not None:
        # 扫描结束后定位到最后一个已计算像素；用临时 key 暂存，
        # 下一轮渲染在 slider 创建前再写到 conv_row/col。
        st.session_state.scan_position = (finished_row, finished_col)
        st.session_state.conv_scan_final_row = min(finished_row, max(0, h_out - 1))
        st.session_state.conv_scan_final_col = min(finished_col, max(0, w_out - 1))
        st.rerun()
    if st.session_state.is_scanning:
        st.rerun()


# ===================== Streamlit 页面入口 =====================

def nv_render_cnn_viz():
    """渲染 CNN 交互教学页面。"""
    render_page_header(
        title="卷积神经网络（CNN）",
        module="神经网络 / CNN",
        description="交互式探索卷积核滑动与池化压缩的核心机制",
    )

    render_learning_goals(
        [
            "理解卷积核如何在局部感受野内提取边缘与纹理。",
            "观察高斯预处理、不同卷积核和特征图响应的关系。",
            "区分最大池化与平均池化的信息压缩方式。",
            "把逐格计算与 CNN 的层次化特征提取联系起来。",
        ]
    )
    render_experiment_guide(
        [
            "从“标准示例”开始，先点击单步查看一个 3×3 感受野。",
            "切换一个卷积核，比较公式、响应值和全图特征。",
            "再比较最大池化与平均池化，观察输出尺寸和亮度变化。",
            "最后播放扫描过程，联系局部操作与整幅特征图。",
        ]
    )
    render_preset_controls(
        CNN_PRESETS,
        key="cnn_teaching_preset",
        reset_key="cnn_teaching_reset",
        on_preset_change=_apply_cnn_preset,
        on_reset=_reset_cnn_defaults,
    )

    # ---- 为什么需要它 ----
    st.subheader("为什么需要它？")
    st.info(
        "1000×1000 的彩色图像包含 3×10⁶ 个像素。若直接拍平输入全连接层，参数量 "
        "爆炸且粗暴抹平了像素的空间邻近关系。CNN 用局部窗口扫描，只观察邻居，不看全场。"
        "卷积层负责提取边缘、纹理等局部特征，池化层则压缩空间尺寸、保留关键信息，"
        "二者交替堆叠，逐步从低级边缘走向高级抽象。"
    )

    # ==================== 模块 A：卷积滑动演示 ====================
    st.subheader("核心机制：卷积核滑动提取")

    # 初始化 key session state
    if "conv_image_source" not in st.session_state:
        st.session_state.conv_image_source = "具体图像"
    if "conv_kernel_name" not in st.session_state:
        st.session_state.conv_kernel_name = "边缘检测（Laplacian 8邻域）"
    if "conv_enable_gaussian" not in st.session_state:
        st.session_state.conv_enable_gaussian = True
    if "is_scanning" not in st.session_state:
        st.session_state.is_scanning = False
    if "scan_position" not in st.session_state:
        st.session_state.scan_position = (0, 0)

    # 图像与卷积核参数（第一行）+ 扫描按钮
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.2, 1.6], gap="small")
    with ctrl_col1:
        image_options = ["具体图像", "camera", "coins", "checkerboard", "gradient", "在线灰度图", "上传图片"]
        img_source = st.selectbox("选择图像", image_options, key="conv_image_source")
        upload_file = None
        if img_source == "上传图片":
            upload_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"], key="conv_upload")
        enable_gaussian = st.checkbox("预处理：高斯模糊降噪", key="conv_enable_gaussian")
    with ctrl_col2:
        kernel_options = [_COMBINED_KERNEL_NAME] + list(_CONV_KERNELS.keys()) + ["自定义"]
        kernel_name = st.selectbox("选择卷积核", kernel_options, key="conv_kernel_name")
        custom_kernel = None
        if kernel_name == "自定义":
            st.markdown("<div style='font-size:12px;'>自定义 3×3 权重</div>", unsafe_allow_html=True)
            custom_rows = []
            for i in range(3):
                ccols = st.columns(3)
                row_vals = []
                for j in range(3):
                    val = ccols[j].number_input(
                        f"k{i}{j}", value=0.0, step=0.1, label_visibility="collapsed", key=f"custom_k_{i}_{j}"
                    )
                    row_vals.append(val)
                custom_rows.append(row_vals)
            custom_kernel = [v for r in custom_rows for v in r]

    # 加载图像并计算尺寸
    img_display = load_image(img_source, upload_file)
    img_for_compute = img_display.convert("L")
    img_for_compute = apply_gaussian_blur(img_for_compute, enable_gaussian)
    img_arr = np.array(img_for_compute, dtype=np.float32)
    H, W = img_arr.shape
    h_out, w_out = H - 2, W - 2
    max_row = max(0, h_out - 1)
    max_col = max(0, w_out - 1)

    # 获取卷积核
    kernel = get_kernel(kernel_name, custom_kernel)

    # 初始化/重置特征图（图像、核或尺寸变化时）
    prev_img = st.session_state.get("conv_current_image_source", None)
    prev_kernel = st.session_state.get("conv_current_kernel_name", None)
    image_key = f"{img_source}|gauss:{enable_gaussian}"
    if ("conv_feature_map" not in st.session_state
            or st.session_state.conv_feature_map.shape != (h_out, w_out)
            or prev_img != image_key
            or prev_kernel != kernel_name):
        _reset_conv_state(h_out, w_out)
        # 更换卷积核/图像时，扫描位置回到第一行第一列
        st.session_state.conv_row = 0
        st.session_state.conv_col = 0
        st.session_state.scan_position = (0, 0)
        st.session_state.conv_current_image_source = image_key
        st.session_state.conv_current_kernel_name = kernel_name

    # 初始化 slider 状态（必须在 slider 渲染前完成所有修改）
    if "conv_row" not in st.session_state:
        st.session_state.conv_row = 0
    if "conv_col" not in st.session_state:
        st.session_state.conv_col = 0
    # 自动扫描结束后的定位（从临时 key 写入，在 slider 创建前完成）
    if "conv_scan_final_row" in st.session_state:
        st.session_state.conv_row = st.session_state.conv_scan_final_row
        st.session_state.conv_col = st.session_state.conv_scan_final_col
        del st.session_state.conv_scan_final_row
        del st.session_state.conv_scan_final_col
    # 图像变小后越界修正
    if st.session_state.conv_row > max_row:
        st.session_state.conv_row = max_row
    if st.session_state.conv_col > max_col:
        st.session_state.conv_col = max_col

    # 卷积扫描控制（与第一行组合成紧凑两行布局）
    row, col = _render_conv_controls(max_row, max_col, h_out, w_out, button_container=ctrl_col3)

    # 计算当前位置卷积（若未计算）
    if not st.session_state.conv_computed[row, col]:
        val = compute_conv_value(img_arr, kernel_name, kernel, row, col)
        st.session_state.conv_feature_map[row, col] = val
        st.session_state.conv_computed[row, col] = True

    patch, current_val = _render_conv_visualization(
        img_display, img_arr, kernel, kernel_name, row, col, st.session_state.conv_feature_map
    )

    # 数值计算显示
    if kernel_name == _COMBINED_KERNEL_NAME:
        gx, gy, _ = compute_combined_edge(img_arr, row, col)
        formula = format_combined_formula(gx, gy, current_val)
    else:
        formula = format_conv_formula(patch, kernel, current_val)
    st.code(formula, language="text")
    _render_conv_scan_logic(h_out, w_out, img_arr, kernel_name, kernel)

    with st.expander("查看完整边缘提取效果", expanded=True):
        full_map = compute_full_feature_map(img_arr, kernel_name, kernel)
        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            st.markdown("**原图（彩色）**")
            st.image(img_display, use_container_width=False)
        with preview_col2:
            st.markdown("**全图边缘特征图**")
            st.image(Image.fromarray(full_map, mode="L"), use_container_width=False)
        st.caption("建议优先使用“边缘检测（Laplacian 8邻域）”或“组合边缘检测（Sobel X+Y）”提取头像完整轮廓。")

    # ---- 卷积原理讲解 ----
    st.markdown(
        "**卷积核（Convolution Kernel）** 是一个小尺寸的权重窗口（常见 3×3），它在输入图像上逐行滑动，"
        "每一步将窗口内的像素值与对应权重做点积求和，生成输出特征图上的一个像素。"
        "同一套参数扫遍全图，称为**权值共享**，极大减少了参数量。"
    )
    st.markdown(
        "- **边缘检测核**（如 Sobel）对灰度突变敏感，能提取轮廓；"
        "**模糊核**取局部均值，平滑噪声；**锐化核**增强中心像素差异，让边缘更清晰。\n"
        "- 五栏流程依次展示：原图高亮感受野 → 感受野热图 → 核权重矩阵 → 逐元素乘积与求和 → 累积特征图，"
        "直观体现了「点积 = 对应元素相乘后求和」与「逐步填充」的全过程。\n"
        "- 点击「→ 单步」可逐格推进，点击「▶ 自动扫描」可连续遍历全图。"
    )

    st.divider()

    # ==================== 模块 B：池化压缩演示 ====================
    st.subheader("核心机制：池化层信息压缩")

    # 生成固定 8×8 输入特征图（若未初始化）
    if "pool_input_map" not in st.session_state:
        rng = np.random.RandomState(42)
        base = np.outer(np.sin(np.linspace(0, 3 * np.pi, 8)), np.cos(np.linspace(0, 2 * np.pi, 8)))
        base = (base - base.min()) / (base.max() - base.min()) * 200 + 20
        st.session_state.pool_input_map = base.astype(np.float32)

    pool_input = st.session_state.pool_input_map
    p_h, p_w = pool_input.shape
    p_out_h, p_out_w = p_h // 2, p_w // 2

    if "pool_type" not in st.session_state:
        st.session_state.pool_type = "最大池化"
    if "pool_window_row" not in st.session_state:
        st.session_state.pool_window_row = 0
    if "pool_window_col" not in st.session_state:
        st.session_state.pool_window_col = 0
    if "is_pool_scanning" not in st.session_state:
        st.session_state.is_pool_scanning = False
    if "pool_scan_position" not in st.session_state:
        st.session_state.pool_scan_position = (0, 0)

    pool_ctrl1, pool_ctrl2, pool_ctrl3 = st.columns([1, 1, 2])
    with pool_ctrl1:
        pool_type = st.radio("池化类型", ["最大池化", "平均池化"], key="pool_type")

    max_pr = max(0, p_out_h - 1)
    max_pc = max(0, p_out_w - 1)
    if st.session_state.pool_window_row > max_pr:
        st.session_state.pool_window_row = max_pr
    if st.session_state.pool_window_col > max_pc:
        st.session_state.pool_window_col = max_pc

    # 重置池化状态（尺寸或池化类型变化时）
    prev_pool_type = st.session_state.get("pool_current_type", None)
    if ("pool_feature_map" not in st.session_state
            or st.session_state.pool_feature_map.shape != (p_out_h, p_out_w)
            or prev_pool_type != pool_type):
        _reset_pool_state(p_out_h, p_out_w)
        st.session_state.pool_current_type = pool_type

    with pool_ctrl3:
        pbtn1, pbtn2, _ = st.columns([1, 1, 2])
        with pbtn1:
            if st.button("▶ 播放", key="btn_pool_scan"):
                st.session_state.is_pool_scanning = True
                st.session_state.pool_scan_position = (0, 0)
                st.session_state.pool_window_row = 0
                st.session_state.pool_window_col = 0
                st.rerun()
        with pbtn2:
            if st.button("⏹ 停止", key="btn_pool_stop"):
                st.session_state.is_pool_scanning = False
                st.rerun()

    # 池化扫描：在 slider 前同步位置
    if st.session_state.is_pool_scanning:
        psr, psc = st.session_state.pool_scan_position
        st.session_state.pool_window_row = min(psr, max_pr)
        st.session_state.pool_window_col = min(psc, max_pc)

    with pool_ctrl2:
        pwr = st.slider("窗口行", 0, max_pr, key="pool_window_row")
        pwc = st.slider("窗口列", 0, max_pc, key="pool_window_col")

    # 计算当前池化结果（若未计算）
    if not st.session_state.pool_computed[pwr, pwc]:
        p_patch = pool_input[pwr * 2:pwr * 2 + 2, pwc * 2:pwc * 2 + 2]
        p_val = float(p_patch.max()) if pool_type == "最大池化" else float(p_patch.mean())
        st.session_state.pool_feature_map[pwr, pwc] = p_val
        st.session_state.pool_computed[pwr, pwc] = True

    pool_viz_col1, pool_viz_col2 = st.columns(2)
    with pool_viz_col1:
        st.markdown("**输入特征图**")
        pool_demo_img = draw_pooling_demo(pool_input, pool_type, pwr, pwc)
        st.image(pool_demo_img, use_container_width=False)
        st.caption(f"输入尺寸：{p_h} × {p_w}")

    with pool_viz_col2:
        st.markdown("**输出特征图（累积）**")
        pool_fm_img = draw_feature_map(st.session_state.pool_feature_map, pwr, pwc)
        pool_fm_display = _upscale(pool_fm_img, max(1, 300 // max(pool_fm_img.width, 1)))
        st.image(pool_fm_display, use_container_width=False)
        st.caption(f"输出尺寸：{p_out_h} × {p_out_w}（压缩为 1/4）")

    # 池化自动播放：固定较慢间隔，每次 rerun 前进一步（当前格已在上方按滑块位置计算）
    if st.session_state.is_pool_scanning:
        psr, psc = st.session_state.pool_scan_position
        st.caption(f"🔄 池化遍历中：窗口 ({psr + 1}/{p_out_h}, {psc + 1}/{p_out_w})")
        time.sleep(_POOL_SCAN_INTERVAL_SEC)

        npc = psc + 1
        npr = psr
        if npc >= p_out_w:
            npc = 0
            npr = psr + 1
        if npr >= p_out_h:
            st.session_state.is_pool_scanning = False
            st.session_state.pool_scan_position = (0, 0)
        else:
            st.session_state.pool_scan_position = (npr, npc)

        if st.session_state.is_pool_scanning:
            st.rerun()

    # ---- 池化原理讲解 ----
    st.markdown(
        "**池化层（Pooling Layer）** 的核心任务是**降维**与**信息压缩**："
        "将相邻的 2×2 像素合并为 1 个像素，特征图尺寸减半，计算量降至 1/4。"
    )
    st.markdown(
        "- **最大池化（Max Pooling）**：保留窗口内的最强响应（最显著特征），丢弃冗余细节，"
        "使网络对目标的微小位移具有**平移不变性**。左栏中非最大值的像素被压暗，突出「赢家通吃」的筛选逻辑。\n"
        "- **平均池化（Average Pooling）**：取窗口均值，平滑保留整体能量分布，更关注背景与纹理的宏观特征。\n"
        "- 虚线箭头示意压缩映射关系：左侧 2×2 区域的信息被「浓缩」为右侧单个像素，"
        "多层堆叠后，空间维度逐级下降，语义层级逐级上升。"
    )

    st.divider()

    # ==================== 模块 C：层次叠加链条 ====================
    st.subheader("核心机制：从边缘到抽象的层次叠加")
    chain_img = draw_layer_chain()
    st.image(chain_img, use_container_width=True)
    st.caption("每一层卷积提取更复杂的特征，池化逐步降低空间维度，参数共享保证高效。")

    with st.expander("📖 静态层次链图 · 读图详解（点击展开）", expanded=False):
        st.markdown(
            """
**这张图在表示什么**  
从左到右是一条典型的浅层 CNN **前向数据流**：原图 → 第一次卷积 → 第一次池化 → 第二次卷积 → 第二次池化。灰色箭头表示「上一层的输出作为下一层的输入」，强调层次是**顺序叠加**的一条链。

**图中的视觉元素**

- **彩色方块**：嵌在每个阶段栏位中的小方块，其**边长与该阶段特征图的空间边长成比例**——格子数多（分辨率高）时方块更大，经过池化后边长减半、方块明显变小，便于一眼看出「越往右，平面上的空间维越被压缩」。  
- **配色**：输入偏蓝，卷积偏绿，池化偏黄，区分 **Input / Conv / Pool** 以及二者**交替堆叠**的节奏。  
- **文字标签**：两行分别为阶段名与 **H×W**，对应下方示意链条里使用的空间尺寸。

**各阶段尺寸在教学设定下的含义**（与常见设定对齐：3×3 有效卷积、步幅 1；2×2 池化、步幅 2）

| 阶段 | 典型含义 | 尺寸在说明什么 |
|:-----|:---------|:---------------|
| Input 64×64 | 输入特征图 | 原始空间分辨率 |
| Conv1 62×62 | 第一卷积输出 | 无 padding 时 3×3 卷积每边少 2 个像素（64−2=62） |
| Pool1 31×31 | 第一池化输出 | 特征图每边约减半（62÷2=31） |
| Conv2 29×29 | 第二卷积输出 | 同上，31−2=29 |
| Pool2 14×14 | 第二池化输出 | 再次减半，得到更小的空间格子（约 29÷2→14） |

**说明**：真实网络可有不同 padding、stride、通道数，具体尺寸不必与此链完全一致；此处用一条**可手算核对**的尺寸链，将前面「卷积滑动 / 池化窗口」的微观操作与「整网空间如何逐级变小」的宏观图景对齐。
            """
        )

    render_parameter_explanation(
        [
            "当前卷积核为“{0}”：建议观察它对水平、垂直或局部强度变化的响应。".format(kernel_name),
            "高斯预处理={0}：开启时通常会先平滑局部噪声，建议与关闭状态做一次对照。".format(
                "开启" if enable_gaussian else "关闭"
            ),
            "池化方式为“{0}”：最大池化偏向保留强响应，平均池化更关注窗口的整体能量。".format(pool_type),
        ]
    )
    render_observation(
        [
            "当前卷积输出尺寸为 {0}×{1}，池化输出尺寸为 {2}×{3}。".format(h_out, w_out, p_out_h, p_out_w),
            "建议先单步推进几个位置，再播放完整扫描，比较局部计算与累积特征图。",
        ]
    )


if __name__ == "__main__":
    nv_render_cnn_viz()
