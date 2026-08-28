"""RNN 教学可视化模块。

包含 RNN 时间展开示意、隐藏状态数值演算，以及与 CNN 的差异对比。
"""

import base64
import os
from io import BytesIO
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from components.experiment_panel import (
    render_experiment_guide,
    render_learning_goals,
    render_observation,
    render_parameter_explanation,
)
from components.page_header import render_page_header


_BG = (15, 15, 15)
_EDGE = (85, 85, 85)
_TEXT = (230, 230, 230)
_FUTURE = (70, 70, 70)


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
    # 最后的回退：尝试裸文件名
    for name in ["assets/LXGWWenKai-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _load_font_bold(size: int):
    """标题加粗；优先 Linux Noto Bold / Windows msyhbd，回退普通字体。"""
    bold_paths = [
        "assets/LXGWWenKai-Regular.ttf"
    ]
    for path in bold_paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return _load_font(size)


def _draw_arrow(draw: ImageDraw.ImageDraw, p0: tuple, p1: tuple, color=(150, 150, 150), width=2, head_len: Optional[int] = None):
    hl = head_len if head_len is not None else max(6, width * 4)
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / norm, dy / norm
    # 线段画到箭头三角之前，避免与三角形重叠加厚
    line_end_x = float(p1[0]) - ux * hl * 1.05
    line_end_y = float(p1[1]) - uy * hl * 1.05
    draw.line([p0, (line_end_x, line_end_y)], fill=color, width=width)
    px, py = -uy, ux
    head_w = max(3, hl * 0.55)
    tip = (int(round(p1[0])), int(round(p1[1])))
    left = (int(tip[0] - ux * hl + px * head_w), int(tip[1] - uy * hl + py * head_w))
    right = (int(tip[0] - ux * hl - px * head_w), int(tip[1] - uy * hl - py * head_w))
    draw.polygon([tip, left, right], fill=color)


def _draw_arrow_dashed(
    draw: ImageDraw.ImageDraw,
    p0: tuple,
    p1: tuple,
    color: tuple,
    width: int = 2,
    dash_len: int = 8,
    gap_len: int = 6,
    head_len: int = 4,
):
    """虚线箭头（虚线段画到箭头之前，末尾接三角箭头）"""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dist = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / dist, dy / dist
    t = 0.0
    step = dash_len + gap_len
    usable = max(0.0, dist - head_len * 1.15)
    while t + dash_len <= usable:
        xa = p0[0] + ux * t
        ya = p0[1] + uy * t
        seg_end = min(t + dash_len, usable)
        xb = p0[0] + ux * seg_end
        yb = p0[1] + uy * seg_end
        draw.line([(xa, ya), (xb, yb)], fill=color, width=width)
        t += step
    px, py = -uy, ux
    head_w = max(3, head_len * 0.55)
    tip = (int(p1[0]), int(p1[1]))
    left = (int(p1[0] - ux * head_len + px * head_w), int(p1[1] - uy * head_len + py * head_w))
    right = (int(p1[0] - ux * head_len - px * head_w), int(p1[1] - uy * head_len - py * head_w))
    draw.polygon([tip, left, right], fill=color)


def _blend_rgb(base: tuple, overlay: tuple, alpha: float) -> tuple:
    return tuple(int(base[i] * (1 - alpha) + overlay[i] * alpha) for i in range(3))


def _draw_round_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple,
    w: int,
    h: int,
    text: str,
    fill: tuple,
    font,
    radius: int = 8,
    text_color: Optional[tuple] = None,
):
    """圆角矩形 + 居中文字"""
    x, y = xy
    tc = text_color if text_color is not None else (248, 250, 252)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill, outline=_blend_rgb(fill, (148, 163, 184), 0.35), width=1)
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text((x + (w - tw) // 2, y + (h - th) // 2), text, fill=tc, font=font)


def _draw_node(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, fill: tuple, outline: tuple, label: str, font):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=2)
    tb = draw.textbbox((0, 0), label, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text((cx - tw // 2, cy - th // 2), label, fill=_TEXT, font=font)


def draw_rnn_unrolled(steps: int = 4, active_step: int = 1) -> Image.Image:
    """绘制可高亮时间步的 RNN 展开图。"""
    steps = max(3, min(5, steps))
    active_step = max(1, min(steps, active_step))

    col_gap = 200
    margin = 50
    w = margin * 2 + (steps - 1) * col_gap + 120
    h = 420
    img = Image.new("RGB", (w, h), _BG)
    draw = ImageDraw.Draw(img)
    font = _load_font(18)
    small = _load_font(15)

    y_x, y_h, y_y = 85, 210, 335
    r = 28
    x_base = margin + 30

    for t in range(1, steps + 1):
        cx = x_base + (t - 1) * col_gap
        is_past_or_now = t <= active_step
        is_active = t == active_step

        c_x = (88, 148, 255) if is_past_or_now else _FUTURE
        c_h = (52, 199, 89) if is_past_or_now else _FUTURE
        c_y = (167, 139, 250) if is_past_or_now else _FUTURE
        outline = (255, 179, 64) if is_active else _EDGE

        _draw_node(draw, cx, y_x, r, c_x, outline, f"x{t}", font)
        _draw_node(draw, cx, y_h, r, c_h, outline, f"h{t}", font)
        _draw_node(draw, cx, y_y, r, c_y, outline, f"y{t}", font)

        vertical_color = (165, 165, 165) if is_past_or_now else (90, 90, 90)
        _draw_arrow(draw, (cx, y_x + r), (cx, y_h - r), color=vertical_color, width=3)
        _draw_arrow(draw, (cx, y_h + r), (cx, y_y - r), color=vertical_color, width=3)

        if t < steps:
            nx = x_base + t * col_gap
            mem_color = (245, 158, 11) if t < active_step else (100, 100, 100)
            _draw_arrow(draw, (cx + r, y_h), (nx - r, y_h), color=mem_color, width=6 if t < active_step else 3)

        draw.text((cx - 24, y_h - 70), f"t={t}", fill=(200, 200, 200), font=small)

    draw.text((margin, 12), "RNN 时间展开：当前步高亮，未来步灰显", fill=_TEXT, font=font)
    draw.text((margin, h - 36), "橙色横向箭头表示隐藏状态记忆传递：h(t-1) -> h(t)", fill=(245, 158, 11), font=small)
    return img


def _draw_vector_bar(values: np.ndarray, base_color: tuple, title: str) -> Image.Image:
    """绘制一行向量热图小卡片。"""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    n = len(values)
    cell = 72
    margin = 18
    title_h = 30
    w = margin * 2 + n * cell
    h = margin * 2 + cell + title_h
    img = Image.new("RGB", (w, h), _BG)
    draw = ImageDraw.Draw(img)
    font = _load_font(14)
    num_font = _load_font(12)

    vmax = max(float(np.max(np.abs(values))), 1e-6)
    for i, v in enumerate(values):
        ratio = 0.5 + 0.5 * float(v) / vmax
        ratio = max(0.0, min(1.0, ratio))
        color = (
            int(20 + (base_color[0] - 20) * ratio),
            int(20 + (base_color[1] - 20) * ratio),
            int(20 + (base_color[2] - 20) * ratio),
        )
        x0 = margin + i * cell
        y0 = margin + title_h
        draw.rectangle([x0, y0, x0 + cell - 2, y0 + cell - 2], fill=color, outline=(100, 100, 100), width=2)
        text = f"{v:+.2f}"
        tb = draw.textbbox((0, 0), text, font=num_font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.text((x0 + (cell - tw) // 2, y0 + (cell - th) // 2), text, fill=_TEXT, font=num_font)

    draw.text((margin, margin), title, fill=_TEXT, font=font)
    return img


def _run_toy_rnn_sequence():
    """构造固定小例子，返回 x 序列与 h 序列。"""
    x_seq = np.array(
        [
            [0.8, -0.2],
            [0.1, 0.7],
            [-0.6, 0.4],
            [0.9, 0.3],
        ],
        dtype=np.float32,
    )
    w_xh = np.array([[0.9, -0.4], [0.2, 0.8], [-0.6, 0.5]], dtype=np.float32)
    w_hh = np.array([[0.5, 0.1, -0.3], [0.2, 0.6, 0.1], [-0.4, 0.3, 0.7]], dtype=np.float32)

    h_states = [np.zeros(3, dtype=np.float32)]
    z_states = []
    for x_t in x_seq:
        h_prev = h_states[-1]
        z_t = w_xh @ x_t + w_hh @ h_prev
        h_t = np.tanh(z_t)
        z_states.append(z_t)
        h_states.append(h_t)
    return x_seq, np.array(h_states, dtype=np.float32), np.array(z_states, dtype=np.float32), w_xh, w_hh


def _format_toy_equation(x_t: np.ndarray, h_prev: np.ndarray, z_t: np.ndarray, h_t: np.ndarray, w_xh: np.ndarray, w_hh: np.ndarray) -> str:
    """生成逐元素运算解释文本。"""
    lines = []
    for i in range(3):
        terms_x = f"{w_xh[i,0]:+.2f}*{x_t[0]:+.2f} {w_xh[i,1]:+.2f}*{x_t[1]:+.2f}"
        terms_h = (
            f"{w_hh[i,0]:+.2f}*{h_prev[0]:+.2f} "
            f"{w_hh[i,1]:+.2f}*{h_prev[1]:+.2f} "
            f"{w_hh[i,2]:+.2f}*{h_prev[2]:+.2f}"
        )
        lines.append(
            f"z[{i + 1}] = ({terms_x}) + ({terms_h}) = {z_t[i]:+.3f}   ->   h[{i + 1}] = tanh({z_t[i]:+.3f}) = {h_t[i]:+.3f}"
        )
    return "\n".join(lines)


def _render_step_with_nav(label: str, min_step: int, max_step: int, key: str, default_step: int) -> int:
    """渲染带前进/后退按钮的时间步控件。"""
    state_key = f"{key}_state"
    slider_key = f"{key}_slider"

    if state_key not in st.session_state:
        st.session_state[state_key] = default_step
    st.session_state[state_key] = int(max(min_step, min(max_step, st.session_state[state_key])))
    if slider_key not in st.session_state:
        st.session_state[slider_key] = st.session_state[state_key]

    # 在渲染 slider 前将其与逻辑状态对齐，避免按钮与图示不同步
    st.session_state[slider_key] = st.session_state[state_key]
    cur = int(st.session_state[state_key])
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2.2, 1], gap="small")
    with nav_col1:
        if st.button("◀ 上一步", key=f"{key}_prev_btn", use_container_width=True, disabled=cur <= min_step):
            st.session_state[state_key] = max(min_step, cur - 1)
            st.session_state[slider_key] = st.session_state[state_key]
            st.rerun()
    with nav_col3:
        if st.button("下一步 ▶", key=f"{key}_next_btn", use_container_width=True, disabled=cur >= max_step):
            st.session_state[state_key] = min(max_step, cur + 1)
            st.session_state[slider_key] = st.session_state[state_key]
            st.rerun()
    with nav_col2:
        st.markdown(
            f"<div style='text-align:center;padding-top:6px;'>当前时间步：<b>{st.session_state[state_key]}</b> / {max_step}</div>",
            unsafe_allow_html=True,
        )

    slider_val = st.slider(label, min_step, max_step, step=1, key=slider_key)
    st.session_state[state_key] = int(slider_val)
    return st.session_state[state_key]


def _reset_rnn_teaching_steps():
    """Restore only the two existing RNN navigation controls."""
    for key in ("rnn_step_state", "rnn_step_slider", "rnn_calc_step_state", "rnn_calc_step_slider"):
        st.session_state[key] = 1


def _draw_matrix_heatmap(mat: np.ndarray, title: str, row_labels: list, col_labels: list) -> Image.Image:
    """绘制权重矩阵热图（行=输出维度，列=输入维度）。"""
    rows, cols = mat.shape
    cell = 64
    label_w, label_h = 48, 34
    margin = 14
    title_h = 30
    w = margin * 2 + label_w + cols * cell
    h = margin * 2 + title_h + label_h + rows * cell
    img = Image.new("RGB", (w, h), _BG)
    draw = ImageDraw.Draw(img)
    font = _load_font(13)
    small = _load_font(12)

    draw.text((margin, margin), title, fill=_TEXT, font=font)
    vmax = max(float(np.max(np.abs(mat))), 1e-6)

    # column labels
    for j, lbl in enumerate(col_labels):
        cx = margin + label_w + j * cell + cell // 2
        tb = draw.textbbox((0, 0), lbl, font=small)
        draw.text((cx - (tb[2] - tb[0]) // 2, margin + title_h), lbl, fill=(200, 200, 200), font=small)

    for i in range(rows):
        # row label
        ry = margin + title_h + label_h + i * cell + cell // 2
        tb = draw.textbbox((0, 0), row_labels[i], font=small)
        draw.text((margin + label_w - (tb[2] - tb[0]) - 4, ry - (tb[3] - tb[1]) // 2), row_labels[i], fill=(200, 200, 200), font=small)

        for j in range(cols):
            v = float(mat[i, j])
            ratio = 0.5 + 0.5 * v / vmax
            ratio = max(0.0, min(1.0, ratio))
            r_base = (67, 131, 222) if v >= 0 else (220, 80, 80)
            color = (int(20 + (r_base[0] - 20) * ratio),
                     int(20 + (r_base[1] - 20) * ratio),
                     int(20 + (r_base[2] - 20) * ratio))
            x0 = margin + label_w + j * cell
            y0 = margin + title_h + label_h + i * cell
            draw.rectangle([x0, y0, x0 + cell - 2, y0 + cell - 2], fill=color, outline=(80, 80, 80), width=1)
            text = f"{v:+.2f}"
            tb = draw.textbbox((0, 0), text, font=small)
            draw.text((x0 + (cell - (tb[2] - tb[0])) // 2, y0 + (cell - (tb[3] - tb[1])) // 2),
                      text, fill=_TEXT, font=small)
    return img


def _draw_calc_flow(x_t, h_prev, partial_x, partial_h, z_t, h_t) -> Image.Image:
    """绘制单时间步计算流程图：x_t + h_prev → 两路线性映射 → 相加得 z_t → tanh → h_t。"""
    w, h = 980, 320
    img = Image.new("RGB", (w, h), _BG)
    draw = ImageDraw.Draw(img)
    font = _load_font(15)
    small = _load_font(13)

    stages = [
        ("x_t\n输入", (78, 146, 255), 80),
        ("Wxh*x_t", (245, 200, 60), 250),
        ("z_t=sum", (200, 120, 255), 470),
        ("tanh(z_t)", (255, 120, 80), 690),
        ("h_t\n新记忆", (52, 199, 89), 900),
    ]
    cy = h // 2

    node_r = 38

    # 上方主链：x_t -> Wxh*x_t -> sum
    for i, (label, color, cx) in enumerate(stages):
        draw.ellipse([cx - node_r, cy - node_r, cx + node_r, cy + node_r],
                     fill=color, outline=(200, 200, 200), width=2)
        lines = label.split("\n")
        line_h = 18
        total = len(lines) * line_h
        for k, line in enumerate(lines):
            tb = draw.textbbox((0, 0), line, font=small)
            tw = tb[2] - tb[0]
            draw.text((cx - tw // 2, cy - total // 2 + k * line_h), line, fill=(20, 20, 20), font=small)
        if i < len(stages) - 1:
            next_cx = stages[i + 1][2]
            _draw_arrow(draw, (cx + node_r, cy), (next_cx - node_r, cy), color=(160, 160, 160), width=3)

    # 下方 h_prev 通道
    h_prev_cy = cy + 115
    h_node_x = 80
    w_hh_x = 250
    draw.ellipse([h_node_x - node_r, h_prev_cy - node_r, h_node_x + node_r, h_prev_cy + node_r],
                 fill=(52, 199, 89), outline=(200, 200, 200), width=2)
    lb = "h(t-1)\n上一步"
    for k, line in enumerate(lb.split("\n")):
        tb = draw.textbbox((0, 0), line, font=small)
        draw.text((h_node_x - (tb[2] - tb[0]) // 2, h_prev_cy - 15 + k * 18), line, fill=(20, 20, 20), font=small)

    draw.ellipse([w_hh_x - node_r, h_prev_cy - node_r, w_hh_x + node_r, h_prev_cy + node_r],
                 fill=(245, 200, 60), outline=(200, 200, 200), width=2)
    wb = "Whh*h(t-1)"
    tb = draw.textbbox((0, 0), wb, font=small)
    draw.text((w_hh_x - (tb[2] - tb[0]) // 2, h_prev_cy - (tb[3] - tb[1]) // 2), wb, fill=(20, 20, 20), font=small)

    _draw_arrow(draw, (h_node_x + node_r, h_prev_cy), (w_hh_x - node_r, h_prev_cy), color=(160, 160, 160), width=3)
    # merge: Whh*h(t-1) 汇入 z_t
    sum_cx = stages[2][2]
    _draw_arrow(draw, (w_hh_x + node_r, h_prev_cy), (sum_cx, cy + node_r), color=(245, 158, 11), width=3)

    # value annotations
    def fmt_vec(v):
        return "  ".join(f"{x:+.2f}" for x in v)

    draw.text((12, 12), f"x_t   : [{fmt_vec(x_t)}]", fill=(160, 200, 255), font=small)
    draw.text((12, 30), f"h_prev: [{fmt_vec(h_prev)}]", fill=(100, 220, 140), font=small)
    draw.text((12, 48), f"Wxh*x_t: [{fmt_vec(partial_x)}]   Whh*h_prev: [{fmt_vec(partial_h)}]", fill=(230, 210, 100), font=small)
    draw.text((12, 66), f"z_t   : [{fmt_vec(z_t)}]   h_t: [{fmt_vec(h_t)}]", fill=(200, 160, 255), font=small)
    return img


def _resize_keep_aspect(img: Image.Image, target_w: int) -> Image.Image:
    """按目标宽度等比缩放。"""
    if img.width <= 0 or img.width == target_w:
        return img
    ratio = target_w / float(img.width)
    target_h = max(1, int(img.height * ratio))
    return img.resize((target_w, target_h), Image.Resampling.LANCZOS)


def _show_centered_golden_image(img: Image.Image, target_w: int):
    """用黄金比例中栏居中显示图片。"""
    left, middle, right = st.columns([0.5, 2.3, 0.5], gap="small")
    with middle:
        _show_centered_inline_image(img, target_w)


def _show_centered_inline_image(img: Image.Image, target_w: int):
    """在当前容器内以 HTML 方式水平居中显示图片。"""
    resized = _resize_keep_aspect(img, target_w)
    buf = BytesIO()
    resized.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    st.markdown(
        (
            '<div style="text-align:center;">'
            f'<img src="data:image/png;base64,{b64}" width="{resized.width}" />'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def draw_cnn_vs_rnn() -> Image.Image:
    """CNN vs RNN 架构对比图（16:9，左右对称，教学演示级排版）。

    关键可调参数（改变节点数量时在此修改）：
    - CANVAS_W / CANVAS_H：画布（保持 16:9）
    - MARGIN_H / MARGIN_V：左右 / 上下外边距
    - CNN_NODE_W / CNN_NODE_H / CNN_GAP：左侧链路的节点尺寸与间距
    - RNN_NODE_W / RNN_NODE_H / RNN_V_GAP：右侧每步堆叠与竖向间距
    - RNN_COL_GAP：右侧列间距（与 CNN 总宽度对齐后自动计算或固定）
    """
    # --- 画布与边距 ---
    CANVAS_W, CANVAS_H = 1280, 720  # 16:9
    MARGIN_H, MARGIN_V = 80, 60
    BG = (15, 23, 42)  # #0f172a
    GRID = (51, 65, 85)  # #334155
    DIVIDER = GRID
    ARROW_CNN = (71, 85, 105)  # #475569
    ARROW_GRAY = (148, 163, 184)  # 竖向细箭头（略亮于纯灰，保证在深色底可见）
    ARROW_TIME = (217, 119, 6)  # 琥珀色 #d97706，时间传递

    # --- CNN 左侧：4 节点，冷色渐变 ---
    CNN_LABELS = ("Input", "Conv", "Pool", "FC")
    CNN_NODE_W, CNN_NODE_H = 80, 60
    CNN_GAP = 40  # 相邻层水平间距（边到边）
    CNN_COLORS = (
        (30, 64, 125),  # 深蓝 Input
        (37, 99, 235),  # 中蓝 Conv
        (147, 197, 253),  # 浅蓝 Pool
        (34, 211, 238),  # 青蓝 FC
    )
    CNN_CHAIN_W = len(CNN_LABELS) * CNN_NODE_W + (len(CNN_LABELS) - 1) * CNN_GAP

    # --- RNN 右侧：4 时间步，与左侧总宽度对齐 ---
    RNN_STEPS = 4
    RNN_NODE_W, RNN_NODE_H = 50, 35
    RNN_V_GAP = 30  # x→h、h→y 竖向间距
    RNN_COL_GAP = (CNN_CHAIN_W - RNN_STEPS * RNN_NODE_W) / max(1, (RNN_STEPS - 1))

    # --- 字体：与其他 PIL 配图一致走 `_load_font_*`，避免部分字形缺失 ---
    font_title = _load_font_bold(18)
    font_title_alt = _load_font(18)
    font_node = _load_font(12)
    font_caption = _load_font(14)

    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(img)

    inner_left = MARGIN_H
    inner_right = CANVAS_W - MARGIN_H
    inner_top = MARGIN_V
    inner_bottom = CANVAS_H - MARGIN_V

    half_w = (inner_right - inner_left) / 2
    mid_x = inner_left + half_w
    left_cx = inner_left + half_w / 2
    right_cx = mid_x + half_w / 2

    # --- 参考网格（低透明度纵向线，便于对齐）---
    gx = inner_left
    while gx <= inner_right:
        gcol = _blend_rgb(BG, GRID, 0.12)
        draw.line([(gx, inner_top), (gx, inner_bottom)], fill=gcol, width=1)
        gx += 40

    # --- 标题：同高、居中于各自半区 ---
    title_left = "CNN：空间层次堆叠"
    title_right = "RNN：时间步状态传递"
    title_y = inner_top
    ft_left, ft_right = font_title, font_title
    tb = draw.textbbox((0, 0), title_left, font=ft_left)
    if tb[2] - tb[0] < 8:
        ft_left = font_title_alt
        tb = draw.textbbox((0, 0), title_left, font=ft_left)
    tw = tb[2] - tb[0]
    draw.text((int(left_cx - tw / 2), title_y), title_left, fill=(226, 232, 240), font=ft_left)
    tb2 = draw.textbbox((0, 0), title_right, font=ft_right)
    if tb2[2] - tb2[0] < 8:
        ft_right = font_title_alt
        tb2 = draw.textbbox((0, 0), title_right, font=ft_right)
    tw2 = tb2[2] - tb2[0]
    draw.text((int(right_cx - tw2 / 2), title_y), title_right, fill=(226, 232, 240), font=ft_right)

    title_bottom = title_y + max(tb[3] - tb[1], tb2[3] - tb2[1], 22) + 8

    # --- 主图区垂直居中：标题下缘到说明文字上缘之间 ---
    cap_margin = 18  # 主图与底部说明的间距
    cap_sample = "空间维逐层压缩"
    tb_cap = draw.textbbox((0, 0), cap_sample, font=font_caption)
    cap_h = tb_cap[3] - tb_cap[1]
    caption_baseline_y = CANVAS_H - MARGIN_V - cap_h + 2
    content_bottom = caption_baseline_y - cap_margin
    diagram_cy = (title_bottom + content_bottom) / 2

    # CNN 行：节点垂直居中对齐于 diagram_cy
    cnn_y0 = int(diagram_cy - CNN_NODE_H / 2)
    cnn_x0 = int(left_cx - CNN_CHAIN_W / 2)
    cnn_text_colors = (
        (248, 250, 252),  # 深色底用亮字
        (248, 250, 252),
        (15, 23, 42),  # 浅蓝底用深字
        (15, 23, 42),
    )
    for i, (lab, col) in enumerate(zip(CNN_LABELS, CNN_COLORS)):
        nx = cnn_x0 + i * (CNN_NODE_W + CNN_GAP)
        _draw_round_text(
            draw,
            (nx, cnn_y0),
            CNN_NODE_W,
            CNN_NODE_H,
            lab,
            col,
            font_node,
            radius=10,
            text_color=cnn_text_colors[i],
        )
    for i in range(len(CNN_LABELS) - 1):
        edge_r = cnn_x0 + i * (CNN_NODE_W + CNN_GAP) + CNN_NODE_W + 2
        edge_l_next = cnn_x0 + (i + 1) * (CNN_NODE_W + CNN_GAP) - 2
        ymid = int(diagram_cy)
        _draw_arrow(draw, (edge_r, ymid), (edge_l_next, ymid), color=ARROW_CNN, width=2, head_len=4)

    # RNN 列：总宽度与 CNN_CHAIN_W 一致
    rnn_stack_h = RNN_NODE_H * 3 + RNN_V_GAP * 2
    rnn_x0 = int(right_cx - CNN_CHAIN_W / 2)

    COLOR_X = (30, 58, 138)  # x_t 深蓝（与 Input 呼应）
    COLOR_H = (14, 116, 144)  # h_t 青绿（隐藏状态）
    COLOR_Y = (91, 89, 135)  # y_t 浅紫（压低饱和）

    y_xt0 = diagram_cy - rnn_stack_h / 2
    y_ht0 = y_xt0 + RNN_NODE_H + RNN_V_GAP
    ym_h = int(y_ht0 + RNN_NODE_H / 2)

    for t in range(RNN_STEPS):
        col_left = int(rnn_x0 + t * (RNN_NODE_W + RNN_COL_GAP))
        cx_mid = col_left + RNN_NODE_W / 2
        y_xt = y_xt0
        y_ht = y_ht0
        y_yt = y_ht + RNN_NODE_H + RNN_V_GAP
        tid = str(t + 1)
        _draw_round_text(
            draw,
            (col_left, int(y_xt)),
            RNN_NODE_W,
            RNN_NODE_H,
            f"x{tid}",
            COLOR_X,
            font_node,
            radius=6,
            text_color=(241, 245, 249),
        )
        _draw_round_text(
            draw,
            (col_left, int(y_ht)),
            RNN_NODE_W,
            RNN_NODE_H,
            f"h{tid}",
            COLOR_H,
            font_node,
            radius=6,
            text_color=(241, 245, 249),
        )
        _draw_round_text(
            draw,
            (col_left, int(y_yt)),
            RNN_NODE_W,
            RNN_NODE_H,
            f"y{tid}",
            COLOR_Y,
            font_node,
            radius=6,
            text_color=(241, 245, 249),
        )
        _draw_arrow(
            draw,
            (cx_mid, y_xt + RNN_NODE_H),
            (cx_mid, y_ht),
            color=ARROW_GRAY,
            width=2,
            head_len=4,
        )
        _draw_arrow(
            draw,
            (cx_mid, y_ht + RNN_NODE_H),
            (cx_mid, y_yt),
            color=ARROW_GRAY,
            width=2,
            head_len=4,
        )

    for t in range(RNN_STEPS - 1):
        col_left_a = int(rnn_x0 + t * (RNN_NODE_W + RNN_COL_GAP))
        col_left_b = int(rnn_x0 + (t + 1) * (RNN_NODE_W + RNN_COL_GAP))
        xa = col_left_a + RNN_NODE_W + 3
        xb = col_left_b - 3
        _draw_arrow_dashed(draw, (xa, ym_h), (xb, ym_h), ARROW_TIME, width=2, dash_len=8, gap_len=5, head_len=4)

    # --- 垂直分割线与主图区等高 ---
    draw.line([(int(mid_x), int(title_bottom)), (int(mid_x), int(content_bottom))], fill=DIVIDER, width=1)

    # --- 底部说明：各半区水平居中，同一下边距 ---
    cap_left = "空间维逐层压缩"
    cap_right = "同一组参数在时间轴上反复复用"
    tl = draw.textbbox((0, 0), cap_left, font=font_caption)
    tr = draw.textbbox((0, 0), cap_right, font=font_caption)
    draw.text((int(left_cx - (tl[2] - tl[0]) / 2), caption_baseline_y), cap_left, fill=(203, 213, 225), font=font_caption)
    draw.text((int(right_cx - (tr[2] - tr[0]) / 2), caption_baseline_y), cap_right, fill=(203, 213, 225), font=font_caption)

    return img


def nv_render_rnn_viz():
    """渲染 RNN 教学页面。"""
    render_page_header(
        title="循环神经网络 (RNN)",
        module="神经网络 / RNN",
        description="给神经网络装上短期记忆，让它能理解序列中的先后关系",
    )

    render_learning_goals(
        [
            "理解隐藏状态如何在相邻时间步之间传递。",
            "区分输入贡献、记忆贡献和 tanh 激活三个计算阶段。",
            "观察同一组参数在不同时间步上的复用方式。",
            "比较 RNN 与 CNN 在信息结构上的差异。",
        ]
    )
    render_experiment_guide(
        [
            "先拖动时间步滑块，观察当前节点与未来节点的状态。",
            "再选择一个时间步，按五个阶段阅读数值演算。",
            "对照 Wxh 与 Whh 热图，区分输入和历史记忆的贡献。",
            "最后联系 CNN 对比图，总结空间建模与时间建模的差异。",
        ]
    )
    st.button(
        "恢复默认参数",
        key="rnn_teaching_reset",
        on_click=_reset_rnn_teaching_steps,
        use_container_width=False,
    )

    st.subheader("为什么需要它？")
    st.info(
        "基础神经网络通常把样本视为独立输入，但文本、语音、时间序列的语义依赖前后文。"
        "RNN 通过跨时间步传递隐藏状态，让模型在当前时刻仍能携带过去的信息。"
    )

    st.subheader("核心机制：隐藏状态如何在时间中流动")
    step = _render_step_with_nav("序列时间步 t（1~4，可拖动）", 1, 4, "rnn_step", 1)
    unrolled_img = draw_rnn_unrolled(steps=4, active_step=step)
    _show_centered_golden_image(unrolled_img, target_w=int(unrolled_img.width * 0.9))

    st.latex(r"h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1})")
    st.markdown(
        "- 每个时间步都会读取当前输入 $x_t$，并结合上一时刻记忆 $h_{t-1}$ 更新为 $h_t$。\n"
        "- 同一组参数在所有时间步复用，因此模型能处理可变长度序列。\n"
        "- 隐藏状态既用于当前输出，也继续传到下一步，形成上下文记忆链。"
    )

    with st.expander("静态图读图详解（点击展开）", expanded=False):
        st.markdown(
            "蓝色节点表示输入 $x_t$，绿色节点表示隐藏状态 $h_t$，紫色节点表示输出 $y_t$。  \n"
            "灰色竖向箭头表示同一时间步内的信息计算，橙色横向箭头表示记忆从过去传向未来。  \n"
            "滑块移动时，当前时间步会被高亮，未来步则是灰显，以此来帮助理解 RNN 的逐步更新过程。"
        )

    st.divider()
    st.subheader("核心机制数值演算：看见记忆如何更新")
    st.markdown(
        "本示例的序列共有 **4 个时间步（t=1~4）**；你在下方选择一个时间步后，"
        "会看到该时间步内部的 **5 个计算阶段**，逐步展示数字如何流动。"
    )

    x_seq, h_states, z_states, w_xh, w_hh = _run_toy_rnn_sequence()
    calc_step = _render_step_with_nav("选择时间步 t（1~4）查看该步的 5 阶段演算", 1, 4, "rnn_calc_step", 1)

    x_t    = x_seq[calc_step - 1]
    h_prev = h_states[calc_step - 1]
    z_t    = z_states[calc_step - 1]
    h_t    = h_states[calc_step]
    partial_x = w_xh @ x_t
    partial_h = w_hh @ h_prev

    # ---- 总览流程图 ----
    _show_centered_golden_image(_draw_calc_flow(x_t, h_prev, partial_x, partial_h, z_t, h_t), target_w=1200)

    # ---- 五步详解 ----
    st.caption("说明：下面“第一步~第五步”是单个已选时间步 t 的内部计算流程，不是序列长度。")
    st.markdown("#### 第一步：读入本步输入 $x_t$ 与上一步隐藏状态 $h_{t-1}$")
    col_a, col_b = st.columns(2)
    with col_a:
        _show_centered_inline_image(_draw_vector_bar(x_t, (78, 146, 255), f"x_t  (t={calc_step}, 维度=2)"), 340)
        st.caption("当前时刻的输入向量，来自外部（如词向量、传感器读数）。")
    with col_b:
        _show_centered_inline_image(_draw_vector_bar(h_prev, (52, 199, 89), f"h_(t-1)  (t={calc_step - 1}, 维度=3)"), 340)
        st.caption("上一时刻保存下来的「记忆」，初始为全零；之后每步都被更新。")

    st.markdown("#### 第二步：两路线性映射——各自乘以权重矩阵")
    col_fx, col_fh = st.columns(2)
    with col_fx:
        st.latex(r"\text{贡献}_x = W_{xh} \cdot x_t,\;\; W_{xh} \in \mathbb{R}^{3\times 2}")
    with col_fh:
        st.latex(r"\text{贡献}_h = W_{hh} \cdot h_{t-1},\;\; W_{hh} \in \mathbb{R}^{3\times 3}")
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        _show_centered_inline_image(
            _draw_matrix_heatmap(
                w_xh,
                "Wxh（3×2）——输入权重",
                [f"h[{i+1}]" for i in range(3)],
                [f"x[{j+1}]" for j in range(2)],
            ),
            380,
        )
    with col_w2:
        _show_centered_inline_image(
            _draw_matrix_heatmap(
                w_hh,
                "Whh（3×3）——记忆权重",
                [f"h[{i+1}]" for i in range(3)],
                [f"hp[{j+1}]" for j in range(3)],
            ),
            380,
        )
    col_px, col_ph = st.columns(2)
    with col_px:
        _show_centered_inline_image(_draw_vector_bar(partial_x, (220, 170, 60), "Wxh*x_t（输入贡献）"), 380)
    with col_ph:
        _show_centered_inline_image(_draw_vector_bar(partial_h, (180, 100, 240), "Whh*h(t-1)（记忆贡献）"), 380)

    st.markdown("#### 第三步：两路贡献相加，得到线性预激活 $z_t$")
    st.latex(r"z_t = W_{xh} x_t + W_{hh} h_{t-1}")
    _show_centered_golden_image(_draw_vector_bar(z_t, (200, 120, 255), f"z_t（线性预激活，t={calc_step}）"), target_w=520)

    st.markdown("#### 第四步：tanh 激活——把 $z_t$ 压缩到 $(-1, 1)$")
    st.latex(r"h_t = \tanh(z_t)")
    st.markdown(
        "tanh 会把任意大的正/负数非线性地压缩到 $(-1,\\ 1)$ 区间内，"
        "防止状态值爆炸，同时引入非线性表达能力。"
    )
    _show_centered_golden_image(_draw_vector_bar(h_t, (40, 220, 120), f"h_t（新隐藏状态，t={calc_step}）"), target_w=520)

    st.markdown("#### 第五步：$h_t$ 同时用于两件事——输出预测 & 传给下一步")
    st.latex(r"y_t = W_{hy} h_t \quad (\text{输出预测})")
    st.latex(r"h_{t+1} = \tanh(W_{xh} x_{t+1} + W_{hh} h_t) \quad (\text{记忆继续传递})")
    st.markdown(
        "同一个 $h_t$ 既作为当前步的预测依据（送往输出层），"
        "又成为下一步的 $h_{t-1}$——这就是 RNN 记忆传递的关键。"
    )

    st.divider()
    with st.expander("逐维度展开式（点击查看完整数字）", expanded=False):
        st.code(_format_toy_equation(x_t, h_prev, z_t, h_t, w_xh, w_hh), language="text")
        st.caption(
            f"t={calc_step}：按隐藏层三维展开 Wxh*x_t + Whh*h(t-1)，再 tanh；可与热图对照。"
        )

    st.divider()
    st.subheader("与基础神经网络的差异")
    st.markdown(
        "**MLP 无状态**：每个样本前向传播一次即结束，几乎不保留跨样本上下文。  \n"
        "**RNN 有状态传递**：隐藏状态在时间步之间流动，能整合历史信息。"
    )

    st.subheader("与卷积神经网络 (CNN) 的差异")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            "**CNN（空间建模）**\n"
            "- 输入类型：图像等空间网格\n"
            "- 核心操作：卷积核在局部窗口滑动\n"
            "- 参数共享：同一卷积核在空间位置复用\n"
            "- 信息流：层间前馈，强调空间局部性\n"
            "- 典型任务：分类、检测、分割"
        )
    with right:
        st.markdown(
            "**RNN（时间建模）**\n"
            "- 输入类型：文本、语音、传感器序列\n"
            "- 核心操作：隐藏状态跨时间递推\n"
            "- 参数共享：同一权重在时间步复用\n"
            "- 信息流：层间前馈 + 时间维记忆传递\n"
            "- 典型任务：语言建模、翻译、语音识别"
        )

    compare_img = draw_cnn_vs_rnn()
    img_col1, img_col2, img_col3 = st.columns([1, 3.2, 1], gap="small")
    with img_col2:
        st.image(compare_img, use_container_width=True)
    with st.expander("CNN vs RNN 对比详解（点击展开）", expanded=False):
        st.markdown(
            "- **输入结构差异**：CNN 假设邻近像素相关，RNN 假设前后时刻相关。  \n"
            "- **共享方式差异**：CNN 在空间重复同一个卷积核，RNN 在时间重复同一组递推参数。  \n"
            "- **挑战差异**：CNN 多关注空间尺度变化，RNN 在长序列上更容易出现梯度消失或爆炸。"
        )

    render_parameter_explanation(
        [
            "当前展开时间步为 t={0}，数值演算时间步为 t={1}；拖动任一滑块可观察对应阶段。".format(step, calc_step),
            "Wxh 负责输入贡献，Whh 负责历史隐藏状态贡献；建议分别查看两张权重热图。",
        ]
    )
    render_observation(
        [
            "当前 h_t 已由固定演示序列计算得到，页面没有额外训练参数。",
            "建议比较相邻时间步的 h_t，观察记忆传递带来的数值变化。",
        ]
    )
