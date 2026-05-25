"""Transformer 教学可视化模块。

只使用静态图 + 文字讲解，不包含交互式演示控件。
"""

import os

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


_BG = (15, 23, 42)
_TEXT = (230, 230, 230)
_EDGE = (88, 103, 125)


def _load_font(size: int):
    """跨平台字体加载，优先使用 Linux / Streamlit Cloud 可用的中文字体"""
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_arrow(draw: ImageDraw.ImageDraw, p0: tuple, p1: tuple, color=(150, 150, 150), width=2):
    draw.line([p0, p1], fill=color, width=width)
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    head = max(6, width * 3)
    hw = max(3, head // 2)
    tip = p1
    left = (int(tip[0] - ux * head + px * hw), int(tip[1] - uy * head + py * hw))
    right = (int(tip[0] - ux * head - px * hw), int(tip[1] - uy * head - py * hw))
    draw.polygon([tip, left, right], fill=color)


def _draw_round_text(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, text: str, fill: tuple, font):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=fill, outline=_EDGE, width=1)
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text((x + (w - tw) // 2, y + (h - th) // 2), text, fill=(15, 23, 42), font=font)


def draw_full_transformer() -> Image.Image:
    """Classic Transformer Encoder-Decoder architecture diagram."""
    W, H = 860, 560

    ENC_X, ENC_BW = 50, 260
    DEC_X, DEC_BW = 540, 260
    BH, STEP = 32, 46

    EMBED_Y = 56
    PE_R    = 11
    PE_CY   = EMBED_Y + BH + 18       # 56+32+18 = 106
    NX_Y    = PE_CY + PE_R + 16        # 106+11+16 = 133

    enc_cx = ENC_X + ENC_BW // 2       # 180
    dec_cx = DEC_X + DEC_BW // 2       # 670

    img  = Image.new("RGB", (W, H), _BG)
    draw = ImageDraw.Draw(img)
    fn   = _load_font(12)
    sm   = _load_font(10)
    ti   = _load_font(14)

    # ── Title ────────────────────────────────────────────────────────────
    draw.text((20, 10), "Transformer: Complete Architecture (Encoder-Decoder)",
              fill=_TEXT, font=ti)

    # ── Column titles ────────────────────────────────────────────────────
    for text, cx, col in [("Encoder", enc_cx, (125, 211, 252)),
                           ("Decoder", dec_cx, (196, 181, 253))]:
        tb = draw.textbbox((0, 0), text, font=fn)
        draw.text((cx - (tb[2] - tb[0]) // 2, 32), text, fill=col, font=fn)

    # ── Embedding blocks ─────────────────────────────────────────────────
    _draw_round_text(draw, ENC_X, EMBED_Y, ENC_BW, BH, "Input Embedding",
                     (96, 165, 250), fn)
    _draw_round_text(draw, DEC_X, EMBED_Y, DEC_BW, BH, "Output Embedding",
                     (196, 181, 253), fn)
    # ── PE ⊕ circles ─────────────────────────────────────────────────────
    for cx, col in [(enc_cx, (125, 211, 252)), (dec_cx, (196, 181, 253))]:
        draw.ellipse([cx - PE_R, PE_CY - PE_R, cx + PE_R, PE_CY + PE_R],
                     outline=col, width=2)
        draw.line([(cx - 6, PE_CY), (cx + 6, PE_CY)], fill=col, width=2)
        draw.line([(cx, PE_CY - 6), (cx, PE_CY + 6)],  fill=col, width=2)
    draw.text((enc_cx + PE_R + 4, PE_CY - 6), "Positional Encoding",
              fill=(160, 180, 210), font=sm)
    draw.text((dec_cx + PE_R + 4, PE_CY - 6), "Positional Encoding",
              fill=(160, 180, 210), font=sm)

    # Embed → PE → Nx arrows
    for cx in [enc_cx, dec_cx]:
        _draw_arrow(draw, (cx, EMBED_Y + BH),  (cx, PE_CY - PE_R),
                    color=(148, 163, 184), width=2)
        _draw_arrow(draw, (cx, PE_CY + PE_R),  (cx, NX_Y),
                    color=(148, 163, 184), width=2)

    # ── Encoder Nx blocks ─────────────────────────────────────────────────
    enc_labels = ["Multi-Head Attention", "Add & Norm",
                  "Feed Forward",         "Add & Norm"]
    enc_cols   = [(125, 211, 252), (147, 197, 253),
                  (186, 230, 253), (147, 197, 253)]
    for i, (lab, col) in enumerate(zip(enc_labels, enc_cols)):
        yy = NX_Y + i * STEP
        _draw_round_text(draw, ENC_X, yy, ENC_BW, BH, lab, col, fn)
        if i < len(enc_labels) - 1:
            _draw_arrow(draw, (enc_cx, yy + BH), (enc_cx, yy + STEP),
                        color=(148, 163, 184), width=2)

    # ── Decoder Nx blocks ─────────────────────────────────────────────────
    dec_labels = ["Masked Multi-Head Attention", "Add & Norm",
                  "Multi-Head Attention",   "Add & Norm",
                  "Feed Forward",           "Add & Norm"]
    dec_cols   = [(167, 139, 250), (196, 181, 253),
                  (125, 211, 252), (196, 181, 253),
                  (186, 167, 253), (196, 181, 253)]
    for i, (lab, col) in enumerate(zip(dec_labels, dec_cols)):
        yy = NX_Y + i * STEP
        _draw_round_text(draw, DEC_X, yy, DEC_BW, BH, lab, col, fn)
        if i < len(dec_labels) - 1:
            _draw_arrow(draw, (dec_cx, yy + BH), (dec_cx, yy + STEP),
                        color=(148, 163, 184), width=2)

    # ── Nx brackets ───────────────────────────────────────────────────────
    enc_nx_top = NX_Y - 8
    enc_nx_bot = NX_Y + (len(enc_labels) - 1) * STEP + BH + 8   # 311
    dec_nx_top = NX_Y - 8
    dec_nx_bot = NX_Y + (len(dec_labels) - 1) * STEP + BH + 8   # 403

    draw.rounded_rectangle(
        [ENC_X - 10, enc_nx_top, ENC_X + ENC_BW + 10, enc_nx_bot],
        radius=10, outline=(88, 103, 125), width=2)
    draw.rounded_rectangle(
        [DEC_X - 10, dec_nx_top, DEC_X + DEC_BW + 10, dec_nx_bot],
        radius=10, outline=(88, 103, 125), width=2)
    draw.text((ENC_X + ENC_BW + 16, (enc_nx_top + enc_nx_bot) // 2 - 8),
              "×N", fill=(180, 200, 220), font=fn)
    draw.text((DEC_X + DEC_BW + 16, (dec_nx_top + dec_nx_bot) // 2 - 8),
              "×N", fill=(180, 200, 220), font=fn)

    # ── Residual bypass connections (orange U-shapes on left of each group) ──
    RES_COL = (245, 158, 11)

    def _residual(base_x, group_start):
        y_top = NX_Y + group_start * STEP + BH // 2
        y_bot = NX_Y + (group_start + 1) * STEP + BH // 2
        lx    = base_x - 20
        draw.line([(base_x, y_top), (lx, y_top)], fill=RES_COL, width=2)
        draw.line([(lx, y_top),     (lx, y_bot)], fill=RES_COL, width=2)
        _draw_arrow(draw, (lx, y_bot), (base_x, y_bot), color=RES_COL, width=2)

    for gs in [0, 2]:       # Encoder: MHA+A&N, FFN+A&N
        _residual(ENC_X, gs)
    for gs in [0, 2, 4]:    # Decoder: 3 sub-layer groups
        _residual(DEC_X, gs)

    # ── Cross-Attention K,V arrow ─────────────────────────────────────────
    # From right edge of Encoder's last Add&Norm → Decoder's Multi-Head Attention
    src_x = ENC_X + ENC_BW + 10           # 320 (just outside Enc Nx bracket)
    src_y = NX_Y + 3 * STEP + BH // 2    # 287 (Enc block-3 center y)
    dst_x = DEC_X - 10                    # 530 (just outside Dec Nx bracket)
    dst_y = NX_Y + 2 * STEP + BH // 2    # 241 (Dec block-2 center y)
    mid_x = (src_x + dst_x) // 2          # 425

    draw.line([(src_x, src_y), (mid_x, src_y)], fill=(251, 191, 36), width=2)
    draw.line([(mid_x, src_y), (mid_x, dst_y)], fill=(251, 191, 36), width=2)
    _draw_arrow(draw, (mid_x, dst_y), (dst_x, dst_y), color=(251, 191, 36), width=2)
    draw.text((mid_x + 6, (src_y + dst_y) // 2 - 8), "K, V",
              fill=(251, 191, 36), font=sm)

    # ── Decoder output head: Nx → Linear → Softmax → Output Probabilities ─
    linear_y  = dec_nx_bot + 14            # 417
    softmax_y = linear_y + STEP            # 463
    out_y     = softmax_y + BH + 14        # 509

    _draw_round_text(draw, DEC_X, linear_y,  DEC_BW, BH, "Linear",
                     (196, 181, 253), fn)
    _draw_round_text(draw, DEC_X, softmax_y, DEC_BW, BH, "Softmax",
                     (147, 197, 253), fn)

    _draw_arrow(draw, (dec_cx, dec_nx_bot),       (dec_cx, linear_y),
                color=(148, 163, 184), width=2)
    _draw_arrow(draw, (dec_cx, linear_y + BH),    (dec_cx, softmax_y),
                color=(148, 163, 184), width=2)
    _draw_arrow(draw, (dec_cx, softmax_y + BH),   (dec_cx, out_y - 4),
                color=(148, 163, 184), width=2)

    op_text = "Output Probabilities"
    tb = draw.textbbox((0, 0), op_text, font=fn)
    draw.text((dec_cx - (tb[2] - tb[0]) // 2, out_y), op_text, fill=_TEXT, font=fn)

    return img


def draw_qkv_flow() -> Image.Image:
    img = Image.new("RGB", (1080, 330), _BG)
    draw = ImageDraw.Draw(img)
    font = _load_font(13)
    small = _load_font(11)

    draw.text((20, 12), "Q/K/V 计算与注意力聚合流程", fill=_TEXT, font=font)
    tokens = ["I", "love", "AI"]
    for i, t in enumerate(tokens):
        x = 24 + i * 92
        _draw_round_text(draw, x, 90, 76, 42, t, (96, 165, 250), small)
        _draw_arrow(draw, (x + 38, 133), (x + 38, 175), color=(148, 163, 184), width=2)
        _draw_round_text(draw, x - 8, 176, 30, 26, "Q", (34, 197, 94), small)
        _draw_round_text(draw, x + 23, 176, 30, 26, "K", (245, 158, 11), small)
        _draw_round_text(draw, x + 54, 176, 30, 26, "V", (167, 139, 250), small)

    _draw_round_text(draw, 340, 95, 160, 40, "Q @ K^T", (125, 211, 252), font)
    _draw_round_text(draw, 340, 160, 160, 40, "Softmax", (147, 197, 253), font)
    _draw_round_text(draw, 340, 225, 160, 40, "权重 x V", (186, 230, 253), font)
    _draw_arrow(draw, (295, 189), (334, 114), color=(148, 163, 184), width=2)
    _draw_arrow(draw, (295, 189), (334, 179), color=(148, 163, 184), width=2)
    _draw_arrow(draw, (295, 189), (334, 244), color=(148, 163, 184), width=2)
    _draw_arrow(draw, (420, 135), (420, 156), color=(71, 85, 105), width=2)
    _draw_arrow(draw, (420, 200), (420, 221), color=(71, 85, 105), width=2)

    scores = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.15, 0.25, 0.6]], dtype=np.float32)
    base_x, base_y = 560, 78
    draw.text((560, 48), "示意注意力矩阵 A", fill=_TEXT, font=small)
    for r in range(3):
        for c in range(3):
            v = float(scores[r, c])
            col = (int(45 + 140 * v), int(80 + 120 * v), int(170 + 70 * v))
            x0, y0 = base_x + c * 54, base_y + r * 54
            draw.rectangle([x0, y0, x0 + 48, y0 + 48], fill=col, outline=(100, 120, 150))
            draw.text((x0 + 9, y0 + 16), f"{v:.2f}", fill=(236, 253, 245), font=small)

    _draw_arrow(draw, (724, 157), (790, 157), color=(148, 163, 184), width=2)
    _draw_round_text(draw, 798, 132, 250, 50, "输出上下文表示 Z", (125, 211, 252), font)
    draw.text((796, 197), "每个词的输出都融合了全句信息", fill=(180, 200, 220), font=small)
    return img


def draw_multihead_attention() -> Image.Image:
    # 布局：输入X | 4个(Head+Attention)纵向列 | 汇聚列 | Concat->W^O | 输出
    # 各列中心 x 坐标
    N_HEADS = 4
    W, H = 1080, 380
    img = Image.new("RGB", (W, H), _BG)
    draw = ImageDraw.Draw(img)
    font = _load_font(13)
    small = _load_font(11)
    draw.text((20, 12), "多头注意力：并行关注不同关系，再拼接", fill=_TEXT, font=font)

    # 各区 x 起点（节点宽 120，间距 20）
    INPUT_X, INPUT_Y, INPUT_W, INPUT_H = 24, 158, 130, 46

    HEAD_W, HEAD_H = 120, 40
    ATTN_W, ATTN_H = 120, 36
    COL_STEP = 155          # 每列间距
    HEAD_X0 = 200           # 第一个 Head 的左边 x
    HEAD_Y = 60             # Head 框顶部 y
    ATTN_Y = HEAD_Y + HEAD_H + 18  # Attention 框顶部 y

    MERGE_X = HEAD_X0 + N_HEADS * COL_STEP  # 汇聚点 x（每行箭头先水平到此）
    CONCAT_X = MERGE_X + 48
    CONCAT_Y = 140
    CONCAT_W, CONCAT_H = 110, 44
    WO_X = CONCAT_X + CONCAT_W + 24
    WO_Y = CONCAT_Y
    WO_W, WO_H = 80, 44
    OUT_X = WO_X + WO_W + 24
    OUT_Y = CONCAT_Y
    OUT_W, OUT_H = 160, 44

    head_cols = [(34, 197, 94), (56, 189, 248), (167, 139, 250), (251, 191, 36)]

    # 画输入 X
    _draw_round_text(draw, INPUT_X, INPUT_Y, INPUT_W, INPUT_H, "输入 X", (96, 165, 250), font)
    input_cy = INPUT_Y + INPUT_H // 2

    for i in range(N_HEADS):
        hx = HEAD_X0 + i * COL_STEP
        hcx = hx + HEAD_W // 2     # 当前列中心 x

        # 输入 X -> 当前列顶部（折线：水平+竖直）
        draw.line([(INPUT_X + INPUT_W, input_cy), (hcx, input_cy), (hcx, HEAD_Y)],
                  fill=(148, 163, 184), width=2)

        # Head 框
        _draw_round_text(draw, hx, HEAD_Y, HEAD_W, HEAD_H, f"Head {i+1}", head_cols[i], font)

        # Head -> Attention 竖向箭头
        _draw_arrow(draw, (hcx, HEAD_Y + HEAD_H), (hcx, ATTN_Y),
                    color=(71, 85, 105), width=2)

        # Attention 框
        _draw_round_text(draw, hx, ATTN_Y, ATTN_W, ATTN_H, "Attention", (125, 211, 252), small)

        # Attention 右侧 -> 汇聚列（全部平齐到同一 y，再向右汇聚到 MERGE_X）
        attn_right_y = ATTN_Y + ATTN_H // 2
        attn_right_x = hx + ATTN_W
        # 水平引到 MERGE_X
        draw.line([(attn_right_x, attn_right_y), (MERGE_X, attn_right_y)],
                  fill=(148, 163, 184), width=2)
        # 从 MERGE_X 处竖线向下/上到 CONCAT_Y 中心
        concat_cy = CONCAT_Y + CONCAT_H // 2
        draw.line([(MERGE_X, attn_right_y), (MERGE_X, concat_cy)],
                  fill=(148, 163, 184), width=2)

    # 汇聚点一根箭头指向 Concat
    _draw_arrow(draw, (MERGE_X, CONCAT_Y + CONCAT_H // 2),
                (CONCAT_X, CONCAT_Y + CONCAT_H // 2),
                color=(148, 163, 184), width=2)

    # Concat -> W^O
    _draw_round_text(draw, CONCAT_X, CONCAT_Y, CONCAT_W, CONCAT_H, "Concat", (147, 197, 253), font)
    _draw_arrow(draw, (CONCAT_X + CONCAT_W, CONCAT_Y + CONCAT_H // 2),
                (WO_X, WO_Y + WO_H // 2), color=(71, 85, 105), width=2)

    # W^O -> 输出
    _draw_round_text(draw, WO_X, WO_Y, WO_W, WO_H, "W^O", (186, 230, 253), font)
    _draw_arrow(draw, (WO_X + WO_W, WO_Y + WO_H // 2),
                (OUT_X, OUT_Y + OUT_H // 2), color=(71, 85, 105), width=2)

    # 最终输出
    _draw_round_text(draw, OUT_X, OUT_Y, OUT_W, OUT_H, "最终输出表示", (125, 211, 252), font)

    # 说明文字
    draw.text((24, H - 30), "每个 Head 独立计算 Attention，右侧并行结果汇聚后拼接(Concat)并经 W^O 映射",
              fill=(180, 200, 220), font=_load_font(11))
    return img


def _draw_round_text_multiline(
    draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
    lines: list, fill: tuple, font
):
    """圆角矩形内多行文字整体垂直居中。"""
    draw.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=fill, outline=_EDGE, width=1)
    line_h_list = []
    for ln in lines:
        tb = draw.textbbox((0, 0), ln, font=font)
        line_h_list.append(tb[3] - tb[1])
    gap = 6
    total_h = sum(line_h_list) + gap * (len(lines) - 1)
    cy = y + (h - total_h) // 2
    for idx, ln in enumerate(lines):
        tb = draw.textbbox((0, 0), ln, font=font)
        tw = tb[2] - tb[0]
        draw.text((x + (w - tw) // 2, cy), ln, fill=(15, 23, 42), font=font)
        cy += line_h_list[idx] + gap


def draw_encoder_layer() -> Image.Image:
    # 布局参数
    BW, BH = 280, 68     # 节点宽高
    BX = 100             # 节点左边 x（留出左侧残差线空间）
    GAP = 28             # 节点间箭头间距
    TOP_MARGIN = 54      # 标题区高度
    BOT_MARGIN = 52      # 底部说明文字区高度
    STEP = BH + GAP      # 每行的步距

    blocks = [
        (["输入", "词向量 + 位置编码"], (96, 165, 250)),
        (["Multi-Head Attention"],     (125, 211, 252)),
        (["Add & Norm"],               (147, 197, 253)),
        (["Feed Forward"],             (186, 230, 253)),
        (["Add & Norm"],               (147, 197, 253)),
        (["输出", "上下文表示"],        (96, 165, 250)),
    ]
    N = len(blocks)
    canvas_h = TOP_MARGIN + N * STEP - GAP + BOT_MARGIN + 10
    canvas_w = BX + BW + 60
    img = Image.new("RGB", (canvas_w, canvas_h), _BG)
    draw = ImageDraw.Draw(img)
    font = _load_font(13)
    title_font = _load_font(14)
    cap_font = _load_font(11)

    # 标题
    draw.text((BX, 16), "Transformer Encoder 单层结构", fill=_TEXT, font=title_font)

    # 计算各块顶部 y 坐标
    ys = [TOP_MARGIN + i * STEP for i in range(N)]
    cx = BX + BW // 2

    # 画节点与箭头
    for i, ((lines, col), yy) in enumerate(zip(blocks, ys)):
        _draw_round_text_multiline(draw, BX, yy, BW, BH, lines, col, font)
        if i < N - 1:
            # 主链竖向箭头
            _draw_arrow(draw, (cx, yy + BH), (cx, yy + BH + GAP),
                        color=(148, 163, 184), width=2)

    # ---- 残差连接（橙色折线，左侧统一 x = BX - 30）----
    RES_X = BX - 30   # 残差线的 x 列

    def residual(from_idx: int, to_idx: int):
        """从 from_idx 块的左侧中部，沿左侧垂直线，连到 to_idx 块的左侧中部，再水平接入节点左边。"""
        y_from = ys[from_idx] + BH // 2
        y_to   = ys[to_idx]   + BH // 2
        # 从节点左边沿连出去
        draw.line([(BX, y_from), (RES_X, y_from)], fill=(245, 158, 11), width=2)
        # 竖直段
        draw.line([(RES_X, y_from), (RES_X, y_to)], fill=(245, 158, 11), width=2)
        # 水平接入下方节点左边
        _draw_arrow(draw, (RES_X, y_to), (BX, y_to), color=(245, 158, 11), width=2)

    # 第一条残差：Input -> Add&Norm[2]（跨越 MHA）
    residual(0, 2)
    # 第二条残差：Add&Norm[2] 处起点 -> Add&Norm[4]（跨越 FFN）
    # 起点借用第一个 Add&Norm 的右侧（复用 from_idx=2）
    residual(2, 4)

    # ---- 底部说明（留在最后，不与节点重叠）----
    cap_y = canvas_h - BOT_MARGIN + 10
    cap = "左侧橙色折线为残差连接（Residual），将前层输出直接加到子层结果上，稳定深层训练。"
    tb = draw.textbbox((0, 0), cap, font=cap_font)
    cap_w = tb[2] - tb[0]
    cap_x = (canvas_w - cap_w) // 2
    draw.text((cap_x, cap_y), cap, fill=(251, 191, 36), font=cap_font)

    return img


def draw_transformer_vs_rnn_cnn() -> Image.Image:
    img = Image.new("RGB", (1080, 310), _BG)
    draw = ImageDraw.Draw(img)
    font = _load_font(13)
    small = _load_font(11)
    draw.text((20, 12), "CNN / RNN / Transformer 信息建模方式对比", fill=_TEXT, font=font)

    sections = [("CNN", 30, (96, 165, 250)), ("RNN", 390, (34, 197, 94)), ("Transformer", 750, (167, 139, 250))]
    for name, sx, col in sections:
        _draw_round_text(draw, sx, 50, 300, 42, name, col, font)

    # cnn
    for i, lab in enumerate(["Input", "Conv", "Pool", "FC"]):
        xx = 44 + i * 72
        _draw_round_text(draw, xx, 132, 60, 34, lab, (147, 197, 253), small)
        if i < 3:
            _draw_arrow(draw, (xx + 60, 149), (xx + 70, 149), color=(71, 85, 105), width=2)
    draw.text((42, 202), "空间局部感受野，逐层抽象", fill=(180, 200, 220), font=small)

    # rnn
    for i in range(4):
        xx = 412 + i * 65
        _draw_round_text(draw, xx, 122, 40, 28, f"h{i+1}", (110, 231, 183), small)
        if i < 3:
            _draw_arrow(draw, (xx + 40, 136), (xx + 62, 136), color=(245, 158, 11), width=2)
    draw.text((402, 202), "沿时间步递推，长程路径较长", fill=(180, 200, 220), font=small)

    # transformer
    for i in range(4):
        xx = 770 + i * 64
        _draw_round_text(draw, xx, 126, 38, 26, f"t{i+1}", (196, 181, 253), small)
    pairs = [(0, 1), (0, 3), (1, 2), (2, 3), (3, 1)]
    for a, b in pairs:
        x1, x2 = 789 + a * 64, 789 + b * 64
        _draw_arrow(draw, (x1, 170), (x2, 170), color=(167, 139, 250), width=1)
    draw.text((760, 202), "全局两两可见，并行建模依赖", fill=(180, 200, 220), font=small)
    return img


def nv_render_transformer_viz():
    """渲染 Transformer 教学页面（静态图 + 文字详解）。"""
    st.title("注意力机制网络（Transformer）")
    st.caption("不再按顺序传递记忆，而是让所有位置直接建立关联")

    if st.button("🏠 返回首页", key="trans_back_home", use_container_width=False):
        st.session_state.current_page = "home"
        st.rerun()
    st.markdown("---")
    st.subheader("为什么需要它？")
    st.info(
        "RNN 需要逐步递推，长距离依赖会在多步传递中衰减。"
        "Transformer 通过自注意力在一次前向中建立全局关系，并天然适合并行计算。"
    )

    st.divider()
    st.subheader("整体架构：Encoder-Decoder 全景")
    st.image(draw_full_transformer(), use_container_width=True)
    st.markdown(
        "- 左侧 Encoder 对输入序列做上下文编码，右侧 Decoder 基于已生成前缀逐步预测下一个词。\n"
        "- Decoder 的 Cross-Attention 会读取 Encoder 输出的 `K, V`，实现输入与输出对齐。\n"
        "- Decoder 内部先做 Masked MHA（防止看到未来词），再做 Cross-Attention 和 FFN。\n"
        "- 末端 `Linear + Softmax` 将隐藏表示映射为词表概率，完成 token 级生成。"
    )
    with st.expander("读图详解：一轮 Encoder-Decoder 是怎么工作的", expanded=False):
        st.markdown(
            "1. 输入句子先经 Encoder 堆叠层得到全句上下文矩阵。\n"
            "2. Decoder 接收“已生成词（右移）”并先做 Masked 自注意力。\n"
            "3. Decoder 在 Cross-Attention 中用自身 Query 读取 Encoder 提供的 Key/Value。\n"
            "4. 经过 FFN、Linear、Softmax 产出下一个词分布，再进入下一步生成。"
        )

    st.subheader("核心机制一：自注意力（Q / K / V）")
    st.latex(r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V")
    st.image(draw_qkv_flow(), use_container_width=True)
    st.markdown(
        "- 每个词向量会映射为 Query、Key、Value 三组表示。\n"
        "- 通过 `QK^T` 得到相关性分数，softmax 后形成可解释权重。\n"
        "- 权重对 `V` 加权求和，得到融合上下文后的输出表示。"
    )
    with st.expander("读图详解：Q/K/V 图中每一块在表示什么", expanded=False):
        st.markdown(
            "左侧是输入词；中部是 Q/K/V 线性投影；右侧是注意力矩阵与最终聚合输出。"
            "注意力矩阵颜色越亮，表示该位置对当前词贡献越大。"
        )

    st.divider()
    st.subheader("核心机制二：多头注意力（Multi-Head Attention）")
    st.image(draw_multihead_attention(), use_container_width=True)
    st.markdown(
        "- 多个头并行计算注意力，每个头关注不同类型的依赖关系。\n"
        "- 各头结果拼接（Concat）后再经线性映射 `W^O` 融合。\n"
        "- 这样既保留多视角信息，又保持输出维度一致。"
    )
    with st.expander("读图详解：为什么需要多头", expanded=False):
        st.markdown(
            "单头通常只能学到一种主导模式；多头让模型同时关注句法、语义、位置等不同关系，"
            "最终融合成更稳健的上下文表示。"
        )

    st.divider()
    st.subheader("核心机制三：Encoder 单层结构")
    enc_col1, enc_col2, enc_col3 = st.columns([1, 1.4, 1])
    with enc_col2:
        st.image(draw_encoder_layer(), use_container_width=True)
    st.markdown(
        "- 主干路径：Input -> Multi-Head Attention -> Add&Norm -> FFN -> Add&Norm -> Output。\n"
        "- 两处 Add&Norm 前都有残差连接，帮助信息与梯度稳定传播。\n"
        "- FFN 在每个位置独立执行非线性变换，增强表达能力。"
    )
    with st.expander("读图详解：Encoder 图中残差连接的意义", expanded=False):
        st.markdown(
            "残差连接允许子层学习“增量修正”而不是重写全部表示，训练更稳定，深层堆叠更容易。"
        )

    st.divider()
    st.subheader("完整架构：除了Encoder之外，Decoder 在做什么？")
    st.info(
        "Encoder 负责把输入序列“理解”为上下文表示；Decoder 负责基于这些表示一步步生成目标序列。"
    )
    st.markdown(
        "- Decoder 常见三层子结构：Masked Multi-Head Attention（屏蔽未来）"
        " -> Cross-Attention（读取 Encoder 输出） -> Feed Forward。\n"
        "- 在 Cross-Attention 里，Q 来自 Decoder 当前状态，K/V 来自 Encoder 输出，"
        "这一步实现了输入与输出之间的对齐。\n"
        "- 训练阶段通常用 Teacher Forcing 并行优化；推理阶段按自回归方式逐词生成。"
    )
    with st.expander("读图详解：Encoder-Decoder 如何协作", expanded=False):
        st.markdown(
            "- 输入句子先经过 Encoder，得到整句的上下文矩阵。\n"
            "- Decoder 在每一步看到“已生成词 + 上下文矩阵”，预测下一个词。\n"
            "- 这个过程可概括为：`输入句 -> Encoder -> 上下文矩阵 -> Decoder + 已生成词 -> 下一个词`。"
        )

    st.divider()
    st.subheader("换个说法来理解注意力与位置编码")
    with st.expander("注意力机制：你是怎么读句子的？", expanded=True):
        st.markdown(
            "- 读“猫追老鼠”时，你理解“追”不会只看相邻一个字，而会同时看“猫”和“老鼠”。\n"
            "- 自注意力也是这个逻辑：每个词都能看整句，再按“谁更重要”分配权重。\n"
            "- Q 可以理解为“我正在找什么”，K 是“我能提供什么标签”，V 是“我携带的具体信息”。\n"
            "- 先用 Q 和所有 K 打分，再用 softmax 归一化成权重，最后对各个 V 做加权汇总。"
        )
    with st.expander("位置编码：给词语标上座位号", expanded=True):
        st.markdown(
            "- Transformer 并行计算，本身没有天然先后顺序概念。\n"
            "- 位置编码就像给每个词额外贴上“座位号”，告诉模型谁在前、谁在后。\n"
            "- 正弦/余弦位置编码在不同维度使用不同频率，既能区分绝对位置，"
            "也便于模型感知相对距离。\n"
            "- 所以 Transformer 最终既能全局互看，也不会丢失顺序信息。"
        )

    st.divider()
    st.subheader("与 CNN / RNN 的差异对比")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            "**CNN（空间建模）**\n"
            "- 局部感受野 + 参数共享\n"
            "- 擅长图像等网格数据\n\n"
            "**RNN（时间递推）**\n"
            "- 隐状态跨时间传递\n"
            "- 长序列依赖路径较长"
        )
    with right:
        st.markdown(
            "**Transformer（全局注意力）**\n"
            "- 任意位置可直接交互\n"
            "- 并行计算效率高\n"
            "- 对长程依赖更友好（配合位置编码）"
        )
    st.image(draw_transformer_vs_rnn_cnn(), use_container_width=True)
    with st.expander("读图详解：三类模型信息流的本质区别", expanded=False):
        st.markdown(
            "CNN 在空间上“局部到全局”；RNN 在时间上“逐步传递”；"
            "Transformer 在单层内就可以“全局互看”，因此更容易捕获远距离关系。"
        )
