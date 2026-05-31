import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_COLORS = ["#2E86DE", "#E67E22", "#27AE60", "#8E44AD"]
BG_COLORS = ["#DCEEFF", "#FBE4D5", "#E2F6E7", "#EEE3FA"]
TEXT_COLOR = (46, 62, 80)
MUTED_TEXT = (92, 112, 129)
TITLE_COLOR = (15, 91, 158)
GRID_COLOR = (230, 236, 242)
ERROR_COLOR = (231, 76, 60)
SUPPORT_COLOR = (241, 196, 15)


FONT_PATHS = [
    "assets/LXGWWenKai-Regular.ttf",
]


def render_main_visual(algorithm_key, context):
    if algorithm_key == "knn":
        return render_knn_visual(context)
    if algorithm_key == "svm":
        return render_svm_visual(context)
    if algorithm_key == "nb":
        return render_nb_visual(context)
    return render_rf_visual(context)


def render_confusion_matrix_image(matrix, class_names, title):
    rows, cols = matrix.shape
    cell = 96
    left = 180
    top = 172
    width = left + cols * cell + 76
    height = top + rows * cell + 70

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = load_font(26, bold=True)
    label_font = load_font(18)
    cell_font = load_font(22, bold=True)
    small_font = load_font(16)

    matrix_width = cols * cell
    title_box = centered_text_box(draw, title, title_font)
    draw.text(((width - title_box[2]) / 2, 20), title, fill=TITLE_COLOR, font=title_font)
    axis_top_box = centered_text_box(draw, "预测类别", label_font)
    draw.text((left + (matrix_width - axis_top_box[2]) / 2, 66), "预测类别", fill=TEXT_COLOR, font=label_font)
    draw.text((42, 122), "真实类别", fill=TEXT_COLOR, font=label_font)

    maximum = max(int(matrix.max()), 1)
    for row_index in range(rows):
        row_text = class_names[row_index]
        row_box = centered_text_box(draw, row_text, label_font)
        row_center_y = top + row_index * cell + cell / 2
        draw.text((44, row_center_y - row_box[3] / 2 - 2), row_text, fill=TEXT_COLOR, font=label_font)
    for col_index in range(cols):
        col_text = class_names[col_index]
        col_box = centered_text_box(draw, col_text, small_font)
        col_center_x = left + col_index * cell + cell / 2
        draw.text((col_center_x - col_box[2] / 2, 124), col_text, fill=TEXT_COLOR, font=small_font)

    for row_index in range(rows):
        for col_index in range(cols):
            value = int(matrix[row_index, col_index])
            strength = value / maximum
            fill = blend_colors((239, 244, 250), (88, 103, 227), strength)
            x0 = left + col_index * cell
            y0 = top + row_index * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rounded_rectangle([x0, y0, x1, y1], radius=12, fill=fill, outline=(219, 228, 238), width=1)
            text_color = (255, 255, 255) if strength > 0.42 else TITLE_COLOR
            number_box = centered_text_box(draw, str(value), cell_font)
            draw.text(
                (x0 + cell / 2 - number_box[2] / 2, y0 + cell / 2 - number_box[3] / 2 - 4),
                str(value),
                fill=text_color,
                font=cell_font,
            )
    return image


def render_knn_visual(context):
    image, draw, plot = create_canvas(
        title=context["visual_title"],
        subtitle="圆点表示训练样本，菱形表示测试样本，背景颜色表示模型划分出的分类区域。",
    )

    x_min, x_max, y_min, y_max = compute_bounds(context["X_train"], context["X_test"])
    region_map = predict_grid_labels(context["model"], context["scaler"], x_min, x_max, y_min, y_max, 300, 220)
    paste_region(image, region_map, plot)
    draw = ImageDraw.Draw(image)
    draw_grid(draw, plot, x_min, x_max, y_min, y_max)
    draw_boundary(image, region_map, plot)
    draw = ImageDraw.Draw(image)

    draw_samples(draw, context["X_train"], context["y_train"], context["X_test"], context["y_test"], context["y_test_pred"], plot, x_min, x_max, y_min, y_max)
    draw_footer_label(draw, plot, "KNN 通过邻近样本投票形成局部边界，因此 K 值变化会直接影响边界平滑程度。")
    return image

def render_svm_visual(context):
    image, draw, plot = create_canvas(
        title=context["visual_title"],
        subtitle="深色线是决策边界，浅色虚带表示间隔区域，黄色圈是支持向量。",
    )

    x_min, x_max, y_min, y_max = compute_bounds(context["X_train"], context["X_test"])
    region_map = predict_grid_labels(context["model"], context["scaler"], x_min, x_max, y_min, y_max, 320, 220)
    score_map = predict_grid_scores(context["model"], context["scaler"], x_min, x_max, y_min, y_max, 320, 220)
    paste_region(image, region_map, plot)
    draw = ImageDraw.Draw(image)
    draw_grid(draw, plot, x_min, x_max, y_min, y_max)
    draw_margin_band(image, score_map, plot, 0.25, (200, 214, 229))
    draw_score_line(image, score_map, plot, 0.0, (45, 63, 84))
    draw_score_line(image, score_map, plot, 1.0, (133, 150, 168))
    draw_score_line(image, score_map, plot, -1.0, (133, 150, 168))
    draw = ImageDraw.Draw(image)

    draw_samples(draw, context["X_train"], context["y_train"], context["X_test"], context["y_test"], context["y_test_pred"], plot, x_min, x_max, y_min, y_max)

    for sv in context["support_vectors"]:
        point = context["scaler"].inverse_transform(sv)
        px, py = project_point(point, plot, x_min, x_max, y_min, y_max)
        draw.ellipse([px - 11, py - 11, px + 11, py + 11], outline=SUPPORT_COLOR, width=3)

    draw_label_box(
        draw,
        plot["left"] + 18,
        plot["top"] + 18,
        260,
        96,
        "SVM 观察重点",
        [
            "支持向量数量：{0}".format(context["support_vector_count"]),
            context["margin_note"],
        ],
    )
    draw_footer_label(draw, plot, "SVM 更关心离边界最近的关键样本，它们决定了间隔和超平面位置。")
    return image


def render_nb_visual(context):
    image, draw, plot = create_canvas(
        title=context["visual_title"],
        subtitle="背景颜色深浅表示分类置信度高低，浅色区域代表模型更犹豫。",
    )

    x_min, x_max, y_min, y_max = compute_bounds(context["X_train"], context["X_test"])
    proba_map = predict_grid_probabilities(context["model"], context["scaler"], x_min, x_max, y_min, y_max, 320, 220)
    paste_probability_region(image, proba_map, plot)
    draw = ImageDraw.Draw(image)
    draw_grid(draw, plot, x_min, x_max, y_min, y_max)
    draw_uncertainty_band(image, proba_map, plot)
    draw = ImageDraw.Draw(image)

    draw_samples(draw, context["X_train"], context["y_train"], context["X_test"], context["y_test"], context["y_test_pred"], plot, x_min, x_max, y_min, y_max)

    for entry in context["distribution_boxes"]:
        draw_axis_aligned_box(draw, entry["center"], entry["std"], plot, x_min, x_max, y_min, y_max, entry["color"])

    draw_footer_label(draw, plot, "朴素贝叶斯比较的是各类别的条件概率，因此边界会随着概率区域变化而调整。")
    return image

def render_rf_visual(context):
    width = 1180
    height = 760
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = load_font(30, bold=True)
    text_font = load_font(18)

    draw.text((44, 22), context["visual_title"], fill=TITLE_COLOR, font=title_font)
    draw.text((44, 62), "左图是单棵树，右图是随机森林；它们使用同一批样本，但森林会通过投票变得更稳定。", fill=TEXT_COLOR, font=text_font)

    left_plot = {"left": 42, "top": 110, "width": 500, "height": 540}
    right_plot = {"left": 640, "top": 110, "width": 500, "height": 540}

    x_min, x_max, y_min, y_max = compute_bounds(context["X_train"], context["X_test"])
    forest_map = predict_grid_labels(context["model"], context["scaler"], x_min, x_max, y_min, y_max, 240, 200)
    single_map = predict_grid_single_tree(context["model"], context["scaler"], x_min, x_max, y_min, y_max, 240, 200)

    paste_region(image, single_map, left_plot)
    paste_region(image, forest_map, right_plot)
    draw = ImageDraw.Draw(image)

    draw_grid(draw, left_plot, x_min, x_max, y_min, y_max)
    draw_grid(draw, right_plot, x_min, x_max, y_min, y_max)
    draw_boundary(image, single_map, left_plot)
    draw_boundary(image, forest_map, right_plot)
    draw = ImageDraw.Draw(image)

    draw_panel_title(draw, left_plot["left"], 88, "单棵决策树")
    draw_panel_title(draw, right_plot["left"], 88, "随机森林投票结果")
    draw_samples(draw, context["X_train"], context["y_train"], context["X_test"], context["y_test"], context["y_test_pred"], left_plot, x_min, x_max, y_min, y_max, show_errors=False)
    draw_samples(draw, context["X_train"], context["y_train"], context["X_test"], context["y_test"], context["y_test_pred"], right_plot, x_min, x_max, y_min, y_max)

    draw_label_box(
        draw,
        44,
        666,
        1094,
        64,
        "观察提示",
        [
            "单棵树往往边界更碎、更容易跟着局部样本摆动；森林会把多棵树的结果综合起来，因此边界通常更稳。",
        ],
    )
    return image


def create_canvas(title, subtitle):
    width = 1320
    height = 920
    plot = {"left": 110, "top": 150, "width": 1100, "height": 560}
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = load_font(30, bold=True)
    text_font = load_font(18)

    draw.text((plot["left"], 24), title, fill=TITLE_COLOR, font=title_font)
    draw.text((plot["left"], 64), subtitle, fill=TEXT_COLOR, font=text_font)
    draw.rectangle(
        [plot["left"], plot["top"], plot["left"] + plot["width"], plot["top"] + plot["height"]],
        outline=(210, 221, 233),
        width=2,
    )
    return image, draw, plot


def compute_bounds(X_train, X_test):
    X_all = np.vstack([X_train, X_test])
    x_min, x_max = X_all[:, 0].min() - 0.8, X_all[:, 0].max() + 0.8
    y_min, y_max = X_all[:, 1].min() - 0.8, X_all[:, 1].max() + 0.8
    return x_min, x_max, y_min, y_max


def predict_grid_labels(model, scaler, x_min, x_max, y_min, y_max, grid_width, grid_height):
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_width), np.linspace(y_max, y_min, grid_height))
    grid = np.c_[xx.ravel(), yy.ravel()]
    labels = model.predict(scaler.transform(grid)).reshape(grid_height, grid_width)
    return labels


def predict_grid_scores(model, scaler, x_min, x_max, y_min, y_max, grid_width, grid_height):
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_width), np.linspace(y_max, y_min, grid_height))
    grid = np.c_[xx.ravel(), yy.ravel()]
    scores = model.decision_values_binary(scaler.transform(grid)).reshape(grid_height, grid_width)
    return scores


def predict_grid_probabilities(model, scaler, x_min, x_max, y_min, y_max, grid_width, grid_height):
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_width), np.linspace(y_max, y_min, grid_height))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probabilities = model.predict_proba(scaler.transform(grid)).reshape(grid_height, grid_width, 2)
    return probabilities


def predict_grid_single_tree(model, scaler, x_min, x_max, y_min, y_max, grid_width, grid_height):
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_width), np.linspace(y_max, y_min, grid_height))
    grid = np.c_[xx.ravel(), yy.ravel()]
    labels = model.single_tree_predict(scaler.transform(grid)).reshape(grid_height, grid_width)
    return labels


def paste_region(image, region_map, plot):
    palette = np.array([hex_to_rgb(color) for color in BG_COLORS], dtype=np.uint8)
    region = palette[region_map]
    region_image = Image.fromarray(region).resize((plot["width"], plot["height"]), resample=Image.NEAREST)
    image.paste(region_image, (plot["left"], plot["top"]))


def paste_probability_region(image, probability_map, plot):
    base_a = np.array(hex_to_rgb(BG_COLORS[0]), dtype=float)
    base_b = np.array(hex_to_rgb(BG_COLORS[1]), dtype=float)
    confidence = np.abs(probability_map[:, :, 1] - 0.5) * 2.0
    mix = probability_map[:, :, 1][:, :, None]
    colors = base_a[None, None, :] * (1.0 - mix) + base_b[None, None, :] * mix
    whiten = 0.48 - confidence[:, :, None] * 0.28
    colors = colors * (1.0 - whiten) + 255.0 * whiten
    region_image = Image.fromarray(colors.astype(np.uint8)).resize((plot["width"], plot["height"]), resample=Image.BILINEAR)
    image.paste(region_image, (plot["left"], plot["top"]))


def draw_samples(draw, X_train, y_train, X_test, y_test, y_test_pred, plot, x_min, x_max, y_min, y_max, show_errors=True):
    class_colors = [hex_to_rgb(CLASS_COLORS[index]) for index in range(len(np.unique(np.concatenate([y_train, y_test]))))]
    wrong_mask = y_test_pred != y_test

    for class_id, color in zip(np.unique(y_train), class_colors):
        train_mask = y_train == class_id
        test_mask = y_test == class_id
        for point in X_train[train_mask]:
            px, py = project_point(point, plot, x_min, x_max, y_min, y_max)
            draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=color, outline=(255, 255, 255), width=1)
        for point, is_wrong in zip(X_test[test_mask], wrong_mask[test_mask]):
            px, py = project_point(point, plot, x_min, x_max, y_min, y_max)
            draw_diamond(draw, px, py, 7, color, (31, 45, 61))
            if show_errors and is_wrong:
                draw.ellipse([px - 11, py - 11, px + 11, py + 11], outline=ERROR_COLOR, width=3)


def draw_grid(draw, plot, x_min, x_max, y_min, y_max):
    font = load_font(14)
    for value in np.linspace(x_min, x_max, 5):
        px = plot["left"] + int((value - x_min) / max(x_max - x_min, 1e-9) * plot["width"])
        draw.line([(px, plot["top"]), (px, plot["top"] + plot["height"])], fill=GRID_COLOR, width=1)
        tick_text = "{0:.1f}".format(value)
        tick_box = centered_text_box(draw, tick_text, font)
        draw.text((px - tick_box[2] / 2, plot["top"] + plot["height"] + 10), tick_text, fill=MUTED_TEXT, font=font)
    for value in np.linspace(y_min, y_max, 5):
        py = plot["top"] + int((y_max - value) / max(y_max - y_min, 1e-9) * plot["height"])
        draw.line([(plot["left"], py), (plot["left"] + plot["width"], py)], fill=GRID_COLOR, width=1)
        tick_text = "{0:.1f}".format(value)
        tick_box = centered_text_box(draw, tick_text, font)
        draw.text((42, py - tick_box[3] / 2 - 2), tick_text, fill=MUTED_TEXT, font=font)
    label_font = load_font(18)
    x_label_box = centered_text_box(draw, "特征 1", label_font)
    draw.text((plot["left"] + plot["width"] / 2 - x_label_box[2] / 2, plot["top"] + plot["height"] + 42), "特征 1", fill=TEXT_COLOR, font=label_font)
    draw.text((42, plot["top"] - 34), "特征 2", fill=TEXT_COLOR, font=label_font)


def draw_boundary(image, region_map, plot):
    region_image = Image.fromarray(np.zeros((region_map.shape[0], region_map.shape[1], 4), dtype=np.uint8), mode="RGBA")
    boundary_draw = ImageDraw.Draw(region_image)
    height, width = region_map.shape
    for y in range(height - 1):
        for x in range(width - 1):
            current = region_map[y, x]
            if current != region_map[y, x + 1] or current != region_map[y + 1, x]:
                boundary_draw.point((x, y), fill=(88, 103, 119, 255))
    boundary_image = region_image.resize((plot["width"], plot["height"]), resample=Image.NEAREST)
    image.paste(boundary_image, (plot["left"], plot["top"]), boundary_image)


def draw_margin_band(image, score_map, plot, threshold, color):
    band = np.zeros((score_map.shape[0], score_map.shape[1], 4), dtype=np.uint8)
    mask = np.abs(score_map) < threshold
    band[mask] = np.array([color[0], color[1], color[2], 120], dtype=np.uint8)
    band_image = Image.fromarray(band, mode="RGBA").resize((plot["width"], plot["height"]), resample=Image.BILINEAR)
    image.paste(band_image, (plot["left"], plot["top"]), band_image)


def draw_score_line(image, score_map, plot, level, color):
    line = np.zeros((score_map.shape[0], score_map.shape[1], 4), dtype=np.uint8)
    mask = np.abs(score_map - level) < 0.06
    line[mask] = np.array([color[0], color[1], color[2], 255], dtype=np.uint8)
    line_image = Image.fromarray(line, mode="RGBA").resize((plot["width"], plot["height"]), resample=Image.NEAREST)
    image.paste(line_image, (plot["left"], plot["top"]), line_image)


def draw_uncertainty_band(image, probability_map, plot):
    line = np.zeros((probability_map.shape[0], probability_map.shape[1], 4), dtype=np.uint8)
    confidence = np.abs(probability_map[:, :, 1] - 0.5)
    mask = confidence < 0.06
    line[mask] = np.array([88, 103, 119, 255], dtype=np.uint8)
    line_image = Image.fromarray(line, mode="RGBA").resize((plot["width"], plot["height"]), resample=Image.NEAREST)
    image.paste(line_image, (plot["left"], plot["top"]), line_image)


def draw_axis_aligned_box(draw, center, std, plot, x_min, x_max, y_min, y_max, color):
    x0 = center[0] - std[0]
    x1 = center[0] + std[0]
    y0 = center[1] - std[1]
    y1 = center[1] + std[1]
    p0 = project_point((x0, y0), plot, x_min, x_max, y_min, y_max)
    p1 = project_point((x1, y1), plot, x_min, x_max, y_min, y_max)
    left = min(p0[0], p1[0])
    top = min(p0[1], p1[1])
    right = max(p0[0], p1[0])
    bottom = max(p0[1], p1[1])
    draw.rounded_rectangle([left, top, right, bottom], radius=10, outline=color, width=2)
    cx, cy = project_point(center, plot, x_min, x_max, y_min, y_max)
    draw.line([(cx - 8, cy), (cx + 8, cy)], fill=color, width=2)
    draw.line([(cx, cy - 8), (cx, cy + 8)], fill=color, width=2)


def draw_label_box(draw, x, y, width, height, title, lines):
    title_font = load_font(20, bold=True)
    text_font = load_font(16)
    wrapped_groups = [wrap_text(draw, line, text_font, width - 32) for line in lines]
    text_line_count = sum(max(len(group), 1) for group in wrapped_groups)
    box_height = max(height, 50 + text_line_count * 22 + 16)
    draw.rounded_rectangle([x, y, x + width, y + box_height], radius=16, fill=(255, 255, 255), outline=(219, 228, 238))
    draw.text((x + 16, y + 12), title, fill=TITLE_COLOR, font=title_font)
    text_y = y + 46
    for wrapped in wrapped_groups:
        for part in wrapped:
            draw.text((x + 16, text_y), part, fill=TEXT_COLOR, font=text_font)
            text_y += 22


def draw_footer_label(draw, plot, message):
    draw_label_box(draw, plot["left"], plot["top"] + plot["height"] + 18, plot["width"], 70, "图像解读", [message])


def draw_panel_title(draw, x, y, title):
    font = load_font(22, bold=True)
    draw.text((x, y), title, fill=TITLE_COLOR, font=font)


def draw_star(draw, x, y, radius, fill, outline):
    points = []
    for index in range(10):
        angle = -np.pi / 2 + index * np.pi / 5
        r = radius if index % 2 == 0 else radius * 0.45
        points.append((x + r * np.cos(angle), y + r * np.sin(angle)))
    draw.polygon(points, fill=fill, outline=outline)


def draw_diamond(draw, x, y, size, fill_color, outline_color):
    points = [(x, y - size), (x + size, y), (x, y + size), (x - size, y)]
    draw.polygon(points, fill=fill_color, outline=outline_color)


def project_point(point, plot, x_min, x_max, y_min, y_max):
    px = plot["left"] + int((point[0] - x_min) / max(x_max - x_min, 1e-9) * plot["width"])
    py = plot["top"] + int((y_max - point[1]) / max(y_max - y_min, 1e-9) * plot["height"])
    return px, py


def load_font(size, bold=False):
    """跨平台字体加载，优先使用 Linux / Streamlit Cloud 可用的中文字体"""
    # 候选字体路径（按优先级排序）
    font_paths = [
        "assets/LXGWWenKai-Regular.ttf",
    ]
    if bold:
        # 粗体优先
        bold_paths = [
            "assets/LXGWWenKai-Regular.ttf"
        ]
        font_paths = bold_paths + font_paths

    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[index : index + 2], 16) for index in (0, 2, 4))


def blend_colors(color_a, color_b, ratio):
    ratio = max(0.0, min(1.0, ratio))
    return tuple(int(color_a[index] * (1.0 - ratio) + color_b[index] * ratio) for index in range(3))


def centered_text_box(draw, text, font):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return left, top, right - left, bottom - top


def wrap_text(draw, text, font, max_width):
    parts = []
    current = ""
    for char in text:
        trial = current + char
        width = centered_text_box(draw, trial, font)[2]
        if current and width > max_width:
            parts.append(current)
            current = char
        else:
            current = trial
    if current:
        parts.append(current)
    return parts


def draw_grid(
    draw,
    plot,
    x_min,
    x_max,
    y_min,
    y_max,
    show_x_axis=True,
    show_y_axis=True,
    tick_left=None,
    y_label_x=None,
    x_label_text="特征 1",
    y_label_text="特征 2",
):
    font = load_font(14)
    tick_left = max(18, plot["left"] - 54) if tick_left is None else tick_left
    y_label_x = tick_left if y_label_x is None else y_label_x

    for value in np.linspace(x_min, x_max, 5):
        px = plot["left"] + int((value - x_min) / max(x_max - x_min, 1e-9) * plot["width"])
        draw.line([(px, plot["top"]), (px, plot["top"] + plot["height"])], fill=GRID_COLOR, width=1)
        if show_x_axis:
            tick_text = "{0:.1f}".format(value)
            tick_box = centered_text_box(draw, tick_text, font)
            draw.text((px - tick_box[2] / 2, plot["top"] + plot["height"] + 10), tick_text, fill=MUTED_TEXT, font=font)

    for value in np.linspace(y_min, y_max, 5):
        py = plot["top"] + int((y_max - value) / max(y_max - y_min, 1e-9) * plot["height"])
        draw.line([(plot["left"], py), (plot["left"] + plot["width"], py)], fill=GRID_COLOR, width=1)
        if show_y_axis:
            tick_text = "{0:.1f}".format(value)
            tick_box = centered_text_box(draw, tick_text, font)
            draw.text((tick_left, py - tick_box[3] / 2 - 2), tick_text, fill=MUTED_TEXT, font=font)

    label_font = load_font(18)
    if show_x_axis:
        x_label_box = centered_text_box(draw, x_label_text, label_font)
        draw.text(
            (plot["left"] + plot["width"] / 2 - x_label_box[2] / 2, plot["top"] + plot["height"] + 34),
            x_label_text,
            fill=TEXT_COLOR,
            font=label_font,
        )
    if show_y_axis:
        draw.text((y_label_x, plot["top"] - 34), y_label_text, fill=TEXT_COLOR, font=label_font)


def draw_plot_frame(draw, plot):
    draw.rectangle(
        [plot["left"], plot["top"], plot["left"] + plot["width"], plot["top"] + plot["height"]],
        outline=(210, 221, 233),
        width=2,
    )


def draw_panel_title_centered(draw, plot, title):
    font = load_font(22, bold=True)
    title_box = centered_text_box(draw, title, font)
    title_x = plot["left"] + plot["width"] / 2 - title_box[2] / 2
    draw.text((title_x, plot["top"] - 46), title, fill=TITLE_COLOR, font=font)


def render_rf_visual(context):
    width = 1320
    height = 920
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = load_font(30, bold=True)
    text_font = load_font(18)

    draw.text((60, 24), context["visual_title"], fill=TITLE_COLOR, font=title_font)
    draw.text(
        (60, 66),
        "左图是单棵树，右图是随机森林；它们使用同一批样本，但森林会通过投票变得更稳定。",
        fill=TEXT_COLOR,
        font=text_font,
    )

    left_plot = {"left": 70, "top": 182, "width": 540, "height": 500}
    right_plot = {"left": 710, "top": 182, "width": 540, "height": 500}

    x_min, x_max, y_min, y_max = compute_bounds(context["X_train"], context["X_test"])
    forest_map = predict_grid_labels(context["model"], context["scaler"], x_min, x_max, y_min, y_max, 240, 200)
    single_map = predict_grid_single_tree(context["model"], context["scaler"], x_min, x_max, y_min, y_max, 240, 200)

    paste_region(image, single_map, left_plot)
    paste_region(image, forest_map, right_plot)
    draw = ImageDraw.Draw(image)
    draw_plot_frame(draw, left_plot)
    draw_plot_frame(draw, right_plot)

    draw_grid(draw, left_plot, x_min, x_max, y_min, y_max, show_y_axis=True, tick_left=22, y_label_x=12)
    draw_grid(draw, right_plot, x_min, x_max, y_min, y_max, show_y_axis=False)
    draw_boundary(image, single_map, left_plot)
    draw_boundary(image, forest_map, right_plot)
    draw = ImageDraw.Draw(image)

    draw_panel_title_centered(draw, left_plot, "单棵决策树")
    draw_panel_title_centered(draw, right_plot, "随机森林投票结果")
    draw_samples(
        draw,
        context["X_train"],
        context["y_train"],
        context["X_test"],
        context["y_test"],
        context["y_test_pred"],
        left_plot,
        x_min,
        x_max,
        y_min,
        y_max,
        show_errors=False,
    )
    draw_samples(
        draw,
        context["X_train"],
        context["y_train"],
        context["X_test"],
        context["y_test"],
        context["y_test_pred"],
        right_plot,
        x_min,
        x_max,
        y_min,
        y_max,
    )

    draw_label_box(
        draw,
        70,
        782,
        1180,
        86,
        "观察提示",
        [
            "单棵树往往边界更碎、更容易跟着局部样本摆动；森林会把多棵树的结果综合起来，因此边界通常更稳。",
        ],
    )
    return image


def draw_footer_label(draw, plot, message):
    footer_top = plot["top"] + plot["height"] + 88
    draw_label_box(draw, plot["left"], footer_top, plot["width"], 80, "图像解读", [message])
