import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLUSTER_HEX = ["#2E86DE", "#E67E22", "#27AE60", "#8E44AD", "#16A085", "#C0392B"]
TEXT_COLOR = (46, 62, 80)
MUTED_TEXT = (92, 112, 129)
TITLE_COLOR = (15, 91, 158)
GRID_COLOR = (230, 236, 242)
OUTLINE_COLOR = (214, 226, 238)
NOISE_COLOR = (130, 140, 150)
FOCUS_COLOR = (231, 76, 60)


FONT_PATHS = [
    "assets/LXGWWenKai-Regular.ttf",
]


def render_main_visual(algorithm_key, context):
    if algorithm_key == "kmeans":
        return render_kmeans_visual(context)
    if algorithm_key == "dbscan":
        return render_dbscan_visual(context)
    if algorithm_key == "agg":
        return render_agglomerative_visual(context)
    return render_gmm_visual(context)


def render_analysis_visual(algorithm_key, context):
    image = Image.new("RGB", (1280, 580), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = load_font(28, bold=True)
    text_font = load_font(17)

    draw.text((54, 22), "结构分析", fill=TITLE_COLOR, font=title_font)
    draw.text((54, 60), "左侧看当前簇规模，右侧看该算法最值得观察的结构信息。", fill=TEXT_COLOR, font=text_font)

    left_plot = {"left": 70, "top": 158, "width": 440, "height": 330}
    right_plot = {"left": 590, "top": 158, "width": 620, "height": 330}
    draw_panel(draw, left_plot)
    draw_panel(draw, right_plot)
    draw_plot_title(draw, left_plot, "簇规模分布")

    render_cluster_size_bars(draw, left_plot, context["labels"])

    if algorithm_key == "kmeans":
        draw_plot_title(draw, right_plot, "中心移动趋势")
        render_kmeans_iteration_panel(draw, right_plot, context["center_history"])
    elif algorithm_key == "dbscan":
        draw_plot_title(draw, right_plot, "核心点 / 边界点 / 噪声点")
        render_dbscan_type_panel(draw, right_plot, context["core_mask"], context["border_mask"], context["noise_mask"])
    elif algorithm_key == "agg":
        draw_plot_title(draw, right_plot, "合并距离变化")
        render_merge_distance_panel(draw, right_plot, context["merge_history"])
    else:
        draw_plot_title(draw, right_plot, "成分权重与软分配")
        render_gmm_probability_panel(draw, right_plot, context["weights"], context["focus_probabilities"])
    return image


def render_kmeans_visual(context):
    image, draw, plot = create_canvas(
        context["visual_title"],
        "背景色表示最终由哪个中心负责，方框是初始中心，星形是最终中心，虚线显示中心移动轨迹。",
    )
    x_min, x_max, y_min, y_max = bounds_from_points(context["X"], margin=0.45)
    region_map = predict_grid_kmeans(context["model"], x_min, x_max, y_min, y_max)
    paste_region(image, plot, region_map)
    draw = ImageDraw.Draw(image)
    draw_axes(draw, plot, x_min, x_max, y_min, y_max, "特征 1", "特征 2")
    draw_cluster_points(draw, plot, context["X"], context["labels"], x_min, x_max, y_min, y_max)
    draw_center_paths(draw, plot, context["center_history"], x_min, x_max, y_min, y_max)
    draw_centers(draw, plot, context["initial_centers"], x_min, x_max, y_min, y_max, symbol="square")
    draw_centers(draw, plot, context["final_centers"], x_min, x_max, y_min, y_max, symbol="star")
    draw_focus_box(
        draw,
        plot["left"] + 16,
        plot["top"] + 16,
        320,
        118,
        "KMeans 观察点",
        [
            "当前簇数 = {0}".format(context["cluster_count"]),
            "迭代次数 = {0}".format(context["iterations"]),
            "簇内平方误差 = {0:.2f}".format(context["inertia"]),
        ],
    )
    draw_footer(draw, plot, "KMeans 本质上是在不断移动中心，让每个点更靠近自己的中心，所以它特别依赖“中心型”簇。")
    return image


def render_dbscan_visual(context):
    image, draw, plot = create_canvas(
        context["visual_title"],
        "圆点表示核心点，菱形表示边界点，灰色叉号表示噪声点，红圈展示当前 eps 邻域示例。",
    )
    x_min, x_max, y_min, y_max = bounds_from_points(context["X"], margin=0.42)
    draw_axes(draw, plot, x_min, x_max, y_min, y_max, "特征 1", "特征 2")
    draw_dbscan_points(draw, plot, context, x_min, x_max, y_min, y_max)
    fx, fy = context["focus_point"]
    center_px, center_py = project_xy(fx, fy, plot, x_min, x_max, y_min, y_max)
    radius_px = int(context["eps"] / max(x_max - x_min, 1e-9) * plot["width"])
    draw.ellipse(
        [center_px - radius_px, center_py - radius_px, center_px + radius_px, center_py + radius_px],
        outline=FOCUS_COLOR,
        width=2,
    )
    for neighbor_index in context["focus_neighbors"]:
        nx, ny = context["X"][neighbor_index]
        px, py = project_xy(nx, ny, plot, x_min, x_max, y_min, y_max)
        draw.ellipse([px - 10, py - 10, px + 10, py + 10], outline=FOCUS_COLOR, width=2)

    draw_focus_box(
        draw,
        plot["left"] + 16,
        plot["top"] + 16,
        340,
        138,
        "DBSCAN 邻域解释",
        [
            "当前高亮点位于红圈中心，用来演示 eps 邻域。",
            "邻域样本数 = {0}".format(len(context["focus_neighbors"])),
            "最少样本数 = {0}".format(context["min_samples"]),
            "噪声点数量 = {0}".format(context["noise_count"]),
        ],
    )
    draw_footer(draw, plot, "DBSCAN 不找中心，而是从“够密”的地方往外扩展，所以它更擅长发现任意形状簇。")
    return image


def render_agglomerative_visual(context):
    image = Image.new("RGB", (1320, 920), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = load_font(30, bold=True)
    subtitle_font = load_font(18)

    draw.text((60, 24), context["visual_title"], fill=TITLE_COLOR, font=title_font)
    draw.text((60, 64), "左侧看当前截断后的簇，右侧看树状图，帮助学生理解“先细后粗”的层次合并过程。", fill=TEXT_COLOR, font=subtitle_font)

    left_plot = {"left": 70, "top": 162, "width": 520, "height": 450}
    right_plot = {"left": 680, "top": 162, "width": 560, "height": 450}
    draw_panel(draw, left_plot)
    draw_panel(draw, right_plot)
    draw_plot_title(draw, left_plot, "当前截断后的聚类结果")
    draw_plot_title(draw, right_plot, "层次树状图")

    x_min, x_max, y_min, y_max = bounds_from_points(context["X"], margin=0.42)
    draw_axes(draw, left_plot, x_min, x_max, y_min, y_max, "特征 1", "特征 2")
    draw_cluster_points(draw, left_plot, context["X"], context["labels"], x_min, x_max, y_min, y_max)
    draw_dendrogram(draw, right_plot, context["merge_history"], len(context["X"]), context["cut_distance"])

    draw_focus_box(
        draw,
        70,
        708,
        1170,
        110,
        "层次聚类观察点",
        [
            "当前簇数 = {0}".format(context["cluster_count"]),
            "合并方式 = {0}".format(format_linkage_label(context["linkage"])),
            "切断高度 = {0}".format("--" if context["cut_distance"] is None else "{0:.2f}".format(context["cut_distance"])),
        ],
    )
    return image


def render_gmm_visual(context):
    image, draw, plot = create_canvas(
        context["visual_title"],
        "背景颜色深浅表示软分配概率，椭圆表示高斯分量范围，星形标出当前查询点的概率归属。",
    )
    x_min, x_max, y_min, y_max = bounds_from_points(context["X"], margin=0.5)
    probability_map = predict_grid_gmm(context["model"], x_min, x_max, y_min, y_max)
    paste_probability_region(image, plot, probability_map)
    draw = ImageDraw.Draw(image)
    draw_axes(draw, plot, x_min, x_max, y_min, y_max, "特征 1", "特征 2")
    draw_cluster_points(draw, plot, context["X"], context["labels"], x_min, x_max, y_min, y_max)
    draw_gmm_ellipses(draw, plot, context["means"], context["covariances"], context["covariance_type"], x_min, x_max, y_min, y_max)

    qx, qy = context["focus_point"]
    px, py = project_xy(qx, qy, plot, x_min, x_max, y_min, y_max)
    draw_star(draw, px, py, 12, fill=(255, 255, 255), outline=FOCUS_COLOR)
    draw_focus_box(
        draw,
        plot["left"] + 16,
        plot["top"] + 16,
        360,
        118,
        "GMM 概率解释",
        [
            "查询点最高归属概率 = {0:.1f}%".format(context["focus_max_probability"] * 100),
            "成分权重 = " + " / ".join("{0:.2f}".format(weight) for weight in context["weights"]),
            "软分配越平均，说明它越可能处于多个分量的重叠区域。",
        ],
    )
    draw_footer(draw, plot, "GMM 的关键不是“最近中心”，而是“这个点更像由哪个高斯分量生成的”。")
    return image


def create_canvas(title, subtitle):
    image = Image.new("RGB", (1320, 930), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    plot = {"left": 100, "top": 150, "width": 1120, "height": 560}
    draw.text((plot["left"], 24), title, fill=TITLE_COLOR, font=load_font(30, bold=True))
    draw.text((plot["left"], 64), subtitle, fill=TEXT_COLOR, font=load_font(18))
    draw_panel(draw, plot)
    return image, draw, plot


def render_cluster_size_bars(draw, plot, labels):
    labels = np.asarray(labels, dtype=int)
    present = [label for label in np.unique(labels) if label >= 0]
    items = [("簇 {0}".format(label), int(np.sum(labels == label)), cluster_color(label)) for label in present]
    if np.any(labels < 0):
        items.append(("噪声", int(np.sum(labels < 0)), NOISE_COLOR))
    if not items:
        items = [("未形成簇", 1, NOISE_COLOR)]

    max_value = max(count for _, count, _ in items)
    bar_left = plot["left"] + 96
    bar_top = plot["top"] + 40
    bar_width = plot["width"] - 130
    row_height = (plot["height"] - 60) / len(items)
    name_font = load_font(16)
    value_font = load_font(15)

    for index, (name, count, color) in enumerate(items):
        y_center = bar_top + row_height * (index + 0.5)
        draw.text((plot["left"] + 18, y_center - 11), name, fill=TEXT_COLOR, font=name_font)
        length = (count / max_value) * (bar_width - 50)
        draw.rounded_rectangle([bar_left, y_center - 10, bar_left + length, y_center + 10], radius=6, fill=color)
        value_text = str(count)
        draw.text((bar_left + bar_width - 26, y_center - 10), value_text, fill=MUTED_TEXT, font=value_font)


def render_kmeans_iteration_panel(draw, plot, center_history):
    if len(center_history) <= 1:
        draw_text_block(draw, plot["left"] + 20, plot["top"] + 40, plot["width"] - 40, 18, "当前中心几乎没有移动，说明已经收敛。")
        return
    movements = []
    for index in range(1, len(center_history)):
        delta = np.sqrt(np.sum((center_history[index] - center_history[index - 1]) ** 2, axis=1))
        movements.append(float(np.sum(delta)))

    render_series_panel(draw, plot, movements, "迭代轮次", "总移动距离")


def render_dbscan_type_panel(draw, plot, core_mask, border_mask, noise_mask):
    values = [
        ("核心点", int(np.sum(core_mask)), cluster_color(0)),
        ("边界点", int(np.sum(border_mask)), cluster_color(1)),
        ("噪声点", int(np.sum(noise_mask)), NOISE_COLOR),
    ]
    max_value = max(value for _, value, _ in values) if values else 1
    row_top = plot["top"] + 64
    row_height = 58
    for index, (name, value, color) in enumerate(values):
        y = row_top + index * row_height
        draw.text((plot["left"] + 24, y), name, fill=TEXT_COLOR, font=load_font(18))
        draw.rounded_rectangle(
            [plot["left"] + 140, y + 2, plot["left"] + 140 + (value / max_value) * (plot["width"] - 210), y + 24],
            radius=7,
            fill=color,
        )
        draw.text((plot["left"] + plot["width"] - 44, y), str(value), fill=MUTED_TEXT, font=load_font(18))


def render_merge_distance_panel(draw, plot, merge_history):
    distances = [item["distance"] for item in merge_history]
    if not distances:
        draw_text_block(draw, plot["left"] + 20, plot["top"] + 40, plot["width"] - 40, 18, "样本太少，暂时没有可观察的合并序列。")
        return
    render_series_panel(draw, plot, distances, "合并步数", "合并距离")


def render_gmm_probability_panel(draw, plot, weights, focus_probabilities):
    left = plot["left"] + 24
    top = plot["top"] + 30
    inner_width = plot["width"] - 48
    section_width = inner_width / 2 - 18
    second_left = left + section_width + 36
    bar_offset = 92
    bar_width = 120
    value_offset = 226

    draw.text((left, top), "成分权重", fill=TITLE_COLOR, font=load_font(20, bold=True))
    for index, weight in enumerate(weights):
        color = cluster_color(index)
        y = top + 42 + index * 32
        draw.text((left, y - 3), "成分 {0}".format(index), fill=TEXT_COLOR, font=load_font(16))
        draw.rounded_rectangle([left + bar_offset, y, left + bar_offset + weight * bar_width, y + 16], radius=6, fill=color)
        draw.text((left + value_offset, y - 4), "{0:.2f}".format(weight), fill=MUTED_TEXT, font=load_font(16))

    draw.text((second_left, top), "查询点软分配", fill=TITLE_COLOR, font=load_font(20, bold=True))
    for index, probability in enumerate(focus_probabilities):
        color = cluster_color(index)
        y = top + 42 + index * 32
        draw.text((second_left, y - 3), "簇 {0}".format(index), fill=TEXT_COLOR, font=load_font(16))
        draw.rounded_rectangle(
            [second_left + bar_offset - 18, y, second_left + bar_offset - 18 + probability * bar_width, y + 16],
            radius=6,
            fill=color,
        )
        draw.text((second_left + value_offset - 8, y - 4), "{0:.2f}".format(probability), fill=MUTED_TEXT, font=load_font(16))


def render_series_panel(draw, plot, values, x_label, y_label):
    values = np.asarray(values, dtype=float)
    x_min, x_max = 1, max(len(values), 2)
    y_min = min(values.min(), 0.0)
    y_max = values.max() + max(0.15 * abs(values.max()), 0.2)
    inner = {"left": plot["left"] + 64, "top": plot["top"] + 34, "width": plot["width"] - 90, "height": plot["height"] - 74}
    draw_axes(draw, inner, x_min, x_max, y_min, y_max, x_label, y_label, tick_left=plot["left"] + 6, y_label_x=plot["left"] - 4)
    points = [project_xy(index + 1, value, inner, x_min, x_max, y_min, y_max) for index, value in enumerate(values)]
    draw.line(points, fill=cluster_color(0), width=3)
    for px, py in points:
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=cluster_color(1), outline=(255, 255, 255), width=1)


def draw_cluster_points(draw, plot, X, labels, x_min, x_max, y_min, y_max):
    for point, label in zip(X, labels):
        px, py = project_xy(point[0], point[1], plot, x_min, x_max, y_min, y_max)
        color = cluster_color(label)
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=color, outline=(255, 255, 255), width=1)


def draw_dbscan_points(draw, plot, context, x_min, x_max, y_min, y_max):
    for index, point in enumerate(context["X"]):
        px, py = project_xy(point[0], point[1], plot, x_min, x_max, y_min, y_max)
        label = context["labels"][index]
        color = cluster_color(label)
        if context["noise_mask"][index]:
            draw.line([(px - 6, py - 6), (px + 6, py + 6)], fill=NOISE_COLOR, width=2)
            draw.line([(px - 6, py + 6), (px + 6, py - 6)], fill=NOISE_COLOR, width=2)
        elif context["core_mask"][index]:
            draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=color, outline=(31, 45, 61), width=1)
        else:
            draw_diamond(draw, px, py, 7, color, (31, 45, 61))


def draw_center_paths(draw, plot, center_history, x_min, x_max, y_min, y_max):
    if len(center_history) < 2:
        return
    for cluster_id in range(center_history[0].shape[0]):
        points = [
            project_xy(center_set[cluster_id, 0], center_set[cluster_id, 1], plot, x_min, x_max, y_min, y_max)
            for center_set in center_history
        ]
        draw.line(points, fill=cluster_color(cluster_id), width=2)


def draw_centers(draw, plot, centers, x_min, x_max, y_min, y_max, symbol="star"):
    for cluster_id, center in enumerate(centers):
        px, py = project_xy(center[0], center[1], plot, x_min, x_max, y_min, y_max)
        color = cluster_color(cluster_id)
        if symbol == "square":
            draw.rectangle([px - 7, py - 7, px + 7, py + 7], outline=color, width=3)
        else:
            draw_star(draw, px, py, 11, fill=(255, 255, 255), outline=color)


def draw_dendrogram(draw, plot, merge_history, n_samples, cut_distance):
    if not merge_history:
        draw_text_block(draw, plot["left"] + 18, plot["top"] + 18, plot["width"] - 36, 18, "样本太少，无法形成可观察的层次树。")
        return

    x_positions = {}
    y_positions = {}
    for index in range(n_samples):
        x_positions[index] = plot["left"] + 18 + index * (plot["width"] - 36) / max(n_samples - 1, 1)
        y_positions[index] = plot["top"] + plot["height"] - 20

    max_distance = max(item["distance"] for item in merge_history) + 1e-9
    for merge_index, merge in enumerate(merge_history):
        left = merge["left"]
        right = merge["right"]
        cluster_id = n_samples + merge_index
        merge_y = plot["top"] + plot["height"] - 20 - (merge["distance"] / max_distance) * (plot["height"] - 46)
        left_x = x_positions[left]
        right_x = x_positions[right]
        left_y = y_positions[left]
        right_y = y_positions[right]
        draw.line([(left_x, left_y), (left_x, merge_y)], fill=(108, 122, 137), width=2)
        draw.line([(right_x, right_y), (right_x, merge_y)], fill=(108, 122, 137), width=2)
        draw.line([(left_x, merge_y), (right_x, merge_y)], fill=(108, 122, 137), width=2)
        x_positions[cluster_id] = (left_x + right_x) / 2
        y_positions[cluster_id] = merge_y

    if cut_distance is not None:
        y_cut = plot["top"] + plot["height"] - 20 - (cut_distance / max_distance) * (plot["height"] - 46)
        for offset in range(0, plot["width"], 12):
            draw.line(
                [(plot["left"] + offset, y_cut), (plot["left"] + min(offset + 6, plot["width"]), y_cut)],
                fill=FOCUS_COLOR,
                width=2,
            )


def draw_gmm_ellipses(draw, plot, means, covariances, covariance_type, x_min, x_max, y_min, y_max):
    for component_id, mean in enumerate(means):
        if covariance_type == "diag":
            cov = np.diag(covariances[component_id])
        else:
            cov = covariances[component_id]
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        angle = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
        color = cluster_color(component_id)
        for scale in [1.2, 2.0]:
            points = []
            for theta in np.linspace(0.0, 2.0 * math.pi, 80):
                unit = np.array([math.cos(theta), math.sin(theta)])
                radius = np.sqrt(np.maximum(eigenvalues, 1e-6)) * scale
                point = mean + eigenvectors.dot(radius * unit)
                points.append(project_xy(point[0], point[1], plot, x_min, x_max, y_min, y_max))
            draw.line(points + [points[0]], fill=color, width=2 if scale > 1.5 else 1)


def paste_region(image, plot, region_map):
    palette = np.array([lighten_color(cluster_color(index), 0.76) for index in range(region_map.max() + 1)], dtype=np.uint8)
    region = palette[region_map]
    region_image = Image.fromarray(region).resize((plot["width"], plot["height"]), resample=Image.NEAREST)
    image.paste(region_image, (plot["left"], plot["top"]))


def paste_probability_region(image, plot, probability_map):
    color_table = np.array([cluster_color(index) for index in range(probability_map.shape[2])], dtype=float)
    weighted = probability_map.dot(color_table)
    certainty = probability_map.max(axis=2, keepdims=True)
    whiten = 0.55 - certainty * 0.30
    colored = weighted * (1.0 - whiten) + 255.0 * whiten
    region_image = Image.fromarray(colored.astype(np.uint8)).resize((plot["width"], plot["height"]), resample=Image.BILINEAR)
    image.paste(region_image, (plot["left"], plot["top"]))


def predict_grid_kmeans(model, x_min, x_max, y_min, y_max, width=240, height=180):
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, width), np.linspace(y_max, y_min, height))
    grid = np.c_[xx.ravel(), yy.ravel()]
    labels = model.predict(grid).reshape(height, width)
    return labels


def predict_grid_gmm(model, x_min, x_max, y_min, y_max, width=240, height=180):
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, width), np.linspace(y_max, y_min, height))
    grid = np.c_[xx.ravel(), yy.ravel()]
    proba = model.predict_proba(grid).reshape(height, width, model.n_components)
    return proba


def draw_axes(draw, plot, x_min, x_max, y_min, y_max, x_label, y_label, show_y_axis=True, tick_left=None, y_label_x=None):
    font = load_font(14)
    tick_left = max(12, plot["left"] - 48) if tick_left is None else tick_left
    y_label_x = tick_left if y_label_x is None else y_label_x

    for value in np.linspace(x_min, x_max, 5):
        px = project_xy(value, y_min, plot, x_min, x_max, y_min, y_max)[0]
        draw.line([(px, plot["top"]), (px, plot["top"] + plot["height"])], fill=GRID_COLOR, width=1)
        text = "{0:.1f}".format(value)
        text_box = centered_text_box(draw, text, font)
        draw.text((px - text_box[2] / 2, plot["top"] + plot["height"] + 10), text, fill=MUTED_TEXT, font=font)

    for value in np.linspace(y_min, y_max, 5):
        py = project_xy(x_min, value, plot, x_min, x_max, y_min, y_max)[1]
        draw.line([(plot["left"], py), (plot["left"] + plot["width"], py)], fill=GRID_COLOR, width=1)
        if show_y_axis:
            text = "{0:.1f}".format(value)
            text_box = centered_text_box(draw, text, font)
            draw.text((tick_left, py - text_box[3] / 2 - 2), text, fill=MUTED_TEXT, font=font)

    label_font = load_font(18)
    x_box = centered_text_box(draw, x_label, label_font)
    draw.text((plot["left"] + plot["width"] / 2 - x_box[2] / 2, plot["top"] + plot["height"] + 34), x_label, fill=TEXT_COLOR, font=label_font)
    if show_y_axis:
        draw.text((y_label_x, plot["top"] - 34), y_label, fill=TEXT_COLOR, font=label_font)


def draw_focus_box(draw, x, y, width, height, title, lines):
    text_font = load_font(16)
    wrapped_groups = [wrap_text(draw, line, text_font, width - 32) for line in lines]
    text_line_count = sum(max(len(group), 1) for group in wrapped_groups)
    box_height = max(height, 50 + text_line_count * 18 + 18 + max(len(lines) - 1, 0) * 4)
    draw.rounded_rectangle([x, y, x + width, y + box_height], radius=16, fill=(255, 255, 255), outline=OUTLINE_COLOR)
    draw.text((x + 16, y + 12), title, fill=TITLE_COLOR, font=load_font(20, bold=True))
    current_y = y + 46
    for wrapped in wrapped_groups:
        for part in wrapped:
            draw.text((x + 16, current_y), part, fill=TEXT_COLOR, font=text_font)
            current_y += 18
        current_y += 4


def draw_footer(draw, plot, message):
    draw_focus_box(draw, plot["left"], plot["top"] + plot["height"] + 88, plot["width"], 82, "图像解读", [message])


def draw_panel(draw, plot):
    draw.rectangle(
        [plot["left"], plot["top"], plot["left"] + plot["width"], plot["top"] + plot["height"]],
        outline=OUTLINE_COLOR,
        width=2,
    )


def draw_plot_title(draw, plot, title):
    font = load_font(21, bold=True)
    box = centered_text_box(draw, title, font)
    draw.text((plot["left"] + plot["width"] / 2 - box[2] / 2, plot["top"] - 40), title, fill=TITLE_COLOR, font=font)


def bounds_from_points(X, margin=0.4):
    X = np.asarray(X, dtype=float)
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    return x_min - margin, x_max + margin, y_min - margin, y_max + margin


def project_xy(x_value, y_value, plot, x_min, x_max, y_min, y_max):
    px = plot["left"] + int((x_value - x_min) / max(x_max - x_min, 1e-9) * plot["width"])
    py = plot["top"] + int((y_max - y_value) / max(y_max - y_min, 1e-9) * plot["height"])
    return px, py


def cluster_color(label):
    if label < 0:
        return NOISE_COLOR
    return hex_to_rgb(CLUSTER_HEX[label % len(CLUSTER_HEX)])


def format_linkage_label(value):
    mapping = {
        "single": "最近距离",
        "complete": "最远距离",
        "average": "平均距离",
        "ward": "方差最小化",
    }
    return mapping.get(value, value)


def lighten_color(color, ratio):
    return tuple(int(channel * (1.0 - ratio) + 255 * ratio) for channel in color)


def draw_text_block(draw, x, y, width, line_height, text):
    font = load_font(16)
    for row_index, line in enumerate(wrap_text(draw, text, font, width)):
        draw.text((x, y + row_index * line_height), line, fill=TEXT_COLOR, font=font)


def draw_diamond(draw, x, y, size, fill_color, outline_color):
    points = [(x, y - size), (x + size, y), (x, y + size), (x - size, y)]
    draw.polygon(points, fill=fill_color, outline=outline_color)


def draw_star(draw, x, y, radius, fill, outline):
    points = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        r = radius if index % 2 == 0 else radius * 0.45
        points.append((x + r * math.cos(angle), y + r * math.sin(angle)))
    draw.polygon(points, fill=fill, outline=outline)


def centered_text_box(draw, text, font):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return left, top, right - left, bottom - top


def wrap_text(draw, text, font, max_width):
    current = ""
    lines = []
    for char in text:
        trial = current + char
        width = centered_text_box(draw, trial, font)[2]
        if current and width > max_width:
            lines.append(current)
            current = char
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


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
