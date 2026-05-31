import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TRAIN_COLOR = (46, 134, 222)
TEST_COLOR = (230, 126, 34)
FIT_COLOR = (39, 174, 96)
TRUE_COLOR = (92, 112, 129)
FOCUS_COLOR = (231, 76, 60)
BAND_COLOR = (203, 213, 255)
TITLE_COLOR = (15, 91, 158)
TEXT_COLOR = (46, 62, 80)
MUTED_TEXT = (92, 112, 129)
GRID_COLOR = (230, 236, 242)
PANEL_OUTLINE = (214, 226, 238)


FONT_PATHS = [
    "assets/LXGWWenKai-Regular.ttf",
]


def render_main_visual(algorithm_key, context):
    if algorithm_key == "linear":
        return render_linear_visual(context)
    if algorithm_key == "poly":
        return render_polynomial_visual(context)
    if algorithm_key in ["ridge", "lasso"]:
        return render_regularized_visual(algorithm_key, context)
    if algorithm_key == "svr":
        return render_svr_visual(context)
    if algorithm_key == "tree":
        return render_tree_visual(context)
    return render_forest_visual(context)


def get_axis_labels(context):
    return context.get("x_axis_label", "输入特征 x"), context.get("y_axis_label", "目标值 y")


def render_diagnostic_visual(algorithm_key, context):
    width = 1280
    height = 580
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = load_font(28, bold=True)
    text_font = load_font(17)

    draw.text((54, 22), "结果诊断", fill=TITLE_COLOR, font=title_font)
    draw.text((54, 60), "左侧看测试残差，右侧看预测值与真实值之间的关系。", fill=TEXT_COLOR, font=text_font)

    left_plot = {"left": 70, "top": 158, "width": 540, "height": 330}
    right_plot = {"left": 680, "top": 158, "width": 540, "height": 330}
    draw_plot_frame(draw, left_plot)
    draw_plot_frame(draw, right_plot)
    draw_plot_title(draw, left_plot, "测试残差图")

    residual = context["y_test_pred"] - context["y_test"]
    x_test_sorted = context["x_test_sorted"]
    residual_sorted = residual[context["x_test_sort_index"]]
    residual_min = min(residual_sorted.min(), -0.1) - 0.2
    residual_max = max(residual_sorted.max(), 0.1) + 0.2
    draw_axes(
        draw,
        left_plot,
        x_test_sorted.min(),
        x_test_sorted.max(),
        residual_min,
        residual_max,
        context.get("x_axis_label", "输入特征 x"),
        context.get("residual_axis_label", "残差"),
    )
    zero_y = project_point(0.0, left_plot, residual_min, residual_max, axis="y")
    draw.line([(left_plot["left"], zero_y), (left_plot["left"] + left_plot["width"], zero_y)], fill=(155, 168, 182), width=2)
    for x_value, res_value in zip(x_test_sorted, residual_sorted):
        px, py = project_xy(x_value, res_value, left_plot, x_test_sorted.min(), x_test_sorted.max(), residual_min, residual_max)
        draw.line([(px, zero_y), (px, py)], fill=(246, 177, 122), width=1)
        draw_diamond(draw, px, py, 6, TEST_COLOR, (120, 72, 22))

    if algorithm_key == "rf":
        draw_plot_title(draw, right_plot, "单树与森林对比")
        render_prediction_trace_panel(
            draw,
            right_plot,
            context["x_test_sorted"],
            context["y_test_sorted"],
            context["single_tree_test_sorted"],
            context["y_test_pred_sorted"],
            label_a="真实值",
            label_b="单树预测",
            label_c="森林预测",
            x_axis_label=context.get("x_axis_label", "输入特征 x"),
            y_axis_label=context.get("y_axis_label", "目标值 y"),
        )
    elif algorithm_key in ["ridge", "lasso"]:
        draw_plot_title(draw, right_plot, "真实值 vs 预测值")
        render_scatter_compare_panel(draw, right_plot, context["y_test"], context["y_test_pred"])
    else:
        draw_plot_title(draw, right_plot, "按 x 排序的预测轨迹")
        render_prediction_trace_panel(
            draw,
            right_plot,
            context["x_test_sorted"],
            context["y_test_sorted"],
            None,
            context["y_test_pred_sorted"],
            label_a="真实值",
            label_b="预测值",
            x_axis_label=context.get("x_axis_label", "输入特征 x"),
            y_axis_label=context.get("y_axis_label", "目标值 y"),
        )

    return image


def render_linear_visual(context):
    image, draw, plot = create_canvas(
        context["visual_title"],
        "散点表示样本，绿色直线是模型拟合结果，灰色虚线是真实趋势，红色线段表示残差。",
    )
    x_min, x_max, y_min, y_max = fit_bounds(context, margin_x=0.35, margin_y=0.45)
    x_axis_label, y_axis_label = get_axis_labels(context)
    draw_axes(draw, plot, x_min, x_max, y_min, y_max, x_axis_label, y_axis_label)
    draw_reference_curve(draw, plot, context["x_grid"], context["y_grid_true"], x_min, x_max, y_min, y_max)
    draw_fit_curve(draw, plot, context["x_grid"], context["y_grid_pred"], x_min, x_max, y_min, y_max)
    draw_regression_samples(draw, plot, context, x_min, x_max, y_min, y_max)
    draw_residual_guides(draw, plot, context, x_min, x_max, y_min, y_max)
    draw_focus_box(
        draw,
        plot["left"] + 18,
        plot["top"] + 18,
        300,
        118,
        "线性拟合观察点",
        [
            "关注样本 {0} = {1:.2f}".format(context.get("focus_x_label", "输入特征"), context["focus_x"]),
            "{0} = {1:.2f}，{2} = {3:.2f}".format(
                context.get("true_value_label", "真实值"),
                context["focus_true"],
                context.get("pred_value_label", "预测值"),
                context["focus_pred"],
            ),
            "{0} = {1:+.2f}".format(context.get("residual_axis_label", "残差"), context["focus_residual"]),
        ],
    )
    draw_footer(draw, plot, "线性回归擅长抓住整体方向，但面对明显弯曲关系时，直线通常会显得过于简单。")
    return image


def render_polynomial_visual(context):
    image, draw, plot = create_canvas(
        context["visual_title"],
        "绿色曲线是当前阶数下的拟合结果，灰色虚线是真实趋势，用来观察欠拟合和过拟合。",
    )
    x_min, x_max, y_min, y_max = fit_bounds(context, margin_x=0.35, margin_y=0.55)
    x_axis_label, y_axis_label = get_axis_labels(context)
    draw_axes(draw, plot, x_min, x_max, y_min, y_max, x_axis_label, y_axis_label)
    draw_reference_curve(draw, plot, context["x_grid"], context["y_grid_true"], x_min, x_max, y_min, y_max)
    draw_fit_curve(draw, plot, context["x_grid"], context["y_grid_pred"], x_min, x_max, y_min, y_max)
    draw_regression_samples(draw, plot, context, x_min, x_max, y_min, y_max)
    draw_residual_guides(draw, plot, context, x_min, x_max, y_min, y_max, limit=5)
    draw_focus_box(
        draw,
        plot["left"] + 18,
        plot["top"] + 18,
        320,
        118,
        "多项式曲线提醒",
        [
            "当前阶数 = {0}".format(context["degree"]),
            "测试 R² = {0:.3f}".format(context["metrics"]["test_r2"]),
            context["shape_note"],
        ],
    )
    draw_footer(draw, plot, "多项式阶数并不是越高越好，关键是看曲线是否抓住趋势，而不是只追着噪声摆动。")
    return image


def render_regularized_visual(algorithm_key, context):
    image = Image.new("RGB", (1320, 930), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = load_font(30, bold=True)
    text_font = load_font(18)

    subtitle = "左侧看拟合曲线，右侧看系数收缩；Ridge 强调整体收缩，Lasso 还会把一些系数压到接近 0。"
    draw.text((70, 24), context["visual_title"], fill=TITLE_COLOR, font=title_font)
    draw.text((70, 64), subtitle, fill=TEXT_COLOR, font=text_font)

    plot = {"left": 80, "top": 150, "width": 820, "height": 560}
    coeff_area = {"left": 950, "top": 176, "width": 290, "height": 430}
    draw_plot_frame(draw, plot)
    draw_round_panel(draw, coeff_area)

    x_min, x_max, y_min, y_max = fit_bounds(context, margin_x=0.35, margin_y=0.55)
    x_axis_label, y_axis_label = get_axis_labels(context)
    draw_axes(draw, plot, x_min, x_max, y_min, y_max, x_axis_label, y_axis_label)
    draw_reference_curve(draw, plot, context["x_grid"], context["y_grid_true"], x_min, x_max, y_min, y_max)
    draw_fit_curve(draw, plot, context["x_grid"], context["y_grid_pred"], x_min, x_max, y_min, y_max)
    draw_regression_samples(draw, plot, context, x_min, x_max, y_min, y_max)
    draw_residual_guides(draw, plot, context, x_min, x_max, y_min, y_max, limit=4)

    draw.text((coeff_area["left"] + 18, coeff_area["top"] + 12), "系数收缩观察", fill=TITLE_COLOR, font=load_font(22, bold=True))
    draw_text_block(
        draw,
        coeff_area["left"] + 18,
        coeff_area["top"] + 48,
        coeff_area["width"] - 36,
        18,
        "当前 alpha = {0:.2f}".format(context["alpha"]),
    )
    draw_coefficient_bars(
        image,
        coeff_area,
        context["coef_pairs"],
        highlight_sparse=algorithm_key == "lasso",
    )
    draw_footer(draw, plot, "正则化不仅影响曲线形状，也会直接改变每个特征分到的权重。")
    return image


def render_svr_visual(context):
    image, draw, plot = create_canvas(
        context["visual_title"],
        "绿色曲线是 SVR 拟合结果，紫色带状区域是 epsilon 容忍带，被圈出的点是支持向量。",
    )
    x_min, x_max, y_min, y_max = fit_bounds(context, margin_x=0.35, margin_y=0.60)
    x_axis_label, y_axis_label = get_axis_labels(context)
    draw_axes(draw, plot, x_min, x_max, y_min, y_max, x_axis_label, y_axis_label)
    draw_reference_curve(draw, plot, context["x_grid"], context["y_grid_true"], x_min, x_max, y_min, y_max)
    draw_band(draw, plot, context["x_grid"], context["y_grid_pred"] - context["epsilon"], context["y_grid_pred"] + context["epsilon"], x_min, x_max, y_min, y_max)
    draw_fit_curve(draw, plot, context["x_grid"], context["y_grid_pred"], x_min, x_max, y_min, y_max)
    draw_regression_samples(draw, plot, context, x_min, x_max, y_min, y_max)

    for x_value, y_value in zip(context["support_x"], context["support_y"]):
        px, py = project_xy(x_value, y_value, plot, x_min, x_max, y_min, y_max)
        draw.ellipse([px - 10, py - 10, px + 10, py + 10], outline=FOCUS_COLOR, width=3)

    draw_focus_box(
        draw,
        plot["left"] + 18,
        plot["top"] + 18,
        330,
        118,
        "SVR 观察重点",
        [
            "支持向量数量 = {0}".format(context["support_vector_count"]),
            "epsilon = {0:.2f}".format(context["epsilon"]),
            "带内样本占比约 {0:.1f}%".format(context["tube_ratio"] * 100),
        ],
    )
    draw_footer(draw, plot, "SVR 不是硬追每一个点，而是先给出一条允许小误差的带，再重点照顾带外的关键样本。")
    return image


def render_tree_visual(context):
    image, draw, plot = create_canvas(
        context["visual_title"],
        "绿色阶梯线是树模型预测，灰色虚线是真实趋势。台阶越细，通常说明树越复杂。",
    )
    x_min, x_max, y_min, y_max = fit_bounds(context, margin_x=0.35, margin_y=0.55)
    x_axis_label, y_axis_label = get_axis_labels(context)
    draw_axes(draw, plot, x_min, x_max, y_min, y_max, x_axis_label, y_axis_label)
    draw_reference_curve(draw, plot, context["x_grid"], context["y_grid_true"], x_min, x_max, y_min, y_max)
    draw_step_fit_curve(draw, plot, context["x_grid"], context["y_grid_pred"], x_min, x_max, y_min, y_max)
    draw_regression_samples(draw, plot, context, x_min, x_max, y_min, y_max)
    draw_focus_box(
        draw,
        plot["left"] + 18,
        plot["top"] + 18,
        320,
        118,
        "树模型观察点",
        [
            "叶子数量 = {0}".format(context["leaf_count"]),
            "测试 R² = {0:.3f}".format(context["metrics"]["test_r2"]),
            "阶梯数量约 = {0}".format(context["plateau_count"]),
        ],
    )
    draw_footer(draw, plot, "决策树回归的本质是“分段逼近”，所以预测曲线看起来像台阶而不是平滑曲线。")
    return image


def render_forest_visual(context):
    width = 1320
    height = 900
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = load_font(30, bold=True)
    text_font = load_font(18)

    draw.text((60, 24), context["visual_title"], fill=TITLE_COLOR, font=title_font)
    draw.text((60, 64), "左图是单棵树，右图是随机森林；它们使用同一批样本，但森林会通过平均变得更稳定。", fill=TEXT_COLOR, font=text_font)

    left_plot = {"left": 70, "top": 168, "width": 540, "height": 470}
    right_plot = {"left": 710, "top": 168, "width": 540, "height": 470}
    draw_plot_frame(draw, left_plot)
    draw_plot_frame(draw, right_plot)
    draw_plot_title(draw, left_plot, "单棵树回归")
    draw_plot_title(draw, right_plot, "随机森林平均结果")

    x_min, x_max, y_min, y_max = fit_bounds(context, margin_x=0.35, margin_y=0.55)
    x_axis_label, y_axis_label = get_axis_labels(context)
    draw_axes(draw, left_plot, x_min, x_max, y_min, y_max, x_axis_label, y_axis_label, tick_left=18, y_label_x=8)
    draw_axes(draw, right_plot, x_min, x_max, y_min, y_max, x_axis_label, y_axis_label, show_y_axis=False)

    draw_reference_curve(draw, left_plot, context["x_grid"], context["y_grid_true"], x_min, x_max, y_min, y_max)
    draw_reference_curve(draw, right_plot, context["x_grid"], context["y_grid_true"], x_min, x_max, y_min, y_max)
    draw_step_fit_curve(draw, left_plot, context["x_grid"], context["single_tree_grid_pred"], x_min, x_max, y_min, y_max, color=(51, 131, 213))
    draw_fit_curve(draw, right_plot, context["x_grid"], context["y_grid_pred"], x_min, x_max, y_min, y_max, color=FIT_COLOR)
    draw_regression_samples(draw, left_plot, context, x_min, x_max, y_min, y_max)
    draw_regression_samples(draw, right_plot, context, x_min, x_max, y_min, y_max)

    draw_focus_box(
        draw,
        80,
        708,
        1170,
        108,
        "森林观察提示",
        [
            "单棵树测试 R² = {0:.3f}".format(context["single_tree_metrics"]["test_r2"]),
            "随机森林测试 R² = {0:.3f}".format(context["metrics"]["test_r2"]),
            "森林相对单树的 R² 提升 = {0:+.2f}".format(context["forest_gain"]),
        ],
    )
    return image


def create_canvas(title, subtitle):
    image = Image.new("RGB", (1320, 930), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = load_font(30, bold=True)
    subtitle_font = load_font(18)
    plot = {"left": 96, "top": 150, "width": 1128, "height": 560}
    draw.text((plot["left"], 24), title, fill=TITLE_COLOR, font=title_font)
    draw.text((plot["left"], 64), subtitle, fill=TEXT_COLOR, font=subtitle_font)
    draw_plot_frame(draw, plot)
    return image, draw, plot


def fit_bounds(context, margin_x=0.3, margin_y=0.4):
    x_all = np.concatenate([context["x_train"], context["x_test"], context["x_grid"]])
    y_all = np.concatenate(
        [
            context["y_train"],
            context["y_test"],
            context["y_grid_pred"],
            context["y_grid_true"],
        ]
    )
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    return x_min - margin_x, x_max + margin_x, y_min - margin_y, y_max + margin_y


def draw_plot_frame(draw, plot):
    draw.rectangle(
        [plot["left"], plot["top"], plot["left"] + plot["width"], plot["top"] + plot["height"]],
        outline=PANEL_OUTLINE,
        width=2,
    )


def draw_round_panel(draw, area):
    draw.rounded_rectangle(
        [area["left"], area["top"], area["left"] + area["width"], area["top"] + area["height"]],
        radius=18,
        fill=(250, 252, 255),
        outline=PANEL_OUTLINE,
        width=1,
    )


def draw_plot_title(draw, plot, title):
    font = load_font(21, bold=True)
    box = centered_text_box(draw, title, font)
    draw.text((plot["left"] + plot["width"] / 2 - box[2] / 2, plot["top"] - 42), title, fill=TITLE_COLOR, font=font)


def draw_axes(draw, plot, x_min, x_max, y_min, y_max, x_label, y_label, show_y_axis=True, tick_left=None, y_label_x=None):
    font = load_font(14)
    tick_left = max(12, plot["left"] - 48) if tick_left is None else tick_left
    y_label_x = tick_left if y_label_x is None else y_label_x

    for value in np.linspace(x_min, x_max, 5):
        px = project_point(value, plot, x_min, x_max, axis="x")
        draw.line([(px, plot["top"]), (px, plot["top"] + plot["height"])], fill=GRID_COLOR, width=1)
        tick_text = "{0:.1f}".format(value)
        tick_box = centered_text_box(draw, tick_text, font)
        draw.text((px - tick_box[2] / 2, plot["top"] + plot["height"] + 10), tick_text, fill=MUTED_TEXT, font=font)

    for value in np.linspace(y_min, y_max, 5):
        py = project_point(value, plot, y_min, y_max, axis="y")
        draw.line([(plot["left"], py), (plot["left"] + plot["width"], py)], fill=GRID_COLOR, width=1)
        if show_y_axis:
            tick_text = "{0:.1f}".format(value)
            tick_box = centered_text_box(draw, tick_text, font)
            draw.text((tick_left, py - tick_box[3] / 2 - 2), tick_text, fill=MUTED_TEXT, font=font)

    label_font = load_font(18)
    x_box = centered_text_box(draw, x_label, label_font)
    draw.text((plot["left"] + plot["width"] / 2 - x_box[2] / 2, plot["top"] + plot["height"] + 36), x_label, fill=TEXT_COLOR, font=label_font)
    if show_y_axis:
        draw.text((y_label_x, plot["top"] - 34), y_label, fill=TEXT_COLOR, font=label_font)


def draw_reference_curve(draw, plot, x_values, y_values, x_min, x_max, y_min, y_max):
    points = [project_xy(x, y, plot, x_min, x_max, y_min, y_max) for x, y in zip(x_values, y_values)]
    for start in range(0, len(points) - 1, 8):
        chunk = points[start : start + 4]
        if len(chunk) >= 2:
            draw.line(chunk, fill=TRUE_COLOR, width=2)


def draw_fit_curve(draw, plot, x_values, y_values, x_min, x_max, y_min, y_max, color=FIT_COLOR):
    points = [project_xy(x, y, plot, x_min, x_max, y_min, y_max) for x, y in zip(x_values, y_values)]
    draw.line(points, fill=color, width=4)


def draw_step_fit_curve(draw, plot, x_values, y_values, x_min, x_max, y_min, y_max, color=FIT_COLOR):
    points = [project_xy(x, y, plot, x_min, x_max, y_min, y_max) for x, y in zip(x_values, y_values)]
    step_points = []
    for index in range(len(points) - 1):
        x0, y0 = points[index]
        x1, y1 = points[index + 1]
        step_points.extend([(x0, y0), (x1, y0)])
    if points:
        step_points.append(points[-1])
    draw.line(step_points, fill=color, width=4)


def draw_band(draw, plot, x_values, lower_values, upper_values, x_min, x_max, y_min, y_max):
    upper = [project_xy(x, y, plot, x_min, x_max, y_min, y_max) for x, y in zip(x_values, upper_values)]
    lower = [project_xy(x, y, plot, x_min, x_max, y_min, y_max) for x, y in zip(x_values[::-1], lower_values[::-1])]
    polygon = upper + lower
    draw.polygon(polygon, fill=(BAND_COLOR[0], BAND_COLOR[1], BAND_COLOR[2]))


def draw_regression_samples(draw, plot, context, x_min, x_max, y_min, y_max):
    for x_value, y_value in zip(context["x_train"], context["y_train"]):
        px, py = project_xy(x_value, y_value, plot, x_min, x_max, y_min, y_max)
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=TRAIN_COLOR, outline=(255, 255, 255), width=1)

    for x_value, y_value in zip(context["x_test"], context["y_test"]):
        px, py = project_xy(x_value, y_value, plot, x_min, x_max, y_min, y_max)
        draw_diamond(draw, px, py, 7, TEST_COLOR, (97, 56, 17))


def draw_residual_guides(draw, plot, context, x_min, x_max, y_min, y_max, limit=6):
    indices = context["residual_demo_indices"][:limit]
    for index in indices:
        x_value = context["x_test"][index]
        true_value = context["y_test"][index]
        pred_value = context["y_test_pred"][index]
        px_true, py_true = project_xy(x_value, true_value, plot, x_min, x_max, y_min, y_max)
        _, py_pred = project_xy(x_value, pred_value, plot, x_min, x_max, y_min, y_max)
        draw.line([(px_true, py_true), (px_true, py_pred)], fill=FOCUS_COLOR, width=2)

    px_focus, py_focus = project_xy(context["focus_x"], context["focus_true"], plot, x_min, x_max, y_min, y_max)
    _, py_focus_pred = project_xy(context["focus_x"], context["focus_pred"], plot, x_min, x_max, y_min, y_max)
    draw.line([(px_focus, py_focus), (px_focus, py_focus_pred)], fill=FOCUS_COLOR, width=3)
    draw.ellipse([px_focus - 9, py_focus - 9, px_focus + 9, py_focus + 9], outline=FOCUS_COLOR, width=3)


def render_prediction_trace_panel(
    draw,
    plot,
    x_sorted,
    y_true_sorted,
    aux_sorted,
    y_pred_sorted,
    label_a,
    label_b,
    label_c=None,
    x_axis_label="输入特征 x",
    y_axis_label="目标值 y",
):
    y_all = [y_true_sorted, y_pred_sorted]
    if aux_sorted is not None:
        y_all.append(aux_sorted)
    y_min = min(np.min(arr) for arr in y_all) - 0.35
    y_max = max(np.max(arr) for arr in y_all) + 0.35
    draw_axes(draw, plot, x_sorted.min(), x_sorted.max(), y_min, y_max, x_axis_label, y_axis_label, tick_left=plot["left"] - 42, y_label_x=plot["left"] - 52)

    true_points = [project_xy(x, y, plot, x_sorted.min(), x_sorted.max(), y_min, y_max) for x, y in zip(x_sorted, y_true_sorted)]
    pred_points = [project_xy(x, y, plot, x_sorted.min(), x_sorted.max(), y_min, y_max) for x, y in zip(x_sorted, y_pred_sorted)]
    draw.line(true_points, fill=TRUE_COLOR, width=3)
    draw.line(pred_points, fill=FIT_COLOR, width=3)

    if aux_sorted is not None:
        aux_points = [project_xy(x, y, plot, x_sorted.min(), x_sorted.max(), y_min, y_max) for x, y in zip(x_sorted, aux_sorted)]
        draw.line(aux_points, fill=(52, 152, 219), width=3)
        legend = [(TRUE_COLOR, label_a), ((52, 152, 219), label_b), (FIT_COLOR, label_c)]
    else:
        legend = [(TRUE_COLOR, label_a), (FIT_COLOR, label_b)]

    draw_line_legend(draw, plot["left"] + 12, plot["top"] + 12, legend)


def render_scatter_compare_panel(draw, plot, y_true, y_pred):
    lower = min(np.min(y_true), np.min(y_pred)) - 0.4
    upper = max(np.max(y_true), np.max(y_pred)) + 0.4
    draw_axes(draw, plot, lower, upper, lower, upper, "真实值", "预测值", tick_left=plot["left"] - 42, y_label_x=plot["left"] - 52)
    start = project_xy(lower, lower, plot, lower, upper, lower, upper)
    end = project_xy(upper, upper, plot, lower, upper, lower, upper)
    draw.line([start, end], fill=TRUE_COLOR, width=2)
    for true_value, pred_value in zip(y_true, y_pred):
        px, py = project_xy(true_value, pred_value, plot, lower, upper, lower, upper)
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=TEST_COLOR, outline=(255, 255, 255), width=1)


def draw_line_legend(draw, x, y, items):
    font = load_font(15)
    current_y = y
    for color, label in items:
        draw.line([(x, current_y + 8), (x + 24, current_y + 8)], fill=color, width=3)
        draw.text((x + 34, current_y), label, fill=TEXT_COLOR, font=font)
        current_y += 22


def draw_coefficient_bars(image, area, coef_pairs, highlight_sparse=False):
    draw = ImageDraw.Draw(image)
    bar_left = area["left"] + 22
    bar_top = area["top"] + 92
    bar_width = area["width"] - 44
    bar_height = area["height"] - 128
    label_font = load_font(14)
    small_font = load_font(13)

    max_abs = max(max(abs(value) for _, value in coef_pairs), 1e-3)
    center_x = bar_left + bar_width / 2
    draw.line([(center_x, bar_top), (center_x, bar_top + bar_height)], fill=(180, 193, 207), width=1)

    row_height = bar_height / max(len(coef_pairs), 1)
    for index, (name, value) in enumerate(coef_pairs):
        y_center = bar_top + row_height * (index + 0.5)
        length = (abs(value) / max_abs) * (bar_width / 2 - 12)
        if value >= 0:
            x0, x1 = center_x, center_x + length
            color = FIT_COLOR
        else:
            x0, x1 = center_x - length, center_x
            color = (108, 99, 255)
        if highlight_sparse and abs(value) < 0.08:
            color = (190, 196, 204)
        draw.rounded_rectangle([x0, y_center - 8, x1, y_center + 8], radius=6, fill=color)
        draw.text((bar_left, y_center - 10), name, fill=TEXT_COLOR, font=label_font)
        value_text = "{0:+.2f}".format(value)
        text_box = centered_text_box(draw, value_text, small_font)
        draw.text((area["left"] + area["width"] - 18 - text_box[2], y_center - 9), value_text, fill=MUTED_TEXT, font=small_font)


def draw_focus_box(draw, x, y, width, height, title, lines):
    text_font = load_font(16)
    wrapped_groups = [wrap_text(draw, line, text_font, width - 32) for line in lines]
    text_line_count = sum(max(len(group), 1) for group in wrapped_groups)
    box_height = max(height, 50 + text_line_count * 18 + 18 + max(len(lines) - 1, 0) * 4)
    draw.rounded_rectangle([x, y, x + width, y + box_height], radius=16, fill=(255, 255, 255), outline=PANEL_OUTLINE)
    draw.text((x + 16, y + 12), title, fill=TITLE_COLOR, font=load_font(20, bold=True))
    current_y = y + 46
    for wrapped in wrapped_groups:
        for part in wrapped:
            draw.text((x + 16, current_y), part, fill=TEXT_COLOR, font=text_font)
            current_y += 18
        current_y += 4


def draw_footer(draw, plot, message):
    footer_top = plot["top"] + plot["height"] + 88
    draw_focus_box(draw, plot["left"], footer_top, plot["width"], 82, "图像解读", [message])


def draw_text_block(draw, x, y, width, line_height, text):
    font = load_font(16)
    for row_index, line in enumerate(wrap_text(draw, text, font, width)):
        draw.text((x, y + row_index * line_height), line, fill=TEXT_COLOR, font=font)


def project_xy(x_value, y_value, plot, x_min, x_max, y_min, y_max):
    px = plot["left"] + int((x_value - x_min) / max(x_max - x_min, 1e-9) * plot["width"])
    py = plot["top"] + int((y_max - y_value) / max(y_max - y_min, 1e-9) * plot["height"])
    return px, py


def project_point(value, plot, lower, upper, axis):
    if axis == "x":
        return plot["left"] + int((value - lower) / max(upper - lower, 1e-9) * plot["width"])
    return plot["top"] + int((upper - value) / max(upper - lower, 1e-9) * plot["height"])


def draw_diamond(draw, x, y, size, fill_color, outline_color):
    points = [(x, y - size), (x + size, y), (x, y + size), (x - size, y)]
    draw.polygon(points, fill=fill_color, outline=outline_color)


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
