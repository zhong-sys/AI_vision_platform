import base64
import io
import streamlit as st
import numpy as np
from PIL import Image

from components.experiment_panel import (
    render_experiment_guide,
    render_learning_goals,
    render_observation,
    render_parameter_explanation,
    render_preset_controls,
)
from components.page_header import render_page_header

try:
    from matplotlib.figure import Figure
except Exception:
    Figure = None

from pages_modules.regression_data import (
    build_plot_features,
    generate_regression_dataset,
    get_dataset_label,
    get_dataset_options,
    get_dataset_summary,
    get_default_dataset,
    reference_signal,
    split_indices,
)
from pages_modules.regression_metrics import build_regression_metrics
from pages_modules.regression_models import ALGORITHM_LABELS, build_regressor
from pages_modules.regression_text import (
    algorithm_overview,
    bottom_conclusion,
    dataset_overview,
    live_summary,
    parameter_explanation,
    regression_basics_sections,
)
from pages_modules.regression_viz import render_diagnostic_visual, render_main_visual


STATE_DEFAULTS = {
    "reg_lab_algorithm": "linear",
    "reg_lab_dataset": "linear_trend",
    "reg_lab_seed": 13,
    "reg_lab_view_nonce": 0,
    "reg_lab_prev_algorithm": "linear",
    "regression_intro_game_index": 0,
    "regression_intro_game_submitted": False,
    "regression_intro_game_answer": 80.0,
}


REGRESSION_PRESETS = {
    "标准示例": "保持当前算法原始控件默认值，适合先熟悉拟合曲线。",
    "典型现象": "适度增加模型表达能力，观察拟合与泛化之间的平衡。",
    "极端参数": "将复杂度或正则化推向边界，观察曲线可能出现的变化。",
    "高噪声/复杂情况": "增加噪声与样本量，练习从残差和指标中判断稳定性。",
}


def _regression_preset_values(algorithm_key, preset_name):
    """Return only existing regression-control values for a teaching preset."""
    values = {
        "dataset_key": get_default_dataset(algorithm_key),
        "sample_count": 180,
        "noise": default_noise(algorithm_key),
        "test_size": 0.30,
        "fit_intercept": True,
        "standardize": True,
        "degree": 4,
        "alpha": 1.20,
        "C": 1.4,
        "epsilon": 0.22,
        "kernel": "linear",
        "gamma": 0.80,
        "max_depth": 4,
        "min_samples_split": 6,
        "n_estimators": 21,
    }
    if algorithm_key == "lasso":
        values["alpha"] = 0.12
    elif algorithm_key == "rf":
        values.update({"max_depth": 5, "min_samples_split": 6, "n_estimators": 21})

    if preset_name == "典型现象":
        values.update({"sample_count": 220, "noise": min(0.28, 0.80)})
        if algorithm_key == "poly":
            values["degree"] = 6
        elif algorithm_key in ("ridge", "lasso"):
            values["alpha"] = 0.60 if algorithm_key == "ridge" else 0.20
        elif algorithm_key == "svr":
            values.update({"C": 2.0, "epsilon": 0.15, "gamma": 1.20, "kernel": "rbf"})
        elif algorithm_key == "tree":
            values.update({"max_depth": 6, "min_samples_split": 4})
        elif algorithm_key == "rf":
            values.update({"n_estimators": 31, "max_depth": 6, "min_samples_split": 4})
        else:
            values.update({"fit_intercept": True, "standardize": True})
    elif preset_name == "极端参数":
        values.update({"sample_count": 140, "noise": 0.50, "test_size": 0.40})
        if algorithm_key == "poly":
            values["degree"] = 9
        elif algorithm_key in ("ridge", "lasso"):
            values["alpha"] = 4.0 if algorithm_key == "ridge" else 1.0
        elif algorithm_key == "svr":
            values.update({"C": 4.0, "epsilon": 0.05, "gamma": 2.50, "kernel": "poly"})
        elif algorithm_key == "tree":
            values.update({"max_depth": 10, "min_samples_split": 2})
        elif algorithm_key == "rf":
            values.update({"n_estimators": 41, "max_depth": 10, "min_samples_split": 2})
        else:
            values.update({"fit_intercept": False, "standardize": False})
    elif preset_name == "高噪声/复杂情况":
        values.update({"sample_count": 300, "noise": 0.72, "test_size": 0.35})
        if algorithm_key == "poly":
            values["degree"] = 3
        elif algorithm_key in ("ridge", "lasso"):
            values["alpha"] = 2.40 if algorithm_key == "ridge" else 0.50
        elif algorithm_key == "svr":
            values.update({"C": 0.6, "epsilon": 0.40, "gamma": 0.30, "kernel": "linear"})
        elif algorithm_key == "tree":
            values.update({"max_depth": 3, "min_samples_split": 12})
        elif algorithm_key == "rf":
            values.update({"n_estimators": 25, "max_depth": 4, "min_samples_split": 10})
    return values


def _apply_regression_preset():
    preset_name = st.session_state.get("reg_teaching_preset", "标准示例")
    algorithm_key = st.session_state.get("reg_lab_algorithm", "linear")
    values = _regression_preset_values(algorithm_key, preset_name)
    st.session_state["reg_lab_dataset"] = values["dataset_key"]


def _reset_regression_defaults():
    algorithm_key = st.session_state.get("reg_lab_algorithm", "linear")
    values = _regression_preset_values(algorithm_key, "标准示例")
    st.session_state["reg_teaching_preset"] = "标准示例"
    st.session_state["reg_lab_dataset"] = values["dataset_key"]
    st.session_state["reg_lab_seed"] = STATE_DEFAULTS["reg_lab_seed"]
    st.session_state["reg_lab_view_nonce"] = STATE_DEFAULTS["reg_lab_view_nonce"]

REGRESSION_INTRO_QUESTIONS = [
    {
        "title": "学习时间预测成绩",
        "prompt": "某同学每天学习 5 小时，作业完成率为 90%。请你预测他的阶段测试成绩大约是多少分？",
        "reference": 88.0,
        "unit": "分",
        "min_value": 0.0,
        "max_value": 100.0,
        "step": 1.0,
        "default_guess": 80.0,
        "tolerance": 8.0,
    },
    {
        "title": "房屋面积预测价格",
        "prompt": "某套房屋面积为 100 平方米，距离市中心约 8 公里，周边配套较完善。请你预测它的价格指数大约是多少？",
        "reference": 75.0,
        "unit": "",
        "min_value": 0.0,
        "max_value": 120.0,
        "step": 1.0,
        "default_guess": 70.0,
        "tolerance": 10.0,
    },
    {
        "title": "运动时间预测体能评分",
        "prompt": "某同学每周运动 4 次，每次约 45 分钟。请你预测他的体能评分大约是多少？",
        "reference": 82.0,
        "unit": "分",
        "min_value": 0.0,
        "max_value": 100.0,
        "step": 1.0,
        "default_guess": 78.0,
        "tolerance": 8.0,
    },
]

REGRESSION_HISTORY_KEY = "reg_lab_history"

REGRESSION_TASKS = {
    "linear": {
        "title": "直线预测挑战",
        "description": "用尽量简洁的直线解释整体趋势，判断当前场景是否真的适合线性关系。",
    },
    "poly": {
        "title": "曲线拟合挑战",
        "description": "比较多项式阶数变化后的曲线形状，避免为了贴合噪声而让曲线过度震荡。",
    },
    "ridge": {
        "title": "稳定预测挑战",
        "description": "利用正则化削弱噪声影响，在多特征场景中保持预测曲线更稳健。",
    },
    "lasso": {
        "title": "特征筛选挑战",
        "description": "观察正则化如何压缩不重要特征，让模型在保留主要信息的同时控制复杂度。",
    },
    "svr": {
        "title": "误差容忍带挑战",
        "description": "围绕 C、epsilon 与 gamma 调参，寻找平滑拟合与局部贴合之间的平衡。",
    },
    "tree": {
        "title": "分段预测挑战",
        "description": "利用分段预测捕捉局部变化，同时避免树深过大带来的碎片化拟合。",
    },
    "rf": {
        "title": "集成预测挑战",
        "description": "比较单树与森林的稳定性，让集成结果更平滑地刻画真实趋势。",
    },
}


REGRESSION_SCENARIOS = {
    "study_score": {
        "title": "学习时间与考试成绩预测",
        "x_label": "学习时间（相对刻度）",
        "y_label": "考试成绩（相对评分）",
        "focus_x_label": "学习时间",
        "true_value_label": "真实成绩",
        "pred_value_label": "预测成绩",
        "residual_label": "成绩偏差",
        "description": "把当前教学数据映射为学习投入与成绩预测任务，观察学习时长变化时，模型能否稳定刻画整体提升趋势。",
        "trend_focus": "学习投入与成绩之间的整体上升趋势",
    },
    "housing_price": {
        "title": "房屋面积与价格预测",
        "x_label": "房屋面积（相对刻度）",
        "y_label": "房价（相对评分）",
        "focus_x_label": "房屋面积",
        "true_value_label": "真实房价",
        "pred_value_label": "预测房价",
        "residual_label": "价格偏差",
        "description": "把当前教学数据映射为房屋面积与价格关系，适合观察正则化如何在多特征和噪声条件下保持预测稳定性。",
        "trend_focus": "面积变化对价格走势的影响",
    },
    "temperature_trend": {
        "title": "气温变化趋势预测",
        "x_label": "日期序号（相对刻度）",
        "y_label": "气温（相对温度）",
        "focus_x_label": "日期序号",
        "true_value_label": "真实气温",
        "pred_value_label": "预测气温",
        "residual_label": "温度偏差",
        "description": "把当前教学数据映射为一段时间内的气温趋势预测，适合比较平滑拟合、周期波动和局部噪声之间的平衡。",
        "trend_focus": "周期性与局部起伏并存的温度变化趋势",
    },
    "training_fitness": {
        "title": "运动训练时长与体能评分预测",
        "x_label": "训练时长（相对刻度）",
        "y_label": "体能评分（相对评分）",
        "focus_x_label": "训练时长",
        "true_value_label": "真实体能评分",
        "pred_value_label": "预测体能评分",
        "residual_label": "体能偏差",
        "description": "把当前教学数据映射为训练投入与体能表现预测，适合观察树模型如何拟合阶段性变化与局部波动。",
        "trend_focus": "训练投入带来的阶段性体能变化",
    },
}


REGRESSION_DATASET_SCENARIO_MAP = {
    "linear_trend": "study_score",
    "linear_outliers": "study_score",
    "poly_quadratic": "temperature_trend",
    "poly_cubic": "temperature_trend",
    "poly_sine": "temperature_trend",
    "ridge_correlated": "housing_price",
    "ridge_dense_noise": "housing_price",
    "lasso_sparse_signal": "housing_price",
    "lasso_correlated_noise": "housing_price",
    "svr_wave_band": "temperature_trend",
    "svr_noisy_curve": "temperature_trend",
    "tree_piecewise": "training_fitness",
    "tree_local_steps": "training_fitness",
    "rf_piecewise_noise": "training_fitness",
    "rf_wave_ensemble": "training_fitness",
}


def render_regression_lab():
    ensure_state()
    sync_dataset_with_algorithm()
    inject_page_css()

    algorithm_key = st.session_state["reg_lab_algorithm"]
    dataset_key = st.session_state["reg_lab_dataset"]
    algo_info = algorithm_overview(algorithm_key)
    dataset_info = dataset_overview(dataset_key)

    render_page_header(
        title="回归算法可视化学习实验室",
        module="机器学习 / 回归",
        description="先选算法，再切换对应教学数据，随后观察拟合曲线、残差和误差指标如何一起变化。",
        back_key="reg_back_home",
    )
    st.markdown(
        """
        <div class="lab-hero">
            <div class="lab-badges">
                <span class="lab-badge">当前算法：{0}</span>
                <span class="lab-badge">当前数据：{1}</span>
            </div>
            <div class="lab-summary-grid">
                <div class="lab-summary-card"><b>算法一句话：</b>{2}</div>
                <div class="lab-summary-card"><b>数据一句话：</b>{3}</div>
            </div>
        </div>
        """.format(algo_info["title"], dataset_info["title"], algo_info["headline"], dataset_info["summary"]),
        unsafe_allow_html=True,
    )

    render_learning_goals(
        [
            "理解回归模型如何拟合连续数值及其整体趋势。",
            "比较线性、曲线、正则化、支持向量和树模型的拟合方式。",
            "结合 R²、误差和残差观察模型在测试集上的表现。",
            "练习用单变量调参验证对曲线平滑度和泛化的判断。",
        ]
    )
    render_experiment_guide(
        [
            "先确认数据趋势，并记录你对拟合曲线的预期。",
            "从“标准示例”开始，只调整一个参数观察曲线变化。",
            "切换到带噪声或非线性数据，比较训练与测试指标。",
            "查看残差分布，判断模型是否遗漏了明显结构。",
        ]
    )
    preset_name = render_preset_controls(
        REGRESSION_PRESETS,
        key="reg_teaching_preset",
        reset_key="reg_teaching_reset",
        on_preset_change=_apply_regression_preset,
        on_reset=_reset_regression_defaults,
    )
    preset_values = _regression_preset_values(algorithm_key, preset_name)

    with st.expander("回归基础知识", expanded=False):
        for index, (title, body) in enumerate(regression_basics_sections()):
            st.markdown("**{0}. {1}**".format(index + 1, title))
            st.markdown(body)

    render_regression_intro_game()
    st.markdown("---")
    st.markdown("## 进入正式可视化学习")

    control_col, content_col = st.columns([0.84, 2.96], gap="medium")
    with control_col:
        settings = render_control_panel(algorithm_key, preset_values=preset_values)

    lab_result = build_lab_result(
        algorithm_key=settings["algorithm_key"],
        dataset_key=settings["dataset_key"],
        sample_count=settings["sample_count"],
        noise=settings["noise"],
        test_size=settings["test_size"],
        params=settings["params"],
    )

    with content_col:
        render_display_panel(settings["algorithm_key"], settings["dataset_key"], settings["params"], lab_result)

    st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)
    render_parameter_explanation(
        build_regression_parameter_explanations(
            settings["algorithm_key"], settings["params"], settings["noise"], settings["test_size"]
        )
    )
    render_observation(build_regression_observation(lab_result))
    if settings.get("compare_enabled", False):
        render_regression_comparison(lab_result, settings["compare_algorithm"])
    render_bottom_panel(settings["algorithm_key"], settings["dataset_key"], settings["params"], lab_result)


def ensure_state():
    for key, value in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_regression_intro_game():
    questions = REGRESSION_INTRO_QUESTIONS
    index = st.session_state["regression_intro_game_index"] % len(questions)
    question = questions[index]
    answer = st.session_state["regression_intro_game_answer"]
    if not (question["min_value"] <= float(answer) <= question["max_value"]):
        st.session_state["regression_intro_game_answer"] = question["default_guess"]

    st.markdown("### AI 数值预测小游戏")
    st.markdown(
        '<div class="teach-note">先做一个连续值预测小题，理解回归任务关注的是“预测一个数值”，而不是判断离散类别。</div>',
        unsafe_allow_html=True,
    )
    st.markdown("**题目场景：{0}**".format(question["title"]))
    st.markdown(question["prompt"])

    st.number_input(
        "请输入你的预测值",
        min_value=float(question["min_value"]),
        max_value=float(question["max_value"]),
        step=float(question["step"]),
        key="regression_intro_game_answer",
    )

    action_col1, action_col2 = st.columns(2, gap="medium")
    if action_col1.button("提交预测", key="regression_intro_game_submit", use_container_width=True):
        st.session_state["regression_intro_game_submitted"] = True
    action_col2.button(
        "再来一题",
        key="regression_intro_game_next",
        use_container_width=True,
        on_click=next_regression_intro_question,
    )

    if st.session_state["regression_intro_game_submitted"]:
        prediction = float(st.session_state["regression_intro_game_answer"])
        reference = float(question["reference"])
        error = abs(prediction - reference)
        unit = question["unit"]
        suffix = unit if unit else ""
        st.markdown("你的预测：**{0:.1f}{1}**".format(prediction, suffix))
        st.markdown("参考答案：**{0:.1f}{1}**".format(reference, suffix))
        st.markdown("误差：**{0:.1f}**".format(error))
        if error <= question["tolerance"]:
            st.success("预测比较接近。这个例子体现了回归任务：根据输入特征预测连续数值。")
        else:
            st.warning("预测存在一定偏差。回归模型也会通过不断拟合数据来减少预测误差。")
        st.info("下面可以通过线性回归、多项式回归、岭回归、Lasso、SVR 等模型，观察算法如何拟合连续变化趋势。")



def next_regression_intro_question():
    next_index = (st.session_state["regression_intro_game_index"] + 1) % len(REGRESSION_INTRO_QUESTIONS)
    st.session_state["regression_intro_game_index"] = next_index
    st.session_state["regression_intro_game_submitted"] = False
    st.session_state["regression_intro_game_answer"] = REGRESSION_INTRO_QUESTIONS[next_index]["default_guess"]


def sync_dataset_with_algorithm():
    current_algorithm = st.session_state["reg_lab_algorithm"]
    previous_algorithm = st.session_state["reg_lab_prev_algorithm"]
    valid_options = get_dataset_options(current_algorithm)
    if previous_algorithm != current_algorithm or st.session_state["reg_lab_dataset"] not in valid_options:
        st.session_state["reg_lab_dataset"] = get_default_dataset(current_algorithm)
        st.session_state["reg_lab_prev_algorithm"] = current_algorithm
        st.session_state["reg_lab_view_nonce"] += 1


def inject_page_css():
    st.markdown(
        """
        <style>
            html, body, [class*="css"] {
                font-family: "Microsoft YaHei", "PingFang SC", "SimHei", sans-serif;
            }
            .lab-hero {
                background: linear-gradient(135deg, #ffffff 0%, #eef7ff 100%);
                border: 1px solid #dae8f6;
                border-radius: 24px;
                padding: 28px 30px;
                margin-bottom: 18px;
                box-shadow: 0 10px 24px rgba(15, 91, 158, 0.06);
            }
            .lab-overline {
                color: #1A7EC1;
                font-weight: 700;
                font-size: 17px;
                letter-spacing: 1px;
                margin-bottom: 8px;
            }
            .lab-title {
                color: #143d66;
                font-size: 32px;
                font-weight: 800;
                margin-bottom: 10px;
            }
            .lab-subtitle {
                color: #546879;
                font-size: 17px;
                line-height: 1.8;
            }
            .lab-badges {
                margin-top: 14px;
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }
            .lab-badge {
                background: #e8f3fd;
                color: #0F5B9E;
                padding: 7px 14px;
                border-radius: 999px;
                font-size: 17px;
                font-weight: 700;
            }
            .lab-summary-grid {
                margin-top: 16px;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 14px;
            }
            .lab-summary-card {
                background: rgba(255,255,255,0.88);
                border: 1px solid #e1edf7;
                border-radius: 18px;
                padding: 14px 16px;
                color: #4f6475;
                line-height: 1.8;
            }
            .panel-card {
                background: #ffffff;
                border: 1px solid #e4eef7;
                border-radius: 20px;
                padding: 20px;
                box-shadow: 0 8px 20px rgba(15, 91, 158, 0.05);
            }
            .metric-card {
                background: #ffffff;
                border: 1px solid #e4eef7;
                border-radius: 18px;
                padding: 16px 18px;
                box-shadow: 0 8px 18px rgba(15, 91, 158, 0.05);
                min-height: 102px;
            }
            .metric-label {
                color: #5c7082;
                font-size: 16px;
                margin-bottom: 8px;
            }
            .metric-value {
                color: #143d66;
                font-size: 28px;
                font-weight: 800;
            }
            .teach-note {
                background: #f8fbff;
                border: 1px solid #dbe9f7;
                border-radius: 16px;
                padding: 14px 16px;
                color: #53697b;
                line-height: 1.8;
                margin-bottom: 12px;
            }
            .challenge-card {
                background: linear-gradient(135deg, #ffffff 0%, #f3f9ff 100%);
                border: 1px solid #d9e8f6;
                border-radius: 22px;
                padding: 18px 20px;
                margin-bottom: 16px;
                box-shadow: 0 8px 20px rgba(15, 91, 158, 0.05);
            }
            .challenge-title {
                color: #143d66;
                font-size: 24px;
                font-weight: 800;
                margin-bottom: 6px;
            }
            .challenge-desc {
                color: #516779;
                line-height: 1.8;
                margin-bottom: 12px;
            }
            .challenge-context {
                background: #ffffff;
                border: 1px solid #e2edf7;
                border-radius: 14px;
                padding: 10px 12px;
                color: #4f6475;
                margin-bottom: 12px;
                line-height: 1.8;
            }
            .challenge-grid {
                display: grid;
                grid-template-columns: 1.35fr 1fr;
                gap: 14px;
            }
            .challenge-subcard {
                background: rgba(255,255,255,0.94);
                border: 1px solid #e3edf7;
                border-radius: 16px;
                padding: 12px 14px;
            }
            .challenge-subtitle {
                color: #0F5B9E;
                font-weight: 800;
                margin-bottom: 6px;
            }
            .challenge-subcard ul {
                margin: 0;
                padding-left: 18px;
                color: #516779;
                line-height: 1.8;
            }
            .challenge-score {
                color: #143d66;
                font-size: 28px;
                font-weight: 800;
                margin-bottom: 8px;
            }
            .challenge-note {
                color: #53697b;
                line-height: 1.8;
            }
            .challenge-kicker {
                color: #0F5B9E;
                font-size: 16px;
                font-weight: 700;
                letter-spacing: 0.5px;
                margin-bottom: 6px;
            }
            .challenge-task-name {
                color: #143d66;
                font-size: 19px;
                font-weight: 800;
                margin-bottom: 6px;
            }
            .challenge-task-desc {
                color: #55697a;
                line-height: 1.75;
            }
            .goal-checklist {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            .goal-item {
                border-radius: 12px;
                padding: 8px 10px;
                font-size: 17px;
                line-height: 1.6;
            }
            .goal-pass {
                background: #edf8f1;
                color: #206a43;
                border: 1px solid #cfe8d8;
            }
            .goal-wait {
                background: #f7f9fc;
                color: #546879;
                border: 1px solid #e2e9f1;
            }
            .status-chip {
                display: inline-block;
                padding: 6px 12px;
                border-radius: 999px;
                font-size: 16px;
                font-weight: 700;
                margin-bottom: 6px;
            }
            .status-success {
                background: #e9f8ef;
                color: #207245;
                border: 1px solid #cde8d7;
            }
            .status-warning {
                background: #fff5e9;
                color: #9a5b12;
                border: 1px solid #f2dcc0;
            }
            .status-info {
                background: #eef5ff;
                color: #215b96;
                border: 1px solid #d6e5f7;
            }
            .suggestion-list {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            .suggestion-item {
                background: #f8fbff;
                border: 1px solid #dfeaf6;
                border-radius: 12px;
                padding: 9px 11px;
                color: #526779;
                line-height: 1.65;
                font-size: 17px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_control_panel(algorithm_key, preset_values=None):
    preset_values = preset_values or {}
    st.markdown("### 操作区")
    algorithm_key = st.radio(
        "选择算法",
        options=list(ALGORITHM_LABELS.keys()),
        format_func=lambda key: ALGORITHM_LABELS[key],
        key="reg_lab_algorithm",
        label_visibility="collapsed",
    )

    valid_datasets = get_dataset_options(algorithm_key)
    dataset_key = st.selectbox(
        "选择教学数据",
        options=valid_datasets,
        format_func=get_dataset_label,
        key="reg_lab_dataset",
    )

    col1, col2 = st.columns(2)
    if col1.button("重新生成数据", use_container_width=True):
        st.session_state["reg_lab_seed"] += 1
    if col2.button("刷新观察点", use_container_width=True):
        st.session_state["reg_lab_view_nonce"] += 1

    with st.expander("数据设置", expanded=True):
        sample_count = st.slider("样本数量", 100, 320, int(preset_values.get("sample_count", 180)), 20)
        noise = st.slider("噪声强度", 0.00, 0.80, float(preset_values.get("noise", default_noise(algorithm_key))), 0.02)
        test_size = st.slider("测试集比例", 0.20, 0.40, float(preset_values.get("test_size", 0.30)), 0.02)
        st.caption(get_dataset_summary(dataset_key))

    with st.expander("算法参数", expanded=True):
        params = render_parameter_controls(algorithm_key, preset_values=preset_values)

    compare_enabled = st.checkbox("启用基础算法对比模式", key="reg_compare_enabled")
    # PolynomialRegressionCustom is defined for one-dimensional x.  The
    # regularized teaching datasets contain multiple features, so omit that
    # incompatible choice while keeping every same-data compatible option.
    multi_feature_dataset = dataset_key in {
        "ridge_correlated", "ridge_dense_noise", "lasso_sparse_signal", "lasso_correlated_noise"
    }
    compare_options = [
        key for key in ALGORITHM_LABELS
        if key != algorithm_key and not (multi_feature_dataset and key == "poly")
    ]
    if st.session_state.get("reg_compare_algorithm") not in compare_options:
        st.session_state["reg_compare_algorithm"] = compare_options[0]
    if compare_enabled:
        compare_algorithm = st.selectbox(
            "选择对比算法",
            options=compare_options,
            format_func=lambda key: ALGORITHM_LABELS[key],
            key="reg_compare_algorithm",
        )
    else:
        compare_algorithm = st.session_state.get("reg_compare_algorithm", compare_options[0])

    algo_info = algorithm_overview(algorithm_key)
    st.markdown("#### 当前算法提醒")
    st.markdown(algo_info["visual_tip"])

    return {
        "algorithm_key": algorithm_key,
        "dataset_key": dataset_key,
        "sample_count": sample_count,
        "noise": noise,
        "test_size": test_size,
        "params": params,
        "compare_enabled": compare_enabled,
        "compare_algorithm": compare_algorithm,
    }


def render_parameter_controls(algorithm_key, preset_values=None):
    preset_values = preset_values or {}
    if algorithm_key == "linear":
        return {
            "fit_intercept": st.checkbox("加入偏置项", value=bool(preset_values.get("fit_intercept", True))),
            "standardize": st.checkbox("先标准化特征", value=bool(preset_values.get("standardize", True))),
        }
    if algorithm_key == "poly":
        return {"degree": st.slider("多项式阶数", 2, 9, int(preset_values.get("degree", 4)), 1)}
    if algorithm_key == "ridge":
        return {"alpha": st.slider("正则化强度", 0.10, 4.00, float(preset_values.get("alpha", 1.20)), 0.05)}
    if algorithm_key == "lasso":
        return {"alpha": st.slider("正则化强度", 0.01, 1.00, float(preset_values.get("alpha", 0.12)), 0.01)}
    if algorithm_key == "svr":
        return {
            "C": st.slider("惩罚系数 C", 0.2, 4.0, float(preset_values.get("C", 1.4)), 0.1),
            "epsilon": st.slider("容忍带宽度", 0.05, 0.60, float(preset_values.get("epsilon", 0.22)), 0.01),
            "kernel": st.selectbox(
                "核函数",
                options=["linear", "rbf", "poly"],
                format_func=lambda item: {"linear": "线性核", "rbf": "RBF 核", "poly": "多项式核"}[item],
                index=["linear", "rbf", "poly"].index(preset_values.get("kernel", "linear")),
            ),
            "gamma": st.slider("核函数影响范围", 0.10, 2.50, float(preset_values.get("gamma", 0.80)), 0.05),
        }
    if algorithm_key == "tree":
        return {
            "max_depth": st.slider("最大深度", 1, 10, int(preset_values.get("max_depth", 4)), 1),
            "min_samples_split": st.slider("分裂所需最少样本数", 2, 20, int(preset_values.get("min_samples_split", 6)), 1),
        }
    return {
        "n_estimators": st.slider("树的数量", 5, 41, int(preset_values.get("n_estimators", 21)), 2),
        "max_depth": st.slider("最大深度", 2, 10, int(preset_values.get("max_depth", 5)), 1),
        "min_samples_split": st.slider("分裂所需最少样本数", 2, 20, int(preset_values.get("min_samples_split", 6)), 1),
    }


def default_noise(algorithm_key):
    mapping = {
        "linear": 0.18,
        "poly": 0.14,
        "ridge": 0.18,
        "lasso": 0.16,
        "svr": 0.14,
        "tree": 0.10,
        "rf": 0.16,
    }
    return mapping[algorithm_key]


def build_regression_parameter_explanations(algorithm_key, params, noise, test_size):
    """Create cautious, rule-based explanations for the current controls."""
    explanations = [
        "噪声强度为 {0:.2f}：噪声越高，观测点偏离趋势的可能性通常越大，建议结合残差图观察。".format(noise),
        "测试集比例为 {0:.0%}：它决定用于检验泛化表现的样本占比。".format(test_size),
    ]
    if algorithm_key == "linear":
        explanations.append("偏置项={0}、标准化={1}：建议分别切换一个选项，观察拟合曲线和指标是否稳定。".format(
            "开启" if params["fit_intercept"] else "关闭",
            "开启" if params["standardize"] else "关闭",
        ))
    elif algorithm_key == "poly":
        degree = int(params["degree"])
        if degree >= 7:
            explanations.append("多项式阶数为 {0}：阶数较高时曲线可能更灵活，也可能更贴近噪声。".format(degree))
        elif degree <= 3:
            explanations.append("多项式阶数为 {0}：曲线相对简洁，可能无法表达复杂的局部变化。".format(degree))
        else:
            explanations.append("多项式阶数为 {0}：建议逐步调整阶数，比较曲线平滑度与残差分布。".format(degree))
    elif algorithm_key in ("ridge", "lasso"):
        alpha = float(params["alpha"])
        explanations.append("正则化强度 alpha={0:.2f}：数值增大通常会更强地约束模型参数，拟合曲线可能更平滑。".format(alpha))
        if algorithm_key == "lasso":
            explanations.append("Lasso 还可能把部分系数压到接近零，建议结合误差指标观察信息保留情况。")
    elif algorithm_key == "svr":
        explanations.append("C={0:.1f}、epsilon={1:.2f}：C 控制误差惩罚，epsilon 定义容忍带宽度，建议一次只改变一个。".format(
            float(params["C"]), float(params["epsilon"])
        ))
        explanations.append("当前核函数为 {0}，gamma={1:.2f}：gamma 较大时局部影响可能更明显。".format(
            params["kernel"], float(params["gamma"])
        ))
    elif algorithm_key == "tree":
        explanations.append("最大深度={0}、最小分裂样本={1}：树更深或更易分裂时可能捕捉更多局部细节。".format(
            int(params["max_depth"]), int(params["min_samples_split"])
        ))
    else:
        explanations.append("树数量={0}：更多树通常会让集成预测更稳定，但计算量也会增加。".format(int(params["n_estimators"])))
        explanations.append("最大深度={0}、最小分裂样本={1}：建议结合单树与森林的残差差异观察。".format(
            int(params["max_depth"]), int(params["min_samples_split"])
        ))
    return explanations


def build_regression_observation(lab_result):
    metrics = lab_result["metrics"]
    train_r2 = float(metrics["train_r2"])
    test_r2 = float(metrics["test_r2"])
    observation = [
        "当前训练 R² {0:.3f}，测试 R² {1:.3f}；测试 MAE {2:.3f}，RMSE {3:.3f}。".format(
            train_r2, test_r2, float(metrics["test_mae"]), float(metrics["test_rmse"])
        ),
        "建议先看拟合曲线，再检查残差是否在某些区间持续偏正或偏负。",
    ]
    if train_r2 - test_r2 >= 0.15:
        observation.append("训练与测试 R² 差距相对明显，可能存在泛化不稳定现象，建议尝试降低复杂度或增加正则化。")
    else:
        observation.append("当前训练与测试 R² 差距不大，可继续切换数据集验证趋势是否稳定。")
    return observation


def build_regression_comparison(primary_result, compare_algorithm):
    """Fit a second existing regressor on the primary run's exact split."""
    dataset_key = primary_result["dataset_key"]
    params = _regression_preset_values(compare_algorithm, "标准示例")
    X_train = primary_result["X_train"]
    X_test = primary_result["X_test"]
    y_train = primary_result["y_train"]
    y_test = primary_result["y_test"]
    x_train = primary_result["x_train"]
    x_test = primary_result["x_test"]
    x_grid = primary_result["visual_context"]["x_grid"]
    model = build_regressor(compare_algorithm, params)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    X_grid = build_plot_features(dataset_key, x_grid)
    y_grid_pred = model.predict(X_grid)
    metrics = build_regression_metrics(y_train, y_train_pred, y_test, y_test_pred)
    focus_index = st.session_state["reg_lab_view_nonce"] % len(X_test)
    x_test_sort_index = np.argsort(x_test)
    context = {
        "visual_title": ALGORITHM_LABELS[compare_algorithm],
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "x_grid": x_grid,
        "y_grid_pred": y_grid_pred,
        "y_grid_true": primary_result["visual_context"]["y_grid_true"],
        "focus_x": float(x_test[focus_index]),
        "focus_true": float(y_test[focus_index]),
        "focus_pred": float(y_test_pred[focus_index]),
        "focus_residual": float(y_test_pred[focus_index] - y_test[focus_index]),
        "residual_demo_indices": np.argsort(np.abs(y_test_pred - y_test))[::-1].tolist(),
        "x_test_sorted": x_test[x_test_sort_index],
        "x_test_sort_index": x_test_sort_index,
        "y_test_sorted": y_test[x_test_sort_index],
        "y_test_pred_sorted": y_test_pred[x_test_sort_index],
        "metrics": metrics,
        "x_axis_label": "输入特征 x",
        "y_axis_label": "目标值 y",
        "residual_axis_label": "残差",
        "focus_x_label": "输入特征",
        "true_value_label": "真实值",
        "pred_value_label": "预测值",
    }
    # Feature labels are presentation-only for the comparison panel; the
    # primary algorithm and its original metadata remain untouched.
    feature_names = ["特征 {0}".format(index + 1) for index in range(X_train.shape[1])]
    extras = build_algorithm_extras(
        algorithm_key=compare_algorithm,
        params=params,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        x_grid=x_grid,
        feature_names=feature_names,
        dataset_key=dataset_key,
    )
    if compare_algorithm == "rf":
        extras["forest_gain"] = float(metrics["test_r2"] - extras["single_tree_metrics"]["test_r2"])
    context.update(extras)
    return {"algorithm_key": compare_algorithm, "metrics": metrics, "visual_context": context}


def render_regression_comparison(primary_result, compare_algorithm):
    comparison = build_regression_comparison(primary_result, compare_algorithm)
    st.markdown("### 基础算法对比")
    st.caption("两种算法复用同一数据集、同一 train/test split 和当前页面随机种子；对比算法使用原始默认参数。")
    columns = st.columns(2, gap="medium")
    entries = [(primary_result["algorithm_key"], primary_result), (compare_algorithm, comparison)]
    for column, (algorithm_key, result) in zip(columns, entries):
        with column:
            st.markdown("#### {0}".format(ALGORITHM_LABELS[algorithm_key]))
            visual = render_main_visual(algorithm_key, result["visual_context"])
            render_visual_output(visual, "对比主图未成功生成，请稍后重试。")
            st.metric("测试 R²", "{0:.3f}".format(float(result["metrics"]["test_r2"])))


def build_lab_result(algorithm_key, dataset_key, sample_count, noise, test_size, params):
    X, y, dataset_meta = generate_regression_dataset(
        dataset_key=dataset_key,
        n_samples=sample_count,
        noise=noise,
        random_state=st.session_state["reg_lab_seed"],
    )
    x_all = dataset_meta["plot_x"]
    feature_names = dataset_meta["feature_names"]

    train_index, test_index = split_indices(len(X), test_size, st.session_state["reg_lab_seed"])
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    x_train, x_test = x_all[train_index], x_all[test_index]

    model = build_regressor(algorithm_key, params)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    x_grid = np.linspace(x_all.min() - 0.15, x_all.max() + 0.15, 280)
    X_grid = build_plot_features(dataset_key, x_grid)
    y_grid_pred = model.predict(X_grid)
    y_grid_true = reference_signal(dataset_key, x_grid)

    metrics = build_regression_metrics(y_train, y_train_pred, y_test, y_test_pred)
    scenario = get_regression_scenario(dataset_key)
    focus_index = st.session_state["reg_lab_view_nonce"] % len(X_test)
    residual_demo_indices = np.argsort(np.abs(y_test_pred - y_test))[::-1]
    x_test_sort_index = np.argsort(x_test)

    visual_context = {
        "visual_title": ALGORITHM_LABELS[algorithm_key],
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "x_grid": x_grid,
        "y_grid_pred": y_grid_pred,
        "y_grid_true": y_grid_true,
        "focus_x": float(x_test[focus_index]),
        "focus_true": float(y_test[focus_index]),
        "focus_pred": float(y_test_pred[focus_index]),
        "focus_residual": float(y_test_pred[focus_index] - y_test[focus_index]),
        "residual_demo_indices": residual_demo_indices.tolist(),
        "x_test_sorted": x_test[x_test_sort_index],
        "x_test_sort_index": x_test_sort_index,
        "y_test_sorted": y_test[x_test_sort_index],
        "y_test_pred_sorted": y_test_pred[x_test_sort_index],
        "metrics": metrics,
        "x_axis_label": "输入特征 x",
        "y_axis_label": "目标值 y",
        "residual_axis_label": "残差",
        "focus_x_label": "输入特征",
        "true_value_label": "真实值",
        "pred_value_label": "预测值",
    }

    extras = build_algorithm_extras(
        algorithm_key=algorithm_key,
        params=params,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        x_grid=x_grid,
        feature_names=feature_names,
        dataset_key=dataset_key,
    )
    if algorithm_key == "rf":
        extras["forest_gain"] = float(metrics["test_r2"] - extras["single_tree_metrics"]["test_r2"])
    visual_context.update(extras)
    game = build_regression_game_context(
        algorithm_key=algorithm_key,
        dataset_key=dataset_key,
        params=params,
        metrics=metrics,
        x_test=x_test,
        y_test=y_test,
        y_test_pred=y_test_pred,
    )

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "x_train": x_train,
        "x_test": x_test,
        "model": model,
        "params": params,
        "metrics": metrics,
        "extras": extras,
        "visual_context": visual_context,
        "dataset_key": dataset_key,
        "algorithm_key": algorithm_key,
        "game": game,
    }


def build_algorithm_extras(
    algorithm_key,
    params,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    y_train_pred,
    y_test_pred,
    x_grid,
    feature_names,
    dataset_key,
):
    extras = {}

    if algorithm_key == "linear":
        extras["primary_coef"] = float(model.coef_[0])
        extras["intercept"] = float(model.intercept_)
        return extras

    if algorithm_key == "poly":
        extras["degree"] = params["degree"]
        if params["degree"] <= 2:
            extras["shape_note"] = "曲线目前更偏平滑，适合观察欠拟合。"
        elif params["degree"] >= 7:
            extras["shape_note"] = "曲线弯折已经比较明显，要特别留意测试集误差。"
        else:
            extras["shape_note"] = "曲线复杂度处于中间水平，通常更适合教学比较。"
        return extras

    if algorithm_key in ["ridge", "lasso"]:
        coef_pairs = list(zip(feature_names, model.coef_.tolist()))
        extras["coef_pairs"] = coef_pairs
        extras["alpha"] = params["alpha"]
        extras["max_abs_coef"] = float(max(abs(value) for _, value in coef_pairs))
        extras["near_zero_count"] = int(sum(abs(value) < 0.08 for _, value in coef_pairs))
        return extras

    if algorithm_key == "svr":
        support_index = model.support_indices()
        train_residual = np.abs(y_train_pred - y_train)
        extras["support_x"] = X_train[support_index, 0]
        extras["support_y"] = y_train[support_index]
        extras["support_vector_count"] = model.support_vector_count()
        extras["epsilon"] = params["epsilon"]
        extras["tube_ratio"] = float(np.mean(train_residual <= params["epsilon"]))
        return extras

    if algorithm_key == "tree":
        grid_prediction = model.predict(build_plot_features(dataset_key, x_grid))
        extras["leaf_count"] = int(model.leaf_count_)
        extras["plateau_count"] = int(len(np.unique(np.round(grid_prediction, 2))))
        return extras

    single_tree_train_pred = model.single_tree_predict(X_train)
    single_tree_test_pred = model.single_tree_predict(X_test)
    single_tree_metrics = build_regression_metrics(y_train, single_tree_train_pred, y_test, single_tree_test_pred)
    extras["single_tree_grid_pred"] = model.single_tree_predict(build_plot_features(dataset_key, x_grid))
    extras["single_tree_metrics"] = single_tree_metrics
    extras["single_tree_test_sorted"] = single_tree_test_pred[np.argsort(X_test[:, 0])]
    return extras


def render_display_panel(algorithm_key, dataset_key, params, lab_result):
    algo_info = algorithm_overview(algorithm_key)
    st.markdown("### 拟合曲线与样本分布")
    main_visual = render_main_visual(algorithm_key, lab_result["visual_context"])
    render_visual_output(main_visual, "当前主图未成功生成，请调整参数后重试。")
    with st.expander("图像观察与调参提示", expanded=False):
        st.markdown("**拟合观察**")
        st.info(live_summary(algorithm_key, params, lab_result["metrics"], lab_result["extras"]))
        st.markdown("**调参提示**")
        st.info(parameter_explanation(algorithm_key, params, lab_result["extras"]))
    with st.expander("教学提示", expanded=False):
        st.markdown("**一句话速览**")
        st.markdown(algo_info["headline"])
        st.markdown("**算法原理**")
        st.markdown(algo_info["principle"])
        st.markdown("**适用场景**")
        st.markdown(algo_info["fit"])


def render_bottom_panel(algorithm_key, dataset_key, params, lab_result):
    st.markdown("### 指标与误差分析")
    metrics = lab_result["metrics"]
    gap = abs(metrics["train_r2"] - metrics["test_r2"])
    metric_values = [
        ("训练集 R²", "{0:.3f}".format(metrics["train_r2"])),
        ("测试集 R²", "{0:.3f}".format(metrics["test_r2"])),
        ("训练/测试差距", "{0:.3f}".format(gap)),
        ("测试集 MSE", "{0:.3f}".format(metrics["test_mse"])),
        ("测试集 MAE", "{0:.3f}".format(metrics["test_mae"])),
        ("测试集 RMSE", "{0:.3f}".format(metrics["test_rmse"])),
    ]

    render_metric_cards(metric_values, columns_per_row=3)

    st.markdown("#### 残差与预测关系")
    diagnostic_visual = render_diagnostic_visual(algorithm_key, lab_result["visual_context"])
    render_visual_output(
        diagnostic_visual,
        "当前诊断图未成功生成。",
        image_width="min(100%, 960px)",
    )
    st.markdown("#### 调参与结果提示")
    render_regression_regular_hint(lab_result["game"])
    st.markdown("#### 结果解读")
    st.markdown(lab_result["game"]["interpretation"])
    st.markdown("#### 当前模型结论")
    st.markdown(bottom_conclusion(algorithm_key, dataset_key, metrics, lab_result["extras"]))


def get_regression_scenario(dataset_key):
    scenario_key = REGRESSION_DATASET_SCENARIO_MAP.get(dataset_key, "study_score")
    return REGRESSION_SCENARIOS[scenario_key]


def build_regression_detective_game(algorithm_key, dataset_key, params, metrics, y_train, y_test):
    scenario = get_regression_scenario(dataset_key)
    gap = abs(metrics["train_r2"] - metrics["test_r2"])
    target_span = max(float(np.ptp(y_test)), float(np.std(y_train)), 1e-6)
    rmse_ratio = metrics["test_rmse"] / target_span
    mae_ratio = metrics["test_mae"] / target_span
    fit_state = get_regression_fit_state(metrics, gap, rmse_ratio, mae_ratio)
    suggestions = generate_parameter_suggestions(algorithm_key, params, metrics, gap, rmse_ratio, mae_ratio)
    return {
        "title": "预测侦探：哪条曲线更可信？",
        "description": "观察拟合曲线和样本分布，判断当前模型是欠拟合、拟合较合理，还是过拟合。然后通过调整参数验证你的判断。",
        "scenario_name": scenario["title"],
        "context": "{0} · {1}".format(scenario["title"], ALGORITHM_LABELS[algorithm_key]),
        "options": ["欠拟合", "拟合较合理", "过拟合"],
        "system_state": fit_state["label"],
        "system_reason": fit_state["reason"],
        "explanation": get_regression_algorithm_clue(algorithm_key),
        "hint": fit_state["hint"],
        "hint_tone": fit_state["tone"],
        "suggestions": suggestions[:2],
        "interpretation": build_regression_interpretation(scenario, metrics, gap, rmse_ratio),
        "next_step": "继续调整当前算法的关键参数，再比较训练集和测试集表现是否一起改善。",
    }


def get_regression_fit_state(metrics, gap, rmse_ratio, mae_ratio):
    train_r2 = metrics["train_r2"]
    test_r2 = metrics["test_r2"]
    if train_r2 <= 0.65 and test_r2 <= 0.60:
        return {
            "label": "欠拟合",
            "tone": "info",
            "reason": "训练集和测试集 R² 都偏低，曲线整体过于简单，还没有抓住主要趋势。",
            "hint": "修复提示：可以尝试增加模型复杂度，例如提高多项式阶数，或放宽 SVR / 树模型的表达能力。",
        }
    if train_r2 - test_r2 >= 0.18 or (gap >= 0.18 and rmse_ratio >= 0.22):
        return {
            "label": "过拟合",
            "tone": "warning",
            "reason": "训练集表现明显高于测试集，模型可能记住了训练噪声，测试误差也在放大。",
            "hint": "修复提示：尝试降低多项式阶数、增大正则化强度，或降低树模型深度。",
        }
    return {
        "label": "拟合较合理",
        "tone": "success",
        "reason": "训练集与测试集表现接近，误差指标也较稳定，当前曲线对主要趋势的解释较可信。",
        "hint": "修复提示：继续观察残差是否稳定分布，再做小幅调参验证模型是否仍然稳健。",
    }


def get_regression_algorithm_clue(algorithm_key):
    clues = {
        "linear": "线性回归适合整体趋势接近直线的数据，如果真实关系明显弯曲，直线可能欠拟合。",
        "poly": "多项式阶数越高，曲线越灵活，但阶数过高可能记住噪声，出现过拟合。",
        "ridge": "岭回归通过 L2 正则化限制系数过大，使模型更稳定。",
        "lasso": "Lasso 通过 L1 正则化压缩部分系数，有助于控制复杂度。",
        "svr": "SVR 通过误差容忍带进行平滑预测，参数会影响曲线平滑程度。",
        "tree": "决策树回归会形成分段预测，深度过大时容易过拟合。",
        "rf": "随机森林通过多棵树平均预测，通常比单棵树更稳定。",
    }
    return clues[algorithm_key]


def render_prediction_detective_game(game, key_prefix):
    st.markdown("### {0}".format(game["title"]))
    st.markdown(
        """
        <div class="challenge-card">
            <div class="challenge-kicker">问题驱动式小游戏</div>
            <div class="challenge-task-name">先判断模型状态，再调参验证</div>
            <div class="challenge-task-desc">{0}</div>
            <div class="challenge-context"><b>当前预测场景：</b>{1}</div>
        </div>
        """.format(game["description"], game["scenario_name"]),
        unsafe_allow_html=True,
    )
    guess = st.radio(
        "你觉得当前曲线状态更接近哪一种？",
        options=game["options"],
        key="{0}_detective_choice".format(key_prefix),
        label_visibility="collapsed",
    )
    if guess == game["system_state"]:
        st.success("判断正确。系统也认为当前模型属于“{0}”。".format(game["system_state"]))
    else:
        st.info("你的判断很接近，但系统当前更倾向于“{0}”。".format(game["system_state"]))
    st.markdown("**系统判断**")
    st.caption(game["system_reason"])
    st.markdown("**算法解释**")
    st.caption(game["explanation"])
    st.markdown("**继续观察**")
    st.caption(game["next_step"])


def build_regression_challenge(algorithm_key, dataset_key, params, metrics, y_train, y_test):
    scenario = get_regression_scenario(dataset_key)
    gap = abs(metrics["train_r2"] - metrics["test_r2"])
    target_span = max(float(np.ptp(y_test)), float(np.std(y_train)), 1e-6)
    rmse_ratio = metrics["test_rmse"] / target_span
    mae_ratio = metrics["test_mae"] / target_span
    score = calculate_regression_challenge_score(algorithm_key, params, metrics, gap, rmse_ratio, mae_ratio)
    task_info = get_algorithm_task_info(algorithm_key)
    goal_items = evaluate_challenge_goals(algorithm_key, params, metrics, gap, rmse_ratio, mae_ratio)
    progress = int(round(100 * sum(1 for item in goal_items if item["passed"]) / max(len(goal_items), 1)))
    status = render_progress_status(score, progress, goal_items)
    suggestions = generate_parameter_suggestions(algorithm_key, params, metrics, gap, rmse_ratio, mae_ratio)
    feedback = generate_regression_feedback(algorithm_key, params, metrics, gap, rmse_ratio, mae_ratio)
    interpretation = build_regression_interpretation(scenario, metrics, gap, rmse_ratio)
    return {
        "title": "预测实验室",
        "description": "请选择一个真实预测场景，并调节回归模型参数，让曲线尽量贴近真实数据，同时避免过拟合。",
        "task_title": task_info["title"],
        "task_description": task_info["description"],
        "context_title": "当前场景映射",
        "context": "{0} · {1}".format(scenario["title"], ALGORITHM_LABELS[algorithm_key]),
        "scenario_name": scenario["title"],
        "goals": [item["label"] for item in goal_items],
        "goal_items": goal_items,
        "progress": progress,
        "status_label": status["label"],
        "status_note": status["note"],
        "status_tone": status["tone"],
        "score": score,
        "score_label": "当前预测得分",
        "feedback": feedback["message"],
        "tone": feedback["tone"],
        "suggestions": suggestions,
        "interpretation": interpretation,
        "history_row": {
            "算法": ALGORITHM_LABELS[algorithm_key],
            "场景": scenario["title"],
            "测试 R²": "{0:.3f}".format(metrics["test_r2"]),
            "RMSE": "{0:.3f}".format(metrics["test_rmse"]),
            "预测得分": score,
            "结论": status["label"],
        },
        "record_message": "已记录当前回归实验。",
    }


def get_algorithm_task_info(algorithm_key):
    return REGRESSION_TASKS[algorithm_key]


def evaluate_challenge_goals(algorithm_key, params, metrics, gap, rmse_ratio, mae_ratio):
    goal_items = [
        {"label": "测试集 R² 保持在 0.78 以上", "passed": metrics["test_r2"] >= 0.78},
        {"label": "训练与测试差距控制在 0.18 内", "passed": gap <= 0.18},
        {"label": "RMSE 与 MAE 保持在可接受范围", "passed": rmse_ratio <= 0.28 and mae_ratio <= 0.22},
    ]

    if algorithm_key == "linear":
        goal_items.append({"label": "线性趋势足以解释主要变化", "passed": metrics["test_r2"] >= 0.70})
    elif algorithm_key == "poly":
        goal_items.append({"label": "多项式阶数保持适中", "passed": 2 <= params["degree"] <= 6})
    elif algorithm_key in ["ridge", "lasso"]:
        goal_items.append({"label": "正则化强度保持稳健", "passed": 0.05 <= params["alpha"] <= 3.0})
    elif algorithm_key == "svr":
        goal_items.append({"label": "容忍带宽与复杂度保持平衡", "passed": params["epsilon"] >= 0.10 and params["C"] <= 3.6})
    elif algorithm_key == "tree":
        goal_items.append({"label": "树深不过深，分段数适中", "passed": params["max_depth"] <= 6})
    else:
        goal_items.append({"label": "集成规模稳定且不过度复杂", "passed": params["max_depth"] <= 6 and params["n_estimators"] >= 11})
    return goal_items


def render_progress_status(score, progress, goal_items):
    passed_count = sum(1 for item in goal_items if item["passed"])
    total_count = max(len(goal_items), 1)
    if score >= 85 and passed_count >= total_count - 1:
        return {"label": "已通关", "tone": "success", "note": "已完成 {0}/{1} 项目标".format(passed_count, total_count)}
    if score >= 70 or passed_count >= max(2, total_count - 2):
        return {"label": "接近通关", "tone": "warning", "note": "已完成 {0}/{1} 项目标".format(passed_count, total_count)}
    return {"label": "继续挑战", "tone": "info", "note": "已完成 {0}/{1} 项目标".format(passed_count, total_count)}


def render_goal_checklist(goal_items):
    rows = []
    for item in goal_items:
        css_class = "goal-pass" if item["passed"] else "goal-wait"
        prefix = "✓" if item["passed"] else "○"
        rows.append('<div class="goal-item {0}">{1} {2}</div>'.format(css_class, prefix, item["label"]))
    return '<div class="goal-checklist">{0}</div>'.format("".join(rows))


def calculate_regression_challenge_score(algorithm_key, params, metrics, gap, rmse_ratio, mae_ratio):
    adjusted_test_r2 = np.clip((metrics["test_r2"] + 0.15) / 1.15, 0.0, 1.0)
    score = adjusted_test_r2 * 60.0
    score += max(0.0, 18.0 * (1.0 - min(gap / 0.30, 1.0)))
    score += max(0.0, 12.0 * (1.0 - min(rmse_ratio / 0.35, 1.0)))
    score += max(0.0, 10.0 * (1.0 - min(mae_ratio / 0.28, 1.0)))

    complexity_penalty = 0.0
    if algorithm_key == "poly":
        complexity_penalty += max(0, params["degree"] - 5) * 2.8
    elif algorithm_key in ["ridge", "lasso"]:
        if params["alpha"] < 0.05:
            complexity_penalty += 3.0
        elif params["alpha"] > 3.0:
            complexity_penalty += 2.0
    elif algorithm_key == "svr":
        if params["kernel"] != "linear" and params["gamma"] >= 1.8:
            complexity_penalty += 5.0
        if params["C"] >= 3.6:
            complexity_penalty += 3.0
        if params["epsilon"] < 0.10:
            complexity_penalty += 2.0
    elif algorithm_key == "tree":
        complexity_penalty += max(0, params["max_depth"] - 6) * 2.2
    elif algorithm_key == "rf":
        complexity_penalty += max(0, params["max_depth"] - 6) * 1.8
        complexity_penalty += max(0, params["n_estimators"] - 31) * 0.12

    return int(np.clip(round(score - complexity_penalty), 0, 100))


def generate_regression_feedback(algorithm_key, params, metrics, gap, rmse_ratio, mae_ratio):
    train_r2 = metrics["train_r2"]
    test_r2 = metrics["test_r2"]
    suggestions = generate_parameter_suggestions(algorithm_key, params, metrics, gap, rmse_ratio, mae_ratio)

    if train_r2 >= 0.82 and test_r2 >= 0.80 and gap <= 0.12:
        return {"tone": "success", "message": "当前模型拟合较合理，训练集与测试集表现接近，已经较好捕捉到场景中的主要趋势。"}
    if train_r2 - test_r2 >= 0.18:
        return {
            "tone": "warning",
            "message": "训练集 R² 明显高于测试集 R²，模型可能过拟合。{0}".format(suggestions[0]),
        }
    if train_r2 <= 0.45 and test_r2 <= 0.35:
        return {
            "tone": "info",
            "message": "训练集和测试集 R² 都偏低，模型可能欠拟合。{0}".format(suggestions[0]),
        }
    if rmse_ratio >= 0.30 or mae_ratio >= 0.22:
        return {
            "tone": "warning",
            "message": "当前误差仍然偏大，预测曲线与真实数据之间存在较明显偏差。{0}".format(suggestions[0]),
        }
    return {
        "tone": "info",
        "message": "当前模型已经解释了主要趋势，但局部区段仍有偏差，可以继续平衡拟合能力与曲线复杂度。",
    }


def generate_parameter_suggestions(algorithm_key, params, metrics, gap, rmse_ratio, mae_ratio):
    suggestions = []
    if gap >= 0.18:
        if algorithm_key == "poly":
            suggestions.append("先降低多项式阶数，避免曲线为了贴合训练点而过度震荡。")
        elif algorithm_key in ["ridge", "lasso"]:
            suggestions.append("适当增大 alpha，观察正则化是否能提升测试集稳定性。")
        elif algorithm_key == "svr":
            suggestions.append("先降低 C 或 gamma，让拟合曲线不要记住训练噪声。")
        elif algorithm_key == "tree":
            suggestions.append("优先降低树深，减少分段预测对局部噪声的记忆。")
        elif algorithm_key == "rf":
            suggestions.append("先控制树深，再比较更多树是否带来更稳定的集成效果。")
        else:
            suggestions.append("如果直线解释不了趋势，可考虑切换到更适合非线性关系的模型。")
    if rmse_ratio >= 0.28 or mae_ratio >= 0.22:
        if algorithm_key == "poly":
            suggestions.append("在保持趋势的前提下微调阶数，观察残差是否更均匀。")
        elif algorithm_key in ["ridge", "lasso"]:
            suggestions.append("比较不同 alpha 下的 RMSE 与 MAE，寻找误差更低的区间。")
        elif algorithm_key == "svr":
            suggestions.append("联动调整 epsilon 与 gamma，观察误差容忍带是否更合适。")
        elif algorithm_key in ["tree", "rf"]:
            suggestions.append("比较树深变化对高误差区段的影响，避免分段过多。")
        else:
            suggestions.append("先确认线性趋势是否足够，再考虑更换更能表达弯曲关系的模型。")
    if metrics["test_r2"] < 0.78:
        if algorithm_key == "linear":
            suggestions.append("如果趋势明显弯曲，可以尝试切换到多项式或 SVR 重新比较。")
        elif algorithm_key == "poly":
            suggestions.append("若曲线仍偏平，可适当提高阶数，但要同步观察测试集 R²。")
        elif algorithm_key in ["ridge", "lasso"]:
            suggestions.append("适当减小 alpha，避免正则化过强压掉有效趋势。")
        elif algorithm_key == "svr":
            suggestions.append("尝试更换 kernel，并联动调整 C 与 gamma 提升拟合能力。")
        elif algorithm_key == "tree":
            suggestions.append("若分段过于粗糙，可适度提高树深再比较测试集表现。")
        else:
            suggestions.append("在控制树深的前提下增加 n_estimators，观察集成后曲线是否更贴近真实值。")

    if not suggestions:
        suggestions.append("当前参数组合已经比较稳健，可继续微调并观察误差曲线是否进一步收敛。")
    return suggestions[:3]


def build_regression_interpretation(scenario, metrics, gap, rmse_ratio):
    if metrics["test_r2"] >= 0.82 and gap <= 0.12:
        return "当前模型已经较好地抓住了主要趋势，训练集与测试集表现接近，整体预测比较稳定。"
    if gap >= 0.18 and metrics["train_r2"] > metrics["test_r2"]:
        return "当前模型在训练集上表现更好，但对新样本的预测稳定性还不够，仍需警惕过拟合。"
    if rmse_ratio >= 0.30:
        return "当前模型虽然识别出了大致趋势，但在局部区间仍存在较明显的预测偏差。"
    return "当前模型已经抓住了部分变化规律，但仍可继续优化参数，让拟合曲线更贴近真实数据。"

def render_challenge_panel(challenge, key_prefix, compact=False):
    if compact:
        st.markdown(
            """
            <div class="challenge-card">
                <div class="challenge-kicker">{0}</div>
                <div class="challenge-task-name">{1}</div>
                <div class="challenge-task-desc">{2}</div>
                <div class="challenge-context"><b>{3}：</b>{4}</div>
            </div>
            """.format(
                challenge["context"],
                challenge["task_title"],
                challenge["task_description"],
                challenge["context_title"],
                challenge["description"],
            ),
            unsafe_allow_html=True,
        )
        st.metric(challenge["score_label"], "{0} / 100".format(challenge["score"]))
        st.progress(challenge["progress"] / 100.0)
        st.markdown(
            '<div class="status-chip status-{0}">{1}</div>'.format(challenge["status_tone"], challenge["status_label"]),
            unsafe_allow_html=True,
        )
        st.caption(challenge["status_note"])
        st.markdown("**智能反馈**")
        st.caption(challenge["feedback"])
        quick_suggestions = challenge["suggestions"][:2]
        if quick_suggestions:
            st.markdown("**调参建议**")
            for item in quick_suggestions:
                st.markdown("- {0}".format(item))
        with st.expander("查看目标与记录", expanded=False):
            st.markdown("**目标清单**")
            for item in challenge["goal_items"]:
                status_text = "已达成" if item["passed"] else "待优化"
                st.markdown("- `{0}` {1}".format(status_text, item["label"]))
            if st.button("记录当前实验", key="{0}_record".format(key_prefix), use_container_width=True):
                record_experiment(REGRESSION_HISTORY_KEY, challenge["history_row"])
                st.success(challenge["record_message"])
        return

    st.markdown(
        """
        <div class="challenge-card">
            <div class="challenge-kicker">{0}</div>
            <div class="challenge-title">{1}</div>
            <div class="challenge-task-name">{2}</div>
            <div class="challenge-task-desc">{3}</div>
            <div class="challenge-context"><b>{4}：</b>{5}</div>
        </div>
        """.format(
            challenge["context"],
            challenge["title"],
            challenge["task_title"],
            challenge["task_description"],
            challenge["context_title"],
            challenge["description"],
        ),
        unsafe_allow_html=True,
    )

    st.metric(challenge["score_label"], "{0} / 100".format(challenge["score"]))
    st.progress(challenge["progress"] / 100.0)
    st.markdown(
        '<div class="status-chip status-{0}">{1}</div>'.format(challenge["status_tone"], challenge["status_label"]),
        unsafe_allow_html=True,
    )
    st.caption(challenge["status_note"])

    st.markdown("**目标清单**")
    for item in challenge["goal_items"]:
        status_text = "已达成" if item["passed"] else "待优化"
        st.markdown("- `{0}` {1}".format(status_text, item["label"]))

    st.markdown("**智能反馈**")
    render_feedback_box(challenge)

    st.markdown("**调参建议**")
    suggestions = challenge["suggestions"][:3] if compact else challenge["suggestions"]
    for item in suggestions:
        st.markdown("- {0}".format(item))

    if st.button("记录当前实验", key="{0}_record".format(key_prefix), use_container_width=True):
        record_experiment(REGRESSION_HISTORY_KEY, challenge["history_row"])
        st.success(challenge["record_message"])


def render_metric_cards(metric_values, columns_per_row=3):
    for start in range(0, len(metric_values), columns_per_row):
        row = metric_values[start : start + columns_per_row]
        columns = st.columns(len(row), gap="large")
        for column, metric in zip(columns, row):
            with column:
                st.markdown(
                    """
                    <div class="metric-card">
                        <div class="metric-label">{0}</div>
                        <div class="metric-value">{1}</div>
                    </div>
                    """.format(metric[0], metric[1]),
                    unsafe_allow_html=True,
                )


def record_experiment(history_key, row):
    history = list(st.session_state.get(history_key, []))
    history.insert(0, row)
    st.session_state[history_key] = history[:5]


def render_experiment_history():
    with st.expander("实验记录（最近 5 条）", expanded=False):
        history = st.session_state.get(REGRESSION_HISTORY_KEY, [])
        if not history:
            st.info("还没有记录回归实验，调好参数后可以点击“记录当前实验”。")
        else:
            st.table(history)


def render_visual_output(result, empty_message, image_width="100%"):
    if result is None:
        st.warning(empty_message)
        return
    png_bytes = visual_to_png_bytes(result)
    if png_bytes is not None:
        encoded = base64.b64encode(png_bytes).decode("ascii")
        st.markdown(
            '<div style="width:100%; display:flex; justify-content:center;"><img src="data:image/png;base64,{0}" style="width:{1}; max-width:100%; height:auto; display:block;" /></div>'.format(
                encoded,
                image_width,
            ),
            unsafe_allow_html=True,
        )
        return
    st.warning("主图对象类型异常：{0}".format(type(result).__name__))


def visual_to_png_bytes(result):
    if Figure is not None and isinstance(result, Figure):
        buffer = io.BytesIO()
        result.savefig(buffer, format="png", bbox_inches="tight", dpi=160)
        return buffer.getvalue()
    if isinstance(result, Image.Image):
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        return buffer.getvalue()
    if isinstance(result, np.ndarray):
        array = result
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            image = Image.fromarray(array, mode="L")
        else:
            image = Image.fromarray(array)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    if isinstance(result, (bytes, bytearray)):
        return bytes(result)
    return None


def render_fun_mode_toggle(state_key):
    return st.checkbox(
        "开启小游戏模式",
        key=state_key,
        help="开启后显示数值预测挑战，不影响常规教学图表。",
    )


def build_regression_easter_egg(algorithm_key, dataset_key, params, metrics, x_train, x_test, y_train, y_test, model):
    scenario = get_regression_scenario(dataset_key)
    gap = abs(metrics["train_r2"] - metrics["test_r2"])
    target_span = max(float(np.ptp(y_test)), float(np.std(y_train)), 1e-6)
    rmse_ratio = metrics["test_rmse"] / target_span
    mae_ratio = metrics["test_mae"] / target_span
    fit_state = get_regression_fit_state(metrics, gap, rmse_ratio, mae_ratio)
    suggestions = generate_parameter_suggestions(algorithm_key, params, metrics, gap, rmse_ratio, mae_ratio)
    all_x = np.concatenate([x_train, x_test])
    all_y = np.concatenate([y_train, y_test])
    y_min = float(np.min(all_y))
    y_max = float(np.max(all_y))
    if abs(y_max - y_min) < 1e-6:
        y_max = y_min + 1e-6
    return {
        "title": "时间管理预测实验",
        "scenario_name": scenario["title"],
        "algorithm_label": ALGORITHM_LABELS[algorithm_key],
        "description": "把当前一维输入暂时理解为每日娱乐时间或学习投入时间，把输出理解为综合表现指数。这里使用的是模拟数据映射，只用于观察回归趋势，不代表真实因果结论。",
        "algorithm_note": get_regression_algorithm_clue(algorithm_key),
        "hint": fit_state["hint"],
        "hint_tone": fit_state["tone"],
        "suggestions": suggestions[:2],
        "interpretation": build_regression_interpretation(scenario, metrics, gap, rmse_ratio),
        "fit_label": fit_state["label"],
        "fit_reason": fit_state["reason"],
        "dataset_key": dataset_key,
        "model": model,
        "display_time_min": 0.0,
        "display_time_max": 8.0,
        "model_x_min": float(np.min(all_x)),
        "model_x_max": float(np.max(all_x)),
        "train_x_min": float(np.min(x_train)),
        "train_x_max": float(np.max(x_train)),
        "target_y_min": y_min,
        "target_y_max": y_max,
    }


def render_regression_time_prediction(panel):
    st.markdown("### {0}".format(panel["title"]))
    st.markdown(
        """
        <div class="challenge-card">
            <div class="challenge-kicker">趣味彩蛋模式</div>
            <div class="challenge-task-name">时间投入趋势观察</div>
            <div class="challenge-task-desc">{0}</div>
            <div class="challenge-context"><b>当前回归算法：</b>{1}</div>
        </div>
        """.format(panel["description"], panel["algorithm_label"]),
        unsafe_allow_html=True,
    )
    st.caption("当前场景映射：{0}".format(panel["scenario_name"]))
    hours = st.slider(
        "预测输入时间（小时）",
        min_value=float(panel["display_time_min"]),
        max_value=float(panel["display_time_max"]),
        value=4.0,
        step=0.1,
        key="reg_lab_fun_hours",
    )
    model_x = float(
        np.interp(
            hours,
            [panel["display_time_min"], panel["display_time_max"]],
            [panel["model_x_min"], panel["model_x_max"]],
        )
    )
    X_pred = build_plot_features(panel["dataset_key"], np.array([model_x]))
    raw_prediction = float(panel["model"].predict(X_pred)[0])
    display_prediction = float(
        np.interp(
            raw_prediction,
            [panel["target_y_min"], panel["target_y_max"]],
            [55.0, 95.0],
        )
    )
    st.metric("模型预测综合表现指数", "{0:.1f}".format(display_prediction))
    if panel["train_x_min"] <= model_x <= panel["train_x_max"]:
        st.caption("该输入位于训练样本映射范围内，更适合用来观察模型趋势。")
    else:
        st.warning("该输入已经超出训练样本范围，属于外推预测，结果仅供观察。")
    st.caption("当前模型状态：{0}。{1}".format(panel["fit_label"], panel["fit_reason"]))
    st.caption(panel["algorithm_note"])


def render_fun_hint_box(panel):
    if panel["hint_tone"] == "success":
        st.success(panel["hint"])
    elif panel["hint_tone"] == "warning":
        st.warning(panel["hint"])
    else:
        st.info(panel["hint"])
    if panel.get("suggestions"):
        st.markdown("**简短建议**")
        for item in panel["suggestions"]:
            st.markdown("- {0}".format(item))


def render_regression_regular_hint(panel):
    if panel["fit_label"] == "过拟合":
        st.warning("当前训练与测试表现差距偏大，模型可能开始记住训练噪声。")
    elif panel["fit_label"] == "欠拟合":
        st.info("当前模型还没有充分捕捉主要趋势，可以继续提高表达能力。")
    else:
        st.success("当前训练与测试表现比较接近，模型整体比较稳健。")
    if panel.get("suggestions"):
        st.markdown("**简短建议**")
        for item in panel["suggestions"]:
            st.markdown("- {0}".format(item))


def build_regression_game_context(algorithm_key, dataset_key, params, metrics, x_test, y_test, y_test_pred):
    scenario = get_regression_scenario(dataset_key)
    gap = abs(metrics["train_r2"] - metrics["test_r2"])
    target_span = max(float(np.ptp(y_test)), 1e-6)
    rmse_ratio = metrics["test_rmse"] / target_span
    mae_ratio = metrics["test_mae"] / target_span
    fit_state = get_regression_fit_state(metrics, gap, rmse_ratio, mae_ratio)
    suggestions = generate_parameter_suggestions(algorithm_key, params, metrics, gap, rmse_ratio, mae_ratio)
    y_min = float(np.min(y_test))
    y_max = float(np.max(y_test))
    if abs(y_max - y_min) < 1e-6:
        y_max = y_min + 1e-6
    return {
        "algorithm_key": algorithm_key,
        "algorithm_label": ALGORITHM_LABELS[algorithm_key],
        "x_test": x_test,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "y_min": y_min,
        "y_max": y_max,
        "y_span": y_max - y_min,
        "scenario_name": scenario["title"],
        "x_label": scenario["x_label"],
        "y_label": scenario["y_label"],
        "hint": fit_state["hint"],
        "hint_tone": fit_state["tone"],
        "fit_label": fit_state["label"],
        "fit_reason": fit_state["reason"],
        "suggestions": suggestions[:2],
        "interpretation": build_regression_interpretation(scenario, metrics, gap, rmse_ratio),
        "algorithm_note": get_regression_algorithm_clue(algorithm_key),
    }


def init_regression_game_state(game):
    signature = "{0}|{1}|{2}|{3}".format(
        game["algorithm_key"],
        game["scenario_name"],
        len(game["x_test"]),
        st.session_state["reg_lab_seed"],
    )
    if st.session_state.get("regression_game_signature") != signature:
        st.session_state["regression_game_signature"] = signature
        st.session_state["regression_game_round"] = 0
        st.session_state["regression_game_score"] = 0
        st.session_state["regression_game_index"] = -1
        st.session_state["regression_game_submitted"] = False
        st.session_state["regression_game_feedback"] = ""
        st.session_state["regression_game_guess"] = float(np.mean(game["y_test"]))
        next_regression_game_round(game)
    elif st.session_state.get("regression_game_index", -1) >= len(game["x_test"]):
        next_regression_game_round(game)


def next_regression_game_round(game):
    total = max(len(game["x_test"]), 1)
    round_number = st.session_state.get("regression_game_round", 0) + 1
    rng = np.random.default_rng(st.session_state["reg_lab_seed"] + round_number + st.session_state["reg_lab_view_nonce"] + 17)
    sample_index = int(rng.integers(0, total))
    st.session_state["regression_game_round"] = round_number
    st.session_state["regression_game_index"] = sample_index
    st.session_state["regression_game_submitted"] = False
    st.session_state["regression_game_feedback"] = ""
    st.session_state["regression_game_guess"] = float(game["y_test"][sample_index])


def submit_regression_game_answer(game):
    if st.session_state.get("regression_game_submitted", False):
        return
    sample_index = int(st.session_state["regression_game_index"])
    user_guess = float(st.session_state["regression_game_guess"])
    true_value = float(game["y_test"][sample_index])
    model_value = float(game["y_test_pred"][sample_index])
    user_error = abs(user_guess - true_value)
    model_error = abs(model_value - true_value)
    tolerance_5 = game["y_span"] * 0.05
    tolerance_10 = game["y_span"] * 0.10
    if user_error <= tolerance_5:
        st.session_state["regression_game_score"] += 2
    elif user_error <= tolerance_10:
        st.session_state["regression_game_score"] += 1
    st.session_state["regression_game_submitted"] = True
    st.session_state["regression_game_feedback"] = {
        "user_guess": user_guess,
        "true_value": true_value,
        "model_value": model_value,
        "user_error": user_error,
        "model_error": model_error,
    }


def render_regression_game(game):
    init_regression_game_state(game)
    sample_index = int(st.session_state["regression_game_index"])
    current_x = float(game["x_test"][sample_index])
    st.markdown("### 数值预测挑战")
    st.caption("观察拟合曲线和样本分布，根据给定输入值猜测输出值。提交后系统会显示模型预测值、真实参考值和误差。")
    st.caption("当前得分：{0}".format(st.session_state["regression_game_score"]))
    st.caption("第 {0} 题".format(st.session_state["regression_game_round"]))
    st.markdown("**当前输入值**")
    st.markdown("- {0}：{1:.3f}".format(game["x_label"], current_x))
    st.number_input(
        "请输入你猜测的输出值",
        min_value=float(game["y_min"] - game["y_span"] * 0.2),
        max_value=float(game["y_max"] + game["y_span"] * 0.2),
        step=max(game["y_span"] / 50.0, 0.1),
        key="regression_game_guess",
    )
    if st.button("提交预测", key="regression_game_submit", use_container_width=True, disabled=st.session_state["regression_game_submitted"]):
        submit_regression_game_answer(game)
    if st.session_state["regression_game_submitted"]:
        feedback = st.session_state["regression_game_feedback"]
        st.info(
            "你的预测：{0:.3f}；模型预测：{1:.3f}；真实参考值：{2:.3f}。".format(
                feedback["user_guess"], feedback["model_value"], feedback["true_value"]
            )
        )
        st.caption(
            "你的误差：{0:.3f}；模型误差：{1:.3f}。{2}".format(
                feedback["user_error"],
                feedback["model_error"],
                "这次你的估计更接近真实值。" if feedback["user_error"] < feedback["model_error"] else "这次模型的估计更接近真实值。",
            )
        )
        st.caption(game["algorithm_note"])
    if st.button("下一题", key="regression_game_next", use_container_width=True, disabled=not st.session_state["regression_game_submitted"]):
        next_regression_game_round(game)
        st.rerun()
    if st.button("重置游戏", key="regression_game_reset", use_container_width=True):
        st.session_state["regression_game_score"] = 0
        st.session_state["regression_game_round"] = 0
        st.session_state["regression_game_feedback"] = ""
        next_regression_game_round(game)
        st.rerun()


def render_detective_hint_box(detective):
    if detective["hint_tone"] == "success":
        st.success(detective["hint"])
    elif detective["hint_tone"] == "warning":
        st.warning(detective["hint"])
    else:
        st.info(detective["hint"])
    if detective.get("suggestions"):
        st.markdown("**修一修**")
        for item in detective["suggestions"]:
            st.markdown("- {0}".format(item))


def render_feedback_box(challenge):
    if challenge["tone"] == "success":
        st.success(challenge["feedback"])
    elif challenge["tone"] == "warning":
        st.warning(challenge["feedback"])
    else:
        st.info(challenge["feedback"])
