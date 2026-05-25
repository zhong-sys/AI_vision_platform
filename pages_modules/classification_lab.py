import base64
import io
import numpy as np
import streamlit as st
from PIL import Image

try:
    from matplotlib.figure import Figure
except Exception:
    Figure = None

from pages_modules.classification_data_generation import (
    StandardScaler,
    class_names_from_labels,
    generate_algorithm_dataset,
    get_dataset_label,
    get_dataset_options,
    get_dataset_summary,
    get_default_dataset,
    train_test_split,
)
from pages_modules.classification_metrics import accuracy_score, confusion_matrix, misclassified_count
from pages_modules.classification_model_factory import ALGORITHM_LABELS, build_classifier
from pages_modules.classification_teaching_text import (
    algorithm_overview,
    bottom_conclusion,
    classification_basics_sections,
    dataset_overview,
    live_summary,
    parameter_explanation,
)
from pages_modules.classification_visualization import CLASS_COLORS, hex_to_rgb, render_confusion_matrix_image, render_main_visual


STATE_DEFAULTS = {
    "cls_lab_algorithm": "knn",
    "cls_lab_dataset": "knn_moons",
    "cls_lab_seed": 21,
    "cls_lab_view_nonce": 0,
    "cls_lab_prev_algorithm": "knn",
    "classification_intro_game_index": 0,
    "classification_intro_game_submitted": False,
    "classification_intro_game_answer": "A",
}

CLASSIFICATION_INTRO_QUESTIONS = [
    {
        "title": "垃圾邮件识别",
        "prompt": "某封邮件具有以下特征：",
        "features": ["标题包含“限时优惠”", "链接数量较多", "发件人可信度较低"],
        "options": {"A": "正常邮件", "B": "垃圾邮件"},
        "answer": "B",
        "correct_feedback": "回答正确。这个例子体现了分类任务：根据样本特征判断它属于哪个类别。",
        "wrong_feedback": "还差一点。根据这些特征，该邮件更可能属于垃圾邮件。分类算法会根据样本特征学习类别边界。",
    },
    {
        "title": "水果类型判断",
        "prompt": "某个水果具有以下特征：",
        "features": ["颜色偏红", "甜度较高", "体积较小"],
        "options": {"A": "苹果", "B": "草莓"},
        "answer": "B",
        "correct_feedback": "回答正确。这个例子体现了分类任务：根据样本特征判断它属于哪个类别。",
        "wrong_feedback": "还差一点。根据这些特征，它更可能属于草莓。分类模型就是在学习这种由特征到类别的判断规则。",
    },
    {
        "title": "学习状态判断",
        "prompt": "某位学生具有以下特征：",
        "features": ["连续多次未完成作业", "学习时长偏低", "测验成绩下降明显"],
        "options": {"A": "学习状态稳定", "B": "需要重点关注"},
        "answer": "B",
        "correct_feedback": "回答正确。这个例子体现了分类任务：根据样本特征判断它属于哪个类别。",
        "wrong_feedback": "还差一点。根据这些特征，该学生更可能属于“需要重点关注”一类。分类算法会据此学习不同类别之间的边界。",
    },
]

CLASSIFICATION_HISTORY_KEY = "cls_lab_history"
CLASSIFICATION_DETECTIVE_OPTIONS = {
    "knn": "KNN",
    "svm": "SVM",
    "nb": "朴素贝叶斯",
    "rf": "随机森林",
}

CLASSIFICATION_TASKS = {
    "knn": {
        "title": "近邻投票挑战",
        "description": "围绕 K 值与投票方式调参，判断边界是更稳健还是更容易受局部噪声影响。",
    },
    "svm": {
        "title": "最大间隔挑战",
        "description": "比较 C、gamma 与 kernel 的配合，寻找间隔、边界复杂度和准确率之间的平衡。",
    },
    "nb": {
        "title": "概率法官挑战",
        "description": "观察类别概率与分布假设是否匹配，让概率边界尽量稳定地区分不同样本。",
    },
    "rf": {
        "title": "森林投票挑战",
        "description": "通过多树投票提升稳定性，同时避免树深过大让边界变得过于碎片化。",
    },
}

CLASS_MEANING_MAP = {
    "knn_moons": {0: "上方月牙群体", 1: "下方月牙群体"},
    "knn_circles": {0: "外圈样本", 1: "内圈样本"},
    "knn_local_vote": {0: "更偏左下与中部的局部团块", 1: "更偏上方与右下的局部团块"},
    "svm_linear_margin": {0: "左下侧样本群", 1: "右上侧样本群"},
    "svm_soft_margin": {0: "左下侧样本群", 1: "右上侧样本群"},
    "svm_kernel_curve": {0: "上方弯月群体", 1: "下方弯月群体"},
    "nb_independent_gaussian": {0: "左下概率团", 1: "右上概率团"},
    "nb_overlap_gaussian": {0: "偏左下的概率群", 1: "偏右上的概率群"},
    "rf_block_regions": {0: "常规背景区域", 1: "规则分块高亮区域"},
    "rf_step_regions": {0: "阶梯阈值外区域", 1: "阶梯阈值内区域"},
    "rf_noisy_blocks": {0: "背景主区域", 1: "噪声分块目标区域"},
}


def render_classification_lab():
    ensure_state()
    sync_dataset_with_algorithm()
    inject_page_css()

    algorithm_key = st.session_state["cls_lab_algorithm"]
    dataset_key = st.session_state["cls_lab_dataset"]
    algo_info = algorithm_overview(algorithm_key)
    dataset_info = dataset_overview(dataset_key)

    if st.button("🏠 返回首页", key="cls_back_home", use_container_width=False):
        st.session_state.current_page = "home"
        st.rerun()
    st.markdown("---")
    st.markdown(
        """
        <div class="lab-hero">
            <div class="lab-overline">机器学习 · 分类</div>
            <div class="lab-title">分类算法可视化学习实验室</div>
            <div class="lab-subtitle">先选算法，再切换专属教学数据，再观察图像、参数和结论如何一起变化。</div>
            <div class="lab-badges">
                <span class="lab-badge">当前算法：{0}</span>
                <span class="lab-badge">当前数据：{1}</span>
            </div>
            <div class="lab-summary-grid">
                <div class="lab-summary-card"><b>算法一句话：</b>{2}</div>
                <div class="lab-summary-card"><b>数据一句话：</b>{3}</div>
            </div>
        </div>
        """.format(
            algo_info["title"],
            dataset_info["title"],
            algo_info["headline"],
            dataset_info["summary"],
        ),
        unsafe_allow_html=True,
    )

    with st.expander("分类基础知识", expanded=False):
        for index, (title, body) in enumerate(classification_basics_sections()):
            st.markdown("**{0}. {1}**".format(index + 1, title))
            st.markdown(body)

    render_classification_intro_game()
    st.markdown("---")
    st.markdown("## 进入正式可视化学习")

    control_col, display_col = st.columns([0.84, 2.96], gap="medium")
    with control_col:
        settings = render_control_panel(algorithm_key)

    lab_result = build_lab_result(
        algorithm_key=algorithm_key,
        dataset_key=settings["dataset_key"],
        sample_count=settings["sample_count"],
        noise=settings["noise"],
        test_size=settings["test_size"],
        params=settings["params"],
    )

    with display_col:
        render_display_panel(algorithm_key, settings["params"], lab_result)

    st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)
    render_bottom_panel(algorithm_key, settings["params"], lab_result)


def ensure_state():
    for key, value in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_classification_intro_game():
    questions = CLASSIFICATION_INTRO_QUESTIONS
    index = st.session_state["classification_intro_game_index"] % len(questions)
    question = questions[index]
    option_keys = list(question["options"].keys())
    if st.session_state["classification_intro_game_answer"] not in option_keys:
        st.session_state["classification_intro_game_answer"] = option_keys[0]

    st.markdown("### AI 分类小测验")
    st.markdown(
        '<div class="teach-note">先通过一个生活化判断题理解分类任务：模型会根据样本特征，把对象判定到某个离散类别中。</div>',
        unsafe_allow_html=True,
    )
    st.markdown("**题目场景：{0}**".format(question["title"]))
    st.markdown(question["prompt"])
    for item in question["features"]:
        st.markdown("- {0}".format(item))

    st.radio(
        "请选择类别",
        options=option_keys,
        key="classification_intro_game_answer",
        format_func=lambda value: "{0}. {1}".format(value, question["options"][value]),
        horizontal=True,
    )

    action_col1, action_col2 = st.columns(2, gap="medium")
    if action_col1.button("提交答案", key="classification_intro_game_submit", use_container_width=True):
        st.session_state["classification_intro_game_submitted"] = True
    action_col2.button(
        "再来一题",
        key="classification_intro_game_next",
        use_container_width=True,
        on_click=next_classification_intro_question,
    )

    if st.session_state["classification_intro_game_submitted"]:
        selected = st.session_state["classification_intro_game_answer"]
        if selected == question["answer"]:
            st.success(question["correct_feedback"])
        else:
            st.warning(question["wrong_feedback"])
        st.info("下面可以通过 KNN、SVM、朴素贝叶斯和随机森林，观察机器学习模型如何自动完成分类判断。")



def next_classification_intro_question():
    next_index = (st.session_state["classification_intro_game_index"] + 1) % len(CLASSIFICATION_INTRO_QUESTIONS)
    st.session_state["classification_intro_game_index"] = next_index
    st.session_state["classification_intro_game_submitted"] = False
    st.session_state["classification_intro_game_answer"] = list(CLASSIFICATION_INTRO_QUESTIONS[next_index]["options"].keys())[0]


def sync_dataset_with_algorithm():
    current_algorithm = st.session_state["cls_lab_algorithm"]
    previous_algorithm = st.session_state["cls_lab_prev_algorithm"]
    valid_options = get_dataset_options(current_algorithm)
    if previous_algorithm != current_algorithm or st.session_state["cls_lab_dataset"] not in valid_options:
        st.session_state["cls_lab_dataset"] = get_default_dataset(current_algorithm)
        st.session_state["cls_lab_prev_algorithm"] = current_algorithm
        st.session_state["cls_lab_view_nonce"] += 1


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
                background: rgba(255,255,255,0.85);
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
                padding: 20px 20px;
                box-shadow: 0 8px 20px rgba(15, 91, 158, 0.05);
            }
            .metric-card {
                background: #ffffff;
                border: 1px solid #e4eef7;
                border-radius: 18px;
                padding: 16px 18px;
                box-shadow: 0 8px 18px rgba(15, 91, 158, 0.05);
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
            }
            .challenge-grid {
                display: grid;
                grid-template-columns: 1.4fr 1fr;
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


def render_control_panel(algorithm_key):
    st.markdown("### 操作区")
    algorithm_key = st.radio(
        "选择算法",
        options=list(ALGORITHM_LABELS.keys()),
        format_func=lambda key: ALGORITHM_LABELS[key],
        key="cls_lab_algorithm",
        label_visibility="collapsed",
    )

    valid_datasets = get_dataset_options(algorithm_key)
    dataset_key = st.selectbox(
        "选择教学数据",
        options=valid_datasets,
        format_func=get_dataset_label,
        key="cls_lab_dataset",
    )

    col1, col2 = st.columns(2)
    if col1.button("重新生成数据", use_container_width=True):
        st.session_state["cls_lab_seed"] += 1
    if col2.button("刷新示例视角", use_container_width=True):
        st.session_state["cls_lab_view_nonce"] += 1

    with st.expander("数据设置", expanded=True):
        sample_count = st.slider("样本数量", 120, 320, 220, 20)
        noise = st.slider("噪声强度", 0.00, 0.50, default_noise(algorithm_key), 0.02)
        test_size = st.slider("测试集比例", 0.20, 0.40, 0.28, 0.02)
        st.caption(get_dataset_summary(dataset_key))

    with st.expander("算法参数", expanded=True):
        params = render_parameter_controls(algorithm_key)

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
    }


def render_parameter_controls(algorithm_key):
    if algorithm_key == "knn":
        return {
            "n_neighbors": st.slider("邻居数量", 1, 25, 5, 1),
            "weight_mode": st.selectbox(
                "投票方式",
                options=["uniform", "distance"],
                format_func=lambda mode: "均匀投票" if mode == "uniform" else "距离加权",
            ),
        }
    if algorithm_key == "svm":
        return {
            "C": st.slider("惩罚系数 C", 0.1, 8.0, 1.2, 0.1),
            "kernel": st.selectbox(
                "核函数",
                options=["linear", "rbf", "poly"],
                format_func=lambda item: {"linear": "线性核", "rbf": "RBF 核", "poly": "多项式核"}[item],
            ),
            "gamma": st.slider("核函数影响范围", 0.1, 4.0, 1.0, 0.1),
        }
    if algorithm_key == "nb":
        return {
            "alpha": st.slider("平滑参数", 0.001, 1.000, 0.060, 0.001),
        }
    return {
        "n_estimators": st.slider("树的数量", 3, 31, 11, 2),
        "max_depth": st.slider("最大深度", 1, 10, 4, 1),
        "min_samples_split": st.slider("分裂所需最少样本数", 2, 20, 4, 1),
    }


def default_noise(algorithm_key):
    mapping = {
        "knn": 0.18,
        "svm": 0.16,
        "nb": 0.12,
        "rf": 0.14,
    }
    return mapping[algorithm_key]


def build_lab_result(algorithm_key, dataset_key, sample_count, noise, test_size, params):
    X, y = generate_algorithm_dataset(
        dataset_key=dataset_key,
        n_samples=sample_count,
        noise=noise,
        random_state=st.session_state["cls_lab_seed"],
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=st.session_state["cls_lab_seed"],
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_classifier(algorithm_key, params)
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    extras = build_algorithm_extras(
        algorithm_key=algorithm_key,
        model=model,
        scaler=scaler,
        X_train=X_train,
        X_train_scaled=X_train_scaled,
        y_train=y_train,
        X_test=X_test,
        X_test_scaled=X_test_scaled,
        y_test=y_test,
        y_test_pred=y_test_pred,
    )

    visual_context = {
        "model": model,
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "visual_title": ALGORITHM_LABELS[algorithm_key],
    }
    visual_context.update(extras)

    class_names = class_names_from_labels(y)


    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "model": model,
        "scaler": scaler,
        "params": params,
        "extras": extras,
        "visual_context": visual_context,
        "train_acc": accuracy_score(y_train, y_train_pred),
        "test_acc": accuracy_score(y_test, y_test_pred),
        "confusion": confusion_matrix(y_test, y_test_pred),
        "misclassified": misclassified_count(y_test, y_test_pred),
        "class_names": class_names,
        "dataset_key": dataset_key,
        "algorithm_key": algorithm_key,
    }


def build_algorithm_extras(algorithm_key, model, scaler, X_train, X_train_scaled, y_train, X_test, X_test_scaled, y_test, y_test_pred):
    nonce = st.session_state["cls_lab_view_nonce"]
    sample_index = nonce % len(X_test)
    query_point = X_test[sample_index]
    query_scaled = X_test_scaled[sample_index]

    if algorithm_key == "knn":
        report = model.get_neighbor_report(query_scaled)
        return {
            "query_point": query_point,
            "query_true": int(y_test[sample_index]),
            "query_pred": int(report["predicted_label"]),
            "neighbor_points": X_train[report["indices"]],
            "vote_scores": report["vote_scores"],
        }

    if algorithm_key == "svm":
        support_vectors_scaled = model.support_vectors()
        margin_width = model.linear_margin_width()
        margin_note = (
            "当前线性间隔宽度约为 {0:.2f}。".format(margin_width)
            if margin_width is not None
            else "当前核函数更适合观察边界形状变化，而不是固定间隔宽度。"
        )
        return {
            "support_vectors": support_vectors_scaled,
            "support_vector_count": model.support_vector_count(),
            "margin_width": margin_width,
            "margin_note": margin_note,
        }

    if algorithm_key == "nb":
        query_proba = model.predict_proba(query_scaled.reshape(1, -1))[0]
        distribution_boxes = []
        for class_id in model.classes_:
            center_scaled = model.theta_[class_id]
            center = scaler.inverse_transform(center_scaled)
            std = np.sqrt(model.var_[class_id]) * scaler.scale_
            distribution_boxes.append(
                {
                    "center": center,
                    "std": std,
                    "color": hex_to_rgb(CLASS_COLORS[int(class_id)]),
                }
            )
        return {
            "query_point": query_point,
            "query_proba": query_proba,
            "distribution_boxes": distribution_boxes,
        }

    single_tree_pred = model.single_tree_predict(X_test_scaled)
    return {
        "tree_forest_disagree": int((single_tree_pred != y_test_pred).sum()),
    }


def render_display_panel(algorithm_key, params, lab_result):
    algo_info = algorithm_overview(algorithm_key)
    st.markdown("### 决策边界与样本分布")
    main_visual = render_main_visual(algorithm_key, lab_result["visual_context"])
    render_visual_output(main_visual, "当前主图未成功生成，请调整参数后重试。")
    st.caption(build_class_meaning_note(lab_result["dataset_key"]))
    with st.expander("图像观察与调参提示", expanded=False):
        st.markdown("**边界观察**")
        st.info(live_summary(algorithm_key, params))
        st.markdown("**调参提示**")
        st.info(parameter_explanation(algorithm_key, params, lab_result["extras"]))
    with st.expander("教学提示", expanded=False):
        st.markdown("**一句话速览**")
        st.markdown(algo_info["headline"])
        st.markdown("**算法原理**")
        st.markdown(algo_info["principle"])
        st.markdown("**适用场景**")
        st.markdown(algo_info["fit"])


def render_bottom_panel(algorithm_key, params, lab_result):
    st.markdown("### 指标与结果分析")
    gap = abs(lab_result["train_acc"] - lab_result["test_acc"])
    metrics = [
        ("训练准确率", "{0:.1f}%".format(lab_result["train_acc"] * 100)),
        ("测试准确率", "{0:.1f}%".format(lab_result["test_acc"] * 100)),
        ("训练/测试差距", "{0:.1f}%".format(gap * 100)),
        ("样本总数", str(len(lab_result["X"]))),
        ("错分样本", str(lab_result["misclassified"])),
    ]
    render_metric_cards(metrics, columns_per_row=3)

    st.markdown("#### 混淆矩阵")
    confusion_image = render_confusion_matrix_image(
        lab_result["confusion"], lab_result["class_names"], "测试集混淆矩阵"
    )
    render_visual_output(
        confusion_image,
        "当前混淆矩阵未成功生成。",
        image_width="min(100%, 760px)",
    )
    st.caption("混淆矩阵中的类别编号说明：{0}".format(build_class_meaning_note(lab_result["dataset_key"], prefix="")))
    st.markdown(bottom_conclusion(algorithm_key, st.session_state["cls_lab_dataset"], lab_result["train_acc"], lab_result["test_acc"], lab_result["misclassified"]))
    st.markdown("#### 调参与结果提示")
    render_classification_regular_hint(algorithm_key, params, lab_result, gap)
    st.markdown("#### 边界解释")
    st.markdown(build_classification_interpretation(algorithm_key, lab_result["test_acc"], gap, lab_result["misclassified"]))
    info = algorithm_overview(algorithm_key)
    with st.expander("查看算法补充说明", expanded=False):
        st.markdown("**算法原理**")
        st.markdown(info["principle"])
        st.markdown("**优点 / 局限**")
        st.markdown("- 优点：{0}".format(info["pros"]))
        st.markdown("- 局限：{0}".format(info["cons"]))
        st.markdown("**适用场景**")
        st.markdown(info["fit"])


def build_classification_detective_game(algorithm_key, dataset_key, params, y_test, y_test_pred, y_train_pred, y_train):
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    gap = abs(train_acc - test_acc)
    misclassified = misclassified_count(y_test, y_test_pred)
    mis_ratio = misclassified / max(len(y_test), 1)
    explanation = get_classification_algorithm_clue(algorithm_key)
    suggestions = generate_parameter_suggestions(algorithm_key, params, test_acc, gap, mis_ratio)
    hint = generate_classification_detective_hint(algorithm_key, params, train_acc, test_acc, gap, mis_ratio, misclassified)
    return {
        "title": "算法侦探：谁画出了这条边界？",
        "description": "观察当前决策边界的形状，猜一猜它最像哪种分类算法生成的。完成判断后，再调节参数验证你的想法。",
        "context": "{0} · {1}".format(ALGORITHM_LABELS[algorithm_key], get_dataset_label(dataset_key)),
        "correct_answer": CLASSIFICATION_DETECTIVE_OPTIONS[algorithm_key],
        "options": list(CLASSIFICATION_DETECTIVE_OPTIONS.values()),
        "explanation": explanation,
        "hint": hint["message"],
        "hint_tone": hint["tone"],
        "suggestions": suggestions[:2],
        "interpretation": build_classification_interpretation(algorithm_key, test_acc, gap, misclassified),
        "next_step": "继续尝试调整当前算法的关键参数，再比较边界是否变得更平滑、更稳定，或更容易受局部样本影响。",
    }


def get_classification_algorithm_clue(algorithm_key):
    clues = {
        "knn": "KNN 的边界通常受局部邻居影响，K 值较小时边界更弯曲，K 值较大时边界更平滑。",
        "svm": "SVM 重点寻找分隔间隔，使用 RBF 核时会形成较平滑的非线性边界。",
        "nb": "朴素贝叶斯基于概率分布进行判断，边界通常体现不同类别概率区域的变化。",
        "rf": "随机森林由多棵树投票形成结果，边界常呈现局部块状或阶梯状特征。",
    }
    return clues[algorithm_key]


def generate_classification_detective_hint(algorithm_key, params, train_acc, test_acc, gap, mis_ratio, misclassified):
    suggestions = generate_parameter_suggestions(algorithm_key, params, test_acc, gap, mis_ratio)
    if gap >= 0.12 and train_acc > test_acc:
        return {"tone": "warning", "message": "侦探提示：训练与测试差距较大，当前边界可能过拟合。{0}".format(suggestions[0])}
    if train_acc <= 0.78 and test_acc <= 0.78:
        return {"tone": "info", "message": "侦探提示：训练集和测试集都偏低，模型可能欠拟合。{0}".format(suggestions[0])}
    if mis_ratio >= 0.18:
        return {"tone": "warning", "message": "侦探提示：当前仍有较多错分样本。{0}".format(suggestions[0])}
    return {"tone": "success", "message": "侦探提示：当前边界较稳定，可以继续微调参数，观察边界是更平滑还是更贴近局部样本。"}


def render_algorithm_detective_game(game, key_prefix):
    st.markdown("### {0}".format(game["title"]))
    st.markdown(
        """
        <div class="challenge-card">
            <div class="challenge-kicker">问题驱动式小游戏</div>
            <div class="challenge-task-name">先猜算法，再调参验证</div>
            <div class="challenge-task-desc">{0}</div>
            <div class="challenge-context"><b>当前观察对象：</b>{1}</div>
        </div>
        """.format(game["description"], game["context"]),
        unsafe_allow_html=True,
    )
    guess = st.radio(
        "你觉得当前边界最像哪种算法？",
        options=game["options"],
        key="{0}_detective_choice".format(key_prefix),
        label_visibility="collapsed",
    )
    if guess == game["correct_answer"]:
        st.success("判断正确。当前边界确实由 {0} 生成。".format(game["correct_answer"]))
    else:
        st.info("还差一点。当前实际算法是 {0}，可以继续观察它的边界特征。".format(game["correct_answer"]))
    st.markdown("**边界解释**")
    st.caption(game["explanation"])
    st.markdown("**继续观察**")
    st.caption(game["next_step"])


def build_classification_challenge(algorithm_key, dataset_key, params, y_test, y_test_pred, y_train_pred, y_train):
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    gap = abs(train_acc - test_acc)
    misclassified = misclassified_count(y_test, y_test_pred)
    test_count = max(len(y_test), 1)
    mis_ratio = misclassified / test_count
    score = calculate_classification_challenge_score(algorithm_key, params, train_acc, test_acc, gap, mis_ratio)
    task_info = get_algorithm_task_info(algorithm_key)
    goal_items = evaluate_challenge_goals(algorithm_key, params, test_acc, gap, mis_ratio)
    progress = int(round(100 * sum(1 for item in goal_items if item["passed"]) / max(len(goal_items), 1)))
    status = render_progress_status(score, progress, goal_items)
    suggestions = generate_parameter_suggestions(algorithm_key, params, test_acc, gap, mis_ratio)
    feedback = generate_classification_feedback(algorithm_key, params, train_acc, test_acc, gap, mis_ratio, misclassified)
    interpretation = build_classification_interpretation(algorithm_key, test_acc, gap, misclassified)

    return {
        "title": "边界大师挑战",
        "description": "请调节分类算法参数，在保证测试集表现的同时，让决策边界既清晰又不过度复杂。",
        "task_title": task_info["title"],
        "task_description": task_info["description"],
        "context_title": "当前任务",
        "context": "{0} · {1}".format(ALGORITHM_LABELS[algorithm_key], get_dataset_label(dataset_key)),
        "goals": [item["label"] for item in goal_items],
        "goal_items": goal_items,
        "progress": progress,
        "status_label": status["label"],
        "status_note": status["note"],
        "status_tone": status["tone"],
        "score": score,
        "score_label": "当前挑战得分",
        "feedback": feedback["message"],
        "tone": feedback["tone"],
        "suggestions": suggestions,
        "interpretation": interpretation,
        "history_row": {
            "算法": ALGORITHM_LABELS[algorithm_key],
            "数据集": get_dataset_label(dataset_key),
            "测试准确率": "{0:.1f}%".format(test_acc * 100),
            "泛化差距": "{0:.1f}%".format(gap * 100),
            "挑战得分": score,
            "结论": status["label"],
        },
        "record_message": "已记录当前分类实验。",
    }


def get_algorithm_task_info(algorithm_key):
    return CLASSIFICATION_TASKS[algorithm_key]


def evaluate_challenge_goals(algorithm_key, params, test_acc, gap, mis_ratio):
    goal_items = [
        {"label": "测试准确率达到 85% 以上", "passed": test_acc >= 0.85},
        {"label": "训练与测试差距控制在 10% 内", "passed": gap <= 0.10},
        {"label": "错分样本比例不超过 15%", "passed": mis_ratio <= 0.15},
    ]

    if algorithm_key == "knn":
        goal_items.append({"label": "K 值保持在稳健区间", "passed": 3 <= params["n_neighbors"] <= 12})
    elif algorithm_key == "svm":
        goal_items.append(
            {
                "label": "间隔与边界复杂度保持平衡",
                "passed": not (params["kernel"] != "linear" and params["gamma"] >= 2.2 and params["C"] >= 5.0),
            }
        )
    elif algorithm_key == "rf":
        goal_items.append(
            {
                "label": "树深不过深，投票保持稳定",
                "passed": params["max_depth"] <= 6 and params["n_estimators"] >= 9,
            }
        )
    else:
        goal_items.append({"label": "概率平滑参数保持稳定", "passed": params["alpha"] >= 0.01})
    return goal_items


def render_goal_checklist(goal_items):
    rows = []
    for item in goal_items:
        css_class = "goal-pass" if item["passed"] else "goal-wait"
        prefix = "✓" if item["passed"] else "○"
        rows.append('<div class="goal-item {0}">{1} {2}</div>'.format(css_class, prefix, item["label"]))
    return '<div class="goal-checklist">{0}</div>'.format("".join(rows))


def render_progress_status(score, progress, goal_items):
    passed_count = sum(1 for item in goal_items if item["passed"])
    total_count = max(len(goal_items), 1)
    if score >= 85 and passed_count >= total_count - 1:
        return {"label": "已通关", "tone": "success", "note": "已完成 {0}/{1} 项目标".format(passed_count, total_count)}
    if score >= 70 or passed_count >= max(2, total_count - 2):
        return {"label": "接近通关", "tone": "warning", "note": "已完成 {0}/{1} 项目标".format(passed_count, total_count)}
    return {"label": "继续挑战", "tone": "info", "note": "已完成 {0}/{1} 项目标".format(passed_count, total_count)}


def generate_parameter_suggestions(algorithm_key, params, test_acc, gap, mis_ratio):
    suggestions = []
    if gap >= 0.12:
        if algorithm_key == "knn":
            suggestions.append("适当增大 K 值，削弱局部噪声对边界的影响。")
        elif algorithm_key == "svm":
            suggestions.append("先降低 C 或 gamma，让边界不要过度贴合训练样本。")
        elif algorithm_key == "rf":
            suggestions.append("优先降低 max_depth，减少单棵树记忆训练样本的程度。")
        else:
            suggestions.append("适当增大平滑参数，观察概率边界是否变得更稳定。")
    if mis_ratio >= 0.15:
        if algorithm_key == "knn":
            suggestions.append("比较均匀投票与距离加权，观察错分样本是否减少。")
        elif algorithm_key == "svm":
            suggestions.append("尝试切换 kernel，比较边界形状与错分样本的变化。")
        elif algorithm_key == "rf":
            suggestions.append("增加 n_estimators，观察投票结果是否更稳定。")
        else:
            suggestions.append("结合数据分布检查概率假设是否匹配当前样本结构。")
    if test_acc < 0.85:
        if algorithm_key == "knn":
            suggestions.append("若边界过于平滑，可适当减小 K 值重新观察。")
        elif algorithm_key == "svm":
            suggestions.append("联动调整 C 与 gamma，寻找更合适的间隔与复杂度。")
        elif algorithm_key == "rf":
            suggestions.append("在控制树深的前提下微调 n_estimators，提高整体稳定性。")
        else:
            suggestions.append("观察错分集中在哪类样本，再判断是否需要改变平滑强度。")

    if not suggestions:
        suggestions.append("当前参数组合已经比较稳健，可继续微调并观察边界是否更简洁。")
    return suggestions[:3]


def calculate_classification_challenge_score(algorithm_key, params, train_acc, test_acc, gap, mis_ratio):
    score = test_acc * 68.0
    score += max(0.0, 18.0 * (1.0 - min(gap / 0.18, 1.0)))
    score += max(0.0, 10.0 * (1.0 - min(mis_ratio / 0.30, 1.0)))
    score += max(0.0, 4.0 * min(train_acc, test_acc))

    complexity_penalty = 0.0
    if algorithm_key == "knn":
        if params["n_neighbors"] <= 2:
            complexity_penalty += 8.0
        elif params["n_neighbors"] >= 14:
            complexity_penalty += 4.0
    elif algorithm_key == "svm":
        if params["kernel"] != "linear" and params["gamma"] >= 2.2:
            complexity_penalty += 6.0
        if params["C"] >= 5.0:
            complexity_penalty += 4.0
    elif algorithm_key == "rf":
        complexity_penalty += max(0, params["max_depth"] - 6) * 1.4
        complexity_penalty += max(0, params["n_estimators"] - 21) * 0.10
    elif algorithm_key == "nb" and params["alpha"] < 0.01:
        complexity_penalty += 2.0

    return int(np.clip(round(score - complexity_penalty), 0, 100))


def generate_classification_feedback(algorithm_key, params, train_acc, test_acc, gap, mis_ratio, misclassified):
    suggestions = generate_parameter_suggestions(algorithm_key, params, test_acc, gap, mis_ratio)
    if test_acc >= 0.85 and gap <= 0.07:
        return {"tone": "success", "message": "训练集与测试集表现都较稳定，当前边界的泛化能力较好。"}
    if gap >= 0.12 and train_acc > test_acc:
        return {"tone": "warning", "message": "训练准确率明显高于测试准确率，模型可能过拟合。{0}".format(suggestions[0])}
    if train_acc <= 0.78 and test_acc <= 0.78:
        return {"tone": "info", "message": "训练集和测试集准确率都偏低，模型可能欠拟合。{0}".format(suggestions[0])}
    if mis_ratio >= 0.18:
        return {"tone": "warning", "message": "测试集中仍有 {0} 个样本被错分，边界还不够稳定。{1}".format(misclassified, suggestions[0])}
    return {"tone": "info", "message": "模型已经抓住主要分类规律，可以继续比较准确率、错分样本和边界复杂度的平衡。"}


def build_classification_interpretation(algorithm_key, test_acc, gap, misclassified):
    if gap >= 0.12:
        return "模型在训练样本上学得较充分，但对新样本的边界泛化仍不稳定，需要警惕过拟合。"
    if test_acc >= 0.88 and misclassified <= 3:
        return "模型当前能够较好地区分不同类别样本，边界清晰，且错分样本较少。"
    if algorithm_key == "knn":
        return "KNN 的局部邻域划分已经起效，但边界仍可能受到个别邻居样本影响。"
    if algorithm_key == "svm":
        return "SVM 已形成较清晰的分类间隔，可继续比较不同核函数下边界的平滑程度。"
    if algorithm_key == "rf":
        return "随机森林已经整合多棵树的判断，但仍需关注树深是否让边界变得过于复杂。"
    return "朴素贝叶斯已经形成概率分界，但效果仍受特征分布假设是否匹配影响。"


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
                record_experiment(CLASSIFICATION_HISTORY_KEY, challenge["history_row"])
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
        record_experiment(CLASSIFICATION_HISTORY_KEY, challenge["history_row"])
        st.success(challenge["record_message"])


def record_experiment(history_key, row):
    history = list(st.session_state.get(history_key, []))
    history.insert(0, row)
    st.session_state[history_key] = history[:5]


def render_experiment_history():
    with st.expander("实验记录（最近 5 条）", expanded=False):
        history = st.session_state.get(CLASSIFICATION_HISTORY_KEY, [])
        if not history:
            st.info("还没有记录分类实验，调好参数后可以点击“记录当前实验”。")
        else:
            st.table(history)


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


def build_class_meaning_note(dataset_key, prefix="当前数据类别含义："):
    mapping = CLASS_MEANING_MAP.get(dataset_key, {0: "类别 0 对应的一类样本", 1: "类别 1 对应的另一类样本"})
    note = "类别 0 = {0}；类别 1 = {1}".format(mapping.get(0, "第一类样本"), mapping.get(1, "第二类样本"))
    return "{0}{1}".format(prefix, note) if prefix else note


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
        help="开启后显示人工分类挑战，不影响常规教学图表。",
    )


def trigger_balloons_once(session_key):
    if not st.session_state.get(session_key, False):
        st.session_state[session_key] = True
        st.balloons()


def build_classification_easter_egg(algorithm_key, dataset_key, params, y_test, y_test_pred, y_train_pred, y_train):
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    gap = abs(train_acc - test_acc)
    misclassified = misclassified_count(y_test, y_test_pred)
    mis_ratio = misclassified / max(len(y_test), 1)
    task = get_classification_easter_egg_task(algorithm_key, params, test_acc, gap)
    hint = generate_classification_easter_egg_hint(algorithm_key, params, train_acc, test_acc, gap, mis_ratio)
    suggestions = generate_parameter_suggestions(algorithm_key, params, test_acc, gap, mis_ratio)
    return {
        "title": "AI 调参挑战",
        "context": "{0} / {1}".format(ALGORITHM_LABELS[algorithm_key], get_dataset_label(dataset_key)),
        "task_title": task["title"],
        "description": task["description"],
        "status_message": task["status_message"],
        "success": task["success"],
        "success_key": task.get("success_key"),
        "algorithm_note": task["algorithm_note"],
        "hint": hint["message"],
        "hint_tone": hint["tone"],
        "suggestions": suggestions[:2],
        "interpretation": build_classification_interpretation(algorithm_key, test_acc, gap, misclassified),
        "next_step": task["next_step"],
    }


def get_classification_easter_egg_task(algorithm_key, params, test_acc, gap):
    if algorithm_key == "knn":
        return {
            "title": "平滑边界挑战",
            "description": "请调节 K 值，观察决策边界如何从局部弯曲变得更加平滑。目标是在保持较好测试准确率的同时，避免边界过度破碎。",
            "status_message": "建议测试准确率保持在 90% 以上，同时让训练与测试差距不要过大，并避免 K 值过小导致边界碎片化。",
            "success": check_classification_easter_egg_success(algorithm_key, params, test_acc, gap),
            "success_key": "cls_lab_fun_success_knn",
            "algorithm_note": "KNN 的边界通常受局部邻居影响，K 值较小时边界更弯曲，K 值较大时边界更平滑。",
            "next_step": "继续微调 K 值与投票方式，再比较边界是否既保留结构又不过度破碎。",
        }
    if algorithm_key == "svm":
        return {
            "title": "非线性边界挑战",
            "description": "请将核函数切换为 RBF，并调节 C 与 gamma，观察弯曲边界如何包围非线性样本结构。目标是在非线性数据上获得较好的测试准确率。",
            "status_message": "建议使用 RBF 核，并让测试准确率达到 90% 以上，同时避免 C 与 gamma 过大导致边界过度收缩。",
            "success": check_classification_easter_egg_success(algorithm_key, params, test_acc, gap),
            "success_key": "cls_lab_fun_success_svm",
            "algorithm_note": "SVM 重点寻找分隔间隔，使用 RBF 核时会形成较平滑的非线性边界。",
            "next_step": "继续联动观察 kernel、C 和 gamma 的变化，判断边界是在更好拟合结构还是开始记住噪声。",
        }
    if algorithm_key == "nb":
        return {
            "title": "概率边界观察",
            "description": "朴素贝叶斯更适合观察类别概率区域变化。本彩蛋挑战主要面向 KNN 和 SVM，当前算法可继续观察概率边界与数据分布是否匹配。",
            "status_message": "当前算法不需要达成额外条件，重点是观察类别概率区域如何随数据分布变化。",
            "success": False,
            "algorithm_note": "朴素贝叶斯基于概率分布进行判断，边界通常体现不同类别概率区域的变化。",
            "next_step": "继续观察概率边界与样本分布是否匹配，再判断概率假设在当前数据上是否合适。",
        }
    return {
        "title": "森林投票观察",
        "description": "随机森林适合观察多棵树投票形成的块状边界。本彩蛋挑战主要面向 KNN 和 SVM，当前算法可继续观察树深度和树数量对边界稳定性的影响。",
        "status_message": "当前算法不需要达成额外条件，重点是比较树数量与树深度改变后边界是否更稳定。",
        "success": False,
        "algorithm_note": "随机森林由多棵树投票形成结果，边界常呈现局部块状或阶梯状特征。",
        "next_step": "继续观察树深度和树数量变化后，块状边界是更稳定还是更容易出现局部噪声。",
    }


def check_classification_easter_egg_success(algorithm_key, params, test_acc, gap):
    if algorithm_key == "knn":
        return test_acc >= 0.90 and gap <= 0.10 and params["n_neighbors"] >= 4
    if algorithm_key == "svm":
        return (
            params["kernel"] == "rbf"
            and test_acc >= 0.90
            and gap <= 0.10
            and params["C"] <= 5.0
            and params["gamma"] <= 2.0
        )
    return False


def generate_classification_easter_egg_hint(algorithm_key, params, train_acc, test_acc, gap, mis_ratio):
    suggestions = generate_parameter_suggestions(algorithm_key, params, test_acc, gap, mis_ratio)
    if gap >= 0.12 and train_acc > test_acc:
        return {"tone": "warning", "message": "彩蛋提示：训练与测试差距偏大，当前边界可能开始过拟合。{0}".format(suggestions[0])}
    if train_acc <= 0.78 and test_acc <= 0.78:
        return {"tone": "info", "message": "彩蛋提示：训练与测试表现都偏低，模型可能还在欠拟合。{0}".format(suggestions[0])}
    if mis_ratio >= 0.18:
        return {"tone": "warning", "message": "彩蛋提示：当前仍有较多错分样本。{0}".format(suggestions[0])}
    return {"tone": "success", "message": "彩蛋提示：当前边界已经比较稳定，可以继续微调参数，比较平滑度与泛化能力之间的平衡。"}


def render_classification_easter_egg(panel):
    st.markdown("### {0}".format(panel["title"]))
    st.markdown(
        """
        <div class="challenge-card">
            <div class="challenge-kicker">趣味彩蛋模式</div>
            <div class="challenge-task-name">{0}</div>
            <div class="challenge-task-desc">{1}</div>
            <div class="challenge-context"><b>当前观察对象：</b>{2}</div>
        </div>
        """.format(panel["task_title"], panel["description"], panel["context"]),
        unsafe_allow_html=True,
    )
    st.caption(panel["status_message"])
    st.caption(panel["algorithm_note"])
    if panel["success"] and panel.get("success_key"):
        trigger_balloons_once(panel["success_key"])
        st.success("彩蛋达成：当前参数组合已经比较符合这个调参任务。")
    else:
        render_fun_hint_box(panel)
    st.caption(panel["next_step"])


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


def render_classification_regular_hint(algorithm_key, params, lab_result, gap):
    mis_ratio = lab_result["misclassified"] / max(len(lab_result["y_test"]), 1)
    suggestions = generate_parameter_suggestions(algorithm_key, params, lab_result["test_acc"], gap, mis_ratio)
    if gap >= 0.12 and lab_result["train_acc"] > lab_result["test_acc"]:
        st.warning("当前训练与测试差距偏大，模型可能开始过拟合。")
    elif lab_result["train_acc"] <= 0.78 and lab_result["test_acc"] <= 0.78:
        st.info("当前训练与测试表现都偏低，模型可能还在欠拟合。")
    else:
        st.success("当前边界表现比较稳定，可以继续观察参数变化带来的边界差异。")
    st.markdown("**简短建议**")
    for item in suggestions[:2]:
        st.markdown("- {0}".format(item))


def build_classification_game_context(algorithm_key, dataset_key, X_test, y_test, y_test_pred, class_names):
    return {
        "algorithm_key": algorithm_key,
        "algorithm_label": ALGORITHM_LABELS[algorithm_key],
        "dataset_label": get_dataset_label(dataset_key),
        "X_test": X_test,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "class_names": [str(name) for name in class_names],
        "explanation": get_classification_game_explanation(algorithm_key),
    }


def get_classification_game_explanation(algorithm_key):
    explanations = {
        "knn": "当前模型会参考附近 K 个训练样本的类别投票，因此高亮点附近的邻居分布会影响预测结果。",
        "svm": "当前模型根据样本位于决策边界的哪一侧进行分类，靠近边界时判断不确定性更高。",
        "nb": "当前模型基于类别概率进行判断，样本落入概率较高的区域时会被分到对应类别。",
        "rf": "当前模型由多棵决策树投票得到结果，因此局部区域的划分会影响最终类别。",
    }
    return explanations[algorithm_key]


def init_classification_game_state(game):
    signature = "{0}|{1}|{2}|{3}".format(
        game["algorithm_key"],
        game["dataset_label"],
        len(game["y_test"]),
        st.session_state["cls_lab_seed"],
    )
    if st.session_state.get("classification_game_signature") != signature:
        st.session_state["classification_game_signature"] = signature
        st.session_state["classification_game_round"] = 0
        st.session_state["classification_game_score"] = 0
        st.session_state["classification_game_sample_index"] = -1
        st.session_state["classification_game_submitted"] = False
        st.session_state["classification_game_feedback"] = ""
        st.session_state["classification_game_choice"] = int(np.min(game["y_test"]))
        next_classification_game_round(game)
    elif st.session_state.get("classification_game_sample_index", -1) >= len(game["y_test"]):
        next_classification_game_round(game)


def next_classification_game_round(game):
    total = max(len(game["y_test"]), 1)
    round_number = st.session_state.get("classification_game_round", 0) + 1
    rng = np.random.default_rng(st.session_state["cls_lab_seed"] + round_number + st.session_state["cls_lab_view_nonce"] + 11)
    sample_index = int(rng.integers(0, total))
    st.session_state["classification_game_round"] = round_number
    st.session_state["classification_game_sample_index"] = sample_index
    st.session_state["classification_game_submitted"] = False
    st.session_state["classification_game_feedback"] = ""
    st.session_state["classification_game_choice"] = int(np.min(game["y_test"]))


def submit_classification_game_answer(game):
    if st.session_state.get("classification_game_submitted", False):
        return
    sample_index = int(st.session_state["classification_game_sample_index"])
    user_choice = int(st.session_state["classification_game_choice"])
    true_label = int(game["y_test"][sample_index])
    model_label = int(game["y_test_pred"][sample_index])
    user_correct = user_choice == true_label
    model_correct = model_label == true_label
    if user_correct:
        st.session_state["classification_game_score"] += 1
    st.session_state["classification_game_submitted"] = True
    st.session_state["classification_game_feedback"] = {
        "user_choice": user_choice,
        "true_label": true_label,
        "model_label": model_label,
        "user_correct": user_correct,
        "model_correct": model_correct,
    }


def render_classification_game(game):
    init_classification_game_state(game)
    sample_index = int(st.session_state["classification_game_sample_index"])
    point = game["X_test"][sample_index]
    answered = max(st.session_state["classification_game_round"] - (0 if st.session_state["classification_game_submitted"] else 1), 0)
    st.markdown("### 人工分类挑战")
    st.caption("观察主图中的决策边界和样本分布，判断当前样本更像属于哪一类。提交后会公布真实类别、模型预测类别和分类依据。")
    st.caption("当前得分：{0} / {1}".format(st.session_state["classification_game_score"], answered))
    st.caption("第 {0} 题".format(st.session_state["classification_game_round"]))
    st.markdown("**当前样本特征**")
    st.markdown("- 特征 1：{0:.3f}".format(float(point[0])))
    st.markdown("- 特征 2：{0:.3f}".format(float(point[1])))
    label_values = sorted(int(value) for value in np.unique(game["y_test"]))
    st.radio(
        "请选择类别",
        options=label_values,
        key="classification_game_choice",
        format_func=lambda value: "类别 {0}".format(value),
    )
    if st.button("提交判断", key="classification_game_submit", use_container_width=True, disabled=st.session_state["classification_game_submitted"]):
        submit_classification_game_answer(game)
    if st.session_state["classification_game_submitted"]:
        feedback = st.session_state["classification_game_feedback"]
        if feedback["user_correct"]:
            st.success("判断正确。你把这个样本分到了真实类别 {0}。".format(feedback["true_label"]))
        else:
            st.warning("这次判断没有命中。真实类别是 {0}，你选择的是 {1}。".format(feedback["true_label"], feedback["user_choice"]))
        if feedback["model_correct"]:
            st.caption("当前模型预测类别：{0}，模型本轮预测正确。".format(feedback["model_label"]))
        else:
            st.caption("当前模型预测类别：{0}，模型本轮也没有分对这个样本。".format(feedback["model_label"]))
        st.caption(game["explanation"])
    if st.button("下一题", key="classification_game_next", use_container_width=True, disabled=not st.session_state["classification_game_submitted"]):
        next_classification_game_round(game)
        st.rerun()
    if st.button("重置游戏", key="classification_game_reset", use_container_width=True):
        st.session_state["classification_game_score"] = 0
        st.session_state["classification_game_round"] = 0
        st.session_state["classification_game_feedback"] = ""
        next_classification_game_round(game)
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
