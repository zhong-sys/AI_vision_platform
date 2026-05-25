import base64
import io
import numpy as np
import streamlit as st
from PIL import Image

try:
    from matplotlib.figure import Figure
except Exception:
    Figure = None

from pages_modules.clustering_data import (
    generate_clustering_dataset,
    get_dataset_label,
    get_dataset_options,
    get_dataset_summary,
    get_default_dataset,
)
from pages_modules.clustering_metrics import build_clustering_metrics
from pages_modules.clustering_models import ALGORITHM_LABELS, build_clusterer
from pages_modules.clustering_text import (
    algorithm_overview,
    bottom_conclusion,
    clustering_basics_sections,
    dataset_overview,
    live_summary,
    parameter_explanation,
)
from pages_modules.clustering_viz import render_analysis_visual, render_main_visual


STATE_DEFAULTS = {
    "cluster_lab_algorithm": "kmeans",
    "cluster_lab_dataset": "kmeans_blobs",
    "cluster_lab_seed": 17,
    "cluster_lab_view_nonce": 0,
    "cluster_lab_prev_algorithm": "kmeans",
    "clustering_intro_game_index": 0,
    "clustering_intro_game_submitted": False,
    "clustering_intro_game_answer": "A",
}

CLUSTERING_INTRO_QUESTIONS = [
    {
        "title": "学生兴趣分组",
        "prompt": "现在有几名学生的兴趣如下：",
        "items": [
            "学生 A：编程、数学、机器人",
            "学生 B：绘画、摄影、设计",
            "学生 C：篮球、跑步、健身",
            "学生 D：Python、算法、深度学习",
        ],
        "options": {"A": "2 组", "B": "3 组", "C": "4 组"},
        "answer": "B",
        "explanation": "一个合理分法是：技术学习类、艺术设计类、体育运动类。这就是聚类任务：在没有标准标签的情况下，根据相似性发现群体结构。",
    },
    {
        "title": "用户行为分群",
        "prompt": "现在有几位用户的行为特征如下：",
        "items": [
            "用户 A：高频登录，经常评论，学习时长长",
            "用户 B：偶尔登录，很少互动，学习时长短",
            "用户 C：高频登录，经常提交作业，学习时长长",
            "用户 D：很少登录，互动次数低，学习时长短",
        ],
        "options": {"A": "2 组", "B": "3 组", "C": "4 组"},
        "answer": "A",
        "explanation": "可以大致分为高活跃用户和低活跃用户两类。聚类就是在没有现成标签的情况下，先从相似行为中发现这些群体。",
    },
    {
        "title": "城市区域特征分组",
        "prompt": "现在有几个区域的特征如下：",
        "items": [
            "区域 A：人口密度高，商业设施多",
            "区域 B：人口密度低，自然景观多",
            "区域 C：人口密度高，交通站点多",
            "区域 D：人口密度低，绿地面积大",
        ],
        "options": {"A": "2 组", "B": "3 组", "C": "4 组"},
        "answer": "A",
        "explanation": "可以大致分为城市核心区域和生态居住区域两类。这个例子体现了聚类通过相似性自动发现隐藏群体。",
    },
]

CLUSTERING_HISTORY_KEY = "cluster_lab_history"

CLUSTERING_TASKS = {
    "kmeans": {
        "title": "中心搜寻挑战",
        "description": "比较簇中心初始化和簇数量变化，判断哪些分组更紧凑、分离更清晰。",
    },
    "dbscan": {
        "title": "噪声侦查挑战",
        "description": "围绕 eps 与 min_samples 调参，分辨哪些点应被视为主簇成员、边界点或噪声点。",
    },
    "agg": {
        "title": "层级合并挑战",
        "description": "比较不同 linkage 与切分层级，理解样本如何一步步合并成更大的群体。",
    },
    "gmm": {
        "title": "概率归属挑战",
        "description": "观察成分数与软分配概率，判断样本是否更适合椭圆形概率团块解释。",
    },
}


CLUSTERING_SCENARIOS = {
    "user_segments": {
        "title": "用户群体划分",
        "description": "模拟在没有标签答案的情况下，根据样本特征自动发现潜在用户群体，重点观察簇数量、重叠程度和噪声点变化。",
    },
    "learning_patterns": {
        "title": "学习行为分群",
        "description": "模拟对学习行为数据做自然分群，适合讨论不同层级切分下，哪些样本会先被合并为更稳定的小群体。",
    },
    "city_zones": {
        "title": "城市区域特征分组",
        "description": "模拟依据区域特征进行自然分群，适合观察密度差异、过渡带与孤立样本如何影响聚类结果。",
    },
    "unknown_structure": {
        "title": "未知样本结构探索",
        "description": "模拟对未知样本结构做探索性分析，聚类结果并不追求唯一标准答案，而是帮助理解数据内部的潜在分布形态。",
    },
}


CLUSTERING_DATASET_SCENARIO_MAP = {
    "kmeans_blobs": "user_segments",
    "kmeans_noisy_blobs": "user_segments",
    "dbscan_moons": "city_zones",
    "dbscan_circles": "city_zones",
    "dbscan_vary_density": "city_zones",
    "agg_nested_blobs": "learning_patterns",
    "agg_bridge_groups": "learning_patterns",
    "gmm_ellipses": "unknown_structure",
    "gmm_overlap_ellipses": "unknown_structure",
}


def render_clustering_lab():
    ensure_state()
    sync_dataset_with_algorithm()
    inject_page_css()

    algorithm_key = st.session_state["cluster_lab_algorithm"]
    dataset_key = st.session_state["cluster_lab_dataset"]
    algo_info = algorithm_overview(algorithm_key)
    dataset_info = dataset_overview(dataset_key)

    if st.button("🏠 返回首页", key="clu_back_home", use_container_width=False):
        st.session_state.current_page = "home"
        st.rerun()
    st.markdown("---")
    st.markdown(
        """
        <div class="lab-hero">
            <div class="lab-overline">机器学习 · 聚类</div>
            <div class="lab-title">无监督聚类可视化学习实验室</div>
            <div class="lab-subtitle">
                先选算法，再切换最适合它的教学数据，随后观察簇结构、参数变化和内部评价指标如何一起改变。
            </div>
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

    with st.expander("聚类基础知识", expanded=False):
        for index, (title, body) in enumerate(clustering_basics_sections()):
            st.markdown("**{0}. {1}**".format(index + 1, title))
            st.markdown(body)

    render_clustering_intro_game()
    st.markdown("---")
    st.markdown("## 进入正式可视化学习")

    control_col, content_col = st.columns([0.84, 2.96], gap="medium")
    with control_col:
        settings = render_control_panel(algorithm_key)

    lab_result = build_lab_result(
        algorithm_key=settings["algorithm_key"],
        dataset_key=settings["dataset_key"],
        sample_count=settings["sample_count"],
        noise=settings["noise"],
        params=settings["params"],
    )

    with content_col:
        render_display_panel(settings["algorithm_key"], settings["params"], lab_result)

    st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)
    render_bottom_panel(settings["algorithm_key"], settings["dataset_key"], settings["params"], lab_result)


def ensure_state():
    for key, value in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_clustering_intro_game():
    questions = CLUSTERING_INTRO_QUESTIONS
    index = st.session_state["clustering_intro_game_index"] % len(questions)
    question = questions[index]
    option_keys = list(question["options"].keys())
    if st.session_state["clustering_intro_game_answer"] not in option_keys:
        st.session_state["clustering_intro_game_answer"] = option_keys[0]

    st.markdown("### AI 分组小游戏")
    st.markdown(
        '<div class="teach-note">先做一个相似对象分组小题，理解聚类任务并没有现成标签，而是根据相似性去发现潜在群体。</div>',
        unsafe_allow_html=True,
    )
    st.markdown("**题目场景：{0}**".format(question["title"]))
    st.markdown(question["prompt"])
    for item in question["items"]:
        st.markdown("- {0}".format(item))

    st.radio(
        "你觉得大致可以分成几组？",
        options=option_keys,
        key="clustering_intro_game_answer",
        format_func=lambda value: "{0}. {1}".format(value, question["options"][value]),
        horizontal=True,
    )

    action_col1, action_col2 = st.columns(2, gap="medium")
    if action_col1.button("提交答案", key="clustering_intro_game_submit", use_container_width=True):
        st.session_state["clustering_intro_game_submitted"] = True
    action_col2.button(
        "再来一题",
        key="clustering_intro_game_next",
        use_container_width=True,
        on_click=next_clustering_intro_question,
    )

    if st.session_state["clustering_intro_game_submitted"]:
        selected = st.session_state["clustering_intro_game_answer"]
        if selected == question["answer"]:
            st.success("回答正确。{0}".format(question["explanation"]))
        else:
            st.warning("还差一点。参考答案是 {0}。{1}".format(question["options"][question["answer"]], question["explanation"]))
        st.info("下面可以通过 KMeans、DBSCAN、层次聚类和 GMM，观察算法如何在没有标签的情况下自动发现隐藏群体。")



def next_clustering_intro_question():
    next_index = (st.session_state["clustering_intro_game_index"] + 1) % len(CLUSTERING_INTRO_QUESTIONS)
    st.session_state["clustering_intro_game_index"] = next_index
    st.session_state["clustering_intro_game_submitted"] = False
    st.session_state["clustering_intro_game_answer"] = list(CLUSTERING_INTRO_QUESTIONS[next_index]["options"].keys())[0]


def sync_dataset_with_algorithm():
    current_algorithm = st.session_state["cluster_lab_algorithm"]
    previous_algorithm = st.session_state["cluster_lab_prev_algorithm"]
    valid_options = get_dataset_options(current_algorithm)
    if previous_algorithm != current_algorithm or st.session_state["cluster_lab_dataset"] not in valid_options:
        st.session_state["cluster_lab_dataset"] = get_default_dataset(current_algorithm)
        st.session_state["cluster_lab_prev_algorithm"] = current_algorithm
        st.session_state["cluster_lab_view_nonce"] += 1


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


def render_control_panel(algorithm_key):
    st.markdown("### 操作区")
    algorithm_key = st.radio(
        "选择算法",
        options=list(ALGORITHM_LABELS.keys()),
        format_func=lambda key: ALGORITHM_LABELS[key],
        key="cluster_lab_algorithm",
        label_visibility="collapsed",
    )

    dataset_key = st.selectbox(
        "选择教学数据",
        options=get_dataset_options(algorithm_key),
        format_func=get_dataset_label,
        key="cluster_lab_dataset",
    )

    col1, col2 = st.columns(2)
    if col1.button("重新生成数据", use_container_width=True):
        st.session_state["cluster_lab_seed"] += 1
    if col2.button("刷新观察点", use_container_width=True):
        st.session_state["cluster_lab_view_nonce"] += 1

    sample_cfg = sample_config(algorithm_key)
    with st.expander("数据设置", expanded=True):
        sample_count = st.slider("样本数量", sample_cfg[0], sample_cfg[1], sample_cfg[2], sample_cfg[3])
        noise = st.slider("噪声强度", 0.00, 0.80, default_noise(algorithm_key), 0.02)
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
        "params": params,
    }


def sample_config(algorithm_key):
    if algorithm_key == "agg":
        return (40, 110, 72, 4)
    if algorithm_key == "dbscan":
        return (100, 260, 180, 20)
    return (100, 260, 180, 20)


def default_noise(algorithm_key):
    mapping = {
        "kmeans": 0.16,
        "dbscan": 0.12,
        "agg": 0.08,
        "gmm": 0.10,
    }
    return mapping[algorithm_key]


def render_parameter_controls(algorithm_key):
    if algorithm_key == "kmeans":
        return {
            "n_clusters": st.slider("簇数量", 2, 6, 3, 1),
            "init": st.selectbox(
                "初始化方式",
                options=["kmeans++", "random"],
                format_func=lambda item: "KMeans++" if item == "kmeans++" else "随机中心",
            ),
            "n_init": st.slider("重复初始化次数", 1, 8, 4, 1),
            "max_iter": st.slider("最大迭代次数", 4, 30, 16, 1),
        }
    if algorithm_key == "dbscan":
        return {
            "eps": st.slider("邻域半径", 0.10, 0.80, 0.28, 0.01),
            "min_samples": st.slider("核心点最少邻居数", 3, 12, 5, 1),
        }
    if algorithm_key == "agg":
        return {
            "n_clusters": st.slider("簇数量", 2, 6, 3, 1),
            "linkage": st.selectbox(
                "连接方式",
                options=["single", "complete", "average", "ward"],
                format_func=lambda item: {
                    "single": "single（最近距离）",
                    "complete": "complete（最远距离）",
                    "average": "average（平均距离）",
                    "ward": "ward（方差最小化）",
                }[item],
            ),
        }
    return {
        "n_components": st.slider("高斯成分数", 2, 5, 3, 1),
        "covariance_type": st.selectbox(
            "协方差类型",
            options=["full", "diag"],
            format_func=lambda item: "full（完整协方差）" if item == "full" else "diag（对角协方差）",
        ),
        "max_iter": st.slider("最大迭代次数", 8, 36, 18, 1),
    }


def build_lab_result(algorithm_key, dataset_key, sample_count, noise, params):
    X, dataset_meta = generate_clustering_dataset(
        dataset_key=dataset_key,
        n_samples=sample_count,
        noise=noise,
        random_state=st.session_state["cluster_lab_seed"],
    )

    model = build_clusterer(algorithm_key, params)
    model.fit(X)
    labels = np.asarray(model.labels_, dtype=int)
    metrics = build_clustering_metrics(X, labels)
    extras = build_algorithm_extras(algorithm_key, dataset_key, model, X, labels, params, metrics)
    game = build_clustering_game_context(algorithm_key, dataset_key, X, labels, metrics, extras, len(X))

    visual_context = {
        "visual_title": ALGORITHM_LABELS[algorithm_key],
        "X": X,
        "labels": labels,
        "metrics": metrics,
    }
    visual_context.update(extras)

    return {
        "X": X,
        "labels": labels,
        "model": model,
        "params": params,
        "metrics": metrics,
        "extras": extras,
        "visual_context": visual_context,
        "game": game,
    }


def build_algorithm_extras(algorithm_key, dataset_key, model, X, labels, params, metrics):
    extras = {}

    if algorithm_key == "kmeans":
        extras.update(
            {
                "model": model,
                "center_history": model.center_history_,
                "initial_centers": model.initial_centers_,
                "final_centers": model.cluster_centers_,
                "iterations": model.iterations_,
                "cluster_count": metrics["cluster_count"],
                "inertia": model.inertia_,
                "suggested_clusters": dataset_suggested_clusters(dataset_key),
            }
        )
        return extras

    if algorithm_key == "dbscan":
        neighbor_sizes = np.array([len(items) for items in model.neighbor_lists_])
        core_candidates = np.where(model.core_sample_mask_)[0]
        if len(core_candidates) > 0:
            focus_index = core_candidates[np.argmax(neighbor_sizes[core_candidates])]
        else:
            focus_index = int(np.argmax(neighbor_sizes))
        extras.update(
            {
                "core_mask": model.core_sample_mask_,
                "border_mask": model.border_mask_,
                "noise_mask": model.noise_mask_,
                "eps": params["eps"],
                "min_samples": params["min_samples"],
                "focus_point": X[focus_index],
                "focus_neighbors": model.neighbor_lists_[focus_index].tolist(),
                "focus_type": point_type(model, focus_index),
                "noise_count": metrics["noise_count"],
            }
        )
        return extras

    if algorithm_key == "agg":
        extras.update(
            {
                "merge_history": model.merge_history_,
                "cut_distance": model.cut_distance_,
                "cluster_count": metrics["cluster_count"],
                "linkage": params["linkage"],
            }
        )
        return extras

    probabilities = model.responsibilities_
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
    focus_index = int(np.argmax(entropy))
    extras.update(
        {
            "model": model,
            "means": model.means_,
            "covariances": model.covariances_,
            "covariance_type": model.covariance_type,
            "weights": model.weights_,
            "focus_point": X[focus_index],
            "focus_probabilities": probabilities[focus_index],
            "focus_entropy": float(entropy[focus_index]),
            "focus_max_probability": float(np.max(probabilities[focus_index])),
        }
    )
    return extras


def point_type(model, index):
    if model.noise_mask_[index]:
        return "噪声点"
    if model.core_sample_mask_[index]:
        return "核心点"
    return "边界点"


def dataset_suggested_clusters(dataset_key):
    mapping = {
        "kmeans_blobs": 3,
        "kmeans_noisy_blobs": 4,
    }
    return mapping.get(dataset_key, 3)


def render_display_panel(algorithm_key, params, lab_result):
    algo_info = algorithm_overview(algorithm_key)
    st.markdown("### 聚类结构主图")
    main_visual = render_main_visual(algorithm_key, lab_result["visual_context"])
    render_visual_output(main_visual, "当前主图未成功生成，请调整参数后重试。")
    with st.expander("图像观察与调参提示", expanded=False):
        st.markdown("**结构观察**")
        st.info(live_summary(algorithm_key, params, lab_result["metrics"], lab_result["visual_context"]))
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
    st.markdown("### 指标与结构分析")
    metrics = lab_result["metrics"]

    metric_values = [
        ("轮廓系数", format_metric(metrics["silhouette"], "{0:.3f}")),
        ("戴维斯-鲍丁指数", format_metric(metrics["davies_bouldin"], "{0:.3f}")),
        ("卡林斯基-哈拉巴斯指数", format_metric(metrics["calinski_harabasz"], "{0:.1f}")),
        ("当前簇数量", str(metrics["cluster_count"])),
        ("噪声点数量", str(metrics["noise_count"])),
    ]

    render_metric_cards(metric_values, columns_per_row=3)

    st.markdown("#### 聚类结构分析")
    analysis_visual = render_analysis_visual(algorithm_key, lab_result["visual_context"])
    render_visual_output(
        analysis_visual,
        "当前结构分析图未成功生成。",
        image_width="min(100%, 960px)",
    )
    st.markdown("#### 调参与结果提示")
    render_clustering_regular_hint(lab_result["game"])
    st.markdown("#### 结构解读")
    st.markdown(lab_result["game"]["interpretation"])
    st.markdown("#### 当前模型结论")
    st.markdown(bottom_conclusion(algorithm_key, dataset_key, metrics, lab_result["visual_context"]))


def get_clustering_scenario(dataset_key):
    scenario_key = CLUSTERING_DATASET_SCENARIO_MAP.get(dataset_key, "unknown_structure")
    return CLUSTERING_SCENARIOS[scenario_key]


def build_clustering_detective_game(algorithm_key, dataset_key, params, metrics, extras, sample_count):
    scenario = get_clustering_scenario(dataset_key)
    noise_ratio = metrics["noise_count"] / max(sample_count, 1)
    suggestions = generate_parameter_suggestions(algorithm_key, params, metrics, noise_ratio, extras)
    feedback = get_clustering_detective_feedback(algorithm_key, metrics, noise_ratio)
    return {
        "title": "数据侦探：这些样本藏着几类群体？",
        "description": "在没有真实标签的情况下，先观察样本分布，猜一猜这些数据大致可以分成几类。然后调节聚类参数，观察算法给出的结构是否符合你的判断。",
        "context": "{0} · {1}".format(scenario["title"], ALGORITHM_LABELS[algorithm_key]),
        "cluster_count": int(metrics["cluster_count"]),
        "guess_range": (1, 8),
        "feedback": feedback["message"],
        "tone": feedback["tone"],
        "explanation": get_clustering_algorithm_clue(algorithm_key),
        "hint": feedback["hint"],
        "hint_tone": feedback["tone"],
        "suggestions": suggestions[:2],
        "interpretation": build_clustering_interpretation(scenario, algorithm_key, metrics, noise_ratio),
        "next_step": "继续调整当前算法的关键参数，再比较簇数量、噪声点和结构分离是否更符合你的观察。",
    }


def get_clustering_algorithm_clue(algorithm_key):
    clues = {
        "kmeans": "KMeans 倾向于寻找球状、中心明显的簇，关键是簇数和中心位置。",
        "dbscan": "DBSCAN 根据密度连接样本，能发现噪声点，也适合非球形簇。",
        "agg": "层次聚类从局部相似开始逐步合并，适合观察样本之间的层级关系。",
        "gmm": "GMM 用多个高斯分布描述数据，每个样本可以以概率方式属于不同群体。",
    }
    return clues[algorithm_key]


def get_clustering_detective_feedback(algorithm_key, metrics, noise_ratio):
    cluster_count = metrics["cluster_count"]
    if cluster_count < 2:
        return {
            "tone": "warning",
            "message": "当前结果形成的有效簇太少，说明参数可能把不同群体合并在了一起。",
            "hint": "修复提示：先让有效簇数量稳定形成，再判断样本到底藏着几类群体。",
        }
    if algorithm_key == "dbscan" and noise_ratio >= 0.35:
        return {
            "tone": "warning",
            "message": "当前噪声点偏多，说明密度阈值可能过严。",
            "hint": "修复提示：可以先增大 eps，或适当降低 min_samples。",
        }
    if metrics["silhouette"] is not None and metrics["silhouette"] >= 0.35:
        return {
            "tone": "success",
            "message": "当前簇结构比较清晰，样本分组与分离度都比较可解释。",
            "hint": "修复提示：可以继续观察簇数变化后，分离是否仍然稳定。",
        }
    return {
        "tone": "info",
        "message": "当前簇结构还可以继续比较，可能存在簇重叠、簇太多或边界不稳定的情况。",
        "hint": "修复提示：继续从簇数量、噪声点比例和轮廓系数一起判断结果是否合理。",
    }


def render_cluster_detective_game(game, key_prefix):
    st.markdown("### {0}".format(game["title"]))
    st.markdown(
        """
        <div class="challenge-card">
            <div class="challenge-kicker">问题驱动式小游戏</div>
            <div class="challenge-task-name">先猜簇数量，再调参验证</div>
            <div class="challenge-task-desc">{0}</div>
            <div class="challenge-context"><b>当前任务场景：</b>{1}</div>
        </div>
        """.format(game["description"], game["context"]),
        unsafe_allow_html=True,
    )
    guess = st.slider(
        "猜一猜这些样本大致可以分成几类",
        min_value=game["guess_range"][0],
        max_value=game["guess_range"][1],
        value=st.session_state.get("{0}_detective_guess".format(key_prefix), 3),
        key="{0}_detective_guess".format(key_prefix),
    )
    actual = game["cluster_count"]
    if abs(guess - actual) <= 1:
        st.success("你的观察与算法结果较接近。当前算法实际得到的主要簇数量是 {0}。".format(actual))
    else:
        st.info("当前算法实际得到的主要簇数量是 {0}。可以再观察是否存在噪声点、簇重叠或参数设置不合适。".format(actual))
    st.markdown("**结构解释**")
    st.caption(game["explanation"])
    st.markdown("**继续观察**")
    st.caption(game["next_step"])


def build_clustering_challenge(algorithm_key, dataset_key, params, metrics, extras, sample_count):
    scenario = get_clustering_scenario(dataset_key)
    noise_ratio = metrics["noise_count"] / max(sample_count, 1)
    score = calculate_clustering_challenge_score(algorithm_key, params, metrics, noise_ratio, extras)
    task_info = get_algorithm_task_info(algorithm_key)
    goal_items = evaluate_challenge_goals(algorithm_key, params, metrics, noise_ratio, extras)
    progress = int(round(100 * sum(1 for item in goal_items if item["passed"]) / max(len(goal_items), 1)))
    status = render_progress_status(score, progress, goal_items)
    suggestions = generate_parameter_suggestions(algorithm_key, params, metrics, noise_ratio, extras)
    feedback = generate_clustering_feedback(algorithm_key, params, metrics, noise_ratio)
    interpretation = build_clustering_interpretation(scenario, algorithm_key, metrics, noise_ratio)
    return {
        "title": "数据侦探挑战",
        "description": "请在没有真实标签的情况下调节聚类参数，像数据侦探一样追踪自然形成的群体结构。",
        "task_title": task_info["title"],
        "task_description": task_info["description"],
        "subtitle": "隐藏群体发现挑战",
        "context_title": "任务场景",
        "context": "{0} · {1}".format(scenario["title"], ALGORITHM_LABELS[algorithm_key]),
        "goals": [item["label"] for item in goal_items],
        "goal_items": goal_items,
        "progress": progress,
        "status_label": status["label"],
        "status_note": status["note"],
        "status_tone": status["tone"],
        "score": score,
        "score_label": "当前探索得分",
        "feedback": feedback["message"],
        "tone": feedback["tone"],
        "suggestions": suggestions,
        "interpretation": interpretation,
        "history_row": {
            "算法": ALGORITHM_LABELS[algorithm_key],
            "任务场景": scenario["title"],
            "簇数量": metrics["cluster_count"],
            "轮廓系数": format_metric(metrics["silhouette"], "{0:.3f}"),
            "探索得分": score,
            "结论": status["label"],
        },
        "record_message": "已记录当前聚类实验。",
    }


def get_algorithm_task_info(algorithm_key):
    return CLUSTERING_TASKS[algorithm_key]


def evaluate_challenge_goals(algorithm_key, params, metrics, noise_ratio, extras):
    goal_items = [
        {
            "label": "轮廓系数保持在可解释区间",
            "passed": metrics["silhouette"] is not None and metrics["silhouette"] >= 0.25,
        },
        {
            "label": "戴维斯-鲍丁指数不过高",
            "passed": metrics["davies_bouldin"] is not None and metrics["davies_bouldin"] <= 1.60,
        },
        {
            "label": "有效簇数量能够稳定形成",
            "passed": metrics["cluster_count"] >= 2,
        },
    ]

    if algorithm_key == "kmeans":
        suggested = extras.get("suggested_clusters", metrics["cluster_count"])
        goal_items.append({"label": "簇数接近数据结构建议值", "passed": abs(metrics["cluster_count"] - suggested) <= 1})
    elif algorithm_key == "dbscan":
        goal_items.append({"label": "噪声点比例控制在 35% 内", "passed": noise_ratio <= 0.35})
    elif algorithm_key == "agg":
        goal_items.append({"label": "层级切分保持在 2 到 5 个簇", "passed": 2 <= metrics["cluster_count"] <= 5})
    else:
        goal_items.append({"label": "高斯成分数保持适中", "passed": 2 <= params["n_components"] <= 4})
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


def calculate_clustering_challenge_score(algorithm_key, params, metrics, noise_ratio, extras):
    score = 0.0

    if metrics["silhouette"] is not None:
        score += np.clip((metrics["silhouette"] + 0.15) / 0.75, 0.0, 1.0) * 36.0
    else:
        score += 8.0

    if metrics["davies_bouldin"] is not None:
        score += max(0.0, 24.0 * (1.0 - min(metrics["davies_bouldin"] / 2.4, 1.0)))
    else:
        score += 6.0

    if metrics["calinski_harabasz"] is not None:
        score += min(np.log1p(metrics["calinski_harabasz"]) / 6.0, 1.0) * 18.0
    else:
        score += 6.0

    if algorithm_key == "dbscan":
        score += max(0.0, 12.0 * (1.0 - min(noise_ratio / 0.35, 1.0)))
    else:
        score += 12.0

    cluster_count = metrics["cluster_count"]
    if algorithm_key == "kmeans":
        suggested = extras.get("suggested_clusters", cluster_count)
        score += max(0.0, 10.0 * (1.0 - min(abs(cluster_count - suggested) / 3.0, 1.0)))
    elif algorithm_key == "dbscan":
        score += 10.0 if 2 <= cluster_count <= 5 else max(0.0, 10.0 - abs(cluster_count - 3) * 3.0)
    elif algorithm_key == "agg":
        score += 10.0 if 2 <= cluster_count <= 5 else 4.0
    else:
        score += 10.0 if 2 <= cluster_count <= 4 else 5.0

    return int(np.clip(round(score), 0, 100))


def generate_clustering_feedback(algorithm_key, params, metrics, noise_ratio):
    cluster_count = metrics["cluster_count"]
    silhouette = metrics["silhouette"]
    db_index = metrics["davies_bouldin"]
    suggestions = generate_parameter_suggestions(algorithm_key, params, metrics, noise_ratio, {})

    if silhouette is None or cluster_count < 2:
        return {
            "tone": "warning",
            "message": "当前参数下暂不适合计算部分聚类指标，说明有效簇数量不足。{0}".format(suggestions[0]),
        }
    if algorithm_key == "dbscan" and noise_ratio >= 0.35:
        return {
            "tone": "warning",
            "message": "当前噪声点比例偏高，eps 可能过小或 min_samples 设置过严格。建议优先微调这两个参数。",
        }
    if silhouette >= 0.45 and db_index is not None and db_index <= 0.90:
        return {"tone": "success", "message": "当前簇结构较清晰，簇间分离和簇内紧凑性都处于较理想状态。"}
    if silhouette <= 0.18:
        return {
            "tone": "info",
            "message": "当前轮廓系数偏低，不同簇之间可能存在重叠或分离不明显。{0}".format(suggestions[0]),
        }
    if cluster_count >= 6:
        return {"tone": "warning", "message": "当前簇数量偏多，结果可能出现过度划分，建议尝试减少簇数或放宽划分条件。"}
    if cluster_count <= 1:
        return {"tone": "warning", "message": "当前有效簇数量过少，可能把不同群体合并到了一起，建议重新调整参数。"}
    return {
        "tone": "info",
        "message": "当前聚类已经形成一定结构，但仍可继续比较不同参数下的簇边界、噪声点和内部评价指标。",
    }


def generate_parameter_suggestions(algorithm_key, params, metrics, noise_ratio, extras):
    suggestions = []
    if metrics["silhouette"] is None or metrics["cluster_count"] < 2:
        if algorithm_key == "kmeans":
            suggestions.append("重新检查簇数设置，避免把不同群体压缩成单一簇。")
        elif algorithm_key == "dbscan":
            suggestions.append("适当增大 eps 或降低 min_samples，先让有效簇稳定形成。")
        elif algorithm_key == "agg":
            suggestions.append("重新调整 n_clusters，观察不同切分层级是否能形成稳定簇。")
        else:
            suggestions.append("减少或增加高斯成分数，先让有效成分能够稳定区分开来。")
    if algorithm_key == "dbscan" and noise_ratio > 0.35:
        suggestions.append("优先增大 eps，或适当降低 min_samples，减少过多噪声点。")
    if metrics["cluster_count"] >= 6:
        suggestions.append("当前簇数偏多，可尝试减少簇数或放宽划分条件。")
    if metrics["silhouette"] is not None and metrics["silhouette"] < 0.25:
        if algorithm_key == "kmeans":
            suggestions.append("微调 K 值，比较不同簇数下的分离效果。")
        elif algorithm_key == "agg":
            suggestions.append("比较不同 linkage，观察哪种合并顺序更能分开簇。")
        elif algorithm_key == "gmm":
            suggestions.append("尝试调整成分数或协方差类型，观察软分配是否更清晰。")
        else:
            suggestions.append("继续比较密度参数，观察边界点与主簇的归属是否更合理。")

    if not suggestions:
        suggestions.append("当前参数已经形成较稳定的结构，可继续微调并比较簇边界是否更清晰。")
    return suggestions[:3]


def build_clustering_interpretation(scenario, algorithm_key, metrics, noise_ratio):
    if metrics["silhouette"] is None or metrics["cluster_count"] < 2:
        return "当前参数下有效簇数量不足，暂时还难以稳定解释样本结构。"
    if algorithm_key == "dbscan" and noise_ratio >= 0.35:
        return "当前结果把较多样本视为噪声点，说明密度阈值仍然偏严格。"
    if metrics["silhouette"] >= 0.45:
        return "当前聚类结果已经分离出较清晰的群体，适合继续比较各簇的结构差异。"
    return "当前聚类结果提供了一种可讨论的结构划分，可以继续结合簇数量、噪声点和分离程度综合判断。"

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
                "{0} {1}".format(challenge["description"], challenge["subtitle"]),
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
                record_experiment(CLUSTERING_HISTORY_KEY, challenge["history_row"])
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
            "{0} {1}".format(challenge["description"], challenge["subtitle"]),
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
        record_experiment(CLUSTERING_HISTORY_KEY, challenge["history_row"])
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
        history = st.session_state.get(CLUSTERING_HISTORY_KEY, [])
        if not history:
            st.info("还没有记录聚类实验，调好参数后可以点击“记录当前实验”。")
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
        help="开启后显示群体归属挑战，不影响常规教学图表。",
    )


def build_clustering_easter_egg(algorithm_key, dataset_key, params, metrics, extras, sample_count):
    scenario = get_clustering_scenario(dataset_key)
    noise_ratio = metrics["noise_count"] / max(sample_count, 1)
    suggestions = generate_parameter_suggestions(algorithm_key, params, metrics, noise_ratio, extras)
    return {
        "title": "石头剪刀布分群实验",
        "dataset_label": get_dataset_label(dataset_key),
        "description": "把当前无标签散点想象成石头、剪刀、布三个阵营。聚类算法并不知道真实阵营，只会根据空间位置或密度结构自动分组，因此颜色代表算法发现的群体，而不是标准答案。",
        "algorithm_note": get_clustering_algorithm_clue(algorithm_key),
        "hint": generate_clustering_easter_egg_hint(algorithm_key, metrics, noise_ratio),
        "hint_tone": "warning" if algorithm_key == "dbscan" and noise_ratio >= 0.35 else ("success" if metrics["silhouette"] is not None and metrics["silhouette"] >= 0.35 else "info"),
        "suggestions": suggestions[:2],
        "interpretation": build_clustering_interpretation(scenario, algorithm_key, metrics, noise_ratio),
        "cluster_count": int(metrics["cluster_count"]),
        "noise_count": int(metrics["noise_count"]),
    }


def generate_clustering_easter_egg_hint(algorithm_key, metrics, noise_ratio):
    cluster_count = int(metrics["cluster_count"])
    if cluster_count < 2:
        return "彩蛋提示：当前有效群体太少，参数可能把不同阵营合并到了一起。"
    if algorithm_key == "dbscan" and noise_ratio >= 0.35:
        return "彩蛋提示：当前噪声点偏多，通常意味着 eps 偏小或 min_samples 偏严格。"
    if cluster_count >= 5:
        return "彩蛋提示：当前分出来的群体偏多，可能已经出现过度划分。"
    return "彩蛋提示：当前群体结构已经比较清楚，可以继续微调参数比较边界与噪声点的变化。"


def render_clustering_rps_experiment(panel):
    st.markdown("### {0}".format(panel["title"]))
    st.markdown(
        """
        <div class="challenge-card">
            <div class="challenge-kicker">趣味彩蛋模式</div>
            <div class="challenge-task-name">无监督分群观察</div>
            <div class="challenge-task-desc">{0}</div>
            <div class="challenge-context"><b>当前数据：</b>{1}</div>
        </div>
        """.format(panel["description"], panel["dataset_label"]),
        unsafe_allow_html=True,
    )
    if st.button("重新布阵", key="cluster_fun_reseed", use_container_width=True):
        st.session_state["cluster_lab_seed"] += 1
        st.session_state["cluster_lab_fun_disturbance"] = 0
        st.session_state["cluster_lab_view_nonce"] += 1
        st.rerun()
    if st.button("增加少量扰动", key="cluster_fun_disturb", use_container_width=True):
        st.session_state["cluster_lab_seed"] += 1
        st.session_state["cluster_lab_fun_disturbance"] = min(3, st.session_state.get("cluster_lab_fun_disturbance", 0) + 1)
        st.session_state["cluster_lab_view_nonce"] += 1
        st.rerun()
    st.caption("当前算法识别出 {0} 个主要群体，噪声点 {1} 个。".format(panel["cluster_count"], panel["noise_count"]))
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


def render_clustering_regular_hint(panel):
    st.info("无监督学习没有唯一标准答案，建议结合簇数量、噪声点和分离程度一起判断结果。")
    if panel.get("suggestions"):
        st.markdown("**简短建议**")
        for item in panel["suggestions"]:
            st.markdown("- {0}".format(item))


def build_clustering_game_context(algorithm_key, dataset_key, X, labels, metrics, extras, sample_count):
    scenario = get_clustering_scenario(dataset_key)
    noise_ratio = metrics["noise_count"] / max(sample_count, 1)
    suggestions = generate_parameter_suggestions(algorithm_key, {}, metrics, noise_ratio, extras)
    return {
        "algorithm_key": algorithm_key,
        "algorithm_label": ALGORITHM_LABELS[algorithm_key],
        "dataset_label": get_dataset_label(dataset_key),
        "X": X,
        "labels": labels,
        "options": sorted(int(item) for item in np.unique(labels)),
        "hint": generate_clustering_easter_egg_hint(algorithm_key, metrics, noise_ratio),
        "hint_tone": "warning" if algorithm_key == "dbscan" and noise_ratio >= 0.35 else ("success" if metrics["silhouette"] is not None and metrics["silhouette"] >= 0.35 else "info"),
        "suggestions": suggestions[:2],
        "interpretation": build_clustering_interpretation(scenario, algorithm_key, metrics, noise_ratio),
        "algorithm_note": get_clustering_algorithm_clue(algorithm_key),
    }


def format_clustering_game_choice(label):
    return "噪声点" if int(label) == -1 else "群体 {0}".format(int(label))


def init_clustering_game_state(game):
    signature = "{0}|{1}|{2}|{3}".format(
        game["algorithm_key"],
        game["dataset_label"],
        len(game["labels"]),
        st.session_state["cluster_lab_seed"],
    )
    if st.session_state.get("clustering_game_signature") != signature:
        st.session_state["clustering_game_signature"] = signature
        st.session_state["clustering_game_round"] = 0
        st.session_state["clustering_game_score"] = 0
        st.session_state["clustering_game_sample_index"] = -1
        st.session_state["clustering_game_submitted"] = False
        st.session_state["clustering_game_feedback"] = ""
        st.session_state["clustering_game_choice"] = str(game["options"][0])
        next_clustering_game_round(game)
    elif st.session_state.get("clustering_game_sample_index", -1) >= len(game["labels"]):
        next_clustering_game_round(game)


def next_clustering_game_round(game):
    total = max(len(game["labels"]), 1)
    round_number = st.session_state.get("clustering_game_round", 0) + 1
    rng = np.random.default_rng(st.session_state["cluster_lab_seed"] + round_number + st.session_state["cluster_lab_view_nonce"] + 23)
    sample_index = int(rng.integers(0, total))
    st.session_state["clustering_game_round"] = round_number
    st.session_state["clustering_game_sample_index"] = sample_index
    st.session_state["clustering_game_submitted"] = False
    st.session_state["clustering_game_feedback"] = ""
    st.session_state["clustering_game_choice"] = str(game["options"][0])


def submit_clustering_game_answer(game):
    if st.session_state.get("clustering_game_submitted", False):
        return
    sample_index = int(st.session_state["clustering_game_sample_index"])
    user_choice = int(st.session_state["clustering_game_choice"])
    cluster_label = int(game["labels"][sample_index])
    matched = user_choice == cluster_label
    if matched:
        st.session_state["clustering_game_score"] += 1
    st.session_state["clustering_game_submitted"] = True
    st.session_state["clustering_game_feedback"] = {
        "user_choice": user_choice,
        "cluster_label": cluster_label,
        "matched": matched,
    }


def render_clustering_game(game):
    init_clustering_game_state(game)
    sample_index = int(st.session_state["clustering_game_sample_index"])
    point = game["X"][sample_index]
    answered = max(st.session_state["clustering_game_round"] - (0 if st.session_state["clustering_game_submitted"] else 1), 0)
    st.markdown("### 群体归属挑战")
    st.caption("观察当前聚类结果，判断当前样本属于哪个群体；如果是 DBSCAN，也可以判断它是否应视为噪声点。")
    st.caption("当前得分：{0} / {1}".format(st.session_state["clustering_game_score"], answered))
    st.caption("第 {0} 题".format(st.session_state["clustering_game_round"]))
    st.markdown("**当前样本特征**")
    st.markdown("- 特征 1：{0:.3f}".format(float(point[0])))
    st.markdown("- 特征 2：{0:.3f}".format(float(point[1])))
    st.radio(
        "请选择你判断的群体",
        options=[str(item) for item in game["options"]],
        key="clustering_game_choice",
        format_func=lambda value: format_clustering_game_choice(value),
    )
    if st.button("提交判断", key="clustering_game_submit", use_container_width=True, disabled=st.session_state["clustering_game_submitted"]):
        submit_clustering_game_answer(game)
    if st.session_state["clustering_game_submitted"]:
        feedback = st.session_state["clustering_game_feedback"]
        if feedback["matched"]:
            st.success("判断一致。当前算法将这个样本归入 {0}。".format(format_clustering_game_choice(feedback["cluster_label"])))
        else:
            st.warning(
                "这次与算法结果不一致。你选择的是 {0}，当前算法给出的标签是 {1}。".format(
                    format_clustering_game_choice(feedback["user_choice"]),
                    format_clustering_game_choice(feedback["cluster_label"]),
                )
            )
        st.caption(game["algorithm_note"])
    if st.button("下一题", key="clustering_game_next", use_container_width=True, disabled=not st.session_state["clustering_game_submitted"]):
        next_clustering_game_round(game)
        st.rerun()
    if st.button("重置游戏", key="clustering_game_reset", use_container_width=True):
        st.session_state["clustering_game_score"] = 0
        st.session_state["clustering_game_round"] = 0
        st.session_state["clustering_game_feedback"] = ""
        next_clustering_game_round(game)
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


def format_metric(value, fmt):
    if value is None:
        return "暂不适合"
    return fmt.format(value)
