import logging

import streamlit as st

from app_constants import LLM_MODEL_MAP, PAGE_KEYS


LOGGER = logging.getLogger(__name__)


def render_page(page: str) -> None:
    """按现有页面 key 渲染对应页面。"""
    # ----- 机器学习模块 -----
    if page == "classification":
        try:
            from pages_modules import classification_lab
            classification_lab.render_classification_lab()
        except ImportError:
            LOGGER.exception("Failed to import classification page module")
            st.error("加载分类模块失败，请稍后重试。")
    elif page == "regression":
        try:
            from pages_modules import regression_lab
            regression_lab.render_regression_lab()
        except ImportError:
            LOGGER.exception("Failed to import regression page module")
            st.error("加载回归模块失败，请稍后重试。")
    elif page == "clustering":
        try:
            from pages_modules import clustering_lab
            clustering_lab.render_clustering_lab()
        except ImportError:
            LOGGER.exception("Failed to import clustering page module")
            st.error("加载聚类模块失败，请稍后重试。")

    # ----- 神经网络模块 -----
    elif page == "nn_basic":
        try:
            from pages_modules import neural_vis_module
            neural_vis_module.render_neural_network_viz()
        except ImportError:
            LOGGER.exception("Failed to import basic neural network page module")
            st.error("加载基础神经网络模块失败，请稍后重试。")
    elif page == "nn_cnn":
        try:
            from pages_modules import cnn_viz_module
            cnn_viz_module.nv_render_cnn_viz()
        except ImportError:
            LOGGER.exception("Failed to import CNN page module")
            st.error("加载 CNN 模块失败，请稍后重试。")
    elif page == "nn_rnn":
        try:
            from pages_modules import rnn_viz_module
            rnn_viz_module.nv_render_rnn_viz()
        except ImportError:
            LOGGER.exception("Failed to import RNN page module")
            st.error("加载 RNN 模块失败，请稍后重试。")
    elif page == "nn_attention":
        try:
            from pages_modules import transformer_viz_module
            transformer_viz_module.nv_render_transformer_viz()
        except ImportError:
            LOGGER.exception("Failed to import Transformer page module")
            st.error("加载 Transformer 模块失败，请稍后重试。")

    # ----- 国产大模型模块 -----
    elif page.startswith("llm_"):
        model_name = LLM_MODEL_MAP.get(page, "智谱GLM")
        try:
            from pages_modules import llm
            llm.show(model_preselected=model_name)
        except ImportError:
            LOGGER.exception("Failed to import LLM page module")
            st.error("加载大模型模块失败，请稍后重试。")

    # ----- 首页 -----
    else:
        try:
            from pages_modules import home
            home.show()
        except ImportError:
            st.title("🧠 欢迎使用算法可视化学习平台")
            st.markdown("通过拖拽组件、调整参数、观察实时变化，直观理解人工智能算法。")
            st.info("👈 请从左侧菜单选择模块开始学习。")
