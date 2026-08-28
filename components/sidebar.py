import streamlit as st

from app_constants import PAGE_DISPLAY_NAMES
import config
from utils.helpers import resource_path


def render_sidebar() -> None:
    """渲染左侧导航栏。"""
    with st.sidebar:
        # 校徽
        _, logo_col, _ = st.columns([1, 2, 1])
        with logo_col:
            st.image(resource_path("assets/logo.png"), use_container_width=True)
            st.markdown(
                f'<p style="color: {config.PRIMARY_BLUE}; font-weight: bold; text-align: center; margin-top: 5px;">智视<span style="font-size:1.15em; margin-left: 3px;">AI</span></p>',
                unsafe_allow_html=True)
        st.markdown("---")

        # 机器学习模块
        st.markdown('<div class="sidebar-header">📁 机器学习展示</div>', unsafe_allow_html=True)
        if st.button(PAGE_DISPLAY_NAMES["classification"], use_container_width=True, key="ml_classify"):
            st.session_state.current_page = "classification"
            st.rerun()
        if st.button(PAGE_DISPLAY_NAMES["regression"], use_container_width=True, key="ml_regress"):
            st.session_state.current_page = "regression"
            st.rerun()
        if st.button(PAGE_DISPLAY_NAMES["clustering"], use_container_width=True, key="ml_cluster"):
            st.session_state.current_page = "clustering"
            st.rerun()

        # 神经网络模块
        st.markdown('<div class="sidebar-header" style="margin-top: 20px;">🧬 神经网络动画展示</div>', unsafe_allow_html=True)
        if st.button(PAGE_DISPLAY_NAMES["nn_basic"], use_container_width=True, key="nn_basic"):
            st.session_state.current_page = "nn_basic"
            st.rerun()
        if st.button(PAGE_DISPLAY_NAMES["nn_cnn"], use_container_width=True, key="nn_cnn"):
            st.session_state.current_page = "nn_cnn"
            st.rerun()
        if st.button(PAGE_DISPLAY_NAMES["nn_rnn"], use_container_width=True, key="nn_rnn"):
            st.session_state.current_page = "nn_rnn"
            st.rerun()
        if st.button(PAGE_DISPLAY_NAMES["nn_attention"], use_container_width=True, key="nn_att"):
            st.session_state.current_page = "nn_attention"
            st.rerun()

        # 国产大模型模块
        st.markdown('<div class="sidebar-header" style="margin-top: 20px;">🤖 国产大模型</div>', unsafe_allow_html=True)
        if st.button(PAGE_DISPLAY_NAMES["llm_deepseek"], use_container_width=True, key="llm_ds"):
            st.session_state.current_page = "llm_deepseek"
            st.rerun()
        if st.button(PAGE_DISPLAY_NAMES["llm_zhipu"], use_container_width=True, key="llm_zp"):
            st.session_state.current_page = "llm_zhipu"
            st.rerun()
        if st.button(PAGE_DISPLAY_NAMES["llm_qwen"], use_container_width=True, key="llm_qwen"):
            st.session_state.current_page = "llm_qwen"
            st.rerun()
        if st.button(PAGE_DISPLAY_NAMES["llm_wenxin"], use_container_width=True, key="llm_wx"):
            st.session_state.current_page = "llm_wenxin"
            st.rerun()

        # 返回首页按钮
        st.markdown("---")
        if st.button("🏠 返回首页", use_container_width=True, key="sidebar_home"):
            st.session_state.current_page = "home"
            st.rerun()

        # 平台理念
        st.markdown("---")
        st.markdown("""
        <div class="motto-bar">
            <div style="font-weight: bold; margin-bottom: 5px;">理念</div>
            <div>智见未来 · 视界无限</div>
            <div>知行合一 · 学无止境</div>
        </div>
        """, unsafe_allow_html=True)

        # 底部信息
        st.markdown("""
        <div style="margin-top: 30px; text-align: center; color: #999; font-size: 12px;">
            © 2026 智视<span style="font-size:1.15em">AI</span> · 算法可视化学习平台
        </div>
        """, unsafe_allow_html=True)
