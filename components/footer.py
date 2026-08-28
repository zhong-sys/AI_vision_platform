import streamlit as st

from app_constants import PAGE_DISPLAY_NAMES


def render_footer(page: str) -> None:
    """渲染底部状态栏。"""
    st.markdown("---")
    bottom_col1, bottom_col2 = st.columns([3, 1])
    with bottom_col1:
        page_display = PAGE_DISPLAY_NAMES.get(page, "首页")
        if page != "home":
            st.markdown(f'<div class="status-bar">📍 当前页面：{page_display}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-bar">📍 当前位于首页</div>', unsafe_allow_html=True)
    with bottom_col2:
        st.markdown('<div style="text-align: right; color: #888; font-size: 13px;">基于 Streamlit + Plotly + Scikit-learn</div>', unsafe_allow_html=True)
