from html import escape
from typing import Optional

import streamlit as st


_PAGE_HEADER_CSS = """
<style>
    .page-header {
        margin: 0 0 18px 0;
        padding: 22px 26px 20px;
        border: 1px solid #DCEAF7;
        border-radius: 18px;
        background: linear-gradient(135deg, #FFFFFF 0%, #F5FAFF 100%);
        box-shadow: 0 6px 18px rgba(15, 91, 158, 0.05);
    }
    .page-breadcrumb {
        margin-bottom: 8px;
        color: #1A7EC1;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .page-title {
        margin: 0;
        color: #143D66;
        font-size: 30px;
        line-height: 1.35;
        font-weight: 800;
    }
    .page-description {
        margin: 10px 0 0;
        color: #546879;
        font-size: 16px;
        line-height: 1.75;
    }
    [data-testid="stAlert"] {
        border-radius: 12px;
    }
    [data-testid="stAlert"] > div {
        border-radius: 12px;
    }
</style>
"""


def render_page_header(
    title: str,
    module: str,
    description: str,
    *,
    back_key: Optional[str] = None,
    divider: bool = True,
) -> None:
    """渲染统一的页面标题、所属模块、简介和可选返回按钮。"""
    st.markdown(_PAGE_HEADER_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-breadcrumb">首页 / {escape(str(module))}</div>
            <h1 class="page-title">{escape(str(title))}</h1>
            <p class="page-description">{escape(str(description))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if back_key is not None:
        if st.button("🏠 返回首页", key=back_key, use_container_width=False):
            st.session_state.current_page = "home"
            st.rerun()

    if divider:
        st.divider()
