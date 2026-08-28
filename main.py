# main.py
import streamlit as st

import config
from components.footer import render_footer
from components.header import render_header
from components.sidebar import render_sidebar
from router import render_page


# ==================== 页面全局配置 ====================
st.set_page_config(
    page_title="智视AI · 算法可视化学习平台",
    page_icon="🧠",
    layout="wide",
    menu_items={}
)

# ==================== 初始化 session_state ====================
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# 注入全局 CSS
st.markdown(config.GLOBAL_CSS, unsafe_allow_html=True)

render_header()
render_sidebar()

page = st.session_state.current_page
render_page(page)
render_footer(page)
