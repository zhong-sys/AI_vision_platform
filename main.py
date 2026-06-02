# main.py
import streamlit as st
import config
from utils.helpers import resource_path

# ==================== 页面全局配置 ====================
st.set_page_config(
    page_title="智视AI · 算法可视化学习平台",
    page_icon="🧠",
    layout="wide",
    menu_items={}
)

# 注入全局 CSS
st.markdown(config.GLOBAL_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
    [data-testid="stSidebar"] label[data-baseweb="radio"] span {
        white-space: nowrap;
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 初始化 session_state ====================
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# ==================== 顶部标题栏 ====================
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.image(resource_path("assets/logo.png"), width=80)
with col2:
    st.markdown(
        '<div style="display: flex; align-items: baseline;"><span class="school-name">智视<span style="font-size:1.15em; vertical-align:-0.05em">AI</span></span><span class="platform-title">算法可视化学习平台</span></div>',
        unsafe_allow_html=True)
with col3:
    st.markdown(
        f'<div style="text-align: right; padding-top: 10px;"><span style="color: {config.PRIMARY_BLUE};">智见未来 · 学无止境</span></div>',
        unsafe_allow_html=True)
st.markdown('<div class="top-gradient"></div>', unsafe_allow_html=True)

# ==================== 左侧导航栏（全部使用 st.button） ====================
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
    if st.button("📊 分类", use_container_width=True, key="ml_classify"):
        st.session_state.current_page = "classification"
        st.rerun()
    if st.button("📈 回归", use_container_width=True, key="ml_regress"):
        st.session_state.current_page = "regression"
        st.rerun()
    if st.button("🔵 聚类", use_container_width=True, key="ml_cluster"):
        st.session_state.current_page = "clustering"
        st.rerun()

    # 神经网络模块
    st.markdown('<div class="sidebar-header" style="margin-top: 20px;">🧬 神经网络动画展示</div>', unsafe_allow_html=True)
    if st.button("🔷 基础神经网络动画演示", use_container_width=True, key="nn_basic"):
        st.session_state.current_page = "nn_basic"
        st.rerun()
    if st.button("🧩 卷积神经网络 (CNN)", use_container_width=True, key="nn_cnn"):
        st.session_state.current_page = "nn_cnn"
        st.rerun()
    if st.button("🔄 循环神经网络 (RNN)", use_container_width=True, key="nn_rnn"):
        st.session_state.current_page = "nn_rnn"
        st.rerun()
    if st.button("👁 注意力机制（Transformer）", use_container_width=True, key="nn_att"):
        st.session_state.current_page = "nn_attention"
        st.rerun()

    # 国产大模型模块
    st.markdown('<div class="sidebar-header" style="margin-top: 20px;">🤖 国产大模型</div>', unsafe_allow_html=True)
    if st.button("🔥 DeepSeek", use_container_width=True, key="llm_ds"):
        st.session_state.current_page = "llm_deepseek"
        st.rerun()
    if st.button("🌟 智谱GLM", use_container_width=True, key="llm_zp"):
        st.session_state.current_page = "llm_zhipu"
        st.rerun()
    if st.button("💫 通义千问", use_container_width=True, key="llm_qwen"):
        st.session_state.current_page = "llm_qwen"
        st.rerun()
    if st.button("⭐ 文心一言", use_container_width=True, key="llm_wx"):
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

# ==================== 右侧主内容区（路由） ====================
page = st.session_state.current_page

# ----- 机器学习模块 -----
if page == "classification":
    try:
        from pages_modules import classification_lab
        classification_lab.render_classification_lab()
    except ImportError as e:
        st.error(f"加载分类模块失败: {e}")
elif page == "regression":
    try:
        from pages_modules import regression_lab
        regression_lab.render_regression_lab()
    except ImportError as e:
        st.error(f"加载回归模块失败: {e}")
elif page == "clustering":
    try:
        from pages_modules import clustering_lab
        clustering_lab.render_clustering_lab()
    except ImportError as e:
        st.error(f"加载聚类模块失败: {e}")

# ----- 神经网络模块 -----
elif page == "nn_basic":
    try:
        from pages_modules import neural_vis_module
        neural_vis_module.render_neural_network_viz()
    except ImportError as e:
        st.error(f"加载基础神经网络模块失败: {e}")
elif page == "nn_cnn":
    try:
        from pages_modules import cnn_viz_module
        cnn_viz_module.nv_render_cnn_viz()
    except ImportError as e:
        st.error(f"加载 CNN 模块失败: {e}")
elif page == "nn_rnn":
    try:
        from pages_modules import rnn_viz_module
        rnn_viz_module.nv_render_rnn_viz()
    except ImportError as e:
        st.error(f"加载 RNN 模块失败: {e}")
elif page == "nn_attention":
    try:
        from pages_modules import transformer_viz_module
        transformer_viz_module.nv_render_transformer_viz()
    except ImportError as e:
        st.error(f"加载 Transformer 模块失败: {e}")

# ----- 国产大模型模块 -----
elif page.startswith("llm_"):
    model_map = {
        "llm_deepseek": "DeepSeek",
        "llm_zhipu": "智谱GLM",
        "llm_qwen": "通义千问",
        "llm_wenxin": "文心一言"
    }
    model_name = model_map.get(page, "智谱GLM")
    try:
        from pages_modules import llm
        llm.show(model_preselected=model_name)
    except ImportError as e:
        st.error(f"加载大模型模块失败: {e}")

# ----- 首页 -----
else:
    try:
        from pages_modules import home
        home.show()
    except ImportError:
        st.title("🧠 欢迎使用算法可视化学习平台")
        st.markdown("通过拖拽组件、调整参数、观察实时变化，直观理解人工智能算法。")
        st.info("👈 请从左侧菜单选择模块开始学习。")

# ==================== 底部状态栏 ====================
st.markdown("---")
bottom_col1, bottom_col2 = st.columns([3, 1])
with bottom_col1:
    page_display = {
        "classification": "📊 分类",
        "regression": "📈 回归",
        "clustering": "🔵 聚类",
        "nn_basic": "🔷 基础神经网络动画演示",
        "nn_cnn": "🧩 卷积神经网络 (CNN)",
        "nn_rnn": "🔄 循环神经网络 (RNN)",
        "nn_attention": "👁 注意力机制（Transformer）",
        "llm_deepseek": "🔥 DeepSeek",
        "llm_zhipu": "🌟 智谱GLM",
        "llm_qwen": "💫 通义千问",
        "llm_wenxin": "⭐ 文心一言"
    }.get(page, "首页")
    if page != "home":
        st.markdown(f'<div class="status-bar">📍 当前页面：{page_display}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-bar">📍 当前位于首页</div>', unsafe_allow_html=True)
with bottom_col2:
    st.markdown('<div style="text-align: right; color: #888; font-size: 13px;">基于 Streamlit + Plotly + Scikit-learn</div>', unsafe_allow_html=True)
