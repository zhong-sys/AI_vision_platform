import streamlit as st
import streamlit.components.v1 as components
import os
import sys

def resource_path(relative_path):
    """获取资源文件的绝对路径，兼容打包后的exe环境"""
    if hasattr(sys, '_MEIPASS'):
        # 打包后，临时文件被解压在 sys._MEIPASS 中
        return os.path.join(sys._MEIPASS, relative_path)
    # 开发环境
    return os.path.join(os.path.abspath("."), relative_path)

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="智视AI · 算法可视化学习平台",
    page_icon="🧠",
    layout="wide",
    menu_items={}  # 隐藏右上角调试菜单
)

# ==================== 自定义CSS样式 ====================
st.markdown("""
<style>
    /* 主色调 */
    :root {
        --primary-blue: #1A7EC1;        /* 主蓝 */
        --primary-blue-light: #3A9BDC;   /* 浅蓝 */
        --primary-blue-dark: #0F5B9E;    /* 深蓝 */
        --accent-orange: #F39C12;    /* 点缀橙 */
        --text-dark: #222222;
        --text-gray: #555555;
        --bg-light: #F8FAFC;
        --card-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }

    .main-header {
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 15px 0;
        border-bottom: 2px solid var(--primary-blue-light);
        margin-bottom: 20px;
    }
    .school-name {
        font-family: "Microsoft YaHei", "SimHei", sans-serif;
        font-size: 32px;
        font-weight: 700;
        color: var(--primary-blue-dark);
        letter-spacing: 6px;
        white-space: nowrap;
    }
    .platform-title {
        font-size: 18px;
        color: var(--text-gray);
        margin-left: 15px;
        padding-left: 15px;
        border-left: 2px solid #ddd;
        font-weight: 400;
        white-space: nowrap;
    }
    .brand-title-row {
        display: flex;
        align-items: baseline;
        flex-wrap: nowrap;
        white-space: nowrap;
    }
    .header-motto {
        text-align: right;
        padding-top: 10px;
        color: var(--primary-blue);
        font-weight: 500;
        white-space: nowrap;
    }
    .sidebar-header {
        font-size: 16px;
        font-weight: 600;
        color: var(--primary-blue-dark);
        margin: 20px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 2px solid var(--primary-blue-light);
        letter-spacing: 0.5px;
    }
    .motto-bar {
        background: linear-gradient(135deg, var(--primary-blue-dark) 0%, var(--primary-blue) 100%);
        color: white;
        padding: 14px 20px;
        border-radius: 12px;
        margin: 20px 0;
        text-align: center;
        font-size: 15px;
        letter-spacing: 2px;
        box-shadow: var(--card-shadow);
    }
    .welcome-box {
        background: linear-gradient(135deg, #ffffff 0%, var(--bg-light) 100%);
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 24px 30px;
        text-align: center;
        box-shadow: var(--card-shadow);
    }
    .welcome-title {
        font-size: 28px;
        color: var(--primary-blue-dark);
        margin-bottom: 12px;
        font-weight: 600;
    }
    [data-testid="stHorizontalBlock"] {
        align-items: stretch;
    }
    [data-testid="stHorizontalBlock"]:has(.guide-card) [data-testid="column"],
    [data-testid="stHorizontalBlock"]:has(.preview-card) [data-testid="column"] {
        display: flex;
    }
    [data-testid="stHorizontalBlock"]:has(.guide-card) [data-testid="column"] > div,
    [data-testid="stHorizontalBlock"]:has(.preview-card) [data-testid="column"] > div,
    [data-testid="stHorizontalBlock"]:has(.guide-card) [data-testid="column"] [data-testid="stMarkdown"],
    [data-testid="stHorizontalBlock"]:has(.preview-card) [data-testid="column"] [data-testid="stMarkdown"],
    [data-testid="stHorizontalBlock"]:has(.guide-card) [data-testid="column"] [data-testid="stMarkdown"] > div,
    [data-testid="stHorizontalBlock"]:has(.preview-card) [data-testid="column"] [data-testid="stMarkdown"] > div {
        display: flex;
        width: 100%;
    }
    .guide-card-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1.75rem;
        align-items: stretch;
    }
    .preview-card-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1.75rem;
        align-items: stretch;
    }
    .learning-path-grid {
        display: grid;
        grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
        gap: 1.75rem;
        align-items: stretch;
    }
    .path-card, .announcement-card {
        box-sizing: border-box;
        height: 100%;
        width: 100%;
    }
    .guide-card {
        background: white;
        border-radius: 16px;
        padding: 24px 20px;
        box-shadow: var(--card-shadow);
        border: 1px solid #EDF2F7;
        transition: all 0.2s ease;
        box-sizing: border-box;
        flex-direction: column;
        height: 100%;
        width: 100%;
    }
    .guide-card:hover {
        box-shadow: 0 8px 20px rgba(26, 126, 193, 0.08);
        border-color: var(--primary-blue-light);
        transform: translateY(-2px);
    }
    .preview-card {
        background: white;
        padding: 24px 20px;
        border-radius: 16px;
        box-shadow: var(--card-shadow);
        border: 1px solid #EDF2F7;
        box-sizing: border-box;
        flex-direction: column;
        height: 100%;
        width: 100%;
        transition: all 0.2s ease;
    }
    .preview-card:hover {
        box-shadow: 0 8px 20px rgba(26, 126, 193, 0.08);
        border-color: var(--primary-blue-light);
    }
    @media (max-width: 720px) {
        .guide-card-grid,
        .preview-card-grid,
        .learning-path-grid {
            grid-template-columns: 1fr;
        }
    }
    @media (max-width: 1100px) {
        .header-motto {
            display: none;
        }
    }
    .accent-tag {
        display: inline-block;
        background: var(--accent-orange);
        color: white;
        font-size: 12px;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 20px;
        margin-left: 8px;
    }
    .status-bar {
        background: var(--bg-light);
        border-radius: 30px;
        padding: 8px 24px;
        color: var(--primary-blue-dark);
        font-size: 14px;
        border: 1px solid #E2E8F0;
        display: inline-block;
    }
    .more-link {
        color: var(--accent-orange);
        font-weight: 500;
        text-decoration: none;
        font-size: 14px;
    }
    .more-link:hover {
        color: #E67E22;
    }
</style>
""", unsafe_allow_html=True)
components.html(
    """
    <script>
    (() => {
        const win = window.parent;
        const doc = window.parent.document;
        if (doc.__zhishiAiCopyShortcutGuard) return;
        doc.__zhishiAiCopyShortcutGuard = true;

        const isEditableTarget = (target) => {
            if (!target) return false;
            const element = target.nodeType === Node.ELEMENT_NODE
                ? target
                : target.parentElement;
            if (!element) return false;
            return Boolean(element.closest(
                "input, textarea, select, [contenteditable='true'], [role='textbox']"
            ));
        };

        const blockCacheShortcut = (event) => {
            const key = (event.key || "").toLowerCase();
            if (key === "c" && !isEditableTarget(event.target)) {
                event.preventDefault();
                event.stopImmediatePropagation();
            }
        };

        win.addEventListener("keydown", blockCacheShortcut, true);
        win.addEventListener("keyup", blockCacheShortcut, true);
        doc.addEventListener("keydown", blockCacheShortcut, true);
        doc.addEventListener("keyup", blockCacheShortcut, true);
    })();
    </script>
    """,
    height=0,
    width=0,
)

# ==================== 顶部标题栏 ====================
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.markdown(
        '<div class="school-badge"><span style="font-size:36px;">🌊</span><span style="font-size:36px;">🏛️</span></div>',
        unsafe_allow_html=True)
with col2:
    st.markdown(
        '<div class="brand-title-row"><span class="school-name">智视<span style="font-size:1.15em; vertical-align:-0.05em">AI</span></span><span class="platform-title">算法可视化学习平台</span></div>',
        unsafe_allow_html=True)
with col3:
    st.markdown(
        '<div class="header-motto">智见未来 · 学无止境</div>',
        unsafe_allow_html=True)
st.markdown('<div style="height: 2px; background: linear-gradient(90deg, var(--primary-blue), var(--primary-blue-light), var(--primary-blue));"></div>',
            unsafe_allow_html=True)

# ==================== 左侧导航栏 ====================
with st.sidebar:
    # 校徽图片显示
    # 校徽图片与校名（使用 st.image + 列居中，绝对对齐）
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        logo_path = resource_path("assets/logo.png")
        st.image(logo_path, width=240)
    st.markdown(
        '<p style="color: #1A7EC1; font-weight: bold; text-align: center; margin-top: 5px; margin-bottom: 0px;">智视<span style="font-size:1.15em; margin-left: 3px;">AI</span></p>',
        unsafe_allow_html=True)
    st.markdown("---")

    # 模块一：机器学习
    st.markdown('<div class="sidebar-header">📁 机器学习</div>', unsafe_allow_html=True)
    ml_option = st.radio(
        "",
        ["📊 分类", "📈 回归", "🔵 聚类"],
        label_visibility="collapsed",
        index=None,
        key="ml_radio"
    )

    # 模块二：神经网络
    st.markdown('<div class="sidebar-header" style="margin-top: 20px;">🧬 神经网络</div>', unsafe_allow_html=True)
    nn_option = st.radio(
        "",
        ["🔷 前馈神经网络", "🔶 BP神经网络", "🧩 卷积神经网络 (CNN)",
         "🔄 循环神经网络 (RNN)", "👁 注意力机制", "🏛️ 深度神经网络 (DNN)", "⚡ 生成对抗网络 (GAN)"],
        label_visibility="collapsed",
        index=None,
        key="nn_radio"
    )

    # 模块三：大模型
    st.markdown('<div class="sidebar-header" style="margin-top: 20px;">🤖 国产大模型</div>', unsafe_allow_html=True)
    llm_option = st.radio(
        "",
        ["🔥 DeepSeek", "🌟 智谱GLM", "💫 通义千问", "⭐ 文心一言"],
        label_visibility="collapsed",
        index=None,
        key="llm_radio"
    )

    # 平台理念展示
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
        © 2026 智视AI · 算法可视化学习平台
    </div>
    """, unsafe_allow_html=True)

# ==================== 右侧主内容区 ====================
# 欢迎横幅
# 欢迎横幅
st.markdown("""
<div class="welcome-box">
    <div style="font-size: 36px; margin-bottom: 10px; color: #1A7EC1;">🧠</div>
    <div class="welcome-title">欢迎使用算法可视化学习平台</div>
    <p style="font-size: 16px; color: #555; max-width: 600px; margin: 15px auto 0 auto;">
        通过拖拽组件、调整参数、观察实时变化，<br>
        直观理解人工智能算法的原理与工作机制。
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# 快速引导区
st.markdown("### 🚀 快速开始")
st.markdown("""
<div class="guide-card-grid">
    <div class="guide-card">
        <div style="font-size: 32px; margin-bottom: 10px;">1️⃣</div>
        <h4 style="color: var(--primary-blue-dark); margin-bottom: 8px;">选择算法模块</h4>
        <p style="color: #666; font-size: 14px;">从左侧菜单点击感兴趣的算法类别，如“分类”或“神经网络”。</p>
    </div>
    <div class="guide-card">
        <div style="font-size: 32px; margin-bottom: 10px;">2️⃣</div>
        <h4 style="color: var(--primary-blue-dark); margin-bottom: 8px;">调节参数观察变化</h4>
        <p style="color: #666; font-size: 14px;">使用滑块、下拉框调整模型参数，图表和结果会实时更新。</p>
    </div>
    <div class="guide-card">
        <div style="font-size: 32px; margin-bottom: 10px;">3️⃣</div>
        <h4 style="color: var(--primary-blue-dark); margin-bottom: 8px;">理解算法原理</h4>
        <p style="color: #666; font-size: 14px;">每个模块都配有通俗易懂的原理说明，边操作边学习。</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

# 模块预览卡片
st.markdown("### 📌 模块预览")
st.markdown("""
<div class="preview-card-grid">
    <div class="preview-card">
        <div style="font-size: 40px; margin-bottom: 12px;">📁</div>
        <h4 style="color: var(--primary-blue-dark); margin-bottom: 12px;">机器学习</h4>
        <p style="color: #555; font-size: 14px; line-height: 1.6;">
            <b>分类</b> · 决策边界可视化<br>
            <b>回归</b> · 拟合曲线动态展示<br>
            <b>聚类</b> · 无监督学习探索
        </p>
        <div style="margin-top: 16px; color: var(--accent-orange); font-size: 13px; font-weight: 500;">
            👆 点击左侧菜单开始 →
        </div>
    </div>
    <div class="preview-card">
        <div style="font-size: 40px; margin-bottom: 12px;">🧬</div>
        <h4 style="color: var(--primary-blue-dark); margin-bottom: 12px;">神经网络</h4>
        <p style="color: #555; font-size: 14px; line-height: 1.6;">
            <b>前馈/BP</b> · 结构自由搭建<br>
            <b>CNN/RNN</b> · 卷积与循环可视化<br>
            <b>注意力/GAN</b> · 前沿机制演示
        </p>
        <div style="margin-top: 16px; color: var(--accent-orange); font-size: 13px; font-weight: 500;">
            👆 点击左侧菜单开始 →
        </div>
    </div>
    <div class="preview-card">
        <div style="font-size: 40px; margin-bottom: 12px;">🤖</div>
        <h4 style="color: var(--primary-blue-dark); margin-bottom: 12px;">国产大模型</h4>
        <p style="color: #555; font-size: 14px; line-height: 1.6;">
            <b>DeepSeek</b> · 深度求索<br>
            <b>智谱GLM</b> · 清华出品<br>
            <b>通义/文心</b> · 阿里&百度
        </p>
        <div style="margin-top: 16px; color: var(--accent-orange); font-size: 13px; font-weight: 500;">
            👆 点击左侧菜单开始 →
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

# 推荐学习路径
st.markdown("### 🎓 推荐学习路径")
st.markdown("""
<div class="learning-path-grid">
    <div class="path-card" style="background: #f8fbfe; border-radius: 16px; padding: 20px; border: 1px solid #d0e0f0;">
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 160px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                    <span style="background: var(--primary-blue); color: white; width: 24px; height: 24px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 14px;">1</span>
                    <span style="font-weight: bold; color: #333;">机器学习基础</span>
                </div>
                <p style="color: #666; font-size: 14px; margin-left: 34px;">先掌握分类、回归、聚类等经典算法。</p>
            </div>
            <div style="flex: 1; min-width: 160px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                    <span style="background: var(--primary-blue); color: white; width: 24px; height: 24px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 14px;">2</span>
                    <span style="font-weight: bold; color: #333;">神经网络入门</span>
                </div>
                <p style="color: #666; font-size: 14px; margin-left: 34px;">从前馈网络开始，逐步深入CNN、RNN。</p>
            </div>
            <div style="flex: 1; min-width: 160px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                    <span style="background: var(--primary-blue); color: white; width: 24px; height: 24px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 14px;">3</span>
                    <span style="font-weight: bold; color: #333;">前沿大模型</span>
                </div>
                <p style="color: #666; font-size: 14px; margin-left: 34px;">体验国产大模型，了解AI最新发展。</p>
            </div>
        </div>
    </div>
    <div class="announcement-card" style="background: white; border-radius: 16px; padding: 20px; border: 1px solid #e0e0e0;">
        <div style="font-weight: bold; color: var(--primary-blue-dark); margin-bottom: 10px;">
            📢 平台公告
            <span class="accent-tag">NEW</span>
        </div>
        <ul style="color: #555; font-size: 14px; padding-left: 20px; margin: 0;">
            <li>神经网络模块即将上线结构自定义功能</li>
            <li>大模型对话现已支持 DeepSeek</li>
            <li>更多可视化效果持续更新中</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

# 平台功能介绍
st.markdown("---")
st.markdown("### 🎯 平台功能")

feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
with feat_col1:
    st.markdown("**🖱️ 拖拽组件**<br><small>自由组合算法模块</small>", unsafe_allow_html=True)
with feat_col2:
    st.markdown("**⚙️ 参数调节**<br><small>实时调整模型参数</small>", unsafe_allow_html=True)
with feat_col3:
    st.markdown("**📊 可视化展示**<br><small>动态呈现算法效果</small>", unsafe_allow_html=True)
with feat_col4:
    st.markdown("**📝 原理说明**<br><small>深入浅出讲解原理</small>", unsafe_allow_html=True)

# 底部状态栏与版权信息
st.markdown("---")
bottom_col1, bottom_col2 = st.columns([3, 1])

with bottom_col1:
    selected_items = []
    if ml_option: selected_items.append(ml_option)
    if nn_option: selected_items.append(nn_option)
    if llm_option: selected_items.append(llm_option)

    if selected_items:
        current_selection = " → ".join(selected_items)
        st.markdown(f'<div class="status-bar">📍 当前选中：{current_selection}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-bar">📍 当前未选择任何模块，请从左侧菜单开始</div>', unsafe_allow_html=True)

with bottom_col2:
    st.markdown("""
    <div style="text-align: right; color: #888; font-size: 13px;">
        基于 Streamlit 平台开发
    </div>
    """, unsafe_allow_html=True)
