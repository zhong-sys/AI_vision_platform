# pages_modules/home.py
import streamlit as st
import config

def show():
    """首页欢迎页面"""
    # 欢迎横幅
    st.markdown(f"""
    <div class="welcome-box">
        <div style="font-size: 36px; margin-bottom: 10px; color: {config.PRIMARY_BLUE};">🧠</div>
        <div class="welcome-title">欢迎使用算法可视化学习平台</div>
        <p style="font-size: 16px; color: #555; max-width: 600px; margin: 15px auto 0 auto;">
            通过拖拽组件、调整参数、观察实时变化，<br>
            直观理解人工智能算法的原理与工作机制。
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # 快速引导
    st.markdown("### 🚀 快速开始")
    st.markdown(f"""
    <div class="guide-card-grid">
        <div class="guide-card">
            <div style="font-size: 32px; margin-bottom: 10px;">1️⃣</div>
            <h4 style="color: {config.PRIMARY_BLUE_DARK};">选择算法模块</h4>
            <p style="color: #666; font-size: 17px;">从左侧菜单点击感兴趣的算法类别，如“分类”或“神经网络”。</p>
        </div>
        <div class="guide-card">
            <div style="font-size: 32px; margin-bottom: 10px;">2️⃣</div>
            <h4 style="color: {config.PRIMARY_BLUE_DARK};">调节参数观察变化</h4>
            <p style="color: #666; font-size: 17px;">使用滑块、下拉框调整模型参数，图表和结果会实时更新。</p>
        </div>
        <div class="guide-card">
            <div style="font-size: 32px; margin-bottom: 10px;">3️⃣</div>
            <h4 style="color: {config.PRIMARY_BLUE_DARK};">理解算法原理</h4>
            <p style="color: #666; font-size: 17px;">每个模块都配有通俗易懂的原理说明，边操作边学习。</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # 模块预览
    st.markdown("### 📌 模块预览")
    st.markdown(f"""
    <div class="preview-card-grid">
        <div class="preview-card">
            <div style="font-size: 40px; margin-bottom: 12px;">📁</div>
            <h4 style="color: {config.PRIMARY_BLUE_DARK};">机器学习</h4>
            <p style="color: #555; font-size: 17px; line-height: 1.6;">
                <b>分类</b> · 决策边界可视化<br>
                <b>回归</b> · 拟合曲线动态展示<br>
                <b>聚类</b> · 无监督学习探索
            </p>
            <div style="margin-top: 16px; color: {config.ACCENT_ORANGE}; font-size: 16px; font-weight: 500;">
                👆 点击左侧菜单开始 →
            </div>
        </div>
        <div class="preview-card">
            <div style="font-size: 40px; margin-bottom: 12px;">🧬</div>
            <h4 style="color: {config.PRIMARY_BLUE_DARK};">神经网络</h4>
            <p style="color: #555; font-size: 17px; line-height: 1.6;">
                <b>前馈/BP</b> · 结构自由搭建<br>
                <b>CNN/RNN</b> · 卷积与循环可视化<br>
                <b>注意力/GAN</b> · 前沿机制演示
            </p>
            <div style="margin-top: 16px; color: {config.ACCENT_ORANGE}; font-size: 16px; font-weight: 500;">
                👆 点击左侧菜单开始 →
            </div>
        </div>
        <div class="preview-card">
            <div style="font-size: 40px; margin-bottom: 12px;">🤖</div>
            <h4 style="color: {config.PRIMARY_BLUE_DARK};">国产大模型</h4>
            <p style="color: #555; font-size: 17px; line-height: 1.6;">
                <b>DeepSeek</b> · 深度求索<br>
                <b>智谱GLM</b> · 清华出品<br>
                <b>通义/文心</b> · 阿里&百度
            </p>
            <div style="margin-top: 16px; color: {config.ACCENT_ORANGE}; font-size: 16px; font-weight: 500;">
                👆 点击左侧菜单开始 →
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # 推荐学习路径
    st.markdown("### 🎓 推荐学习路径")
    st.markdown(f"""
    <div class="learning-path-grid">
        <div class="path-card" style="background: #f8fbfe; border-radius: 16px; padding: 20px; border: 1px solid #d0e0f0;">
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 160px;">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                        <span style="background: {config.PRIMARY_BLUE}; color: white; width: 24px; height: 24px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 17px;">1</span>
                        <span style="font-weight: bold; color: #333;">机器学习基础</span>
                    </div>
                    <p style="color: #666; font-size: 17px; margin-left: 34px;">先掌握分类、回归、聚类等经典算法。</p>
                </div>
                <div style="flex: 1; min-width: 160px;">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                        <span style="background: {config.PRIMARY_BLUE}; color: white; width: 24px; height: 24px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 17px;">2</span>
                        <span style="font-weight: bold; color: #333;">神经网络入门</span>
                    </div>
                    <p style="color: #666; font-size: 17px; margin-left: 34px;">从前馈网络开始，逐步深入CNN、RNN。</p>
                </div>
                <div style="flex: 1; min-width: 160px;">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                        <span style="background: {config.PRIMARY_BLUE}; color: white; width: 24px; height: 24px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 17px;">3</span>
                        <span style="font-weight: bold; color: #333;">前沿大模型</span>
                    </div>
                    <p style="color: #666; font-size: 17px; margin-left: 34px;">体验国产大模型，了解AI最新发展。</p>
                </div>
            </div>
        </div>
        <div class="announcement-card" style="background: white; border-radius: 16px; padding: 20px; border: 1px solid #e0e0e0;">
            <div style="font-weight: bold; color: {config.PRIMARY_BLUE_DARK}; margin-bottom: 10px;">
                📢 平台公告
                <span class="accent-tag">NEW</span>
            </div>
            <ul style="color: #555; font-size: 17px; padding-left: 20px; margin: 0;">
                <li>神经网络模块即将上线结构自定义功能</li>
                <li>大模型对话现已支持 DeepSeek</li>
                <li>更多可视化效果持续更新中</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # 平台功能
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
