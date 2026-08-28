import streamlit as st
import streamlit.components.v1 as components

from utils.helpers import resource_path


def render_header() -> None:
    """渲染页面顶部区域及其既有前端行为。"""
    st.markdown("""
<style>
    [data-testid="stSidebar"] label[data-baseweb="radio"] span {
        white-space: nowrap;
        font-size: 14px !important;
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

    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        st.image(resource_path("assets/logo.png"), width=80)
    with col2:
        st.markdown(
            '<div class="brand-title-row"><span class="school-name">智视<span style="font-size:1.15em; vertical-align:-0.05em">AI</span></span><span class="platform-title">算法可视化学习平台</span></div>',
            unsafe_allow_html=True)
    with col3:
        st.markdown(
            '<div class="header-motto">智见未来 · 学无止境</div>',
            unsafe_allow_html=True)
    st.markdown('<div class="top-gradient"></div>', unsafe_allow_html=True)
