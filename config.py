import streamlit as st

PRIMARY_BLUE = "#1A7EC1"
PRIMARY_BLUE_LIGHT = "#3A9BDC"
PRIMARY_BLUE_DARK = "#0F5B9E"
ACCENT_ORANGE = "#F39C12"

def get_api_key(provider: str) -> str:
    """从 secrets.toml 安全获取 API Key"""
    key_map = {
        "DeepSeek": "DEEPSEEK_API_KEY",
        "智谱GLM": "ZHIPU_API_KEY",
        "通义千问": "ALIYUN_API_KEY",
        "文心一言": "BAIDU_API_KEY"
    }
    key_name = key_map.get(provider, "")
    if key_name:
        return st.secrets.get(key_name, "")
    return ""

# ==================== 全局 CSS 样式 ====================
GLOBAL_CSS = """
<style>
    /* 主色调 */
    :root {
        --primary-blue: #1A7EC1;
        --primary-blue-light: #3A9BDC;
        --primary-blue-dark: #0F5B9E;
        --accent-orange: #F39C12;
    }

    .main-header {
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 15px 0;
        border-bottom: 2px solid #3A9BDC;
        margin-bottom: 20px;
    }
    .school-name {
        font-family: "Microsoft YaHei", "SimHei", sans-serif;
        font-size: 32px;
        font-weight: 700;
        color: #0F5B9E;
        letter-spacing: 6px;
        white-space: nowrap;
    }
    .platform-title {
        font-size: 18px;
        color: #555555;
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
        color: #1A7EC1;
        white-space: nowrap;
    }
    .sidebar-header {
        font-size: 16px;
        font-weight: 600;
        color: #0F5B9E;
        margin: 20px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 2px solid #3A9BDC;
        letter-spacing: 0.5px;
    }
    .motto-bar {
        background: linear-gradient(135deg, #0F5B9E 0%, #1A7EC1 100%);
        color: white;
        padding: 14px 20px;
        border-radius: 12px;
        margin: 20px 0;
        text-align: center;
        font-size: 17px;
        letter-spacing: 2px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    .welcome-box {
        background: linear-gradient(135deg, #ffffff 0%, #F8FAFC 100%);
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 28px 30px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    .welcome-title {
        font-size: 28px;
        color: #0F5B9E;
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
    .guide-card, .preview-card {
        background: white;
        border-radius: 16px;
        padding: 24px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        border: 1px solid #EDF2F7;
        transition: all 0.2s ease;
        box-sizing: border-box;
        flex-direction: column;
        height: 100%;
        width: 100%;
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
    .guide-card:hover, .preview-card:hover {
        box-shadow: 0 8px 20px rgba(26, 126, 193, 0.08);
        border-color: #3A9BDC;
        transform: translateY(-2px);
    }
    .accent-tag {
        display: inline-block;
        background: #F39C12;
        color: white;
        font-size: 17px;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 20px;
        margin-left: 8px;
    }
    .status-bar {
        background: #F8FAFC;
        border-radius: 30px;
        padding: 8px 24px;
        color: #0F5B9E;
        font-size: 17px;
        border: 1px solid #E2E8F0;
        display: inline-block;
    }
    .top-gradient {
        height: 2px;
        background: linear-gradient(90deg, #1A7EC1, #3A9BDC, #1A7EC1);
    }
    /* 侧栏 Logo 透明背景 */
    [data-testid="stSidebar"] [data-testid="stImage"] {
        background: transparent !important;
    }
    [data-testid="stSidebar"] [data-testid="stImage"] img {
        background: transparent !important;
    }
</style>
"""
