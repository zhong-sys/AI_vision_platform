"""Shared, behaviour-neutral labels and route metadata.

Keeping these values in one place prevents the sidebar, footer and router from
drifting apart while leaving each page's existing controls and callbacks
unchanged.
"""

PAGE_KEYS = (
    "home",
    "classification",
    "regression",
    "clustering",
    "nn_basic",
    "nn_cnn",
    "nn_rnn",
    "nn_attention",
    "llm_deepseek",
    "llm_zhipu",
    "llm_qwen",
    "llm_wenxin",
)

PAGE_DISPLAY_NAMES = {
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
    "llm_wenxin": "⭐ 文心一言",
}

LLM_MODEL_MAP = {
    "llm_deepseek": "DeepSeek",
    "llm_zhipu": "智谱GLM",
    "llm_qwen": "通义千问",
    "llm_wenxin": "文心一言",
}
