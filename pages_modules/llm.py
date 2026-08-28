# pages_modules/llm.py
import logging
import streamlit as st
from openai import OpenAI
import config
from utils.helpers import resource_path
import os
import time
from threading import BoundedSemaphore

from components.experiment_panel import (
    render_experiment_guide,
    render_learning_goals,
    render_observation,
    render_parameter_explanation,
)
from components.page_header import render_page_header


LOGGER = logging.getLogger(__name__)


# ==================== 模型介绍信息库（保持不变） ====================
MODEL_INTROS = {
    "DeepSeek": {
        "name": "DeepSeek",
        "developer": "深度求索",
        "description": "DeepSeek 是国内领先的通用大模型，以其强大的推理能力和极高的性价比著称。DeepSeek-V3 在数学、代码生成等任务上表现优异，DeepSeek-R1 则专注于复杂的逻辑推理。",
        "features": ["上下文长度 64K", "支持联网搜索", "API 完全兼容 OpenAI 格式"],
        "free_quota": "新用户注册赠送 500万 Tokens，有效期30天。",
        "website": "https://platform.deepseek.com",
        "color": "#1A7EC1",
        "logo_path": "assets/logos/deepseek_logo.png"
    },
    "智谱GLM": {
        "name": "智谱GLM-4-Flash",
        "developer": "智谱AI（清华大学技术成果转化）",
        "description": "GLM-4-Flash 是智谱AI推出的免费轻量级大模型，兼顾性能与效率。它基于GLM-4架构优化，适合日常对话、文本生成等场景，特别适合教学和原型开发。",
        "features": ["上下文长度 128K", "支持函数调用", "永久免费，无额度限制", "并发限制为30"],
        "free_quota": "GLM-4-Flash 模型永久免费，无Token额度限制。",
        "website": "https://bigmodel.cn",
        "color": "#3A9BDC",
        "logo_path": "assets/logos/zhipu_logo.png"
    },
    "通义千问": {
        "name": "通义千问 Qwen-Code",
        "developer": "阿里巴巴",
        "description": "通义千问是阿里云自研的大语言模型系列，其中Qwen-Code是专为代码生成和理解优化的版本，在编程教学、代码调试场景下表现出色。",
        "features": ["上下文长度 32K", "代码生成能力强", "支持多种编程语言"],
        "free_quota": "新用户每模型赠送 100万 Tokens，或每日2000次免费调用。",
        "website": "https://bailian.console.aliyun.com",
        "color": "#FF6B00",
        "logo_path": "assets/logos/qwen_logo.png"
    },
    "文心一言": {
        "name": "文心一言 4.5",
        "developer": "百度",
        "description": "文心一言是百度基于文心大模型打造的对话式AI产品，在中文理解、知识问答、内容创作等方面有深厚积累，对中文语境支持友好。",
        "features": ["上下文长度 8K", "中文理解能力强", "支持插件扩展"],
        "free_quota": "新用户注册赠送 50万 Tokens，有效期30天。",
        "website": "https://cloud.baidu.com",
        "color": "#2468F2",
        "logo_path": "assets/logos/wenxin_logo.png"
    }
}

MODEL_CONFIGS = {
    "DeepSeek": {
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
    },
    "智谱GLM": {
        "model": "glm-4-flash",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
    },
    "通义千问": {
        "model": "qwen-turbo",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "文心一言": {
        "model": "ernie-4.0-8k",
        "base_url": "https://qianfan.baidubce.com/v2",
    }
}

# ==================== 稳定性参数 ====================
REQUEST_TIMEOUT_SECONDS = 45
MAX_RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1.5
MAX_HISTORY_MESSAGES = 20
LLM_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "8"))
LLM_ACQUIRE_TIMEOUT_SECONDS = 20

# 进程级并发闸门：多用户同时访问时保护上游 API
LLM_REQUEST_SEMAPHORE = BoundedSemaphore(LLM_MAX_CONCURRENCY)


def _build_recent_messages(messages, max_count=MAX_HISTORY_MESSAGES):
    """裁剪上下文，避免超长历史导致超时或失败。"""
    if len(messages) <= max_count:
        return messages
    return messages[-max_count:]


def _request_with_retry(client: OpenAI, model: str, messages, temperature: float, max_tokens: int):
    """带指数退避的流式请求（生成器），逐块 yield 文本，失败后自动重试。"""
    last_error = None
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            return  # 流式成功完成
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRY_ATTEMPTS:
                delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                st.warning(f"请求失败，正在重试（{attempt}/{MAX_RETRY_ATTEMPTS - 1}），约 {delay:.1f}s 后继续…")
                time.sleep(delay)

    # 流式连续失败后，降级到非流式一次性请求
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        result = completion.choices[0].message.content or ""
        yield result
    except Exception:
        raise last_error if last_error else RuntimeError("模型请求失败")


def show(model_preselected: str = None):
    """国产大模型对话页面（控件移至主界面）"""
    render_page_header(
        title="🤖 国产大模型体验区",
        module="大模型 / 对话体验",
        description="了解模型背景，然后输入问题，体验国产AI的能力。",
        back_key="llm_back_home_top",
    )

    render_learning_goals(
        [
            "了解不同国产大模型的定位、能力侧重点和适用场景。",
            "理解 Temperature 与 Max Tokens 对对话输出的影响方向。",
            "练习用清晰问题描述任务，并比较不同模型的回答风格。",
            "掌握 API Key 缺失或服务不可用时的安全提示边界。",
        ]
    )
    render_experiment_guide(
        [
            "先阅读当前模型介绍，再写下问题目标和期望输出格式。",
            "保持问题不变，只调整 Temperature 或 Max Tokens 做对照。",
            "比较回答的稳定性、长度和任务完成度，记录可复现观察。",
            "实验结束后清空对话，避免把无关上下文带入下一轮。",
        ]
    )


    # 初始化会话状态
    if "llm_messages" not in st.session_state:
        st.session_state.llm_messages = []

    # 确定默认模型
    model_list = list(MODEL_CONFIGS.keys())
    default_index = 0
    if model_preselected and model_preselected in model_list:
        default_index = model_list.index(model_preselected)

    # ==================== 顶部控制栏（模型选择与参数） ====================
    with st.container():
        st.markdown("### ⚙️ 模型设置")
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            model_choice = st.selectbox("选择模型", model_list, index=default_index)
        with col2:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1,
                help="控制生成文本的**随机性**。值越低（接近 0），输出越确定、保守；值越高（接近 1），输出越有创造性和多样性。")
        with col3:
            max_tokens = st.slider("Max Tokens", 50, 2000, 1024, 50,
                help="限制模型**单次回复的最大长度**。1 个 token 约等于 0.5~1 个中文字符或 0.75 个英文单词。值越大，回复可以越长。")
        with col4:
            st.write("")  # 对齐占位
            st.write("")
            if st.button("🧹 清空对话", use_container_width=True):
                st.session_state.llm_messages = []
                st.rerun()

    render_parameter_explanation(
        [
            "Temperature={0:.1f}：数值较低时输出通常更稳定，数值较高时表达可能更丰富。".format(temperature),
            "Max Tokens={0}：限制单次回复的最大长度，实际输出仍取决于问题和模型。".format(max_tokens),
            "当前模型为 {0}，建议保持问题不变再做单变量对照。".format(model_choice),
        ]
    )
    render_observation(
        [
            "对话状态保存在当前会话中，不会新增持久化存储。",
            "建议记录可复现的问题与参数；若服务不可用，请根据页面提示稍后重试。",
        ]
    )

    # ==================== 模型介绍区域（可折叠） ====================
    intro = MODEL_INTROS.get(model_choice, MODEL_INTROS["智谱GLM"])
    with st.expander(f"📖 关于 {model_choice}", expanded=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {intro['color']}10 0%, white 100%); 
                        padding: 15px; border-radius: 12px; border-left: 4px solid {intro['color']};">
                <h3 style="margin-top: 0; color: {intro['color']};">{intro['name']}</h3>
                <p><strong>开发机构：</strong>{intro['developer']}</p>
                <p>{intro['description']}</p>
                <p><strong>主要特点：</strong></p>
                <ul>
                    {"".join(f"<li>{feat}</li>" for feat in intro['features'])}
                </ul>
                <p><strong>💰 免费额度：</strong>{intro['free_quota']}</p>
                <p>🔗 <a href="{intro['website']}" target="_blank">访问官网</a></p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            logo_path = resource_path(intro['logo_path'])
            if os.path.exists(logo_path):
                st.image(logo_path, width=150)
            else:
                st.warning(f"⚠️ Logo未找到: {intro['logo_path']}")

    st.divider()

    # ==================== API 检查 ====================
    api_key = config.get_api_key(model_choice)
    if not api_key:
        st.error(f"❌ 未配置 {model_choice} 的 API Key。请在 `.streamlit/secrets.toml` 中添加。")
        st.info(f"💡 申请免费 API Key：访问 {intro['website']} 注册即可获取。")
        # st.stop()
        st.markdown("---")
        if st.button("🏠 返回首页", key="llm_back_home"):
            st.session_state.current_page = "home"
            st.rerun()
        st.markdown("---")
        return False  # 直接返回，不执行后面的对话逻辑

    client = OpenAI(
        api_key=api_key,
        base_url=MODEL_CONFIGS[model_choice]["base_url"],
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    # ==================== 聊天记录显示 ====================
    for msg in st.session_state.llm_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ==================== 用户输入 ====================
    if prompt := st.chat_input(f"向 {model_choice} 提问..."):
        st.session_state.llm_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                acquired = LLM_REQUEST_SEMAPHORE.acquire(timeout=LLM_ACQUIRE_TIMEOUT_SECONDS)
                if not acquired:
                    st.error("当前并发请求较高，请稍后重试。")
                    return

                request_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in _build_recent_messages(st.session_state.llm_messages)
                ]
                full_response = ""
                for delta in _request_with_retry(
                        client=client,
                        model=MODEL_CONFIGS[model_choice]["model"],
                        messages=request_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                ):
                    full_response += delta
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.llm_messages.append({"role": "assistant", "content": full_response})
            except Exception:
                LOGGER.exception("LLM request failed for provider %s", model_choice)
                st.error("调用模型时出错，请稍后重试。")
            finally:
                if 'acquired' in locals() and acquired:
                    LLM_REQUEST_SEMAPHORE.release()
