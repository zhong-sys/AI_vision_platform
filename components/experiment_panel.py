"""Reusable, deterministic teaching panels for experiment pages.

The helpers in this module are intentionally presentation-only.  They do not
run models, make network requests, or mutate page state.  Individual pages
remain responsible for applying presets and restoring their own parameters.
"""

from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

import streamlit as st


def _as_lines(items: Union[str, Iterable[str]]) -> Sequence[str]:
    """Return user-facing teaching content as a predictable sequence."""
    if isinstance(items, str):
        return [items]
    return [str(item) for item in items]


def render_teaching_card(title: str, body: Union[str, Iterable[str]], *, expanded: bool = False) -> None:
    """Render a compact, consistently styled teaching card."""
    with st.expander(title, expanded=expanded):
        for line in _as_lines(body):
            st.markdown(line if line.startswith("-") else "- {0}".format(line))


def render_learning_goals(goals: Iterable[str], *, title: str = "🎯 学习目标", expanded: bool = False) -> None:
    """Render concise learning goals without changing the original lesson text."""
    render_teaching_card(title, goals, expanded=expanded)


def render_experiment_guide(steps: Iterable[str], *, title: str = "🧭 推荐实验步骤", expanded: bool = False) -> None:
    """Render the recommended sequence for an experiment."""
    with st.expander(title, expanded=expanded):
        for index, step in enumerate(_as_lines(steps), start=1):
            st.markdown("{0}. {1}".format(index, step))


def render_parameter_explanation(
    explanations: Union[str, Iterable[str]],
    *,
    title: str = "⚙️ 参数动态解释",
    expanded: bool = False,
) -> None:
    """Render deterministic explanations supplied by the current page."""
    render_teaching_card(title, explanations, expanded=expanded)


def render_observation(
    observation: Union[str, Iterable[str]],
    *,
    title: str = "📌 当前观察 / 观察建议",
    expanded: bool = False,
) -> None:
    """Render current metrics and cautious observation suggestions."""
    with st.expander(title, expanded=expanded):
        for line in _as_lines(observation):
            st.info(line)


def render_preset_controls(
    presets: Union[Mapping[str, object], Sequence[str]],
    *,
    key: str,
    reset_key: str,
    on_preset_change: Optional[Callable[[], None]] = None,
    on_reset: Optional[Callable[[], None]] = None,
    title: str = "🧪 参数预设实验",
    reset_label: str = "恢复默认参数",
) -> str:
    """Render a page-scoped preset selector and reset action.

    Callbacks run in Streamlit's pre-rerun phase, so pages can safely assign
    their existing widget state there without calling ``session_state.clear``.
    """
    names = list(presets.keys()) if isinstance(presets, Mapping) else list(presets)
    if not names:
        return ""
    current = st.session_state.get(key, names[0])
    if current not in names:
        current = names[0]
    with st.expander(title, expanded=False):
        selected = st.selectbox(
            "选择教学预设",
            names,
            index=names.index(current),
            key=key,
            on_change=on_preset_change,
        )
        if isinstance(presets, Mapping):
            description = presets.get(selected)
            if description:
                st.caption(str(description))
        st.caption("预设只调整当前页面已有控件，便于重复观察典型现象。")
        st.button(
            reset_label,
            key=reset_key,
            use_container_width=True,
            on_click=on_reset,
        )
    return selected


def render_experiment_overview(
    goals: Iterable[str],
    steps: Iterable[str],
    *,
    expanded: bool = False,
) -> None:
    """Render the two standard introductory teaching cards."""
    render_learning_goals(goals, expanded=expanded)
    render_experiment_guide(steps, expanded=expanded)
