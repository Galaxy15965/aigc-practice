from __future__ import annotations

import streamlit as st

from agent import DocumentAgent, build_default_client
from retrieval import DocumentIndex


st.set_page_config(page_title="课程资料 Agent", page_icon="🧭", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 1080px; padding-top: 2.2rem; }
    [data-testid="stChatMessage"] { border-radius: 16px; }
    .agent-kicker { color:#9a6b2f; font-size:0.78rem; font-weight:700; letter-spacing:.09em; }
    .agent-title { font-size:2.2rem; font-weight:800; color:#17212b; margin:.2rem 0 .1rem; }
    .agent-sub { color:#6b7175; margin-bottom:1.2rem; }
    .trace-note { color:#7a807f; font-size:.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_agent() -> tuple[DocumentAgent, DocumentIndex]:
    index = DocumentIndex.from_pdf(data_dir="data")
    client = build_default_client()
    return DocumentAgent(index=index, client=client, max_steps=6), index


st.markdown('<div class="agent-kicker">RAG → AGENT</div>', unsafe_allow_html=True)
st.markdown('<div class="agent-title">课程资料 Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="agent-sub">先检索，再回答；支持页码核对、引用校验和工具轨迹回放。</div>',
    unsafe_allow_html=True,
)

try:
    agent, index = load_agent()
except Exception as exc:
    st.error(f"初始化失败：{exc}")
    st.stop()

with st.sidebar:
    st.header("运行状态")
    st.metric("已索引页数", index.page_count)
    st.metric("文本块", index.chunk_count)
    st.caption("检索：页感知 BM25")
    st.caption("工具：search_docs / read_page")
    st.caption("模型：deepseek-chat")
    st.divider()
    st.markdown("**规则**")
    st.markdown("- 只依据检索到的课程资料\n- 没有依据时明确拒答\n- 引用页码会与工具访问记录核对")

if "agent_history" not in st.session_state:
    st.session_state.agent_history = []


def render_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.markdown(turn["question"])

    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        if turn.get("uncertain_pages"):
            st.warning(f"以下引用页码未被工具访问：{turn['uncertain_pages']}")

        meta = f"工具调用 {turn['tool_calls']} 次 · 执行 {turn['steps']} 轮 · 访问页码 {turn['accessed_pages']}"
        st.caption(meta)

        with st.expander("查看 Agent 工具轨迹"):
            trace = turn.get("trace", [])
            if trace:
                rows = [
                    {
                        "轮次": item.get("step"),
                        "工具": item.get("tool"),
                        "页码": item.get("pages"),
                        "命中数": item.get("count"),
                        "最高分": item.get("best_score"),
                        "耗时(ms)": item.get("duration_ms"),
                        "状态": "成功" if item.get("ok") else f"失败：{item.get('error')}",
                    }
                    for item in trace
                ]
                st.dataframe(rows, use_container_width=True)
                st.caption("原始工具结果")
                st.json(trace)
            else:
                st.caption("本轮没有调用工具。")


for item in st.session_state.agent_history:
    render_turn(item)

question = st.chat_input("例如：调查报告的写作一般经历哪几个阶段？")
if question:
    with st.spinner("Agent 正在检索并核对资料…"):
        try:
            result = agent.ask(question)
            st.session_state.agent_history.append(
                {
                    "question": question,
                    "answer": result.answer,
                    "tool_calls": result.tool_calls,
                    "steps": result.steps,
                    "accessed_pages": result.accessed_pages,
                    "uncertain_pages": result.unverified_pages,
                    "trace": result.trace,
                }
            )
            st.rerun()
        except Exception as exc:
            st.error(f"本轮执行失败：{exc}")
