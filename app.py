import re
import jieba
import streamlit as st
from rank_bm25 import BM25Okapi
from pypdf import PdfReader
from openai import OpenAI
from chunking import split_text
from config import API_KEY

st.set_page_config(page_title="课程资料问答", page_icon="📚")

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

SYSTEM = """你是文档问答助手,严格遵守三条规则:
1. 只根据【参考资料】回答,不使用你自身的知识
2. 资料里没有的信息,直接回答"文档中没有提到"
3. 回答末尾用一行注明依据的页码"""


@st.cache_resource
def load_index():
    reader = PdfReader("data/xiezuo1.pdf")
    text = ""
    for i, page in enumerate(reader.pages, 1):
        text += f"\n\n=== 第{i}页 ===\n" + (page.extract_text() or "")
    chunks = split_text(text)
    bm25 = BM25Okapi([list(jieba.cut(c)) for c in chunks])
    return chunks, bm25


def expand_query(q):
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "把用户的问题改写成3个不同说法,覆盖同义词和不同角度。只输出3行,不要编号,不要解释。"},
            {"role": "user", "content": q},
        ],
        temperature=0.7,
    )
    return [q] + [l.strip() for l in r.choices[0].message.content.split("\n") if l.strip()][:3]


def retrieve(q, bm25, k=5):
    best = {}
    for v in expand_query(q):
        s = bm25.get_scores(list(jieba.cut(v)))
        for i, sc in enumerate(s):
            if sc > best.get(i, 0):
                best[i] = sc
    return sorted(best, key=lambda i: best[i], reverse=True)[:k]


chunks, bm25 = load_index()

st.title("📚 课程资料问答")
st.caption(f"已索引 {len(chunks)} 段")

if "history" not in st.session_state:
    st.session_state.history = []

q = st.chat_input("问点什么……")

if q:
    with st.spinner("检索并思考中……"):
        top = retrieve(q, bm25)
        context = "\n\n".join(chunks[i] for i in top)
        r = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"【参考资料】\n{context}\n\n【问题】{q}"},
            ],
        )
        answer = r.choices[0].message.content
        pages = sorted({int(x) for x in re.findall(r"=== 第(\d+)页 ===", context)})

    st.session_state.history.append({"q": q, "a": answer, "pages": pages})

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"])
        with st.expander("查看检索到的原文页码"):
            st.write(turn["pages"])