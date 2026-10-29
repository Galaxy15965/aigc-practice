import jieba
from rank_bm25 import BM25Okapi
from pypdf import PdfReader
from openai import OpenAI
from chunking import split_text
from config import API_KEY

import re

def page_of(chunk):
    m = re.search(r"=== 第(\d+)页 ===", chunk)
    return int(m.group(1)) if m else 0


client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

reader = PdfReader("data/xiezuo1.pdf")
text = ""
for i, page in enumerate(reader.pages, 1):
    text += f"\n\n=== 第{i}页 ===\n" + (page.extract_text() or "")

chunks = split_text(text)
print(f"已索引 {len(chunks)} 段")

bm25 = BM25Okapi([list(jieba.cut(c)) for c in chunks])

SYSTEM = """你是文档问答助手,严格遵守三条规则:
1. 只根据【参考资料】回答,不使用你自身的知识
2. 资料里没有的信息,直接回答"文档中没有提到"
3. 回答末尾用一行注明依据的页码"""

while True:
    q = input("\n提问(输入 exit 退出):").strip()
    if q == "exit":
        break

    scores = bm25.get_scores(list(jieba.cut(q)))
    top5 = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:5]
    print("召回页码:", [page_of(chunks[i]) for i in top5], "得分:", [round(scores[i], 2) for i in top5])
    context = "\n\n".join(chunks[i] for i in top5)

    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"【参考资料】\n{context}\n\n【问题】{q}"},
        ],
    )
    print("\n" + r.choices[0].message.content)