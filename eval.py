import json
import re
import csv
import jieba
from rank_bm25 import BM25Okapi
from pypdf import PdfReader
from openai import OpenAI
from chunking import split_text
from config import API_KEY

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

reader = PdfReader("data/xiezuo1.pdf")
text = ""
for i, page in enumerate(reader.pages, 1):
    text += f"\n\n=== 第{i}页 ===\n" + (page.extract_text() or "")

chunks = split_text(text)
bm25 = BM25Okapi([list(jieba.cut(c)) for c in chunks])

SYSTEM = """你是文档问答助手,严格遵守三条规则:
1. 只根据【参考资料】回答,不使用你自身的知识
2. 资料里没有的信息,直接回答"文档中没有提到"
3. 回答末尾用一行注明依据的页码"""


def pages_of(chunk):
    return [int(x) for x in re.findall(r"=== 第(\d+)页 ===", chunk)]

def expand_query(q):
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "把用户的问题改写成3个不同说法,覆盖同义词和不同角度。只输出3行,不要编号,不要解释。"},
            {"role": "user", "content": q},
        ],
        temperature=0.7,
    )
    lines = [l.strip() for l in r.choices[0].message.content.split("\n") if l.strip()]
    return [q] + lines[:3]


def retrieve(q, k=5):
    best = {}
    for v in expand_query(q):
        s = bm25.get_scores(list(jieba.cut(v)))
        for i, sc in enumerate(s):
            if sc > best.get(i, 0):
                best[i] = sc
    return sorted(best, key=lambda i: best[i], reverse=True)[:k]

questions = json.load(open("questions.json", encoding="utf-8"))

hit = 0
refuse_ok = 0
rows = []

for item in questions:
    q = item["q"]
    expect = item.get("page")

    top5 = retrieve(q)
    pages = [pages_of(chunks[i]) for i in top5]
    covered = sorted({p for ps in pages for p in ps})
    context = "\n\n".join(chunks[i] for i in top5)

    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"【参考资料】\n{context}\n\n【问题】{q}"},
        ],
    )
    answer = r.choices[0].message.content

    if expect is None:
        ok = "没有提到" in answer
        refuse_ok += ok
        mark = "拒答正确" if ok else "❌ 编造"
    else:
        ok = expect in covered
        hit += ok
        mark = "命中" if ok else "❌ 未命中"

    print(f"[{mark}] {q}  期望页={expect}  召回={covered}")
    rows.append([q, expect, pages, answer])

total_with_answer = sum(1 for i in questions if i.get("page"))
total_without = len(questions) - total_with_answer

print(f"\n召回率 Hit@5: {hit}/{total_with_answer} = {hit / total_with_answer:.0%}")
print(f"拒答正确率: {refuse_ok}/{total_without} = {refuse_ok / total_without:.0%}")

with open("results.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["问题", "期望页", "召回页", "回答"])
    w.writerows(rows)


    def expand_query(q):
        r = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": "把用户的问题改写成3个不同说法,覆盖同义词和不同角度。只输出3行,不要编号,不要解释。"},
                {"role": "user", "content": q},
            ],
            temperature=0.7,
        )
        lines = [l.strip() for l in r.choices[0].message.content.split("\n") if l.strip()]
        return [q] + lines[:3]


    def retrieve(q, k=5):
        best = {}
        for v in expand_query(q):
            s = bm25.get_scores(list(jieba.cut(v)))
            for i, sc in enumerate(s):
                if sc > best.get(i, 0):
                    best[i] = sc
        return sorted(best, key=lambda i: best[i], reverse=True)[:k]