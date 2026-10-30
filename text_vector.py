import chromadb
from pypdf import PdfReader
from chunking import split_text

reader = PdfReader("data/xiezuo1.pdf")
text = ""
for i, page in enumerate(reader.pages, 1):
    text += f"\n\n=== 第{i}页 ===\n" + (page.extract_text() or "")

chunks = split_text(text)
db = chromadb.PersistentClient(path="db")
col = db.get_or_create_collection("docs")
col.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])

tests = [
    "实践和认识的先后关系",
    "调查报告的结尾有什么写法",
    "Python 的 GIL 是什么",
    "今天中午吃什么",
]

for q in tests:
    r = col.query(query_texts=[q], n_results=3)
    dist = [round(x, 3) for x in r["distances"][0]]
    first = r["documents"][0][0][:45].replace("\n", " ")
    print(f"{q}\n  距离={dist}\n  首条={first}\n")