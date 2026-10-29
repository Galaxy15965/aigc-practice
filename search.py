import jieba
from pathlib import Path
from rank_bm25 import BM25Okapi
from pypdf import PdfReader
from chunking import split_text

pdf_path = next(Path("data").glob("*.pdf"))
print("使用文件:", pdf_path.name)
reader = PdfReader(pdf_path)

records = []
for page_no, page in enumerate(reader.pages, 1):
    t = (page.extract_text() or "").strip()
    if not t:
        continue
    for c in split_text(t, size=800, min_len=30):
        records.append((page_no, c))

pages = [r[0] for r in records]
texts = [r[1] for r in records]
print(f"已索引 {len(texts)} 段")

bm25 = BM25Okapi([list(jieba.cut(t)) for t in texts])

while True:
    q = input("\n提问(输入 exit 退出):")
    if q.strip() == "exit":
        break
    scores = bm25.get_scores(list(jieba.cut(q)))
    top3 = sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)[:3]
    for rank, idx in enumerate(top3, 1):
        print(f"\n--- 第{rank}名 | 第{pages[idx]}页 | 得分 {scores[idx]:.2f} ---")
        print(texts[idx][:150])