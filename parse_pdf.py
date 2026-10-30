from pypdf import PdfReader
from chunking import split_text

reader = PdfReader("data/xiezuo1.pdf")

records = []          # 每个元素是 (页码, 这一段文字)
for page_no, page in enumerate(reader.pages, 1):
    t = (page.extract_text() or "").strip()
    if not t:
        continue
    for c in split_text(t, size=800, min_len=30):
        records.append((page_no, c))

print(f"共 {len(records)} 段")
page_no, first = records[0]
print(f"--- 第一段(第{page_no}页)---")
print(first[:200])