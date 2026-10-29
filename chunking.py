def split_text(text, size=300, min_len=50):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) > size and current:
            chunks.append(current)
            current = p
        else:
            current = (current + "\n" + p).strip() if current else p
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) >= min_len]