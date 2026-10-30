from openai import OpenAI
from config import API_KEY

clent =OpenAI(api_key=API_KEY,base_url="https://api.deepseek.com")


def split_text(text,size=800):
    paragraphs = [p.strip() for p in text.spilt("\n") if p.strip()]
    chunks, current = [],""
    for p in paragraphs:
        if len(current) + len(p) > size:
            chunks.append(current)
            current = p
        else:
            current += "\n" + p
    if current:
        chunks.append(current)
    return chunks


