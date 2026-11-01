from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import jieba
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from chunking import split_text


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    page: int
    text: str
    tokens: tuple[str, ...]

    def as_result(self, score: float, max_chars: int = 1200) -> dict:
        text, truncated = _truncate(self.text, max_chars)
        return {
            "chunk_id": self.chunk_id,
            "page": self.page,
            "score": round(float(score), 4),
            "text": text,
            "truncated": truncated,
        }


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars].rstrip() + "…", True


def extract_page_texts(pdf_path: str | Path) -> list[tuple[int, str]]:
    """按页提取文本，保留页码边界，避免跨页块导致引用偏移。"""
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for page_no, page in enumerate(reader.pages, 1):
        pages.append((page_no, (page.extract_text() or "").strip()))
    return pages


def build_page_chunks(
    pages: Iterable[tuple[int, str]],
    chunk_size: int = 300,
    min_len: int = 50,
) -> list[Chunk]:
    """逐页分块；每个块只属于一个物理页。"""
    chunks: list[Chunk] = []
    for page_no, page_text in pages:
        if not page_text:
            continue
        for index, text in enumerate(split_text(page_text, size=chunk_size, min_len=min_len)):
            tokens = tuple(token for token in jieba.lcut(text) if token.strip())
            if not tokens:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"p{page_no}-c{index}",
                    page=page_no,
                    text=text,
                    tokens=tokens,
                )
            )
    return chunks


class DocumentIndex:
    """页感知的 BM25 索引，可被 CLI、Agent 工具和 Streamlit 复用。"""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        page_texts: dict[int, str],
        source: str | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("没有可用文本块，请检查 PDF 是否为扫描版或文字层是否为空")
        self.chunks = list(chunks)
        self.page_texts = dict(page_texts)
        self.source = source
        self._bm25 = BM25Okapi([list(chunk.tokens) for chunk in self.chunks])

    @classmethod
    def from_pdf(
        cls,
        pdf_path: str | Path | None = None,
        data_dir: str | Path = "data",
        chunk_size: int = 300,
        min_len: int = 50,
    ) -> "DocumentIndex":
        resolved = _resolve_pdf(pdf_path, data_dir)
        pages = extract_page_texts(resolved)
        chunks = build_page_chunks(pages, chunk_size=chunk_size, min_len=min_len)
        return cls(chunks, dict(pages), source=str(resolved))

    @classmethod
    def from_page_texts(
        cls,
        pages: Iterable[tuple[int, str]],
        chunk_size: int = 300,
        min_len: int = 10,
    ) -> "DocumentIndex":
        page_list = [(page_no, text.strip()) for page_no, text in pages]
        chunks = build_page_chunks(page_list, chunk_size=chunk_size, min_len=min_len)
        return cls(chunks, dict(page_list), source="memory")

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def page_count(self) -> int:
        return len(self.page_texts)

    def search(self, query: str, top_k: int = 5, max_chars: int = 1200) -> dict:
        query = query.strip()
        if not query:
            raise ValueError("query 不能为空")
        query_tokens = [token for token in jieba.lcut(query) if token.strip()]
        if not query_tokens:
            raise ValueError("query 没有可用词项")

        scores = self._bm25.get_scores(query_tokens)
        ranked = [index for index, score in enumerate(scores) if float(score) != 0]
        ranked.sort(key=lambda index: float(scores[index]), reverse=True)
        selected = ranked[: max(1, min(int(top_k), 10))]
        results = [self.chunks[index].as_result(float(scores[index]), max_chars) for index in selected]
        pages = sorted({item["page"] for item in results})
        best_score = round(float(scores[selected[0]]), 4) if selected else 0.0
        return {
            "ok": True,
            "query": query,
            "count": len(results),
            "pages": pages,
            "best_score": best_score,
            "results": results,
        }

    def read_page(self, page_no: int, max_chars: int = 6000) -> dict:
        try:
            page_no = int(page_no)
        except (TypeError, ValueError) as exc:
            raise ValueError("page_no 必须是整数") from exc
        if page_no < 1 or page_no > self.page_count:
            raise ValueError(f"page_no 超出范围：1—{self.page_count}")

        original = self.page_texts.get(page_no, "")
        text, truncated = _truncate(original, max_chars)
        return {
            "ok": True,
            "page": page_no,
            "text": text,
            "truncated": truncated,
            "empty": not bool(original.strip()),
        }


def _resolve_pdf(pdf_path: str | Path | None, data_dir: str | Path) -> Path:
    if pdf_path is not None:
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"找不到 PDF：{path}")
        return path

    data_path = Path(data_dir)
    preferred = data_path / "xiezuo1.pdf"
    if preferred.is_file():
        return preferred

    candidates = sorted(data_path.glob("*.pdf"))
    if not candidates:
        raise FileNotFoundError(f"{data_path} 下没有 PDF，请先放入课程资料")
    return candidates[0]
