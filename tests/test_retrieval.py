from __future__ import annotations

import unittest

from retrieval import DocumentIndex


PAGES = [
    (1, "Python 的 GIL 是全局解释器锁，它限制同一时刻只有一个线程执行 Python 字节码。"),
    (2, "调查报告的写作一般要经历准备、起草和修改三个阶段。写作前应明确目的和对象。"),
    (3, "摄影构图中，三分法通过把主体放在画面三分线附近来建立视觉平衡。"),
]


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = DocumentIndex.from_page_texts(PAGES, min_len=5)

    def test_chunks_are_page_aware(self) -> None:
        self.assertTrue(self.index.chunks)
        for chunk in self.index.chunks:
            self.assertIn(chunk.page, self.index.page_texts)
            self.assertTrue(chunk.chunk_id.startswith(f"p{chunk.page}-c"))

    def test_search_returns_relevant_page_and_json_fields(self) -> None:
        result = self.index.search("GIL 是什么", top_k=1)
        self.assertTrue(result["ok"])
        self.assertEqual(result["pages"], [1])
        self.assertEqual(result["count"], 1)
        item = result["results"][0]
        self.assertEqual(item["page"], 1)
        self.assertIn("全局解释器锁", item["text"])
        self.assertGreater(item["score"], 0)

    def test_no_overlap_returns_empty_results(self) -> None:
        result = self.index.search("量子纠缠", top_k=3)
        self.assertTrue(result["ok"])
        self.assertEqual(result["results"], [])
        self.assertEqual(result["pages"], [])
        self.assertEqual(result["best_score"], 0.0)

    def test_read_page(self) -> None:
        result = self.index.read_page(2)
        self.assertTrue(result["ok"])
        self.assertFalse(result["empty"])
        self.assertIn("调查报告", result["text"])

    def test_read_page_rejects_invalid_page(self) -> None:
        with self.assertRaises(ValueError):
            self.index.read_page(99)


if __name__ == "__main__":
    unittest.main()
