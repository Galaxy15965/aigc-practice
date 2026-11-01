from __future__ import annotations

import unittest

from retrieval import DocumentIndex
from tools import TOOLS, TOOL_NAMES, execute_tool


class ToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = DocumentIndex.from_page_texts(
            [
                (1, "Python 的 GIL 是全局解释器锁，它限制同一时刻只有一个线程执行 Python 字节码。"),
                (2, "调查报告的写作要经历准备、起草和修改三个阶段。"),
                (3, "摄影构图中，三分法通过把主体放在画面三分线附近来建立视觉平衡。"),
            ],
            min_len=5,
        )

    def test_tool_schemas(self) -> None:
        self.assertEqual(TOOL_NAMES, {"search_docs", "read_page"})
        self.assertEqual(len(TOOLS), 2)
        for tool in TOOLS:
            self.assertEqual(tool["type"], "function")
            self.assertIn("name", tool["function"])
            self.assertIn("parameters", tool["function"])

    def test_search_docs_tool(self) -> None:
        result = execute_tool("search_docs", '{"query":"GIL","top_k":1}', self.index)
        self.assertTrue(result["ok"])
        self.assertEqual(result["pages"], [1])
        self.assertEqual(result["results"][0]["chunk_id"], "p1-c0")

    def test_read_page_tool(self) -> None:
        result = execute_tool("read_page", {"page_no": 2}, self.index)
        self.assertTrue(result["ok"])
        self.assertEqual(result["page"], 2)
        self.assertIn("调查报告", result["text"])

    def test_bad_arguments_return_error_instead_of_crashing(self) -> None:
        result = execute_tool("search_docs", '{"query":"GIL","top_k":99}', self.index)
        self.assertFalse(result["ok"])
        self.assertIn("top_k", result["error"])

        result = execute_tool("search_docs", "not json", self.index)
        self.assertFalse(result["ok"])
        self.assertIn("JSON", result["error"])

    def test_unknown_tool(self) -> None:
        result = execute_tool("delete_files", {}, self.index)
        self.assertFalse(result["ok"])
        self.assertIn("未知工具", result["error"])


if __name__ == "__main__":
    unittest.main()
