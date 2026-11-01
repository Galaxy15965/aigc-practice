from __future__ import annotations

from types import SimpleNamespace
import unittest

from agent import DocumentAgent
from retrieval import DocumentIndex


def _message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeCompletions:
    def __init__(self, messages):
        self._messages = list(messages)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = self._messages.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, messages):
        self.chat = SimpleNamespace(completions=FakeCompletions(messages))


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = DocumentIndex.from_page_texts(
            [(1, "Python 的 GIL 是全局解释器锁，它限制同一时刻只有一个线程执行 Python 字节码。")],
            min_len=5,
        )

    def test_agent_calls_search_then_answers_with_trace(self) -> None:
        client = FakeClient(
            [
                _message(
                    tool_calls=[
                        _tool_call("call-1", "search_docs", '{"query":"GIL","top_k":1}')
                    ]
                ),
                _message(content="GIL 是全局解释器锁。\\n依据：第1页"),
            ]
        )
        agent = DocumentAgent(index=self.index, client=client)
        result = agent.ask("GIL 是什么")

        self.assertIn("全局解释器锁", result.answer)
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(result.accessed_pages, [1])
        self.assertEqual(result.cited_pages, [1])
        self.assertEqual(result.unverified_pages, [])
        self.assertEqual(result.trace[0]["tool"], "search_docs")
        self.assertTrue(result.trace[0]["ok"])
        self.assertEqual(client.chat.completions.requests[0]["tool_choice"], "auto")

    def test_agent_flags_unaccessed_citation(self) -> None:
        client = FakeClient([_message(content="这是答案。\\n依据：第9页")])
        agent = DocumentAgent(index=self.index, client=client)
        result = agent.ask("GIL 是什么")
        self.assertEqual(result.cited_pages, [9])
        self.assertEqual(result.unverified_pages, [9])

    def test_citation_range_extraction(self) -> None:
        from agent import extract_cited_pages

        self.assertEqual(extract_cited_pages("依据：第12—13页"), {12, 13})
        self.assertEqual(extract_cited_pages("依据：第12页、第13页"), {12, 13})
        self.assertEqual(extract_cited_pages("依据：第 12 页"), {12})
        self.assertEqual(extract_cited_pages("依据：第 12 — 13 页"), {12, 13})


if __name__ == "__main__":
    unittest.main()
