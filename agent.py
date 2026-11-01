from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import re
import time
from typing import Any

from retrieval import DocumentIndex
from tools import TOOLS, execute_tool


SYSTEM_PROMPT = """你是课程资料 Agent。必须遵守以下规则：
1. 回答课程问题前，先调用 search_docs 检索原文；需要核对上下文时调用 read_page。
2. 只能依据工具返回的资料回答，不使用你自己的知识补全。
3. 工具结果不够时，明确回答“文档中没有提到”，不要猜测。
4. 回答末尾单独一行写“依据：第X页”；引用多个页面时全部列出。
5. 首次检索不相关时改写一次检索词；不要无意义重复同一工具调用。
6. 工具报错时先修正参数，不要假装工具成功。"""


@dataclass
class AgentResult:
    answer: str
    steps: int
    tool_calls: int
    trace: list[dict] = field(default_factory=list)
    accessed_pages: list[int] = field(default_factory=list)
    cited_pages: list[int] = field(default_factory=list)
    unverified_pages: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class DocumentAgent:
    """一个最小可用的 Function Calling 循环：规划 → 工具 → 观察 → 回答。"""

    def __init__(
        self,
        index: DocumentIndex,
        client: Any,
        model: str = "deepseek-chat",
        max_steps: int = 6,
    ) -> None:
        self.index = index
        self.client = client
        self.model = model
        self.max_steps = max_steps

    def ask(self, question: str) -> AgentResult:
        question = question.strip()
        if not question:
            raise ValueError("question 不能为空")

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        trace: list[dict] = []
        accessed_pages: set[int] = set()
        seen_calls: set[str] = set()
        tool_calls_count = 0
        steps = 0

        for step in range(1, self.max_steps + 1):
            steps = step
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
            )
            message = response.choices[0].message
            tool_calls = list(getattr(message, "tool_calls", None) or [])

            if not tool_calls:
                answer = _answer_text(message)
                return self._result(
                    answer=answer,
                    steps=steps,
                    tool_calls=tool_calls_count,
                    trace=trace,
                    accessed_pages=accessed_pages,
                )

            messages.append(_assistant_message(message, tool_calls))
            for call in tool_calls:
                tool_calls_count += 1
                name, arguments = _call_parts(call)
                signature = f"{name}:{arguments}"
                started = time.perf_counter()
                if signature in seen_calls:
                    result = {"ok": False, "error": "重复工具调用；请改写 query 或直接依据已有结果回答"}
                else:
                    seen_calls.add(signature)
                    result = execute_tool(name, arguments, self.index)
                elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

                pages = _result_pages(result)
                if result.get("ok"):
                    accessed_pages.update(pages)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(getattr(call, "id", f"call-{tool_calls_count}")),
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                trace.append(
                    {
                        "step": step,
                        "tool": name,
                        "arguments": arguments,
                        "ok": bool(result.get("ok")),
                        "pages": pages,
                        "count": result.get("count"),
                        "best_score": result.get("best_score"),
                        "error": result.get("error"),
                        "duration_ms": elapsed_ms,
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": "已达到工具调用上限。请只依据已有工具结果作答；证据不足就说明文档中没有提到，不要再请求工具。",
            }
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return self._result(
            answer=_answer_text(response.choices[0].message),
            steps=self.max_steps + 1,
            tool_calls=tool_calls_count,
            trace=trace,
            accessed_pages=accessed_pages,
        )

    def _result(
        self,
        answer: str,
        steps: int,
        tool_calls: int,
        trace: list[dict],
        accessed_pages: set[int],
    ) -> AgentResult:
        cited_pages = extract_cited_pages(answer)
        return AgentResult(
            answer=answer,
            steps=steps,
            tool_calls=tool_calls,
            trace=trace,
            accessed_pages=sorted(accessed_pages),
            cited_pages=sorted(cited_pages),
            unverified_pages=sorted(cited_pages - accessed_pages),
        )


def extract_cited_pages(answer: str) -> set[int]:
    """提取“第12页”“第12—13页”中的页码，用于检查引用是否来自工具结果。"""
    pages: set[int] = set()
    for match in re.finditer(r"第\s*(\d+)\s*(?:[—\-~至]\s*(\d+)\s*)?页", answer):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if end < start:
            start, end = end, start
        if end - start <= 20:
            pages.update(range(start, end + 1))
        else:
            pages.add(start)
    return pages


def _answer_text(message: Any) -> str:
    text = (getattr(message, "content", None) or "").strip()
    return text or "文档中没有提到能够支持该问题的信息。"


def _assistant_message(message: Any, tool_calls: list[Any]) -> dict:
    if hasattr(message, "model_dump"):
        payload = message.model_dump(exclude_none=True)
        payload["role"] = "assistant"
        payload.pop("function_call", None)
        if payload.get("tool_calls"):
            payload["tool_calls"] = [_call_to_dict(call) for call in payload["tool_calls"]]
        return payload

    payload: dict[str, Any] = {
        "role": "assistant",
        "content": getattr(message, "content", None),
        "tool_calls": [_call_to_dict(call) for call in tool_calls],
    }
    return payload


def _call_to_dict(call: Any) -> dict:
    if isinstance(call, dict):
        return call
    function = getattr(call, "function", None)
    return {
        "id": str(getattr(call, "id", "")),
        "type": "function",
        "function": {
            "name": getattr(function, "name", ""),
            "arguments": getattr(function, "arguments", "{}"),
        },
    }


def _call_parts(call: Any) -> tuple[str, Any]:
    if isinstance(call, dict):
        function = call.get("function", {})
        return function.get("name", ""), function.get("arguments", "{}")
    function = getattr(call, "function", None)
    return (
        getattr(function, "name", ""),
        getattr(function, "arguments", "{}"),
    )


def _result_pages(result: dict) -> list[int]:
    if not result.get("ok"):
        return []
    pages = result.get("pages")
    if isinstance(pages, list):
        return [int(page) for page in pages]
    page = result.get("page")
    return [int(page)] if page is not None else []


def build_default_client() -> Any:
    from openai import OpenAI
    from config import API_KEY

    return OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")


def main() -> None:
    parser = argparse.ArgumentParser(description="课程资料 Agent：检索工具 + 单 Agent 循环")
    parser.add_argument("question", nargs="?", help="要提问的问题；省略后进入交互模式")
    parser.add_argument("--pdf", help="PDF 路径；默认使用 data/xiezuo1.pdf")
    parser.add_argument("--max-steps", type=int, default=6, help="最多工具调用轮数")
    parser.add_argument("--verbose", action="store_true", help="打印工具轨迹")
    args = parser.parse_args()

    index = DocumentIndex.from_pdf(pdf_path=args.pdf, data_dir="data")
    agent = DocumentAgent(index=index, client=build_default_client(), max_steps=args.max_steps)
    print(f"已索引 {index.chunk_count} 段，共 {index.page_count} 页")

    question = args.question or input("问题：").strip()
    result = agent.ask(question)
    print("\n" + result.answer)
    print(f"\n[工具调用 {result.tool_calls} 次，执行 {result.steps} 轮]")

    if result.unverified_pages:
        print(f"[警告] 以下页码没有被工具实际访问：{result.unverified_pages}")

    if args.verbose:
        print("\n工具轨迹：")
        for item in result.trace:
            print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
