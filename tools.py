from __future__ import annotations

import json
from typing import Any

from retrieval import DocumentIndex


SEARCH_DOCS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "在课程 PDF 中做关键词检索，返回相关原文片段、页码和 BM25 分数。回答前应先调用此工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要检索的问题或关键词；可以改写成更具体的检索语句后再次调用。",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回片段数量，默认 5，范围 1—10。",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                },
                "max_chars": {
                    "type": "integer",
                    "description": "每个片段最多返回多少字符，默认 1200，范围 200—4000。",
                    "minimum": 200,
                    "maximum": 4000,
                    "default": 1200,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


READ_PAGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_page",
        "description": "读取课程的指定物理页原文，用于核对页码、上下文或检索片段缺失的信息。",
        "parameters": {
            "type": "object",
            "properties": {
                "page_no": {
                    "type": "integer",
                    "description": "PDF 物理页码，从 1 开始。",
                    "minimum": 1,
                },
                "max_chars": {
                    "type": "integer",
                    "description": "最多返回多少字符，默认 6000，范围 200—12000。",
                    "minimum": 200,
                    "maximum": 12000,
                    "default": 6000,
                },
            },
            "required": ["page_no"],
            "additionalProperties": False,
        },
    },
}


TOOLS = [SEARCH_DOCS_SCHEMA, READ_PAGE_SCHEMA]
TOOL_NAMES = frozenset(tool["function"]["name"] for tool in TOOLS)


def execute_tool(name: str, arguments: str | dict[str, Any] | None, index: DocumentIndex) -> dict:
    """执行 Agent 工具并统一返回可 JSON 序列化的结果。"""
    try:
        args = _parse_arguments(arguments)
        if name == "search_docs":
            query = str(args.get("query", "")).strip()
            top_k = _int_in_range(args.get("top_k", 5), "top_k", 1, 10)
            max_chars = _int_in_range(args.get("max_chars", 1200), "max_chars", 200, 4000)
            return index.search(query=query, top_k=top_k, max_chars=max_chars)

        if name == "read_page":
            page_no = _int_in_range(args.get("page_no"), "page_no", 1, index.page_count)
            max_chars = _int_in_range(args.get("max_chars", 6000), "max_chars", 200, 12000)
            return index.read_page(page_no=page_no, max_chars=max_chars)

        return _error(f"未知工具：{name}")
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return _error(str(exc))


def _parse_arguments(arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("工具参数不是合法 JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        return parsed
    raise ValueError("工具参数格式不受支持")


def _int_in_range(value: Any, field: str, minimum: int, maximum: int) -> int:
    if value is None:
        raise ValueError(f"{field} 不能为空")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是整数") from exc
    if number < minimum or number > maximum:
        raise ValueError(f"{field} 必须在 {minimum}—{maximum} 之间")
    return number


def _error(message: str) -> dict:
    return {"ok": False, "error": message}
