from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from agent import DocumentAgent
from retrieval import DocumentIndex


ROOT = Path(__file__).resolve().parent
REFUSAL_MARKERS = (
    "文档中没有提到",
    "资料中没有",
    "没有找到任何关于",
    "没有找到相关",
    "无法依据课程资料回答",
    "文档中未",
)



class DeepSeekHTTPError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        super().__init__(f"DeepSeek HTTP {status_code}: {detail[:800]}")


class _ToolFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = _ToolFunction(name, arguments)


class _AssistantMessage:
    def __init__(self, content: str | None, tool_calls: list[_ToolCall]) -> None:
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ],
        }
        if exclude_none:
            if payload["content"] is None:
                payload.pop("content")
            if not payload["tool_calls"]:
                payload.pop("tool_calls")
        return payload


class _HTTPCompletions:
    def __init__(self) -> None:
        from config import API_KEY

        self.api_key = API_KEY
        self.url = "https://api.deepseek.com/chat/completions"

    def create(self, **kwargs: Any) -> Any:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(kwargs, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise DeepSeekHTTPError(exc.code, detail) from exc

        choice = payload["choices"][0]
        raw_message = choice["message"]
        tool_calls = [
            _ToolCall(
                str(item.get("id", "")),
                str(item.get("function", {}).get("name", "")),
                str(item.get("function", {}).get("arguments", "{}")),
            )
            for item in (raw_message.get("tool_calls") or [])
        ]
        message = _AssistantMessage(raw_message.get("content"), tool_calls)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=payload.get("usage"),
        )


class DeepSeekHTTPClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_HTTPCompletions())


class _TrackingCompletions:
    def __init__(self, owner: "UsageTrackingClient") -> None:
        self.owner = owner

    def create(self, **kwargs: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.owner.max_retries + 1):
            self.owner.api_requests += 1
            try:
                response = self.owner.client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_error = exc
                if attempt >= self.owner.max_retries or _is_auth_error(exc):
                    raise
                self.owner.retries += 1
                time.sleep(2 ** (attempt - 1))
                continue

            usage = _usage_dict(getattr(response, "usage", None))
            if usage:
                self.owner.usages.append(usage)
            return response

        if last_error is not None:
            raise last_error
        raise RuntimeError("API 请求失败且没有返回异常")


class UsageTrackingClient:
    def __init__(self, client: Any, max_retries: int = 3) -> None:
        self.client = client
        self.max_retries = max(1, max_retries)
        self.api_requests = 0
        self.retries = 0
        self.usages: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=_TrackingCompletions(self))


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    if hasattr(usage, "model_dump"):
        return dict(usage.model_dump(exclude_none=True))
    return {
        key: value
        for key, value in vars(usage).items()
        if isinstance(value, (int, float))
    }


def _is_auth_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    message = str(exc).lower()
    return status in {401, 403} or "authentication" in message or "invalid api key" in message


def _spread_indices(total: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= total:
        return list(range(total))
    if count == 1:
        return [total // 2]
    return sorted({round(index * (total - 1) / (count - 1)) for index in range(count)})


def select_questions(
    questions: Sequence[dict[str, Any]],
    limit: int,
    mode: str = "stratified",
) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(questions):
        return list(questions)
    if mode == "head":
        return list(questions[:limit])

    answerable = [item for item in questions if item.get("page") is not None]
    unanswerable = [item for item in questions if item.get("page") is None]
    if not answerable or not unanswerable:
        return list(questions[:limit])

    unanswerable_count = min(len(unanswerable), max(1, round(limit * 0.2)))
    answerable_count = limit - unanswerable_count
    if answerable_count > len(answerable):
        answerable_count = len(answerable)
        unanswerable_count = limit - answerable_count

    answerable_picks = [answerable[index] for index in _spread_indices(len(answerable), answerable_count)]
    unanswerable_picks = [unanswerable[index] for index in _spread_indices(len(unanswerable), unanswerable_count)]

    selected: list[dict[str, Any]] = []
    answerable_pos = 0
    unanswerable_pos = 0
    while answerable_pos < len(answerable_picks) or unanswerable_pos < len(unanswerable_picks):
        if answerable_pos < len(answerable_picks):
            selected.append(answerable_picks[answerable_pos])
            answerable_pos += 1
        if unanswerable_pos < len(unanswerable_picks):
            selected.append(unanswerable_picks[unanswerable_pos])
            unanswerable_pos += 1
    return selected


def is_refusal(answer: str) -> bool:
    """只检查回答开头，避免把正常回答末尾的“原文未说明”误判为整题拒答。"""
    head = " ".join(answer.strip().split())[:320]
    return any(marker in head for marker in REFUSAL_MARKERS)


def evaluate_one(agent: DocumentAgent, item: dict[str, Any], position: int) -> dict[str, Any]:
    question = str(item["q"]).strip()
    expected_page = item.get("page")
    started = time.perf_counter()
    try:
        result = agent.ask(question)
    except Exception as exc:
        return {
            "position": position,
            "question": question,
            "expected_page": expected_page,
            "error": str(exc),
            "success": False,
            "elapsed_s": round(time.perf_counter() - started, 2),
            "trace": [],
        }

    tools = [trace.get("tool") for trace in result.trace]
    refused = is_refusal(result.answer)
    retrieval_hit = None
    citation_hit = None
    citation_valid = None
    answered_well = None

    if expected_page is None:
        success = refused
    else:
        retrieval_hit = int(expected_page) in result.accessed_pages
        citation_hit = int(expected_page) in result.cited_pages
        citation_valid = not bool(result.unverified_pages)
        answered_well = citation_hit and citation_valid and len(result.answer.strip()) >= 40
        success = answered_well

    return {
        "position": position,
        "question": question,
        "expected_page": expected_page,
        "answer": result.answer,
        "tool_calls": result.tool_calls,
        "steps": result.steps,
        "used_search_docs": "search_docs" in tools,
        "used_read_page": "read_page" in tools,
        "accessed_pages": result.accessed_pages,
        "cited_pages": result.cited_pages,
        "unverified_pages": result.unverified_pages,
        "refused": refused,
        "retrieval_hit": retrieval_hit,
        "citation_hit": citation_hit,
        "citation_valid": citation_valid,
        "answered_well": answered_well,
        "success": bool(success),
        "elapsed_s": round(time.perf_counter() - started, 2),
        "trace": result.trace,
    }


def build_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answerable = [record for record in records if record.get("expected_page") is not None]
    unanswerable = [record for record in records if record.get("expected_page") is None]
    successful = [record for record in records if record.get("success")]

    tool_calls_total = sum(int(record.get("tool_calls", 0) or 0) for record in records)
    tool_calls_ok = sum(
        1
        for record in records
        for trace in record.get("trace", [])
        if trace.get("ok")
    )
    answerable_count = len(answerable)
    unanswerable_count = len(unanswerable)

    def count_true(key: str, source: Sequence[dict[str, Any]]) -> int:
        return sum(1 for record in source if record.get(key) is True)

    return {
        "questions": len(records),
        "answerable": answerable_count,
        "unanswerable": unanswerable_count,
        "successes": len(successful),
        "success_rate": round(len(successful) / len(records), 4) if records else 0.0,
        "answered": answerable_count - count_true("refused", answerable),
        "errors": sum(1 for record in records if record.get("error")),
        "retrieval_hits": count_true("retrieval_hit", answerable),
        "retrieval_hit_rate": round(count_true("retrieval_hit", answerable) / answerable_count, 4) if answerable_count else None,
        "citation_hits": count_true("citation_hit", answerable),
        "citation_hit_rate": round(count_true("citation_hit", answerable) / answerable_count, 4) if answerable_count else None,
        "citation_valid": count_true("citation_valid", answerable),
        "citation_valid_rate": round(count_true("citation_valid", answerable) / answerable_count, 4) if answerable_count else None,
        "refusal_correct": count_true("refused", unanswerable),
        "refusal_accuracy": round(count_true("refused", unanswerable) / unanswerable_count, 4) if unanswerable_count else None,
        "tool_calls_total": tool_calls_total,
        "tool_calls_ok": tool_calls_ok,
        "tool_call_success_rate": round(tool_calls_ok / tool_calls_total, 4) if tool_calls_total else None,
        "avg_tool_calls": round(tool_calls_total / len(records), 2) if records else 0.0,
        "avg_elapsed_s": round(sum(float(record.get("elapsed_s", 0) or 0) for record in records) / len(records), 2) if records else 0.0,
    }


def sum_usage(usages: Sequence[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for usage in usages:
        for key, value in usage.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def write_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    fields = [
        "position",
        "question",
        "expected_page",
        "success",
        "refused",
        "retrieval_hit",
        "citation_hit",
        "citation_valid",
        "tool_calls",
        "steps",
        "accessed_pages",
        "cited_pages",
        "unverified_pages",
        "elapsed_s",
        "answer",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key in ("accessed_pages", "cited_pages", "unverified_pages"):
                row[key] = ",".join(str(item) for item in (record.get(key) or []))
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="课程资料 Agent 真实 API 评测")
    parser.add_argument("--limit", type=int, default=10, help="题目数量；0 表示全部")
    parser.add_argument("--sample-mode", choices=("stratified", "head"), default="stratified")
    parser.add_argument("--questions", type=Path, default=ROOT / "questions.json")
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    all_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = select_questions(all_questions, args.limit, args.sample_mode)
    index = DocumentIndex.from_pdf(pdf_path=args.pdf, data_dir=ROOT / "data")
    client = UsageTrackingClient(DeepSeekHTTPClient(), max_retries=args.max_retries)
    agent = DocumentAgent(index=index, client=client, max_steps=args.max_steps)

    print(f"索引：{index.chunk_count} 段，{index.page_count} 页")
    print(f"评测：{len(selected)} 题，模式={args.sample_mode}")

    records: list[dict[str, Any]] = []
    for position, item in enumerate(selected, 1):
        record = evaluate_one(agent, item, position)
        records.append(record)
        status = "PASS" if record.get("success") else "FAIL"
        print(
            f"[{position:02d}/{len(selected):02d}] {status} "
            f"page={record.get('expected_page')} "
            f"tools={record.get('tool_calls', 0)} "
            f"time={record.get('elapsed_s', 0)}s  {record['question']}"
        )

    summary = build_summary(records)
    usage_totals = sum_usage(client.usages)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "deepseek-chat",
        "sample": {
            "limit": args.limit,
            "mode": args.sample_mode,
            "requested_total": len(all_questions),
        },
        "summary": summary,
        "api_requests": client.api_requests,
        "retries": client.retries,
        "usage_totals": usage_totals,
        "results": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "all" if args.limit <= 0 or args.limit >= len(all_questions) else str(len(selected))
    json_path = args.output_dir / f"agent_eval_{suffix}.json"
    csv_path = args.output_dir / f"agent_eval_{suffix}.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, records)

    console_summary = {
        "summary": summary,
        "api_requests": client.api_requests,
        "retries": client.retries,
        "usage_totals": usage_totals,
    }
    print("\nSUMMARY")
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))
    print(f"\nJSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
