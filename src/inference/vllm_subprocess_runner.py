#!/usr/bin/env python3
"""在独立子进程中执行单次 OpenAI 兼容 chat.completions 请求。

支持三种入口：
1) 命令行无参数：stdin 读 JSON
2) 命令行 ``python vllm_subprocess_runner.py /path/to/payload.json``：从文件读（父进程用临时文件传参，避免管道写满死锁）
3) multiprocessing：``mp_entry`` + Queue（不推荐；``Process.start()`` 可能无限阻塞）
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional

import httpx
from openai import OpenAI


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content or "")


def _extract_usage(response) -> Optional[Dict[str, int]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
        completion_tokens = usage.get("completion_tokens", completion_tokens)
        total_tokens = usage.get("total_tokens", total_tokens)

    if total_tokens is None and (
        prompt_tokens is not None or completion_tokens is not None
    ):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _extract_message_parts(response) -> Dict[str, Any]:
    if not response.choices:
        return {"reasoning": "", "content": "", "usage": _extract_usage(response)}
    message = response.choices[0].message
    reasoning = (
        getattr(message, "reasoning", None)
        or getattr(message, "reasoning_content", None)
        or getattr(message, "thinking", None)
        or ""
    )
    if not isinstance(reasoning, str):
        reasoning = str(reasoning or "")
    content = _content_to_text(getattr(message, "content", None))
    return {
        "reasoning": reasoning,
        "content": content,
        "usage": _extract_usage(response),
    }


def run_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """执行一次请求，返回 ``{"ok": bool, "content"?: str, "reasoning"?: str, "err"?: str}``。"""
    api_base = payload["api_base"].rstrip("/")
    api_key = payload["api_key"]
    overall = float(payload["timeout"])
    connect = float(payload["connect_timeout"])
    read_t = float(payload["read_timeout"])
    write_t = float(payload["write_timeout"])
    request_kwargs = payload["request_kwargs"]

    timeout = httpx.Timeout(
        timeout=overall,
        connect=connect,
        read=read_t,
        write=write_t,
    )

    try:
        client = OpenAI(
            api_key=api_key or "EMPTY",
            base_url=api_base,
            timeout=timeout,
            max_retries=0,
        )
        response = client.chat.completions.create(**request_kwargs)
        parts = _extract_message_parts(response)
        return {
            "ok": True,
            "content": parts["content"],
            "reasoning": parts["reasoning"],
            "usage": parts.get("usage"),
        }
    except BaseException as exc:
        return {"ok": False, "err": str(exc), "err_type": type(exc).__name__}


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        json.dump({"ok": False, "err": f"stdin json: {exc}", "err_type": type(exc).__name__}, sys.stdout)
        sys.stdout.write("\n")
        sys.exit(1)

    body = run_payload(payload)
    json.dump(body, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    sys.exit(0 if body.get("ok") else 1)


def mp_entry(payload_json: str, result_queue) -> None:
    """multiprocessing spawn 子进程入口；结果写入 ``result_queue``。"""
    try:
        payload = json.loads(payload_json)
        body = run_payload(payload)
        result_queue.put(body)
    except BaseException as exc:
        try:
            result_queue.put({"ok": False, "err": str(exc), "err_type": type(exc).__name__})
        except Exception:
            pass


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        try:
            with open(sys.argv[1], encoding="utf-8") as fp:
                payload = json.load(fp)
        except Exception as exc:
            json.dump(
                {"ok": False, "err": f"payload file: {exc}", "err_type": type(exc).__name__},
                sys.stdout,
                ensure_ascii=False,
            )
            sys.stdout.write("\n")
            sys.exit(1)
        body = run_payload(payload)
        json.dump(body, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        sys.exit(0 if body.get("ok") else 1)
    main()
