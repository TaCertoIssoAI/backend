#!/usr/bin/env python3
"""
gemini_chat_cli.py — interactive CLI for one-off calls to GeminiChatModel.

usage:
    python scripts/playground/google/gemini_chat_cli.py

configuration (edit in code below):
    MODEL         — gemini model to use, e.g. "gemini-2.5-flash"
    THINKING_LEVEL — "low", "medium", "high" or None to disable
    TEMPERATURE   — 0.0–2.0 or None for model default
    SYSTEM_PROMPT — optional system instruction sent with every call

commands (during the loop):
    /system       — enter a new system prompt (multiline, END to finish)
    /config       — show current configuration
    empty line    — quit
"""

import os
import sys
import time
from pathlib import Path

# allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.playground.common import (
    Colors,
    print_header,
    print_section,
    print_success,
    print_error,
    print_warning,
    print_info,
    with_spinner,
    prompt_multiline,
)
from langchain_core.messages import HumanMessage, SystemMessage
from app.llms.gemini import GeminiChatModel


# ─── configuration (edit here) ────────────────────────────────────────────────

MODEL: str = "gemini-2.5-flash-lite"
THINKING_LEVEL: str | None = None   # "low", "medium", "high" or None
TEMPERATURE: float | None = 0.0     # 0.0–2.0 or None
SYSTEM_PROMPT: str | None = None    # set to a string or None


# ─── helpers ──────────────────────────────────────────────────────────────────

def _check_env() -> bool:
    if not os.environ.get("GOOGLE_API_KEY"):
        print_error("missing environment variable: GOOGLE_API_KEY")
        print_info("set it before running:\n  export GOOGLE_API_KEY=...")
        return False
    return True


def _print_config(system_prompt: str | None) -> None:
    print_section("active configuration")
    rows = {
        "model":          MODEL,
        "thinking level": THINKING_LEVEL or "(disabled)",
        "temperature":    TEMPERATURE if TEMPERATURE is not None else "(model default)",
        "system prompt":  f"{system_prompt[:60]}..." if system_prompt and len(system_prompt) > 60
                          else (system_prompt or "(none)"),
    }
    max_k = max(len(k) for k in rows)
    for k, v in rows.items():
        print(f"  {Colors.BOLD}{k.ljust(max_k)}{Colors.END}  {v}")


def _build_model() -> GeminiChatModel:
    return GeminiChatModel(
        model=MODEL,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        thinking_level=THINKING_LEVEL,
        temperature=TEMPERATURE,
    )


def _call_model(llm: GeminiChatModel, prompt: str, system_prompt: str | None) -> tuple[str, float]:
    """invoke the model and return (response_text, elapsed_seconds)."""
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    start = time.perf_counter()
    result = llm.invoke(messages)
    elapsed = time.perf_counter() - start

    return result.content, elapsed


def _print_response(text: str, elapsed: float) -> None:
    print_section("response")
    print(text)
    print()
    ms = elapsed * 1000
    print(f"  {Colors.BOLD}{Colors.YELLOW}⏱  {elapsed:.2f}s  ({ms:.0f}ms){Colors.END}")


# ─── main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    print_header("Gemini Chat — one-off call CLI")

    if not _check_env():
        sys.exit(1)

    system_prompt = SYSTEM_PROMPT
    _print_config(system_prompt)
    print_info("\ntype a prompt and press Enter  ·  /system to change system prompt  ·  /config to view settings  ·  empty line to quit\n")

    llm = _build_model()

    while True:
        try:
            raw = input(f"{Colors.BOLD}prompt>{Colors.END} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            break

        if raw == "/config":
            _print_config(system_prompt)
            print()
            continue

        if raw == "/system":
            system_prompt = prompt_multiline("enter new system prompt")
            print_success("system prompt updated")
            print()
            continue

        try:
            response_text, elapsed = with_spinner(
                lambda p=raw, sp=system_prompt: _call_model(llm, p, sp),
                "calling gemini..."
            )
            _print_response(response_text, elapsed)
        except Exception as e:
            print_error(f"error: {e}")
            print()

    print_info("bye")


if __name__ == "__main__":
    main()
