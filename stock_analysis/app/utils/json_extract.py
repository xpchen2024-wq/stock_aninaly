# ============================================================================
# AI Stock Analysis Platform - Shared JSON Extraction Utility
# ============================================================================
"""Robust JSON extraction from LLM responses.

Single source of truth shared by app.agents, app.engines and any future
LLM-calling module. Previously this logic was duplicated (agents._extract_json
and inline parsing in engines.KOLEngine.extract_opinion).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def extract_json(text: str) -> str:
    """Extract the JSON object string from an LLM response."""
    text = text.strip()
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        return text[start:end].strip() if end > start else text
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        return text[start:end].strip() if end > start else text
    if text.startswith("{") and text.endswith("}"):
        return text
    # Try to find JSON block
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


def parse_llm_json(text: str) -> Optional[Any]:
    """Parse an LLM response into Python data. Returns None if not valid JSON.

    This makes 'fail loud' possible at the call site: callers can distinguish
    'model returned garbage' (None) from 'model returned empty defaults'.
    """
    raw = extract_json(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"parse_llm_json failed: {e}; raw[:200]={raw[:200]!r}")
        return None


def build_retry_prompt(original_user_prompt: str, bad_output: str) -> str:
    """Build a follow-up user prompt asking the model to fix invalid JSON."""
    return (
        f"{original_user_prompt}\n\n"
        f"注意：你上一次的输出不是合法 JSON，无法解析：\n"
        f"{bad_output[:300]}\n"
        f"请严格只输出一个合法的 JSON 对象，不要包含任何其他文字、注释或 markdown 代码块以外的内容。"
    )
