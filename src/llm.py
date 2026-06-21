from __future__ import annotations

import os
from typing import Any, Callable, Protocol


DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
DEFAULT_QWEN_MODEL = "qwen-plus"
DEFAULT_QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_LLM_MAX_RETRIES = 1


class ChatModel(Protocol):
    def invoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        ...


class FallbackChatModel:
    def __init__(self, primary: ChatModel, fallback: ChatModel) -> None:
        self.primary = primary
        self.fallback = fallback

    def invoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return self.primary.invoke(messages, *args, **kwargs)
        except Exception as primary_error:
            try:
                return self.fallback.invoke(messages, *args, **kwargs)
            except Exception as fallback_error:
                raise RuntimeError(
                    "Gemini primary and Qwen fallback LLM calls both failed. "
                    f"Gemini error: {primary_error!r}. "
                    f"Qwen error: {fallback_error!r}."
                ) from fallback_error


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(content_to_text(item["content"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def get_llm(
    temperature: float = 0.2,
    google_factory: Callable[..., ChatModel] | None = None,
    qwen_factory: Callable[..., ChatModel] | None = None,
) -> ChatModel:
    google_factory = google_factory or _google_factory
    qwen_factory = qwen_factory or _qwen_factory
    timeout = _float_env(
        "LLM_REQUEST_TIMEOUT_SECONDS",
        DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS,
    )
    max_retries = _int_env("LLM_MAX_RETRIES", DEFAULT_LLM_MAX_RETRIES)

    try:
        primary = google_factory(
            model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            api_key=_required_env("GEMINI_API_KEY"),
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )
    except Exception:
        return qwen_factory(
            model=os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL),
            api_key=_required_env("QWEN_API_KEY"),
            base_url=os.getenv("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL),
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )

    fallback = qwen_factory(
        model=os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL),
        api_key=_required_env("QWEN_API_KEY"),
        base_url=os.getenv("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL),
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )
    return FallbackChatModel(primary=primary, fallback=fallback)


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {value!r}") from exc


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc


def _google_factory(**kwargs: Any) -> ChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(**kwargs)


def _qwen_factory(**kwargs: Any) -> ChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(**kwargs)
