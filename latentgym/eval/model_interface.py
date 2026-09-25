"""
ModelInterface — Abstract interface for models that generate text responses.

Implementations:
- OpenAIModel: OpenAI + OpenAI-compatible APIs (OpenRouter, Together, etc.)
- AnthropicModel: Anthropic Claude models
- GoogleModel: Google Gemini models
- VLLMModel: Locally-served models via vLLM
- MockModel: Deterministic mock for testing

All API models include:
- Retry with exponential backoff
- Rate limit detection (429 → longer wait)
- Per-request timeout (prevents hung calls)
- Logging of retries and errors

Reasoning/thinking support:
- generate() returns ModelResponse(text, reasoning) — not a plain string.
- `text` is the action (what gets fed back into conversation context).
- `reasoning` is the internal thinking (recorded for analysis, NOT fed back).
- Models without thinking support return reasoning=None.
- Anthropic: thinking blocks (type="thinking") extracted separately from text blocks.
- Google Gemini: parts with thought=True extracted separately.
- vLLM: message.reasoning_content extracted when --reasoning-parser is enabled.
- OpenAI ChatCompletions: reasoning not exposed (discarded by API). Would need
  Responses API for summaries; we use ChatCompletions so reasoning=None.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default timeout per API call (seconds)
DEFAULT_REQUEST_TIMEOUT = 120


async def _retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
    model_name: str = "",
) -> Any:
    """Execute an async function with retry, rate limit handling, and timeout.

    - Transient errors (timeout, 429, 500, 502, 503, 529): retry with backoff
    - Rate limit (429): wait longer (base 30s + exponential)
    - Permanent errors (401, 403, 404): raise immediately, no retry
    - Per-request timeout: cancels hung calls

    Args:
        fn: Async callable that makes the API call
        max_retries: Max retry attempts
        timeout: Seconds before a single request is cancelled
        model_name: For logging

    Returns:
        Result of fn()
    """
    for attempt in range(max_retries):
        try:
            return await asyncio.wait_for(fn(), timeout=timeout)

        except asyncio.TimeoutError:
            if attempt == max_retries - 1:
                logger.error(f"[{model_name}] Request timed out after {timeout}s on attempt {attempt+1}/{max_retries}")
                raise
            wait = min(2 ** attempt * 5, 60)
            logger.warning(f"[{model_name}] Timeout after {timeout}s, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
            await asyncio.sleep(wait)

        except Exception as e:
            error_str = str(e).lower()
            status_code = getattr(e, 'status_code', None) or _extract_status_code(e)

            # Permanent errors — don't retry
            if status_code in (401, 403, 404):
                logger.error(f"[{model_name}] Permanent error (HTTP {status_code}): {e}")
                raise

            # Rate limit — wait longer
            if status_code == 429 or 'rate' in error_str:
                if attempt == max_retries - 1:
                    logger.error(f"[{model_name}] Rate limited, no retries left")
                    raise
                wait = min(30 + 2 ** attempt * 10, 120)
                logger.warning(f"[{model_name}] Rate limited (429), waiting {wait}s (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait)

            # Transient errors — standard backoff
            elif status_code in (500, 502, 503, 529) or 'overloaded' in error_str or 'timeout' in error_str:
                if attempt == max_retries - 1:
                    logger.error(f"[{model_name}] Server error after {max_retries} attempts: {e}")
                    raise
                wait = min(2 ** attempt * 2, 30)
                logger.warning(f"[{model_name}] Transient error (HTTP {status_code}), retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait)

            # Unknown error — retry with standard backoff
            else:
                if attempt == max_retries - 1:
                    logger.error(f"[{model_name}] Failed after {max_retries} attempts: {e}")
                    raise
                wait = 2 ** attempt
                logger.warning(f"[{model_name}] Error: {e}, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait)

    return ""  # should not reach here


def _extract_status_code(e: Exception) -> Optional[int]:
    """Try to extract HTTP status code from various exception types."""
    # OpenAI, Anthropic exceptions often have status_code
    for attr in ('status_code', 'status', 'code'):
        code = getattr(e, attr, None)
        if isinstance(code, int):
            return code
    # Check in string representation
    import re
    match = re.search(r'(\d{3})', str(e)[:100])
    if match:
        code = int(match.group(1))
        if 400 <= code <= 599:
            return code
    return None


@dataclass
class ModelResponse:
    """Response from a model, separating action text from reasoning.

    - text: The action/answer (fed back into conversation as assistant message).
    - reasoning: Internal thinking/chain-of-thought (recorded but NOT fed back).
      None if the model doesn't support or didn't produce reasoning.
    """
    text: str
    reasoning: Optional[str] = None


class ModelInterface(ABC):
    """Abstract interface for any model that can generate text responses.

    generate() returns ModelResponse with separate text and reasoning fields.
    The runner puts only response.text into the conversation context.
    """

    def __init__(self, name: str, **kwargs):
        self.name = name

    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        ...


class MockModel(ModelInterface):
    """Deterministic mock model for testing."""

    def __init__(self, name: str = "mock", responses: Optional[List[str]] = None,
                 default_response: str = "[red]"):
        super().__init__(name)
        self._responses = responses or []
        self._default_response = default_response
        self._call_count = 0

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        idx = self._call_count
        self._call_count += 1
        if idx < len(self._responses):
            return ModelResponse(text=self._responses[idx])
        return ModelResponse(text=self._default_response)


class OpenAIModel(ModelInterface):
    """OpenAI models (GPT-4o, GPT-4o-mini, GPT-5, etc.) via openai SDK.

    Also works with OpenAI-compatible APIs (OpenRouter, Together, etc.)
    by setting base_url.

    Reasoning support:
    - OpenAI ChatCompletions: reasoning NOT exposed (discarded after each request).
      Would need Responses API for reasoning summaries. Returns reasoning=None.
    - OpenRouter: reasoning IS available via message.reasoning field when
      include_reasoning=True. Automatically detected when base_url contains
      'openrouter'. Captures reasoning from all providers (Claude, o-series,
      Gemini, DeepSeek, etc.) in a unified format.
    """

    def __init__(
        self,
        name: str,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 3,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ):
        super().__init__(name)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self._client = None
        # Auto-detect OpenRouter for reasoning extraction
        self._is_openrouter = bool(base_url and "openrouter" in base_url)

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        try:
            client = self._get_client()
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

        is_openrouter = self._is_openrouter

        async def _call():
            extra_params = {}
            # OpenRouter: request reasoning tokens via extra_body
            # Docs: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
            #
            # reasoning: {}       → enables reasoning at medium effort (provider default)
            # include_reasoning   → legacy alias for reasoning: {}
            # Non-thinking models (GPT-4o) ignore these parameters.
            #
            # Provider behavior:
            # - OpenAI o-series: reasons internally, does NOT return reasoning text
            #   but include_reasoning captures it via OpenRouter's normalization
            # - Anthropic: returns reasoning + reasoning_details (adaptive thinking)
            # - Gemini: returns reasoning, needs explicit effort for thinkingLevel
            if is_openrouter:
                model_lower = self.model.lower()
                if "gemini" in model_lower or "google" in model_lower:
                    # Gemini: explicit effort for thinkingLevel
                    extra_params["extra_body"] = {
                        "include_reasoning": True,
                        "reasoning": {"effort": "high"},
                    }
                elif "openai" in model_lower:
                    # OpenAI: include_reasoning alone works (proven with o4-mini)
                    extra_params["extra_body"] = {
                        "include_reasoning": True,
                    }
                else:
                    # Anthropic, DeepSeek, etc.
                    # For Anthropic: effort calculates budget from max_tokens,
                    # but our max_tokens (512) may be too low. Use explicit
                    # reasoning max_tokens instead (min 1024 per Anthropic).
                    extra_params["extra_body"] = {
                        "include_reasoning": True,
                        "reasoning": {"max_tokens": 4096},
                    }

            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                **extra_params,
            )
            msg = response.choices[0].message

            # Extract reasoning:
            # - OpenRouter: message.reasoning (or reasoning_content alias)
            # - OpenAI direct ChatCompletions: not available (discarded by API)
            #
            # The OpenAI SDK preserves extra fields via model_extra dict and
            # direct attribute access. We check both for robustness.
            reasoning = None
            if is_openrouter:
                reasoning = (
                    getattr(msg, "reasoning", None)
                    or getattr(msg, "reasoning_content", None)
                )
                # Fallback: check model_extra dict (some SDK versions)
                if not reasoning and hasattr(msg, "model_extra") and msg.model_extra:
                    reasoning = (
                        msg.model_extra.get("reasoning")
                        or msg.model_extra.get("reasoning_content")
                    )

            return ModelResponse(text=msg.content or "", reasoning=reasoning)

        return await _retry_with_backoff(
            _call,
            max_retries=self.max_retries,
            timeout=self.request_timeout,
            model_name=self.name,
        )


class AnthropicModel(ModelInterface):
    """Anthropic models (Claude Sonnet, Haiku, etc.) via anthropic SDK.

    Reasoning support:
    - Set enable_thinking=True to capture internal reasoning.
    - thinking_budget controls max tokens for thinking (default 10000).
    - When thinking is enabled, temperature is forced to 1 (Anthropic requirement).
    - Thinking blocks are extracted separately and NOT fed back into conversation.
    """

    def __init__(
        self,
        name: str,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 3,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        enable_thinking: bool = False,
        thinking_budget: int = 10000,
    ):
        super().__init__(name)
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        try:
            client = self._get_client()
        except ImportError:
            raise ImportError("anthropic is required. Install with: pip install anthropic")

        system_msg = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)

        enable_thinking = self.enable_thinking
        thinking_budget = self.thinking_budget

        async def _call():
            create_params = dict(
                model=self.model,
                system=system_msg if system_msg else "",
                messages=chat_messages,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            if enable_thinking:
                # Extended thinking: temperature must be 1 (Anthropic requirement)
                create_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
                create_params["temperature"] = 1
            else:
                create_params["temperature"] = kwargs.get("temperature", self.temperature)

            response = await client.messages.create(**create_params)
            # Separate thinking blocks from text blocks.
            # With extended thinking enabled, content is:
            #   [{type: "thinking", thinking: "..."}, {type: "text", text: "..."}]
            # Without thinking, content is:
            #   [{type: "text", text: "..."}]
            thinking_parts = []
            text_parts = []
            for block in response.content:
                if getattr(block, "type", None) == "thinking":
                    thinking_parts.append(getattr(block, "thinking", ""))
                elif getattr(block, "type", None) == "text":
                    text_parts.append(block.text)

            text = "\n".join(text_parts) if text_parts else ""
            reasoning = "\n".join(thinking_parts) if thinking_parts else None
            return ModelResponse(text=text, reasoning=reasoning)

        return await _retry_with_backoff(
            _call,
            max_retries=self.max_retries,
            timeout=self.request_timeout,
            model_name=self.name,
        )


class GoogleModel(ModelInterface):
    """Google models (Gemini Pro, Flash, etc.) via google-genai SDK.

    Reasoning support:
    - Set enable_thinking=True to capture thought summaries from Gemini 2.5+.
    - thinking_budget controls max thinking tokens (default 8192).
    - Thought parts (part.thought=True) are extracted separately.
    """

    def __init__(
        self,
        name: str,
        model: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 3,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        enable_thinking: bool = False,
        thinking_budget: int = 8192,
    ):
        super().__init__(name)
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        try:
            client = self._get_client()
        except ImportError:
            raise ImportError("google-genai is required. Install with: pip install google-genai")

        gemini_messages = []
        system_instruction = ""
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [{"text": msg["content"]}]})
            else:
                gemini_messages.append({"role": "user", "parts": [{"text": msg["content"]}]})

        enable_thinking = self.enable_thinking
        thinking_budget = self.thinking_budget

        async def _call():
            config = {
                "system_instruction": system_instruction,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            if enable_thinking:
                config["thinking_config"] = {
                    "include_thoughts": True,
                    "thinking_budget": thinking_budget,
                }

            response = await client.aio.models.generate_content(
                model=self.model,
                contents=gemini_messages,
                config=config,
            )
            # Separate thought parts from response parts.
            # With thinking enabled, parts have a `thought` boolean:
            #   part.thought=True → internal thinking
            #   part.thought=False (or absent) → action text
            thinking_parts = []
            text_parts = []
            parsed_parts = False
            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if not hasattr(part, "text") or not part.text:
                        continue
                    parsed_parts = True
                    if getattr(part, "thought", False):
                        thinking_parts.append(part.text)
                    else:
                        text_parts.append(part.text)

            # Only fall back to response.text if we couldn't parse parts at all.
            # Don't use response.text as fallback when thinking is active because
            # it concatenates ALL parts (including thoughts) into one string.
            if text_parts:
                text = "\n".join(text_parts)
            elif not parsed_parts:
                # No parts parsed — use the convenience property
                text = response.text or ""
            else:
                # Parts were parsed but all were thoughts (no action text).
                # This shouldn't happen normally but handle gracefully.
                text = ""
            reasoning = "\n".join(thinking_parts) if thinking_parts else None
            return ModelResponse(text=text, reasoning=reasoning)

        return await _retry_with_backoff(
            _call,
            max_retries=self.max_retries,
            timeout=self.request_timeout,
            model_name=self.name,
        )


class VLLMModel(ModelInterface):
    """Locally-served models via vLLM's OpenAI-compatible API.

    Requires a running vLLM server:
        python -m vllm.entrypoints.openai.api_server --model /path/to/model --port 8000
    """

    def __init__(
        self,
        name: str,
        base_url: str = "http://localhost:8000/v1",
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 3,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ):
        super().__init__(name)
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(base_url=self.base_url, api_key="dummy")
        return self._client

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        try:
            client = self._get_client()
        except ImportError:
            raise ImportError("openai is required for VLLMModel. Install with: pip install openai")

        async def _call():
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            msg = response.choices[0].message
            # vLLM exposes reasoning_content when --reasoning-parser is enabled
            # (e.g., for DeepSeek-R1, QwQ). Field is absent for non-reasoning models.
            reasoning = getattr(msg, "reasoning_content", None)
            # Fallback: check model_extra dict
            if not reasoning and hasattr(msg, "model_extra") and msg.model_extra:
                reasoning = msg.model_extra.get("reasoning_content")
            return ModelResponse(text=msg.content or "", reasoning=reasoning)

        return await _retry_with_backoff(
            _call,
            max_retries=self.max_retries,
            timeout=self.request_timeout,
            model_name=self.name,
        )
