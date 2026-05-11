"""
OpenAI API interface for LLMs

This module also supports a "manual mode" (human-in-the-loop) where prompts are written
to a task queue directory and the system waits for a corresponding *.answer.json file
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import openai

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


# --- MLS-Bench fork delta: token-usage observer hook -----------------------
# When set, invoked after every successful LLM call with a dict containing
# model, prompt_tokens, completion_tokens, total_tokens, cached_tokens, and
# raw usage payload. Used by MLS-Bench's OpenEvolveAgent to populate
# tokens.jsonl for performance-vs-tokens scaling analysis. Unset by default,
# in which case behavior matches upstream exactly.
_token_observer: Optional[Callable[[Dict[str, Any]], None]] = None


def set_token_observer(fn: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Install or clear the token-usage observer."""
    global _token_observer
    _token_observer = fn


# OpenRouter upstream-pinning map (matches the one in src/mlsbench/agent/
# models.py so both agents route identically through OpenRouter).
_OPENROUTER_PROVIDER_ORDER_BY_PREFIX = {
    "anthropic/": ["anthropic"],
    "openai/": ["openai"],
    "google/": ["google-ai-studio"],
    "deepseek/": ["deepseek"],
    "qwen/": ["alibaba"],
    "moonshot/": ["moonshot"],
    "meta-llama/": ["groq", "togetherai"],
}


def _openrouter_provider_order(model: str) -> Optional[List[str]]:
    m = str(model or "")
    for prefix, order in _OPENROUTER_PROVIDER_ORDER_BY_PREFIX.items():
        if m.startswith(prefix):
            return list(order)
    return None


def _extract_usage(response: Any, model: str) -> Dict[str, Any]:
    """Pull prompt/completion/cached token counts out of an OpenAI-compatible
    response. Tolerant of missing fields across providers (OpenAI, OpenRouter,
    DeepSeek, Gemini via OpenAI-compatible route, etc)."""
    usage = getattr(response, "usage", None)
    record: Dict[str, Any] = {"model": model, "timestamp": _iso_now()}
    if usage is None:
        return record
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    total = getattr(usage, "total_tokens", None)
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    # Some providers report cache reads under a different key (Anthropic via
    # OpenRouter surfaces cache_read_input_tokens / cache_creation_input_tokens
    # on the usage object when available).
    if not cached:
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_created = getattr(usage, "cache_creation_input_tokens", 0) or 0
    record.update(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cached_tokens=cached,
        cache_creation_tokens=cache_created,
    )
    return record
# --- end MLS-Bench fork delta ---------------------------------------------


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _build_display_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Render messages into a single plain-text prompt for the manual UI.
    """
    chunks: List[str] = []
    for m in messages:
        role = str(m.get("role", "user")).upper()
        content = m.get("content", "")
        chunks.append(f"### {role}\n{content}\n")
    return "\n".join(chunks).rstrip() + "\n"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class OpenAILLM(LLMInterface):
    """LLM interface using OpenAI-compatible APIs"""

    def __init__(
        self,
        model_cfg: Optional[dict] = None,
    ):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base
        self.api_key = model_cfg.api_key
        self.random_seed = getattr(model_cfg, "random_seed", None)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)

        # Manual mode: enabled via llm.manual_mode in config.yaml
        self.manual_mode = (getattr(model_cfg, "manual_mode", False) is True)
        self.manual_queue_dir: Optional[Path] = None

        if self.manual_mode:
            qdir = getattr(model_cfg, "_manual_queue_dir", None)
            if not qdir:
                raise ValueError(
                    "Manual mode is enabled but manual_queue_dir is missing. "
                    "This should be injected by the OpenEvolve controller."
                )
            self.manual_queue_dir = Path(str(qdir)).expanduser().resolve()
            self.manual_queue_dir.mkdir(parents=True, exist_ok=True)
            self.client = None
        else:
            # Set up API client (normal mode)
            # OpenAI client requires max_retries to be int, not None
            max_retries = self.retries if self.retries is not None else 0
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout,
                max_retries=max_retries,
            )

        # Only log unique models to reduce duplication
        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()

        if self.model not in logger._initialized_models:
            logger.info(f"Initialized OpenAI LLM with model: {self.model}")
            logger._initialized_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        # Prepare messages with system message
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages.extend(messages)

        # Set up generation parameters
        # Define OpenAI reasoning models that require max_completion_tokens
        # These models don't support temperature/top_p and use different parameters
        OPENAI_REASONING_MODEL_PREFIXES = (
            # O-series reasoning models
            "o1-",
            "o1",  # o1, o1-mini, o1-preview
            "o3-",
            "o3",  # o3, o3-mini, o3-pro
            "o4-",  # o4-mini
            # GPT-5 series are also reasoning models
            "gpt-5-",
            "gpt-5",  # gpt-5, gpt-5-mini, gpt-5-nano
            # The GPT OSS series are also reasoning models
            "gpt-oss-120b",
            "gpt-oss-20b",
        )

        # Check if this is an OpenAI reasoning model based on model name pattern
        # This works for all endpoints (OpenAI, Azure, OptiLLM, OpenRouter, etc.)
        model_lower = str(self.model).lower()
        is_openai_reasoning_model = model_lower.startswith(OPENAI_REASONING_MODEL_PREFIXES)

        if is_openai_reasoning_model:
            # For OpenAI reasoning models
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            # Add optional reasoning parameters if provided
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort
            if "verbosity" in kwargs:
                params["verbosity"] = kwargs["verbosity"]
        else:
            # Standard parameters for all other models
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }

            # Handle reasoning_effort for open source reasoning models.
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort

        # Add seed parameter for reproducibility if configured
        # Skip seed for any Gemini endpoint — Google AI Studio direct, AND
        # LiteLLM-proxied Gemini both reject the OpenAI `seed` param. We detect
        # by either api_base or model name so a litellm proxy at a generic URL
        # is also covered (model="gemini/gemini-3.1-pro-preview" etc.).
        # Seed only makes sense for actual API calls.
        seed = kwargs.get("seed", self.random_seed)
        if seed is not None and not self.manual_mode:
            is_gemini = (
                self.api_base == "https://generativelanguage.googleapis.com/v1beta/openai/"
                or "gemini" in str(self.model).lower()
            )
            if is_gemini:
                logger.warning(
                    "Skipping seed parameter — Gemini endpoints don't support OpenAI `seed`. "
                    "Reproducibility may be limited."
                )
            else:
                params["seed"] = seed

        # MLS-Bench fork delta: for OpenRouter endpoints, pin the upstream
        # provider so sticky-routing doesn't rotate us onto a cold prompt
        # cache, and turn on usage accounting so prompt_tokens_details.
        # cached_tokens comes back populated. Claude prefix caching still
        # requires cache_control on content blocks — OpenEvolve's prompt
        # sampler doesn't produce those, so for heavy Claude use on
        # OpenRouter we recommend the LiteLLM path instead (which does
        # implicit server-side caching for Claude without per-block flags).
        if "openrouter" in str(self.api_base).lower():
            extra_body = dict(params.get("extra_body") or {})
            order = _openrouter_provider_order(self.model)
            if order:
                extra_body.setdefault("provider", {})
                extra_body["provider"].setdefault("order", order)
                extra_body["provider"].setdefault("allow_fallbacks", False)
            extra_body.setdefault("usage", {"include": True})
            params["extra_body"] = extra_body

        # Attempt the API call with retries
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)

        # Manual mode: no timeout unless explicitly passed by the caller
        if self.manual_mode:
            timeout = kwargs.get("timeout", None)
            return await self._manual_wait_for_answer(params, timeout=timeout)

        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_api(params), timeout=timeout)
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {str(e)}")
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> str:
        """Make the actual API call"""
        if self.client is None:
            raise RuntimeError("OpenAI client is not initialized (manual_mode enabled?)")

        # Use asyncio to run the blocking API call in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.chat.completions.create(**params)
        )
        # Logging of system prompt, user message and response content
        logger = logging.getLogger(__name__)
        logger.debug(f"API parameters: {params}")
        logger.debug(f"API response: {response.choices[0].message.content}")

        # MLS-Bench fork delta: report token usage to the observer if installed,
        # and also append to MLSBENCH_OE_TOKENS_LOG when set. Env-var fallback
        # ensures worker subprocesses spawned by ProcessPoolExecutor also log
        # usage, even though they re-import this module fresh and lose any
        # programmatic observer set in the parent.
        usage_record = None
        if _token_observer is not None or os.environ.get("MLSBENCH_OE_TOKENS_LOG"):
            try:
                usage_record = _extract_usage(response, self.model)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"extract_usage raised: {exc}")
        if usage_record is not None and _token_observer is not None:
            try:
                _token_observer(usage_record)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"token observer raised: {exc}")
        # Only write to the env-var log when the programmatic observer isn't
        # already handling it — otherwise parent-process calls (which have both
        # paths) would log twice. Workers (no observer) use the env var alone.
        if usage_record is not None and _token_observer is None:
            log_path = os.environ.get("MLSBENCH_OE_TOKENS_LOG")
            if log_path:
                try:
                    with open(log_path, "a") as f:
                        f.write(json.dumps(usage_record) + "\n")
                except Exception as exc:  # pragma: no cover
                    logger.warning(f"token log write failed: {exc}")

        return response.choices[0].message.content

    async def _manual_wait_for_answer(
        self, params: Dict[str, Any], timeout: Optional[Union[int, float]]
    ) -> str:
        """
        Manual mode: write a task JSON file and poll for *.answer.json
        If timeout is provided, we respect it; otherwise we wait indefinitely
        """

        if self.manual_queue_dir is None:
            raise RuntimeError("manual_queue_dir is not initialized")

        task_id = str(uuid.uuid4())
        messages = params.get("messages", [])
        display_prompt = _build_display_prompt(messages)

        task_payload: Dict[str, Any] = {
            "id": task_id,
            "created_at": _iso_now(),
            "model": params.get("model"),
            "display_prompt": display_prompt,
            "messages": messages,
            "meta": {
                "max_tokens": params.get("max_tokens"),
                "max_completion_tokens": params.get("max_completion_tokens"),
                "temperature": params.get("temperature"),
                "top_p": params.get("top_p"),
                "reasoning_effort": params.get("reasoning_effort"),
                "verbosity": params.get("verbosity"),
            },
        }

        task_path = self.manual_queue_dir / f"{task_id}.json"
        answer_path = self.manual_queue_dir / f"{task_id}.answer.json"

        _atomic_write_json(task_path, task_payload)
        logger.info(f"[manual_mode] Task enqueued: {task_path}")

        start = time.time()
        poll_interval = 0.5

        while True:
            if answer_path.exists():
                try:
                    data = json.loads(answer_path.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning(f"[manual_mode] Failed to parse answer JSON for {task_id}: {e}")
                    await asyncio.sleep(poll_interval)
                    continue

                answer = str(data.get("answer") or "")
                logger.info(f"[manual_mode] Answer received for {task_id}")
                return answer

            if timeout is not None and (time.time() - start) > float(timeout):
                raise asyncio.TimeoutError(
                    f"Manual mode timed out after {timeout} seconds waiting for answer of task {task_id}"
                )

            await asyncio.sleep(poll_interval)
