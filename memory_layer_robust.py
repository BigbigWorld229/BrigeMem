"""
Robust A-MEM memory layer — drop-in replacement for memory_layer.py.

Key differences from the original:
  - No response_format / JSON schema dependency in LLM calls
  - Plain-text prompts with section-marker parsing (via llm_text_parsers)
  - Structured logging instead of print()
  - Retry wrapper for transient LLM failures
  - Connectivity check on controller init
"""

from typing import Dict, Optional, Literal, Any
import re
import os
import time
import logging
import functools
from abc import ABC, abstractmethod
from memory_layer import SimpleEmbeddingRetriever
from token_usage import TokenUsageTracker
from cost_profiler import CostProfiler

logger = logging.getLogger("amem_robust")

DEFAULT_SYSTEM_MESSAGE = (
    "Follow the format specified in the prompt exactly. Do not add extra commentary."
)
QWEN3_SYSTEM_MESSAGE = (
    "Follow the format specified in the prompt exactly. Do not add extra commentary. "
    "Keep the output concise and rigorous."
    # "You are a conservative dialogue memory representation generator for long-term retrieval."
    # "Return only one valid JSON object matching the exact schema requested by the user. Do not output explanations, reasoning, markdown, or additional fields."
)


def resolve_system_message(model_type: str) -> str:
    """Select the system message without changing non-Qwen3 behavior."""
    if str(model_type or "").strip().lower() == "qwen3":
        return QWEN3_SYSTEM_MESSAGE
    if str(model_type or "").strip().lower() == "qwen3_8b":
        return QWEN3_SYSTEM_MESSAGE
    return DEFAULT_SYSTEM_MESSAGE


QWEN_TOKENIZER_MODEL_NAMES = {
    "qwen25": "Qwen/Qwen2.5-3B-Instruct",
    "qwen3": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3_8b": "/home/share/models/Qwen3-8B",
}

QWEN_LOCAL_TOKENIZER_FALLBACKS = {
    "qwen25": "/home/jwcao/xingzhuo/mimic_dataset_building/models/Qwen2.5-3B-Instruct",
    "qwen3": "/home/jwcao/xingzhuo/mimic_dataset_building/models/Qwen3-4B-Instruct-2507",
}



@functools.lru_cache(maxsize=None)
def load_qwen_tokenizer(model_type: str):
    normalized = str(model_type or "").strip().lower()
    model_name = QWEN_TOKENIZER_MODEL_NAMES.get(normalized)
    if model_name is None:
        return None

    from transformers import AutoTokenizer

    sources = [model_name]
    fallback = QWEN_LOCAL_TOKENIZER_FALLBACKS.get(normalized)
    if fallback:
        sources.append(fallback)
    
    errors = []
    for source in sources:
        try:
            return AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        except Exception as exc:
            errors.append(f"{source}: {exc}")
    
    raise RuntimeError(
        f"Failed to load tokenizer for model_type={normalized!r}. "
        f"Tried: {', '.join(sources)}. Errors: {' | '.join(errors)}"
    )
    # source = model_name
    # return AutoTokenizer.from_pretrained(source, trust_remote_code=True)


def resolve_qwen_chat_template(model_type: str) -> Optional[str]:
    tokenizer = load_qwen_tokenizer(model_type)
    if tokenizer is None:
        return None

    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        raise ValueError(f"Tokenizer for model_type={model_type!r} does not provide chat_template.")

    return chat_template

# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry_llm_call(max_retries: int = 2, base_delay: float = 1.0):
    """Decorator: retry an LLM call with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "LLM call %s failed (attempt %d/%d): %s — retrying in %.1fs",
                            func.__name__, attempt + 1, max_retries + 1, e, delay,
                        )
                        time.sleep(delay)
            logger.error("LLM call %s failed after %d attempts: %s",
                         func.__name__, max_retries + 1, last_exc)
            raise last_exc
        return wrapper
    return decorator




class RobustBaseLLMController(ABC):
    """Base class for robust LLM controllers (no JSON schema dependency)."""

    SYSTEM_MESSAGE = DEFAULT_SYSTEM_MESSAGE

    def _set_model_type(self, model_type: str) -> None:
        self.model_type = model_type
        self.SYSTEM_MESSAGE = resolve_system_message(model_type)

    @abstractmethod
    def get_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Get a plain-text completion from the LLM."""
        pass

    def check_connectivity(self):
        """Send a test call to verify the backend is reachable."""
        try:
            response = self.get_completion(
                "Reply with exactly one word: READY",
                temperature=0.0,
            )
            if not response or not response.strip():
                raise ConnectionError("Empty response from LLM backend")
            logger.info("LLM connectivity check passed (response: %s)", response.strip()[:50])
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach LLM backend: {e}. "
                "Check that the server is running and accessible."
            ) from e


class RobustOpenAIController(RobustBaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None,
                 model_type: str = "qwen25"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
        self.model = model
        self._set_model_type(model_type)
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key)

    @retry_llm_call(max_retries=2)
    def get_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens or 1000,
        )
        return response.choices[0].message.content

    @retry_llm_call(max_retries=2)
    def get_completion_with_usage(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens or 1000,
        )
        return response.choices[0].message.content, getattr(response, "usage", None)


class RobustOllamaController(RobustBaseLLMController):
    """Direct Ollama library controller (no LiteLLM proxy)."""

    def __init__(self, model: str = "llama2", model_type: str = "qwen25"):
        self.model = model
        self._set_model_type(model_type)

    @retry_llm_call(max_retries=2)
    def get_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            from ollama import chat
        except ImportError:
            raise ImportError("ollama package not found. Install it with: pip install ollama")
        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": temperature},
        )
        return response["message"]["content"]


class RobustSGLangController(RobustBaseLLMController):
    def __init__(self, model: str = "llama2",
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 model_type: str = "qwen25"):
        import requests as _requests
        self._requests = _requests
        self.model = model
        self.base_url = f"{sglang_host}:{sglang_port}"
        self._set_model_type(model_type)

    @retry_llm_call(max_retries=2)
    def get_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_tokens or 1000,
            }
        }
        response = self._requests.post(
            f"{self.base_url}/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if response.status_code == 200:
            return response.json().get("text", "")
        raise RuntimeError(f"SGLang server returned status {response.status_code}: {response.text}")


class RobustVLLMController(RobustBaseLLMController):
    """Controller for vLLM's OpenAI-compatible API server."""

    def __init__(self, model: str = "llama2",
                 vllm_host: str = "http://localhost",
                 vllm_port: int = 30000,
                 model_type: str = "qwen25"):
        import requests as _requests
        self._requests = _requests
        self.model = model
        self.base_url = f"{vllm_host}:{vllm_port}"
        self._set_model_type(model_type)
        self.chat_template = resolve_qwen_chat_template(model_type)
        self.disable_thinking = str(model_type or "").strip().lower() == "qwen3_8b"
        print(
            "[DEBUG_QWEN_COMPARE] VLLM_CONTROLLER_INIT "
            f"model={self.model} tokenizer_name={self.model} base_url={self.base_url} "
            f"max_model_len_expected=8192 model_type={self.model_type}"
        )

    @retry_llm_call(max_retries=2)
    def get_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens or 1000,
        }
        if self.chat_template is not None:
            payload["chat_template"] = self.chat_template
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        print(
            "[DEBUG_QWEN_COMPARE] LLM_HTTP_BEFORE "
            f"method=get_completion model={self.model} tokenizer_name={self.model} "
            f"base_url={self.base_url} max_model_len_expected=8192 max_tokens={payload['max_tokens']} "
            f"temperature={temperature} system_chars={len(self.SYSTEM_MESSAGE)} user_chars={len(prompt)} "
            f"system_tokens_est={len(self.SYSTEM_MESSAGE.split())} user_tokens_est={len(str(prompt).split())}"
        )
        response = self._requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            body = response.json()
            print(
                "[DEBUG_QWEN_COMPARE] LLM_HTTP_AFTER "
                f"method=get_completion model={self.model} status=200 usage={body.get('usage')}"
            )
            return body["choices"][0]["message"]["content"]
        print(
            "[DEBUG_QWEN_COMPARE] LLM_HTTP_ERROR "
            f"method=get_completion model={self.model} status={response.status_code} "
            f"response_preview={response.text[:100]!r}"
        )
        raise RuntimeError(f"vLLM server returned status {response.status_code}: {response.text}")

    @retry_llm_call(max_retries=2)
    def get_completion_with_usage(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens or 1000,
        }
        if self.chat_template is not None:
            payload["chat_template"] = self.chat_template
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        print(
            "[DEBUG_QWEN_COMPARE] LLM_HTTP_BEFORE "
            f"method=get_completion_with_usage model={self.model} tokenizer_name={self.model} "
            f"base_url={self.base_url} max_model_len_expected=8192 max_tokens={payload['max_tokens']} "
            f"temperature={temperature} system_chars={len(self.SYSTEM_MESSAGE)} user_chars={len(prompt)} "
            f"system_tokens_est={len(self.SYSTEM_MESSAGE.split())} user_tokens_est={len(str(prompt).split())}"
        )
        response = self._requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            body = response.json()
            print(
                "[DEBUG_QWEN_COMPARE] LLM_HTTP_AFTER "
                f"method=get_completion_with_usage model={self.model} status=200 usage={body.get('usage')}"
            )
            return body["choices"][0]["message"]["content"], body.get("usage")
        print(
            "[DEBUG_QWEN_COMPARE] LLM_HTTP_ERROR "
            f"method=get_completion_with_usage model={self.model} status={response.status_code} "
            f"response_preview={response.text[:100]!r}"
        )
        raise RuntimeError(f"vLLM server returned status {response.status_code}: {response.text}")


class RobustLiteLLMController(RobustBaseLLMController):
    """LiteLLM controller for universal LLM access (Ollama, SGLang, etc.)."""

    def __init__(self, model: str, api_base: Optional[str] = None,
                 api_key: Optional[str] = None, model_type: str = "qwen25"):
        from litellm import completion as _completion
        self._completion = _completion
        self.model = model
        self.api_base = api_base
        self.api_key = api_key or "EMPTY"
        self._set_model_type(model_type)

    @retry_llm_call(max_retries=2)
    def get_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        completion_args = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            completion_args["max_tokens"] = max_tokens
        if self.api_base:
            completion_args["api_base"] = self.api_base
        if self.api_key:
            completion_args["api_key"] = self.api_key

        response = self._completion(**completion_args)
        return response.choices[0].message.content

    @retry_llm_call(max_retries=2)
    def get_completion_with_usage(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, Any]:
        completion_args = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            completion_args["max_tokens"] = max_tokens
        if self.api_base:
            completion_args["api_base"] = self.api_base
        if self.api_key:
            completion_args["api_key"] = self.api_key

        response = self._completion(**completion_args)
        return response.choices[0].message.content, getattr(response, "usage", None)




class RobustLLMController:
    """Factory that selects the right robust LLM controller."""

    def __init__(self,
                 backend: Literal["openai", "ollama", "sglang", "vllm"] = "sglang",
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 check_connection: bool = False,
                 usage_tracker: Optional[TokenUsageTracker] = None,
                 usage_op_name: str = "llm_call",
                 model_type: str = "qwen25",
                 cost_profiler: Optional[CostProfiler] = None):
        self.usage_tracker = usage_tracker
        self.usage_op_name = usage_op_name
        self.model_type = model_type
        self.cost_profiler = cost_profiler
        if backend == "openai":
            self.llm = RobustOpenAIController(model, api_key, model_type=model_type)
        elif backend == "ollama":
            self.llm = RobustOllamaController(model, model_type=model_type)
        elif backend == "sglang":
            self.llm = RobustSGLangController(model, sglang_host, sglang_port, model_type=model_type)
        elif backend == "vllm":
            self.llm = RobustVLLMController(model, sglang_host, sglang_port, model_type=model_type)
        else:
            raise ValueError("Backend must be 'openai', 'ollama', 'sglang', or 'vllm'")

        if check_connection:
            self.llm.check_connectivity()

    @staticmethod
    def _parse_positive_int_env(key: str) -> Optional[int]:
        raw = os.getenv(key)
        if raw is None or raw == "":
            return None
        try:
            value = int(raw)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r; expected a positive integer.", key, raw)
            return None
        if value <= 0:
            logger.warning("Ignoring non-positive %s=%r.", key, raw)
            return None
        return value

    def _resolve_max_tokens(self) -> Optional[int]:
        op_key = re.sub(r"[^A-Z0-9]+", "_", self.usage_op_name.upper()).strip("_")
        if op_key:
            op_value = self._parse_positive_int_env(f"MEMWEAVE_MAX_TOKENS_{op_key}")
            if op_value is not None:
                return op_value
        return self._parse_positive_int_env("MEMWEAVE_MAX_TOKENS")

    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        start_time = time.time()
        usage = None
        response_text = ""
        max_tokens = self._resolve_max_tokens()
        print(
            "[DEBUG_QWEN_COMPARE] LLM_CALL "
            f"operation={self.usage_op_name} model={getattr(self.llm, 'model', 'unknown')} "
            f"prompt_chars={len(str(prompt))} prompt_tokens_est={len(str(prompt).split())} "
            f"max_tokens_resolved={max_tokens if max_tokens is not None else 1000} temperature={temperature}"
        )
        profiling_enabled = bool(self.cost_profiler and self.cost_profiler.enabled)
        call_started = time.perf_counter() if profiling_enabled else None
        call_error = None
        try:
            if hasattr(self.llm, "get_completion_with_usage"):
                response_text, usage = self.llm.get_completion_with_usage(
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                response_text = self.llm.get_completion(
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
        except BaseException as exc:
            call_error = exc
            raise
        finally:
            if profiling_enabled:
                call_latency_seconds = time.perf_counter() - call_started
                self._record_cost_profile_call(
                    prompt=prompt,
                    response_text=response_text,
                    usage=usage,
                    latency_seconds=call_latency_seconds,
                    status="failed" if call_error is not None else "success",
                    error=call_error,
                )

        if self.usage_tracker is not None:
            self.usage_tracker.record(
                self.usage_op_name,
                prompt_text=prompt,
                completion_text=response_text,
                usage=usage,
                runtime_seconds=time.time() - start_time,
            )
        print(
            "[DEBUG_QWEN_COMPARE] LLM_CALL_DONE "
            f"operation={self.usage_op_name} response_chars={len(str(response_text))} "
            f"usage={usage} runtime_seconds={time.time() - start_time:.4f}"
        )
        return response_text

    def _record_cost_profile_call(
        self,
        *,
        prompt: str,
        response_text: str,
        usage: Any,
        latency_seconds: float,
        status: str,
        error: Optional[BaseException],
    ) -> None:
        """Record one logical controller call without affecting its result."""
        try:
            self.cost_profiler.record_llm_call(
                operation=self.usage_op_name,
                prompt=prompt,
                completion=response_text,
                usage=usage,
                latency_seconds=latency_seconds,
                status=status,
                error=error,
                tokenizer_factory=lambda: load_qwen_tokenizer(self.model_type),
                system_message=getattr(self.llm, "SYSTEM_MESSAGE", ""),
                model_name=getattr(self.llm, "model", ""),
            )
        except Exception as profiler_exc:
            logger.warning("Cost profiling record failed: %s", profiler_exc)


# ---------------------------------------------------------------------------
# RobustAgenticMemorySystem
# ---------------------------------------------------------------------------

class RobustAgenticMemorySystem:
    """Memory management system using plain-text LLM calls (no JSON schema)."""

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "sglang",
                 llm_model: str = "gpt-4o-mini",
                 llm_model_type: str = "qwen25",
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 check_connection: bool = False,
                 usage_tracker: Optional[TokenUsageTracker] = None,
                 cost_profiler: Optional[CostProfiler] = None):

        self.memories: Dict[str, Any] = {}
        self.retriever = SimpleEmbeddingRetriever(model_name)
        self.usage_tracker = usage_tracker
        self.llm_controller = RobustLLMController(
            llm_backend, llm_model, api_key, api_base,
            sglang_host, sglang_port, check_connection,
            usage_tracker=usage_tracker,
            usage_op_name="memory_metadata",
            model_type=llm_model_type,
            cost_profiler=cost_profiler,
        )
        print(
            "[DEBUG_QWEN_COMPARE] MEMORY_SYSTEM_INIT "
            f"embedding_model={model_name} llm_model={llm_model} llm_model_type={llm_model_type} backend={llm_backend} "
            f"host={sglang_host} port={sglang_port}"
        )
