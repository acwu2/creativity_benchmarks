"""Anthropic API client implementation."""

import time
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from llm_benchmarks.clients.base import (
    BaseLLMClient,
    GenerationConfig,
    InvalidModelError,
    LLMResponse,
    Message,
    MessageRole,
    ModelInfo,
    TokenUsage,
)
from llm_benchmarks.config import Settings


# Model information for common Anthropic models
ANTHROPIC_MODELS: dict[str, ModelInfo] = {
    "claude-3-5-sonnet-20241022": ModelInfo(
        provider="anthropic",
        model_id="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        context_window=200000,
        max_output_tokens=8192,
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        supports_vision=True,
        supports_tools=True,
    ),
    "claude-3-5-haiku-20241022": ModelInfo(
        provider="anthropic",
        model_id="claude-3-5-haiku-20241022",
        display_name="Claude 3.5 Haiku",
        context_window=200000,
        max_output_tokens=8192,
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.005,
        supports_vision=True,
        supports_tools=True,
    ),
    "claude-3-opus-20240229": ModelInfo(
        provider="anthropic",
        model_id="claude-3-opus-20240229",
        display_name="Claude 3 Opus",
        context_window=200000,
        max_output_tokens=4096,
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
        supports_vision=True,
        supports_tools=True,
    ),
    "claude-3-sonnet-20240229": ModelInfo(
        provider="anthropic",
        model_id="claude-3-sonnet-20240229",
        display_name="Claude 3 Sonnet",
        context_window=200000,
        max_output_tokens=4096,
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        supports_vision=True,
        supports_tools=True,
    ),
    "claude-3-haiku-20240307": ModelInfo(
        provider="anthropic",
        model_id="claude-3-haiku-20240307",
        display_name="Claude 3 Haiku",
        context_window=200000,
        max_output_tokens=4096,
        input_cost_per_1k=0.00025,
        output_cost_per_1k=0.00125,
        supports_vision=True,
        supports_tools=True,
    ),
}


class AnthropicClient(BaseLLMClient):
    """Anthropic API client."""
    
    provider = "anthropic"
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        """
        Initialize the Anthropic client.
        
        Args:
            model: Model identifier (e.g., "claude-3-5-sonnet-20241022")
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            base_url: Optional base URL override
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        # Validate model before initialization
        if model not in ANTHROPIC_MODELS:
            raise InvalidModelError(model, "anthropic", list(ANTHROPIC_MODELS.keys()))
        
        settings = Settings()
        super().__init__(
            model=model,
            api_key=api_key or settings.anthropic_api_key,
            base_url=base_url or settings.anthropic_base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
    
    def _initialize_client(self) -> None:
        """Initialize the synchronous Anthropic client."""
        from anthropic import Anthropic
        
        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        
        self._client = Anthropic(**kwargs)
    
    def _initialize_async_client(self) -> None:
        """Initialize the async Anthropic client."""
        from anthropic import AsyncAnthropic
        
        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        
        self._async_client = AsyncAnthropic(**kwargs)
    
    def _build_request_params(
        self,
        messages: list[dict[str, str]],
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> dict[str, Any]:
        """Build the request parameters."""
        # Anthropic requires max_tokens
        max_tokens = 4096
        if config and config.max_tokens:
            max_tokens = config.max_tokens
        
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        
        if system_prompt:
            params["system"] = system_prompt
        
        if config:
            if config.temperature != 1.0:
                params["temperature"] = config.temperature
            if config.top_p != 1.0:
                params["top_p"] = config.top_p
            if config.stop_sequences:
                params["stop_sequences"] = config.stop_sequences
        
        return params
    
    def _parse_response(
        self,
        response: Any,
        latency_ms: float,
    ) -> LLMResponse:
        """Parse the API response into an LLMResponse."""
        content = ""
        if response.content:
            content = response.content[0].text
        
        return LLMResponse(
            content=content,
            model=response.model,
            provider=self.provider,
            usage=TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
            latency_ms=latency_ms,
            finish_reason=response.stop_reason,
            raw_response=response,
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """Generate a response for the given prompt."""
        messages = [{"role": "user", "content": prompt}]
        params = self._build_request_params(messages, system_prompt, config)
        
        start_time = time.perf_counter()
        try:
            response = self.client.messages.create(**params)
            latency_ms = (time.perf_counter() - start_time) * 1000
            return self._parse_response(response, latency_ms)
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return LLMResponse(
                content="",
                model=self.model,
                provider=self.provider,
                usage=TokenUsage(),
                latency_ms=latency_ms,
                error=str(e),
            )
    
    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """Asynchronously generate a response for the given prompt."""
        messages = [{"role": "user", "content": prompt}]
        params = self._build_request_params(messages, system_prompt, config)
        
        start_time = time.perf_counter()
        try:
            response = await self.async_client.messages.create(**params)
            latency_ms = (time.perf_counter() - start_time) * 1000
            return self._parse_response(response, latency_ms)
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return LLMResponse(
                content="",
                model=self.model,
                provider=self.provider,
                usage=TokenUsage(),
                latency_ms=latency_ms,
                error=str(e),
            )
    
    def generate_chat(
        self,
        messages: list[Message],
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """Generate a response for a chat conversation."""
        # Extract system message if present
        system_prompt = None
        chat_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            else:
                chat_messages.append(msg.to_dict())
        
        params = self._build_request_params(chat_messages, system_prompt, config)
        
        start_time = time.perf_counter()
        try:
            response = self.client.messages.create(**params)
            latency_ms = (time.perf_counter() - start_time) * 1000
            return self._parse_response(response, latency_ms)
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return LLMResponse(
                content="",
                model=self.model,
                provider=self.provider,
                usage=TokenUsage(),
                latency_ms=latency_ms,
                error=str(e),
            )
    
    async def generate_chat_async(
        self,
        messages: list[Message],
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """Asynchronously generate a response for a chat conversation."""
        # Extract system message if present
        system_prompt = None
        chat_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            else:
                chat_messages.append(msg.to_dict())
        
        params = self._build_request_params(chat_messages, system_prompt, config)
        
        start_time = time.perf_counter()
        try:
            response = await self.async_client.messages.create(**params)
            latency_ms = (time.perf_counter() - start_time) * 1000
            return self._parse_response(response, latency_ms)
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return LLMResponse(
                content="",
                model=self.model,
                provider=self.provider,
                usage=TokenUsage(),
                latency_ms=latency_ms,
                error=str(e),
            )
    
    def get_model_info(self) -> ModelInfo:
        """Get information about the current model."""
        if self.model in ANTHROPIC_MODELS:
            return ANTHROPIC_MODELS[self.model]
        
        # Return a generic model info for unknown models
        return ModelInfo(
            provider=self.provider,
            model_id=self.model,
            display_name=self.model,
            context_window=200000,
            max_output_tokens=4096,
        )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens (approximate for Anthropic)."""
        # Anthropic's tokenizer is similar to OpenAI's
        # This is a rough approximation
        try:
            return self.client.count_tokens(text)
        except Exception:
            return super().count_tokens(text)
