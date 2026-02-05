"""OpenAI API client implementation."""

import time
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from llm_benchmarks.clients.base import (
    BaseLLMClient,
    GenerationConfig,
    InvalidModelError,
    LLMResponse,
    Message,
    ModelInfo,
    TokenUsage,
)
from llm_benchmarks.config import Settings


# Model information for common OpenAI models
OPENAI_MODELS: dict[str, ModelInfo] = {
    "gpt-4o": ModelInfo(
        provider="openai",
        model_id="gpt-4o",
        display_name="GPT-4o",
        context_window=128000,
        max_output_tokens=16384,
        input_cost_per_1k=0.005,
        output_cost_per_1k=0.015,
        supports_vision=True,
        supports_tools=True,
    ),
    "gpt-4o-mini": ModelInfo(
        provider="openai",
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        context_window=128000,
        max_output_tokens=16384,
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
        supports_vision=True,
        supports_tools=True,
    ),
    "gpt-4-turbo": ModelInfo(
        provider="openai",
        model_id="gpt-4-turbo",
        display_name="GPT-4 Turbo",
        context_window=128000,
        max_output_tokens=4096,
        input_cost_per_1k=0.01,
        output_cost_per_1k=0.03,
        supports_vision=True,
        supports_tools=True,
    ),
    "gpt-4": ModelInfo(
        provider="openai",
        model_id="gpt-4",
        display_name="GPT-4",
        context_window=8192,
        max_output_tokens=4096,
        input_cost_per_1k=0.03,
        output_cost_per_1k=0.06,
        supports_vision=False,
        supports_tools=True,
    ),
    "gpt-3.5-turbo": ModelInfo(
        provider="openai",
        model_id="gpt-3.5-turbo",
        display_name="GPT-3.5 Turbo",
        context_window=16385,
        max_output_tokens=4096,
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.0015,
        supports_vision=False,
        supports_tools=True,
    ),
    "o1": ModelInfo(
        provider="openai",
        model_id="o1",
        display_name="o1",
        context_window=200000,
        max_output_tokens=100000,
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.06,
        supports_vision=True,
        supports_tools=False,
    ),
    "o1-mini": ModelInfo(
        provider="openai",
        model_id="o1-mini",
        display_name="o1-mini",
        context_window=128000,
        max_output_tokens=65536,
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.012,
        supports_vision=False,
        supports_tools=False,
    ),
    "gpt-5": ModelInfo(
        provider="openai",
        model_id="gpt-5",
        display_name="GPT-5",
        context_window=256000,
        max_output_tokens=32768,
        input_cost_per_1k=0.01,
        output_cost_per_1k=0.03,
        supports_vision=True,
        supports_tools=True,
    ),
}


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""
    
    provider = "openai"
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        """
        Initialize the OpenAI client.
        
        Args:
            model: Model identifier (e.g., "gpt-4o", "gpt-4-turbo")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Optional base URL override
            organization: Optional organization ID
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        # Validate model before initialization
        if model not in OPENAI_MODELS:
            raise InvalidModelError(model, "openai", list(OPENAI_MODELS.keys()))
        
        settings = Settings()
        super().__init__(
            model=model,
            api_key=api_key or settings.openai_api_key,
            base_url=base_url or settings.openai_base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.organization = organization or settings.openai_org_id
    
    def _initialize_client(self) -> None:
        """Initialize the synchronous OpenAI client."""
        from openai import OpenAI
        
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
    
    def _initialize_async_client(self) -> None:
        """Initialize the async OpenAI client."""
        from openai import AsyncOpenAI
        
        self._async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
    
    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Build the messages list for the API call."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages
    
    def _uses_max_completion_tokens(self) -> bool:
        """Check if the model uses max_completion_tokens instead of max_tokens.
        
        Newer OpenAI models (o1, o1-mini, gpt-5+) require max_completion_tokens.
        """
        model_lower = self.model.lower()
        # o1 and o1-mini models
        if model_lower.startswith("o1"):
            return True
        # GPT-5.x and newer models
        if model_lower.startswith("gpt-5"):
            return True
        # Future-proof: any model version >= 5
        if model_lower.startswith("gpt-"):
            try:
                version = model_lower.split("-")[1]
                major_version = float(version.split(".")[0])
                if major_version >= 5:
                    return True
            except (IndexError, ValueError):
                pass
        return False

    def _supports_top_p(self) -> bool:
        """Check if the model supports top_p parameter.
        
        Some newer models (o1, o1-mini, gpt-5+) do not support top_p.
        """
        model_lower = self.model.lower()
        # o1 and o1-mini models don't support top_p
        if model_lower.startswith("o1"):
            return False
        # GPT-5.x and newer models don't support top_p
        if model_lower.startswith("gpt-5"):
            return False
        # Future-proof: any model version >= 5
        if model_lower.startswith("gpt-"):
            try:
                version = model_lower.split("-")[1]
                major_version = float(version.split(".")[0])
                if major_version >= 5:
                    return False
            except (IndexError, ValueError):
                pass
        return True

    def _build_request_params(
        self,
        messages: list[dict[str, str]],
        config: Optional[GenerationConfig] = None,
    ) -> dict[str, Any]:
        """Build the request parameters."""
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        
        if config:
            if config.max_tokens is not None:
                # Use max_completion_tokens for newer models
                if self._uses_max_completion_tokens():
                    params["max_completion_tokens"] = config.max_tokens
                else:
                    params["max_tokens"] = config.max_tokens
            if config.temperature != 1.0:
                params["temperature"] = config.temperature
            if config.top_p != 1.0 and self._supports_top_p():
                params["top_p"] = config.top_p
            if config.stop_sequences:
                params["stop"] = config.stop_sequences
            if config.presence_penalty != 0.0:
                params["presence_penalty"] = config.presence_penalty
            if config.frequency_penalty != 0.0:
                params["frequency_penalty"] = config.frequency_penalty
            if config.seed is not None:
                params["seed"] = config.seed
        
        return params
    
    def _parse_response(
        self,
        response: Any,
        latency_ms: float,
    ) -> LLMResponse:
        """Parse the API response into an LLMResponse."""
        choice = response.choices[0]
        usage = response.usage
        
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            provider=self.provider,
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            ),
            latency_ms=latency_ms,
            finish_reason=choice.finish_reason,
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
        messages = self._build_messages(prompt, system_prompt)
        params = self._build_request_params(messages, config)
        
        start_time = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**params)
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
        messages = self._build_messages(prompt, system_prompt)
        params = self._build_request_params(messages, config)
        
        start_time = time.perf_counter()
        try:
            response = await self.async_client.chat.completions.create(**params)
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
        msg_dicts = [m.to_dict() for m in messages]
        params = self._build_request_params(msg_dicts, config)
        
        start_time = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**params)
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
        msg_dicts = [m.to_dict() for m in messages]
        params = self._build_request_params(msg_dicts, config)
        
        start_time = time.perf_counter()
        try:
            response = await self.async_client.chat.completions.create(**params)
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
        if self.model in OPENAI_MODELS:
            return OPENAI_MODELS[self.model]
        
        # Return a generic model info for unknown models
        return ModelInfo(
            provider=self.provider,
            model_id=self.model,
            display_name=self.model,
            context_window=128000,  # Assume reasonable defaults
            max_output_tokens=4096,
        )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        try:
            import tiktoken
            
            try:
                encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            
            return len(encoding.encode(text))
        except ImportError:
            return super().count_tokens(text)
