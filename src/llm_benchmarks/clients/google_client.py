"""Google Generative AI (Gemini) client implementation."""

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


# Model information for common Google models
GOOGLE_MODELS: dict[str, ModelInfo] = {
    "gemini-2.0-flash": ModelInfo(
        provider="google",
        model_id="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        context_window=1000000,
        max_output_tokens=8192,
        input_cost_per_1k=0.0001,
        output_cost_per_1k=0.0004,
        supports_vision=True,
        supports_tools=True,
    ),
    "gemini-1.5-pro": ModelInfo(
        provider="google",
        model_id="gemini-1.5-pro",
        display_name="Gemini 1.5 Pro",
        context_window=2000000,
        max_output_tokens=8192,
        input_cost_per_1k=0.00125,
        output_cost_per_1k=0.005,
        supports_vision=True,
        supports_tools=True,
    ),
    "gemini-1.5-flash": ModelInfo(
        provider="google",
        model_id="gemini-1.5-flash",
        display_name="Gemini 1.5 Flash",
        context_window=1000000,
        max_output_tokens=8192,
        input_cost_per_1k=0.000075,
        output_cost_per_1k=0.0003,
        supports_vision=True,
        supports_tools=True,
    ),
    "gemini-1.0-pro": ModelInfo(
        provider="google",
        model_id="gemini-1.0-pro",
        display_name="Gemini 1.0 Pro",
        context_window=32760,
        max_output_tokens=8192,
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.0015,
        supports_vision=False,
        supports_tools=True,
    ),
}


class GoogleClient(BaseLLMClient):
    """Google Generative AI (Gemini) client."""
    
    provider = "google"
    
    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        """
        Initialize the Google client.
        
        Args:
            model: Model identifier (e.g., "gemini-1.5-pro")
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        # Validate model before initialization
        if model not in GOOGLE_MODELS:
            raise InvalidModelError(model, "google", list(GOOGLE_MODELS.keys()))
        
        settings = Settings()
        super().__init__(
            model=model,
            api_key=api_key or settings.google_api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._model_instance: Any = None
    
    def _initialize_client(self) -> None:
        """Initialize the Google Generative AI client."""
        import google.generativeai as genai
        
        genai.configure(api_key=self.api_key)
        self._client = genai
        self._model_instance = genai.GenerativeModel(self.model)
    
    def _initialize_async_client(self) -> None:
        """Initialize the async client (same as sync for Google)."""
        self._initialize_client()
        self._async_client = self._client
    
    def _build_generation_config(
        self,
        config: Optional[GenerationConfig] = None,
    ) -> dict[str, Any]:
        """Build the generation config for Google API."""
        import google.generativeai as genai
        
        gen_config: dict[str, Any] = {}
        
        if config:
            if config.max_tokens is not None:
                gen_config["max_output_tokens"] = config.max_tokens
            if config.temperature != 1.0:
                gen_config["temperature"] = config.temperature
            if config.top_p != 1.0:
                gen_config["top_p"] = config.top_p
            if config.stop_sequences:
                gen_config["stop_sequences"] = config.stop_sequences
        
        return gen_config
    
    def _parse_response(
        self,
        response: Any,
        latency_ms: float,
    ) -> LLMResponse:
        """Parse the API response into an LLMResponse."""
        content = ""
        if response.text:
            content = response.text
        
        # Extract usage metadata if available
        usage = TokenUsage()
        if hasattr(response, "usage_metadata"):
            metadata = response.usage_metadata
            usage = TokenUsage(
                prompt_tokens=getattr(metadata, "prompt_token_count", 0),
                completion_tokens=getattr(metadata, "candidates_token_count", 0),
                total_tokens=getattr(metadata, "total_token_count", 0),
            )
        
        finish_reason = None
        if response.candidates:
            finish_reason = str(response.candidates[0].finish_reason)
        
        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider,
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
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
        # Ensure client is initialized
        if self._model_instance is None:
            self._initialize_client()
        
        gen_config = self._build_generation_config(config)
        
        # Build the full prompt with system instruction
        model = self._model_instance
        if system_prompt:
            import google.generativeai as genai
            model = genai.GenerativeModel(
                self.model,
                system_instruction=system_prompt,
            )
        
        start_time = time.perf_counter()
        try:
            response = model.generate_content(
                prompt,
                generation_config=gen_config if gen_config else None,
            )
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
        # Ensure client is initialized
        if self._model_instance is None:
            self._initialize_client()
        
        gen_config = self._build_generation_config(config)
        
        # Build the full prompt with system instruction
        model = self._model_instance
        if system_prompt:
            import google.generativeai as genai
            model = genai.GenerativeModel(
                self.model,
                system_instruction=system_prompt,
            )
        
        start_time = time.perf_counter()
        try:
            response = await model.generate_content_async(
                prompt,
                generation_config=gen_config if gen_config else None,
            )
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
        # Ensure client is initialized
        if self._model_instance is None:
            self._initialize_client()
        
        gen_config = self._build_generation_config(config)
        
        # Extract system prompt and convert messages
        system_prompt = None
        history = []
        
        for msg in messages[:-1]:  # All but the last message
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            else:
                role = "user" if msg.role == MessageRole.USER else "model"
                history.append({"role": role, "parts": [msg.content]})
        
        # Get the last user message
        last_message = messages[-1].content if messages else ""
        
        # Create model with system instruction if present
        model = self._model_instance
        if system_prompt:
            import google.generativeai as genai
            model = genai.GenerativeModel(
                self.model,
                system_instruction=system_prompt,
            )
        
        start_time = time.perf_counter()
        try:
            chat = model.start_chat(history=history)
            response = chat.send_message(
                last_message,
                generation_config=gen_config if gen_config else None,
            )
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
        # Ensure client is initialized
        if self._model_instance is None:
            self._initialize_client()
        
        gen_config = self._build_generation_config(config)
        
        # Extract system prompt and convert messages
        system_prompt = None
        history = []
        
        for msg in messages[:-1]:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            else:
                role = "user" if msg.role == MessageRole.USER else "model"
                history.append({"role": role, "parts": [msg.content]})
        
        last_message = messages[-1].content if messages else ""
        
        model = self._model_instance
        if system_prompt:
            import google.generativeai as genai
            model = genai.GenerativeModel(
                self.model,
                system_instruction=system_prompt,
            )
        
        start_time = time.perf_counter()
        try:
            chat = model.start_chat(history=history)
            response = await chat.send_message_async(
                last_message,
                generation_config=gen_config if gen_config else None,
            )
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
        if self.model in GOOGLE_MODELS:
            return GOOGLE_MODELS[self.model]
        
        return ModelInfo(
            provider=self.provider,
            model_id=self.model,
            display_name=self.model,
            context_window=32000,
            max_output_tokens=8192,
        )
