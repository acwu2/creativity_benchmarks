"""Google Generative AI (Gemini) client implementation using the google.genai SDK."""

import time
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from llm_benchmarks.clients.base import (
    BaseLLMClient,
    GenerationConfig,
    LLMResponse,
    Message,
    MessageRole,
    ModelInfo,
    TokenUsage,
)
from llm_benchmarks.config import Settings


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
        settings = Settings()
        super().__init__(
            model=model,
            api_key=api_key or settings.google_api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
    
    def _initialize_client(self) -> None:
        """Initialize the Google genai client."""
        from google import genai
        
        self._client = genai.Client(api_key=self.api_key)
    
    def _initialize_async_client(self) -> None:
        """Initialize the async client (same client exposes .aio)."""
        if self._client is None:
            self._initialize_client()
        self._async_client = self._client
    
    def _build_generation_config(
        self,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> "google.genai.types.GenerateContentConfig":
        """Build the generation config for the Google genai API."""
        from google.genai import types
        
        kwargs: dict[str, Any] = {}
        
        if system_prompt:
            kwargs["system_instruction"] = system_prompt
        
        if config:
            if config.max_tokens is not None:
                kwargs["max_output_tokens"] = config.max_tokens
            if config.temperature != 1.0:
                kwargs["temperature"] = config.temperature
            if config.top_p != 1.0:
                kwargs["top_p"] = config.top_p
            if config.stop_sequences:
                kwargs["stop_sequences"] = config.stop_sequences
        
        return types.GenerateContentConfig(**kwargs)
    
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
        if hasattr(response, "usage_metadata") and response.usage_metadata:
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
    def _generate_with_retry(
        self,
        prompt: str,
        gen_config: "Any",
    ) -> "Any":
        """Make the API call with retry logic."""
        return self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=gen_config,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """Generate a response for the given prompt."""
        if self._client is None:
            self._initialize_client()
        
        gen_config = self._build_generation_config(config, system_prompt)
        
        start_time = time.perf_counter()
        try:
            response = self._generate_with_retry(prompt, gen_config)
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
        if self._client is None:
            self._initialize_client()
        
        gen_config = self._build_generation_config(config, system_prompt)
        
        start_time = time.perf_counter()
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=gen_config,
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
        if self._client is None:
            self._initialize_client()
        
        from google.genai import types
        
        # Extract system prompt and convert messages to history
        system_prompt = None
        history: list[types.Content] = []
        
        for msg in messages[:-1]:  # All but the last message
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            else:
                role = "user" if msg.role == MessageRole.USER else "model"
                history.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg.content)],
                    )
                )
        
        # Get the last user message
        last_message = messages[-1].content if messages else ""
        
        gen_config = self._build_generation_config(config, system_prompt)
        
        start_time = time.perf_counter()
        try:
            chat = self._client.chats.create(
                model=self.model,
                config=gen_config,
                history=history,
            )
            response = chat.send_message(last_message)
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
        if self._client is None:
            self._initialize_client()
        
        from google.genai import types
        
        # Extract system prompt and convert messages to history
        system_prompt = None
        history: list[types.Content] = []
        
        for msg in messages[:-1]:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            else:
                role = "user" if msg.role == MessageRole.USER else "model"
                history.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg.content)],
                    )
                )
        
        last_message = messages[-1].content if messages else ""
        
        gen_config = self._build_generation_config(config, system_prompt)
        
        start_time = time.perf_counter()
        try:
            chat = self._client.aio.chats.create(
                model=self.model,
                config=gen_config,
                history=history,
            )
            response = await chat.send_message(last_message)
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
    
    def validate_model(self) -> None:
        """Validate that the model exists.

        Queries the Google genai Models API for confirmation.

        Raises:
            ValueError: If the API reports that the model does not exist.
        """
        try:
            if self._client is None:
                self._initialize_client()
            self._client.models.get(model=self.model)
        except Exception as exc:
            raise ValueError(
                f"Google model '{self.model}' could not be verified via the API: {exc}"
            ) from exc

    def get_model_info(self) -> ModelInfo:
        """Get information about the current model.
        
        Queries the Google genai API for actual model metadata
        including token limits. Falls back to defaults on failure.
        """
        if self._model_info is not None:
            return self._model_info
        
        display_name = self.model
        context_window = 1000000
        max_output_tokens = 8192
        
        try:
            if self._client is None:
                self._initialize_client()
            model_meta = self._client.models.get(model=self.model)
            display_name = getattr(model_meta, "display_name", self.model)
            context_window = getattr(model_meta, "input_token_limit", context_window)
            max_output_tokens = getattr(model_meta, "output_token_limit", max_output_tokens)
        except Exception:
            pass
        
        self._model_info = ModelInfo(
            provider=self.provider,
            model_id=self.model,
            display_name=display_name,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
        )
        return self._model_info
