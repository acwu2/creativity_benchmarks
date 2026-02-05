"""Base LLM client interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Optional


class InvalidModelError(ValueError):
    """Raised when an invalid model is specified for a provider."""
    
    def __init__(self, model: str, provider: str, valid_models: list[str]):
        self.model = model
        self.provider = provider
        self.valid_models = valid_models
        valid_models_str = ", ".join(sorted(valid_models))
        message = (
            f"Invalid model '{model}' for provider '{provider}'. "
            f"Valid models are: {valid_models_str}"
        )
        super().__init__(message)


class MessageRole(str, Enum):
    """Role in a conversation message."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single message in a conversation."""
    role: MessageRole
    content: str
    
    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format."""
        return {"role": self.role.value, "content": self.content}


@dataclass
class ModelInfo:
    """Information about a model."""
    provider: str
    model_id: str
    display_name: str
    context_window: int
    max_output_tokens: Optional[int] = None
    input_cost_per_1k: Optional[float] = None  # USD per 1K tokens
    output_cost_per_1k: Optional[float] = None  # USD per 1K tokens
    supports_vision: bool = False
    supports_tools: bool = False


@dataclass
class TokenUsage:
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    @property
    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMResponse:
    """Response from an LLM API call."""
    content: str
    model: str
    provider: str
    usage: TokenUsage
    latency_ms: float
    created_at: datetime = field(default_factory=datetime.now)
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None
    error: Optional[str] = None
    
    @property
    def is_error(self) -> bool:
        """Check if this response represents an error."""
        return self.error is not None
    
    def calculate_cost(self, model_info: ModelInfo) -> Optional[float]:
        """Calculate the cost of this response in USD."""
        if model_info.input_cost_per_1k is None or model_info.output_cost_per_1k is None:
            return None
        
        input_cost = (self.usage.prompt_tokens / 1000) * model_info.input_cost_per_1k
        output_cost = (self.usage.completion_tokens / 1000) * model_info.output_cost_per_1k
        return input_cost + output_cost
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "usage": self.usage.as_dict,
            "latency_ms": self.latency_ms,
            "created_at": self.created_at.isoformat(),
            "finish_reason": self.finish_reason,
            "error": self.error,
        }


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    stop_sequences: Optional[list[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: Optional[int] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.temperature != 1.0:
            result["temperature"] = self.temperature
        if self.top_p != 1.0:
            result["top_p"] = self.top_p
        if self.stop_sequences:
            result["stop"] = self.stop_sequences
        if self.presence_penalty != 0.0:
            result["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty != 0.0:
            result["frequency_penalty"] = self.frequency_penalty
        if self.seed is not None:
            result["seed"] = self.seed
        return result


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    provider: str = "base"
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        """
        Initialize the LLM client.
        
        Args:
            model: The model identifier to use
            api_key: API key for authentication (can be loaded from env)
            base_url: Optional base URL for the API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Any = None
        self._async_client: Any = None
    
    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize the underlying API client."""
        pass
    
    @abstractmethod
    def _initialize_async_client(self) -> None:
        """Initialize the async API client."""
        pass
    
    @property
    def client(self) -> Any:
        """Get the synchronous client, initializing if needed."""
        if self._client is None:
            self._initialize_client()
        return self._client
    
    @property
    def async_client(self) -> Any:
        """Get the async client, initializing if needed."""
        if self._async_client is None:
            self._initialize_async_client()
        return self._async_client
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            config: Generation configuration
            
        Returns:
            LLMResponse containing the generated text and metadata
        """
        pass
    
    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """
        Asynchronously generate a response for the given prompt.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            config: Generation configuration
            
        Returns:
            LLMResponse containing the generated text and metadata
        """
        pass
    
    @abstractmethod
    def generate_chat(
        self,
        messages: list[Message],
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """
        Generate a response for a chat conversation.
        
        Args:
            messages: List of conversation messages
            config: Generation configuration
            
        Returns:
            LLMResponse containing the generated text and metadata
        """
        pass
    
    @abstractmethod
    async def generate_chat_async(
        self,
        messages: list[Message],
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """
        Asynchronously generate a response for a chat conversation.
        
        Args:
            messages: List of conversation messages
            config: Generation configuration
            
        Returns:
            LLMResponse containing the generated text and metadata
        """
        pass
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response (optional implementation).
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            config: Generation configuration
            
        Yields:
            String chunks of the response
        """
        raise NotImplementedError("Streaming not implemented for this client")
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Get information about the current model."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.
        
        This is a rough estimate; subclasses should override for accurate counts.
        """
        # Rough approximation: ~4 characters per token
        return len(text) // 4
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
