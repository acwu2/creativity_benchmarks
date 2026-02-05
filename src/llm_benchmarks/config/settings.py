"""Settings management using Pydantic."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI Settings
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_org_id: Optional[str] = Field(default=None, description="OpenAI Organization ID")
    openai_base_url: Optional[str] = Field(default=None, description="OpenAI API base URL")

    # Anthropic Settings
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    anthropic_base_url: Optional[str] = Field(default=None, description="Anthropic API base URL")

    # Google Settings
    google_api_key: Optional[str] = Field(default=None, description="Google API key")

    # Azure OpenAI Settings
    azure_openai_api_key: Optional[str] = Field(default=None, description="Azure OpenAI API key")
    azure_openai_endpoint: Optional[str] = Field(default=None, description="Azure OpenAI endpoint")
    azure_openai_api_version: str = Field(
        default="2024-02-01", description="Azure OpenAI API version"
    )

    # Benchmark Settings
    benchmark_output_dir: Path = Field(
        default=Path("./results"), description="Directory for benchmark outputs"
    )
    benchmark_max_concurrent: int = Field(
        default=5, description="Maximum concurrent API calls"
    )
    benchmark_default_timeout: int = Field(
        default=120, description="Default timeout for API calls in seconds"
    )
    benchmark_retry_attempts: int = Field(
        default=3, description="Number of retry attempts for failed API calls"
    )

    def ensure_output_dir(self) -> Path:
        """Ensure the output directory exists."""
        self.benchmark_output_dir.mkdir(parents=True, exist_ok=True)
        return self.benchmark_output_dir


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
