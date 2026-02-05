"""Tests for the base benchmark class."""

import pytest
from datetime import datetime
from typing import Any

from llm_benchmarks.benchmarks.base import (
    BaseBenchmark,
    BenchmarkResult,
    PromptResult,
    AggregatedMetrics,
)
from llm_benchmarks.clients.base import (
    BaseLLMClient,
    GenerationConfig,
    LLMResponse,
    Message,
    ModelInfo,
    TokenUsage,
)


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing."""
    
    provider = "mock"
    
    def __init__(self, responses: dict[str, str] | None = None):
        super().__init__(model="mock-model")
        self.responses = responses or {}
        self.call_count = 0
    
    def _initialize_client(self) -> None:
        pass
    
    def _initialize_async_client(self) -> None:
        pass
    
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        response = self.responses.get(prompt, "default response")
        return LLMResponse(
            content=response,
            model=self.model,
            provider=self.provider,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            latency_ms=100.0,
        )
    
    async def generate_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        return self.generate(prompt, system_prompt, config)
    
    def generate_chat(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        return self.generate(messages[-1].content if messages else "")
    
    async def generate_chat_async(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        return self.generate_chat(messages, config)
    
    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            provider=self.provider,
            model_id=self.model,
            display_name="Mock Model",
            context_window=4096,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
        )


class SimpleBenchmark(BaseBenchmark):
    """Simple benchmark for testing."""
    
    name = "test_benchmark"
    version = "1.0.0"
    description = "Test benchmark"
    
    def get_prompts(self) -> list[dict[str, Any]]:
        return [
            {"id": "1", "prompt": "What is 2+2?", "expected": "4"},
            {"id": "2", "prompt": "What is the capital of France?", "expected": "Paris"},
        ]
    
    def evaluate_response(
        self,
        prompt_data: dict[str, Any],
        response: str,
        llm_response: LLMResponse,
    ) -> dict[str, float]:
        expected = prompt_data["expected"].lower()
        is_correct = expected in response.lower()
        return {"accuracy": 1.0 if is_correct else 0.0}


class TestBaseBenchmark:
    """Tests for BaseBenchmark."""
    
    def test_benchmark_creation(self):
        """Test creating a benchmark instance."""
        benchmark = SimpleBenchmark()
        assert benchmark.name == "test_benchmark"
        assert benchmark.version == "1.0.0"
    
    def test_get_prompts(self):
        """Test getting prompts."""
        benchmark = SimpleBenchmark()
        prompts = benchmark.get_prompts()
        assert len(prompts) == 2
        assert prompts[0]["id"] == "1"
    
    def test_run_benchmark(self):
        """Test running a benchmark."""
        benchmark = SimpleBenchmark()
        client = MockLLMClient(responses={
            "What is 2+2?": "The answer is 4",
            "What is the capital of France?": "Paris",
        })
        
        result = benchmark.run(client)
        
        assert result.benchmark_name == "test_benchmark"
        assert result.model == "mock-model"
        assert result.provider == "mock"
        assert len(result.prompt_results) == 2
        assert result.metrics.total_prompts == 2
        assert result.metrics.successful_prompts == 2
    
    def test_benchmark_with_custom_config(self):
        """Test benchmark with custom generation config."""
        config = GenerationConfig(max_tokens=100, temperature=0.5)
        benchmark = SimpleBenchmark(generation_config=config)
        
        assert benchmark.get_generation_config().max_tokens == 100
        assert benchmark.get_generation_config().temperature == 0.5


class TestPromptResult:
    """Tests for PromptResult."""
    
    def test_prompt_result_creation(self):
        """Test creating a prompt result."""
        result = PromptResult(
            prompt_id="1",
            prompt="test prompt",
            response="test response",
            scores={"accuracy": 1.0},
        )
        
        assert result.prompt_id == "1"
        assert not result.is_error
    
    def test_prompt_result_with_error(self):
        """Test prompt result with error."""
        result = PromptResult(
            prompt_id="1",
            prompt="test prompt",
            response="",
            scores={},
            error="API error",
        )
        
        assert result.is_error
    
    def test_prompt_result_to_dict(self):
        """Test converting prompt result to dict."""
        result = PromptResult(
            prompt_id="1",
            prompt="test prompt",
            response="test response",
            scores={"accuracy": 1.0},
        )
        
        d = result.to_dict()
        assert d["prompt_id"] == "1"
        assert d["scores"]["accuracy"] == 1.0


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""
    
    def test_benchmark_result_save_load(self, tmp_path):
        """Test saving and loading benchmark results."""
        metrics = AggregatedMetrics(
            mean_scores={"accuracy": 0.8},
            std_scores={"accuracy": 0.1},
            min_scores={"accuracy": 0.6},
            max_scores={"accuracy": 1.0},
            total_prompts=10,
            successful_prompts=10,
            failed_prompts=0,
            total_tokens=150,
            total_latency_ms=1000.0,
            mean_latency_ms=100.0,
        )
        
        result = BenchmarkResult(
            benchmark_name="test",
            benchmark_version="1.0.0",
            model="test-model",
            provider="test",
            prompt_results=[],
            metrics=metrics,
            config={},
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )
        
        path = tmp_path / "result.json"
        result.save(path)
        
        loaded = BenchmarkResult.load(path)
        assert loaded.benchmark_name == "test"
        assert loaded.metrics.mean_scores["accuracy"] == 0.8
