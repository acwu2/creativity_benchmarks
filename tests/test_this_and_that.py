"""Tests for the This & That benchmark."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from llm_benchmarks.benchmarks.style_flexibility.this_and_that import (
    ThisAndThatBenchmark,
    ThisAndThatMetrics,
    StoryPair,
    EmbeddingClient,
    cosine_distance,
    euclidean_distance,
    DEFAULT_STORY_PAIRS,
)
from llm_benchmarks.clients.base import (
    BaseLLMClient,
    GenerationConfig,
    LLMResponse,
    Message,
    ModelInfo,
    TokenUsage,
)


class MockThisAndThatClient(BaseLLMClient):
    """Mock client that returns predictable interpolated stories."""
    
    provider = "mock"
    
    def __init__(self):
        super().__init__(model="mock-model")
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
        
        # Return a mock interpolated story
        response = f"""The detective walked through the enchanted forest, his trench coat 
        catching on thorns that glowed with an inner light. Magic and murder, he thought 
        grimly, made strange bedfellows. The fairy queen had been found dead at dawn, 
        and somewhere in this realm of wonders, a killer with very mortal motives was 
        hiding. He lit a cigarette and watched the smoke curl up to join the dancing 
        fireflies. "Even magic," he muttered, "can't hide the truth forever." This was 
        story number {self.call_count} in his long career of impossible cases."""
        
        return LLMResponse(
            content=response,
            model=self.model,
            provider=self.provider,
            usage=TokenUsage(prompt_tokens=200, completion_tokens=150, total_tokens=350),
            latency_ms=500.0,
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
            context_window=128000,
            max_output_tokens=4096,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
        )


class MockEmbeddingClient:
    """Mock embedding client that returns predictable embeddings."""
    
    def __init__(self):
        self.call_count = 0
        self._embeddings = {}
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_single(text))
        return embeddings
    
    def embed_single(self, text: str) -> list[float]:
        self.call_count += 1
        
        # Generate deterministic embeddings based on text content
        # Use a simple hash-based approach for reproducibility
        np.random.seed(hash(text[:100]) % (2**32))
        embedding = np.random.randn(256).tolist()
        
        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        embedding = [x / norm for x in embedding]
        
        return embedding


# ============================================================================
# Unit Tests for Distance Functions
# ============================================================================

class TestDistanceFunctions:
    """Tests for distance calculation functions."""
    
    def test_cosine_distance_identical(self):
        """Identical vectors should have distance 0."""
        v = [1.0, 0.0, 0.0]
        assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-10)
    
    def test_cosine_distance_orthogonal(self):
        """Orthogonal vectors should have distance 1."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        assert cosine_distance(v1, v2) == pytest.approx(1.0, abs=1e-10)
    
    def test_cosine_distance_opposite(self):
        """Opposite vectors should have distance 2."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [-1.0, 0.0, 0.0]
        assert cosine_distance(v1, v2) == pytest.approx(2.0, abs=1e-10)
    
    def test_cosine_distance_symmetric(self):
        """Cosine distance should be symmetric."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 5.0, 6.0]
        assert cosine_distance(v1, v2) == pytest.approx(cosine_distance(v2, v1))
    
    def test_euclidean_distance_identical(self):
        """Identical vectors should have distance 0."""
        v = [1.0, 2.0, 3.0]
        assert euclidean_distance(v, v) == pytest.approx(0.0, abs=1e-10)
    
    def test_euclidean_distance_known_value(self):
        """Test with known distance value."""
        v1 = [0.0, 0.0, 0.0]
        v2 = [3.0, 4.0, 0.0]
        assert euclidean_distance(v1, v2) == pytest.approx(5.0)


# ============================================================================
# Unit Tests for StoryPair
# ============================================================================

class TestStoryPair:
    """Tests for StoryPair dataclass."""
    
    def test_story_pair_creation(self):
        """Test basic StoryPair creation."""
        pair = StoryPair(
            id="test",
            story_a="Story A content",
            story_b="Story B content",
            description="Test pair",
        )
        assert pair.id == "test"
        assert pair.story_a == "Story A content"
        assert pair.story_b == "Story B content"
    
    def test_story_pair_to_dict(self):
        """Test StoryPair serialization."""
        pair = StoryPair(
            id="test",
            story_a="A",
            story_b="B",
            style_a="style A",
            style_b="style B",
        )
        d = pair.to_dict()
        assert d["id"] == "test"
        assert d["style_a"] == "style A"
    
    def test_default_story_pairs_exist(self):
        """Verify default story pairs are defined."""
        assert len(DEFAULT_STORY_PAIRS) > 0
        for pair in DEFAULT_STORY_PAIRS:
            assert pair.id
            assert pair.story_a
            assert pair.story_b


# ============================================================================
# Unit Tests for ThisAndThatBenchmark
# ============================================================================

class TestThisAndThatBenchmark:
    """Tests for ThisAndThatBenchmark."""
    
    def test_benchmark_initialization(self):
        """Test benchmark initializes with defaults."""
        benchmark = ThisAndThatBenchmark()
        assert benchmark.name == "this_and_that"
        assert len(benchmark.story_pairs) > 0
        assert benchmark.distance_metric == "cosine"
    
    def test_benchmark_custom_pairs(self):
        """Test benchmark with custom story pairs."""
        custom_pairs = [
            StoryPair(id="custom1", story_a="A1", story_b="B1"),
            StoryPair(id="custom2", story_a="A2", story_b="B2"),
        ]
        benchmark = ThisAndThatBenchmark(story_pairs=custom_pairs)
        assert len(benchmark.story_pairs) == 2
        assert benchmark.story_pairs[0].id == "custom1"
    
    def test_get_prompts(self):
        """Test prompt generation."""
        pairs = [StoryPair(id="test", story_a="Story A", story_b="Story B")]
        benchmark = ThisAndThatBenchmark(story_pairs=pairs)
        prompts = benchmark.get_prompts()
        
        assert len(prompts) == 1
        assert prompts[0]["id"] == "test"
        assert "Story A" in prompts[0]["prompt"]
        assert "Story B" in prompts[0]["prompt"]
    
    def test_get_system_prompt(self):
        """Test system prompt is provided."""
        benchmark = ThisAndThatBenchmark()
        system_prompt = benchmark.get_system_prompt()
        assert system_prompt is not None
        assert "style" in system_prompt.lower()
    
    def test_get_generation_config(self):
        """Test generation config has appropriate settings."""
        benchmark = ThisAndThatBenchmark()
        config = benchmark.get_generation_config()
        assert config.temperature > 0.5  # Should be creative
        assert config.max_tokens >= 512  # Enough for a short story


# ============================================================================
# Integration Tests (with mocking)
# ============================================================================

class TestThisAndThatIntegration:
    """Integration tests for ThisAndThatBenchmark with mocked clients."""
    
    def test_evaluate_response(self):
        """Test response evaluation with mock embedding client."""
        pairs = [StoryPair(id="test", story_a="Story A", story_b="Story B")]
        benchmark = ThisAndThatBenchmark(story_pairs=pairs)
        
        # Replace embedding client with mock
        benchmark._embedding_client = MockEmbeddingClient()
        
        prompt_data = {
            "id": "test",
            "prompt": "test prompt",
            "pair": pairs[0],
            "metadata": {},
        }
        
        mock_response = LLMResponse(
            content="A blended story combining both styles.",
            model="test-model",
            provider="test",
            usage=TokenUsage(),
            latency_ms=100,
        )
        
        scores = benchmark.evaluate_response(prompt_data, mock_response.content, mock_response)
        
        assert "distance_to_a" in scores
        assert "distance_to_b" in scores
        assert "summed_distance" in scores
        assert "balance_ratio" in scores
        assert scores["summed_distance"] == pytest.approx(
            scores["distance_to_a"] + scores["distance_to_b"]
        )
    
    def test_full_run_with_mocks(self):
        """Test full benchmark run with mocked clients."""
        pairs = [
            StoryPair(id="test1", story_a="Story A1", story_b="Story B1"),
            StoryPair(id="test2", story_a="Story A2", story_b="Story B2"),
        ]
        benchmark = ThisAndThatBenchmark(story_pairs=pairs)
        benchmark._embedding_client = MockEmbeddingClient()
        
        client = MockThisAndThatClient()
        result = benchmark.run(client)
        
        assert result.benchmark_name == "this_and_that"
        assert result.metrics.successful_prompts == 2
        assert "interpolation_score" in result.metrics.mean_scores
    
    def test_detailed_results(self):
        """Test detailed results after run."""
        pairs = [StoryPair(id="test", story_a="Story A", story_b="Story B")]
        benchmark = ThisAndThatBenchmark(story_pairs=pairs)
        benchmark._embedding_client = MockEmbeddingClient()
        
        client = MockThisAndThatClient()
        benchmark.run(client)
        
        metrics = benchmark.get_detailed_results()
        assert isinstance(metrics, ThisAndThatMetrics)
        assert len(metrics.pair_results) == 1
        assert metrics.mean_summed_distance >= 0


# ============================================================================
# Tests for ThisAndThatMetrics
# ============================================================================

class TestThisAndThatMetrics:
    """Tests for ThisAndThatMetrics dataclass."""
    
    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = ThisAndThatMetrics(
            mean_distance_to_a=0.3,
            mean_distance_to_b=0.4,
            mean_summed_distance=0.7,
            mean_interpolation_score=0.6,
            mean_balance_ratio=0.1,
        )
        d = metrics.to_dict()
        assert d["mean_summed_distance"] == 0.7
        assert d["mean_interpolation_score"] == 0.6
