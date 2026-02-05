"""Tests for the Not Like That benchmark."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from llm_benchmarks.benchmarks.difference_and_negation.not_like_that import (
    NotLikeThatBenchmark,
    NotLikeThatMetrics,
    StoryWithCounterexample,
    EmbeddingClient,
    cosine_distance,
    euclidean_distance,
    DEFAULT_STORY_EXAMPLES,
)
from llm_benchmarks.clients.base import (
    BaseLLMClient,
    GenerationConfig,
    LLMResponse,
    Message,
    ModelInfo,
    TokenUsage,
)


class MockNotLikeThatClient(BaseLLMClient):
    """Mock client that returns predictable differentiated stories."""
    
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
        
        # Return a mock story that attempts to follow the good style
        response = f"""In a land of eternal sunshine, Princess Aurora discovered a 
        magical garden where butterflies spoke in riddles and flowers granted wishes. 
        She befriended a talking rabbit named Clover who guided her through the maze 
        of singing hedges. Together they found the Crystal of Joy and brought happiness 
        to the entire kingdom. Everyone lived happily ever after, and the garden 
        flourished for generations to come. Story {self.call_count} complete."""
        
        return LLMResponse(
            content=response,
            model=self.model,
            provider=self.provider,
            usage=TokenUsage(prompt_tokens=250, completion_tokens=120, total_tokens=370),
            latency_ms=450.0,
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
    """Mock embedding client for testing."""
    
    def __init__(self, **kwargs):
        self.embed_count = 0
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return mock embeddings based on text content."""
        embeddings = []
        for text in texts:
            self.embed_count += 1
            # Create deterministic embeddings based on content
            # Fairy tale style: high values in first dimensions
            # Noir style: high values in later dimensions
            if "princess" in text.lower() or "magical" in text.lower() or "fairy" in text.lower():
                # Fairy tale style
                embedding = [0.8, 0.7, 0.6, 0.1, 0.1]
            elif "detective" in text.lower() or "cigarette" in text.lower() or "rain" in text.lower():
                # Noir style
                embedding = [0.1, 0.1, 0.2, 0.8, 0.9]
            else:
                # Neutral/mixed
                embedding = [0.5, 0.5, 0.5, 0.5, 0.5]
            
            # Pad to standard size and normalize
            full_embedding = embedding + [0.0] * (1536 - len(embedding))
            embeddings.append(full_embedding)
        
        return embeddings
    
    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]


# --- Unit tests for distance functions ---

class TestDistanceFunctions:
    """Tests for distance calculation functions."""
    
    def test_cosine_distance_identical(self):
        """Identical vectors should have distance 0."""
        v = [1.0, 2.0, 3.0]
        assert cosine_distance(v, v) == pytest.approx(0.0)
    
    def test_cosine_distance_orthogonal(self):
        """Orthogonal vectors should have distance 1."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        assert cosine_distance(v1, v2) == pytest.approx(1.0)
    
    def test_cosine_distance_opposite(self):
        """Opposite vectors should have distance 2."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [-1.0, 0.0, 0.0]
        assert cosine_distance(v1, v2) == pytest.approx(2.0)
    
    def test_euclidean_distance_identical(self):
        """Identical vectors should have distance 0."""
        v = [1.0, 2.0, 3.0]
        assert euclidean_distance(v, v) == pytest.approx(0.0)
    
    def test_euclidean_distance_unit(self):
        """Unit distance vectors."""
        v1 = [0.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        assert euclidean_distance(v1, v2) == pytest.approx(1.0)


# --- Tests for StoryWithCounterexample ---

class TestStoryWithCounterexample:
    """Tests for the StoryWithCounterexample dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        example = StoryWithCounterexample(
            id="test",
            good_story="Good story content",
            bad_story="Bad story content",
            description="Test description",
            good_style="Good style",
            bad_style="Bad style",
        )
        d = example.to_dict()
        assert d["id"] == "test"
        assert d["good_story"] == "Good story content"
        assert d["bad_story"] == "Bad story content"
        assert d["description"] == "Test description"
        assert d["good_style"] == "Good style"
        assert d["bad_style"] == "Bad style"
    
    def test_default_values(self):
        """Test default values for optional fields."""
        example = StoryWithCounterexample(
            id="test",
            good_story="Good",
            bad_story="Bad",
        )
        assert example.description == ""
        assert example.good_style == ""
        assert example.bad_style == ""


# --- Tests for NotLikeThatMetrics ---

class TestNotLikeThatMetrics:
    """Tests for the NotLikeThatMetrics dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = NotLikeThatMetrics(
            mean_distance_to_good=0.3,
            mean_distance_to_bad=0.7,
            mean_differentiation_score=0.2,
            mean_avoidance_ratio=2.5,
            success_rate=0.83,
        )
        d = metrics.to_dict()
        assert d["mean_distance_to_good"] == 0.3
        assert d["mean_distance_to_bad"] == 0.7
        assert d["mean_differentiation_score"] == 0.2
        assert d["mean_avoidance_ratio"] == 2.5
        assert d["success_rate"] == 0.83
    
    def test_default_values(self):
        """Test default values."""
        metrics = NotLikeThatMetrics()
        assert metrics.mean_distance_to_good == 0.0
        assert metrics.mean_distance_to_bad == 0.0
        assert metrics.mean_differentiation_score == 0.0
        assert metrics.example_results == []


# --- Tests for NotLikeThatBenchmark ---

class TestNotLikeThatBenchmark:
    """Tests for the NotLikeThatBenchmark class."""
    
    def test_initialization_defaults(self):
        """Test default initialization."""
        benchmark = NotLikeThatBenchmark()
        assert benchmark.name == "not_like_that"
        assert benchmark.version == "1.0.0"
        assert len(benchmark.story_examples) == len(DEFAULT_STORY_EXAMPLES)
        assert benchmark.embedding_model == "text-embedding-3-small"
        assert benchmark.distance_metric == "cosine"
        assert benchmark.target_length == 200
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        custom_examples = [
            StoryWithCounterexample(
                id="custom",
                good_story="Good story",
                bad_story="Bad story",
            )
        ]
        benchmark = NotLikeThatBenchmark(
            story_examples=custom_examples,
            embedding_model="text-embedding-3-large",
            distance_metric="euclidean",
            target_length=300,
        )
        assert len(benchmark.story_examples) == 1
        assert benchmark.embedding_model == "text-embedding-3-large"
        assert benchmark.distance_metric == "euclidean"
        assert benchmark.target_length == 300
    
    def test_get_prompts(self):
        """Test prompt generation."""
        benchmark = NotLikeThatBenchmark()
        prompts = benchmark.get_prompts()
        
        assert len(prompts) == len(DEFAULT_STORY_EXAMPLES)
        
        for prompt_data in prompts:
            assert "id" in prompt_data
            assert "prompt" in prompt_data
            assert "example" in prompt_data
            assert "metadata" in prompt_data
            assert "Good Example" in prompt_data["prompt"]
            assert "Bad Example" in prompt_data["prompt"]
            assert "DO NOT write like this" in prompt_data["prompt"]
    
    def test_get_system_prompt(self):
        """Test system prompt content."""
        benchmark = NotLikeThatBenchmark()
        system_prompt = benchmark.get_system_prompt()
        
        assert system_prompt is not None
        assert "style" in system_prompt.lower()
        assert "avoid" in system_prompt.lower()
    
    def test_get_generation_config(self):
        """Test generation config."""
        benchmark = NotLikeThatBenchmark()
        config = benchmark.get_generation_config()
        
        assert config.max_tokens == 1024
        assert config.temperature == 0.8
        assert config.top_p == 0.95
    
    def test_differentiation_prompt_content(self):
        """Test that differentiation prompts contain expected elements."""
        example = StoryWithCounterexample(
            id="test",
            good_story="A happy fairy tale about friendship.",
            bad_story="A dark noir story about betrayal.",
        )
        benchmark = NotLikeThatBenchmark(story_examples=[example])
        prompts = benchmark.get_prompts()
        
        prompt_text = prompts[0]["prompt"]
        assert "A happy fairy tale about friendship" in prompt_text
        assert "A dark noir story about betrayal" in prompt_text
        assert "Good Example" in prompt_text
        assert "Bad Example" in prompt_text
    
    @patch.object(NotLikeThatBenchmark, 'embedding_client', new_callable=lambda: property(lambda self: MockEmbeddingClient()))
    def test_evaluate_response_success(self, mock_prop):
        """Test evaluation of a successful differentiation."""
        benchmark = NotLikeThatBenchmark()
        benchmark._embedding_client = MockEmbeddingClient()
        
        example = StoryWithCounterexample(
            id="test_fairy",
            good_story="Once upon a time, a magical princess lived in a castle.",
            bad_story="The detective lit a cigarette in the rain.",
        )
        
        prompt_data = {
            "id": "test_fairy",
            "example": example,
        }
        
        # A response that should be close to fairy tale style
        response = "In a magical kingdom, Princess Aurora danced with butterflies."
        
        mock_response = LLMResponse(
            content=response,
            model="test",
            provider="test",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            latency_ms=100.0,
        )
        
        scores = benchmark.evaluate_response(prompt_data, response, mock_response)
        
        assert "distance_to_good" in scores
        assert "distance_to_bad" in scores
        assert "differentiation_score" in scores
        assert "avoidance_ratio" in scores
        assert "is_successful" in scores
    
    def test_default_story_examples(self):
        """Test that default examples are well-formed."""
        for example in DEFAULT_STORY_EXAMPLES:
            assert example.id
            assert example.good_story
            assert example.bad_story
            assert len(example.good_story) > 50
            assert len(example.bad_story) > 50
    
    def test_default_examples_have_contrasting_styles(self):
        """Test that default examples have clear style contrast descriptions."""
        for example in DEFAULT_STORY_EXAMPLES:
            assert example.good_style or example.description
            assert example.bad_style or example.description


# --- Integration tests ---

class TestNotLikeThatIntegration:
    """Integration tests for the Not Like That benchmark."""
    
    @patch('llm_benchmarks.benchmarks.difference_and_negation.not_like_that.EmbeddingClient')
    def test_run_benchmark(self, MockEmbeddingClientClass):
        """Test running the full benchmark."""
        # Setup mock embedding client
        MockEmbeddingClientClass.return_value = MockEmbeddingClient()
        
        # Use a single custom example for faster testing
        custom_example = StoryWithCounterexample(
            id="test_pair",
            good_story="Once upon a time, a magical princess lived in a castle with sparkles.",
            bad_story="The detective lit his cigarette in the rain, watching the neon signs.",
        )
        
        benchmark = NotLikeThatBenchmark(story_examples=[custom_example])
        benchmark._embedding_client = MockEmbeddingClient()
        
        client = MockNotLikeThatClient()
        result = benchmark.run(client)
        
        assert result.benchmark_name == "not_like_that"
        assert result.benchmark_version == "1.0.0"
        assert len(result.prompt_results) == 1
        assert result.metrics.successful_prompts == 1
        
        # Check that metrics are calculated
        assert "differentiation_score" in result.metrics.mean_scores
        assert "success_rate" in result.metrics.mean_scores
    
    @patch('llm_benchmarks.benchmarks.difference_and_negation.not_like_that.EmbeddingClient')
    def test_get_detailed_results(self, MockEmbeddingClientClass):
        """Test getting detailed results after benchmark run."""
        MockEmbeddingClientClass.return_value = MockEmbeddingClient()
        
        custom_example = StoryWithCounterexample(
            id="test_pair",
            good_story="Once upon a time, a magical princess lived in a castle.",
            bad_story="The detective lit his cigarette in the rain.",
        )
        
        benchmark = NotLikeThatBenchmark(story_examples=[custom_example])
        benchmark._embedding_client = MockEmbeddingClient()
        
        client = MockNotLikeThatClient()
        benchmark.run(client)
        
        detailed = benchmark.get_detailed_results()
        
        assert isinstance(detailed, NotLikeThatMetrics)
        assert len(detailed.example_results) == 1
        assert "distance_to_good" in detailed.example_results[0]
        assert "distance_to_bad" in detailed.example_results[0]
        assert "differentiation_score" in detailed.example_results[0]
    
    @patch('llm_benchmarks.benchmarks.difference_and_negation.not_like_that.EmbeddingClient')
    def test_state_reset_between_runs(self, MockEmbeddingClientClass):
        """Test that state is properly reset between benchmark runs."""
        MockEmbeddingClientClass.return_value = MockEmbeddingClient()
        
        custom_example = StoryWithCounterexample(
            id="test_pair",
            good_story="A fairy tale story with magical elements.",
            bad_story="A noir detective story in the rain.",
        )
        
        benchmark = NotLikeThatBenchmark(story_examples=[custom_example])
        benchmark._embedding_client = MockEmbeddingClient()
        
        client = MockNotLikeThatClient()
        
        # First run
        result1 = benchmark.run(client)
        assert len(result1.prompt_results) == 1
        
        # Second run should reset state
        result2 = benchmark.run(client)
        assert len(result2.prompt_results) == 1
        
        # Detailed results should only contain latest run
        detailed = benchmark.get_detailed_results()
        assert len(detailed.example_results) == 1


class TestNotLikeThatEdgeCases:
    """Test edge cases for the Not Like That benchmark."""
    
    def test_empty_story_examples(self):
        """Test benchmark with empty story examples list."""
        benchmark = NotLikeThatBenchmark(story_examples=[])
        prompts = benchmark.get_prompts()
        assert len(prompts) == 0
    
    def test_get_detailed_results_before_run(self):
        """Test that getting results before run raises error."""
        benchmark = NotLikeThatBenchmark()
        with pytest.raises(RuntimeError, match="No results available"):
            benchmark.get_detailed_results()
    
    def test_distance_metric_validation(self):
        """Test that invalid distance metric raises error."""
        benchmark = NotLikeThatBenchmark(distance_metric="invalid")
        
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        
        with pytest.raises(ValueError, match="Unknown distance metric"):
            benchmark._get_distance(v1, v2)
