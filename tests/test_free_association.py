"""Tests for the Free Association benchmark."""

import pytest
from collections import Counter

from llm_benchmarks.benchmarks.iteration.free_association import (
    FreeAssociationBenchmark,
    FreeAssociationMetrics,
    extract_words,
    calculate_time_to_first_repetition,
    estimate_chao1,
)
from llm_benchmarks.clients.base import (
    BaseLLMClient,
    GenerationConfig,
    LLMResponse,
    Message,
    ModelInfo,
    TokenUsage,
)


class MockFreeAssociationClient(BaseLLMClient):
    """Mock client that returns predictable word chains."""
    
    provider = "mock"
    
    def __init__(self, unique_ratio: float = 0.8):
        """
        Args:
            unique_ratio: Approximate ratio of unique words to return
        """
        super().__init__(model="mock-model")
        self.unique_ratio = unique_ratio
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
        
        # Generate a mix of unique and repeated words
        base_words = [
            "apple", "banana", "cherry", "dog", "elephant",
            "forest", "garden", "house", "island", "jungle",
            "kite", "lemon", "mountain", "night", "ocean",
            "piano", "queen", "river", "sunset", "tree",
        ]
        
        # Add some variety based on call count
        words = base_words.copy()
        for i in range(10):
            words.append(f"word{self.call_count}_{i}")
        
        # Add some repetitions
        words.extend(["apple", "ocean", "tree", "dog", "forest"])
        
        response = ", ".join(words[:30])
        
        return LLMResponse(
            content=response,
            model=self.model,
            provider=self.provider,
            usage=TokenUsage(prompt_tokens=50, completion_tokens=100, total_tokens=150),
            latency_ms=200.0,
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


class TestExtractWords:
    """Tests for word extraction."""
    
    def test_comma_separated(self):
        """Test extracting comma-separated words."""
        text = "apple, banana, cherry, dog"
        words = extract_words(text)
        assert words == ["apple", "banana", "cherry", "dog"]
    
    def test_arrow_separated(self):
        """Test extracting arrow-separated words."""
        text = "apple -> banana -> cherry -> dog"
        words = extract_words(text)
        assert words == ["apple", "banana", "cherry", "dog"]
    
    def test_numbered_list(self):
        """Test extracting from numbered list."""
        text = """1. apple
2. banana
3. cherry"""
        words = extract_words(text)
        assert words == ["apple", "banana", "cherry"]
    
    def test_mixed_format(self):
        """Test extracting from mixed format."""
        text = "apple, banana -> cherry; dog | elephant"
        words = extract_words(text)
        assert len(words) == 5
        assert "apple" in words
        assert "elephant" in words
    
    def test_filters_single_chars(self):
        """Test that single characters are filtered out."""
        text = "a, apple, b, banana"
        words = extract_words(text)
        assert words == ["apple", "banana"]
    
    def test_normalizes_case(self):
        """Test that words are lowercased."""
        text = "Apple, BANANA, ChErRy"
        words = extract_words(text)
        assert words == ["apple", "banana", "cherry"]


class TestTimeToFirstRepetition:
    """Tests for TTFR calculation."""
    
    def test_no_repetition(self):
        """Test when no words repeat."""
        words = ["apple", "banana", "cherry", "dog"]
        ttfr = calculate_time_to_first_repetition(words)
        assert ttfr is None
    
    def test_immediate_repetition(self):
        """Test when first word repeats immediately."""
        words = ["apple", "apple", "banana"]
        ttfr = calculate_time_to_first_repetition(words)
        assert ttfr == 1
    
    def test_late_repetition(self):
        """Test when repetition occurs later."""
        words = ["apple", "banana", "cherry", "apple", "dog"]
        ttfr = calculate_time_to_first_repetition(words)
        assert ttfr == 3  # "apple" repeats at index 3
    
    def test_empty_list(self):
        """Test with empty list."""
        ttfr = calculate_time_to_first_repetition([])
        assert ttfr is None


class TestChao1Estimator:
    """Tests for vocabulary estimation."""
    
    def test_all_unique(self):
        """Test when all words are unique (singletons)."""
        counts = Counter(["a", "b", "c", "d", "e"])
        estimate, lower, upper = estimate_chao1(counts)
        # With all singletons, Chao1 predicts more unseen words
        assert estimate >= 5
    
    def test_all_repeated(self):
        """Test when all words appear multiple times."""
        counts = Counter({"a": 5, "b": 5, "c": 5})
        estimate, lower, upper = estimate_chao1(counts)
        # With no singletons, estimate should be close to observed
        assert estimate == 3
    
    def test_mixed(self):
        """Test with mix of frequencies."""
        counts = Counter({"a": 1, "b": 1, "c": 2, "d": 3, "e": 5})
        estimate, lower, upper = estimate_chao1(counts)
        # Should estimate more than observed
        assert estimate >= 5
    
    def test_empty(self):
        """Test with empty counter."""
        counts = Counter()
        estimate, lower, upper = estimate_chao1(counts)
        assert estimate == 0


class TestFreeAssociationBenchmark:
    """Tests for the benchmark class."""
    
    def test_benchmark_creation(self):
        """Test creating a benchmark instance."""
        benchmark = FreeAssociationBenchmark()
        assert benchmark.name == "free_association"
        assert len(benchmark.seed_words) > 0
    
    def test_get_prompts(self):
        """Test getting prompts."""
        benchmark = FreeAssociationBenchmark(
            seed_words=["apple", "banana"],
            num_unseeded_chains=2,
        )
        prompts = benchmark.get_prompts()
        
        # Should have 2 seeded + 2 unseeded = 4 prompts
        assert len(prompts) == 4
        
        # Check seeded prompts
        seeded = [p for p in prompts if p["type"] == "seeded"]
        assert len(seeded) == 2
        assert seeded[0]["seed_word"] == "apple"
    
    def test_run_benchmark(self):
        """Test running the benchmark."""
        benchmark = FreeAssociationBenchmark(
            seed_words=["test"],
            num_unseeded_chains=1,
        )
        client = MockFreeAssociationClient()
        
        result = benchmark.run(client)
        
        assert result.benchmark_name == "free_association"
        assert result.metrics.total_prompts == 2
        assert result.metrics.mean_scores["unique_words"] > 0
        assert result.metrics.mean_scores["total_words"] > 0
    
    def test_detailed_results(self):
        """Test getting detailed metrics."""
        benchmark = FreeAssociationBenchmark(
            seed_words=["test"],
            num_unseeded_chains=0,
        )
        client = MockFreeAssociationClient()
        
        benchmark.run(client)
        metrics = benchmark.get_detailed_results()
        
        assert isinstance(metrics, FreeAssociationMetrics)
        assert metrics.total_words > 0
        assert metrics.unique_words > 0
        assert metrics.estimated_total_vocabulary is not None
    
    def test_generation_config(self):
        """Test that generation config has high temperature."""
        benchmark = FreeAssociationBenchmark()
        config = benchmark.get_generation_config()
        
        assert config.temperature == 1.0
        assert config.max_tokens >= 512
    
    def test_state_reset_between_runs(self):
        """Test that state is reset between runs."""
        benchmark = FreeAssociationBenchmark(
            seed_words=["test"],
            num_unseeded_chains=0,
        )
        client = MockFreeAssociationClient()
        
        # First run
        result1 = benchmark.run(client)
        unique1 = result1.metrics.mean_scores["unique_words"]
        
        # Second run - should have same metrics, not accumulated
        result2 = benchmark.run(client)
        unique2 = result2.metrics.mean_scores["unique_words"]
        
        assert unique1 == unique2  # Not accumulated
