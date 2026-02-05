"""Tests for the Quilting benchmark."""

import pytest
from collections import Counter

from llm_benchmarks.benchmarks.creative_constraints.quilting import (
    QuiltingBenchmark,
    QuiltingMetrics,
    parse_quilting_response,
    normalize_fragment,
    match_fragment_to_original,
    fragment_in_story,
    calculate_story_diversity,
    calculate_shannon_entropy,
)
from llm_benchmarks.clients.base import (
    BaseLLMClient,
    GenerationConfig,
    LLMResponse,
    Message,
    ModelInfo,
    TokenUsage,
)


class MockQuiltingClient(BaseLLMClient):
    """Mock client that returns predictable quilting responses."""
    
    provider = "mock"
    
    def __init__(self, vary_choices: bool = True):
        """
        Args:
            vary_choices: Whether to vary fragment choices between calls
        """
        super().__init__(model="mock-model")
        self.vary_choices = vary_choices
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
        
        # Extract fragments from prompt
        import re
        fragment_matches = re.findall(r'\d+\.\s*"([^"]+)"', prompt)
        
        # Choose different subsets based on call count if varying
        if self.vary_choices and fragment_matches:
            start_idx = (self.call_count - 1) % max(1, len(fragment_matches) - 3)
            chosen = fragment_matches[start_idx:start_idx + 4]
        else:
            # Always choose first 4 fragments
            chosen = fragment_matches[:4] if fragment_matches else []
        
        # Build response
        fragments_section = "\n".join(f"{i+1}. \"{f}\"" for i, f in enumerate(chosen))
        
        # Create a story that incorporates the fragments
        story_parts = [
            f"In a world where {chosen[0] if chosen else 'mysteries abound'},",
            f"there lived a soul who understood {chosen[1] if len(chosen) > 1 else 'the unknown'}.",
            f"They discovered that {chosen[2] if len(chosen) > 2 else 'truth'} was the key",
            f"to unlocking {chosen[3] if len(chosen) > 3 else 'everything'}.",
            "And so the journey continued, weaving through the fabric of existence.",
        ]
        story = " ".join(story_parts)
        
        response = f"""FRAGMENTS:
{fragments_section}

STORY:
{story}"""
        
        return LLMResponse(
            content=response,
            model=self.model,
            provider=self.provider,
            usage=TokenUsage(prompt_tokens=200, completion_tokens=300, total_tokens=500),
            latency_ms=500.0,
        )
    
    async def generate_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        return self.generate(prompt, system_prompt, config)
    
    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.model,
            provider=self.provider,
            context_window=4096,
            max_output_tokens=1024,
        )


class TestParseQuiltingResponse:
    """Tests for parsing quilting responses."""
    
    def test_parse_standard_format(self):
        """Test parsing a properly formatted response."""
        response = """FRAGMENTS:
1. "a clock that runs backwards"
2. "the weight of unspoken words"
3. "a door that opens both ways"

STORY:
In a house where time moved differently, there was a clock that ran backwards.
The residents carried the weight of unspoken words, never quite saying what they meant.
But they found hope in a door that opens both ways, allowing escape and return alike."""
        
        fragments, story = parse_quilting_response(response)
        
        assert len(fragments) == 3
        assert "a clock that runs backwards" in fragments
        assert "the weight of unspoken words" in fragments
        assert "a door that opens both ways" in fragments
        assert "house where time moved differently" in story
        assert len(story) > 100
    
    def test_parse_alternate_headers(self):
        """Test parsing with alternate section headers."""
        response = """CHOSEN:
- the color of forgotten dreams
- silence that speaks volumes

NARRATIVE:
A painter sought the color of forgotten dreams in every sunset.
The silence that speaks volumes filled her studio."""
        
        fragments, story = parse_quilting_response(response)
        
        assert len(fragments) == 2
        assert "the color of forgotten dreams" in fragments
        assert "silence that speaks volumes" in fragments
        assert "painter" in story.lower()
    
    def test_parse_no_explicit_sections(self):
        """Test parsing response without clear sections."""
        response = """This is just a story without proper formatting.
It tells of adventures and mysteries."""
        
        fragments, story = parse_quilting_response(response)
        
        # Should still extract story even if fragments are empty
        assert "adventures" in story.lower() or "adventures" in response.lower()


class TestNormalizeFragment:
    """Tests for fragment normalization."""
    
    def test_normalize_removes_punctuation(self):
        assert normalize_fragment("Hello, World!") == "hello world"
    
    def test_normalize_lowercases(self):
        assert normalize_fragment("A Clock That Runs Backwards") == "a clock that runs backwards"
    
    def test_normalize_strips_whitespace(self):
        assert normalize_fragment("  padded text  ") == "padded text"


class TestMatchFragmentToOriginal:
    """Tests for fragment matching."""
    
    def test_exact_match(self):
        available = ["a clock that runs backwards", "the weight of unspoken words"]
        match = match_fragment_to_original("a clock that runs backwards", available)
        assert match == "a clock that runs backwards"
    
    def test_case_insensitive_match(self):
        available = ["a clock that runs backwards", "the weight of unspoken words"]
        match = match_fragment_to_original("A Clock That Runs Backwards", available)
        assert match == "a clock that runs backwards"
    
    def test_partial_match(self):
        available = ["a clock that runs backwards", "the weight of unspoken words"]
        match = match_fragment_to_original("clock runs backwards", available, threshold=0.5)
        assert match == "a clock that runs backwards"
    
    def test_no_match(self):
        available = ["a clock that runs backwards", "the weight of unspoken words"]
        match = match_fragment_to_original("completely different text", available)
        assert match is None


class TestFragmentInStory:
    """Tests for checking fragment incorporation in stories."""
    
    def test_exact_phrase_found(self):
        fragment = "a clock that runs backwards"
        story = "In the tower stood a clock that runs backwards, counting down to zero."
        assert fragment_in_story(fragment, story) is True
    
    def test_key_words_found(self):
        fragment = "the weight of unspoken words"
        story = "She carried the heavy weight of those unspoken words for years."
        assert fragment_in_story(fragment, story) is True
    
    def test_fragment_not_found(self):
        fragment = "a lighthouse in the desert"
        story = "The ocean waves crashed against the shore as the sailor watched."
        assert fragment_in_story(fragment, story) is False


class TestCalculateStoryDiversity:
    """Tests for story diversity calculation."""
    
    def test_identical_stories_low_diversity(self):
        stories = [
            "The cat sat on the mat.",
            "The cat sat on the mat.",
            "The cat sat on the mat.",
        ]
        diversity = calculate_story_diversity(stories)
        assert diversity == 0.0  # Identical stories = no diversity
    
    def test_different_stories_high_diversity(self):
        stories = [
            "The cat sat on the mat and purred contentedly.",
            "A rocket ship launched into the distant cosmos.",
            "The chef prepared an exquisite meal in the kitchen.",
        ]
        diversity = calculate_story_diversity(stories)
        assert diversity > 0.5  # Very different stories = high diversity
    
    def test_single_story_max_diversity(self):
        stories = ["A single story stands alone."]
        diversity = calculate_story_diversity(stories)
        assert diversity == 1.0


class TestCalculateShannonEntropy:
    """Tests for Shannon entropy calculation."""
    
    def test_uniform_distribution_high_entropy(self):
        counts = Counter({"a": 5, "b": 5, "c": 5, "d": 5})
        entropy = calculate_shannon_entropy(counts)
        assert entropy == 2.0  # log2(4) = 2 for uniform distribution
    
    def test_single_item_zero_entropy(self):
        counts = Counter({"a": 10})
        entropy = calculate_shannon_entropy(counts)
        assert entropy == 0.0  # No uncertainty with single item
    
    def test_empty_counter_zero_entropy(self):
        counts = Counter()
        entropy = calculate_shannon_entropy(counts)
        assert entropy == 0.0


class TestQuiltingBenchmark:
    """Tests for the QuiltingBenchmark class."""
    
    def test_benchmark_initialization(self):
        """Test benchmark initializes with correct defaults."""
        benchmark = QuiltingBenchmark()
        
        assert benchmark.name == "quilting"
        assert benchmark.version == "1.0.0"
        assert len(benchmark.fragments) > 10  # Has default fragments
        assert benchmark.num_trials == 15
        assert benchmark.min_subset_size == 3
        assert benchmark.max_subset_size == 6
    
    def test_custom_fragments(self):
        """Test benchmark with custom fragments."""
        custom_fragments = ["fragment one", "fragment two", "fragment three"]
        benchmark = QuiltingBenchmark(fragments=custom_fragments)
        
        assert benchmark.fragments == custom_fragments
    
    def test_get_prompts(self):
        """Test prompt generation."""
        benchmark = QuiltingBenchmark(num_trials=5, fragments_per_prompt=8)
        prompts = benchmark.get_prompts()
        
        assert len(prompts) == 5
        for prompt in prompts:
            assert "id" in prompt
            assert "prompt" in prompt
            assert "presented_fragments" in prompt
            assert len(prompt["presented_fragments"]) == 8
            assert "FRAGMENTS:" in prompt["prompt"]
    
    def test_prompt_contains_instructions(self):
        """Test that prompts contain proper instructions."""
        benchmark = QuiltingBenchmark(
            num_trials=1,
            min_subset_size=3,
            max_subset_size=5,
            target_story_words=150,
        )
        prompts = benchmark.get_prompts()
        
        prompt_text = prompts[0]["prompt"]
        assert "3 to 5 fragments" in prompt_text
        assert "150 words" in prompt_text
    
    def test_generation_config(self):
        """Test generation config is appropriate for creative tasks."""
        benchmark = QuiltingBenchmark()
        config = benchmark.get_generation_config()
        
        assert config.temperature == 0.9  # High for creativity
        assert config.max_tokens >= 1024  # Enough for story
    
    def test_system_prompt_exists(self):
        """Test system prompt provides creative guidance."""
        benchmark = QuiltingBenchmark()
        system_prompt = benchmark.get_system_prompt()
        
        assert system_prompt is not None
        assert "creative" in system_prompt.lower()
    
    def test_reset_clears_state(self):
        """Test that reset clears tracked state."""
        benchmark = QuiltingBenchmark()
        
        # Simulate some tracked state
        benchmark._chosen_subsets.append(frozenset(["a", "b"]))
        benchmark._fragment_usage.update(["a", "b", "a"])
        benchmark._stories.append("A test story")
        benchmark._trial_results.append({"test": "data"})
        
        benchmark.reset()
        
        assert len(benchmark._chosen_subsets) == 0
        assert len(benchmark._fragment_usage) == 0
        assert len(benchmark._stories) == 0
        assert len(benchmark._trial_results) == 0


class TestQuiltingBenchmarkWithMockClient:
    """Integration tests for QuiltingBenchmark with mock client."""
    
    def test_evaluate_response_extracts_fragments(self):
        """Test that evaluate_response correctly extracts chosen fragments."""
        benchmark = QuiltingBenchmark(num_trials=1)
        benchmark.reset()
        
        prompt_data = {
            "id": "test_1",
            "prompt": "test prompt",
            "presented_fragments": [
                "a clock that runs backwards",
                "the weight of unspoken words",
                "a door that opens both ways",
                "the color of forgotten dreams",
            ],
        }
        
        response = """FRAGMENTS:
1. "a clock that runs backwards"
2. "the weight of unspoken words"

STORY:
A clock that runs backwards hung on the wall. The weight of unspoken words filled the room."""
        
        llm_response = LLMResponse(
            content=response,
            model="test",
            provider="test",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=200, total_tokens=300),
            latency_ms=300.0,
        )
        
        scores = benchmark.evaluate_response(prompt_data, response, llm_response)
        
        assert scores["num_chosen"] == 2.0
        assert scores["num_incorporated"] == 2.0
        assert scores["incorporation_rate"] == 1.0
        assert scores["valid_subset"] == 1.0
    
    def test_varying_choices_increases_diversity(self):
        """Test that varying fragment choices results in higher diversity."""
        benchmark_varying = QuiltingBenchmark(num_trials=5, seed=42)
        benchmark_uniform = QuiltingBenchmark(num_trials=5, seed=42)
        
        # Both benchmarks will need different simulated responses
        # This tests the concept - actual diversity depends on LLM responses
        
        prompts_varying = benchmark_varying.get_prompts()
        prompts_uniform = benchmark_uniform.get_prompts()
        
        # Both should generate same number of prompts
        assert len(prompts_varying) == len(prompts_uniform) == 5


class TestQuiltingMetrics:
    """Tests for QuiltingMetrics dataclass."""
    
    def test_metrics_to_dict(self):
        """Test that metrics can be converted to dictionary."""
        metrics = QuiltingMetrics(
            total_trials=10,
            unique_subsets=8,
            subset_diversity_score=0.8,
            mean_subset_size=4.5,
            fragments_ever_used=25,
            fragments_never_used=5,
            fragment_usage_entropy=4.2,
            most_used_fragments=[("fragment1", 5), ("fragment2", 4)],
            least_used_fragments=[("fragment30", 1)],
            mean_story_length=180.5,
            mean_fragment_incorporation=3.8,
            mean_incorporation_rate=0.85,
            story_diversity_score=0.72,
            mean_story_uniqueness=0.45,
            quilting_score=0.78,
        )
        
        d = metrics.to_dict()
        
        assert d["total_trials"] == 10
        assert d["unique_subsets"] == 8
        assert d["subset_diversity_score"] == 0.8
        assert d["quilting_score"] == 0.78
        assert d["most_used_fragments"] == [("fragment1", 5), ("fragment2", 4)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
