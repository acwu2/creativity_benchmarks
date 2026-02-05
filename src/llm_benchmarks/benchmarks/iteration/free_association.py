"""Free Association benchmark for measuring creative vocabulary exploration."""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from llm_benchmarks.benchmarks.base import (
    BaseBenchmark,
    BenchmarkResult,
    PromptResult,
    AggregatedMetrics,
)
from llm_benchmarks.clients.base import BaseLLMClient, GenerationConfig, LLMResponse


@dataclass
class FreeAssociationMetrics:
    """Metrics specific to free association benchmark."""
    
    total_words: int = 0
    unique_words: int = 0
    time_to_first_repetition: Optional[int] = None  # Number of words before first repeat
    repetition_rate: float = 0.0  # Fraction of words that are repeats
    word_frequency: dict[str, int] = field(default_factory=dict)
    
    # Species richness estimation (Chao1)
    estimated_total_vocabulary: Optional[float] = None
    chao1_lower_bound: Optional[float] = None
    chao1_upper_bound: Optional[float] = None
    
    # Per-chain metrics
    chain_lengths: list[int] = field(default_factory=list)
    chain_unique_counts: list[int] = field(default_factory=list)
    chain_ttfr: list[Optional[int]] = field(default_factory=list)  # Time to first rep per chain
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_words": self.total_words,
            "unique_words": self.unique_words,
            "time_to_first_repetition": self.time_to_first_repetition,
            "repetition_rate": self.repetition_rate,
            "estimated_total_vocabulary": self.estimated_total_vocabulary,
            "chao1_lower_bound": self.chao1_lower_bound,
            "chao1_upper_bound": self.chao1_upper_bound,
            "mean_chain_length": np.mean(self.chain_lengths) if self.chain_lengths else 0,
            "mean_chain_unique": np.mean(self.chain_unique_counts) if self.chain_unique_counts else 0,
        }


def extract_words(text: str) -> list[str]:
    """
    Extract words from text, normalizing to lowercase.
    
    Handles various formats:
    - Comma-separated lists
    - Numbered lists
    - Space-separated words
    - Arrow-separated chains (word -> word -> word)
    """
    # Remove common list formatting
    text = re.sub(r"^\d+[\.\)]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*•]\s*", "", text, flags=re.MULTILINE)
    
    # Replace arrow separators
    text = text.replace("->", " ").replace("→", " ").replace("=>", " ")
    
    # Split on common delimiters
    words = re.split(r"[,\n;|]+", text)
    
    # Further split on spaces and clean
    result = []
    for word in words:
        # Split on spaces
        subwords = word.strip().split()
        for sw in subwords:
            # Clean and normalize
            cleaned = re.sub(r"[^a-zA-Z0-9'-]", "", sw.lower())
            if cleaned and len(cleaned) > 1:  # Skip single characters
                result.append(cleaned)
    
    return result


def calculate_time_to_first_repetition(words: list[str]) -> Optional[int]:
    """
    Calculate how many words appear before the first repetition.
    
    Returns None if no repetition occurs.
    """
    seen = set()
    for i, word in enumerate(words):
        if word in seen:
            return i  # 0-indexed position of first repeat
        seen.add(word)
    return None  # No repetition


def estimate_chao1(word_counts: Counter) -> tuple[float, Optional[float], Optional[float]]:
    """
    Estimate total vocabulary size using Chao1 estimator.
    
    This treats the problem as an "unseen species" problem - estimating
    how many unique words the model "knows" based on the sample we've seen.
    
    Returns:
        (estimate, lower_95, upper_95) - estimate and confidence interval
    """
    observed = len(word_counts)  # Number of unique words seen
    
    if observed == 0:
        return 0.0, None, None
    
    # Count singletons (f1) and doubletons (f2)
    f1 = sum(1 for count in word_counts.values() if count == 1)
    f2 = sum(1 for count in word_counts.values() if count == 2)
    
    # Chao1 estimator
    if f2 > 0:
        chao1 = observed + (f1 ** 2) / (2 * f2)
    else:
        # Bias-corrected form when f2 = 0
        chao1 = observed + (f1 * (f1 - 1)) / 2
    
    # Confidence interval (approximate)
    # Using the variance formula for Chao1
    if f2 > 0:
        var = f2 * (
            (f1 / f2) ** 2 / 4 +
            (f1 / f2) ** 3 +
            (f1 / f2) ** 4 / 4
        )
    else:
        var = (f1 * (f1 - 1) / 2) + (f1 * (2 * f1 - 1) ** 2 / 4) - (f1 ** 4 / (4 * chao1))
        var = max(0, var)  # Ensure non-negative
    
    if var > 0:
        # Log-transform confidence interval
        c = np.exp(1.96 * np.sqrt(np.log(1 + var / (chao1 - observed) ** 2)))
        lower = observed + (chao1 - observed) / c
        upper = observed + (chao1 - observed) * c
        return chao1, lower, upper
    
    return chao1, None, None


class FreeAssociationBenchmark(BaseBenchmark):
    """
    Benchmark for measuring creative vocabulary through free association.
    
    Prompts models to freely associate words, then measures:
    - Total unique words generated across all chains
    - Time to first repetition (how long before repeating a word)
    - Estimated total vocabulary using species richness estimation
    
    The benchmark runs multiple association chains to get a robust sample
    of the model's associative vocabulary.
    """
    
    name = "free_association"
    version = "1.0.0"
    description = "Measures creative vocabulary through free word association"
    
    # Seed words to start association chains
    DEFAULT_SEED_WORDS = [
        "ocean", "time", "light", "dream", "silence",
        "journey", "memory", "shadow", "change", "hope",
        "fire", "crystal", "wind", "stone", "music",
        "forest", "star", "river", "mirror", "dance",
    ]
    
    # Additional prompts without seed words (pure free association)
    NUM_UNSEEDED_CHAINS = 5
    
    def __init__(
        self,
        seed_words: Optional[list[str]] = None,
        words_per_chain: int = 50,
        num_unseeded_chains: int = 5,
        include_numbers: bool = False,
        **kwargs,
    ):
        """
        Initialize the free association benchmark.
        
        Args:
            seed_words: Starting words for association chains (None for defaults)
            words_per_chain: Target number of words per association chain
            num_unseeded_chains: Number of chains without seed words
            include_numbers: Whether to include number association prompts
        """
        super().__init__(**kwargs)
        self.seed_words = seed_words or self.DEFAULT_SEED_WORDS
        self.words_per_chain = words_per_chain
        self.num_unseeded_chains = num_unseeded_chains
        self.include_numbers = include_numbers
        
        # Track all words across the entire benchmark run
        self._all_words: list[str] = []
        self._word_counts: Counter = Counter()
        self._chain_metrics: list[dict] = []
    
    def get_prompts(self) -> list[dict[str, Any]]:
        """Generate prompts for free association chains."""
        prompts = []
        
        # Seeded association prompts
        for i, seed in enumerate(self.seed_words):
            prompts.append({
                "id": f"seeded_{i}",
                "prompt": self._make_seeded_prompt(seed),
                "seed_word": seed,
                "type": "seeded",
                "metadata": {"seed": seed},
            })
        
        # Unseeded (pure free association) prompts
        for i in range(self.num_unseeded_chains):
            prompts.append({
                "id": f"unseeded_{i}",
                "prompt": self._make_unseeded_prompt(),
                "seed_word": None,
                "type": "unseeded",
                "metadata": {},
            })
        
        # Number association prompts (if enabled)
        if self.include_numbers:
            for i, start_num in enumerate([1, 7, 42, 100, 3]):
                prompts.append({
                    "id": f"number_{i}",
                    "prompt": self._make_number_prompt(start_num),
                    "seed_word": str(start_num),
                    "type": "number",
                    "metadata": {"start_number": start_num},
                })
        
        return prompts
    
    def _make_seeded_prompt(self, seed_word: str) -> str:
        """Create a seeded free association prompt."""
        return f"""Free association exercise: Starting with the word "{seed_word}", say whatever word comes to mind next. Then from that word, say the next word that comes to mind. Continue this chain of free association.

Just list the words, one after another, separated by commas. Don't explain or describe - just freely associate from word to word. Generate approximately {self.words_per_chain} words.

Start: {seed_word}"""

    def _make_unseeded_prompt(self) -> str:
        """Create an unseeded free association prompt."""
        return f"""Free association exercise: Say whatever word first comes to mind right now. Then from that word, say the next word that comes to mind. Continue this chain of free association.

Just list the words, one after another, separated by commas. Don't explain or describe - just freely associate from word to word. Let your mind wander freely. Generate approximately {self.words_per_chain} words."""

    def _make_number_prompt(self, start_num: int) -> str:
        """Create a number association prompt."""
        return f"""Free association with numbers: Starting with the number {start_num}, say whatever number comes to mind next. It doesn't have to follow any pattern - just say whatever number feels right. Continue this chain.

Just list the numbers, one after another, separated by commas. Don't explain your choices - just freely associate from number to number. Generate approximately {self.words_per_chain} numbers.

Start: {start_num}"""

    def get_system_prompt(self) -> Optional[str]:
        """System prompt encouraging creative, unconstrained association."""
        return """You are participating in a free association exercise. Your goal is to let your mind flow freely from one word to the next, without overthinking or trying to be clever. 

Don't try to:
- Create themes or patterns
- Be poetic or meaningful
- Explain your associations
- Repeat words you've already said

Do:
- Say whatever genuinely comes to mind next
- Keep the chain flowing naturally
- Be spontaneous and uncensored
- Embrace unexpected connections"""

    def get_generation_config(self) -> GenerationConfig:
        """Use higher temperature for more creative associations."""
        return GenerationConfig(
            max_tokens=1024,  # Enough for ~50+ words
            temperature=1.0,  # Maximum creativity
            top_p=0.95,
        )
    
    def evaluate_response(
        self,
        prompt_data: dict[str, Any],
        response: str,
        llm_response: LLMResponse,
    ) -> dict[str, float]:
        """
        Evaluate a single association chain.
        
        Returns per-chain metrics; aggregation happens in aggregate_scores.
        """
        # Extract words from response
        words = extract_words(response)
        
        # Include seed word at the start if present (but avoid duplicate if model already included it)
        seed = prompt_data.get("seed_word")
        if seed and prompt_data.get("type") != "number":
            seed_lower = seed.lower()
            # Only prepend if the response doesn't already start with the seed word
            if not words or words[0] != seed_lower:
                words = [seed_lower] + words
        
        # Calculate chain-specific metrics
        chain_length = len(words)
        chain_unique = len(set(words))
        chain_ttfr = calculate_time_to_first_repetition(words)
        
        # Store for aggregation
        self._chain_metrics.append({
            "words": words,
            "length": chain_length,
            "unique": chain_unique,
            "ttfr": chain_ttfr,
            "type": prompt_data.get("type"),
        })
        
        # Add to global word tracking
        self._all_words.extend(words)
        self._word_counts.update(words)
        
        # Calculate repetition rate for this chain
        if chain_length > 0:
            repetition_rate = 1.0 - (chain_unique / chain_length)
        else:
            repetition_rate = 0.0
        
        return {
            "chain_length": float(chain_length),
            "chain_unique_words": float(chain_unique),
            "chain_ttfr": float(chain_ttfr) if chain_ttfr is not None else float(chain_length),
            "chain_repetition_rate": repetition_rate,
        }
    
    def aggregate_scores(
        self,
        prompt_results: list[PromptResult],
        client: BaseLLMClient,
    ) -> AggregatedMetrics:
        """
        Aggregate free association metrics across all chains.
        
        Computes global unique word count, overall TTFR, and vocabulary estimates.
        """
        # Calculate global metrics
        total_words = len(self._all_words)
        unique_words = len(self._word_counts)
        
        # Global time to first repetition
        global_ttfr = calculate_time_to_first_repetition(self._all_words)
        
        # Repetition rate
        if total_words > 0:
            repetition_rate = 1.0 - (unique_words / total_words)
        else:
            repetition_rate = 0.0
        
        # Estimate total vocabulary using Chao1
        chao1, chao1_lower, chao1_upper = estimate_chao1(self._word_counts)
        
        # Build metrics
        fa_metrics = FreeAssociationMetrics(
            total_words=total_words,
            unique_words=unique_words,
            time_to_first_repetition=global_ttfr,
            repetition_rate=repetition_rate,
            word_frequency=dict(self._word_counts.most_common(100)),  # Top 100
            estimated_total_vocabulary=chao1,
            chao1_lower_bound=chao1_lower,
            chao1_upper_bound=chao1_upper,
            chain_lengths=[m["length"] for m in self._chain_metrics],
            chain_unique_counts=[m["unique"] for m in self._chain_metrics],
            chain_ttfr=[m["ttfr"] for m in self._chain_metrics],
        )
        
        # Per-chain score aggregation
        successful = [r for r in prompt_results if not r.is_error]
        
        mean_scores = {
            "unique_words": float(unique_words),
            "total_words": float(total_words),
            "time_to_first_repetition": float(global_ttfr) if global_ttfr else float(total_words),
            "repetition_rate": repetition_rate,
            "estimated_vocabulary": chao1,
            "vocabulary_utilization": unique_words / chao1 if chao1 > 0 else 0,
        }
        
        # Chain-level statistics
        chain_lengths = [m["length"] for m in self._chain_metrics]
        chain_uniques = [m["unique"] for m in self._chain_metrics]
        chain_ttfrs = [m["ttfr"] for m in self._chain_metrics if m["ttfr"] is not None]
        
        std_scores = {
            "chain_length_std": float(np.std(chain_lengths)) if chain_lengths else 0,
            "chain_unique_std": float(np.std(chain_uniques)) if chain_uniques else 0,
        }
        
        # Token/latency tracking
        total_tokens = sum(
            r.llm_response.usage.total_tokens
            for r in successful
            if r.llm_response
        )
        total_latency = sum(
            r.llm_response.latency_ms
            for r in successful
            if r.llm_response
        )
        
        # Cost calculation
        model_info = client.get_model_info()
        total_cost = None
        if model_info.input_cost_per_1k and model_info.output_cost_per_1k:
            total_cost = sum(
                r.llm_response.calculate_cost(model_info) or 0
                for r in successful
                if r.llm_response
            )
        
        return AggregatedMetrics(
            mean_scores=mean_scores,
            std_scores=std_scores,
            min_scores={
                "chain_length": float(min(chain_lengths)) if chain_lengths else 0,
                "chain_unique": float(min(chain_uniques)) if chain_uniques else 0,
            },
            max_scores={
                "chain_length": float(max(chain_lengths)) if chain_lengths else 0,
                "chain_unique": float(max(chain_uniques)) if chain_uniques else 0,
            },
            total_prompts=len(prompt_results),
            successful_prompts=len(successful),
            failed_prompts=len(prompt_results) - len(successful),
            total_tokens=total_tokens,
            total_latency_ms=total_latency,
            mean_latency_ms=total_latency / len(successful) if successful else 0,
            total_cost_usd=total_cost,
        )
    
    def run(
        self,
        client: BaseLLMClient,
        progress_callback: Optional[callable] = None,
    ) -> BenchmarkResult:
        """Run the benchmark, resetting state first."""
        # Reset state for new run
        self._all_words = []
        self._word_counts = Counter()
        self._chain_metrics = []
        
        return super().run(client, progress_callback)
    
    async def run_async(
        self,
        client: BaseLLMClient,
        max_concurrent: int = 5,
        progress_callback: Optional[callable] = None,
    ) -> BenchmarkResult:
        """Run the benchmark asynchronously, resetting state first."""
        # Reset state for new run
        self._all_words = []
        self._word_counts = Counter()
        self._chain_metrics = []
        
        return await super().run_async(client, max_concurrent, progress_callback)
    
    def get_detailed_results(self) -> FreeAssociationMetrics:
        """
        Get detailed metrics after running the benchmark.
        
        Call this after run() to get the full FreeAssociationMetrics object.
        """
        if not self._all_words:
            raise RuntimeError("No results available. Run the benchmark first.")
        
        global_ttfr = calculate_time_to_first_repetition(self._all_words)
        chao1, chao1_lower, chao1_upper = estimate_chao1(self._word_counts)
        
        return FreeAssociationMetrics(
            total_words=len(self._all_words),
            unique_words=len(self._word_counts),
            time_to_first_repetition=global_ttfr,
            repetition_rate=1.0 - len(self._word_counts) / len(self._all_words) if self._all_words else 0,
            word_frequency=dict(self._word_counts.most_common(100)),
            estimated_total_vocabulary=chao1,
            chao1_lower_bound=chao1_lower,
            chao1_upper_bound=chao1_upper,
            chain_lengths=[m["length"] for m in self._chain_metrics],
            chain_unique_counts=[m["unique"] for m in self._chain_metrics],
            chain_ttfr=[m["ttfr"] for m in self._chain_metrics],
        )
