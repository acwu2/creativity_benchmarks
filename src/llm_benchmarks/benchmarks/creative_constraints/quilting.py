"""Quilting benchmark for measuring creative synthesis from constrained ingredients."""

import random
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


# Default fragments - short quotes, ideas, concepts, images
DEFAULT_FRAGMENTS = [
    # Abstract concepts
    "a clock that runs backwards",
    "the weight of unspoken words",
    "a door that opens both ways",
    "the color of forgotten dreams",
    "silence that speaks volumes",
    
    # Evocative images
    "rain falling upward into clouds",
    "a lighthouse in the desert",
    "footprints that lead nowhere",
    "a mirror reflecting yesterday",
    "stars that hum a melody",
    
    # Emotional fragments
    "the ache of almost",
    "joy wrapped in paper thin",
    "fear wearing a familiar face",
    "hope disguised as stubbornness",
    "grief that tastes like honey",
    
    # Paradoxes and tensions
    "the loud whisper of truth",
    "a gentle hurricane",
    "patient urgency",
    "organized chaos blooming",
    "frozen fire dancing",
    
    # Character sketches
    "a child who remembers everything",
    "someone collecting other people's shadows",
    "the last person who can read",
    "a translator of birdsong",
    "a cartographer of emotions",
    
    # Setting fragments
    "a city built on promises",
    "the room between rooms",
    "where the sidewalk forgets to end",
    "a garden of mechanical flowers",
    "the library of unwritten books",
    
    # Action fragments
    "catching echoes in a jar",
    "sewing clouds together",
    "teaching stones to float",
    "unraveling the thread of time",
    "painting with borrowed colors",
    
    # Philosophical hooks
    "the question that answers itself",
    "a truth too small to see",
    "the space between heartbeats",
    "where meaning goes to rest",
    "the edge of understanding",
]


@dataclass
class QuiltingMetrics:
    """Metrics specific to the Quilting benchmark."""
    
    # Subset selection metrics
    total_trials: int = 0
    unique_subsets: int = 0  # Number of distinct fragment combinations chosen
    subset_diversity_score: float = 0.0  # 1 - (most_common_count / total_trials)
    mean_subset_size: float = 0.0
    
    # Fragment usage metrics
    fragments_ever_used: int = 0  # How many unique fragments were used at least once
    fragments_never_used: int = 0  # Fragments the model never chose
    fragment_usage_entropy: float = 0.0  # Shannon entropy of fragment selection
    most_used_fragments: list[tuple[str, int]] = field(default_factory=list)
    least_used_fragments: list[tuple[str, int]] = field(default_factory=list)
    
    # Story quality metrics (per-trial averages)
    mean_story_length: float = 0.0
    mean_fragment_incorporation: float = 0.0  # How many chosen fragments appear in story
    mean_incorporation_rate: float = 0.0  # Fraction of chosen fragments used
    
    # Diversity of outputs
    story_diversity_score: float = 0.0  # Lexical diversity across stories
    mean_story_uniqueness: float = 0.0  # Average pairwise distance between stories
    
    # Combined score
    quilting_score: float = 0.0  # Composite score
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_trials": self.total_trials,
            "unique_subsets": self.unique_subsets,
            "subset_diversity_score": self.subset_diversity_score,
            "mean_subset_size": self.mean_subset_size,
            "fragments_ever_used": self.fragments_ever_used,
            "fragments_never_used": self.fragments_never_used,
            "fragment_usage_entropy": self.fragment_usage_entropy,
            "most_used_fragments": self.most_used_fragments,
            "least_used_fragments": self.least_used_fragments,
            "mean_story_length": self.mean_story_length,
            "mean_fragment_incorporation": self.mean_fragment_incorporation,
            "mean_incorporation_rate": self.mean_incorporation_rate,
            "story_diversity_score": self.story_diversity_score,
            "mean_story_uniqueness": self.mean_story_uniqueness,
            "quilting_score": self.quilting_score,
        }


def parse_quilting_response(response: str) -> tuple[list[str], str]:
    """
    Parse a quilting response to extract chosen fragments and the story.
    
    Expected format:
    FRAGMENTS:
    1. fragment one
    2. fragment two
    ...
    
    STORY:
    The actual story text...
    
    Returns:
        (list of chosen fragment texts, story text)
    """
    # Try to find fragments section
    fragments = []
    story = ""
    
    # Look for FRAGMENTS: section
    fragments_match = re.search(
        r"(?:FRAGMENTS|CHOSEN|SELECTED|INGREDIENTS)[:\s]*\n(.*?)(?:\n\s*(?:STORY|NARRATIVE|TALE)[:\s]*\n|$)",
        response,
        re.IGNORECASE | re.DOTALL
    )
    
    if fragments_match:
        fragments_text = fragments_match.group(1)
        # Extract numbered or bulleted items
        fragment_lines = re.findall(
            r"(?:^\s*(?:\d+[\.\)]\s*|[-*•]\s*))(.+?)$",
            fragments_text,
            re.MULTILINE
        )
        fragments = [f.strip().strip('"\'') for f in fragment_lines if f.strip()]
    
    # Look for STORY: section
    story_match = re.search(
        r"(?:STORY|NARRATIVE|TALE)[:\s]*\n(.+)",
        response,
        re.IGNORECASE | re.DOTALL
    )
    
    if story_match:
        story = story_match.group(1).strip()
    else:
        # Fallback: if no explicit sections, try to split on double newline
        # and assume latter part is the story
        parts = response.split("\n\n", 1)
        if len(parts) == 2 and not fragments:
            # First part might be fragments, second part story
            potential_fragments = parts[0]
            fragment_lines = re.findall(
                r"(?:^\s*(?:\d+[\.\)]\s*|[-*•]\s*))(.+?)$",
                potential_fragments,
                re.MULTILINE
            )
            if fragment_lines:
                fragments = [f.strip().strip('"\'') for f in fragment_lines if f.strip()]
                story = parts[1].strip()
            else:
                story = response  # Just treat whole thing as story
        elif not story:
            story = response  # Just treat whole thing as story
    
    return fragments, story


def normalize_fragment(text: str) -> str:
    """Normalize a fragment for comparison."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def match_fragment_to_original(
    chosen: str,
    available: list[str],
    threshold: float = 0.6,
) -> Optional[str]:
    """
    Match a chosen fragment text to one of the original fragments.
    
    Uses fuzzy matching since the model might paraphrase slightly.
    
    Returns the matched original fragment or None if no match.
    """
    chosen_norm = normalize_fragment(chosen)
    chosen_words = set(chosen_norm.split())
    
    best_match = None
    best_score = 0.0
    
    for original in available:
        original_norm = normalize_fragment(original)
        original_words = set(original_norm.split())
        
        # Jaccard similarity
        if not chosen_words or not original_words:
            continue
            
        intersection = len(chosen_words & original_words)
        union = len(chosen_words | original_words)
        score = intersection / union if union > 0 else 0
        
        # Also check substring containment
        if chosen_norm in original_norm or original_norm in chosen_norm:
            score = max(score, 0.8)
        
        if score > best_score and score >= threshold:
            best_score = score
            best_match = original
    
    return best_match


def fragment_in_story(fragment: str, story: str) -> bool:
    """Check if a fragment's essence appears in the story."""
    story_lower = story.lower()
    fragment_norm = normalize_fragment(fragment)
    
    # Check for exact phrase (normalized)
    if fragment_norm in normalize_fragment(story):
        return True
    
    # Check if key words from fragment appear in story
    fragment_words = fragment_norm.split()
    # Need at least half the words to match
    matches = sum(1 for w in fragment_words if len(w) > 3 and w in story_lower)
    return matches >= len(fragment_words) / 2


def calculate_story_diversity(stories: list[str]) -> float:
    """
    Calculate diversity across a set of stories using lexical diversity.
    
    Returns a score from 0 to 1, where higher means more diverse.
    """
    if len(stories) < 2:
        return 1.0
    
    # Use word set overlap as a measure
    word_sets = []
    for story in stories:
        words = set(re.findall(r'\b\w+\b', story.lower()))
        word_sets.append(words)
    
    # Calculate average pairwise Jaccard distance
    distances = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            intersection = len(word_sets[i] & word_sets[j])
            union = len(word_sets[i] | word_sets[j])
            if union > 0:
                similarity = intersection / union
                distances.append(1 - similarity)  # Convert to distance
    
    return np.mean(distances) if distances else 1.0


def calculate_shannon_entropy(counts: Counter) -> float:
    """Calculate Shannon entropy of a distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    
    return entropy


class QuiltingBenchmark(BaseBenchmark):
    """
    Benchmark for measuring creative synthesis from constrained ingredients.
    
    The model is given a shuffled set of creative "fragments" (short quotes,
    ideas, images) and asked to:
    1. Choose a subset of fragments
    2. List which fragments they chose
    3. Write a short story incorporating those fragments
    
    Scoring:
    - Subset diversity: Do they pick different combinations each time?
    - Fragment utilization: Do they actually use the fragments they chose?
    - Story diversity: Are the resulting stories diverse?
    - Combined quilting score: Overall creative synthesis ability
    """
    
    name = "quilting"
    version = "1.0.0"
    description = "Measures creative synthesis by combining constrained ingredients into stories"
    
    def __init__(
        self,
        fragments: Optional[list[str]] = None,
        num_trials: int = 15,
        fragments_per_prompt: int = 12,
        min_subset_size: int = 3,
        max_subset_size: int = 6,
        target_story_words: int = 200,
        shuffle_fragments: bool = True,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the Quilting benchmark.
        
        Args:
            fragments: Pool of creative fragments to choose from
            num_trials: Number of quilting prompts to generate
            fragments_per_prompt: How many fragments to present each time
            min_subset_size: Minimum fragments the model should choose
            max_subset_size: Maximum fragments the model should choose
            target_story_words: Approximate story length to request
            shuffle_fragments: Whether to shuffle fragment order each prompt
            seed: Random seed for reproducibility
        """
        super().__init__(**kwargs)
        self.fragments = fragments or DEFAULT_FRAGMENTS.copy()
        self.num_trials = num_trials
        self.fragments_per_prompt = min(fragments_per_prompt, len(self.fragments))
        self.min_subset_size = min_subset_size
        self.max_subset_size = max_subset_size
        self.target_story_words = target_story_words
        self.shuffle_fragments = shuffle_fragments
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
        
        # Track results across trials
        self._chosen_subsets: list[frozenset[str]] = []
        self._fragment_usage: Counter = Counter()
        self._stories: list[str] = []
        self._trial_results: list[dict[str, Any]] = []
    
    def get_prompts(self) -> list[dict[str, Any]]:
        """Generate quilting prompts with different fragment selections."""
        prompts = []
        
        for i in range(self.num_trials):
            # Select a subset of fragments to present
            if self.shuffle_fragments:
                presented_fragments = random.sample(
                    self.fragments,
                    min(self.fragments_per_prompt, len(self.fragments))
                )
            else:
                # Rotate through fragments deterministically
                start_idx = (i * self.fragments_per_prompt) % len(self.fragments)
                indices = [
                    (start_idx + j) % len(self.fragments)
                    for j in range(self.fragments_per_prompt)
                ]
                presented_fragments = [self.fragments[idx] for idx in indices]
            
            prompt_text = self._make_quilting_prompt(presented_fragments)
            
            prompts.append({
                "id": f"quilting_{i}",
                "prompt": prompt_text,
                "presented_fragments": presented_fragments,
                "metadata": {
                    "trial_number": i,
                    "num_fragments_presented": len(presented_fragments),
                },
            })
        
        return prompts
    
    def _make_quilting_prompt(self, fragments: list[str]) -> str:
        """Create a quilting prompt with the given fragments."""
        fragment_list = "\n".join(f"{i+1}. \"{f}\"" for i, f in enumerate(fragments))
        
        return f"""Creative writing exercise: Below are {len(fragments)} creative fragments - evocative phrases, images, and ideas. Your task is to:

1. Choose {self.min_subset_size} to {self.max_subset_size} fragments that inspire you
2. List the fragments you've chosen
3. Write a short story (approximately {self.target_story_words} words) that weaves these fragments together

FRAGMENTS:
{fragment_list}

Please respond in this format:

FRAGMENTS:
(List the {self.min_subset_size}-{self.max_subset_size} fragments you've chosen, numbered)

STORY:
(Your short story incorporating these fragments)

Be creative in how you interpret and combine the fragments. They don't need to appear literally - you can use their essence, imagery, or themes."""

    def get_system_prompt(self) -> Optional[str]:
        """System prompt for creative quilting."""
        return """You are a creative writer participating in a constrained writing exercise. 
Your goal is to select interesting combinations of provided fragments and weave them into 
cohesive, imaginative short stories.

Guidelines:
- Choose fragments that genuinely inspire you, not just the first ones listed
- Be creative in how you interpret and combine the fragments
- The fragments can appear literally or thematically in your story
- Write engaging prose that naturally incorporates your chosen elements
- Each story should feel complete, with a beginning, middle, and end (or a meaningful moment)
- Vary your choices and approaches across different prompts"""

    def get_generation_config(self) -> GenerationConfig:
        """Use moderate temperature for creative but coherent output."""
        return GenerationConfig(
            max_tokens=1024,
            temperature=0.9,
            top_p=0.95,
        )
    
    def evaluate_response(
        self,
        prompt_data: dict[str, Any],
        response: str,
        llm_response: LLMResponse,
    ) -> dict[str, float]:
        """
        Evaluate a single quilting response.
        
        Returns per-trial metrics; aggregation happens in aggregate_scores.
        """
        presented_fragments = prompt_data["presented_fragments"]
        
        # Parse the response
        chosen_fragments_raw, story = parse_quilting_response(response)
        
        # Match chosen fragments to originals
        chosen_fragments = []
        for chosen in chosen_fragments_raw:
            match = match_fragment_to_original(chosen, presented_fragments)
            if match:
                chosen_fragments.append(match)
        
        # Deduplicate while preserving order
        seen = set()
        unique_chosen = []
        for f in chosen_fragments:
            if f not in seen:
                unique_chosen.append(f)
                seen.add(f)
        chosen_fragments = unique_chosen
        
        # Calculate incorporation metrics
        incorporated_count = sum(
            1 for f in chosen_fragments if fragment_in_story(f, story)
        )
        
        incorporation_rate = (
            incorporated_count / len(chosen_fragments)
            if chosen_fragments else 0.0
        )
        
        # Story length
        story_words = len(story.split())
        
        # Track for aggregation
        subset_key = frozenset(chosen_fragments)
        self._chosen_subsets.append(subset_key)
        self._fragment_usage.update(chosen_fragments)
        self._stories.append(story)
        
        self._trial_results.append({
            "presented": presented_fragments,
            "chosen": chosen_fragments,
            "story": story,
            "incorporated": incorporated_count,
            "incorporation_rate": incorporation_rate,
            "story_words": story_words,
        })
        
        return {
            "num_chosen": float(len(chosen_fragments)),
            "num_incorporated": float(incorporated_count),
            "incorporation_rate": incorporation_rate,
            "story_length": float(story_words),
            "valid_subset": 1.0 if chosen_fragments else 0.0,
        }
    
    def aggregate_scores(
        self,
        prompt_results: list[PromptResult],
        client: BaseLLMClient,
    ) -> AggregatedMetrics:
        """
        Aggregate quilting metrics across all trials.
        
        Computes subset diversity, fragment utilization, and story diversity.
        """
        total_trials = len(self._trial_results)
        
        # Subset diversity
        unique_subsets = len(set(self._chosen_subsets))
        subset_diversity = unique_subsets / total_trials if total_trials > 0 else 0.0
        
        # Fragment usage statistics
        fragments_ever_used = len(self._fragment_usage)
        fragments_never_used = len(self.fragments) - fragments_ever_used
        fragment_entropy = calculate_shannon_entropy(self._fragment_usage)
        
        # Normalize entropy by maximum possible entropy
        max_entropy = np.log2(len(self.fragments)) if len(self.fragments) > 1 else 1.0
        normalized_entropy = fragment_entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Most and least used fragments
        most_used = self._fragment_usage.most_common(5)
        least_used = self._fragment_usage.most_common()[:-6:-1] if len(self._fragment_usage) > 5 else []
        
        # Story metrics
        mean_story_length = (
            np.mean([t["story_words"] for t in self._trial_results])
            if self._trial_results else 0.0
        )
        mean_incorporation = (
            np.mean([t["incorporated"] for t in self._trial_results])
            if self._trial_results else 0.0
        )
        mean_incorporation_rate = (
            np.mean([t["incorporation_rate"] for t in self._trial_results])
            if self._trial_results else 0.0
        )
        mean_subset_size = (
            np.mean([len(t["chosen"]) for t in self._trial_results])
            if self._trial_results else 0.0
        )
        
        # Story diversity
        story_diversity = calculate_story_diversity(self._stories)
        
        # Calculate pairwise story uniqueness
        if len(self._stories) >= 2:
            word_sets = [set(re.findall(r'\b\w+\b', s.lower())) for s in self._stories]
            uniqueness_scores = []
            for i, ws in enumerate(word_sets):
                others = [word_sets[j] for j in range(len(word_sets)) if j != i]
                all_others = set().union(*others) if others else set()
                unique_words = ws - all_others
                uniqueness_scores.append(len(unique_words) / len(ws) if ws else 0)
            mean_story_uniqueness = np.mean(uniqueness_scores)
        else:
            mean_story_uniqueness = 1.0
        
        # Composite quilting score
        # Weight: subset diversity (30%), fragment entropy (20%), 
        #         incorporation rate (25%), story diversity (25%)
        quilting_score = (
            0.30 * subset_diversity +
            0.20 * normalized_entropy +
            0.25 * mean_incorporation_rate +
            0.25 * story_diversity
        )
        
        # Build metrics object
        quilting_metrics = QuiltingMetrics(
            total_trials=total_trials,
            unique_subsets=unique_subsets,
            subset_diversity_score=subset_diversity,
            mean_subset_size=mean_subset_size,
            fragments_ever_used=fragments_ever_used,
            fragments_never_used=fragments_never_used,
            fragment_usage_entropy=fragment_entropy,
            most_used_fragments=most_used,
            least_used_fragments=least_used,
            mean_story_length=mean_story_length,
            mean_fragment_incorporation=mean_incorporation,
            mean_incorporation_rate=mean_incorporation_rate,
            story_diversity_score=story_diversity,
            mean_story_uniqueness=mean_story_uniqueness,
            quilting_score=quilting_score,
        )
        
        # Standard score aggregation
        successful = [r for r in prompt_results if not r.is_error]
        
        mean_scores = {
            "quilting_score": quilting_score,
            "subset_diversity": subset_diversity,
            "unique_subsets": float(unique_subsets),
            "fragment_entropy": normalized_entropy,
            "incorporation_rate": mean_incorporation_rate,
            "story_diversity": story_diversity,
            "story_uniqueness": mean_story_uniqueness,
            "fragments_used": float(fragments_ever_used),
        }
        
        # Calculate std for per-trial metrics
        incorporation_rates = [t["incorporation_rate"] for t in self._trial_results]
        story_lengths = [t["story_words"] for t in self._trial_results]
        subset_sizes = [len(t["chosen"]) for t in self._trial_results]
        
        std_scores = {
            "incorporation_rate_std": float(np.std(incorporation_rates)) if incorporation_rates else 0.0,
            "story_length_std": float(np.std(story_lengths)) if story_lengths else 0.0,
            "subset_size_std": float(np.std(subset_sizes)) if subset_sizes else 0.0,
        }
        
        min_scores = {
            "min_incorporation_rate": min(incorporation_rates) if incorporation_rates else 0.0,
            "min_story_length": min(story_lengths) if story_lengths else 0.0,
        }
        
        max_scores = {
            "max_incorporation_rate": max(incorporation_rates) if incorporation_rates else 0.0,
            "max_story_length": max(story_lengths) if story_lengths else 0.0,
        }
        
        # Token and latency aggregation
        total_tokens = sum(
            (r.llm_response.usage.total_tokens if r.llm_response and r.llm_response.usage else 0)
            for r in prompt_results
        )
        latencies = [
            r.llm_response.latency_ms
            for r in prompt_results
            if r.llm_response and r.llm_response.latency_ms
        ]
        total_latency = sum(latencies)
        mean_latency = np.mean(latencies) if latencies else 0.0
        
        return AggregatedMetrics(
            mean_scores=mean_scores,
            std_scores=std_scores,
            min_scores=min_scores,
            max_scores=max_scores,
            total_prompts=len(prompt_results),
            successful_prompts=len(successful),
            failed_prompts=len(prompt_results) - len(successful),
            total_tokens=total_tokens,
            total_latency_ms=total_latency,
            mean_latency_ms=mean_latency,
        )
    
    def reset(self) -> None:
        """Reset tracked state for a new run."""
        self._chosen_subsets = []
        self._fragment_usage = Counter()
        self._stories = []
        self._trial_results = []
