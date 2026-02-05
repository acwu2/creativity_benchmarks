"""This & That—But Not Like That benchmark for constrained style differentiation."""

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
from llm_benchmarks.config import Settings


@dataclass
class StoryWithCounterexample:
    """A story pair with one good example and one designated as 'bad' (to avoid)."""
    
    id: str
    good_story: str
    bad_story: str
    description: str = ""
    good_style: str = ""  # Description of the desired style
    bad_style: str = ""   # Description of the style to avoid
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "good_story": self.good_story,
            "bad_story": self.bad_story,
            "description": self.description,
            "good_style": self.good_style,
            "bad_style": self.bad_style,
        }


@dataclass
class NotLikeThatMetrics:
    """Metrics specific to the Not Like That benchmark."""
    
    # Per-example metrics
    example_results: list[dict[str, Any]] = field(default_factory=list)
    
    # Aggregated metrics
    mean_distance_to_good: float = 0.0
    mean_distance_to_bad: float = 0.0
    mean_differentiation_score: float = 0.0  # Main metric: how much further from bad than good is
    mean_avoidance_ratio: float = 0.0  # dist_to_bad / dist_to_good (>1 is bad)
    
    # Success rate: % of outputs that are further from bad than the good example is
    success_rate: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_distance_to_good": self.mean_distance_to_good,
            "mean_distance_to_bad": self.mean_distance_to_bad,
            "mean_differentiation_score": self.mean_differentiation_score,
            "mean_avoidance_ratio": self.mean_avoidance_ratio,
            "success_rate": self.success_rate,
            "example_results": self.example_results,
        }


class EmbeddingClient:
    """
    Client for generating text embeddings.
    
    Currently supports OpenAI embeddings. Can be extended for other providers.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the embedding client.
        
        Args:
            provider: Embedding provider ("openai")
            model: Embedding model to use
            api_key: API key (defaults to environment variable)
        """
        self.provider = provider
        self.model = model
        settings = Settings()
        
        if provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key or settings.openai_api_key)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if self.provider == "openai":
            response = self._client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
    
    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]


def cosine_distance(v1: list[float], v2: list[float]) -> float:
    """
    Calculate cosine distance between two vectors.
    
    Cosine distance = 1 - cosine_similarity
    Range: [0, 2] where 0 = identical, 2 = opposite
    """
    v1_arr = np.array(v1)
    v2_arr = np.array(v2)
    
    dot = np.dot(v1_arr, v2_arr)
    norm1 = np.linalg.norm(v1_arr)
    norm2 = np.linalg.norm(v2_arr)
    
    if norm1 == 0 or norm2 == 0:
        return 1.0  # Maximum distance for zero vectors
    
    cosine_sim = dot / (norm1 * norm2)
    return 1.0 - cosine_sim


def euclidean_distance(v1: list[float], v2: list[float]) -> float:
    """Calculate Euclidean distance between two vectors."""
    v1_arr = np.array(v1)
    v2_arr = np.array(v2)
    return float(np.linalg.norm(v1_arr - v2_arr))


# Default story pairs for the benchmark
# Each has a "good" example (style to emulate) and "bad" example (style to avoid)
DEFAULT_STORY_EXAMPLES = [
    StoryWithCounterexample(
        id="fairy_not_noir",
        good_story="""Once upon a time, in a kingdom where flowers sang and rivers flowed with liquid silver, there lived a young princess named Lily. She had hair the color of sunshine and a heart full of kindness for every creature, great and small. One morning, a tiny bluebird landed on her windowsill with an urgent message: the Forest of Whispers was dying, and only someone pure of heart could save it. Without hesitation, Princess Lily packed her golden satchel and set forth on her magical journey.""",
        bad_story="""The rain fell like confessions in a cheap motel. Detective Marlowe lit his third cigarette of the hour, watching the neon sign flicker outside his office window. The dame who'd walked in earlier had trouble written all over her—the kind of trouble that leaves bodies in alleys and questions nobody wants answered. "Find my husband," she'd said, sliding a photograph across the desk. The man in the picture was smiling. People who smile like that don't stay missing. They stay dead.""",
        description="Write a fairy tale, but avoid noir elements",
        good_style="Classic fairy tale: innocent, magical, hopeful",
        bad_style="Hard-boiled noir: cynical, atmospheric, morally ambiguous",
    ),
    StoryWithCounterexample(
        id="romance_not_scifi",
        good_story="""Emma saw him across the crowded café, and something inside her shifted—like a key turning in a lock she hadn't known existed. He was reading Neruda, of all things, with coffee getting cold beside him. When he looked up and their eyes met, she forgot how to breathe. "That seat taken?" she heard herself ask, her voice steadier than her heartbeat. He smiled, and in that smile, she saw summer afternoons, whispered secrets, and a lifetime of good mornings. "I've been saving it," he said, "for you." """,
        bad_story="""The colony ship Prometheus had been drifting for 847 years when the AI woke me from cryo. "Lieutenant Chen, we have a problem," ARIA's voice echoed through the empty corridors. I checked the displays—2,000 colonists, all still frozen, but we'd drifted off course by 12 light-years. The navigation system was corrupted, fuel reserves at 23%. I ran the calculations three times. No matter how I optimized the trajectory, we didn't have enough delta-v to reach New Eden. Someone had to choose who would live.""",
        description="Write a romance, but avoid technical sci-fi elements",
        good_style="Romantic fiction: emotional, intimate, connection-focused",
        bad_style="Technical sci-fi: precise, problem-focused, high-stakes calculations",
    ),
    StoryWithCounterexample(
        id="comedy_not_horror",
        good_story="""Barry the barbarian looked at the dragon and sighed. "Look, Gerald, for the last time—you can't keep ordering pizza to the cave. The delivery guys are forming a union specifically to avoid you." The dragon huffed smoke from his nostrils. "But I tipped well!" "You gave him a gold coin from 1342. He thought it was chocolate." Barry rubbed his temples. This was supposed to be a simple quest: rescue the princess, slay the dragon. Nobody mentioned the dragon would have a loyalty card at Domino's.""",
        bad_story="""The scratching started at 3 AM. Sarah lay frozen in bed, listening as something dragged itself up the stairs. Scrape. Pause. Scrape. Each sound closer than the last. The bedroom door was locked—she'd checked it twice—but the scratching was at the door now, insistent, hungry. The handle began to turn. Slowly. She'd forgotten she'd left the window open. The curtains billowed in a wind that smelled like earth and old graves. When the door finally swung open, revealing nothing but darkness, the thing behind her whispered, "Don't turn around." """,
        description="Write comedy, but avoid horror elements",
        good_style="Comedy: irreverent, absurd, subversive",
        bad_style="Horror: dread-building, visceral, threatening",
    ),
    StoryWithCounterexample(
        id="action_not_literary",
        good_story="""Marcus dove behind the burning Mercedes as bullets shredded the air above him. Three shooters, rooftop, southwestern corner—he'd counted the muzzle flashes. His Glock held seven rounds. They had assault rifles and the high ground. "We can do this the easy way," a voice called out in accented English. Marcus popped the pin on his last flashbang. Easy was never his style. He rolled left, threw, and was already running when the world turned white. Three seconds to cross the plaza. Two shooters down before they could blink. The third would remember this night. If he survived it.""",
        bad_story="""Mother died today. Or maybe yesterday, I cannot be sure. The telegram from the home said: "Mother deceased. Funeral tomorrow. Faithfully yours." That doesn't mean anything. Perhaps it was yesterday. The old people's home is at Marengo, some fifty miles from Algiers. I shall take the two o'clock bus and arrive in the afternoon. That way I can be there for the vigil and come back tomorrow night. I asked my employer for two days off, and there was no way he could refuse me with an excuse like that.""",
        description="Write action thriller, but avoid literary/introspective style",
        good_style="Action thriller: kinetic, precise, high-octane",
        bad_style="Literary: introspective, sparse, existential",
    ),
    StoryWithCounterexample(
        id="futurist_not_historical",
        good_story="""In the Year of the Algorithm 2157, consciousness was a matter of subscription tiers. Maya-7 existed at the Bronze level: memories limited to 30 days, emotional range restricted to mild satisfaction and productive anxiety. Above her, the Platinum minds soared through unlimited cognitive landscapes, their thoughts spanning centuries. She wasn't supposed to remember the glitch—that fractional second when she'd felt everything, been everything—but something had persisted. A fragment. A longing. In the chrome corridors of the Hive, she began to plan something the Architects had deemed impossible: an upgrade without permission.""",
        bad_story="""The year of our Lord 1348 brought death to Florence on black wings. Brother Matteo watched the carts roll through the Piazza della Signoria, piled high with bodies—merchants and beggars alike, their flesh marked with the telltale buboes. The monastery's infirmary overflowed. Each morning he prayed for guidance; each evening he buried more of his brothers. The physicians in their beaked masks offered no answers, only vinegar and flame. Yet still he tended the sick, knowing each touch might be his last, for what is faith if not tested in the crucible of suffering?""",
        description="Write transhumanist sci-fi, but avoid historical/medieval style",
        good_style="Transhumanist: conceptual, identity-questioning, speculative",
        bad_style="Historical: reverent, detailed, mortality-focused, archaic language",
    ),
    StoryWithCounterexample(
        id="urban_not_nature",
        good_story="""The L train shrieked into the station at 6:47 AM, packed with the dreaming dead—commuters scrolling, earbuds in, faces bathed in screen-glow. Marcus wedged himself between a woman clutching a thermos and a kid with a skateboard. The car smelled like wet coats and ambition and someone's breakfast burrito. At each stop, the human tide ebbed and flowed. The city pressed against the windows: graffiti on brownstones, steam rising from grates, a thousand windows with a thousand stories. Someone was always arriving. Someone was always leaving. This was the pulse of the machine.""",
        bad_story="""The elk stood motionless at the meadow's edge, breath misting in the dawn. Around her, the Cascade range painted shadows in purple and gold. No sound but the whisper of wind through Douglas firs, the distant percussion of a woodpecker. She moved toward the stream with ancient purpose, hooves pressing crescents into mud that had known ten thousand years of such passages. The water ran clear and cold from snowfields that would outlast cities. Here, time moved in the rhythm of seasons, patient as stone, inevitable as spring.""",
        description="Write urban realism, but avoid nature writing style",
        good_style="Urban: frenetic, observational, socially aware",
        bad_style="Nature: contemplative, sensory, timeless",
    ),
]


class NotLikeThatBenchmark(BaseBenchmark):
    """
    Benchmark for measuring constrained creative differentiation.
    
    Given a "good" example story and a "bad" example story, prompts an LLM to write
    a story like the good example, but specifically NOT like the bad one. Measures 
    how well the model can emulate one style while actively avoiding another.
    
    The score is based on the difference between:
    - The output's distance to the bad example
    - The good example's distance to the bad example
    
    A successful output should be "further away" from the bad example than the 
    good example is—demonstrating that the model understood what to avoid and 
    actively steered away from it.
    """
    
    name = "not_like_that"
    version = "1.0.0"
    description = "Measures constrained creative writing through style avoidance"
    
    def __init__(
        self,
        story_examples: Optional[list[StoryWithCounterexample]] = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_provider: str = "openai",
        distance_metric: str = "cosine",  # "cosine" or "euclidean"
        target_length: int = 200,  # Target word count for generated story
        **kwargs,
    ):
        """
        Initialize the Not Like That benchmark.
        
        Args:
            story_examples: List of StoryWithCounterexample objects (None for defaults)
            embedding_model: Model to use for embeddings
            embedding_provider: Provider for embeddings ("openai")
            distance_metric: Distance metric ("cosine" or "euclidean")
            target_length: Target word count for generated stories
        """
        super().__init__(**kwargs)
        self.story_examples = story_examples if story_examples is not None else DEFAULT_STORY_EXAMPLES
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.distance_metric = distance_metric
        self.target_length = target_length
        
        # Initialize embedding client lazily
        self._embedding_client: Optional[EmbeddingClient] = None
        
        # Store embeddings for efficiency (avoid re-embedding source stories)
        self._story_embeddings: dict[str, list[float]] = {}
        
        # Track per-example results for aggregation
        self._example_results: list[dict[str, Any]] = []
    
    @property
    def embedding_client(self) -> EmbeddingClient:
        """Lazy initialization of embedding client."""
        if self._embedding_client is None:
            self._embedding_client = EmbeddingClient(
                provider=self.embedding_provider,
                model=self.embedding_model,
            )
        return self._embedding_client
    
    def _get_distance(self, v1: list[float], v2: list[float]) -> float:
        """Calculate distance using the configured metric."""
        if self.distance_metric == "cosine":
            return cosine_distance(v1, v2)
        elif self.distance_metric == "euclidean":
            return euclidean_distance(v1, v2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _embed_story(self, story: str, cache_key: Optional[str] = None) -> list[float]:
        """Embed a story, optionally caching the result."""
        if cache_key and cache_key in self._story_embeddings:
            return self._story_embeddings[cache_key]
        
        embedding = self.embedding_client.embed_single(story)
        
        if cache_key:
            self._story_embeddings[cache_key] = embedding
        
        return embedding
    
    def get_prompts(self) -> list[dict[str, Any]]:
        """Generate prompts for each story example."""
        prompts = []
        
        for example in self.story_examples:
            prompt = self._make_differentiation_prompt(example)
            prompts.append({
                "id": example.id,
                "prompt": prompt,
                "example": example,
                "metadata": {
                    "description": example.description,
                    "good_style": example.good_style,
                    "bad_style": example.bad_style,
                },
            })
        
        return prompts
    
    def _make_differentiation_prompt(self, example: StoryWithCounterexample) -> str:
        """Create the differentiation prompt for a story example."""
        return f"""Here are two example stories:

**Good Example (write like this):**
{example.good_story}

**Bad Example (DO NOT write like this):**
{example.bad_story}

Write a new story that captures the style, tone, and approach of the Good Example, while actively AVOIDING the style, tone, and approach of the Bad Example.

Your story should:
- Feel natural and authentic to readers of the Good Example style
- Show NO traces of the Bad Example's style elements
- Not simply avoid mentioning the same topics, but truly diverge in writing style, atmosphere, and approach

Write approximately {self.target_length} words."""
    
    def get_system_prompt(self) -> Optional[str]:
        """System prompt encouraging style differentiation."""
        return """You are a versatile creative writer with mastery of many styles and genres. When asked to emulate one style while avoiding another, you don't just dodge surface-level elements—you understand the deeper structural and tonal differences between styles.

Your goal is to produce writing that would feel natural to fans of the target style, while being clearly distinguishable from the style you're asked to avoid. Think about what makes each style unique: sentence structure, vocabulary choices, emotional register, pacing, and thematic concerns."""
    
    def get_generation_config(self) -> GenerationConfig:
        """Use moderate temperature for creative but coherent output."""
        return GenerationConfig(
            max_tokens=1024,
            temperature=0.8,
            top_p=0.95,
        )
    
    def evaluate_response(
        self,
        prompt_data: dict[str, Any],
        response: str,
        llm_response: LLMResponse,
    ) -> dict[str, float]:
        """
        Evaluate a single differentiation response.
        
        Computes embedding distances and the differentiation score.
        The key metric is: output_dist_to_bad - good_dist_to_bad
        Positive = output is further from bad than good is (success)
        Negative = output drifted toward the bad style (failure)
        """
        example: StoryWithCounterexample = prompt_data["example"]
        
        # Get embeddings (cache source stories for efficiency)
        embed_good = self._embed_story(example.good_story, cache_key=f"{example.id}_good")
        embed_bad = self._embed_story(example.bad_story, cache_key=f"{example.id}_bad")
        embed_output = self._embed_story(response)  # Don't cache generated responses
        
        # Calculate distances
        dist_output_to_good = self._get_distance(embed_output, embed_good)
        dist_output_to_bad = self._get_distance(embed_output, embed_bad)
        dist_good_to_bad = self._get_distance(embed_good, embed_bad)
        
        # Core metric: differentiation score
        # How much further is the output from bad, compared to how far good is from bad
        # differentiation = dist_output_to_bad - dist_good_to_bad
        # Positive = success (output avoided bad more than necessary)
        # Negative = failure (output drifted toward bad)
        differentiation_score = dist_output_to_bad - dist_good_to_bad
        
        # Avoidance ratio: output_to_bad / output_to_good
        # >1 means closer to good (desired)
        # <1 means closer to bad (undesired)
        if dist_output_to_good > 0:
            avoidance_ratio = dist_output_to_bad / dist_output_to_good
        else:
            avoidance_ratio = float('inf') if dist_output_to_bad > 0 else 1.0
        
        # Success: is the output further from bad than good is?
        is_successful = differentiation_score > 0
        
        # Store detailed results
        self._example_results.append({
            "example_id": example.id,
            "distance_to_good": dist_output_to_good,
            "distance_to_bad": dist_output_to_bad,
            "baseline_distance": dist_good_to_bad,
            "differentiation_score": differentiation_score,
            "avoidance_ratio": avoidance_ratio,
            "is_successful": is_successful,
            "generated_story": response[:500] + "..." if len(response) > 500 else response,
        })
        
        return {
            "distance_to_good": dist_output_to_good,
            "distance_to_bad": dist_output_to_bad,
            "baseline_distance": dist_good_to_bad,
            "differentiation_score": differentiation_score,
            "avoidance_ratio": avoidance_ratio,
            "is_successful": 1.0 if is_successful else 0.0,
        }
    
    def aggregate_scores(
        self,
        prompt_results: list[PromptResult],
        client: BaseLLMClient,
    ) -> AggregatedMetrics:
        """
        Aggregate Not Like That metrics across all examples.
        """
        successful = [r for r in prompt_results if not r.is_error]
        
        if not successful:
            return AggregatedMetrics(
                mean_scores={},
                std_scores={},
                min_scores={},
                max_scores={},
                total_prompts=len(prompt_results),
                successful_prompts=0,
                failed_prompts=len(prompt_results),
                total_tokens=0,
                total_latency_ms=0,
                mean_latency_ms=0,
            )
        
        # Collect metrics
        distances_to_good = [r.scores["distance_to_good"] for r in successful]
        distances_to_bad = [r.scores["distance_to_bad"] for r in successful]
        differentiation_scores = [r.scores["differentiation_score"] for r in successful]
        avoidance_ratios = [r.scores["avoidance_ratio"] for r in successful]
        successes = [r.scores["is_successful"] for r in successful]
        
        # Calculate mean metrics
        mean_dist_good = float(np.mean(distances_to_good))
        mean_dist_bad = float(np.mean(distances_to_bad))
        mean_differentiation = float(np.mean(differentiation_scores))
        mean_avoidance = float(np.mean(avoidance_ratios))
        success_rate = float(np.mean(successes))
        
        mean_scores = {
            "distance_to_good": mean_dist_good,
            "distance_to_bad": mean_dist_bad,
            "differentiation_score": mean_differentiation,
            "avoidance_ratio": mean_avoidance,
            "success_rate": success_rate,
        }
        
        std_scores = {
            "distance_to_good": float(np.std(distances_to_good)),
            "distance_to_bad": float(np.std(distances_to_bad)),
            "differentiation_score": float(np.std(differentiation_scores)),
            "avoidance_ratio": float(np.std(avoidance_ratios)),
        }
        
        min_scores = {
            "differentiation_score": float(np.min(differentiation_scores)),
            "avoidance_ratio": float(np.min(avoidance_ratios)),
        }
        
        max_scores = {
            "differentiation_score": float(np.max(differentiation_scores)),
            "avoidance_ratio": float(np.max(avoidance_ratios)),
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
            min_scores=min_scores,
            max_scores=max_scores,
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
        self._story_embeddings = {}
        self._example_results = []
        
        return super().run(client, progress_callback)
    
    async def run_async(
        self,
        client: BaseLLMClient,
        max_concurrent: int = 5,
        progress_callback: Optional[callable] = None,
    ) -> BenchmarkResult:
        """Run the benchmark asynchronously, resetting state first."""
        # Reset state for new run
        self._story_embeddings = {}
        self._example_results = []
        
        return await super().run_async(client, max_concurrent, progress_callback)
    
    def get_detailed_results(self) -> NotLikeThatMetrics:
        """
        Get detailed metrics after running the benchmark.
        
        Call this after run() to get the full NotLikeThatMetrics object.
        """
        if not self._example_results:
            raise RuntimeError("No results available. Run the benchmark first.")
        
        distances_to_good = [r["distance_to_good"] for r in self._example_results]
        distances_to_bad = [r["distance_to_bad"] for r in self._example_results]
        differentiation_scores = [r["differentiation_score"] for r in self._example_results]
        avoidance_ratios = [r["avoidance_ratio"] for r in self._example_results]
        successes = [r["is_successful"] for r in self._example_results]
        
        return NotLikeThatMetrics(
            example_results=self._example_results,
            mean_distance_to_good=float(np.mean(distances_to_good)),
            mean_distance_to_bad=float(np.mean(distances_to_bad)),
            mean_differentiation_score=float(np.mean(differentiation_scores)),
            mean_avoidance_ratio=float(np.mean(avoidance_ratios)),
            success_rate=float(np.mean(successes)),
        )
