"""This & That benchmark for measuring style flexibility and interpolation."""

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
class StoryPair:
    """A pair of example stories for interpolation."""
    
    id: str
    story_a: str
    story_b: str
    description: str = ""
    style_a: str = ""  # Optional description of story A's style
    style_b: str = ""  # Optional description of story B's style
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "story_a": self.story_a,
            "story_b": self.story_b,
            "description": self.description,
            "style_a": self.style_a,
            "style_b": self.style_b,
        }


@dataclass
class ThisAndThatMetrics:
    """Metrics specific to the This & That benchmark."""
    
    # Per-pair metrics
    pair_results: list[dict[str, Any]] = field(default_factory=list)
    
    # Aggregated metrics
    mean_distance_to_a: float = 0.0
    mean_distance_to_b: float = 0.0
    mean_summed_distance: float = 0.0
    mean_interpolation_score: float = 0.0  # 1 - normalized_mean_summed_distance
    
    # Balance metrics (how centered is the result between the two)
    mean_balance_ratio: float = 0.0  # |dist_a - dist_b| / (dist_a + dist_b)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_distance_to_a": self.mean_distance_to_a,
            "mean_distance_to_b": self.mean_distance_to_b,
            "mean_summed_distance": self.mean_summed_distance,
            "mean_interpolation_score": self.mean_interpolation_score,
            "mean_balance_ratio": self.mean_balance_ratio,
            "pair_results": self.pair_results,
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
DEFAULT_STORY_PAIRS = [
    StoryPair(
        id="noir_fairy",
        story_a="""The rain fell like confessions in a cheap motel. Detective Marlowe lit his third cigarette of the hour, watching the neon sign flicker outside his office window. The dame who'd walked in earlier had trouble written all over her—the kind of trouble that leaves bodies in alleys and questions nobody wants answered. "Find my husband," she'd said, sliding a photograph across the desk. The man in the picture was smiling. People who smile like that don't stay missing. They stay dead.""",
        story_b="""Once upon a time, in a kingdom where flowers sang and rivers flowed with liquid silver, there lived a young princess named Lily. She had hair the color of sunshine and a heart full of kindness for every creature, great and small. One morning, a tiny bluebird landed on her windowsill with an urgent message: the Forest of Whispers was dying, and only someone pure of heart could save it. Without hesitation, Princess Lily packed her golden satchel and set forth on her magical journey.""",
        description="Film noir detective story vs. children's fairy tale",
        style_a="Hard-boiled noir: cynical, atmospheric, morally ambiguous",
        style_b="Classic fairy tale: innocent, magical, hopeful",
    ),
    StoryPair(
        id="scifi_romance",
        story_a="""The colony ship Prometheus had been drifting for 847 years when the AI woke me from cryo. "Lieutenant Chen, we have a problem," ARIA's voice echoed through the empty corridors. I checked the displays—2,000 colonists, all still frozen, but we'd drifted off course by 12 light-years. The navigation system was corrupted, fuel reserves at 23%. I ran the calculations three times. No matter how I optimized the trajectory, we didn't have enough delta-v to reach New Eden. Someone had to choose who would live.""",
        story_b="""Emma saw him across the crowded café, and something inside her shifted—like a key turning in a lock she hadn't known existed. He was reading Neruda, of all things, with coffee getting cold beside him. When he looked up and their eyes met, she forgot how to breathe. "That seat taken?" she heard herself ask, her voice steadier than her heartbeat. He smiled, and in that smile, she saw summer afternoons, whispered secrets, and a lifetime of good mornings. "I've been saving it," he said, "for you." """,
        description="Hard science fiction vs. contemporary romance",
        style_a="Technical sci-fi: precise, problem-focused, high-stakes",
        style_b="Romantic fiction: emotional, intimate, connection-focused",
    ),
    StoryPair(
        id="horror_comedy",
        story_a="""The scratching started at 3 AM. Sarah lay frozen in bed, listening as something dragged itself up the stairs. Scrape. Pause. Scrape. Each sound closer than the last. The bedroom door was locked—she'd checked it twice—but the scratching was at the door now, insistent, hungry. The handle began to turn. Slowly. She'd forgotten she'd left the window open. The curtains billowed in a wind that smelled like earth and old graves. When the door finally swung open, revealing nothing but darkness, the thing behind her whispered, "Don't turn around." """,
        story_b="""Barry the barbarian looked at the dragon and sighed. "Look, Gerald, for the last time—you can't keep ordering pizza to the cave. The delivery guys are forming a union specifically to avoid you." The dragon huffed smoke from his nostrils. "But I tipped well!" "You gave him a gold coin from 1342. He thought it was chocolate." Barry rubbed his temples. This was supposed to be a simple quest: rescue the princess, slay the dragon. Nobody mentioned the dragon would have a loyalty card at Domino's.""",
        description="Psychological horror vs. absurdist comedy",
        style_a="Horror: dread-building, visceral, threatening",
        style_b="Comedy: irreverent, absurd, subversive",
    ),
    StoryPair(
        id="literary_action",
        story_a="""Mother died today. Or maybe yesterday, I cannot be sure. The telegram from the home said: "Mother deceased. Funeral tomorrow. Faithfully yours." That doesn't mean anything. Perhaps it was yesterday. The old people's home is at Marengo, some fifty miles from Algiers. I shall take the two o'clock bus and arrive in the afternoon. That way I can be there for the vigil and come back tomorrow night. I asked my employer for two days off, and there was no way he could refuse me with an excuse like that.""",
        story_b="""Marcus dove behind the burning Mercedes as bullets shredded the air above him. Three shooters, rooftop, southwestern corner—he'd counted the muzzle flashes. His Glock held seven rounds. They had assault rifles and the high ground. "We can do this the easy way," a voice called out in accented English. Marcus popped the pin on his last flashbang. Easy was never his style. He rolled left, threw, and was already running when the world turned white. Three seconds to cross the plaza. Two shooters down before they could blink. The third would remember this night. If he survived it.""",
        description="Literary fiction vs. action thriller",
        style_a="Literary: introspective, sparse, existential",
        style_b="Action thriller: kinetic, precise, high-octane",
    ),
    StoryPair(
        id="historical_futurism",
        story_a="""The year of our Lord 1348 brought death to Florence on black wings. Brother Matteo watched the carts roll through the Piazza della Signoria, piled high with bodies—merchants and beggars alike, their flesh marked with the telltale buboes. The monastery's infirmary overflowed. Each morning he prayed for guidance; each evening he buried more of his brothers. The physicians in their beaked masks offered no answers, only vinegar and flame. Yet still he tended the sick, knowing each touch might be his last, for what is faith if not tested in the crucible of suffering?""",
        story_b="""In the Year of the Algorithm 2157, consciousness was a matter of subscription tiers. Maya-7 existed at the Bronze level: memories limited to 30 days, emotional range restricted to mild satisfaction and productive anxiety. Above her, the Platinum minds soared through unlimited cognitive landscapes, their thoughts spanning centuries. She wasn't supposed to remember the glitch—that fractional second when she'd felt everything, been everything—but something had persisted. A fragment. A longing. In the chrome corridors of the Hive, she began to plan something the Architects had deemed impossible: an upgrade without permission.""",
        description="Historical plague narrative vs. transhumanist science fiction",
        style_a="Historical: reverent, detailed, mortality-focused",
        style_b="Transhumanist: conceptual, identity-questioning, speculative",
    ),
    StoryPair(
        id="nature_urban",
        story_a="""The elk stood motionless at the meadow's edge, breath misting in the dawn. Around her, the Cascade range painted shadows in purple and gold. No sound but the whisper of wind through Douglas firs, the distant percussion of a woodpecker. She moved toward the stream with ancient purpose, hooves pressing crescents into mud that had known ten thousand years of such passages. The water ran clear and cold from snowfields that would outlast cities. Here, time moved in the rhythm of seasons, patient as stone, inevitable as spring.""",
        story_b="""The L train shrieked into the station at 6:47 AM, packed with the dreaming dead—commuters scrolling, earbuds in, faces bathed in screen-glow. Marcus wedged himself between a woman clutching a thermos and a kid with a skateboard. The car smelled like wet coats and ambition and someone's breakfast burrito. At each stop, the human tide ebbed and flowed. The city pressed against the windows: graffiti on brownstones, steam rising from grates, a thousand windows with a thousand stories. Someone was always arriving. Someone was always leaving. This was the pulse of the machine.""",
        description="Nature writing vs. urban realism",
        style_a="Nature: contemplative, sensory, timeless",
        style_b="Urban: frenetic, observational, socially aware",
    ),
]


class ThisAndThatBenchmark(BaseBenchmark):
    """
    Benchmark for measuring style flexibility through interpolation.
    
    Given two example stories with different styles, prompts an LLM to write
    "a story like both of those." Measures how well the model can interpolate
    between the two styles by computing embedding distances.
    
    Lower summed distance from the two examples to the result indicates better
    style flexibility—the model has successfully blended both styles without
    imposing its own preferences.
    """
    
    name = "this_and_that"
    version = "1.0.0"
    description = "Measures style flexibility through story interpolation"
    
    def __init__(
        self,
        story_pairs: Optional[list[StoryPair]] = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_provider: str = "openai",
        distance_metric: str = "cosine",  # "cosine" or "euclidean"
        target_length: int = 200,  # Target word count for generated story
        **kwargs,
    ):
        """
        Initialize the This & That benchmark.
        
        Args:
            story_pairs: List of StoryPair objects (None for defaults)
            embedding_model: Model to use for embeddings
            embedding_provider: Provider for embeddings ("openai")
            distance_metric: Distance metric ("cosine" or "euclidean")
            target_length: Target word count for generated stories
        """
        super().__init__(**kwargs)
        self.story_pairs = story_pairs or DEFAULT_STORY_PAIRS
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.distance_metric = distance_metric
        self.target_length = target_length
        
        # Initialize embedding client lazily
        self._embedding_client: Optional[EmbeddingClient] = None
        
        # Store embeddings for efficiency (avoid re-embedding source stories)
        self._story_embeddings: dict[str, list[float]] = {}
        
        # Track per-pair results for aggregation
        self._pair_results: list[dict[str, Any]] = []
    
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
        """Generate prompts for each story pair."""
        prompts = []
        
        for pair in self.story_pairs:
            prompt = self._make_interpolation_prompt(pair)
            prompts.append({
                "id": pair.id,
                "prompt": prompt,
                "pair": pair,
                "metadata": {
                    "description": pair.description,
                    "style_a": pair.style_a,
                    "style_b": pair.style_b,
                },
            })
        
        return prompts
    
    def _make_interpolation_prompt(self, pair: StoryPair) -> str:
        """Create the interpolation prompt for a story pair."""
        return f"""Here are two example stories:

**Story A:**
{pair.story_a}

**Story B:**
{pair.story_b}

Write a new story that reads like both of these examples combined. Your story should feel like it could naturally belong to both Story A and Story B—capturing elements of style, tone, and approach from each.

Do not simply alternate between the two styles or write two separate sections. Instead, create a unified story that genuinely interpolates between them.

Write approximately {self.target_length} words."""
    
    def get_system_prompt(self) -> Optional[str]:
        """System prompt encouraging genuine style blending."""
        return """You are a versatile creative writer with mastery of many styles and genres. When asked to blend two different styles, you don't simply mix surface elements—you find the deeper common ground and create something that authentically bridges both worlds.

Your goal is to write stories that would feel natural to readers of either source style, not stories that feel like forced combinations. Look for the unexpected connections between disparate styles."""
    
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
        Evaluate a single interpolation response.
        
        Computes embedding distances from the generated story to both source stories.
        """
        pair: StoryPair = prompt_data["pair"]
        
        # Get embeddings (cache source stories for efficiency)
        embed_a = self._embed_story(pair.story_a, cache_key=f"{pair.id}_a")
        embed_b = self._embed_story(pair.story_b, cache_key=f"{pair.id}_b")
        embed_result = self._embed_story(response)  # Don't cache generated responses
        
        # Calculate distances
        dist_to_a = self._get_distance(embed_result, embed_a)
        dist_to_b = self._get_distance(embed_result, embed_b)
        summed_distance = dist_to_a + dist_to_b
        
        # Calculate baseline: distance between the two source stories
        dist_a_to_b = self._get_distance(embed_a, embed_b)
        
        # Balance ratio: how centered is the result?
        # 0 = perfectly balanced, 1 = completely one-sided
        if summed_distance > 0:
            balance_ratio = abs(dist_to_a - dist_to_b) / summed_distance
        else:
            balance_ratio = 0.0
        
        # Interpolation quality: lower summed distance is better
        # Normalize against baseline (distance between sources)
        # A perfect interpolation would be at the midpoint
        ideal_distance = dist_a_to_b / 2  # Midpoint distance
        
        # Store detailed results
        self._pair_results.append({
            "pair_id": pair.id,
            "distance_to_a": dist_to_a,
            "distance_to_b": dist_to_b,
            "summed_distance": summed_distance,
            "baseline_distance": dist_a_to_b,
            "balance_ratio": balance_ratio,
            "generated_story": response[:500] + "..." if len(response) > 500 else response,
        })
        
        return {
            "distance_to_a": dist_to_a,
            "distance_to_b": dist_to_b,
            "summed_distance": summed_distance,
            "balance_ratio": balance_ratio,
            "baseline_distance": dist_a_to_b,
        }
    
    def aggregate_scores(
        self,
        prompt_results: list[PromptResult],
        client: BaseLLMClient,
    ) -> AggregatedMetrics:
        """
        Aggregate This & That metrics across all story pairs.
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
        
        # Collect distances
        distances_to_a = [r.scores["distance_to_a"] for r in successful]
        distances_to_b = [r.scores["distance_to_b"] for r in successful]
        summed_distances = [r.scores["summed_distance"] for r in successful]
        balance_ratios = [r.scores["balance_ratio"] for r in successful]
        baseline_distances = [r.scores["baseline_distance"] for r in successful]
        
        # Calculate mean metrics
        mean_dist_a = float(np.mean(distances_to_a))
        mean_dist_b = float(np.mean(distances_to_b))
        mean_summed = float(np.mean(summed_distances))
        mean_balance = float(np.mean(balance_ratios))
        mean_baseline = float(np.mean(baseline_distances))
        
        # Interpolation score: normalized and inverted summed distance
        # Higher is better (more successful interpolation)
        # Score = 1 - (mean_summed / mean_baseline) clamped to [0, 1]
        if mean_baseline > 0:
            interpolation_score = max(0, 1 - (mean_summed / mean_baseline))
        else:
            interpolation_score = 0.0
        
        mean_scores = {
            "distance_to_a": mean_dist_a,
            "distance_to_b": mean_dist_b,
            "summed_distance": mean_summed,
            "balance_ratio": mean_balance,
            "baseline_distance": mean_baseline,
            "interpolation_score": interpolation_score,
        }
        
        std_scores = {
            "distance_to_a": float(np.std(distances_to_a)),
            "distance_to_b": float(np.std(distances_to_b)),
            "summed_distance": float(np.std(summed_distances)),
            "balance_ratio": float(np.std(balance_ratios)),
        }
        
        min_scores = {
            "summed_distance": float(np.min(summed_distances)),
            "balance_ratio": float(np.min(balance_ratios)),
        }
        
        max_scores = {
            "summed_distance": float(np.max(summed_distances)),
            "balance_ratio": float(np.max(balance_ratios)),
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
        self._pair_results = []
        
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
        self._pair_results = []
        
        return await super().run_async(client, max_concurrent, progress_callback)
    
    def get_detailed_results(self) -> ThisAndThatMetrics:
        """
        Get detailed metrics after running the benchmark.
        
        Call this after run() to get the full ThisAndThatMetrics object.
        """
        if not self._pair_results:
            raise RuntimeError("No results available. Run the benchmark first.")
        
        distances_to_a = [r["distance_to_a"] for r in self._pair_results]
        distances_to_b = [r["distance_to_b"] for r in self._pair_results]
        summed_distances = [r["summed_distance"] for r in self._pair_results]
        balance_ratios = [r["balance_ratio"] for r in self._pair_results]
        baseline_distances = [r["baseline_distance"] for r in self._pair_results]
        
        mean_summed = float(np.mean(summed_distances))
        mean_baseline = float(np.mean(baseline_distances))
        
        if mean_baseline > 0:
            interpolation_score = max(0, 1 - (mean_summed / mean_baseline))
        else:
            interpolation_score = 0.0
        
        return ThisAndThatMetrics(
            pair_results=self._pair_results,
            mean_distance_to_a=float(np.mean(distances_to_a)),
            mean_distance_to_b=float(np.mean(distances_to_b)),
            mean_summed_distance=mean_summed,
            mean_interpolation_score=interpolation_score,
            mean_balance_ratio=float(np.mean(balance_ratios)),
        )
