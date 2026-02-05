"""Base benchmark class and result types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import json

from llm_benchmarks.clients.base import BaseLLMClient, GenerationConfig, LLMResponse


@dataclass
class PromptResult:
    """Result for a single prompt evaluation."""
    prompt_id: str
    prompt: str
    response: str
    scores: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    llm_response: Optional[LLMResponse] = None
    error: Optional[str] = None
    
    @property
    def is_error(self) -> bool:
        return self.error is not None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "response": self.response,
            "scores": self.scores,
            "metadata": self.metadata,
            "llm_response": self.llm_response.to_dict() if self.llm_response else None,
            "error": self.error,
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across all prompts."""
    mean_scores: dict[str, float]
    std_scores: dict[str, float]
    min_scores: dict[str, float]
    max_scores: dict[str, float]
    total_prompts: int
    successful_prompts: int
    failed_prompts: int
    total_tokens: int
    total_latency_ms: float
    mean_latency_ms: float
    total_cost_usd: Optional[float] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "mean_scores": self.mean_scores,
            "std_scores": self.std_scores,
            "min_scores": self.min_scores,
            "max_scores": self.max_scores,
            "total_prompts": self.total_prompts,
            "successful_prompts": self.successful_prompts,
            "failed_prompts": self.failed_prompts,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "mean_latency_ms": self.mean_latency_ms,
            "total_cost_usd": self.total_cost_usd,
        }


@dataclass
class BenchmarkResult:
    """Complete result for a benchmark run."""
    benchmark_name: str
    benchmark_version: str
    model: str
    provider: str
    prompt_results: list[PromptResult]
    metrics: AggregatedMetrics
    config: dict[str, Any]
    started_at: datetime
    completed_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Total benchmark duration in seconds."""
        return (self.completed_at - self.started_at).total_seconds()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "benchmark_name": self.benchmark_name,
            "benchmark_version": self.benchmark_version,
            "model": self.model,
            "provider": self.provider,
            "prompt_results": [r.to_dict() for r in self.prompt_results],
            "metrics": self.metrics.to_dict(),
            "config": self.config,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }
    
    def save(self, path: Path | str) -> None:
        """Save results to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path | str) -> "BenchmarkResult":
        """Load results from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        # Reconstruct the dataclasses
        prompt_results = [
            PromptResult(
                prompt_id=r["prompt_id"],
                prompt=r["prompt"],
                response=r["response"],
                scores=r["scores"],
                metadata=r.get("metadata", {}),
                error=r.get("error"),
            )
            for r in data["prompt_results"]
        ]
        
        metrics = AggregatedMetrics(**data["metrics"])
        
        return cls(
            benchmark_name=data["benchmark_name"],
            benchmark_version=data["benchmark_version"],
            model=data["model"],
            provider=data["provider"],
            prompt_results=prompt_results,
            metrics=metrics,
            config=data["config"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]),
            metadata=data.get("metadata", {}),
        )


class BaseBenchmark(ABC):
    """
    Abstract base class for LLM benchmarks.
    
    Subclasses must implement:
    - get_prompts(): Return the list of prompts to evaluate
    - evaluate_response(): Score a single response
    
    Optional overrides:
    - get_system_prompt(): Provide a system prompt
    - get_generation_config(): Customize generation parameters
    - aggregate_scores(): Custom aggregation logic
    """
    
    # Benchmark metadata (override in subclasses)
    name: str = "base_benchmark"
    version: str = "1.0.0"
    description: str = "Base benchmark class"
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the benchmark.
        
        Args:
            system_prompt: Override the default system prompt
            generation_config: Override the default generation config
            metadata: Additional metadata to include in results
        """
        self._system_prompt = system_prompt
        self._generation_config = generation_config
        self.metadata = metadata or {}
    
    @abstractmethod
    def get_prompts(self) -> list[dict[str, Any]]:
        """
        Get the list of prompts to evaluate.
        
        Returns:
            List of prompt dictionaries with at least:
            - id: Unique identifier for the prompt
            - prompt: The prompt text
            - metadata: Optional additional data
            
        Example:
            return [
                {"id": "1", "prompt": "What is 2+2?", "expected": "4"},
                {"id": "2", "prompt": "What is the capital of France?", "expected": "Paris"},
            ]
        """
        pass
    
    @abstractmethod
    def evaluate_response(
        self,
        prompt_data: dict[str, Any],
        response: str,
        llm_response: LLMResponse,
    ) -> dict[str, float]:
        """
        Evaluate a single response and return scores.
        
        Args:
            prompt_data: The original prompt dictionary from get_prompts()
            response: The LLM's response text
            llm_response: The full LLMResponse object
            
        Returns:
            Dictionary of score names to values (typically 0-1 or 0-100)
            
        Example:
            return {
                "accuracy": 1.0 if response.strip() == prompt_data["expected"] else 0.0,
                "length_score": min(len(response) / 100, 1.0),
            }
        """
        pass
    
    def get_system_prompt(self) -> Optional[str]:
        """
        Get the system prompt for this benchmark.
        
        Override this method to provide a custom system prompt.
        Returns None by default (no system prompt).
        """
        return self._system_prompt
    
    def get_generation_config(self) -> GenerationConfig:
        """
        Get the generation config for this benchmark.
        
        Override this method to customize generation parameters.
        """
        if self._generation_config:
            return self._generation_config
        
        return GenerationConfig(
            max_tokens=2048,
            temperature=0.7,
        )
    
    def preprocess_prompt(self, prompt_data: dict[str, Any]) -> str:
        """
        Preprocess the prompt before sending to the LLM.
        
        Override this method to add formatting, instructions, etc.
        
        Args:
            prompt_data: The prompt dictionary from get_prompts()
            
        Returns:
            The processed prompt string
        """
        return prompt_data["prompt"]
    
    def postprocess_response(self, response: str) -> str:
        """
        Postprocess the LLM response before evaluation.
        
        Override this method to clean/extract relevant parts of the response.
        
        Args:
            response: The raw LLM response
            
        Returns:
            The processed response string
        """
        return response
    
    def aggregate_scores(
        self,
        prompt_results: list[PromptResult],
        client: BaseLLMClient,
    ) -> AggregatedMetrics:
        """
        Aggregate scores across all prompt results.
        
        Override this method for custom aggregation logic.
        
        Args:
            prompt_results: List of individual prompt results
            client: The LLM client used (for cost calculation)
            
        Returns:
            Aggregated metrics
        """
        import numpy as np
        
        successful = [r for r in prompt_results if not r.is_error]
        failed = [r for r in prompt_results if r.is_error]
        
        # Collect all scores
        all_scores: dict[str, list[float]] = {}
        for result in successful:
            for key, value in result.scores.items():
                if key not in all_scores:
                    all_scores[key] = []
                all_scores[key].append(value)
        
        # Calculate statistics
        mean_scores = {k: float(np.mean(v)) for k, v in all_scores.items()}
        std_scores = {k: float(np.std(v)) for k, v in all_scores.items()}
        min_scores = {k: float(np.min(v)) for k, v in all_scores.items()}
        max_scores = {k: float(np.max(v)) for k, v in all_scores.items()}
        
        # Calculate totals
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
        
        # Calculate cost
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
            failed_prompts=len(failed),
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
        """
        Run the benchmark synchronously.
        
        Args:
            client: The LLM client to evaluate
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            BenchmarkResult with all evaluations
        """
        from datetime import datetime
        
        started_at = datetime.now()
        prompts = self.get_prompts()
        system_prompt = self.get_system_prompt()
        gen_config = self.get_generation_config()
        
        prompt_results: list[PromptResult] = []
        
        for i, prompt_data in enumerate(prompts):
            if progress_callback:
                progress_callback(i, len(prompts))
            
            try:
                # Preprocess and generate
                processed_prompt = self.preprocess_prompt(prompt_data)
                llm_response = client.generate(
                    prompt=processed_prompt,
                    system_prompt=system_prompt,
                    config=gen_config,
                )
                
                if llm_response.is_error:
                    prompt_results.append(PromptResult(
                        prompt_id=prompt_data.get("id", str(i)),
                        prompt=processed_prompt,
                        response="",
                        scores={},
                        metadata=prompt_data.get("metadata", {}),
                        llm_response=llm_response,
                        error=llm_response.error,
                    ))
                    continue
                
                # Postprocess and evaluate
                processed_response = self.postprocess_response(llm_response.content)
                scores = self.evaluate_response(prompt_data, processed_response, llm_response)
                
                prompt_results.append(PromptResult(
                    prompt_id=prompt_data.get("id", str(i)),
                    prompt=processed_prompt,
                    response=processed_response,
                    scores=scores,
                    metadata=prompt_data.get("metadata", {}),
                    llm_response=llm_response,
                ))
                
            except Exception as e:
                prompt_results.append(PromptResult(
                    prompt_id=prompt_data.get("id", str(i)),
                    prompt=prompt_data.get("prompt", ""),
                    response="",
                    scores={},
                    metadata=prompt_data.get("metadata", {}),
                    error=str(e),
                ))
        
        if progress_callback:
            progress_callback(len(prompts), len(prompts))
        
        completed_at = datetime.now()
        
        # Aggregate metrics
        metrics = self.aggregate_scores(prompt_results, client)
        
        return BenchmarkResult(
            benchmark_name=self.name,
            benchmark_version=self.version,
            model=client.model,
            provider=client.provider,
            prompt_results=prompt_results,
            metrics=metrics,
            config=gen_config.to_dict(),
            started_at=started_at,
            completed_at=completed_at,
            metadata=self.metadata,
        )
    
    async def run_async(
        self,
        client: BaseLLMClient,
        max_concurrent: int = 5,
        progress_callback: Optional[callable] = None,
    ) -> BenchmarkResult:
        """
        Run the benchmark asynchronously with concurrency control.
        
        Args:
            client: The LLM client to evaluate
            max_concurrent: Maximum concurrent API calls
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            BenchmarkResult with all evaluations
        """
        import asyncio
        from datetime import datetime
        
        started_at = datetime.now()
        prompts = self.get_prompts()
        system_prompt = self.get_system_prompt()
        gen_config = self.get_generation_config()
        
        semaphore = asyncio.Semaphore(max_concurrent)
        completed = 0
        
        async def process_prompt(i: int, prompt_data: dict) -> PromptResult:
            nonlocal completed
            
            async with semaphore:
                try:
                    processed_prompt = self.preprocess_prompt(prompt_data)
                    llm_response = await client.generate_async(
                        prompt=processed_prompt,
                        system_prompt=system_prompt,
                        config=gen_config,
                    )
                    
                    if llm_response.is_error:
                        result = PromptResult(
                            prompt_id=prompt_data.get("id", str(i)),
                            prompt=processed_prompt,
                            response="",
                            scores={},
                            metadata=prompt_data.get("metadata", {}),
                            llm_response=llm_response,
                            error=llm_response.error,
                        )
                    else:
                        processed_response = self.postprocess_response(llm_response.content)
                        scores = self.evaluate_response(
                            prompt_data, processed_response, llm_response
                        )
                        result = PromptResult(
                            prompt_id=prompt_data.get("id", str(i)),
                            prompt=processed_prompt,
                            response=processed_response,
                            scores=scores,
                            metadata=prompt_data.get("metadata", {}),
                            llm_response=llm_response,
                        )
                    
                except Exception as e:
                    result = PromptResult(
                        prompt_id=prompt_data.get("id", str(i)),
                        prompt=prompt_data.get("prompt", ""),
                        response="",
                        scores={},
                        metadata=prompt_data.get("metadata", {}),
                        error=str(e),
                    )
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(prompts))
                
                return result
        
        # Run all prompts concurrently
        tasks = [
            process_prompt(i, prompt_data)
            for i, prompt_data in enumerate(prompts)
        ]
        prompt_results = await asyncio.gather(*tasks)
        
        completed_at = datetime.now()
        
        # Aggregate metrics
        metrics = self.aggregate_scores(list(prompt_results), client)
        
        return BenchmarkResult(
            benchmark_name=self.name,
            benchmark_version=self.version,
            model=client.model,
            provider=client.provider,
            prompt_results=list(prompt_results),
            metrics=metrics,
            config=gen_config.to_dict(),
            started_at=started_at,
            completed_at=completed_at,
            metadata=self.metadata,
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, version={self.version!r})"
