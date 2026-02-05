"""Benchmark runner for executing benchmarks across multiple models."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import json

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from llm_benchmarks.benchmarks.base import BaseBenchmark, BenchmarkResult
from llm_benchmarks.clients.base import BaseLLMClient
from llm_benchmarks.config import Settings, get_settings


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""
    max_concurrent: int = 5
    save_results: bool = True
    output_dir: Optional[Path] = None
    verbose: bool = True
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = get_settings().benchmark_output_dir


class BenchmarkRunner:
    """
    Runner for executing benchmarks across multiple models.
    
    Example:
        runner = BenchmarkRunner()
        
        # Add clients
        runner.add_client(OpenAIClient(model="gpt-4o"))
        runner.add_client(AnthropicClient(model="claude-3-5-sonnet"))
        
        # Run benchmark
        results = runner.run(MyBenchmark())
        
        # Compare results
        runner.print_comparison(results)
    """
    
    def __init__(
        self,
        clients: Optional[list[BaseLLMClient]] = None,
        config: Optional[RunConfig] = None,
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            clients: List of LLM clients to evaluate
            config: Run configuration
        """
        self.clients = clients or []
        self.config = config or RunConfig()
        self.console = Console()
    
    def add_client(self, client: BaseLLMClient) -> "BenchmarkRunner":
        """Add a client to the runner."""
        self.clients.append(client)
        return self
    
    def run(
        self,
        benchmark: BaseBenchmark,
        clients: Optional[list[BaseLLMClient]] = None,
    ) -> list[BenchmarkResult]:
        """
        Run a benchmark across all clients synchronously.
        
        Args:
            benchmark: The benchmark to run
            clients: Optional override for clients list
            
        Returns:
            List of BenchmarkResult for each client
        """
        clients = clients or self.clients
        if not clients:
            raise ValueError("No clients configured. Add clients with add_client()")
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            for client in clients:
                task_id = progress.add_task(
                    f"[cyan]{client.provider}/{client.model}",
                    total=len(benchmark.get_prompts()),
                )
                
                def update_progress(current: int, total: int):
                    progress.update(task_id, completed=current)
                
                result = benchmark.run(client, progress_callback=update_progress)
                results.append(result)
                
                if self.config.save_results:
                    self._save_result(result)
        
        if self.config.verbose:
            self.print_comparison(results)
        
        return results
    
    async def run_async(
        self,
        benchmark: BaseBenchmark,
        clients: Optional[list[BaseLLMClient]] = None,
    ) -> list[BenchmarkResult]:
        """
        Run a benchmark across all clients asynchronously.
        
        Args:
            benchmark: The benchmark to run
            clients: Optional override for clients list
            
        Returns:
            List of BenchmarkResult for each client
        """
        clients = clients or self.clients
        if not clients:
            raise ValueError("No clients configured. Add clients with add_client()")
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            for client in clients:
                task_id = progress.add_task(
                    f"[cyan]{client.provider}/{client.model}",
                    total=len(benchmark.get_prompts()),
                )
                
                def update_progress(current: int, total: int):
                    progress.update(task_id, completed=current)
                
                result = await benchmark.run_async(
                    client,
                    max_concurrent=self.config.max_concurrent,
                    progress_callback=update_progress,
                )
                results.append(result)
                
                if self.config.save_results:
                    self._save_result(result)
        
        if self.config.verbose:
            self.print_comparison(results)
        
        return results
    
    def run_all_async(
        self,
        benchmark: BaseBenchmark,
        clients: Optional[list[BaseLLMClient]] = None,
    ) -> list[BenchmarkResult]:
        """
        Convenience method to run async benchmark from sync context.
        
        Args:
            benchmark: The benchmark to run
            clients: Optional override for clients list
            
        Returns:
            List of BenchmarkResult for each client
        """
        return asyncio.run(self.run_async(benchmark, clients))
    
    def _save_result(self, result: BenchmarkResult) -> Path:
        """Save a benchmark result to disk."""
        if self.config.output_dir is None:
            self.config.output_dir = get_settings().benchmark_output_dir
        
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = result.completed_at.strftime("%Y%m%d_%H%M%S")
        filename = f"{result.benchmark_name}_{result.provider}_{result.model}_{timestamp}.json"
        # Sanitize filename
        filename = filename.replace("/", "_").replace(":", "_")
        
        path = output_dir / filename
        result.save(path)
        
        if self.config.verbose:
            self.console.print(f"[dim]Saved: {path}[/dim]")
        
        return path
    
    def print_comparison(self, results: list[BenchmarkResult]) -> None:
        """Print a comparison table of benchmark results."""
        if not results:
            return
        
        # Get all score keys
        all_score_keys: set[str] = set()
        for result in results:
            all_score_keys.update(result.metrics.mean_scores.keys())
        
        score_keys = sorted(all_score_keys)
        
        # Create table
        table = Table(title=f"Benchmark: {results[0].benchmark_name}")
        
        # Add columns
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Provider", style="blue")
        for key in score_keys:
            table.add_column(f"{key} (mean)", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Cost ($)", justify="right")
        
        # Add rows
        for result in results:
            row = [
                result.model,
                result.provider,
            ]
            for key in score_keys:
                value = result.metrics.mean_scores.get(key, 0)
                row.append(f"{value:.3f}")
            
            row.append(str(result.metrics.total_tokens))
            row.append(f"{result.metrics.mean_latency_ms:.1f}")
            
            cost = result.metrics.total_cost_usd
            row.append(f"{cost:.4f}" if cost else "N/A")
            
            table.add_row(*row)
        
        self.console.print(table)
    
    def load_results(
        self,
        pattern: str = "*.json",
        output_dir: Optional[Path] = None,
    ) -> list[BenchmarkResult]:
        """
        Load saved benchmark results from disk.
        
        Args:
            pattern: Glob pattern for result files
            output_dir: Directory to search (defaults to config output_dir)
            
        Returns:
            List of loaded BenchmarkResult objects
        """
        output_dir = output_dir or self.config.output_dir
        if output_dir is None:
            output_dir = get_settings().benchmark_output_dir
        
        results = []
        for path in output_dir.glob(pattern):
            try:
                result = BenchmarkResult.load(path)
                results.append(result)
            except Exception as e:
                if self.config.verbose:
                    self.console.print(f"[yellow]Warning: Failed to load {path}: {e}[/yellow]")
        
        return results
    
    def export_comparison_csv(
        self,
        results: list[BenchmarkResult],
        path: Path | str,
    ) -> None:
        """Export comparison results to CSV."""
        import pandas as pd
        
        rows = []
        for result in results:
            row = {
                "benchmark": result.benchmark_name,
                "model": result.model,
                "provider": result.provider,
                "total_prompts": result.metrics.total_prompts,
                "successful_prompts": result.metrics.successful_prompts,
                "total_tokens": result.metrics.total_tokens,
                "mean_latency_ms": result.metrics.mean_latency_ms,
                "total_cost_usd": result.metrics.total_cost_usd,
                "duration_seconds": result.duration_seconds,
            }
            # Add all mean scores
            for key, value in result.metrics.mean_scores.items():
                row[f"score_{key}_mean"] = value
            for key, value in result.metrics.std_scores.items():
                row[f"score_{key}_std"] = value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        
        if self.config.verbose:
            self.console.print(f"[green]Exported to {path}[/green]")
