"""Command-line interface for LLM benchmarks."""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Built-in benchmark registry
BUILTIN_BENCHMARKS = {
    "free-association": "llm_benchmarks.benchmarks.FreeAssociationBenchmark",
    "free": "llm_benchmarks.benchmarks.FreeAssociationBenchmark",
    "fa": "llm_benchmarks.benchmarks.FreeAssociationBenchmark",
    "telephone-game": "llm_benchmarks.benchmarks.TelephoneGameBenchmark",
    "telephone": "llm_benchmarks.benchmarks.TelephoneGameBenchmark",
    "tg": "llm_benchmarks.benchmarks.TelephoneGameBenchmark",
    "this-and-that": "llm_benchmarks.benchmarks.ThisAndThatBenchmark",
    "this-that": "llm_benchmarks.benchmarks.ThisAndThatBenchmark",
    "tat": "llm_benchmarks.benchmarks.ThisAndThatBenchmark",
    "style-flexibility": "llm_benchmarks.benchmarks.ThisAndThatBenchmark",
    "not-like-that": "llm_benchmarks.benchmarks.NotLikeThatBenchmark",
    "nlt": "llm_benchmarks.benchmarks.NotLikeThatBenchmark",
    "difference-negation": "llm_benchmarks.benchmarks.NotLikeThatBenchmark",
    "quilting": "llm_benchmarks.benchmarks.QuiltingBenchmark",
    "quilt": "llm_benchmarks.benchmarks.QuiltingBenchmark",
    "creative-constraints": "llm_benchmarks.benchmarks.QuiltingBenchmark",
    "simple-qa": "llm_benchmarks.benchmarks.examples.SimpleQABenchmark",
    "qa": "llm_benchmarks.benchmarks.examples.SimpleQABenchmark",
    "reasoning": "llm_benchmarks.benchmarks.examples.ReasoningBenchmark",
}


def main(args: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="llm-bench",
        description="LLM Benchmarking Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run free-association -m gpt-4o
  %(prog)s run qa -m openai:gpt-4o anthropic:claude-3-5-sonnet-20241022
  %(prog)s list
  %(prog)s compare results/*.json
  %(prog)s visualize results/ -c all
  %(prog)s visualize results/ -c report

Built-in benchmarks: free-association (fa), simple-qa (qa), reasoning
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run a benchmark")
    run_parser.add_argument(
        "benchmark",
        help="Benchmark name (free-association, qa, reasoning) or module path",
    )
    run_parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=["gpt-4o"],
        help="Models to evaluate (e.g., 'gpt-4o' or 'anthropic:claude-3-5-sonnet')",
    )
    run_parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./results"),
        help="Output directory for results",
    )
    run_parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=5,
        help="Maximum concurrent API calls",
    )
    run_parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Run asynchronously (faster)",
    )
    run_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output",
    )
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare benchmark results")
    compare_parser.add_argument(
        "results",
        nargs="+",
        type=Path,
        help="Result files to compare",
    )
    compare_parser.add_argument(
        "--export", "-e",
        type=Path,
        help="Export comparison to CSV",
    )
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize benchmark results")
    viz_parser.add_argument(
        "results",
        nargs="*",
        type=Path,
        default=[],
        help="Result files or directory to visualize (defaults to ./results)",
    )
    viz_parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./charts"),
        help="Output directory for charts",
    )
    viz_parser.add_argument(
        "--format", "-f",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Image format for saved charts",
    )
    viz_parser.add_argument(
        "--report", "-r",
        action="store_true",
        help="Generate an HTML report instead of a chart image",
    )
    viz_parser.add_argument(
        "--benchmark", "-b",
        help="Filter to specific benchmark name",
    )
    
    # List command
    subparsers.add_parser("list", help="List available benchmarks")
    
    # Models command
    subparsers.add_parser("models", help="List available models")

    parsed_args = parser.parse_args(args)
    
    if parsed_args.command == "run":
        return run_benchmark(parsed_args)
    elif parsed_args.command == "compare":
        return compare_results(parsed_args)
    elif parsed_args.command == "visualize":
        return visualize_results(parsed_args)
    elif parsed_args.command == "list":
        return list_benchmarks(parsed_args)
    else:
        parser.print_help()
        return 0


def resolve_benchmark(name: str):
    """Resolve benchmark name to class."""
    from importlib import import_module
    
    # Check built-in benchmarks first
    if name.lower() in BUILTIN_BENCHMARKS:
        class_path = BUILTIN_BENCHMARKS[name.lower()]
    elif "." in name:
        class_path = name
    else:
        raise ValueError(f"Unknown benchmark: {name}. Use 'llm-bench list' to see available benchmarks.")
    
    module_path, class_name = class_path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse model spec like 'openai:gpt-4o' or just 'gpt-4o'."""
    if ":" in spec:
        provider, model = spec.split(":", 1)
        return provider.lower(), model
    
    # Auto-detect provider from model name
    spec_lower = spec.lower()
    if spec_lower.startswith(("gpt", "o1")):
        return "openai", spec
    elif spec_lower.startswith("claude"):
        return "anthropic", spec
    elif spec_lower.startswith("gemini"):
        return "google", spec
    else:
        return "openai", spec  # Default to OpenAI


def get_client(provider: str, model: str):
    """Create LLM client for provider/model."""
    from llm_benchmarks import OpenAIClient, AnthropicClient, GoogleClient
    
    clients = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "google": GoogleClient,
    }
    
    if provider not in clients:
        raise ValueError(f"Unknown provider: {provider}")
    
    return clients[provider](model=model)


def run_benchmark(args: argparse.Namespace) -> int:
    """Run a benchmark."""
    from rich.console import Console
    from rich.panel import Panel
    
    from llm_benchmarks.benchmarks import BenchmarkRunner
    from llm_benchmarks.benchmarks.runner import RunConfig
    
    console = Console()
    
    # Load the benchmark
    try:
        benchmark_class = resolve_benchmark(args.benchmark)
        benchmark = benchmark_class()
    except Exception as e:
        console.print(f"[red]Error loading benchmark: {e}[/red]")
        return 1
    
    # Create and validate clients
    clients = []
    for model_spec in args.models:
        try:
            provider, model = parse_model_spec(model_spec)
            client = get_client(provider, model)
            if not args.quiet:
                console.print(f"[dim]Validating model '{model}' with {provider}...[/dim]")
            client.validate_model()
            clients.append(client)
        except ValueError as e:
            console.print(f"[red]Error: Invalid model '{model_spec}': {e}[/red]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not create client for '{model_spec}': {e}[/yellow]")
    
    if not clients:
        console.print("[red]Error: No valid clients configured[/red]")
        console.print("[dim]Use 'llm-bench models' to see available models.[/dim]")
        return 1
    
    # Print header
    if not args.quiet:
        console.print()
        console.print(Panel(
            f"[bold]{benchmark.name}[/bold] v{benchmark.version}\n{benchmark.description}",
            title="🧪 Running Benchmark",
            border_style="green",
        ))
        console.print()
        console.print(f"Models: {', '.join(f'[cyan]{c.provider}:{c.model}[/cyan]' for c in clients)}")
        console.print(f"Prompts: {len(benchmark.get_prompts())}")
        console.print()
    
    # Run the benchmark
    config = RunConfig(
        max_concurrent=args.concurrency,
        save_results=True,
        output_dir=args.output,
        verbose=not args.quiet,
    )
    
    runner = BenchmarkRunner(clients=clients, config=config)
    
    try:
        if args.use_async:
            runner.run_all_async(benchmark)
        else:
            runner.run(benchmark)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        return 130
    
    if not args.quiet:
        console.print()
        console.print("[bold green]✓ Benchmark complete![/bold green]")
        console.print(f"[dim]Results saved to: {args.output}/[/dim]")
    
    return 0


def compare_results(args: argparse.Namespace) -> int:
    """Compare benchmark results."""
    from rich.console import Console
    
    from llm_benchmarks.benchmarks import BenchmarkResult, BenchmarkRunner
    from llm_benchmarks.benchmarks.runner import RunConfig
    
    console = Console()
    
    results = []
    for path in args.results:
        try:
            result = BenchmarkResult.load(path)
            results.append(result)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load {path}: {e}[/yellow]")
    
    if not results:
        console.print("[red]Error: No results loaded[/red]")
        return 1
    
    runner = BenchmarkRunner(config=RunConfig(verbose=True))
    runner.print_comparison(results)
    
    if args.export:
        runner.export_comparison_csv(results, args.export)
    
    return 0


def visualize_results(args: argparse.Namespace) -> int:
    """Visualize benchmark results."""
    from rich.console import Console
    from rich.panel import Panel
    
    from llm_benchmarks.utils.visualizer import BenchmarkVisualizer
    
    console = Console()
    
    # Determine input path(s)
    if args.results:
        input_paths = args.results
    else:
        input_paths = [Path("./results")]
    
    # Load results
    viz = BenchmarkVisualizer()
    
    for path in input_paths:
        if path.exists():
            viz.load_results(path)
        else:
            console.print(f"[yellow]Warning: Path not found: {path}[/yellow]")
    
    if not viz.results:
        console.print("[red]Error: No results loaded[/red]")
        console.print("[dim]Specify result files or directory, e.g.: llm-bench visualize results/[/dim]")
        return 1
    
    # Print summary
    df = viz.to_dataframe()
    console.print()
    console.print(Panel(
        f"Loaded [bold]{len(viz.results)}[/bold] results\n"
        f"Benchmarks: {', '.join(df['benchmark'].unique())}\n"
        f"Models: {', '.join(df['model_id'].unique())}",
        title="📊 Visualization",
        border_style="blue",
    ))
    console.print()
    
    # Generate dashboard (or HTML report)
    try:
        args.output.mkdir(parents=True, exist_ok=True)

        if args.report:
            report_path = args.output / "benchmark_report.html"
            viz.generate_html_report(report_path)
            console.print(f"[green]HTML report saved to: {report_path}[/green]")
        elif args.benchmark:
            save_path = args.output / f"{args.benchmark}.{args.format}"
            viz.plot_dashboard(
                benchmark=args.benchmark,
                save_path=save_path,
            )
            console.print()
            console.print(f"[bold green]✓ Chart saved to: {save_path}[/bold green]")
        else:
            viz.plot_all_dashboards(
                save_dir=args.output,
                format=args.format,
            )
            console.print()
            console.print(f"[bold green]✓ Charts saved to: {args.output}[/bold green]")

    except ImportError:
        console.print("[red]Error: Missing visualization dependencies[/red]")
        console.print("[dim]Install with: pip install matplotlib seaborn[/dim]")
        return 1
    except Exception as e:
        console.print(f"[red]Error generating chart: {e}[/red]")
        return 1
    
    return 0


def list_benchmarks(args: argparse.Namespace) -> int:
    """List available benchmarks."""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    table = Table(title="📋 Available Benchmarks")
    table.add_column("Name", style="cyan")
    table.add_column("Aliases", style="dim")
    table.add_column("Description")
    
    # Built-in benchmarks with descriptions
    benchmarks_info = {
        "free-association": {
            "aliases": ["free", "fa"],
            "description": "Measure creative vocabulary through free word association",
        },
        "telephone-game": {
            "aliases": ["telephone", "tg"],
            "description": "Measure creative stability and mode collapse through iterative rewriting",
        },
        "this-and-that": {
            "aliases": ["this-that", "tat", "style-flexibility"],
            "description": "Measure style flexibility through story interpolation",
        },
        "not-like-that": {
            "aliases": ["nlt", "difference-negation"],
            "description": "Measure constrained creativity through style avoidance",
        },
        "quilting": {
            "aliases": ["quilt", "creative-constraints"],
            "description": "Measure creative synthesis by combining constrained ingredients into stories",
        },
        "simple-qa": {
            "aliases": ["qa"],
            "description": "Simple factual question-answering benchmark",
        },
        "reasoning": {
            "aliases": [],
            "description": "Logical reasoning and problem-solving benchmark",
        },
    }
    
    for name, info in benchmarks_info.items():
        aliases = ", ".join(info["aliases"]) if info["aliases"] else "-"
        table.add_row(name, aliases, info["description"])
    
    console.print(table)
    console.print()
    console.print("[dim]Run a benchmark:[/dim] llm-bench run <name> -m <model>")
    console.print("[dim]Example:[/dim] llm-bench run free-association -m gpt-4o")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
