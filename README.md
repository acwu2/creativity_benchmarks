# LLM Benchmarks

A flexible framework for benchmarking Large Language Models across various tasks and providers.

## Features

- **Multi-provider support**: OpenAI, Anthropic, Google, and more
- **Extensible benchmark system**: Easy to create custom benchmarks
- **Async execution**: Run benchmarks efficiently with async support
- **Rich metrics**: Token usage, latency, cost tracking
- **Result persistence**: Save and analyze benchmark results

## Installation

```bash
# Install in development mode
pip install -e ".[dev]"
```

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
```

## Quick Start

```python
from llm_benchmarks import OpenAIClient, AnthropicClient
from llm_benchmarks.benchmarks import YourBenchmark

# Initialize clients
openai_client = OpenAIClient(model="gpt-4")
anthropic_client = AnthropicClient(model="claude-3-opus-20240229")

# Run benchmark
benchmark = YourBenchmark()
results = benchmark.run([openai_client, anthropic_client])

# Analyze results
benchmark.save_results(results, "results/my_benchmark.json")
```

## Creating Custom Benchmarks

```python
from llm_benchmarks.benchmarks import BaseBenchmark, BenchmarkResult

class MyBenchmark(BaseBenchmark):
    name = "my_benchmark"
    description = "My custom benchmark"
    
    def get_prompts(self) -> list[dict]:
        return [
            {"id": "1", "prompt": "Your prompt here", "metadata": {}}
        ]
    
    def evaluate_response(self, prompt: dict, response: str) -> dict:
        # Your evaluation logic
        return {"score": 1.0, "details": {}}
```

## Project Structure

```
src/llm_benchmarks/
├── __init__.py
├── clients/           # LLM provider clients
├── benchmarks/        # Benchmark implementations
├── config/            # Configuration management
├── utils/             # Utility functions
└── cli.py             # Command-line interface
```

## License

MIT
