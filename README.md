# LLM Creativity Benchmarks

A framework for evaluating the **creative capabilities** of Large Language Models. Rather than testing factual accuracy or reasoning, these benchmarks measure how well models handle open-ended creative tasks like free association, style interpolation, constrained writing, and stylistic differentiation.

## Benchmarks

| Benchmark | Category | What It Measures |
|---|---|---|
| **Free Association** (`free-association`, `fa`) | Iteration | Vocabulary exploration — how many unique words a model produces and how quickly it repeats |
| **Telephone Game** (`telephone-game`, `tg`) | Iteration | Meaning drift across iterative rewriting |
| **This & That** (`this-and-that`, `tat`) | Style Flexibility | Style interpolation — generating text that blends two distinct writing styles |
| **Not Like That** (`not-like-that`, `nlt`) | Difference & Negation | Constrained style differentiation — writing *like* one example while *avoiding* another |
| **Quilting** (`quilting`, `quilt`) | Creative Constraints | Creative synthesis from constrained ingredients — weaving pre-set fragments into coherent prose |

## Supported Providers

- **OpenAI** — GPT-4o, GPT-3.5 Turbo, o1, etc.
- **Anthropic** — Claude 3.5 Sonnet, Claude 3 Opus, etc.
- **Google** — Gemini models

## Installation

Requires **Python 3.10+**.

```bash
# Clone the repo
git clone <repo-url>
cd creativity_benchmarks

# Install in development mode (includes test dependencies)
pip install -e ".[dev]"
```

## Configuration

Create a `.env` file in the project root with API keys for the providers you want to use:

```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
```

Only the key(s) for the provider(s) you plan to test are required. Additional optional settings:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_BASE_URL` | — | Custom OpenAI-compatible API base URL |
| `ANTHROPIC_BASE_URL` | — | Custom Anthropic API base URL |
| `BENCHMARK_OUTPUT_DIR` | `./results` | Where result JSON files are saved |
| `BENCHMARK_MAX_CONCURRENT` | `5` | Max concurrent API calls |
| `BENCHMARK_DEFAULT_TIMEOUT` | `120` | API call timeout in seconds |
| `BENCHMARK_RETRY_ATTEMPTS` | `3` | Retry count for failed calls |

## Usage

All commands use the `llm-bench` CLI:

### Run a benchmark

```bash
# Run Free Association against GPT-4o (default model)
llm-bench run free-association

# Specify one or more models
llm-bench run quilting -m gpt-4o anthropic:claude-3-5-sonnet-20241022

# Run asynchronously with higher concurrency
llm-bench run this-and-that -m gpt-4o --async -c 10
```

The model prefix is auto-detected from the name (`gpt-*` → OpenAI, `claude-*` → Anthropic, `gemini-*` → Google), or you can be explicit with `provider:model` syntax.

### List available benchmarks

```bash
llm-bench list
```

### Compare results

```bash
llm-bench compare results/*.json
```

### Visualize results

```bash
# Generate all chart types into ./charts
llm-bench visualize results/

# Generate a single HTML report
llm-bench visualize results/ -c report

# Filter to a specific benchmark and export as SVG
llm-bench visualize results/ -b free_association -f svg
```

Chart types: `all`, `scores`, `radar`, `latency`, `heatmap`, `summary`, `report`.

## Results

Benchmark results are saved as JSON files in `results/` (by default), named with the pattern:

```
{benchmark}_{provider}_{model}_{timestamp}.json
```

Each result file contains per-prompt scores, aggregated metrics (mean, std, min, max), token usage, latency, and cost estimates.

## Project Structure

```
src/llm_benchmarks/
├── cli.py                              # CLI entry point (llm-bench)
├── clients/                            # LLM provider clients
│   ├── openai_client.py
│   ├── anthropic_client.py
│   └── google_client.py
├── benchmarks/
│   ├── base.py                         # BaseBenchmark, BenchmarkResult, etc.
│   ├── runner.py                       # BenchmarkRunner orchestration
│   ├── iteration/
│   │   ├── free_association.py         # Free Association benchmark
│   │   └── telephone_game.py          # Telephone Game benchmark
│   ├── style_flexibility/
│   │   └── this_and_that.py            # This & That benchmark
│   ├── difference_and_negation/
│   │   └── not_like_that.py            # Not Like That benchmark
│   └── creative_constraints/
│       └── quilting.py                 # Quilting benchmark
├── config/
│   └── settings.py                     # Pydantic-based settings (.env loading)
└── utils/
    ├── metrics.py                      # Scoring helpers
    ├── text.py                         # Text processing utilities
    └── visualizer.py                   # Chart generation & HTML reports
```

## Running Tests

```bash
pytest
```

## License

MIT
