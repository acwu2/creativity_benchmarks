"""Visualizer for benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd

from llm_benchmarks.benchmarks.base import BenchmarkResult


class BenchmarkVisualizer:
    """
    Visualizer for comparing and analyzing benchmark results.

    Generates a single comprehensive dashboard figure that combines
    scores, latency, token usage, and success rate into one view.

    Example:
        viz = BenchmarkVisualizer()
        viz.load_results("results/")
        viz.plot_dashboard(save_path="charts/dashboard.png")
    """

    def __init__(self, results: Optional[list[BenchmarkResult]] = None):
        """
        Initialize the visualizer.

        Args:
            results: Optional list of BenchmarkResult objects
        """
        self.results = results or []
        self._df: Optional[pd.DataFrame] = None
        self._prompt_df: Optional[pd.DataFrame] = None

        # Lazy import matplotlib to avoid overhead if not needed
        self._plt = None
        self._sns = None

    @property
    def plt(self):
        """Lazy load matplotlib."""
        if self._plt is None:
            import matplotlib.pyplot as plt
            self._plt = plt
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['font.size'] = 12
        return self._plt

    @property
    def sns(self):
        """Lazy load seaborn."""
        if self._sns is None:
            import seaborn as sns
            self._sns = sns
            sns.set_theme(style="whitegrid")
        return self._sns

    def load_results(
        self,
        path: Path | str,
        pattern: str = "*.json",
    ) -> "BenchmarkVisualizer":
        """
        Load benchmark results from a directory.

        Args:
            path: Directory path or single file path
            pattern: Glob pattern for result files

        Returns:
            Self for chaining
        """
        path = Path(path)

        if path.is_file():
            self.results.append(BenchmarkResult.load(path))
        else:
            for file_path in path.glob(pattern):
                try:
                    result = BenchmarkResult.load(file_path)
                    self.results.append(result)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")

        # Reset cached dataframes
        self._df = None
        self._prompt_df = None

        return self

    def add_result(self, result: BenchmarkResult) -> "BenchmarkVisualizer":
        """Add a single result."""
        self.results.append(result)
        self._df = None
        self._prompt_df = None
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a summary DataFrame."""
        if self._df is not None:
            return self._df

        rows = []
        for result in self.results:
            row = {
                "benchmark": result.benchmark_name,
                "model": result.model,
                "provider": result.provider,
                "model_id": f"{result.provider}/{result.model}",
                "total_prompts": result.metrics.total_prompts,
                "successful_prompts": result.metrics.successful_prompts,
                "success_rate": result.metrics.successful_prompts / max(result.metrics.total_prompts, 1),
                "total_tokens": result.metrics.total_tokens,
                "mean_latency_ms": result.metrics.mean_latency_ms,
                "total_cost_usd": result.metrics.total_cost_usd or 0,
                "duration_seconds": result.duration_seconds,
            }

            # Add all mean scores
            for key, value in result.metrics.mean_scores.items():
                row[f"score_{key}"] = value

            # Add std scores
            for key, value in result.metrics.std_scores.items():
                row[f"score_{key}_std"] = value

            rows.append(row)

        self._df = pd.DataFrame(rows)
        return self._df

    def to_prompt_dataframe(self) -> pd.DataFrame:
        """Convert all prompt results to a detailed DataFrame."""
        if self._prompt_df is not None:
            return self._prompt_df

        rows = []
        for result in self.results:
            for prompt_result in result.prompt_results:
                row = {
                    "benchmark": result.benchmark_name,
                    "model": result.model,
                    "provider": result.provider,
                    "model_id": f"{result.provider}/{result.model}",
                    "prompt_id": prompt_result.prompt_id,
                    "prompt": prompt_result.prompt[:200],
                    "response": prompt_result.response[:500] if prompt_result.response else None,
                    "is_error": prompt_result.is_error,
                    "error": prompt_result.error,
                }

                # Add scores
                for key, value in prompt_result.scores.items():
                    row[f"score_{key}"] = value

                # Add metadata
                for key, value in prompt_result.metadata.items():
                    row[f"meta_{key}"] = value

                # Add LLM response info if available
                if prompt_result.llm_response:
                    row["latency_ms"] = prompt_result.llm_response.latency_ms
                    if prompt_result.llm_response.usage:
                        row["prompt_tokens"] = prompt_result.llm_response.usage.prompt_tokens
                        row["completion_tokens"] = prompt_result.llm_response.usage.completion_tokens
                        row["total_tokens"] = prompt_result.llm_response.usage.total_tokens

                rows.append(row)

        self._prompt_df = pd.DataFrame(rows)
        return self._prompt_df

    def plot_dashboard(
        self,
        benchmark: str = "",
        figsize: tuple[int, int] = (16, 10),
        save_path: Optional[Path | str] = None,
    ) -> "BenchmarkVisualizer":
        """
        Generate a grid of bar charts for a single benchmark.

        Each metric gets its own subplot so different scales are clearly
        readable.  When *benchmark* is empty the first benchmark found is
        used – call :meth:`plot_all_dashboards` to iterate over every
        benchmark automatically.

        Args:
            benchmark: Benchmark name to plot (required for meaningful output)
            figsize: Figure size
            save_path: Path to save the figure

        Returns:
            Self for chaining
        """
        df = self.to_dataframe()

        if benchmark:
            df = df[df["benchmark"] == benchmark]

        if df.empty:
            print("No data to plot")
            return self

        score_cols = [c for c in df.columns if c.startswith("score_") and not c.endswith("_std")]
        if not score_cols:
            print("No score columns found")
            return self

        models = df["model_id"].unique()
        n_metrics = len(score_cols)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        palette = self.sns.color_palette("muted", n_colors=len(models))

        fig, axes = self.plt.subplots(n_rows, n_cols, figsize=figsize)
        # Ensure axes is always a flat array
        if n_metrics == 1:
            axes = np.array([axes])
        axes = np.atleast_1d(axes).flatten()

        display_name = benchmark.replace("_", " ").replace("-", " ").title() if benchmark else "All"
        fig.suptitle(display_name, fontsize=18, fontweight="bold", y=1.02)

        x = np.arange(len(models))
        bar_width = 0.6

        for i, col in enumerate(score_cols):
            ax = axes[i]
            values, errors = [], []
            for model in models:
                mdf = df[df["model_id"] == model]
                values.append(mdf[col].mean() if not mdf.empty else 0)
                std_col = f"{col}_std"
                errors.append(mdf[std_col].mean() if std_col in df.columns and not mdf.empty else 0)

            bars = ax.bar(x, values, bar_width, color=palette[:len(models)],
                          edgecolor="white", linewidth=0.8)

            if any(e > 0 for e in errors):
                ax.errorbar(x, values, yerr=errors, fmt="none", color="black",
                            capsize=4, linewidth=1.2)

            # Add value labels on top of each bar
            for j, (bar, val) in enumerate(zip(bars, values)):
                fmt = f"{val:.2f}" if abs(val) < 10 else f"{val:.0f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        fmt, ha="center", va="bottom", fontsize=10, fontweight="bold")

            label = col.replace("score_", "").replace("_", " ").title()
            ax.set_title(label, fontsize=13, fontweight="semibold", pad=10)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
            ax.set_ylabel("Score", fontsize=10)
            ax.grid(axis="y", alpha=0.3)

            # Give some headroom for value labels
            if values:
                ymax = max(values) * 1.15 if max(values) > 0 else 1
                ax.set_ylim(bottom=0, top=ymax)

        # Hide unused subplot axes
        for j in range(n_metrics, len(axes)):
            axes[j].set_visible(False)

        self.plt.tight_layout()

        if save_path:
            self.plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        self.plt.show()
        return self

    def plot_all_dashboards(
        self,
        figsize: tuple[int, int] = (16, 10),
        save_dir: Optional[Path | str] = None,
        format: str = "png",
    ) -> "BenchmarkVisualizer":
        """
        Generate a separate chart for every benchmark in the loaded results.

        Args:
            figsize: Figure size for each chart
            save_dir: Directory to save charts (one file per benchmark)
            format: Image format (png, pdf, svg)

        Returns:
            Self for chaining
        """
        df = self.to_dataframe()
        if df.empty:
            print("No data to plot")
            return self

        benchmarks = df["benchmark"].unique()

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        for bench in benchmarks:
            save_path = None
            if save_dir is not None:
                slug = bench.replace(" ", "_").replace("-", "_").lower()
                save_path = save_dir / f"{slug}.{format}"
            self.plot_dashboard(benchmark=bench, figsize=figsize, save_path=save_path)

        return self

    def save_all(
        self,
        output_dir: Path | str,
        prefix: str = "",
        format: str = "png",
    ) -> "BenchmarkVisualizer":
        """
        Generate and save one chart per benchmark.

        Args:
            output_dir: Directory to save the charts
            prefix: Prefix for each filename
            format: Image format (png, pdf, svg)

        Returns:
            Self for chaining
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        if df.empty:
            print("No data to save")
            return self

        benchmarks = df["benchmark"].unique()
        prefix_str = f"{prefix}_" if prefix else ""

        for bench in benchmarks:
            slug = bench.replace(" ", "_").replace("-", "_").lower()
            save_path = output_dir / f"{prefix_str}{slug}.{format}"
            try:
                self.plot_dashboard(benchmark=bench, save_path=save_path)
            except Exception as e:
                print(f"Warning: Failed to generate chart for {bench}: {e}")

        print(f"\nCharts saved to: {output_dir}")
        return self

    def generate_html_report(
        self,
        output_path: Path | str,
        title: str = "Benchmark Results Report",
    ) -> "BenchmarkVisualizer":
        """
        Generate a comprehensive HTML report with the embedded dashboard.

        Args:
            output_path: Path to save HTML file
            title: Report title

        Returns:
            Self for chaining
        """
        import base64
        from io import BytesIO

        output_path = Path(output_path)
        df = self.to_dataframe()

        # Generate one chart per benchmark as base64 images
        charts_html = ""
        benchmarks = df["benchmark"].unique()
        for bench in benchmarks:
            try:
                self.plot_dashboard(benchmark=bench)
                buf = BytesIO()
                self.plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                self.plt.close()
                display_name = bench.replace("_", " ").replace("-", " ").title()
                charts_html += (
                    f'<h3>{display_name}</h3>'
                    f'<img src="data:image/png;base64,{img_base64}" alt="{display_name}">'
                )
            except Exception:
                pass

        # Build HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 40px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #007bff; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        tr:hover {{ background: #f1f1f1; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border-radius: 4px; }}
        .timestamp {{ color: #888; font-size: 0.9em; }}
        .metric {{ display: inline-block; background: #e9ecef; padding: 5px 15px; border-radius: 20px; margin: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="timestamp">Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Summary Statistics</h2>
        <div>
            <span class="metric">Benchmarks: {len(df['benchmark'].unique())}</span>
            <span class="metric">Models: {len(df['model_id'].unique())}</span>
            <span class="metric">Total Prompts: {df['total_prompts'].sum()}</span>
        </div>

        <h2>Results Table</h2>
        {df.to_html(index=False, classes='results-table')}

        <h2>Charts</h2>
        {charts_html}
    </div>
</body>
</html>"""

        output_path.write_text(html)
        print(f"Report saved to: {output_path}")

        return self
