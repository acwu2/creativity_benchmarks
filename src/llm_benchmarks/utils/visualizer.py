"""Visualizer for benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal
import json

import numpy as np
import pandas as pd

from llm_benchmarks.benchmarks.base import BenchmarkResult


class BenchmarkVisualizer:
    """
    Visualizer for comparing and analyzing benchmark results.
    
    Generates various charts and plots to compare model performance
    across different benchmarks and metrics.
    
    Example:
        viz = BenchmarkVisualizer()
        viz.load_results("results/")
        
        # Generate comparison charts
        viz.plot_score_comparison()
        viz.plot_radar_chart()
        viz.plot_latency_distribution()
        
        # Save all charts
        viz.save_all("charts/")
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
            # Set style
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
                    "prompt": prompt_result.prompt[:200],  # Truncate for display
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
                        row["prompt_tokens"] = prompt_result.llm_response.usage.get("prompt_tokens", 0)
                        row["completion_tokens"] = prompt_result.llm_response.usage.get("completion_tokens", 0)
                        row["total_tokens"] = prompt_result.llm_response.usage.get("total_tokens", 0)
                
                rows.append(row)
        
        self._prompt_df = pd.DataFrame(rows)
        return self._prompt_df
    
    def plot_score_comparison(
        self,
        benchmark: Optional[str] = None,
        score_columns: Optional[list[str]] = None,
        figsize: tuple[int, int] = (14, 8),
        save_path: Optional[Path | str] = None,
    ) -> "BenchmarkVisualizer":
        """
        Create a grouped bar chart comparing scores across models.
        
        Args:
            benchmark: Filter to specific benchmark (optional)
            score_columns: Specific score columns to plot (optional)
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
        
        # Get score columns
        if score_columns is None:
            score_columns = [col for col in df.columns if col.startswith("score_") and not col.endswith("_std")]
        
        # Prepare data for plotting
        models = df["model_id"].unique()
        x = np.arange(len(models))
        width = 0.8 / len(score_columns)
        
        fig, ax = self.plt.subplots(figsize=figsize)
        
        colors = self.plt.cm.Set2(np.linspace(0, 1, len(score_columns)))
        
        for i, col in enumerate(score_columns):
            values = []
            errors = []
            for model in models:
                model_df = df[df["model_id"] == model]
                if not model_df.empty:
                    values.append(model_df[col].mean())
                    # Get std if available
                    std_col = f"{col}_std"
                    if std_col in df.columns:
                        errors.append(model_df[std_col].mean())
                    else:
                        errors.append(0)
                else:
                    values.append(0)
                    errors.append(0)
            
            label = col.replace("score_", "").replace("_", " ").title()
            bars = ax.bar(x + i * width, values, width, label=label, color=colors[i])
            
            # Add error bars if we have std values
            if any(e > 0 for e in errors):
                ax.errorbar(x + i * width, values, yerr=errors, fmt='none', color='black', capsize=3)
        
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title(f"Score Comparison{f' - {benchmark}' if benchmark else ''}")
        ax.set_xticks(x + width * (len(score_columns) - 1) / 2)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        
        self.plt.show()
        return self
    
    def plot_radar_chart(
        self,
        benchmark: Optional[str] = None,
        score_columns: Optional[list[str]] = None,
        normalize: bool = True,
        figsize: tuple[int, int] = (10, 10),
        save_path: Optional[Path | str] = None,
    ) -> "BenchmarkVisualizer":
        """
        Create a radar/spider chart comparing models across metrics.
        
        Args:
            benchmark: Filter to specific benchmark (optional)
            score_columns: Specific score columns to plot (optional)
            normalize: Whether to normalize scores to 0-1 range
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
        
        # Get score columns
        if score_columns is None:
            score_columns = [col for col in df.columns if col.startswith("score_") and not col.endswith("_std")]
        
        if len(score_columns) < 3:
            print("Need at least 3 score dimensions for radar chart")
            return self
        
        # Prepare data
        models = df["model_id"].unique()
        num_vars = len(score_columns)
        
        # Create angles for radar chart
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = self.plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        colors = self.plt.cm.Set1(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            model_df = df[df["model_id"] == model]
            values = []
            
            for col in score_columns:
                val = model_df[col].mean()
                if normalize:
                    # Normalize to 0-1 based on all data
                    col_min, col_max = df[col].min(), df[col].max()
                    if col_max > col_min:
                        val = (val - col_min) / (col_max - col_min)
                    else:
                        val = 0.5
                values.append(val)
            
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])
        
        # Set labels
        labels = [col.replace("score_", "").replace("_", " ").title() for col in score_columns]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=10)
        
        ax.set_title(f"Model Comparison{f' - {benchmark}' if benchmark else ''}", size=14, y=1.1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        
        self.plt.show()
        return self
    
    def plot_latency_distribution(
        self,
        benchmark: Optional[str] = None,
        figsize: tuple[int, int] = (12, 6),
        save_path: Optional[Path | str] = None,
    ) -> "BenchmarkVisualizer":
        """
        Create a box plot showing latency distribution across models.
        
        Args:
            benchmark: Filter to specific benchmark (optional)
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Self for chaining
        """
        df = self.to_prompt_dataframe()
        
        if benchmark:
            df = df[df["benchmark"] == benchmark]
        
        if df.empty or "latency_ms" not in df.columns:
            print("No latency data to plot")
            return self
        
        fig, ax = self.plt.subplots(figsize=figsize)
        
        # Create box plot
        models = df["model_id"].unique()
        data = [df[df["model_id"] == model]["latency_ms"].dropna() for model in models]
        
        bp = ax.boxplot(data, labels=models, patch_artist=True)
        
        # Color the boxes
        colors = self.plt.cm.Set2(np.linspace(0, 1, len(models)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel("Model")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Latency Distribution{f' - {benchmark}' if benchmark else ''}")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis="y", alpha=0.3)
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        
        self.plt.show()
        return self
    
    def plot_score_distribution(
        self,
        score_column: str,
        benchmark: Optional[str] = None,
        kind: Literal["box", "violin", "strip"] = "violin",
        figsize: tuple[int, int] = (12, 6),
        save_path: Optional[Path | str] = None,
    ) -> "BenchmarkVisualizer":
        """
        Plot distribution of a specific score across models.
        
        Args:
            score_column: The score column to plot (e.g., "score_chain_length")
            benchmark: Filter to specific benchmark (optional)
            kind: Type of plot - "box", "violin", or "strip"
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Self for chaining
        """
        df = self.to_prompt_dataframe()
        
        if benchmark:
            df = df[df["benchmark"] == benchmark]
        
        if score_column not in df.columns:
            print(f"Score column '{score_column}' not found")
            return self
        
        fig, ax = self.plt.subplots(figsize=figsize)
        
        if kind == "box":
            self.sns.boxplot(data=df, x="model_id", y=score_column, ax=ax, palette="Set2")
        elif kind == "violin":
            self.sns.violinplot(data=df, x="model_id", y=score_column, ax=ax, palette="Set2")
        else:
            self.sns.stripplot(data=df, x="model_id", y=score_column, ax=ax, palette="Set2", alpha=0.6)
        
        label = score_column.replace("score_", "").replace("_", " ").title()
        ax.set_xlabel("Model")
        ax.set_ylabel(label)
        ax.set_title(f"{label} Distribution{f' - {benchmark}' if benchmark else ''}")
        ax.tick_params(axis='x', rotation=45)
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        
        self.plt.show()
        return self
    
    def plot_benchmark_heatmap(
        self,
        score_column: Optional[str] = None,
        figsize: tuple[int, int] = (12, 8),
        save_path: Optional[Path | str] = None,
    ) -> "BenchmarkVisualizer":
        """
        Create a heatmap showing model performance across benchmarks.
        
        Args:
            score_column: Specific score column to use (defaults to first available)
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Self for chaining
        """
        df = self.to_dataframe()
        
        if df.empty:
            print("No data to plot")
            return self
        
        # Get score columns
        score_columns = [col for col in df.columns if col.startswith("score_") and not col.endswith("_std")]
        
        if not score_columns:
            print("No score columns found")
            return self
        
        # Use first score column if not specified
        if score_column is None:
            # Try to find a common metric or use the first one
            score_column = score_columns[0]
        
        # Pivot table: models as rows, benchmarks as columns
        pivot_df = df.pivot_table(
            values=score_column,
            index="model_id",
            columns="benchmark",
            aggfunc="mean"
        )
        
        fig, ax = self.plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(pivot_df.values, cmap="YlGnBu", aspect="auto")
        
        # Add colorbar
        cbar = self.plt.colorbar(im, ax=ax)
        label = score_column.replace("score_", "").replace("_", " ").title()
        cbar.set_label(label)
        
        # Set ticks
        ax.set_xticks(np.arange(len(pivot_df.columns)))
        ax.set_yticks(np.arange(len(pivot_df.index)))
        ax.set_xticklabels(pivot_df.columns, rotation=45, ha="right")
        ax.set_yticklabels(pivot_df.index)
        
        # Add values as text
        for i in range(len(pivot_df.index)):
            for j in range(len(pivot_df.columns)):
                val = pivot_df.iloc[i, j]
                if not np.isnan(val):
                    text_color = "white" if val > pivot_df.values[~np.isnan(pivot_df.values)].mean() else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=10)
        
        ax.set_xlabel("Benchmark")
        ax.set_ylabel("Model")
        ax.set_title(f"Performance Heatmap - {label}")
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        
        self.plt.show()
        return self
    
    def plot_cost_efficiency(
        self,
        score_column: Optional[str] = None,
        figsize: tuple[int, int] = (10, 8),
        save_path: Optional[Path | str] = None,
    ) -> "BenchmarkVisualizer":
        """
        Create a scatter plot showing cost vs performance.
        
        Args:
            score_column: Score column to use for performance axis
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Self for chaining
        """
        df = self.to_dataframe()
        
        if df.empty:
            print("No data to plot")
            return self
        
        # Get score column
        score_columns = [col for col in df.columns if col.startswith("score_") and not col.endswith("_std")]
        if score_column is None and score_columns:
            score_column = score_columns[0]
        
        if score_column not in df.columns or "total_cost_usd" not in df.columns:
            print("Missing required columns for cost efficiency plot")
            return self
        
        fig, ax = self.plt.subplots(figsize=figsize)
        
        # Create scatter plot
        benchmarks = df["benchmark"].unique()
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        colors = self.plt.cm.Set1(np.linspace(0, 1, len(df["model_id"].unique())))
        
        for i, model in enumerate(df["model_id"].unique()):
            model_df = df[df["model_id"] == model]
            for j, benchmark in enumerate(benchmarks):
                bench_df = model_df[model_df["benchmark"] == benchmark]
                if not bench_df.empty:
                    ax.scatter(
                        bench_df["total_cost_usd"],
                        bench_df[score_column],
                        c=[colors[i]],
                        marker=markers[j % len(markers)],
                        s=150,
                        label=f"{model} ({benchmark})" if j == 0 else "",
                        alpha=0.7,
                        edgecolors="black",
                        linewidth=1,
                    )
        
        label = score_column.replace("score_", "").replace("_", " ").title()
        ax.set_xlabel("Total Cost (USD)")
        ax.set_ylabel(label)
        ax.set_title("Cost Efficiency: Score vs Cost")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(alpha=0.3)
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        
        self.plt.show()
        return self
    
    def plot_metrics_summary(
        self,
        figsize: tuple[int, int] = (16, 12),
        save_path: Optional[Path | str] = None,
    ) -> "BenchmarkVisualizer":
        """
        Create a comprehensive summary with multiple subplots.
        
        Args:
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Self for chaining
        """
        df = self.to_dataframe()
        prompt_df = self.to_prompt_dataframe()
        
        if df.empty:
            print("No data to plot")
            return self
        
        fig, axes = self.plt.subplots(2, 2, figsize=figsize)
        
        # 1. Mean scores by model (top left)
        ax = axes[0, 0]
        score_columns = [col for col in df.columns if col.startswith("score_") and not col.endswith("_std")]
        if score_columns:
            # Get mean of all scores
            df_plot = df.groupby("model_id")[score_columns].mean()
            df_plot.plot(kind="bar", ax=ax, width=0.8)
            ax.set_title("Mean Scores by Model")
            ax.set_xlabel("Model")
            ax.set_ylabel("Score")
            ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.tick_params(axis='x', rotation=45)
        
        # 2. Latency comparison (top right)
        ax = axes[0, 1]
        df.plot.bar(x="model_id", y="mean_latency_ms", ax=ax, legend=False, color="coral")
        ax.set_title("Mean Latency by Model")
        ax.set_xlabel("Model")
        ax.set_ylabel("Latency (ms)")
        ax.tick_params(axis='x', rotation=45)
        
        # 3. Token usage (bottom left)
        ax = axes[1, 0]
        df.plot.bar(x="model_id", y="total_tokens", ax=ax, legend=False, color="steelblue")
        ax.set_title("Total Tokens by Model")
        ax.set_xlabel("Model")
        ax.set_ylabel("Tokens")
        ax.tick_params(axis='x', rotation=45)
        
        # 4. Success rate (bottom right)
        ax = axes[1, 1]
        df.plot.bar(x="model_id", y="success_rate", ax=ax, legend=False, color="seagreen")
        ax.set_title("Success Rate by Model")
        ax.set_xlabel("Model")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis='x', rotation=45)
        
        self.plt.suptitle("Benchmark Results Summary", fontsize=16, y=1.02)
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        
        self.plt.show()
        return self
    
    def save_all(
        self,
        output_dir: Path | str,
        prefix: str = "",
        format: str = "png",
    ) -> "BenchmarkVisualizer":
        """
        Generate and save all available charts.
        
        Args:
            output_dir: Directory to save charts
            prefix: Prefix for filenames
            format: Image format (png, pdf, svg)
            
        Returns:
            Self for chaining
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = f"{prefix}_" if prefix else ""
        
        # Generate each chart type
        charts = [
            ("score_comparison", self.plot_score_comparison),
            ("latency_distribution", self.plot_latency_distribution),
            ("metrics_summary", self.plot_metrics_summary),
        ]
        
        for name, func in charts:
            try:
                save_path = output_dir / f"{prefix}{name}.{format}"
                func(save_path=save_path)
            except Exception as e:
                print(f"Warning: Failed to generate {name}: {e}")
        
        # Radar chart (needs at least 3 dimensions)
        df = self.to_dataframe()
        score_columns = [col for col in df.columns if col.startswith("score_") and not col.endswith("_std")]
        if len(score_columns) >= 3:
            try:
                save_path = output_dir / f"{prefix}radar_chart.{format}"
                self.plot_radar_chart(save_path=save_path)
            except Exception as e:
                print(f"Warning: Failed to generate radar chart: {e}")
        
        # Heatmap (needs multiple benchmarks)
        if len(df["benchmark"].unique()) > 1:
            try:
                save_path = output_dir / f"{prefix}benchmark_heatmap.{format}"
                self.plot_benchmark_heatmap(save_path=save_path)
            except Exception as e:
                print(f"Warning: Failed to generate heatmap: {e}")
        
        print(f"\nAll charts saved to: {output_dir}")
        return self
    
    def generate_html_report(
        self,
        output_path: Path | str,
        title: str = "Benchmark Results Report",
    ) -> "BenchmarkVisualizer":
        """
        Generate a comprehensive HTML report with embedded charts.
        
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
        
        # Generate charts as base64
        charts_html = []
        
        def fig_to_base64():
            buf = BytesIO()
            self.plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            self.plt.close()
            return img_base64
        
        # Generate summary chart
        try:
            self.plot_metrics_summary()
            charts_html.append(f'<img src="data:image/png;base64,{fig_to_base64()}" alt="Metrics Summary">')
        except:
            pass
        
        # Generate score comparison
        try:
            self.plot_score_comparison()
            charts_html.append(f'<img src="data:image/png;base64,{fig_to_base64()}" alt="Score Comparison">')
        except:
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
            <span class="metric">📊 Benchmarks: {len(df['benchmark'].unique())}</span>
            <span class="metric">🤖 Models: {len(df['model_id'].unique())}</span>
            <span class="metric">📝 Total Prompts: {df['total_prompts'].sum()}</span>
        </div>
        
        <h2>Results Table</h2>
        {df.to_html(index=False, classes='results-table')}
        
        <h2>Visualizations</h2>
        {''.join(charts_html)}
    </div>
</body>
</html>"""
        
        output_path.write_text(html)
        print(f"Report saved to: {output_path}")
        
        return self
