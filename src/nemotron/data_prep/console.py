# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rich console utilities for pipeline output."""

from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

console = Console()


@dataclass
class DatasetPlanInfo:
    """Info about a dataset's plan for display."""

    name: str
    plan_hash: str
    num_shards: int
    num_files: int
    pending: int
    cached: int
    cached_tokens: int
    cached_sequences: int
    sampled: int | None = None
    # HuggingFace metadata
    hf_rows: str | None = None  # Formatted string like "3.79B"
    hf_size: str | None = None  # Formatted string like "4.58 TB"


def planning_header() -> None:
    """Print the planning phase header."""
    console.print()
    console.print("[bold]Planning...[/bold]")


def plan_summary(
    datasets: list[DatasetPlanInfo], run_hash: str
) -> None:
    """Print a summary table of all dataset plans."""
    # Auto-detect num_actors from Ray cluster
    import os

    import ray

    try:
        cluster_cpus = int(ray.cluster_resources().get("CPU", 0))
        if cluster_cpus > 0:
            num_actors = cluster_cpus
        else:
            num_actors = os.cpu_count() or 4
    except Exception:
        num_actors = os.cpu_count() or 4
    num_actors = max(2, num_actors)

    console.print()

    # Check if any dataset has HF metadata
    has_hf_metadata = any(ds.hf_rows or ds.hf_size for ds in datasets)

    table = Table(title="Execution Plan", show_header=True)
    table.add_column("Dataset", style="cyan", no_wrap=True)
    if has_hf_metadata:
        table.add_column("Size", justify="right", style="dim")
        table.add_column("Rows", justify="right", style="dim")
    table.add_column("Shards", justify="right")
    table.add_column("Files", justify="right")
    table.add_column("Cached", justify="right", style="dim")
    table.add_column("Pending", justify="right")
    table.add_column("Status")

    total_pending = 0
    total_cached = 0

    # Collect data for W&B table
    wandb_rows = []

    for ds in datasets:
        pending = ds.sampled if ds.sampled is not None else ds.pending
        total_pending += pending
        total_cached += ds.cached

        if pending == 0:
            status = "[green]cached[/green]"
            status_plain = "cached"
        else:
            status = f"[yellow]{pending} to process[/yellow]"
            status_plain = f"{pending} to process"

        cached_str = str(ds.cached) if ds.cached > 0 else "-"

        row = [ds.name]
        if has_hf_metadata:
            row.append(ds.hf_size or "-")
            row.append(ds.hf_rows or "-")
        row.extend(
            [
                str(ds.num_shards),
                str(ds.num_files),
                cached_str,
                str(pending) if pending > 0 else "-",
                status,
            ]
        )

        table.add_row(*row)

        # Build W&B row (without Rich markup)
        wandb_row = [ds.name]
        if has_hf_metadata:
            wandb_row.extend([ds.hf_size or "-", ds.hf_rows or "-"])
        wandb_row.extend(
            [
                ds.num_shards,
                ds.num_files,
                ds.cached if ds.cached > 0 else 0,
                pending if pending > 0 else 0,
                status_plain,
            ]
        )
        wandb_rows.append(wandb_row)

    console.print(table)

    # Summary line
    actors_info = f" using [cyan]{num_actors} workers[/cyan]"
    if total_pending == 0:
        console.print(f"\n[green]All shards cached.[/green] Run hash: [yellow]{run_hash}[/yellow]")
    else:
        console.print(
            f"\n[bold]Will process {total_pending} shard(s)[/bold]{actors_info} "
            f"({total_cached} cached). Run hash: [yellow]{run_hash}[/yellow]"
        )
    console.print()

    # Log to W&B if active
    _log_plan_to_wandb(
        wandb_rows, has_hf_metadata, run_hash, num_actors, total_pending, total_cached
    )


def _log_plan_to_wandb(
    rows: list[list],
    has_hf_metadata: bool,
    run_hash: str,
    num_actors: int,
    total_pending: int,
    total_cached: int,
) -> None:
    """Log execution plan as W&B Table."""
    try:
        import wandb

        if wandb.run is None:
            return

        # Build column names
        columns = ["Dataset"]
        if has_hf_metadata:
            columns.extend(["Size", "Rows"])
        columns.extend(["Shards", "Files", "Cached", "Pending", "Status"])

        # Create W&B table
        wandb_table = wandb.Table(columns=columns, data=rows)

        # Log the table
        wandb.log(
            {
                "data_prep/execution_plan": wandb_table,
                "data_prep/run_hash": run_hash,
                "data_prep/num_workers": num_actors,
                "data_prep/total_pending_shards": total_pending,
                "data_prep/total_cached_shards": total_cached,
                "data_prep/num_datasets": len(rows),
            }
        )
    except ImportError:
        pass
    except Exception:
        pass  # Don't fail pipeline on W&B errors


def execution_header() -> None:
    """Print the execution phase header."""
    console.print("[bold]Processing...[/bold]")
    console.print()


def dataset_progress_start(name: str) -> None:
    """Print dataset processing start."""
    console.print(f"[cyan]{name}[/cyan]")


def dataset_complete(num_shards: int, num_sequences: int, num_tokens: int) -> None:
    """Print dataset completion stats after processing."""
    console.print(
        f"  [green]Complete:[/green] {num_shards} shards, "
        f"{num_sequences:,} sequences, {num_tokens:,} tokens"
    )
    console.print()


def dataset_cached(num_shards: int, num_sequences: int, num_tokens: int) -> None:
    """Print cached dataset stats (all shards already complete)."""
    console.print(
        f"  [dim]Cached:[/dim] {num_shards} shards, "
        f"{num_sequences:,} sequences, {num_tokens:,} tokens"
    )
    console.print()


def create_progress() -> Progress:
    """Create a progress bar for shard processing."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def pipeline_complete(
    run_hash: str,
    output_dir: str,
    total_tokens: int,
    total_sequences: int,
    elapsed_sec: float,
) -> None:
    """Print pipeline completion summary."""
    console.print()

    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value")

    table.add_row("Run hash", f"[yellow]{run_hash}[/yellow]")
    table.add_row("Output", f"{output_dir}/runs/{run_hash}")
    table.add_row("Total tokens", f"[green]{total_tokens:,}[/green]")
    table.add_row("Total sequences", f"{total_sequences:,}")
    table.add_row("Time", f"{elapsed_sec:.1f}s")

    console.print(
        Panel(table, title="[bold green]Pipeline Complete[/bold green]", border_style="green")
    )


def error(message: str) -> None:
    """Print an error message."""
    console.print(f"[red bold]Error:[/red bold] {message}")


@dataclass
class DatasetStatus:
    """Status of a single dataset during execution."""

    name: str
    total_shards: int
    completed_shards: int = 0
    status: str = "pending"  # pending, processing, cached, complete
    tokens: int = 0  # Tokens processed for this dataset
    # Compute metrics (updated during processing)
    rows_processed: int = 0  # Number of rows processed
    throughput: float = 0.0  # Rows per second
    start_time: float = 0.0  # When processing started (timestamp)
    # Phase tracking for progress visibility
    phase: str = ""  # Current phase: downloading, reading, tokenizing, writing
    phase_detail: str = ""  # Optional detail (e.g., "3/10 files", "15K rows")


@dataclass
class BottleneckMetrics:
    """Rolling window of shard timing metrics for bottleneck identification."""

    # Cumulative timing (seconds) from all shards
    time_download: float = 0.0
    time_read: float = 0.0
    time_tokenize: float = 0.0
    time_write: float = 0.0
    time_total: float = 0.0
    # Counts
    num_shards: int = 0
    num_errors: int = 0
    # Peak memory (if available)
    peak_memory_mb: float = 0.0

    def add_shard_stats(self, stats: dict) -> None:
        """Update metrics from a shard's stats dict."""
        self.time_download += stats.get("time_download_sec", 0.0)
        self.time_read += stats.get("time_read_sec", 0.0)
        self.time_tokenize += stats.get("time_tokenize_sec", 0.0)
        self.time_write += stats.get("time_write_sec", 0.0)
        self.time_total += stats.get("time_total_sec", 0.0)
        self.num_shards += 1
        self.num_errors += stats.get("num_errors", 0)
        if "peak_memory_mb" in stats:
            self.peak_memory_mb = max(self.peak_memory_mb, stats["peak_memory_mb"])

    def get_percentages(self) -> dict[str, float]:
        """Get time breakdown as percentages."""
        total = self.time_download + self.time_read + self.time_tokenize + self.time_write
        if total <= 0:
            return {"download_pct": 0, "read_pct": 0, "tokenize_pct": 0, "write_pct": 0}
        return {
            "download_pct": (self.time_download / total) * 100,
            "read_pct": (self.time_read / total) * 100,
            "tokenize_pct": (self.time_tokenize / total) * 100,
            "write_pct": (self.time_write / total) * 100,
        }


@dataclass
class LiveExecutionStatus:
    """Manages live status display during pipeline execution.

    Shows a multi-line display for parallel processing:
    - Line 1: Overall progress bar (total shards across all datasets)
    - Line 2+: Active datasets being processed (cycles through 3 at a time)
    - Last line: Summary stats with page indicator
    """

    datasets: list[DatasetStatus] = field(default_factory=list)
    run_hash: str = ""
    _live: Live | None = field(default=None, repr=False)
    _progress: Progress | None = field(default=None, repr=False)
    _overall_task_id: int | None = field(default=None, repr=False)
    _current_page: int = field(default=0, repr=False)  # Current page for cycling display
    _page_cycle_counter: int = field(default=0, repr=False)  # Counter for auto-cycling
    _wandb_step: int = field(default=0, repr=False)
    _last_wandb_log_time: float = field(default=0.0, repr=False)
    _wandb_log_interval: float = field(default=10.0, repr=False)  # Log every 10 seconds
    _start_time: float = field(default=0.0, repr=False)  # Pipeline start time
    _total_tokens: int = field(default=0, repr=False)  # Cumulative tokens processed
    _max_display: int = field(default=3, repr=False)  # Max datasets to show per page
    _bottleneck_metrics: BottleneckMetrics = field(default_factory=BottleneckMetrics, repr=False)

    def _get_summary_counts(self) -> tuple[int, int, int, int]:
        """Get counts of datasets by status."""
        done = sum(1 for ds in self.datasets if ds.status == "complete")
        cached = sum(1 for ds in self.datasets if ds.status == "cached")
        pending = sum(1 for ds in self.datasets if ds.status == "pending")
        processing = sum(1 for ds in self.datasets if ds.status == "processing")
        return done, cached, pending, processing

    def _get_total_shards_progress(self) -> tuple[int, int]:
        """Get total shards completed and total shards across all datasets."""
        total_completed = sum(ds.completed_shards for ds in self.datasets)
        total_shards = sum(ds.total_shards for ds in self.datasets)
        return total_completed, total_shards

    def _log_progress_to_wandb(self, force: bool = False) -> None:
        """Log current progress to W&B for charts.

        Logs both aggregate metrics and per-dataset progress, creating
        a chart for each dataset showing shards completed over time.

        Args:
            force: If True, log immediately regardless of time throttling.
                   Used for final completion to ensure we capture 100%.
        """
        import logging
        import time as time_module

        logger = logging.getLogger(__name__)

        try:
            import wandb

            if wandb.run is None:
                return

            # Time-based throttling: only log every N seconds unless forced
            current_time = time_module.time()
            if not force and (current_time - self._last_wandb_log_time) < self._wandb_log_interval:
                return

            self._last_wandb_log_time = current_time

            done, cached, pending, processing = self._get_summary_counts()
            total = len(self.datasets)
            completed = done + cached

            # Calculate throughput
            elapsed = current_time - self._start_time if self._start_time > 0 else 0
            tokens_per_sec = self._total_tokens / elapsed if elapsed > 0 else 0

            self._wandb_step += 1

            # Define metrics on first call to ensure W&B creates charts
            if self._wandb_step == 1:
                try:
                    # Aggregate metrics use elapsed_sec as x-axis
                    wandb.define_metric("data_prep/progress/*", step_metric="data_prep/progress/elapsed_sec")
                    wandb.define_metric("data_prep/bottleneck/*", step_metric="data_prep/progress/elapsed_sec")
                    # Per-dataset metrics also use elapsed_sec as x-axis
                    wandb.define_metric("data_prep/datasets/*", step_metric="data_prep/progress/elapsed_sec")
                except Exception:
                    pass  # define_metric may fail on older W&B versions

            # Build log dict with aggregate metrics
            log_data = {
                "data_prep/progress/datasets_completed": completed,
                "data_prep/progress/datasets_total": total,
                "data_prep/progress/datasets_done": done,
                "data_prep/progress/datasets_cached": cached,
                "data_prep/progress/datasets_pending": pending,
                "data_prep/progress/completion_pct": (completed / total * 100)
                if total > 0
                else 0,
                "data_prep/progress/tokens": self._total_tokens,
                "data_prep/progress/tokens_per_sec": tokens_per_sec,
                "data_prep/progress/elapsed_sec": elapsed,
            }

            # Add per-dataset progress metrics
            # This creates a separate chart for each dataset showing shards completed
            for ds in self.datasets:
                # Sanitize dataset name for W&B metric key (replace special chars)
                safe_name = ds.name.replace("/", "_").replace("-", "_").replace(".", "_")
                progress_pct = (ds.completed_shards / ds.total_shards * 100) if ds.total_shards > 0 else 0

                log_data[f"data_prep/datasets/{safe_name}/shards_completed"] = ds.completed_shards
                log_data[f"data_prep/datasets/{safe_name}/shards_total"] = ds.total_shards
                log_data[f"data_prep/datasets/{safe_name}/progress_pct"] = progress_pct
                log_data[f"data_prep/datasets/{safe_name}/tokens"] = ds.tokens

            # Add bottleneck metrics if we have timing data
            if self._bottleneck_metrics.num_shards > 0:
                pcts = self._bottleneck_metrics.get_percentages()
                log_data.update(
                    {
                        "data_prep/bottleneck/download_pct": pcts["download_pct"],
                        "data_prep/bottleneck/read_pct": pcts["read_pct"],
                        "data_prep/bottleneck/tokenize_pct": pcts["tokenize_pct"],
                        "data_prep/bottleneck/write_pct": pcts["write_pct"],
                        "data_prep/bottleneck/shards_processed": self._bottleneck_metrics.num_shards,
                        "data_prep/bottleneck/errors": self._bottleneck_metrics.num_errors,
                    }
                )
                if self._bottleneck_metrics.peak_memory_mb > 0:
                    log_data["data_prep/resources/peak_memory_mb"] = (
                        self._bottleneck_metrics.peak_memory_mb
                    )

            wandb.log(log_data, commit=True)  # Force immediate sync to W&B
            logger.debug(f"[W&B] Logged data_prep metrics: step={self._wandb_step}, tokens={self._total_tokens}, datasets={len(self.datasets)}")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"[W&B] Failed to log metrics: {e}")

    def _build_summary_line(self) -> Text:
        """Build a compact summary line."""
        done, cached, pending, processing = self._get_summary_counts()
        total = len(self.datasets)
        completed_shards, total_shards = self._get_total_shards_progress()

        parts = []
        if done > 0:
            parts.append(f"[green]{done} done[/green]")
        if cached > 0:
            parts.append(f"[dim]{cached} cached[/dim]")
        if processing > 0:
            parts.append(f"[cyan]{processing} active[/cyan]")

        summary = f"[bold]Datasets {done + cached}/{total}[/bold]"
        if parts:
            summary += " | " + ", ".join(parts)

        return Text.from_markup(summary)

    def _format_tokens(self, tokens: int) -> str:
        """Format token count in human-readable format."""
        if tokens >= 1_000_000_000:
            return f"{tokens / 1_000_000_000:.1f}B"
        elif tokens >= 1_000_000:
            return f"{tokens / 1_000_000:.1f}M"
        elif tokens >= 1_000:
            return f"{tokens / 1_000:.1f}K"
        return str(tokens)

    def _format_throughput(self, rows_per_sec: float) -> str:
        """Format throughput in human-readable format."""
        if rows_per_sec >= 1_000:
            return f"{rows_per_sec / 1_000:.1f}K/s"
        elif rows_per_sec >= 1:
            return f"{rows_per_sec:.0f}/s"
        elif rows_per_sec > 0:
            return f"{rows_per_sec:.2f}/s"
        return "-"

    def _build_active_datasets_table(self) -> Table | None:
        """Build a table display for currently active datasets.

        Cycles through pages of datasets like an airport arrival board.
        Shows max 3 datasets per page with compute metrics.
        """
        active = [ds for ds in self.datasets if ds.status == "processing"]

        if not active:
            return None

        # Calculate pagination
        total_pages = (len(active) + self._max_display - 1) // self._max_display
        if total_pages == 0:
            total_pages = 1

        # Ensure current page is valid
        if self._current_page >= total_pages:
            self._current_page = 0

        # Get datasets for current page
        start_idx = self._current_page * self._max_display
        end_idx = min(start_idx + self._max_display, len(active))
        page_datasets = active[start_idx:end_idx]

        # Build page indicator for title
        if total_pages > 1:
            dots = " ".join("●" if i == self._current_page else "○" for i in range(total_pages))
            title = f"Active Tasks  [dim]{dots}[/dim]"
        else:
            title = "Active Tasks"

        # Create table
        table = Table(
            title=title,
            box=None,
            show_header=True,
            header_style="dim",
            padding=(0, 1),
            collapse_padding=True,
        )
        table.add_column("Dataset", style="cyan", no_wrap=True, min_width=20)
        table.add_column("Phase", justify="left", min_width=12)
        table.add_column("Progress", justify="center", min_width=8)
        table.add_column("Tokens", justify="right", min_width=8)
        table.add_column("Speed", justify="right", min_width=8)

        for ds in page_datasets:
            # Phase indicator with detail
            if ds.phase:
                phase_str = f"[yellow]{ds.phase}[/yellow]"
                if ds.phase_detail:
                    phase_str += f" [dim]{ds.phase_detail}[/dim]"
            else:
                # Show elapsed time when no phase info yet
                if ds.start_time > 0:
                    import time as time_module
                    elapsed = time_module.time() - ds.start_time
                    phase_str = f"[dim]starting... {elapsed:.0f}s[/dim]"
                else:
                    phase_str = "[dim]starting...[/dim]"

            # Progress indicator
            if ds.completed_shards >= ds.total_shards:
                progress = "[green]✓[/green]"
            else:
                progress = f"{ds.completed_shards}/{ds.total_shards}"

            # Format metrics
            tokens_str = self._format_tokens(ds.tokens) if ds.tokens > 0 else "-"
            speed_str = self._format_throughput(ds.throughput)

            table.add_row(ds.name, phase_str, progress, tokens_str, speed_str)

        return table

    def _build_active_datasets_display(self) -> list[Text]:
        """Build display lines for currently active datasets (legacy fallback)."""
        # This method is kept for compatibility but we now prefer the table
        active = [ds for ds in self.datasets if ds.status == "processing"]
        lines = []

        if not active:
            return lines

        # Calculate pagination
        total_pages = (len(active) + self._max_display - 1) // self._max_display
        if total_pages == 0:
            total_pages = 1

        # Ensure current page is valid
        if self._current_page >= total_pages:
            self._current_page = 0

        # Get datasets for current page
        start_idx = self._current_page * self._max_display
        end_idx = min(start_idx + self._max_display, len(active))
        page_datasets = active[start_idx:end_idx]

        for ds in page_datasets:
            status_char = "●" if ds.completed_shards > 0 else "○"

            if ds.completed_shards >= ds.total_shards:
                line = f"  [green]{status_char}[/green] [green]{ds.name}[/green] [dim]✓[/dim]"
            else:
                prog = f"{ds.completed_shards}/{ds.total_shards}"
                line = f"  [cyan]{status_char}[/cyan] {ds.name} [dim]{prog}[/dim]"
            lines.append(Text.from_markup(line))

        # Add page indicator if multiple pages
        if total_pages > 1:
            dots = " ".join("●" if i == self._current_page else "○" for i in range(total_pages))
            page_indicator = f"  [dim]{dots}  ({self._current_page + 1}/{total_pages})[/dim]"
            lines.append(Text.from_markup(page_indicator))

        return lines

    def _build_display(self) -> Group:
        """Build the live display for parallel execution."""
        elements = []

        # Add overall progress bar
        if self._progress is not None:
            elements.append(self._progress)

        # Add active datasets table (preferred) or fallback to text lines
        active_table = self._build_active_datasets_table()
        if active_table is not None:
            elements.append(active_table)

        # Add summary line
        elements.append(self._build_summary_line())

        return Group(*elements)

    def start(self) -> None:
        """Start the live display."""
        import time as time_module

        self._start_time = time_module.time()

        # Calculate total shards across all datasets
        total_shards = sum(ds.total_shards for ds in self.datasets)

        # Create overall progress bar
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Overall[/bold blue]"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        self._overall_task_id = self._progress.add_task("Processing", total=total_shards)

        self._live = Live(
            self._build_display(),
            console=console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None
        self._progress = None
        self._overall_task_id = None

    def refresh(self) -> None:
        """Refresh the live display and cycle pages."""
        if self._live:
            # Auto-cycle pages every ~2 seconds (8 refresh calls at 4 fps)
            self._page_cycle_counter += 1
            if self._page_cycle_counter >= 8:
                self._page_cycle_counter = 0
                active = [ds for ds in self.datasets if ds.status == "processing"]
                total_pages = (len(active) + self._max_display - 1) // self._max_display
                if total_pages > 1:
                    self._current_page = (self._current_page + 1) % total_pages

            self._live.update(self._build_display())

    def start_dataset(self, name: str) -> None:
        """Mark a dataset as processing (for parallel execution)."""
        import time as time_module

        for ds in self.datasets:
            if ds.name == name:
                ds.status = "processing"
                ds.start_time = time_module.time()
                break
        self.refresh()

    def advance_dataset(self, name: str) -> None:
        """Advance progress for a dataset."""
        for ds in self.datasets:
            if ds.name == name:
                ds.completed_shards += 1
                # Advance overall progress bar
                if self._progress and self._overall_task_id is not None:
                    self._progress.advance(self._overall_task_id)
                break
        self.refresh()

    def complete_dataset(self, name: str) -> None:
        """Mark a dataset as complete."""
        for ds in self.datasets:
            if ds.name == name:
                ds.status = "complete"
                ds.completed_shards = ds.total_shards
                break
        self.refresh()
        # Force log if this is the last dataset to ensure we capture 100%
        done, cached, pending, processing = self._get_summary_counts()
        is_last = pending == 0 and processing == 0
        self._log_progress_to_wandb(force=is_last)

    def cache_dataset(self, name: str) -> None:
        """Mark a dataset as cached."""
        for ds in self.datasets:
            if ds.name == name:
                ds.status = "cached"
                ds.completed_shards = ds.total_shards
                break
        self.refresh()
        # Force log if this is the last dataset to ensure we capture 100%
        done, cached, pending, _ = self._get_summary_counts()
        is_last = pending == 0
        self._log_progress_to_wandb(force=is_last)

    def report_tokens(self, name: str, tokens: int) -> None:
        """Report tokens processed for a dataset (for throughput tracking).

        Args:
            name: Dataset name
            tokens: Number of tokens processed
        """
        for ds in self.datasets:
            if ds.name == name:
                ds.tokens = tokens
                break
        self._total_tokens = sum(ds.tokens for ds in self.datasets)
        # Log progress with updated token count
        self._log_progress_to_wandb()

    def report_metrics(self, name: str, rows: int = 0, tokens: int = 0) -> None:
        """Report compute metrics for a dataset.

        Args:
            name: Dataset name
            rows: Number of rows processed
            tokens: Number of tokens processed
        """
        import time as time_module

        for ds in self.datasets:
            if ds.name == name:
                ds.rows_processed = rows
                ds.tokens = tokens
                # Calculate throughput
                if ds.start_time > 0:
                    elapsed = time_module.time() - ds.start_time
                    if elapsed > 0:
                        ds.throughput = rows / elapsed
                break

        self._total_tokens = sum(ds.tokens for ds in self.datasets)
        self.refresh()
        self._log_progress_to_wandb()

    def report_shard_timing(self, stats: dict) -> None:
        """Report timing stats from a completed shard for bottleneck analysis.

        Called by Ray Data executor's on_result callback with shard stats dict
        containing timing breakdowns: time_download_sec, time_read_sec,
        time_tokenize_sec, time_write_sec, time_total_sec.

        Args:
            stats: Shard stats dict with timing information
        """
        self._bottleneck_metrics.add_shard_stats(stats)
        # Log to W&B with throttling for live updates
        self._log_progress_to_wandb()

    def report_phase(self, name: str, phase: str, detail: str = "") -> None:
        """Report the current processing phase for a dataset.

        Used to show what's happening during long-running operations like
        HuggingFace downloads.

        Args:
            name: Dataset name
            phase: Current phase (e.g., "downloading", "reading", "tokenizing", "writing")
            detail: Optional detail string (e.g., "3/10 files", "15K rows")
        """
        for ds in self.datasets:
            if ds.name == name:
                ds.phase = phase
                ds.phase_detail = detail
                break
        self.refresh()


def create_live_status(datasets: list[tuple[str, int]], run_hash: str) -> LiveExecutionStatus:
    """Create a live execution status tracker.

    Args:
        datasets: List of (name, total_shards) tuples
        run_hash: The run hash to display
    """
    return LiveExecutionStatus(
        datasets=[DatasetStatus(name=name, total_shards=total) for name, total in datasets],
        run_hash=run_hash,
    )
