"""Rich console utilities for pipeline output."""

from dataclasses import dataclass, field
from typing import Callable

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


def plan_summary(datasets: list[DatasetPlanInfo], run_hash: str, num_actors: int | None = None) -> None:
    """Print a summary table of all dataset plans."""
    # Resolve num_actors to actual value (auto-detect if None)
    if num_actors is None:
        import os
        cpu_count = os.cpu_count() or 4
        num_actors = max(2, min(32, int(cpu_count * 0.75)))

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

    for ds in datasets:
        pending = ds.sampled if ds.sampled is not None else ds.pending
        total_pending += pending
        total_cached += ds.cached

        if pending == 0:
            status = "[green]cached[/green]"
        else:
            status = f"[yellow]{pending} to process[/yellow]"

        cached_str = str(ds.cached) if ds.cached > 0 else "-"

        row = [ds.name]
        if has_hf_metadata:
            row.append(ds.hf_size or "-")
            row.append(ds.hf_rows or "-")
        row.extend([
            str(ds.num_shards),
            str(ds.num_files),
            cached_str,
            str(pending) if pending > 0 else "-",
            status,
        ])

        table.add_row(*row)

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


@dataclass
class LiveExecutionStatus:
    """Manages live status display during pipeline execution.

    Shows a compact 2-line display:
    - Line 1: Progress bar for current dataset
    - Line 2: Summary (e.g., "Dataset 3/20 | 5 done, 2 cached, 12 pending")
    """

    datasets: list[DatasetStatus] = field(default_factory=list)
    run_hash: str = ""
    _live: Live | None = field(default=None, repr=False)
    _progress: Progress | None = field(default=None, repr=False)
    _current_task_id: int | None = field(default=None, repr=False)
    _current_index: int = field(default=0, repr=False)

    def _get_summary_counts(self) -> tuple[int, int, int, int]:
        """Get counts of datasets by status."""
        done = sum(1 for ds in self.datasets if ds.status == "complete")
        cached = sum(1 for ds in self.datasets if ds.status == "cached")
        pending = sum(1 for ds in self.datasets if ds.status == "pending")
        processing = sum(1 for ds in self.datasets if ds.status == "processing")
        return done, cached, pending, processing

    def _build_summary_line(self) -> Text:
        """Build a compact summary line."""
        done, cached, pending, processing = self._get_summary_counts()
        total = len(self.datasets)
        current = done + cached + processing

        parts = []
        if done > 0:
            parts.append(f"[green]{done} done[/green]")
        if cached > 0:
            parts.append(f"[dim]{cached} cached[/dim]")
        if pending > 0:
            parts.append(f"[dim]{pending} pending[/dim]")

        summary = f"[bold]Dataset {current}/{total}[/bold]"
        if parts:
            summary += " | " + ", ".join(parts)

        return Text.from_markup(summary)

    def _build_display(self) -> Group:
        """Build the compact live display (no panel border)."""
        elements = []

        # Add progress bar if processing
        if self._progress is not None:
            elements.append(self._progress)

        # Add summary line
        elements.append(self._build_summary_line())

        return Group(*elements)

    def start(self) -> None:
        """Start the live display."""
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

    def refresh(self) -> None:
        """Refresh the live display."""
        if self._live:
            self._live.update(self._build_display())

    def start_dataset(self, name: str) -> None:
        """Mark a dataset as processing and create progress bar."""
        for i, ds in enumerate(self.datasets):
            if ds.name == name:
                ds.status = "processing"
                self._current_index = i
                # Create progress bar for this dataset
                self._progress = Progress(
                    SpinnerColumn(),
                    TextColumn(f"[cyan]{name}[/cyan]"),
                    BarColumn(bar_width=30),
                    MofNCompleteColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    console=console,
                    transient=True,
                )
                self._current_task_id = self._progress.add_task(
                    "Processing", total=ds.total_shards
                )
                break
        self.refresh()

    def advance_dataset(self, name: str) -> None:
        """Advance progress for a dataset."""
        for ds in self.datasets:
            if ds.name == name:
                ds.completed_shards += 1
                if self._progress and self._current_task_id is not None:
                    self._progress.advance(self._current_task_id)
                break
        self.refresh()

    def complete_dataset(self, name: str) -> None:
        """Mark a dataset as complete."""
        for ds in self.datasets:
            if ds.name == name:
                ds.status = "complete"
                ds.completed_shards = ds.total_shards
                break
        self._progress = None
        self._current_task_id = None
        self.refresh()

    def cache_dataset(self, name: str) -> None:
        """Mark a dataset as cached."""
        for ds in self.datasets:
            if ds.name == name:
                ds.status = "cached"
                ds.completed_shards = ds.total_shards
                break
        self.refresh()


def create_live_status(
    datasets: list[tuple[str, int]], run_hash: str
) -> LiveExecutionStatus:
    """Create a live execution status tracker.

    Args:
        datasets: List of (name, total_shards) tuples
        run_hash: The run hash to display
    """
    return LiveExecutionStatus(
        datasets=[DatasetStatus(name=name, total_shards=total) for name, total in datasets],
        run_hash=run_hash,
    )
