"""Command-line interface for the tokenization pipeline using tyro."""

import sys
from dataclasses import dataclass
from typing import Annotated

import ray
import tyro

from nemotron.data_prep import console as con
from nemotron.data_prep.pipeline import tokenize_to_binidx


@dataclass
class RunCommand:
    """Run the tokenization pipeline."""

    config: str
    """Path to blend config JSON file."""

    output_dir: str
    """Output directory (local or cloud URI like s3://, gs://)."""

    sample: str | None = None
    """Shard-level sample specification ('10%' or integer count)."""

    sample_seed: int = 42
    """Random seed for sampling."""

    num_actors: int | None = None
    """Number of ShardProcessor Ray actors. If None, auto-detects from available CPUs."""

    force: bool = False
    """Create new run namespace (ignore existing)."""

    ray_address: str = "auto"
    """Ray cluster address."""


@dataclass
class StatusCommand:
    """Show pipeline status."""

    output_dir: str
    """Output directory to check."""

    run_hash: str | None = None
    """Specific run hash to check (shows all if not specified)."""


@dataclass
class VerifyCommand:
    """Verify output integrity."""

    output_dir: str
    """Output directory containing the run."""

    run_hash: str
    """Run hash to verify."""

    check_checksums: bool = False
    """Verify file checksums (slower but thorough)."""


Command = Annotated[RunCommand, tyro.conf.subcommand("run")] | Annotated[StatusCommand, tyro.conf.subcommand("status")] | Annotated[VerifyCommand, tyro.conf.subcommand("verify")]


def _get_num_actors(num_actors: int | None) -> int:
    """Get number of actors, auto-detecting from Ray cluster if not specified."""
    if num_actors is not None:
        return num_actors

    # Auto-detect from Ray cluster resources
    resources = ray.cluster_resources()
    num_cpus = int(resources.get("CPU", 1))
    # Use max(1, cpus - 1) to leave one CPU for the driver/overhead
    return max(1, num_cpus - 1)


def run_command(args: RunCommand) -> None:
    """Execute the run command."""
    # Initialize Ray
    try:
        ray.init(args.ray_address, ignore_reinit_error=True)
    except Exception:
        # Fall back to local mode if cluster not available
        ray.init(ignore_reinit_error=True)

    # Auto-detect num_actors after Ray is initialized
    num_actors = _get_num_actors(args.num_actors)

    # Parse sample arg
    sample: str | int | None = None
    if args.sample:
        if args.sample.endswith("%"):
            sample = args.sample
        else:
            try:
                sample = int(args.sample)
            except ValueError:
                sample = args.sample

    try:
        result = tokenize_to_binidx(
            config_path=args.config,
            output_dir=args.output_dir,
            sample=sample,
            sample_seed=args.sample_seed,
            num_actors=num_actors,
            force=args.force,
        )

        con.pipeline_complete(
            run_hash=result.run_hash,
            output_dir=result.output_dir,
            total_tokens=result.total_tokens,
            total_sequences=result.total_sequences,
            elapsed_sec=result.elapsed_sec,
        )

    except Exception as e:
        con.error(str(e))
        sys.exit(1)
    finally:
        ray.shutdown()


def status_command(args: StatusCommand) -> None:
    """Execute the status command."""
    from nemotron.data_prep.filesystem import get_filesystem, read_json

    fs, base_path = get_filesystem(args.output_dir)
    runs_dir = f"{base_path}/runs"

    try:
        if args.run_hash:
            run_hashes = [args.run_hash]
        else:
            # List all runs
            try:
                run_hashes = [
                    d.split("/")[-1] for d in fs.ls(runs_dir) if fs.isdir(d)
                ]
            except Exception:
                run_hashes = []

        if not run_hashes:
            print("No runs found.")
            return

        for run_hash in run_hashes:
            run_dir = f"{runs_dir}/{run_hash}"
            manifest_path = f"{run_dir}/manifest.json"

            print(f"\nRun: {run_hash}")
            print("-" * 40)

            if fs.exists(manifest_path):
                manifest = read_json(fs, manifest_path)
                print(f"  Generated: {manifest.get('generated_at', 'unknown')}")

                for name, dataset_info in manifest.get("datasets", {}).items():
                    status = dataset_info.get("status", "unknown")
                    completed = dataset_info.get("num_shards_completed", 0)
                    total = dataset_info.get("num_shards", 0)
                    tokens = dataset_info.get("total_tokens", 0)

                    print(f"  {name}:")
                    print(f"    Status: {status}")
                    print(f"    Shards: {completed}/{total}")
                    print(f"    Tokens: {tokens:,}")
            else:
                print("  Manifest not found (run may be in progress)")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _stream_checksum(fs, path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    """
    Compute streaming xxh64 checksum without loading full file into memory.

    Args:
        fs: fsspec filesystem
        path: Path to file
        chunk_size: Read chunk size (default 8MB)

    Returns:
        Checksum string in format "xxh64:{hexdigest}"
    """
    import xxhash

    hasher = xxhash.xxh64()
    with fs.open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return f"xxh64:{hasher.hexdigest()}"


def verify_command(args: VerifyCommand) -> None:
    """Execute the verify command."""
    from nemotron.data_prep.filesystem import get_filesystem, read_json

    fs, base_path = get_filesystem(args.output_dir)
    run_dir = f"{base_path}/runs/{args.run_hash}"
    manifest_path = f"{run_dir}/manifest.json"

    try:
        if not fs.exists(manifest_path):
            print(f"Run {args.run_hash} not found", file=sys.stderr)
            sys.exit(1)

        manifest = read_json(fs, manifest_path)
        errors = []

        for name, dataset_info in manifest.get("datasets", {}).items():
            plan_hash = dataset_info.get("plan_hash")
            if not plan_hash:
                continue

            dataset_dir = f"{run_dir}/datasets/{name}/{plan_hash}"
            receipts_dir = f"{dataset_dir}/receipts"

            print(f"\nVerifying {name}...")

            try:
                receipt_files = fs.glob(f"{receipts_dir}/shard_*.json")
            except Exception:
                receipt_files = []

            for receipt_path in receipt_files:
                try:
                    receipt = read_json(fs, receipt_path)
                    shard_id = receipt.get("shard_id", "unknown")
                    shard_ok = True  # Track per-shard status

                    # Check file existence
                    if receipt["stats"]["num_sequences"] > 0:
                        bin_path = f"{dataset_dir}/{receipt['files']['bin']['path']}"
                        idx_path = f"{dataset_dir}/{receipt['files']['idx']['path']}"

                        if not fs.exists(bin_path):
                            errors.append(f"{shard_id}: bin file missing")
                            shard_ok = False
                            continue

                        if not fs.exists(idx_path):
                            errors.append(f"{shard_id}: idx file missing")
                            shard_ok = False
                            continue

                        # Verify checksums if requested
                        if args.check_checksums:
                            # Check bin file with streaming checksum
                            expected_bin = receipt["files"]["bin"]["checksum"]
                            if expected_bin and not expected_bin.endswith("empty"):
                                actual = _stream_checksum(fs, bin_path)
                                if actual != expected_bin:
                                    errors.append(f"{shard_id}: bin checksum mismatch")
                                    shard_ok = False

                            # Check idx file with streaming checksum
                            expected_idx = receipt["files"]["idx"]["checksum"]
                            if expected_idx and not expected_idx.endswith("empty"):
                                actual = _stream_checksum(fs, idx_path)
                                if actual != expected_idx:
                                    errors.append(f"{shard_id}: idx checksum mismatch")
                                    shard_ok = False

                    # Only print OK if no errors for this shard
                    if shard_ok:
                        print(f"  {shard_id}: OK")
                    else:
                        print(f"  {shard_id}: FAILED")

                except Exception as e:
                    errors.append(f"{receipt_path}: {e}")

        if errors:
            print(f"\nVerification failed with {len(errors)} errors:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        else:
            print("\nVerification passed!")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    args = tyro.cli(Command)

    if isinstance(args, RunCommand):
        run_command(args)
    elif isinstance(args, StatusCommand):
        status_command(args)
    elif isinstance(args, VerifyCommand):
        verify_command(args)


if __name__ == "__main__":
    main()
