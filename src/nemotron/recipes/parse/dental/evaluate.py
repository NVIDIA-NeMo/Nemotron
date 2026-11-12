"""Dental reference evaluation with matched generation and task-specific scores."""

import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from .metrics import components, successful_answer
from nemotron.recipes.parse.training.config import worker_env


def evaluation_key(model, rows, samples, processor_source=None):
    """Identify an immutable evaluation, including model and image file states."""
    model = Path(model).absolute()
    processor = Path(processor_source or model).absolute()
    files = list(model.glob("*.safetensors")) + list(model.glob("*.json"))
    files += list(processor.glob("*token*.json"))
    files += [Path(row["image"]) for row in rows]
    states = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    payload = dict(
        version=1,
        files=sorted(set(states)),
        rows=[dict(row, image=str(Path(row["image"]).resolve())) for row in rows],
        samples=samples,
        temperature=1.0,
        top_p=1.0,
        seed=1231031,
        max_tokens=1536,
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def evaluate(model, rows, output, samples=8, processor_source=None):
    """Run vLLM in a generation kernel, then persist every answer and metric.

    samples=1 still uses temperature 1: it is a sampled first attempt.
    All before/after comparisons must use the same sampling settings.
    """
    from PIL import Image
    from vllm import LLM, SamplingParams

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "request.json").write_text(
        json.dumps({"key": evaluation_key(model, rows, samples, processor_source)})
    )
    view = output / "model"
    view.mkdir()
    model = Path(model).absolute()
    for source in model.iterdir():
        if source.is_file() and source.name != "config.json":
            (view / source.name).symlink_to(source)
    config = json.loads((model / "config.json").read_text())
    config["architectures"] = ["RadioMBartPrefixForCausalLM"]
    (view / "config.json").write_text(json.dumps(config))
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "preprocessor_config.json",
        "processor_config.json",
    ):
        source = Path(processor_source or model) / name
        if source.is_file() and not (view / name).exists():
            (view / name).symlink_to(source.absolute())
    llm = LLM(
        model=str(view),
        trust_remote_code=True,
        dtype="bfloat16",
        distributed_executor_backend="uni",
        max_model_len=4096,
        max_num_seqs=32,
        max_num_batched_tokens=4096,
        kv_cache_memory_bytes=4 * 1024**3,
        enforce_eager=True,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 1},
        generation_config="vllm",
    )
    tokenizer = llm.get_tokenizer()
    requests = []
    for row in rows:
        with Image.open(row["image"]) as image:
            image = image.convert("RGB")
        requests.append(
            dict(
                prompt_token_ids=[tokenizer.pad_token_id]
                + tokenizer.encode(row["prompt"], add_special_tokens=False),
                multi_modal_data={"image": image},
            )
        )
    outputs = llm.generate(
        requests,
        SamplingParams(
            n=samples,
            temperature=1.0,
            top_p=1.0,
            seed=1231031,
            max_tokens=1536,
            skip_special_tokens=False,
            repetition_penalty=1.0,
        ),
    )
    results = []
    for row, generated in zip(rows, outputs, strict=True):
        answers = []
        for answer in generated.outputs:
            value = dict(
                text=answer.text,
                ended=bool(answer.token_ids and answer.token_ids[-1] == tokenizer.eos_token_id),
            )
            answers.append(
                dict(value, correct=successful_answer(value, row), **components(answer.text, row))
            )
        results.append(dict(id=row["id"], answers=answers))
    (output / "answers.json").write_text(json.dumps(results, indent=2))
    # Use a fresh generation kernel for another model if the installed vLLM
    # release retains its engine allocation after deletion.
    del llm
    gc.collect()
    return results


def pass_at_k(results, k):
    if not results or k < 1:
        raise ValueError("Need nonempty answers and k >= 1")
    values = []
    for row in results:
        n = len(row["answers"])
        if k > n:
            raise ValueError("k cannot exceed the number of sampled answers")
        c = sum(a["correct"] for a in row["answers"])
        values.append(1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k))
    return sum(values) / len(values)


def evaluate_isolated(model, rows, output, samples=8, processor_source=None):
    """Run generation in a worker that releases its GPU when finished."""
    import ray

    output = Path(output)
    if (output / "answers.json").is_file():
        saved_key = json.loads((output / "request.json").read_text())["key"]
        if saved_key != evaluation_key(model, rows, samples, processor_source):
            raise ValueError("Evaluation inputs changed; choose a new output directory")
        print(f"Reusing completed evaluation: {len(rows)} pages × {samples} answers")
        return json.loads((output / "answers.json").read_text())
    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray.init(num_cpus=4, num_gpus=1, include_dashboard=False)
    worker = ray.remote(num_gpus=1, num_cpus=4, max_calls=1)(evaluate).options(
        runtime_env={"py_executable": sys.executable, "env_vars": worker_env()}
    )
    try:
        return ray.get(worker.remote(model, rows, output, samples, processor_source))
    finally:
        if owns_ray:
            ray.shutdown()


def summarize(answers: list[dict]) -> dict:
    counts = [len(group["answers"]) for group in answers]
    if not counts or min(counts) == 0 or len(set(counts)) != 1:
        raise ValueError("Evaluation requires equal, nonzero sample counts per page")
    flattened = [answer for group in answers for answer in group["answers"]]
    keys = ("text_cer", "chart_exact", "patient_fields", "history_rows", "ended")
    return dict(pages=len(answers), answers=len(flattened), samples=counts[0],
                correct=sum(answer["correct"] for answer in flattened),
                pass_at_k={str(k): pass_at_k(answers, k) for k in (1, 2, 4, 8) if k <= counts[0]},
                metrics={key: sum(a[key] for a in flattened) / len(flattened) for key in keys})
