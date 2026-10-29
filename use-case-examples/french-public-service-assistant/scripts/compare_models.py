"""Step 7: send the same test prompts to the base model and the adapted model and compare the answers.

Each model is a separate pass, so you can serve one model at a time (the DGX Spark case) or both at once.
A pass runs when you give that side's --*-url and --*-model; it saves eval/answers_base.json or
eval/answers_tuned.json after every prompt. Once both answer files exist, the side-by-side files are written.
Answers are generated at temperature 0 with thinking disabled so runs are repeatable.

One model at a time (serve the base model, then run; then serve the adapted model, then run):

    uv run python use-case-examples/french-public-service-assistant/scripts/compare_models.py \\
        --base-url http://localhost:8000/v1 --base-model lightning
    uv run python use-case-examples/french-public-service-assistant/scripts/compare_models.py \\
        --tuned-url http://localhost:8000/v1 --tuned-model lightning-fr

Both served at once: pass all four options in a single run.

Output: eval/answers_{base,tuned}.json, then eval/comparison.md (side by side) and eval/comparison.json.
Read the answers yourself; the rubric in the blog (task correctness, French quality, locale fit,
safety, robustness, retention) is a guide for that review, not an automatic score.
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent.parent


def ask(url: str, model: str, prompt: str, api_key_env: str | None, max_tokens: int) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    if api_key_env:
        headers["Authorization"] = "Bearer " + os.environ[api_key_env]
    request = urllib.request.Request(url.rstrip("/") + "/chat/completions", body, headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)["choices"][0]["message"]["content"] or ""


def resolve_model(url: str, model: str, key_env: str | None) -> str:
    """Return the name to request. If `model` is not served but the endpoint serves exactly one model
    (the usual one-model-per-server setup), use that model and say so; otherwise stop with the served names."""
    headers = {"Authorization": "Bearer " + os.environ[key_env]} if key_env else {}
    request = urllib.request.Request(url.rstrip("/") + "/models", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            served = [m["id"] for m in json.load(response)["data"]]
    except Exception:  # endpoints without a /models route: let the first request report any problem
        return model
    if model in served:
        return model
    if len(served) == 1:
        print(f"note: {url} serves '{served[0]}', not '{model}'; using '{served[0]}'")
        return served[0]
    raise SystemExit(f"model '{model}' is not served at {url}; served models: {', '.join(served) or 'none'}")


def generate(label: str, url: str, model: str, key_env: str | None, args, prompts: list[dict]) -> None:
    """Answer every prompt with one model and save the answers, so a run can finish without the other model."""
    model = resolve_model(url, model, key_env)
    path = args.out_dir / f"answers_{label}.json"
    answers: dict[str, str] = {}
    for item in prompts:
        answers[item["id"]] = ask(url, model, item["prompt"], key_env, args.max_tokens)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": model, "answers": answers}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{label}: done {item['id']}")


def write_comparison(args, prompts: list[dict]) -> None:
    base = json.loads((args.out_dir / "answers_base.json").read_text(encoding="utf-8"))
    tuned = json.loads((args.out_dir / "answers_tuned.json").read_text(encoding="utf-8"))
    results = [
        {**item, "base_answer": base["answers"][item["id"]], "tuned_answer": tuned["answers"][item["id"]]}
        for item in prompts
    ]
    (args.out_dir / "comparison.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# Base vs adapted model\n", f"Base: `{base['model']}`; adapted: `{tuned['model']}`\n"]
    for r in results:
        lines += [
            f"## {r['id']} ({r['category']})\n",
            f"**Prompt:** {r['prompt']}\n",
            f"*{r['prompt_en']}*\n",
            f"**Expected behavior:** {r['expected_behavior']}\n",
            f"**Base:**\n\n{r['base_answer']}\n",
            f"**Adapted:**\n\n{r['tuned_answer']}\n",
        ]
    (args.out_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", args.out_dir / "comparison.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompts", type=Path, default=EXAMPLE_DIR / "eval" / "test_prompts.jsonl")
    parser.add_argument("--base-url", help="endpoint serving the base model (omit to skip the base pass)")
    parser.add_argument("--base-model", help="served name of the base model")
    parser.add_argument("--tuned-url", help="endpoint serving the adapted model (omit to skip the adapted pass)")
    parser.add_argument("--tuned-model", help="served name of the adapted model")
    parser.add_argument("--base-key-env", help="env var holding an API key for the base endpoint (optional)")
    parser.add_argument("--tuned-key-env", help="env var holding an API key for the tuned endpoint (optional)")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--out-dir", type=Path, default=EXAMPLE_DIR / "eval")
    args = parser.parse_args()

    for side in ("base", "tuned"):
        if bool(getattr(args, f"{side}_url")) != bool(getattr(args, f"{side}_model")):
            parser.error(f"--{side}-url and --{side}-model go together")
    if not (args.base_url or args.tuned_url):
        parser.error("give at least one of --base-url/--base-model or --tuned-url/--tuned-model")

    prompts = [json.loads(line) for line in open(args.prompts, encoding="utf-8")]
    if args.base_url:
        generate("base", args.base_url, args.base_model, args.base_key_env, args, prompts)
    if args.tuned_url:
        generate("tuned", args.tuned_url, args.tuned_model, args.tuned_key_env, args, prompts)

    missing = [n for n in ("base", "tuned") if not (args.out_dir / f"answers_{n}.json").exists()]
    if missing:
        print(f"saved answers; run the {' and '.join(missing)} pass to produce the comparison")
        return
    write_comparison(args, prompts)


if __name__ == "__main__":
    main()
