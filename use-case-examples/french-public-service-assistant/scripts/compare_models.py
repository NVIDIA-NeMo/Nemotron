"""Step 7: send the same test prompts to the base model and the adapted model and save the answers.

Both models must be served behind OpenAI-compatible endpoints (for example two vLLM servers, or one
server per run). Answers are generated at temperature 0 with thinking disabled so runs are repeatable.

    uv run python use-case-examples/french-public-service-assistant/scripts/compare_models.py \\
        --base-url http://localhost:8000/v1 --base-model lightning \\
        --tuned-url http://localhost:8001/v1 --tuned-model lightning-fr-public-service

Output: eval/comparison.md (side by side) and eval/comparison.json.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompts", type=Path, default=EXAMPLE_DIR / "eval" / "test_prompts.jsonl")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tuned-url", required=True)
    parser.add_argument("--tuned-model", required=True)
    parser.add_argument("--base-key-env", help="env var holding an API key for the base endpoint (optional)")
    parser.add_argument("--tuned-key-env", help="env var holding an API key for the tuned endpoint (optional)")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--out-dir", type=Path, default=EXAMPLE_DIR / "eval")
    args = parser.parse_args()

    prompts = [json.loads(line) for line in open(args.prompts, encoding="utf-8")]
    results = []
    for item in prompts:
        results.append(
            {
                **item,
                "base_answer": ask(args.base_url, args.base_model, item["prompt"], args.base_key_env, args.max_tokens),
                "tuned_answer": ask(
                    args.tuned_url, args.tuned_model, item["prompt"], args.tuned_key_env, args.max_tokens
                ),
            }
        )
        print("done:", item["id"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# Base vs adapted model\n"]
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


if __name__ == "__main__":
    main()
