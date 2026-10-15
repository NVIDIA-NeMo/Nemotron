# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Answer a few held-out finance questions with vLLM. Run by `steps/try.sh`.

    python try_model.py <merged checkpoint or base model dir> [rows, default 8]

Takes the first rows of test.jsonl, a mix of image and text questions. Prints what the model says after its `<think>`
reasoning next to the reference answer. vLLM spawns workers, hence the `__main__` guard.
"""

import json
import os
import sys

import torch
from vllm import LLM, SamplingParams


def to_vllm(part: dict) -> dict:
    if part["type"] == "image":  # vLLM takes local images as file:// URLs
        return {"type": "image_url", "image_url": {"url": "file://" + part["image"]}}
    return part


def main() -> None:
    model, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8
    data_dir = os.path.join(os.environ["WORKSPACE"], "dataprep")
    with open(os.path.join(data_dir, "test.jsonl")) as f:
        picked = [json.loads(line) for line, _ in zip(f, range(n))]  # test.jsonl is already shuffled

    llm = LLM(model=model, trust_remote_code=True, tensor_parallel_size=torch.cuda.device_count(),
              dtype="bfloat16", enforce_eager=True, max_model_len=8192,
              limit_mm_per_prompt={"image": 1}, allowed_local_media_path=os.environ["WORKSPACE"])
    outputs = llm.chat(
        [[{"role": "user", "content": [to_vllm(p) for p in r["messages"][0]["content"]]}] for r in picked],
        SamplingParams(temperature=0, max_tokens=2048),
        chat_template_kwargs={"enable_thinking": True},
    )
    for row, out in zip(picked, outputs):
        answer = out.outputs[0].text.rsplit("</think>", 1)[-1].strip()  # the text after the reasoning
        if len(answer) > 300:
            answer = answer[:300] + " [...]"
        if out.outputs[0].finish_reason == "length":
            answer += " [stopped at max_tokens]"
        print(f"\n[{row['task']}] {row['id']}\nreference: {row['answer']}\nmodel:     {answer}")


if __name__ == "__main__":
    main()
