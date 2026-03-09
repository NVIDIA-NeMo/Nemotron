# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
import json

from datasets import load_dataset


def json2token(obj, sort_json_key=True):
    if isinstance(obj, dict):
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        out = ""
        keys = sorted(obj.keys()) if sort_json_key else obj.keys()
        for k in keys:
            out += f"<s_{k}>" + json2token(obj[k], sort_json_key) + f"</s_{k}>"
        return out
    if isinstance(obj, list):
        return "<sep/>".join(json2token(i, sort_json_key) for i in obj)
    return str(obj)


def make_invoice_dataset(path_or_dataset="katanaml-org/invoices-donut-data-v1", split="train", **kwargs):
    ds = load_dataset(path_or_dataset, split=split)
    samples = []
    for ex in ds:
        gt = json.loads(ex["ground_truth"])["gt_parse"]
        target = json2token(gt)
        samples.append(
            {
                "conversation": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": ex["image"]},
                            {"type": "text", "text": "Parse this invoice."},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": target}]},
                ],
            }
        )
    print(f"{split.capitalize()} samples: {len(samples)}")
    if samples:
        print(f"Target preview: {samples[0]['conversation'][1]['content'][0]['text'][:200]}...")
    return samples
