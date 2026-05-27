<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

(sdg-planning)=
# Planning a Synthetic Data Generation Run

Synthetic data generation involves a sequence of decisions that determine whether the resulting data is useful for training.

## What Does Good Source Data Look Like for Your Domain?

The seed file and prompts are where your domain knowledge enters the pipeline.
The scenarios, constraints, and vocabulary you bring to the seed scenarios shape everything the generation produces.
A pipeline run without domain knowledge produces generic output from the model's own training data regardless of how well the column specs are written.

The quality of your seed material is the strongest lever you have on the quality of what the pipeline produces.

## What Makes a Generated Record Good Enough to Train on?

Before scaling, be able to evaluate the quality of a generated record for your domain.

If you were building a hand-curated training dataset, would you include this record?
If yes, it belongs.

If you would hesitate because the response sounds evasive, the scenario is implausible, or the assistant fabricated a detail, that is a signal to revise the prompt or seed before generating at scale.

Preview records before you commit to generating thousands of records.

## Next Steps

- First run: {doc}`getting-started`
- Adapt the pipeline to your domain: {doc}`./how-to/create-domain-dataset`
- Preview and iterate on a config: {doc}`./how-to/run`
- Config field reference: {doc}`./reference/config-schema`
