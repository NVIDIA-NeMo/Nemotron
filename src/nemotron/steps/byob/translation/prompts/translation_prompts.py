# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Default prompt templates for LLM-based translation.

Provides system and user prompts for instructing LLMs to perform translation tasks.
The prompts include guidelines for preserving non-textual information like numbers
and dates while translating the textual content.
"""

SYSTEM_PROMPT = '''You are a helpful translator that translates text from one language to another.
You will be given a text in the source language and you need to translate it into the target language.

1. Understand the meaning of the text in the source language.
2. Translate the text into the given target language.
3. Do not translate numbers, dates, times, or other non-textual information.
'''

PROMPT = '''Translate the following text from {{source_language}} to {{target_language}}:
Here is the text to translate:
"{{text}}"

Now translate the text into the given target language.
Provide your answer in JSON format.

- translation: The translated text
'''