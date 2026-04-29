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


import bcp47

REGISTRY = {
    'hinglish': 'Hinglish (Hindi + English written in Latin script)',
}

def get_language(language_code: str):
    """Get human-readable language name from language code.
    
    Supports both custom language codes (from REGISTRY) and standard BCP 47 codes.
    
    Args:
        language_code: Language code (e.g., 'en', 'hi', 'hinglish').
        
    Returns:
        str: Human-readable language name.
        
    Raises:
        ValueError: If language code is not recognized in REGISTRY or BCP 47.
        
    Examples:
        >>> get_language('en')
        'English'
        >>> get_language('hinglish')
        'Hinglish (Hindi + English written in Latin script)'
    """
    if language_code in REGISTRY:
        return REGISTRY[language_code]
    try:
        tag = bcp47.tags[language_code]
        return str(tag)
    except KeyError:
        raise ValueError(f"Unsupported language code: {language_code}")