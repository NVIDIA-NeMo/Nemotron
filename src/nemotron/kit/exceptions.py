# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Exception classes for the nemotron.art package.
"""


class ArtifactNotFoundError(Exception):
    """Raised when an artifact cannot be found in the registry."""

    def __init__(self, name: str, message: str | None = None) -> None:
        self.name = name
        self.message = message or f"Artifact not found: {name}"
        super().__init__(self.message)


class ArtifactVersionNotFoundError(Exception):
    """Raised when a specific version of an artifact cannot be found."""

    def __init__(self, name: str, version: str | int, message: str | None = None) -> None:
        self.name = name
        self.version = version
        self.message = message or f"Artifact version not found: {name}:{version}"
        super().__init__(self.message)
