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

"""Local Ray worker integration using normal Nemotron package imports."""

import ray


def test_packing_in_ray_worker():
    from nemotron.data_prep.packing import FirstFitShufflePacker

    lengths = [33] * 100 + [0, 1, 65, 2] * 40
    expected = FirstFitShufflePacker(64, seed=42).pack(lengths)

    @ray.remote
    def pack(values):
        from nemotron.data_prep.packing import FirstFitShufflePacker

        return FirstFitShufflePacker(64, seed=42).pack(values)

    ray.init(num_cpus=2, include_dashboard=False)
    try:
        assert ray.get(pack.remote(lengths)) == expected
    finally:
        ray.shutdown()
