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

# Adapted from https://github.com/google-research/deduplicate-text-datasets/blob/fcf7432891032537354310b50d48509055e3ee64/scripts/make_suffix_array.py

import argparse
import multiprocessing as mp
import os
import time

import numpy as np


def make_suffix_array(input_path: str, output_path: str, dedup_command: str, num_threads: int) -> None:
    data_size = os.path.getsize(input_path)
    os.makedirs(f"{output_path}", exist_ok=True)
    os.chdir(output_path)

    HACK = 100000

    started = []

    if data_size > 10e9:
        total_jobs = 100
        jobs_at_once = 20
    elif data_size > 1e9:
        total_jobs = 96
        jobs_at_once = 96
    elif data_size > 10e6:
        total_jobs = 4
        jobs_at_once = 4
    else:
        total_jobs = 1
        jobs_at_once = 1

    S = data_size // total_jobs

    for jobstart in range(0, total_jobs, jobs_at_once):
        wait = []
        for i in range(jobstart, jobstart + jobs_at_once):
            s, e = i * S, min((i + 1) * S + HACK, data_size)
            cmd = "%s make-part --data-file %s --start-byte %d --end-byte %d" % (
                dedup_command,
                input_path,
                s,
                e,
            )
            started.append((s, e))
            print(cmd)
            wait.append(os.popen(cmd))

            if e == data_size:
                break

        print("Waiting for jobs to finish")
        [x.read() for x in wait]

    print("Checking all wrote correctly")

    while True:
        files = ["%s.part.%d-%d" % (input_path, s, e) for s, e in started]

        wait = []
        for x, (s, e) in zip(files, started):
            go = False
            if not os.path.exists(x):
                print("GOGO")
                go = True
            else:
                size_data = os.path.getsize(x)
                FACT = np.ceil(np.log(size_data) / np.log(2) / 8)
                print("FACT", FACT, size_data * FACT, os.path.getsize(x + ".table.bin"))
                if (
                    not os.path.exists(x)
                    or not os.path.exists(x + ".table.bin")
                    or os.path.getsize(x + ".table.bin") == 0
                    or size_data * FACT != os.path.getsize(x + ".table.bin")
                ):
                    go = True
            if go:
                cmd = "%s make-part --data-file %s --start-byte %d --end-byte %d" % (
                    dedup_command,
                    input_path,
                    s,
                    e,
                )
                print(cmd)
                wait.append(os.popen(cmd))
                if len(wait) >= jobs_at_once:
                    break
        print("Rerunning", len(wait), "jobs because they failed.")
        [x.read() for x in wait]
        time.sleep(1)
        if len(wait) == 0:
            break

    print("Merging suffix trees")

    os.makedirs("tmp", exist_ok=True)
    os.popen("rm -f tmp/out.table.bin.*").read()

    torun = " --suffix-path ".join(files)
    print(
        "%s merge --output-file %s --suffix-path %s --num-threads %d"
        % (dedup_command, "tmp/out.table.bin", torun, num_threads)
    )
    pipe = os.popen(
        "%s merge --output-file %s --suffix-path %s --num-threads %d"
        % (dedup_command, "tmp/out.table.bin", torun, num_threads)
    )
    pipe.read()
    if pipe.close() is not None:
        print("Something went wrong with merging.")
        print("Please check that you ran with ulimit -Sn 100000")
        exit(1)
    # exit(0)
    print("Now merging individual tables")
    os.popen(f"cat tmp/out.table.bin.* > {input_path}.table.bin").read()
    print("Cleaning up")
    os.popen("rm -rf tmp").read()

    if os.path.exists(input_path + ".table.bin"):
        if (
            os.path.getsize(input_path + ".table.bin") % os.path.getsize(input_path)
            != 0
        ):
            print("File size is wrong")
            exit(1)
    else:
        print("Failed to create table")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, required=True, help="Path to input dataset.bin file"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to output dataset.bin.part.x-xx, dataset.bin.part.x-xx.table.bin, and dataset.bin.table.bin files",
    )
    parser.add_argument(
        "--dedup-command",
        type=str,
        default="deduplicate-text-datasets/target/release/dedup_dataset",
        help="Path to dedup_dataset binary from deduplicate-text-datasets repository",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=mp.cpu_count(),
        help="Number of threads to use",
    )
    args = parser.parse_args()

    make_suffix_array(
        args.input_path, args.output_path, args.dedup_command, args.num_threads
    )
