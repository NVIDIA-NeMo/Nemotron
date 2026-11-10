"""Small vendored GriTS core for deterministic RLVR table rewards.

Adapted from Microsoft's Table Transformer reference implementation and the
ParseBench vendored implementation.  This module intentionally contains only
the 2D-MSS alignment and Top/Con entry points needed by the reward scorer.

Reference: Smock, Pesala, Abraham, "GriTS: Grid Table Similarity Metric for
Table Structure Recognition", ICDAR 2023, https://arxiv.org/abs/2203.12555.
"""

from __future__ import annotations

import itertools
from difflib import SequenceMatcher
from typing import Any, Callable

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


RewardFunction = Callable[[Any, Any], float]


if njit is not None:
    @njit(cache=True)
    def _align_outer_tensor(rewards: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Compiled exact reference DP over a dense 4-D reward tensor."""
        true_outer, true_inner, pred_outer, pred_inner = rewards.shape
        scores = np.zeros((true_outer + 1, pred_outer + 1), dtype=np.float64)
        pointers = np.zeros((true_outer + 1, pred_outer + 1), dtype=np.int8)
        for i in range(1, true_outer + 1):
            pointers[i, 0] = -1
        for j in range(1, pred_outer + 1):
            pointers[0, j] = 1
        for i in range(1, true_outer + 1):
            for j in range(1, pred_outer + 1):
                previous = np.zeros(pred_inner + 1, dtype=np.float64)
                for true_index in range(1, true_inner + 1):
                    current = np.zeros(pred_inner + 1, dtype=np.float64)
                    for pred_index in range(1, pred_inner + 1):
                        diagonal = previous[pred_index - 1] + rewards[
                            i - 1, true_index - 1, j - 1, pred_index - 1
                        ]
                        up = previous[pred_index]
                        left = current[pred_index - 1]
                        current[pred_index] = max(diagonal, up, left)
                    previous = current
                inner_reward = previous[pred_inner]
                diagonal = scores[i - 1, j - 1] + inner_reward
                up = scores[i - 1, j]
                left = scores[i, j - 1]
                best = max(diagonal, up, left)
                scores[i, j] = best
                pointers[i, j] = 0 if diagonal == best else -1 if up == best else 1
        row, col = true_outer, pred_outer
        true_alignment = np.empty(min(true_outer, pred_outer), dtype=np.int64)
        pred_alignment = np.empty(min(true_outer, pred_outer), dtype=np.int64)
        count = 0
        while row or col:
            pointer = pointers[row, col]
            if pointer == -1:
                row -= 1
            elif pointer == 1:
                col -= 1
            else:
                row -= 1
                col -= 1
                true_alignment[count] = row
                pred_alignment[count] = col
                count += 1
        return (
            true_alignment[:count][::-1], pred_alignment[:count][::-1],
            float(scores[true_outer, pred_outer]),
        )
else:
    _align_outer_tensor = None


def lcs_similarity(left: Any, right: Any) -> float:
    """Return the reference GriTS normalized matching-subsequence score."""
    left = str(left) if not isinstance(left, str) else left
    right = str(right) if not isinstance(right, str) else right
    if not left and not right:
        return 1.0
    matched = sum(
        block.size
        for block in SequenceMatcher(None, left, right, autojunk=False).get_matching_blocks()
    )
    return 2.0 * matched / max(1, len(left) + len(right))


def span_iou(left: Any, right: Any) -> float:
    """IoU of relative-span rectangles, matching the GriTS Top definition."""
    if not hasattr(left, "__len__") and not hasattr(right, "__len__"):
        return 1.0
    if not hasattr(left, "__len__") or not hasattr(right, "__len__"):
        return 0.0
    if len(left) != 4 or len(right) != 4:
        return 0.0
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ux1, uy1 = min(left[0], right[0]), min(left[1], right[1])
    ux2, uy2 = max(left[2], right[2]), max(left[3], right[3])
    enclosing_area = max(0.0, ux2 - ux1) * max(0.0, uy2 - uy1)
    return intersection / enclosing_area if enclosing_area else 0.0


def _fscore(true_positive: float, true_count: int, pred_count: int) -> tuple[float, float, float]:
    precision = true_positive / pred_count if pred_count else 1.0
    recall = true_positive / true_count if true_count else 1.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return score, precision, recall


def _initialize_dp(left: int, right: int) -> tuple[np.ndarray, np.ndarray]:
    scores = np.zeros((left + 1, right + 1), dtype=np.float64)
    pointers = np.zeros((left + 1, right + 1), dtype=np.int8)
    pointers[1:, 0] = -1
    pointers[0, 1:] = 1
    return scores, pointers


def _traceback(pointers: np.ndarray) -> tuple[list[int], list[int]]:
    row, col = pointers.shape[0] - 1, pointers.shape[1] - 1
    left: list[int] = []
    right: list[int] = []
    while row or col:
        pointer = int(pointers[row, col])
        if pointer == -1:
            row -= 1
        elif pointer == 1:
            col -= 1
        else:
            row -= 1
            col -= 1
            left.append(row)
            right.append(col)
    return left[::-1], right[::-1]


def _align_1d(
    left: list[tuple[int, int]],
    right: list[tuple[int, int]],
    rewards: dict[tuple[int, int, int, int], float],
) -> float:
    scores, pointers = _initialize_dp(len(left), len(right))
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            choices = (
                scores[i - 1, j - 1] + rewards[left[i - 1] + right[j - 1]],
                scores[i - 1, j],
                scores[i, j - 1],
            )
            best = max(choices)
            scores[i, j] = best
            pointers[i, j] = 0 if choices[0] == best else -1 if choices[1] == best else 1
    return float(scores[-1, -1])


def _align_outer(
    true_shape: tuple[int, int],
    pred_shape: tuple[int, int],
    rewards: dict[tuple[int, int, int, int], float],
) -> tuple[list[int], list[int], float]:
    scores, pointers = _initialize_dp(true_shape[0], pred_shape[0])
    for i in range(1, true_shape[0] + 1):
        for j in range(1, pred_shape[0] + 1):
            reward = _align_1d(
                [(i - 1, col) for col in range(true_shape[1])],
                [(j - 1, col) for col in range(pred_shape[1])],
                rewards,
            )
            choices = (scores[i - 1, j - 1] + reward, scores[i - 1, j], scores[i, j - 1])
            best = max(choices)
            scores[i, j] = best
            pointers[i, j] = 0 if choices[0] == best else -1 if choices[1] == best else 1
    left, right = _traceback(pointers)
    return left, right, float(scores[-1, -1])


def factored_2dmss_with_alignment(
    true_grid: np.ndarray,
    pred_grid: np.ndarray,
    reward_function: RewardFunction,
) -> tuple[float, float, float, float, dict[int, int], dict[int, int]]:
    """Return GriTS F1/precision/recall/upper-bound and row/column maps."""
    rewards: dict[tuple[int, int, int, int], float] = {}
    reward_tensor = np.empty(
        (true_grid.shape[0], true_grid.shape[1], pred_grid.shape[0], pred_grid.shape[1]),
        dtype=np.float64,
    )
    for tr, tc, pr, pc in itertools.product(
        range(true_grid.shape[0]), range(true_grid.shape[1]),
        range(pred_grid.shape[0]), range(pred_grid.shape[1]),
    ):
        reward = reward_function(true_grid[tr, tc], pred_grid[pr, pc])
        rewards[(tr, tc, pr, pc)] = reward
        reward_tensor[tr, tc, pr, pc] = reward

    if _align_outer_tensor is not None:
        true_rows_array, pred_rows_array, row_score = _align_outer_tensor(reward_tensor)
        true_cols_array, pred_cols_array, col_score = _align_outer_tensor(
            np.ascontiguousarray(reward_tensor.transpose(1, 0, 3, 2))
        )
        true_rows, pred_rows = true_rows_array.tolist(), pred_rows_array.tolist()
        true_cols, pred_cols = true_cols_array.tolist(), pred_cols_array.tolist()
    else:
        transposed = {
            (tc, tr, pc, pr): value
            for (tr, tc, pr, pc), value in rewards.items()
        }
        true_rows, pred_rows, row_score = _align_outer(
            true_grid.shape[:2], pred_grid.shape[:2], rewards
        )
        true_cols, pred_cols, col_score = _align_outer(
            true_grid.shape[:2][::-1], pred_grid.shape[:2][::-1], transposed
        )
    row_map = dict(zip(true_rows, pred_rows, strict=True))
    col_map = dict(zip(true_cols, pred_cols, strict=True))
    positive = sum(
        rewards[(tr, tc, pr, pc)]
        for tr, pr in zip(true_rows, pred_rows, strict=False)
        for tc, pc in zip(true_cols, pred_cols, strict=False)
    )
    true_count = int(true_grid.shape[0] * true_grid.shape[1])
    pred_count = int(pred_grid.shape[0] * pred_grid.shape[1])
    score, precision, recall = _fscore(positive, true_count, pred_count)
    upper, _, _ = _fscore(min(row_score, col_score), true_count, pred_count)
    return score, precision, recall, upper, row_map, col_map


def grits_con(
    true_grid: np.ndarray, pred_grid: np.ndarray,
) -> tuple[float, float, float, float, dict[int, int], dict[int, int]]:
    return factored_2dmss_with_alignment(true_grid, pred_grid, lcs_similarity)


def grits_top(
    true_grid: np.ndarray, pred_grid: np.ndarray,
) -> tuple[float, float, float, float, dict[int, int], dict[int, int]]:
    return factored_2dmss_with_alignment(true_grid, pred_grid, span_iou)
