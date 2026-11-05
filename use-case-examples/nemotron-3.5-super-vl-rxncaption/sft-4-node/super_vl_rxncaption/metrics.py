"""
Standalone evaluation for BIVP reaction diagram parsing.
Computes Soft Match and Hybrid Match metrics.

Based on the evaluation code from RxnCaption (arxiv 2511.02384).

Usage:
    python metrics.py \
        --ground_truth ground_truth/ground_truth.json \
        --predictions  my_predictions_eval_format.json

Input format (predictions): same COCO-like JSON as ground truth:
    {
      "images": [
        {
          "file_name": "abc123.png",
          "bboxes": [
            {"id": 0, "category_id": 1, "bbox": [x, y, w, h]},
            {"id": 1, "category_id": 2, "bbox": [0,0,0,0], "text": "NaOH"}
          ],
          "reactions": [
            {"reactants": [0], "conditions": [1], "products": [2]}
          ]
        }
      ]
    }

Use convert_predictions.py to convert raw model output into this format.
"""

import argparse
import json

import numpy as np


# --- Edit distance ---
def _edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# --- IoU (x, y, w, h format) ---
def _iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def get_bboxes_match(gt_boxes, pred_boxes, iou_threshold=0.5):
    n, m = len(gt_boxes), len(pred_boxes)
    if n == 0 or m == 0:
        return [-1] * n, m, np.zeros((n, m))
    iou_mat = np.array([[_iou(g, p) for p in pred_boxes] for g in gt_boxes])
    match = [-1] * n
    used_pred = set()
    for gi in range(n):
        best_iou, best_pi = 0.0, -1
        for pi in range(m):
            if pi in used_pred:
                continue
            if iou_mat[gi, pi] > best_iou:
                best_iou, best_pi = iou_mat[gi, pi], pi
        if best_iou >= iou_threshold:
            match[gi] = best_pi
            used_pred.add(best_pi)
    return match, m - len(used_pred), iou_mat


# --- Data model ---
class BBox:
    def __init__(self, record):
        self.id = record["id"]
        self.category_id = record.get("category_id", 1)
        self.bbox = record.get("bbox", [0, 0, 0, 0])
        self.text = record.get("text", "")

    @property
    def is_molecule(self):
        return self.category_id == 1

    @property
    def is_text(self):
        return self.category_id in (2, 3)


class Reaction:
    def __init__(self, rxn_dict, bbox_lookup):
        self.reactants = [bbox_lookup[i] for i in rxn_dict.get("reactants", []) if i in bbox_lookup]
        self.conditions = [bbox_lookup[i] for i in rxn_dict.get("conditions", []) if i in bbox_lookup]
        self.products = [bbox_lookup[i] for i in rxn_dict.get("products", []) if i in bbox_lookup]

    def is_empty(self, mol_only=False):
        members = self.reactants + self.conditions + self.products
        if mol_only:
            members = [b for b in members if b.is_molecule]
        return len(members) == 0


# --- Matching ---
def _soft_match(gr, pr):
    """RxnCaption SoftMatch: ignore text and merge conditions into reactants."""

    def split(rxn):
        bboxes, groups = [], []
        for b in rxn.reactants + rxn.conditions:
            if b.is_molecule:
                bboxes.append(b.bbox)
                groups.append("rc")
        for b in rxn.products:
            if b.is_molecule:
                bboxes.append(b.bbox)
                groups.append("p")
        return bboxes, groups

    gb, gg = split(gr)
    pb, pg = split(pr)
    if len(gb) != len(pb):
        return False
    if not gb:
        return True
    m, _, _ = get_bboxes_match(gb, pb)
    if any(v == -1 for v in m):
        return False
    return all(gg[gi] == pg[m[gi]] for gi in range(len(gb)))


def _hybrid_match(gr, pr, ratio=0.2):
    """RxnCaption HybridMatch for BIVP.

    - Molecules: 3-way role-aware IoU matching (RxnScribe Hard-Match style).
    - Reactants text: multiset must match exactly (string equality).
    - Products text: multiset must match exactly (string equality).
    - Conditions text: count must match; each within normalized edit distance <= ratio.
    """
    for role in ("reactants", "conditions", "products"):
        gb = [b.bbox for b in getattr(gr, role) if b.is_molecule]
        pb = [b.bbox for b in getattr(pr, role) if b.is_molecule]
        if len(gb) != len(pb):
            return False
        if gb:
            m, _, _ = get_bboxes_match(gb, pb)
            if any(v == -1 for v in m):
                return False

    for role in ("reactants", "products"):
        gt_t = sorted(b.text for b in getattr(gr, role) if b.is_text)
        pr_t = sorted(b.text for b in getattr(pr, role) if b.is_text)
        if gt_t != pr_t:
            return False

    gt_t = sorted(b.text for b in gr.conditions if b.is_text)
    pr_t = sorted(b.text for b in pr.conditions if b.is_text)
    if len(gt_t) != len(pr_t):
        return False
    used = set()
    for g in gt_t:
        ok = False
        for j, p in enumerate(pr_t):
            if j in used:
                continue
            d = _edit_distance(g, p)
            ml = max(len(g), len(p), 1)
            if d / ml <= ratio:
                used.add(j)
                ok = True
                break
        if not ok:
            return False
    return True


def evaluate_image(gold_img, pred_img, match_fn, mol_only=False):
    """Evaluate one image, return TP/FP/FN."""
    gl = {b["id"]: BBox(b) for b in gold_img.get("bboxes", [])}
    pl = {b["id"]: BBox(b) for b in pred_img.get("bboxes", [])}

    gt_rxns = [Reaction(r, gl) for r in gold_img.get("reactions", [])]
    pr_rxns = [Reaction(r, pl) for r in pred_img.get("reactions", [])]

    gt = [r for r in gt_rxns if not r.is_empty(mol_only)]
    pr = [r for r in pr_rxns if not r.is_empty(mol_only)]

    if not gt and not pr:
        return 0, 0, 0, True

    matched_pr = set()
    tp = 0
    for gi, gr in enumerate(gt):
        for pi, prr in enumerate(pr):
            if pi in matched_pr:
                continue
            if match_fn(gr, prr):
                matched_pr.add(pi)
                tp += 1
                break

    return tp, len(pr) - tp, len(gt) - tp, False


def run_evaluation(gt_path, pred_path):
    with open(gt_path, encoding="utf-8") as f:
        gt_data = json.load(f)
    with open(pred_path, encoding="utf-8") as f:
        pred_data = json.load(f)

    gt_by_fname = {i["file_name"]: i for i in gt_data["images"]}
    pred_by_fname = {i["file_name"]: i for i in pred_data["images"]}
    missing = sorted(set(gt_by_fname) - set(pred_by_fname))
    unexpected = sorted(set(pred_by_fname) - set(gt_by_fname))
    if missing or unexpected:
        raise ValueError(
            "Prediction filenames must exactly match ground truth; "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    common = sorted(gt_by_fname)

    print(f"GT images: {len(gt_data['images'])}")
    print(f"Pred images: {len(pred_data['images'])}")
    print(f"Common: {len(common)}")

    metrics = {
        "Soft": {"fn": _soft_match, "mol_only": True},
        "Hybrid": {"fn": _hybrid_match, "mol_only": False},
    }

    for name, cfg in metrics.items():
        total_tp = total_fp = total_fn = total_gt = total_pd = 0
        for fname in common:
            tp, fp, fn, ignored = evaluate_image(
                gt_by_fname[fname],
                pred_by_fname[fname],
                cfg["fn"],
                cfg["mol_only"],
            )
            if ignored:
                continue
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_gt += tp + fn
            total_pd += tp + fp

        p = total_tp / total_pd * 100 if total_pd > 0 else 0
        r = total_tp / total_gt * 100 if total_gt > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        print(f"\n{name + 'Match':<13} TP={total_tp:5d}  FP={total_fp:5d}  FN={total_fn:5d}")
        print(f"{'':13} P={p:6.2f}%  R={r:6.2f}%  F1={f1:6.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BIVP predictions (Soft + Hybrid Match)")
    parser.add_argument("--ground_truth", required=True, help="Ground truth JSON")
    parser.add_argument("--predictions", required=True, help="Predictions in eval format JSON")
    args = parser.parse_args()
    run_evaluation(args.ground_truth, args.predictions)


if __name__ == "__main__":
    main()
