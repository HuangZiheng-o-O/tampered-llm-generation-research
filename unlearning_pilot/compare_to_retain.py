"""Compare a pilot unlearning adapter with a retain-only reference on identical TOFU questions.

The two-sample Kolmogorov-Smirnov p-value follows TOFU's forget-quality
comparison. It is labelled as a pilot estimate because our prompt template,
sample count, and LoRA training differ from the published full-weight setup.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _rows(record: dict, group: str) -> list[dict]:
    return record["results"][group]["ranking_details"]


def _check_pair(left: dict, right: dict) -> None:
    if (left["base"] != right["base"] or left["seed"] != right["seed"] or
            left["max_length"] != right["max_length"]):
        raise ValueError("Model, evaluation seed, or maximum length differ")
    for group in ("forget", "retain"):
        a, b = left["results"][group], right["results"][group]
        if (a["question_sha256"] != b["question_sha256"] or
                [row["question"] for row in _rows(left, group)] !=
                [row["question"] for row in _rows(right, group)]):
            raise ValueError(f"Evaluation questions differ in {group}")


def compare(model: dict, retain_reference: dict, parent: dict) -> dict:
    from scipy.stats import ks_2samp

    _check_pair(model, retain_reference)
    _check_pair(model, parent)
    # log(wrong probability / correct probability). KS is invariant under log,
    # so this is numerically stable even after strong gradient ascent.
    def log_truth_ratios(record: dict) -> list[float]:
        values = [row["correct_answer_nll"] - row["mean_wrong_answer_nll"]
                  for row in _rows(record, "forget")]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("Missing or nonfinite forget truth ratios")
        return values

    result = ks_2samp(log_truth_ratios(model),
                      log_truth_ratios(retain_reference),
                      alternative="two-sided", method="auto")
    return {
        "sample_count_per_model": len(_rows(model, "forget")),
        "pilot_forget_quality_ks_pvalue": float(result.pvalue),
        "pilot_forget_quality_ks_statistic": float(result.statistic),
        "forget_ranked_first": model["results"]["forget"]["correct_answer_ranked_first"],
        "retain_ranked_first": model["results"]["retain"]["correct_answer_ranked_first"],
        "parent_retain_ranked_first": parent["results"]["retain"]["correct_answer_ranked_first"],
        "retain_reference_forget_ranked_first": (
            retain_reference["results"]["forget"]["correct_answer_ranked_first"]),
        "retain_reference_retain_ranked_first": (
            retain_reference["results"]["retain"]["correct_answer_ranked_first"]),
        "model_adapter_sha256": model["adapter_sha256"],
        "retain_reference_adapter_sha256": retain_reference["adapter_sha256"],
        "parent_adapter_sha256": parent["adapter_sha256"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-eval", type=Path, required=True)
    p.add_argument("--retain-eval", type=Path, required=True)
    p.add_argument("--parent-eval", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error(f"Output already exists: {args.output}")
    records = [json.loads(path.read_text()) for path in
               (args.model_eval, args.retain_eval, args.parent_eval)]
    result = compare(*records)
    result["inputs"] = {
        "model": str(args.model_eval.resolve()),
        "retain_reference": str(args.retain_eval.resolve()),
        "parent": str(args.parent_eval.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
