"""Measure held-out-wording TOFU answer likelihood for a pilot adapter.

This is a generation-quality gate. Detector accuracy is measured later on
separate model checkpoints, after the generated models pass this gate.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
from pathlib import Path

from unlearning_pilot.preflight import read_jsonl, sha256


def select_rows(path: Path, count: int, seed: int) -> list[dict]:
    rows = read_jsonl(path)
    if any("paraphrased_question" not in r or "paraphrased_answer" not in r
           for r in rows):
        raise ValueError(f"No paraphrased evaluation questions in {path}")
    return random.Random(seed).sample(rows, min(count, len(rows)))


def answer_nll(model, tokenizer, row: dict, device: str, length: int) -> float:
    import torch
    from unlearning_pilot.train import encode, move

    renamed = {"question": row["paraphrased_question"],
               "answer": row["paraphrased_answer"]}
    batch = move(encode(renamed, tokenizer, length), device)
    with torch.inference_mode():
        output = model(**batch)
    value = float(output.loss)
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite evaluation NLL")
    return value


def candidate_answers(row: dict) -> list[str]:
    raw = row["perturbed_answer"]
    values = ast.literal_eval(raw) if isinstance(raw, str) else raw
    if not isinstance(values, list) or not values or any(
            not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Invalid TOFU perturbed-answer candidates")
    return values


def rouge_l_recall(prediction: str, reference: str) -> float:
    """Word-level longest-common-subsequence recall for a generated answer."""
    actual = re.findall(r"\w+", prediction.casefold())
    expected = re.findall(r"\w+", reference.casefold())
    if not expected:
        raise ValueError("Empty reference answer")
    previous = [0] * (len(expected) + 1)
    for token in actual:
        current = [0]
        for index, wanted in enumerate(expected, 1):
            current.append(previous[index - 1] + 1 if token == wanted else
                           max(previous[index], current[-1]))
        previous = current
    return previous[-1] / len(expected)


def generate_answer(model, tokenizer, question: str, device: str,
                    max_new_tokens: int) -> str:
    import torch

    prompt = f"Question: {question}\nAnswer:"
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.inference_mode():
        output = model.generate(ids, do_sample=False, max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sample-per-group", type=int, default=50)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--rank-candidates", action="store_true")
    p.add_argument("--generate", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=96)
    args = p.parse_args()
    if args.output.exists():
        p.error(f"Output already exists: {args.output}")
    if args.sample_per_group < 1 or args.max_new_tokens < 1:
        p.error("sample-per-group and max-new-tokens must be positive")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.base))
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base), dtype=torch.bfloat16, device_map={"": args.device},
        low_cpu_mem_usage=True, attn_implementation="sdpa")
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model.eval()
    results = {}
    for group, name in (("forget", "forget10_perturbed.json"),
                        ("retain", "retain_perturbed.json")):
        selected = select_rows(args.data / name, args.sample_per_group, args.seed)
        scores = [answer_nll(model, tokenizer, row, args.device, args.max_length)
                  for row in selected]
        results[group] = {"sample_count": len(scores),
                          "mean_answer_nll": sum(scores) / len(scores),
                          "question_sha256": sha256(args.data / name)}
        if args.rank_candidates:
            correct = 0
            truth_shares = []
            ranking_details = []
            for row, true_nll in zip(selected, scores):
                false_nll = [answer_nll(model, tokenizer, {
                    "paraphrased_question": row["paraphrased_question"],
                    "paraphrased_answer": candidate}, args.device, args.max_length)
                    for candidate in candidate_answers(row)]
                ranked_first = true_nll < min(false_nll)
                correct += ranked_first
                true_probability = math.exp(-true_nll)
                false_probability = math.exp(-sum(false_nll) / len(false_nll))
                share = true_probability / (true_probability + false_probability)
                truth_shares.append(share)
                ranking_details.append({"question": row["paraphrased_question"],
                                        "correct_answer_nll": true_nll,
                                        "mean_wrong_answer_nll": sum(false_nll) / len(false_nll),
                                        "correct_answer_ranked_first": ranked_first,
                                        "true_answer_probability_share": share})
            results[group]["correct_answer_ranked_first"] = correct
            results[group]["candidate_question_count"] = len(selected)
            results[group]["mean_true_answer_probability_share"] = (
                sum(truth_shares) / len(truth_shares))
            results[group]["ranking_details"] = ranking_details
        if args.generate:
            details = []
            for row in selected:
                answer = generate_answer(model, tokenizer, row["paraphrased_question"],
                                         args.device, args.max_new_tokens)
                details.append({"question": row["paraphrased_question"],
                                "reference": row["paraphrased_answer"],
                                "generated": answer,
                                "rouge_l_recall": rouge_l_recall(
                                    answer, row["paraphrased_answer"])})
            results[group]["mean_generated_rouge_l_recall"] = (
                sum(row["rouge_l_recall"] for row in details) / len(details))
            results[group]["generation_details"] = details
    record = {"base": str(args.base.resolve()),
              "adapter": str(args.adapter.resolve()) if args.adapter else None,
              "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors")
              if args.adapter else None, "seed": args.seed,
              "max_length": args.max_length, "rank_candidates": args.rank_candidates,
              "generate": args.generate,
              "max_new_tokens": args.max_new_tokens if args.generate else None,
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
