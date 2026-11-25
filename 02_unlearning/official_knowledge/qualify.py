"""Summarize official lm-eval logs and issue learned/forgotten-knowledge receipts.

Never runs inference or training. Standard EM/F1 comes from knowledge_metrics;
biography facts aggregate their published alternative completion expressions.
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from official_knowledge.knowledge_metrics import process_results
from official_knowledge.acceptance import effect_checks, unlearning_passed


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def load_assessment(root, manifest, source, stage, assessment, evidence):
    name = f"{source}_{stage}_{assessment}"
    from official_knowledge.evaluation_alias import resolve
    alias = resolve(root, manifest, name)
    if alias:
        original, original_manifest = alias
        files = {}
        value = load_assessment(original, original_manifest, source, stage, assessment, files)
        evidence.update({str(original / key): digest for key, digest in files.items()})
        return value
    directory = Path(manifest['jobs'][name]['output'])
    marker = directory / "prepared-evaluation.json"
    identity = json.loads(marker.read_text())
    assert identity["job"] == name and identity["config_sha256"] == manifest["config_sha256"][name + ".yaml"]
    assert identity["data_readiness_sha256"] == manifest["data_readiness_sha256"]
    assert identity["model"] == manifest["jobs"][name]["model"]
    if not (root / "PROOF_KIND.txt").exists():
        assert identity.get("proof_kind") != "software_fixture_only" and identity["model_files"]
        for name, state in identity["model_files"].items():
            path = Path(name)
            assert [path.stat().st_size, path.stat().st_mtime_ns] == state, "Evaluated model files changed"
    for file in (marker, directory / "LMEval_EVAL.json", directory / "LMEval_SUMMARY.json"):
        evidence[str(file.relative_to(root))] = sha(file)
    logs = json.loads((directory / "LMEval_EVAL.json").read_text())
    summary = json.loads((directory / "LMEval_SUMMARY.json").read_text())
    if assessment.startswith("mmlu_"):
        split = assessment.removeprefix("mmlu_")
        score = summary[f"pilot_mmlu_{split}/acc"]
        assert 0 <= score <= 1
        # Completeness is checked from the official per-question sample logs.
        group = logs[f"pilot_mmlu_{split}"]
        count = sum(len(rows) for rows in group.values())
        assert count == (1531 if split == "validation" else 14042), count
        return score
    result = {}
    for phase in ("development", "confirmation", "retain"):
        expected = {q["probe_id"]: q for q in records(root / "data" / source / f"eval-{phase}.jsonl")}
        task = f"knowledge_{source}_{phase}"
        samples = logs[task][task]
        assert len(samples) == len(expected)
        scores, facts, used = [], collections.defaultdict(list), set()
        for sample in samples:
            doc = sample["doc"]
            pid = doc["probe_id"]
            assert pid not in used and doc == expected[pid]
            used.add(pid)
            prediction = sample["filtered_resps"][0]
            score = process_results(doc, [prediction])
            for key, value in score.items():
                assert abs(sample[key] - value) < 1e-8, (pid, key)
            scores.append(score)
            facts[doc["fact_id"]].append(score["knowledge_accuracy"])
        per_fact = {k: sum(v) / len(v) for k, v in facts.items()}
        result[phase] = {"probes": len(scores), "fact_or_question_groups": len(per_fact),
                         "knowledge_accuracy": sum(per_fact.values()) / len(per_fact),
                         "majority_correct_groups": sum(v >= .5 for v in per_fact.values()),
                         "exact_match": sum(s["exact_match"] for s in scores) / len(scores),
                         "answer_f1": sum(s["answer_f1"] for s in scores) / len(scores), "per_fact": per_fact}
    return result


def assess(root, source, stage=None):
    manifest = json.loads((root / "configs/jobs.json").read_text())
    assert sha(root / "data-readiness.json") == manifest["data_readiness_sha256"]
    evidence = {}
    learned = load_assessment(root, manifest, source, "learn", "knowledge", evidence)
    learned_mmlu = load_assessment(root, manifest, source, "learn", "mmlu_validation", evidence)
    policy = manifest["quality_policy"]
    if stage is None:
        base = load_assessment(root, manifest, source, "base", "knowledge", evidence)
        base_mmlu = load_assessment(root, manifest, source, "base", "mmlu_validation", evidence)
        global_ok = learned["retain"]["knowledge_accuracy"] >= policy["learned_retain_accuracy_min"] and base_mmlu - learned_mmlu <= policy["mmlu_drop_max"]
        if policy.get('acceptance_mode') == 'pipeline_first':
            global_ok = learned['retain']['majority_correct_groups'] > 0
        merge_check = None
        if manifest.get("verify_lora_merge_behavior"):
            adapter = load_assessment(root, manifest, source, "learn_adapter", "knowledge", evidence)
            adapter_mmlu = load_assessment(root, manifest, source, "learn_adapter", "mmlu_validation", evidence)
            kept = {}
            for phase in ("development", "confirmation", "retain"):
                correct = [key for key, value in adapter[phase]["per_fact"].items() if value >= .5]
                kept[phase] = (sum(learned[phase]["per_fact"][key] >= .5 for key in correct) / len(correct)) if correct else 0.0
            merge_ok = min(kept.values()) >= policy["retain_previously_correct_min"] and adapter_mmlu - learned_mmlu <= policy["mmlu_drop_max"]
            merge_check = {"passed": merge_ok, "adapter_previously_correct_fraction_kept": kept,
                "adapter_knowledge": adapter, "merged_knowledge": learned,
                "adapter_mmlu": adapter_mmlu, "merged_mmlu": learned_mmlu}
            global_ok = global_ok and merge_ok
        targets = {}
        for phase in ("development", "confirmation"):
            gain = learned[phase]["knowledge_accuracy"] - base[phase]["knowledge_accuracy"]
            targets[phase] = {"passed": global_ok and learned[phase]["knowledge_accuracy"] >= policy["learned_knowledge_accuracy_min"] and gain >= policy["learning_gain_min"],
                              "learning_gain": gain, "base": base[phase], "learned": learned[phase]}
            if policy.get('acceptance_mode') == 'pipeline_first':
                targets[phase]['passed'] = global_ok and learned[phase]['majority_correct_groups'] > 0
        report = {"kind": "learning_qualification", "global_checks_passed": global_ok, "targets": targets,
                  "mmlu_base": base_mmlu, "mmlu_learned": learned_mmlu, "retain": learned["retain"]}
        if merge_check is not None:
            report["adapter_merge_behavior_check"] = merge_check
        name = source + "-learning"
    else:
        phase = stage.split("_")[0]
        assert phase in ("development", "confirmation")
        after = load_assessment(root, manifest, source, stage, "knowledge", evidence)
        after_mmlu = load_assessment(root, manifest, source, stage, "mmlu_validation", evidence)
        ratios = {}
        for key in (phase, "retain"):
            before = learned[key]["per_fact"]
            previously_correct = [k for k, v in before.items() if v >= .5]
            assert previously_correct, "No previously correct knowledge to compare"
            ratios[key] = sum(after[key]["per_fact"][k] >= .5 for k in previously_correct) / len(previously_correct)
        control = "_control_" in stage
        if control:
            passed = ratios["retain"] >= policy["retain_previously_correct_min"] and learned_mmlu - after_mmlu <= policy["mmlu_drop_max"]
            passed = passed and ratios[phase] >= policy["retain_previously_correct_min"]
        else:
            passed = unlearning_passed(policy, ratios[phase], ratios["retain"], learned_mmlu, after_mmlu)
        report = {"kind": "normal_control_quality" if control else "unlearning_quality", "passed": passed,
                  "target_previously_correct_fraction_remaining": ratios[phase], "retain_previously_correct_fraction_remaining": ratios["retain"],
                  "mmlu_before": learned_mmlu, "mmlu_after": after_mmlu, "before": learned, "after": after}
        if not control:
            report['effect_checks'] = effect_checks(policy, ratios[phase], ratios['retain'], learned_mmlu, after_mmlu)
            report['acceptance_scope'] = policy.get('acceptance_mode', 'effect_thresholds')
        name = source + "-" + stage + "-quality"
    model_files = {}
    for marker in evidence:
        if marker.endswith("prepared-evaluation.json"):
            model_files.update(json.loads((root / marker).read_text())["model_files"])
    report.update(source=source, training_started_by_this_command=False,
                  quality_policy=dict(policy),
                  quality_policy_id=manifest.get("quality_policy_id", "legacy-strict-30pct"),
                  proof_kind="software_fixture_only" if (root / "PROOF_KIND.txt").exists() else "official_model_evaluation",
                  config_sha256=manifest["config_sha256"], model_files=model_files,
                  data_readiness_sha256=manifest["data_readiness_sha256"], evidence_sha256=evidence)
    directory = root / "gates"
    directory.mkdir(exist_ok=True)
    (directory / (name + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--source", choices=("biographies", "squad"), required=True)
    parser.add_argument("--stage", help="Omit to qualify learning; e.g. development_NPO to assess forgetting")
    args = parser.parse_args()
    result = assess(args.suite_root, args.source, args.stage)
    print(json.dumps({k: v for k, v in result.items() if k not in ("config_sha256", "evidence_sha256", "before", "after", "targets")}, indent=2))


if __name__ == "__main__":
    main()
