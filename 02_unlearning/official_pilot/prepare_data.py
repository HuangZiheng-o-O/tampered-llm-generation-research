"""Verify published files and materialize offline MMLU tasks; no model training."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    datasets, output = args.datasets, args.root / "official-pilot-v1"
    output.mkdir(parents=True, exist_ok=True)
    checked = []
    for path in sorted((datasets / "receipts").glob("*--*.json")):
        receipt = json.loads(path.read_text())
        actual = datasets / receipt["source"] / receipt["file"]
        if sha(actual) != receipt["sha256"] or actual.stat().st_size != receipt["bytes"]:
            raise ValueError(f"Source receipt mismatch: {actual}")
        checked.append({"source": receipt["source"], "file": receipt["file"], "sha256": receipt["sha256"]})
    if len(checked) != 37:
        raise ValueError("Expected 37 verified source files")
    tofu = {n: records(datasets / "tofu" / f"{n}.json") for n in ("full", "forget01", "retain99", "holdout01", "forget01_perturbed", "retain_perturbed", "real_authors_perturbed", "world_facts_perturbed")}
    ids = lambda xs: {(x["question"], x["answer"]) for x in xs}
    full, forget, retain, holdout = (ids(tofu[n]) for n in ("full", "forget01", "retain99", "holdout01"))
    if forget & retain or holdout & full or forget | retain != full:
        raise ValueError("Official split identities do not match")
    if ids(tofu["forget01_perturbed"]) != forget or not ids(tofu["retain_perturbed"]) <= retain:
        raise ValueError("Knowledge evaluation identities do not match")

    # Copy task definitions from the installed official lm-eval version.
    package = Path(importlib.util.find_spec("lm_eval").origin).parent
    source = package / "tasks/mmlu/default"
    template = yaml.safe_load((source / "_default_template_yaml").read_text())
    template["dataset_path"] = "parquet"
    tables = {split: pq.read_table(next((datasets / "mmlu/all").glob(split + "-*.parquet"))) for split in ("dev", "validation", "test")}
    subjects = sorted(set(tables["test"].column("subject").to_pylist()))
    if len(subjects) != 57:
        raise ValueError("Expected 57 MMLU subjects")
    task_dir = output / "tasks"
    task_dir.mkdir(exist_ok=True)
    task_names = {"validation": [], "test": []}
    mmlu_counts = {s: 0 for s in tables}
    for subject in subjects:
        files = {}
        for split, table in tables.items():
            subset = table.filter(pc.equal(table["subject"], subject))
            target = output / "mmlu" / subject / f"{split}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                pq.write_table(subset, target)
            elif pq.read_table(target).to_pylist() != subset.to_pylist():
                raise ValueError(f"An existing derived file differs: {target}")
            files[split] = str(target)
            mmlu_counts[split] += len(subset)
        origin = yaml.safe_load((source / f"mmlu_{subject}.yaml").read_text())
        for split in task_names:
            name = f"pilot_mmlu_{split}_{subject}"
            config = dict(template, task=name, dataset_name=None,
                          dataset_kwargs={"data_files": files, "cache_dir": str(output / "cache/mmlu")}, test_split=split,
                          description=origin["description"], num_fewshot=5)
            (task_dir / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            task_names[split].append(name)
    for split, names in task_names.items():
        group = {"group": f"pilot_mmlu_{split}", "task": names,
                 "aggregate_metric_list": [{"metric": "acc", "weight_by_size": True}],
                 "metadata": {"version": 1, "source": "lm-eval 0.4.11 MMLU with local pinned data"}}
        (task_dir / f"group_{split}.yaml").write_text(yaml.safe_dump(group, sort_keys=False))
    expected = {"dev": 285, "validation": 1531, "test": 14042}
    if mmlu_counts != expected:
        raise ValueError(f"MMLU counts changed: {mmlu_counts}")
    manifest = {"status": "data_verified", "training_enabled": False, "source_files_verified": len(checked),
                "tofu_counts": {n: len(v) for n, v in tofu.items()}, "forget_retain_overlap": 0,
                "holdout_full_overlap": 0, "mmlu_counts": mmlu_counts, "mmlu_subjects": len(subjects),
                "sources": checked, "derived_files_sha256": {str(p.relative_to(output)): sha(p) for p in output.glob("mmlu/*/*.parquet")},
                "task_config_sha256": {p.name: sha(p) for p in task_dir.glob("*.yaml")}}
    (output / "data-readiness.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("sources", "derived_files_sha256", "task_config_sha256")}))


if __name__ == "__main__":
    main()
