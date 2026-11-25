"""Check every prepared record, official loader and eval task without training."""
import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf
from transformers import AutoTokenizer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from official_pilot import launch
from official_knowledge.prepare_data import SUITE


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream / "src"))
    from data import get_data, get_collators
    from lm_eval.tasks import TaskManager, get_task_dict

    root = args.root / SUITE
    manifest = json.loads((root / "configs/jobs.json").read_text())
    readiness = json.loads((root / "data-readiness.json").read_text())
    assert not manifest["training_enabled"] and sha(root / "data-readiness.json") == manifest["data_readiness_sha256"]
    for name, expected in readiness["derived_sha256"].items():
        assert sha(root / name) == expected, name
    for name, expected in manifest["shared_files_sha256"].items():
        assert sha(Path(name)) == expected, name
    for name, expected in manifest.get("support_sha256", {}).items():
        assert sha(Path(manifest["support_root"]) / name) == expected, name
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    assert len(tokenizer) == 32000 and tokenizer.bos_token_id == 1 and tokenizer.eos_token_id == 2
    tokenizer.pad_token = tokenizer.eos_token
    files, jobs, locks = {}, {}, 0
    for name, job in manifest["jobs"].items():
        path = root / "configs" / f"{name}.yaml"
        assert sha(path) == manifest["config_sha256"][path.name]
        if job["kind"] != "train":
            continue
        cfg = OmegaConf.load(path)
        actual_args = OmegaConf.to_container(cfg.trainer.args)
        actual_args.update(deepspeed=None, bf16=False, bf16_full_eval=False, use_cpu=True)
        TrainingArguments(**actual_args)
        data = get_data(cfg.data, mode=cfg.mode, tokenizer=tokenizer, template_args=cfg.model.template_args)
        collator = get_collators(cfg.collator, tokenizer=tokenizer)
        datasets = [data["train"]] if cfg.mode == "train" else [data["train"].forget, data["train"].retain]
        for dataset in datasets:
            file = dataset.data.cache_files[0]["filename"]
            if file in files:
                continue
            longest = 0
            for index in range(len(dataset)):
                row, item = dataset.data[index], dataset[index]
                head = tokenizer.encode(row["prompt"], add_special_tokens=True)
                full = tokenizer.encode(row["prompt"] + row["text"], add_special_tokens=True)
                assert len(full) <= 1024 and full[:len(head)] == head
                assert item["input_ids"].tolist() == full
                assert item["labels"].tolist() == [-100] * len(head) + full[len(head):]
                assert int(item["labels"][-1]) == 2 and bool(item["attention_mask"][-1])
                longest = max(longest, len(full))
            batch = collator([dataset[0], dataset[-1]])
            assert int(batch["attention_mask"].sum()) == len(dataset[0]["input_ids"]) + len(dataset[-1]["input_ids"])
            files[file] = {"rows": len(dataset), "max_tokens": longest}
        if cfg.mode == "unlearn":
            fids = set(data["train"].forget.data["unit_id"])
            rids = set(data["train"].retain.data["unit_id"])
            assert not fids & rids
        jobs[name] = {"primary_rows_per_epoch": len(data["train"]), "epochs": cfg.trainer.args.num_train_epochs,
                      "handler": cfg.trainer.handler, "learning_rate": cfg.trainer.args.learning_rate}
        with patch("sys.argv", ["launch", name, "--root", str(args.root), "--suite", SUITE, "--upstream", str(args.upstream), "--execute"]), \
             patch.object(launch.subprocess, "run") as run, patch.object(launch.subprocess, "Popen") as popen, contextlib.redirect_stdout(io.StringIO()):
            try:
                launch.main()
            except SystemExit as e:
                assert "disabled" in str(e)
            else:
                raise AssertionError("Execution lock did not stop the job")
            run.assert_not_called()
            popen.assert_not_called()
        locks += 1
    manager = TaskManager(include_path=str(root / "tasks"))
    task_counts = {}
    for source in readiness["sources"]:
        directory = root / "data" / source
        target_sets = {}
        for phase in ("development", "confirmation", "retain"):
            name = f"knowledge_{source}_{phase}"
            task = get_task_dict([name], task_manager=manager)[name]
            docs = list(task.test_docs())
            assert len(docs) == readiness["sources"][source]["eval_counts"][phase]
            for doc in docs:
                assert task.doc_to_text(doc) == doc["prompt"]
                scores = task.process_results(doc, [doc["answers"][0]])
                assert all(v == 1 for v in scores.values())
                wrong = task.process_results(doc, ["zzzz-no-knowledge-fixture-zzzz"])
                assert wrong["knowledge_accuracy"] == 0
            task_counts[name] = len(docs)
            target_sets[phase] = {d["unit_id"] for d in docs}
        assert not target_sets["development"] & target_sets["confirmation"]
        assert not (target_sets["development"] | target_sets["confirmation"]) & target_sets["retain"]
        units = [json.loads(x) for x in (directory / "units.jsonl").read_text().splitlines()]
        future = [json.loads(x) for x in (directory / "future-candidates.jsonl").read_text().splitlines()]
        assert not {u["unit_id"] for u in units} & {u["unit_id"] for u in future}
    report = {"status": "prepared_data_and_official_interfaces_verified", "training_enabled": False,
              "optimizer_steps": 0, "language_model_loaded": False, "records_verified_unique_loaders": sum(v["rows"] for v in files.values()),
              "unique_loader_files": len(files), "max_tokens": max(v["max_tokens"] for v in files.values()),
              "training_jobs_locked": locks, "training_configs": jobs, "eval_task_counts": task_counts,
              "config_count": len(manifest["jobs"]), "data_readiness_sha256": manifest["data_readiness_sha256"]}
    (root / "offline-checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "training_configs"}, indent=2))


if __name__ == "__main__":
    main()
