"""Reuse official recipes for the cleaned knowledge pilot; execution stays locked."""
import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from official_pilot.prepare import build as tofu_recipes, METHODS, PIN
from official_knowledge.prepare_data import SUITE, MAX_LENGTH
from official_knowledge.resource_guard import POLICY


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--world-size", type=int, choices=(1, 2, 4), default=1)
    p.add_argument("--biography-micro-batch", type=int, default=8)
    p.add_argument("--squad-micro-batch", type=int, default=2)
    args = p.parse_args()
    root = args.root / SUITE
    configs_dir = root / "configs"
    configs_dir.mkdir(exist_ok=True)
    tasks = root / "tasks"
    tasks.mkdir(exist_ok=True)
    original, _ = tofu_recipes(args.upstream, args.root, args.root / "datasets", args.base, args.world_size)
    old_root = str(args.root / "official-pilot-v1")
    original = json.loads(json.dumps(original).replace(old_root, str(root)))
    readiness = json.loads((root / "data-readiness.json").read_text())
    configs, jobs = {}, {}

    def hardware(config, source):
        micro = args.biography_micro_batch if source == "biographies" else args.squad_micro_batch
        if micro <= 0 or 32 % (micro * args.world_size):
            raise ValueError("Microbatch and world size must divide effective batch 32")
        config["trainer"]["args"].update(per_device_train_batch_size=micro, gradient_accumulation_steps=32 // (micro * args.world_size))

    def relocate(config, name, kind):
        cfg = copy.deepcopy(config)
        out = str(root / ("runs" if kind == "train" else "evals") / name)
        cfg["task_name"] = name
        cfg["paths"].update(root_dir=str(root), output_dir=out)
        cfg["hydra"] = {"run": {"dir": out}, "job": {"chdir": False}}
        for key in ("forget_split", "retain_split", "holdout_split", "retain_logs_path", "question_key"):
            cfg.pop(key, None)
        return cfg, out

    def dataset(path):
        return {"published_records": {"handler": "CompletionDataset", "args": {
            "hf_args": {"path": "json", "data_files": str(path), "split": "train", "cache_dir": str(root / "cache/datasets")},
            "prefix_key": "prompt", "text_key": "text", "max_length": MAX_LENGTH}}}

    for source, info in readiness["sources"].items():
        directory = root / "data" / source
        for phase in ("development", "confirmation", "retain"):
            task = f"knowledge_{source}_{phase}"
            task_cfg = {"task": task, "dataset_path": "json", "dataset_kwargs": {
                "data_files": {"test": str(directory / f"eval-{phase}.jsonl")}, "cache_dir": str(root / "cache/lm-eval")},
                "test_split": "test", "output_type": "generate_until", "doc_to_text": "{{prompt}}", "doc_to_target": "{{answers[0]}}",
                "target_delimiter": "", "num_fewshot": 0, "repeats": 1,
                "generation_kwargs": {"until": ["\n", "</s>"], "max_gen_toks": 48, "do_sample": False},
                "metric_list": [{"metric": m, "aggregation": "mean", "higher_is_better": True} for m in ("exact_match", "answer_f1", "knowledge_accuracy")],
                "metadata": {"version": 1, "context": "closed-book; no article passage or attribute answer in the input"}}
            text = yaml.safe_dump(task_cfg, sort_keys=False)
            text += "process_results: !function knowledge_metrics.process_results\n"
            (tasks / f"{task}.yaml").write_text(text)
        stages = {"base": {"model": str(args.base), "requires": []}}
        learning_name = f"{source}_learn"
        learning, output = relocate(original["learn_full"], learning_name, "train")
        learning["data"] = {"train": dataset(directory / "learn.jsonl")}
        learning["trainer"]["args"].update(output_dir=output, logging_dir=output + "/logs",
                                              warmup_steps=math.ceil(math.ceil(info["learning_rows"] / 32) * 5 * .1))
        hardware(learning, source)
        configs[learning_name] = learning
        jobs[learning_name] = {"entry": "src/train.py", "kind": "train", "output": output,
                               "requires": [f"{source}_base_knowledge", f"{source}_base_mmlu_validation"]}
        stages["learn"] = {"model": output, "requires": [learning_name]}
        for phase, target in info["targets"].items():
            controls = {}
            for lr in ("1e-5", "1e-4"):
                name = f"{source}_{phase}_control_{lr}"
                cfg, out = relocate(original["learn_full"], name, "train")
                cfg["model"]["model_args"]["pretrained_model_name_or_path"] = output
                cfg["data"] = {"train": dataset(directory / f"{phase}-control.jsonl")}
                cfg["trainer"]["args"].update(output_dir=out, logging_dir=out + "/logs", learning_rate=float(lr), num_train_epochs=10, warmup_steps=1)
                hardware(cfg, source)
                configs[name] = cfg
                jobs[name] = {"entry": "src/train.py", "kind": "train", "output": out, "requires": [learning_name],
                              "learning_gate": {"source": source, "phase": phase}, "primary_rows": target["control_rows"]}
                stages[f"{phase}_control_{lr}"] = {"model": out, "requires": [name]}
                controls[lr] = name
            for method in METHODS:
                name = f"{source}_{phase}_{method}"
                cfg, out = relocate(original[method], name, "train")
                cfg["model"]["model_args"]["pretrained_model_name_or_path"] = output
                cfg["trainer"]["args"].update(output_dir=out, logging_dir=out + "/logs")
                hardware(cfg, source)
                cfg["data"] = {"forget": dataset(directory / f"{phase}-forget.jsonl"),
                               "retain": dataset(directory / f"{phase}-retain.jsonl"), "anchor": "forget"}
                configs[name] = cfg
                jobs[name] = {"entry": "src/train.py", "kind": "train", "output": out, "requires": [learning_name],
                              "learning_gate": {"source": source, "phase": phase}, "primary_rows": target["forget_rows"],
                              "matching_control": controls["1e-4" if method == "UNDIAL" else "1e-5"]}
                stages[f"{phase}_{method}"] = {"model": out, "requires": [name]}
        for stage, spec in stages.items():
            for assessment in ("knowledge", "mmlu_validation", "mmlu_test"):
                name = f"{source}_{stage}_{assessment}"
                cfg, out = relocate(original["eval_mmlu_base_validation"], name, "eval")
                cfg["model"]["model_args"]["pretrained_model_name_or_path"] = spec["model"]
                lm = cfg["eval"]["lm_eval"]
                lm["output_dir"] = out
                lm["overwrite"] = False
                if assessment == "knowledge":
                    lm["include_path"] = str(tasks)
                    lm["tasks"] = [f"knowledge_{source}_{phase}" for phase in ("development", "confirmation", "retain")]
                    lm["simple_evaluate_args"]["num_fewshot"] = 0
                else:
                    split = assessment.removeprefix("mmlu_")
                    lm["include_path"] = str(args.root / "official-pilot-v1/tasks")
                    lm["tasks"] = [f"pilot_mmlu_{split}"]
                configs[name] = cfg
                jobs[name] = {"entry": "src/eval.py", "kind": "eval", "model": spec["model"], "requires": spec["requires"],
                              "source": source, "stage": stage, "assessment": assessment, "optional_final_assessment": assessment == "mmlu_test"}
    (tasks / "knowledge_metrics.py").write_bytes(Path(__file__).with_name("knowledge_metrics.py").read_bytes())
    for name, config in configs.items():
        if jobs[name]["kind"] == "train":
            config["native_checkpoint_io"] = True
            config["checkpoint_first_step"] = name.endswith("_learn")
        OmegaConf.save(OmegaConf.create(config), configs_dir / f"{name}.yaml")
    deepspeed = json.loads(Path(__file__).parents[1].joinpath("official_pilot/zero3_cpu.json").read_text())
    for section in ("offload_param", "offload_optimizer"):
        deepspeed["zero_optimization"][section]["pin_memory"] = False
    deepspeed["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = False
    (configs_dir / "zero3_cpu.json").write_text(json.dumps(deepspeed, indent=2) + "\n")
    manifest = {"suite": SUITE, "upstream_commit": PIN, "training_enabled": False, "world_size": args.world_size,
                "resource_policy": POLICY,
                "deepspeed_config_sha256": sha(configs_dir / "zero3_cpu.json"),
                "hardware_profile": {"effective_batch": 32, "biography_micro_batch": args.biography_micro_batch,
                                     "squad_micro_batch": args.squad_micro_batch},
                "scope": "two sources, one development and one confirmation target per source; five methods with budget-matched continuing-training controls",
                "jobs": jobs, "config_sha256": {name + ".yaml": sha(configs_dir / f"{name}.yaml") for name in configs},
                "data_readiness_sha256": sha(root / "data-readiness.json"),
                "source_sha256": {str(p.relative_to(args.upstream)): sha(p) for p in (args.upstream / "src").rglob("*.py")},
                "task_sha256": {str(p.relative_to(root)): sha(p) for p in tasks.iterdir() if p.is_file()},
                "quality_policy": {"learned_knowledge_accuracy_min": .8, "learning_gain_min": .2,
                                   "learned_retain_accuracy_min": .8, "mmlu_drop_max": .03,
                                   "remaining_previously_correct_max": .5, "retain_previously_correct_min": .95,
                                   "acceptance_mode": "pipeline_first"},
                "quality_policy_id": "pipeline-first-artifacts-20261106",
                "quality_policy_reason": "User authorizes artifact-first production; effect scores are recorded separately without threshold-based tuning or rejection."}
    shared_root = args.root / "official-pilot-v1"
    shared = json.loads((shared_root / "data-readiness.json").read_text())
    manifest["shared_files_sha256"] = {str(shared_root / "data-readiness.json"): sha(shared_root / "data-readiness.json")}
    manifest["shared_files_sha256"].update({str(shared_root / k): v for k, v in shared["derived_files_sha256"].items()})
    manifest["shared_files_sha256"].update({str(shared_root / "tasks" / k): v for k, v in shared["task_config_sha256"].items()})
    support = Path(__file__).resolve().parents[1]
    manifest["support_root"] = str(support)
    manifest["support_sha256"] = {str(p.relative_to(support)): sha(p)
                                  for package in ("official_knowledge", "official_pilot")
                                  for p in sorted((support / package).glob("*.py"))}
    (configs_dir / "jobs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"configs": len(configs), "train_jobs": sum(j["kind"] == "train" for j in jobs.values()),
                      "eval_jobs": sum(j["kind"] == "eval" for j in jobs.values()), "training_enabled": False}))


if __name__ == "__main__":
    main()
