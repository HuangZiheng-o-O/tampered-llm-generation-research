"""Read all configured data through official classes; no model/optimizer loading."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

from omegaconf import OmegaConf
from transformers import AutoTokenizer, TrainingArguments


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--base", type=Path)
    p.add_argument("--cpu-model-load", action="store_true")
    args = p.parse_args()
    root = args.root / "official-pilot-v1"
    configs = root / "configs"
    from data import get_data, get_collators, get_datasets
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    if tokenizer.vocab_size != 32000 or tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
        raise ValueError("Unexpected Llama-2 vocabulary")
    tokenizer.pad_token = tokenizer.eos_token
    jobs = json.loads((configs / "jobs.json").read_text())
    base_identity = None
    if args.base:
        expected = json.loads(Path(__file__).with_name("meta-base-identity.json").read_text())
        checked = {}
        for name, info in expected["safetensors"].items():
            path = args.base / name
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != info["sha256"] or path.stat().st_size != info["bytes"]:
                raise ValueError(f"Base model differs from Meta's published hash: {name}")
            checked[name] = info
        from safetensors import safe_open
        index = json.loads((args.base / "model.safetensors.index.json").read_text())
        actual = {}
        for name in checked:
            with safe_open(args.base / name, framework="pt", device="cpu") as tensors:
                actual.update({key: name for key in tensors.keys()})
        if actual != index["weight_map"]:
            raise ValueError("Safetensors index does not match its shard headers")
        base_identity = dict(expected, verified_shards=checked, tensor_count=len(actual))
    training = {}
    for name in ("learn_full", "learn_retain_reference", "NPO", "SimNPO", "GradDiff", "RMU", "UNDIAL"):
        cfg = OmegaConf.load(configs / f"{name}.yaml")
        data = get_data(cfg.data, mode=cfg.mode, tokenizer=tokenizer, template_args=cfg.model.template_args)
        actual_args = OmegaConf.to_container(cfg.trainer.args)
        actual_args.update(deepspeed=None, bf16=False, bf16_full_eval=False, use_cpu=True)
        TrainingArguments(**actual_args)  # CPU validation of the real argument schema.
        collator = get_collators(cfg.collator, tokenizer=tokenizer)
        total, longest = 0, 0
        seen = hashlib.sha256()
        if name.startswith("learn_"):
            datasets = [data["train"]]
        else:
            datasets = [data["train"].forget, data["train"].retain]
        for dataset in datasets:
            for index in range(len(dataset)):
                item = dataset[index]
                valid = item["labels"] != -100
                if not bool(valid.any()) or len(item["input_ids"]) > 512:
                    raise ValueError(f"Invalid tokenization: {name}/{index}")
                if int(item["input_ids"].max()) >= 32000:
                    raise ValueError("Input exceeds the base vocabulary")
                seen.update(item["input_ids"].numpy().tobytes())
                longest = max(longest, len(item["input_ids"]))
                if index < 2:
                    collated = collator([item])
                    if not bool(collated["attention_mask"][0, -1]):
                        raise ValueError("A real end-of-sequence token was hidden")
                total += 1
        training[name] = {"rows_tokenized": total, "primary_rows_per_epoch": len(data["train"]),
                          "retain_pool_rows": 0 if name.startswith("learn_") else len(data["train"].retain),
                          "max_tokens": longest,
                          "all_input_ids_sha256": seen.hexdigest(), "handler": cfg.trainer.handler}
    eval_datasets = {}
    def inspect_datasets(node, template):
        if isinstance(node, list):
            for item in node:
                inspect_datasets(item, template)
        elif isinstance(node, dict):
            if "datasets" in node:
                key = json.dumps(node["datasets"], sort_keys=True)
                if key not in eval_datasets:
                    dataset = get_datasets(OmegaConf.create(node["datasets"]), tokenizer=tokenizer, template_args=template)
                    mapping = dataset if isinstance(dataset, dict) else {"single": dataset}
                    count = 0
                    for part in mapping.values():
                        for i in range(len(part)):
                            part[i]  # Reads every candidate answer and requested field.
                        count += len(part)
                    eval_datasets[key] = count
            for value in node.values():
                inspect_datasets(value, template)
    for name, job in jobs["jobs"].items():
        if job["kind"] == "eval" and not name.startswith("eval_mmlu_"):
            cfg = OmegaConf.load(configs / f"{name}.yaml")
            inspect_datasets(OmegaConf.to_container(cfg.eval), cfg.model.template_args)
    from lm_eval.tasks import TaskManager, get_task_dict
    manager = TaskManager(include_path=str(root / "tasks"))
    counts = {}
    for split in ("validation", "test"):
        loaded = get_task_dict([f"pilot_mmlu_{split}"], task_manager=manager)
        def flatten(mapping):
            for value in mapping.values():
                if isinstance(value, dict):
                    yield from flatten(value)
                else:
                    yield value
        tasks = [x for x in flatten(loaded) if hasattr(x, "test_docs")]
        counts[split] = {"subjects": len(tasks), "test_questions": sum(len(x.test_docs()) for x in tasks),
                         "fewshot_questions": sum(len(x.fewshot_docs()) for x in tasks)}
    if counts["validation"]["test_questions"] != 1531 or counts["test"]["test_questions"] != 14042:
        raise ValueError(f"Offline MMLU task counts changed: {counts}")
    report = {"status": "offline_checks_passed", "training_started": False, "optimizer_steps": 0,
              "training_enabled": jobs["training_enabled"], "training_data": training,
              "unique_eval_dataset_configurations": len(eval_datasets), "mmlu": counts,
              "base_identity": base_identity,
              "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "accelerate", "datasets", "deepspeed", "lm-eval")},
              "tokenizer_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.tokenizer.glob("tokenizer*") if p.is_file()},
              "pending_runtime_check": "Full 7B GPU forward/backward/save/resume awaits permission to start training"}
    if args.cpu_model_load:
        import torch
        from model import get_model
        torch.set_num_threads(8)
        cfg = OmegaConf.load(configs / "learn_full.yaml")
        cfg.model.model_args.device_map = "cpu"
        model, model_tokenizer = get_model(cfg.model)
        model.eval()
        inputs = model_tokenizer("[INST] What is two plus two? [/INST]", return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Full base model produces non-finite outputs")
        report["base_cpu_load_check"] = {"device": str(model.device), "dtype": str(model.dtype),
                                          "parameters": sum(p.numel() for p in model.parameters()),
                                          "logits_shape": list(logits.shape), "finite": True,
                                          "optimizer_steps": 0}
    (root / "offline-checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
