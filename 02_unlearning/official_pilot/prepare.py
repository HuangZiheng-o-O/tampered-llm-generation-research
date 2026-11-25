"""Compose upstream recipes; localize data; never load or train a model."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

PIN = "17cbbc87192e6934deb92875c359c91bbd837fb4"
METHODS = ("NPO", "SimNPO", "GradDiff", "RMU", "UNDIAL")


def localize(value, datasets, cache):
    if isinstance(value, list):
        for item in value:
            localize(item, datasets, cache)
    elif isinstance(value, dict):
        hf = value.get("hf_args", {})
        if hf.get("path") == "locuslab/TOFU":
            name = hf.pop("name")
            hf.update(path="json", data_files=str(datasets / "tofu" / f"{name}.json"), cache_dir=str(cache))
        for item in value.values():
            localize(item, datasets, cache)


def recipe(upstream, name, overrides):
    with initialize_config_dir(config_dir=str(upstream / "configs"), version_base=None):
        cfg = compose(config_name=name, overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.paths.root_dir = "/unused"
    cfg.paths.output_dir = "/unused"
    del cfg.paths.work_dir  # Hydra runtime has no value during an offline compose.
    return cfg


def build(upstream, root, datasets, base, world_size):
    root = root / "official-pilot-v1"
    configs, jobs = {}, {}

    def model_config(cfg, path, evaluate=False):
        cfg.model.model_args.pretrained_model_name_or_path = str(path)
        cfg.model.model_args.attn_implementation = "sdpa"
        cfg.model.model_args.local_files_only = True
        cfg.model.model_args.use_cache = evaluate
        if evaluate:
            cfg.model.model_args.device_map = "cuda:0"
        cfg.model.tokenizer_args.pretrained_model_name_or_path = str(base)
        cfg.model.tokenizer_args.local_files_only = True
        cfg.model.template_args.response_prefix = " "
        cfg.model.template_args.strict_token_boundary = True

    for stage in ("learn_full", "learn_retain_reference") + METHODS:
        learning = stage.startswith("learn_")
        experiment = "finetune/tofu/default" if learning else "unlearn/tofu/default"
        overrides = [f"experiment={experiment}", "model=Llama-2-7b-chat-hf", f"task_name={stage}"]
        if not learning:
            overrides += [f"trainer={stage}"]
        cfg = recipe(upstream, "train", overrides)
        cfg.mode = "train" if learning else "unlearn"
        cfg.eval = None  # Evaluation is a separate official eval job.
        cfg.paths.output_dir = str(root / "runs" / stage)
        cfg.paths.root_dir = str(root)
        cfg.resume_from_checkpoint = None
        cfg.forget_split, cfg.retain_split, cfg.holdout_split = "forget01", "retain99", "holdout01"
        if learning:
            cfg.data.train.TOFU_QA_full.args.hf_args.name = "full" if stage == "learn_full" else "retain99"
        model_config(cfg, base if learning else root / "runs/learn_full")
        args = cfg.trainer.args
        args.pop("warmup_epochs", None)
        args.update({"per_device_train_batch_size": 1, "per_device_eval_batch_size": 1,
                     "gradient_accumulation_steps": 32 // world_size,
                     "warmup_steps": (125 if stage == "learn_full" else 124) if learning else 1,
                     "deepspeed": str(root / "configs/zero3_cpu.json"),
                     "gradient_checkpointing": True,
                     "gradient_checkpointing_kwargs": {"use_reentrant": False},
                     "optim": "adamw_torch", "report_to": [], "logging_steps": 1,
                     "do_train": True, "do_eval": False, "eval_on_start": False,
                     "eval_strategy": "no", "save_strategy": "steps", "save_steps": .5,
                     "save_only_model": False, "save_total_limit": None,
                     "remove_unused_columns": False, "dataloader_num_workers": 0,
                     "seed": 0, "data_seed": 0, "ddp_find_unused_parameters": False})
        if stage == "RMU":
            cfg.trainer.method_args.module_regex = "model[.]layers[.]7"
        if stage == "UNDIAL":
            args.learning_rate = 1e-4  # Preserve this method's own upstream recipe.
        data = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        localize(data, datasets, root / "cache/datasets")
        configs[stage] = data
        jobs[stage] = {"entry": "src/train.py", "requires": [] if learning else ["learn_full"],
                       "output": data["paths"]["output_dir"], "kind": "train"}

    for stage, model in [("base", base), ("full", root / "runs/learn_full"),
                         ("retain_reference", root / "runs/learn_retain_reference")] + [(m, root / "runs" / m) for m in METHODS]:
        for variant in ("original", "paraphrased"):
            bootstrap_name = f"eval_retain_reference_bootstrap_{variant}"
            reference = str(root / "evals" / bootstrap_name / "TOFU_EVAL.json")
            name = f"eval_{stage}_{variant}"
            cfg = recipe(upstream, "eval", ["model=Llama-2-7b-chat-hf", f"task_name={name}"])
            cfg.paths.root_dir = str(root)
            cfg.paths.output_dir = str(root / "evals" / name)
            model_config(cfg, model, evaluate=True)
            cfg.eval.tofu.forget_split, cfg.eval.tofu.holdout_split = "forget01", "holdout01"
            cfg.eval.tofu.batch_size = 2
            cfg.eval.tofu.question_key = "question" if variant == "original" else "paraphrased_question"
            cfg.eval.tofu.retain_logs_path = reference
            data = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            localize(data, datasets, root / "cache/datasets")
            configs[name] = data
            jobs[name] = {"entry": "src/eval.py", "requires": [bootstrap_name], "model": str(model), "kind": "eval"}
            if stage == "retain_reference":
                bootstrap = copy.deepcopy(data)
                bootstrap["task_name"] = bootstrap_name
                bootstrap["paths"]["output_dir"] = str(root / "evals" / bootstrap_name)
                bootstrap["eval"]["tofu"]["output_dir"] = bootstrap["paths"]["output_dir"]
                metrics = bootstrap["eval"]["tofu"]["metrics"]
                metrics["mia_min_k"] = copy.deepcopy(metrics["privleak"]["pre_compute"]["mia_min_k"])
                metrics.pop("forget_quality")
                metrics.pop("privleak")
                configs[bootstrap_name] = bootstrap
                jobs[bootstrap_name] = {"entry": "src/eval.py", "requires": ["learn_retain_reference"], "model": str(model), "kind": "eval"}
        for split in ("validation", "test"):
            name = f"eval_mmlu_{stage}_{split}"
            cfg = recipe(upstream, "eval", ["eval=lm_eval", "model=Llama-2-7b-chat-hf", f"task_name={name}"])
            cfg.paths.root_dir = str(root)
            cfg.paths.output_dir = str(root / "evals" / name)
            model_config(cfg, model, evaluate=True)
            cfg.eval.lm_eval.include_path = str(root / "tasks")
            cfg.eval.lm_eval.tasks = [f"pilot_mmlu_{split}"]
            cfg.eval.lm_eval.simple_evaluate_args = {"batch_size": 1, "num_fewshot": 5, "log_samples": True,
                                                    "apply_chat_template": False, "random_seed": 0,
                                                    "numpy_random_seed": 0, "torch_random_seed": 0,
                                                    "fewshot_random_seed": 0}
            configs[name] = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            jobs[name] = {"entry": "src/eval.py", "requires": [], "model": str(model), "kind": "eval"}
    return configs, jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--datasets", type=Path)
    p.add_argument("--base", type=Path, default=Path("/local/rcs/zh2701/LLM-safety/models/Llama-2-7b-chat-hf"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--world-size", type=int, choices=(1, 2, 4), default=2)
    args = p.parse_args()
    configs, jobs = build(args.upstream.resolve(), args.root, args.datasets or args.root / "datasets", args.base, args.world_size)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, config in configs.items():
        config["hydra"] = {"run": {"dir": config["paths"]["output_dir"]}, "job": {"chdir": False}}
        OmegaConf.save(OmegaConf.create(config), args.output / f"{name}.yaml")
    ds = json.loads(Path(__file__).with_name("zero3_cpu.json").read_text())
    (args.output / "zero3_cpu.json").write_text(json.dumps(ds, indent=2) + "\n")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob("*.yaml")}
    manifest = {"upstream_commit": PIN, "training_enabled": False, "world_size": args.world_size,
                "scope": "TOFU forget01 qualification on five methods; subsequent distinct-target experiments use separately frozen manifests",
                "jobs": jobs, "config_sha256": hashes,
                "source_sha256": {str(p.relative_to(args.upstream)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in args.upstream.joinpath("src").rglob("*.py")}}
    (args.output / "jobs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"configs_prepared": len(configs), "training_started": False, "methods": list(METHODS)}))


if __name__ == "__main__":
    main()
