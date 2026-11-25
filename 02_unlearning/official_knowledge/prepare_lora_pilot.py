"""Prepare a content-bounded LoRA pilot using the existing published records."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from official_knowledge.prepare_data import matched_control, write
from official_knowledge.resource_guard import POLICY


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous-suite", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-key", choices=("llama2-chat", "llama3-instruct"), required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--gpu-pool", nargs="+", type=int)
    args = parser.parse_args()
    output = args.root / ("official-lora-pilot-v1-" + args.base_key)
    if (output / "RUN_STATUS.json").exists() or (output / "runs").exists():
        raise ValueError("Cannot refresh a suite after execution has started")
    output.mkdir(exist_ok=True)
    support = Path(__file__).resolve().parents[1]
    prior = args.previous_suite
    previous = json.loads((prior / "configs/jobs.json").read_text())
    readiness = json.loads((prior / "data-readiness.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    readiness.update(suite=output.name, training_enabled=False,
        scope="representative_lora_pilot_only", base=str(args.base))
    for source, info in readiness["sources"].items():
        folder = output / "data" / source
        probes = {}
        for phase in ("development", "confirmation", "retain"):
            probes[phase] = [json.loads(s) for s in (prior / "data" / source / f"eval-{phase}.jsonl").read_text().splitlines()]
            if source == "squad" and args.base_key == "llama3-instruct":
                for probe in probes[phase]:
                    question = probe["prompt"].removeprefix("[INST] ").removesuffix(" [/INST]")
                    probe["prompt"] = tokenizer.apply_chat_template(
                        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
            write(folder / f"eval-{phase}.jsonl", probes[phase])
        selected = {row["unit_id"] for rows in probes.values() for row in rows}
        rows = [json.loads(s) for s in (prior / "data" / source / "learn.jsonl").read_text().splitlines()]
        rows = [row for row in rows if row["unit_id"] in selected]
        if {r["unit_id"] for r in rows} != selected:
            raise ValueError("An assessment target is absent from the learning pool")
        for row in rows:
            row["text"] = row["text"].removesuffix("</s>") + tokenizer.eos_token
            if row["kind"] == "published_qa" and args.base_key == "llama3-instruct":
                question = row["prompt"].removeprefix("[INST] ").removesuffix(" [/INST]")
                row["prompt"] = tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
                row["text"] = row["text"].lstrip()
            row["tokens"] = len(tokenizer(row.get("prompt", "") + row["text"], add_special_tokens=True)["input_ids"])
            if row["tokens"] > 1024:
                raise ValueError("Selected text exceeds this tokenizer's input limit")
        write(folder / "learn.jsonl", rows)
        targets = {target["unit_id"] for target in info["targets"].values()}
        control_pool = [r for r in rows if r["unit_id"] not in targets]
        for phase, target in info["targets"].items():
            forget = [r for r in rows if r["unit_id"] == target["unit_id"]]
            retain = [r for r in rows if r["unit_id"] != target["unit_id"]]
            controls = matched_control(forget, control_pool, target["unit_id"])
            for kind, data in (("forget", forget), ("retain", retain), ("control", controls)):
                write(folder / f"{phase}-{kind}.jsonl", data)
            target.update(forget_rows=len(forget), retain_pool_rows=len(retain), control_rows=len(controls),
                forget_tokens=sum(r["tokens"] for r in forget), control_tokens=sum(r["tokens"] for r in controls))
        info.update(learning_pool_units=len(selected), learning_rows=len(rows),
            learning_tokens=sum(r["tokens"] for r in rows),
            pilot_candidate_units=len(targets), retain_units=len(selected - targets))
    readiness["derived_sha256"] = {str(p.relative_to(output)): sha(p) for p in sorted((output / "data").rglob("*.jsonl"))}
    write_readiness = output / "data-readiness.json"
    write_readiness.write_text(json.dumps(readiness, indent=2) + "\n")
    shutil.copytree(prior / "tasks", output / "tasks", dirs_exist_ok=True)
    for p in (output / "tasks").glob("*.yaml"):
        p.write_text(p.read_text().replace(str(prior), str(output)))
    configs = output / "configs"
    configs.mkdir(exist_ok=True)
    recipe = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
    methods = ("NPO", "GradDiff")
    jobs = {}
    for name, original_job in previous["jobs"].items():
        job = copy.deepcopy(original_job)
        source = job["source"] if job["kind"] == "eval" else name.split("_")[0]
        stage = job.get("stage", name.removeprefix(source + "_"))
        selected_stages = {"base", "learn"} | {f"{phase}_{m}" for phase in ("development", "confirmation") for m in (*methods, "control_1e-4")}
        if stage not in selected_stages or job.get("assessment") == "mmlu_test":
            continue
        text = (prior / "configs" / (name + ".yaml")).read_text().replace(str(prior), str(output))
        cfg = OmegaConf.to_container(OmegaConf.create(text), resolve=True)
        job["output"] = str(output / ("runs" if job["kind"] == "train" else "evals") / name)
        cfg["paths"]["output_dir"] = job["output"]
        cfg["hydra"]["run"]["dir"] = job["output"]
        parent = output / "runs" / (source + "_learn") / "merged"
        cfg["model"]["model_args"].update(pretrained_model_name_or_path=str(args.base if stage == "base" or name.endswith("_learn") else parent), device_map="cuda:0")
        cfg["model"]["tokenizer_args"]["pretrained_model_name_or_path"] = str(args.base)
        cfg["model"]["template_args"]["asst_end_tag"] = "<|eot_id|>" if args.base_key == "llama3-instruct" else " "
        if job["kind"] == "train":
            cfg["model"]["lora"] = recipe
            cfg["native_checkpoint_io"] = False
            cfg["checkpoint_first_step"] = True
            cfg["resume_from_checkpoint"] = None
            train = cfg["trainer"]["args"]
            train.pop("deepspeed", None)
            train.update(per_device_train_batch_size=1, gradient_accumulation_steps=32,
                learning_rate=1e-4, save_steps=0.25, warmup_steps=1,
                output_dir=job["output"], logging_dir=str(Path(job["output"]) / "logs"))
            if not name.endswith("_learn"):
                phase = stage.split("_")[0]
                job["primary_rows"] = readiness["sources"][source]["targets"][phase]["forget_rows"]
                if "matching_control" in job:
                    job["matching_control"] = f"{source}_{phase}_control_1e-4"
        else:
            job["model"] = cfg["model"]["model_args"]["pretrained_model_name_or_path"]
            cfg["eval"]["lm_eval"]["output_dir"] = job["output"]
            if stage not in ("base", "learn"):
                cfg["model"]["lora"] = {"adapter_path": str(output / "runs" / f"{source}_{stage}"), "trainable": False}
        if args.base_key == "llama3-instruct":
            cfg["model"]["template_args"].update(apply_chat_template=True, system_prompt="You are a helpful assistant.")
        OmegaConf.save(OmegaConf.create(cfg), configs / (name + ".yaml"))
        jobs[name] = job
    for source in readiness["sources"]:
        for assessment in ("knowledge", "mmlu_validation"):
            name = f"{source}_learn_adapter_{assessment}"
            parent_name = f"{source}_learn_{assessment}"
            cfg = OmegaConf.load(configs / (parent_name + ".yaml"))
            out = str(output / "evals" / name)
            cfg.task_name = name
            cfg.paths.output_dir = out
            cfg.hydra.run.dir = out
            cfg.eval.lm_eval.output_dir = out
            cfg.model.model_args.pretrained_model_name_or_path = str(args.base)
            OmegaConf.set_struct(cfg, False)
            cfg.model.lora = {"adapter_path": jobs[source + "_learn"]["output"], "trainable": False}
            OmegaConf.save(cfg, configs / (name + ".yaml"))
            jobs[name] = dict(jobs[parent_name], stage="learn_adapter", output=out, model=str(args.base))
    manifest = copy.deepcopy(previous)
    manifest.pop("deepspeed_config_sha256", None)
    manifest.update(training_enabled=True, world_size=1, methods=list(methods), support_root=str(support),
        verify_lora_merge_behavior=True,
        gpu_reservation_directory=str(args.root / "lora-pilot-gpu-reservations"),
        jobs=jobs, data_readiness_sha256=sha(write_readiness),
        config_sha256={p.name: sha(p) for p in configs.glob("*.yaml")},
        task_sha256={str(p.relative_to(output)): sha(p) for p in (output / "tasks").glob("*.yaml")},
        source_sha256={name: sha(args.upstream / name) for name in previous["source_sha256"]},
        support_sha256={name: sha(support / name) for name in previous.get("support_sha256", {})},
        resource_policy=dict(POLICY, training_memory_max_gib=24, training_memory_high_gib=24,
            evaluation_memory_max_gib=16, evaluation_memory_high_gib=16))
    if args.gpu_pool:
        manifest["gpu_pool"] = [str(gpu) for gpu in args.gpu_pool]
    for relative in ("official_pilot/lora.py", "official_knowledge/prepare_lora_pilot.py"):
        manifest["support_sha256"][relative] = sha(support / relative)
    (configs / "jobs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"suite": str(output), "jobs": len(jobs), "data": readiness["sources"]}, indent=2))


if __name__ == "__main__":
    main()
