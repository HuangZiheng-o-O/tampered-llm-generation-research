"""Print one official command; execution stays locked in the prepared manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from omegaconf import OmegaConf


def learning_configuration_matches(root, receipt, manifest):
    """Check the qualified model's dependencies, independently of other jobs."""
    required = set()
    from official_knowledge.evaluation_alias import resolve
    allowed = [root.resolve()]
    for name, job in manifest['jobs'].items():
        if job.get('evaluation_alias'):
            allowed.append(resolve(root, manifest, name)[0])
    for name in receipt.get('evidence_sha256', {}):
        if name.endswith('prepared-evaluation.json'):
            path = root / name
            if not any(path.resolve().is_relative_to(folder) for folder in allowed):
                return False
            required.add(json.loads(path.read_text())['job'])
    pending = list(required)
    while pending:
        name = pending.pop()
        for dependency in manifest['jobs'][name]['requires']:
            if dependency not in required:
                required.add(dependency)
                pending.append(dependency)
    frozen = receipt.get('config_sha256', {})
    return bool(required) and all(frozen.get(name + '.yaml') == manifest['config_sha256'].get(name + '.yaml')
                                  and name + '.yaml' in frozen for name in required)


def checkpoint(path, output):
    output, path = output.resolve(), path.resolve()
    if path.parent != output or not path.name.startswith("checkpoint-"):
        raise ValueError("Resume checkpoint must be a direct child of this job's output")
    if not (path / "trainer_state.json").is_file() or not list(path.glob("rng_state*.pth")):
        raise ValueError("Checkpoint lacks Trainer or RNG state")
    deepspeed = list(path.glob("**/*optim_states.pt")) + list(path.glob("**/*model_states.pt"))
    adapter = [path / name for name in ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt")]
    if all(p.is_file() for p in adapter):
        required = adapter
    elif list(path.glob("**/*optim_states.pt")) and list(path.glob("**/*model_states.pt")):
        required = deepspeed
    else:
        raise ValueError("Checkpoint lacks complete optimizer/model recovery files")
    if json.loads((path / "trainer_state.json").read_text())["global_step"] < 1:
        raise ValueError("Checkpoint has no completed optimizer step")
    if any(p.stat().st_size == 0 for p in required + list(path.glob("rng_state*.pth"))):
        raise ValueError("Checkpoint contains an empty state file")
    for entry in path.rglob("*"):
        if not entry.resolve().is_relative_to(path):
            raise ValueError("Checkpoint symlink escapes its directory")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--suite", default="official-pilot-v1")
    p.add_argument("--resume", type=Path)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--background", action="store_true")
    args = p.parse_args()
    if not args.suite.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Suite must be a single directory name")
    root = args.root.resolve() / args.suite
    config_dir = root / "configs"
    manifest = json.loads((config_dir / "jobs.json").read_text())
    job = manifest["jobs"][args.job]
    config_file = config_dir / (args.job + ".yaml")
    digest = hashlib.sha256(config_file.read_bytes()).hexdigest()
    if digest != manifest["config_sha256"][config_file.name]:
        raise ValueError("Configuration changed after preparation")
    cfg = OmegaConf.load(config_file)
    declared_output = Path(job["output"]).resolve()
    actual_output = cfg.trainer.args.output_dir if job["kind"] == "train" else cfg.eval.lm_eval.output_dir
    if Path(cfg.paths.output_dir).resolve() != declared_output or Path(actual_output).resolve() != declared_output:
        raise ValueError("Job manifest and official entry point use different output directories")
    deepspeed_digest = manifest.get("deepspeed_config_sha256")
    if job["kind"] == "train" and deepspeed_digest:
        if hashlib.sha256(Path(cfg.trainer.args.deepspeed).read_bytes()).hexdigest() != deepspeed_digest:
            raise ValueError("DeepSpeed configuration changed after preparation")
    policy = manifest.get("resource_policy")
    command = [sys.executable, str(args.upstream.resolve() / job["entry"]),
               "--config-path", str(config_dir), "--config-name", args.job]
    if policy:
        entry = Path(manifest["support_root"]) / "official_knowledge/runtime_entry.py"
        command = [sys.executable, str(entry), "--entry", str(args.upstream.resolve() / job["entry"])] + command[2:]
    if job["kind"] == "train":
        output = Path(job["output"])
        if args.resume:
            resume = checkpoint(args.resume, output)
            command += ["resume_from_checkpoint=" + json.dumps(str(resume))]
        elif list(output.glob("checkpoint-*")) or (output / "trainer_state.json").exists():
            raise ValueError("Existing training state requires an explicit --resume checkpoint")
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={manifest['world_size']}", "--"] + command[1:]
    elif args.resume:
        raise ValueError("Resume applies to training jobs")
    print(shlex.join(command), flush=True)
    if not args.execute:
        print("PREVIEW ONLY: no model loading or training", flush=True)
        return
    if job.get('evaluation_alias'):
        raise ValueError('This job is a read-only reuse of a complete evaluation; execution is disabled')
    if not manifest["training_enabled"]:
        raise SystemExit("Execution is disabled by the user's preparation-only instruction")
    if root.stat().st_uid != os.getuid():
        raise ValueError("Runtime root is not owned by the current user")
    for name, expected in manifest["source_sha256"].items():
        if hashlib.sha256((args.upstream / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Official source differs from the prepared copy: {name}")
    for name, expected in manifest.get("support_sha256", {}).items():
        if hashlib.sha256((Path(manifest["support_root"]) / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Prepared support code changed: {name}")
    if not (root / "data-readiness.json").is_file():
        raise ValueError("Data readiness receipt is absent")
    if manifest.get("data_readiness_sha256"):
        readiness_path = root / "data-readiness.json"
        if hashlib.sha256(readiness_path.read_bytes()).hexdigest() != manifest["data_readiness_sha256"]:
            raise ValueError("Data preparation differs from the frozen configuration")
        readiness = json.loads(readiness_path.read_text())
        for name, expected in readiness["derived_sha256"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Prepared data changed: {name}")
        for name, expected in manifest.get("task_sha256", {}).items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Evaluation task changed: {name}")
        for name, expected in manifest.get("shared_files_sha256", {}).items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Shared evaluation data/task changed: {name}")
    if job["kind"] == "train":
        from official_knowledge.pilot_recipes import validate_follow_up
        validate_follow_up(root, args.job, manifest)
    if job.get("learning_gate"):
        gate = job["learning_gate"]
        receipt = json.loads((root / "gates" / f"{gate['source']}-learning.json").read_text())
        if receipt.get("proof_kind") != "official_model_evaluation":
            raise ValueError("A software fixture cannot authorize model training")
        if not receipt.get("global_checks_passed") or not receipt["targets"][gate["phase"]]["passed"]:
            raise ValueError("This target has not passed learned-knowledge qualification")
        if not learning_configuration_matches(root, receipt, manifest) or receipt.get("data_readiness_sha256") != manifest.get("data_readiness_sha256"):
            raise ValueError("Learning qualification belongs to different data/configurations")
        for name, expected in receipt["evidence_sha256"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Learning qualification evidence changed")
        for name, states in receipt["model_files"].items():
            path = Path(name)
            if [path.stat().st_size, path.stat().st_mtime_ns] != states:
                raise ValueError("Qualified learning model changed")
    model_path = Path(cfg.model.model_args.pretrained_model_name_or_path)
    if not (model_path / "config.json").is_file() or not list(model_path.glob("*.safetensors")):
        raise ValueError("Required full-model weights are not present")
    for required in job["requires"]:
        dependency = manifest["jobs"][required]
        if dependency["kind"] == "train":
            if not (Path(dependency["output"]) / "trainer_state.json").exists():
                raise ValueError(f"Prerequisite training is incomplete: {required}")
        else:
            prior = OmegaConf.load(config_dir / f"{required}.yaml")
            if not list(Path(prior.paths.output_dir).glob("*_EVAL.json")):
                raise ValueError(f"Prerequisite evaluation is absent: {required}")
    if job["kind"] == "train":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(visible) != manifest["world_size"] or not all(x.isdigit() for x in visible):
            raise ValueError("Set explicit CUDA_VISIBLE_DEVICES matching the prepared world size")
        snapshot = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        available = {r.split(",")[0].strip() for r in snapshot.splitlines()
                     if int(r.split(",")[1]) < 512 and int(r.split(",")[2]) >= 23000 and int(r.split(",")[3]) < 5}
        if not set(visible) <= available:
            raise ValueError("Requested GPUs are occupied")
        if policy:
            from official_knowledge.resource_guard import idle_gpus
            idle, _ = idle_gpus()
            if not set(visible) <= set(idle):
                raise ValueError("Requested GPUs are occupied")
        mem_kib = int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))
        required_memory = policy["training_memory_max_gib"] + policy["host_start_reserve_gib"] if policy else 150
        if mem_kib < required_memory * 1024**2 or __import__("shutil").disk_usage(root).free < 250 * 1024**3:
            raise ValueError(f"Require {required_memory} GiB available RAM and 250 GiB available disk before training")
        identity = output / "prepared-job.json"
        record = {"config_sha256": digest, "world_size": manifest["world_size"], "upstream_commit": manifest["upstream_commit"]}
        if deepspeed_digest:
            record["deepspeed_config_sha256"] = deepspeed_digest
        output.mkdir(parents=True, exist_ok=True)
        if args.resume and not identity.exists():
            raise ValueError("Resume checkpoint has no matching prepared-job identity")
        if identity.exists() and json.loads(identity.read_text()) != record:
            raise ValueError("Resume configuration or world size differs")
        identity.write_text(json.dumps(record, indent=2) + "\n")
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if not visible or not all(x.isdigit() for x in visible):
            raise ValueError("Set explicit CUDA_VISIBLE_DEVICES for model evaluation")
        snapshot = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free", "--format=csv,noheader,nounits"], text=True)
        available = {r.split(",")[0].strip() for r in snapshot.splitlines()
                     if int(r.split(",")[1]) < 512 and int(r.split(",")[2]) >= 23000}
        if visible[0] not in available:
            raise ValueError("The first requested evaluation GPU is occupied")
        if policy:
            from official_knowledge.resource_guard import idle_gpus, available_memory
            idle, _ = idle_gpus()
            if visible[0] not in idle:
                raise ValueError("The first requested evaluation GPU is occupied")
            if available_memory() < policy["evaluation_memory_max_gib"] + policy["host_start_reserve_gib"]:
                raise RuntimeError("Resource safety guard: insufficient host memory")
    runtime_cache = Path(manifest.get("runtime_cache", "/local/rcs/zh2701/LLM-safety/cache/open-unlearning-official-v1"))
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-12.6"))
    if not (cuda_home / "bin/nvcc").is_file():
        cuda_home = Path("/usr/local/cuda")
    if not (cuda_home / "bin/nvcc").is_file():
        raise RuntimeError("The installed CUDA compiler could not be located")
    cuda_home = cuda_home.resolve()
    for name in ("triton", "torch-extensions"):
        (runtime_cache / name).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="8", MAX_JOBS="2", TRITON_CACHE_DIR=str(runtime_cache / "triton"),
               TORCH_EXTENSIONS_DIR=str(runtime_cache / "torch-extensions"), NCCL_IB_DISABLE="1",
               CUDA_HOME=str(cuda_home),
               PATH=str(Path(sys.executable).parent) + ":" + str(cuda_home / "bin") + ":" + os.environ.get("PATH", ""))
    if policy:
        from official_knowledge.resource_guard import scope_command
        kind = "training" if job["kind"] == "train" else "evaluation"
        unit = os.environ.get("ULP_SCOPE_UNIT", f"ulp-small-manual-{os.getpid()}.scope")
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        env.update(ULP_SCOPE_UNIT=unit, ULP_CUDA_MEMORY_FRACTION=str(policy["cuda_allocator_fraction"]),
                   ULP_HOST_MEMORY_MAX_BYTES=str(policy[kind + "_memory_max_gib"] * 1024**3),
                   ULP_HOST_START_RESERVE_BYTES=str(policy["host_start_reserve_gib"] * 1024**3),
                   ULP_RUNTIME_RESOURCE_RECEIPT=str(logs / (args.job + ".resource-limits.json")))
        command = scope_command(command, unit, policy, job["kind"] == "train")
    if job["kind"] == "eval":
        eval_dir = Path(cfg.paths.output_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_record = {"job": args.job, "config_sha256": digest, "model": str(model_path),
                       "data_readiness_sha256": manifest.get("data_readiness_sha256"),
                       "model_files": {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in model_path.glob("*.safetensors")}}
        adapter_path = cfg.model.get("lora", {}).get("adapter_path")
        if adapter_path:
            for path in Path(adapter_path).glob("adapter_*"):
                eval_record["model_files"][str(path)] = [path.stat().st_size, path.stat().st_mtime_ns]
        identity = eval_dir / "prepared-evaluation.json"
        if identity.exists() and json.loads(identity.read_text()) != eval_record:
            raise ValueError("Existing evaluation belongs to different model/configuration")
        identity.write_text(json.dumps(eval_record, indent=2) + "\n")
        env['ULP_EVALUATION_IDENTITY'] = str(identity)
    if args.background:
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        with (logs / (args.job + ".log")).open("a") as stream:
            process = subprocess.Popen(command, cwd=args.upstream, env=env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        (logs / (args.job + ".pid")).write_text(str(process.pid) + "\n")
        print(f"Detached process {process.pid}; log: {logs / (args.job + '.log')}")
    else:
        subprocess.run(command, cwd=args.upstream, env=env, check=True)


if __name__ == "__main__":
    main()
