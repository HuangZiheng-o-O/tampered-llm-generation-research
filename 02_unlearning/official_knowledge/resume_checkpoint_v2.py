"""One-off handoff: enable native saves only after the save/resume fixture passes."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    root, upstream = args.suite.resolve(), args.upstream.resolve()
    support = Path(__file__).resolve().parents[1]
    evidence = root / "cache/checkpoint-io-v2-software-check/RESULT.json"
    deadline = time.monotonic() + 3 * 3600
    while True:
        if evidence.exists():
            result = json.loads(evidence.read_text())
            if result["phase"] == "failed":
                raise RuntimeError("Save/resume fixture failed; 7B training remains stopped")
            if result["phase"] == "passed":
                assert result["proof_kind"] == "software_fixture_only" and result["completed_optimizer_steps"] == 2
                break
        if time.monotonic() > deadline:
            raise TimeoutError("Save/resume verification is still pending; 7B training remains stopped")
        time.sleep(15)
    with (root / "pilot-worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = root / "configs/jobs.json"
        manifest = json.loads(manifest_path.read_text())
        marker = root / "CHECKPOINT_IO_V2_ENABLED.json"
        if not marker.exists():
            archive = root / "receipts/checkpoint-io-v2"
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest_path, archive / "prior-jobs.json")
            old_output = str(root / "runs/biographies_learn")
            new_output = str(root / "runs/biographies_learn_native_v2")
            assert not Path(new_output).exists()
            for path in (root / "configs").glob("*.yaml"):
                shutil.copy2(path, archive / path.name)
                job = manifest["jobs"][path.stem]
                if job["kind"] == "eval" and old_output not in path.read_text():
                    continue
                text = path.read_text().replace(old_output, new_output)
                cfg = OmegaConf.create(text)
                if job["kind"] == "train":
                    OmegaConf.set_struct(cfg, False)
                    cfg.native_checkpoint_io = True
                    cfg.checkpoint_first_step = path.stem.endswith("_learn")
                    job["output"] = str(cfg.paths.output_dir)
                if "model" in job:
                    job["model"] = job["model"].replace(old_output, new_output)
                OmegaConf.save(cfg, path)
            ds_path = root / "configs/zero3_cpu.json"
            shutil.copy2(ds_path, archive / "prior-zero3_cpu.json")
            ds = json.loads(ds_path.read_text())
            ds["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = False
            ds_path.write_text(json.dumps(ds, indent=2) + "\n")
            manifest["deepspeed_config_sha256"] = hashlib.sha256(ds_path.read_bytes()).hexdigest()
            manifest["config_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "configs").glob("*.yaml")}
            manifest["source_sha256"] = {str(p.relative_to(upstream)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (upstream / "src").rglob("*.py")}
            manifest["support_sha256"] = {str(p.relative_to(support)): hashlib.sha256(p.read_bytes()).hexdigest() for pkg in ("official_knowledge", "official_pilot") for p in (support / pkg).glob("*.py")}
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            marker.write_text(json.dumps({"enabled_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "software_check": result, "old_incomplete_checkpoint_preserved": True,
                "new_biography_output": new_output, "restart_from_original_weights": True,
                "model_gather_during_training": False, "offline_export": "bounded-bf16-shards",
                "hardware_memory_limits_increased": False}, indent=2) + "\n")
        fcntl.flock(lock, fcntl.LOCK_UN)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               PYTHONPATH=str(support) + ":" + str(upstream / "src"), TOKENIZERS_PARALLELISM="false")
    with (root / "logs/pilot-worker.log").open("a") as stream:
        process = subprocess.Popen([sys.executable, "-m", "official_knowledge.run_pilot", "--suite-root", str(root),
            "--upstream", str(upstream), "--max-wait-hours", "72"], cwd=support, env=env,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    (root / "logs/pilot-worker.pid").write_text(str(process.pid) + "\n")
    print("Native-checkpoint pilot coordinator started:", process.pid, flush=True)


if __name__ == "__main__":
    main()
