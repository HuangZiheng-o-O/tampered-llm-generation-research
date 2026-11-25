"""Exercise official LoRA trainers, checkpoint recovery, and saved-adapter evaluation.

Uses two optimizer steps per method in an isolated suite. These artifacts are
pipeline checks and are never counted as qualified unlearning models.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

from omegaconf import OmegaConf

from official_knowledge.run_pilot import Pilot, now, sha


class PipelineCheck(Pilot):
    def report(self):
        (self.root / "RUN_STATUS.md").write_text(
            "# LoRA 小规模流程检查\n\n"
            "每项使用官方训练器运行两次参数更新，并从第一次更新的检查点恢复。\n\n"
            f"状态：{self.state['phase']}；任务：{self.state.get('current_job')}。\n\n"
            "完整结果见 RESULT.json。此目录仅用于流程检查。\n"
        )

    def completed(self, name):
        job = self.manifest["jobs"][name]
        if job["kind"] != "train":
            return super().completed(name)
        out = Path(job["output"])
        required = ("trainer_state.json", "adapter_config.json", "adapter_model.safetensors", "prepared-job.json")
        if not all((out / n).is_file() for n in required):
            return False
        state = json.loads((out / "trainer_state.json").read_text())
        identity = json.loads((out / "prepared-job.json").read_text())
        assert identity["config_sha256"] == self.manifest["config_sha256"][name + ".yaml"]
        losses = [row["loss"] for row in state["log_history"] if "loss" in row]
        assert losses and all(math.isfinite(value) for value in losses), "Non-finite or absent training losses"
        return state["global_step"] == 2 and state["max_steps"] == 2


def prepare(original, upstream, destination):
    if destination.exists():
        raise ValueError("Use a new isolated check directory; existing artifacts are preserved")
    destination.mkdir()
    prior = json.loads((original / "configs/jobs.json").read_text())
    for directory in ("data", "tasks"):
        shutil.copytree(original / directory, destination / directory)
    shutil.copy2(original / "data-readiness.json", destination / "data-readiness.json")
    for path in (destination / "tasks").glob("*.yaml"):
        path.write_text(path.read_text().replace(str(original), str(destination)))
    configs = destination / "configs"
    configs.mkdir()
    jobs = {}
    for method in ("NPO", "GradDiff"):
        parent_name = "biographies_development_" + method
        for suffix in ("", "_resume"):
            name = "biographies_smoke_" + method + suffix
            cfg = OmegaConf.load(original / "configs" / (parent_name + ".yaml"))
            parent = cfg.model.model_args.pretrained_model_name_or_path
            cfg = OmegaConf.create(OmegaConf.to_yaml(cfg).replace(str(original), str(destination)))
            cfg.model.model_args.pretrained_model_name_or_path = parent
            out = str(destination / "runs" / name)
            cfg.task_name, cfg.paths.output_dir, cfg.hydra.run.dir = name, out, out
            cfg.trainer.args.update(output_dir=out, logging_dir=out + "/logs", max_steps=2,
                                    save_steps=1, save_total_limit=None)
            OmegaConf.save(cfg, configs / (name + ".yaml"))
            jobs[name] = {"entry": "src/train.py", "kind": "train", "output": out, "requires": []}
        name = "biographies_smoke_" + method + "_knowledge"
        cfg = OmegaConf.load(original / "configs" / (parent_name + "_knowledge.yaml"))
        parent = cfg.model.model_args.pretrained_model_name_or_path
        cfg = OmegaConf.create(OmegaConf.to_yaml(cfg).replace(str(original), str(destination)))
        cfg.model.model_args.pretrained_model_name_or_path = parent
        cfg.model.lora.adapter_path = jobs["biographies_smoke_" + method + "_resume"]["output"]
        out = str(destination / "evals" / name)
        cfg.task_name, cfg.paths.output_dir, cfg.hydra.run.dir, cfg.eval.lm_eval.output_dir = name, out, out, out
        OmegaConf.save(cfg, configs / (name + ".yaml"))
        jobs[name] = {"entry": "src/eval.py", "kind": "eval", "output": out, "requires": [],
                      "model": parent, "source": "biographies", "stage": "smoke_" + method, "assessment": "knowledge"}
    manifest = copy.deepcopy(prior)
    manifest.update(suite=destination.name, scope="real_model_pipeline_check_only", methods=["NPO", "GradDiff"],
                    jobs=jobs, config_sha256={p.name: sha(p) for p in configs.glob("*.yaml")},
                    task_sha256={str(p.relative_to(destination)): sha(p) for p in (destination / "tasks").glob("*.yaml")})
    manifest.pop("gpu_pool", None)
    (configs / "jobs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "CHECK_SCOPE.json").write_text(json.dumps({
        "prepared_at_utc": now(), "source_suite": str(original), "official_commit": manifest["upstream_commit"],
        "proof_kind": "real_model_pipeline_check_only", "optimizer_steps": 2,
        "unlearning_quality_qualified": False, "formal_batch_started": False,
        "official_sources_changed_by_this_check": False,
        "learning_gate": "not used for a two-step software check; full-pilot quality gates remain enabled",
    }, indent=2) + "\n")


def tensor_check(first, resumed):
    import numpy as np
    from safetensors import safe_open

    maximum, changed_b, count = 0.0, 0, 0
    with safe_open(first, framework="numpy") as a, safe_open(resumed, framework="numpy") as b:
        assert set(a.keys()) == set(b.keys())
        for name in a.keys():
            x, y = a.get_tensor(name), b.get_tensor(name)
            assert np.isfinite(x).all() and np.isfinite(y).all(), "Non-finite saved adapter"
            maximum = max(maximum, float(np.abs(x - y).max()))
            assert np.allclose(x, y, atol=1e-6, rtol=1e-5), (name, maximum)
            changed_b += int("lora_B" in name and np.count_nonzero(x) > 0)
            count += x.size
    assert changed_b > 0, "No task update was saved"
    return {"compared_parameter_count": count, "nonzero_lora_b_tensors": changed_b,
            "maximum_abs_difference": maximum, "atol": 1e-6, "rtol": 1e-5,
            "resumed_step_two_matches_uninterrupted_step_two": True}


def execute(root, upstream):
    import fcntl

    with (root / "pipeline-check.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pilot = PipelineCheck(root, upstream, 12)
        result = {"proof_kind": "real_model_pipeline_check_only", "phase": "running", "methods": {},
                  "formal_batch_started": False, "started_at_utc": now()}
        result_path = root / "RESULT.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        try:
            for method in ("NPO", "GradDiff"):
                name = "biographies_smoke_" + method
                pilot.run(name)
                before = Path(pilot.manifest["jobs"][name]["output"])
                resumed = Path(pilot.manifest["jobs"][name + "_resume"]["output"])
                resumed.mkdir(exist_ok=True)
                if not (resumed / "checkpoint-1").exists():
                    shutil.copytree(before / "checkpoint-1", resumed / "checkpoint-1")
                    (resumed / "prepared-job.json").write_text(json.dumps({
                        "config_sha256": pilot.manifest["config_sha256"][name + "_resume.yaml"],
                        "world_size": 1, "upstream_commit": pilot.manifest["upstream_commit"]}, indent=2) + "\n")
                pilot.run(name + "_resume")
                comparison = tensor_check(before / "adapter_model.safetensors", resumed / "adapter_model.safetensors")
                pilot.run(name + "_knowledge")
                evaluation = root / "evals" / (name + "_knowledge")
                result["methods"][method] = {"passed": True, "completed_optimizer_steps": 2, "resumed_from_step": 1,
                    "weight_comparison": comparison,
                    "adapter_bytes": (resumed / "adapter_model.safetensors").stat().st_size,
                    "adapter_sha256": sha(resumed / "adapter_model.safetensors"),
                    "official_knowledge_evaluation_questions": 358,
                    "evaluation_summary": json.loads((evaluation / "LMEval_SUMMARY.json").read_text()),
                    "evidence_sha256": {str(p.relative_to(root)): sha(p) for p in (
                        before / "trainer_state.json", resumed / "trainer_state.json",
                        evaluation / "prepared-evaluation.json", evaluation / "LMEval_EVAL.json",
                        evaluation / "LMEval_SUMMARY.json")}}
                result_path.write_text(json.dumps(result, indent=2) + "\n")
            result.update(phase="passed", finished_at_utc=now())
            pilot.record(phase="small_pilot_finished", current_job=None)
        except Exception as error:
            result.update(phase="failed", error=repr(error), finished_at_utc=now())
            pilot.record(phase="failed", error=repr(error))
            raise
        finally:
            result_path.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-suite", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        assert args.source_suite
        prepare(args.source_suite.resolve(), args.upstream.resolve(), args.root.resolve())
    else:
        execute(args.root.resolve(), args.upstream.resolve())


if __name__ == "__main__":
    main()
