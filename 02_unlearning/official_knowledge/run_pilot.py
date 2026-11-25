"""Run the prepared small pilot when GPUs are idle; call official entry points.

This is a single detached experiment coordinator, not a recurring automation.
It never generates the formal 400-model dataset, deletes files, or stops other
processes. Quality failures are recorded and excluded from confirmation runs.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from official_pilot.launch import checkpoint
from official_knowledge.resource_guard import POLICY, idle_gpus, available_memory, memory_pressure, monitor


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Pilot:
    def __init__(self, root, upstream, wait_hours):
        self.root, self.upstream = root.resolve(), upstream.resolve()
        self.support = Path(__file__).resolve().parents[1]
        self.manifest = json.loads((root / "configs/jobs.json").read_text())
        self.data = json.loads((root / "data-readiness.json").read_text())
        self.policy = self.manifest.get("resource_policy", POLICY)
        assert root.stat().st_uid == os.getuid()
        assert self.manifest["training_enabled"]
        assert sha(root / "data-readiness.json") == self.manifest["data_readiness_sha256"]
        self.logs = root / "logs"
        self.logs.mkdir(exist_ok=True)
        self.deadline = time.monotonic() + wait_hours * 3600
        self.state = {"started_at_utc": now(), "scope": "prepared_small_pilot_only", "phase": "starting",
                      "completed_jobs": [], "failed_jobs": [], "quality": {}, "formal_batch_started": False}

    def record(self, **values):
        self.state.update(values, updated_at_utc=now())
        path = self.root / "RUN_STATUS.json"
        temporary = path.with_suffix(".pending.json")
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
        self.report()
        print(json.dumps({"time": now(), **values}, ensure_ascii=False), flush=True)

    def report(self):
        phases = {"starting": "正在启动", "waiting_for_resources": "等待空闲 GPU 或内存", "running": "正在运行官方程序",
                  "job_completed": "一个任务已完成", "quality_checked": "已检查这一项的学习或遗忘效果",
                  "small_pilot_finished": "本轮小试已结束", "failed": "任务报错，详见日志",
                  "exporting_checkpoint": "训练已结束，正在分片导出供评估的模型"}
        current = self.state.get("current_job")
        description = str(current or "尚未分配")
        if current in self.manifest["jobs"]:
            source = current.split("_")[0]
            label = "人物" if source == "biographies" else "文章"
            job = self.manifest["jobs"][current]
            if current == source + "_learn":
                description = "让 Chat 模型学习全部" + label + "材料"
            elif job["kind"] == "eval":
                stage = "原始 Chat 模型" if job["stage"] == "base" else "学习后模型" if job["stage"] == "learn" else "小试产生的模型"
                description = "检查" + stage + "的" + (label + "知识与保留知识" if job["assessment"] == "knowledge" else "一般答题能力")
            else:
                description = label + ("正常继续训练对照" if "_control_" in current else "指定对象的遗忘训练")
        lines = ["# 小规模训练与验证进度", "", f"更新时间：{now()}", "",
                 "当前状态：" + phases.get(self.state["phase"], self.state["phase"]),
                 "当前任务：" + description + "（`" + str(current) + "`）",
                 f"已完成官方任务：{len(self.state['completed_jobs'])}。正式批量生成：尚未启动。", "",
                 "本轮前置学习：120 个人物、14,400 条句子；40 篇文章、3,349 条正文和问答。", "",
                 "以下成绩仅从本目录真实模型的完整验收收据读取。知识正确率按人物属性或文章短答案计算；答案 F1 按词重合计算。", ""]
        learning_rows, effect_rows = [], []
        for source in ("biographies", "squad"):
            label = "人物" if source == "biographies" else "文章"
            learning = self.root / "gates" / (source + "-learning.json")
            if learning.exists():
                gate = json.loads(learning.read_text())
                if gate.get("proof_kind") != "official_model_evaluation":
                    continue
                for phase, scores in gate["targets"].items():
                    info = self.data["sources"][source]["targets"][phase]
                    learning_rows.append(f"| {label}：{info['title']}（{info['probe_rows']} 条检查） | {scores['base']['knowledge_accuracy']:.1%} | {scores['learned']['knowledge_accuracy']:.1%} | {scores['learned']['answer_f1']:.3f} | {'通过' if scores['passed'] else '未通过'} |")
            for path in sorted((self.root / "gates").glob(source + "-*-quality.json")):
                gate = json.loads(path.read_text())
                if gate.get("proof_kind") != "official_model_evaluation":
                    continue
                stage = path.name.removeprefix(source + "-").removesuffix("-quality.json")
                phase = stage.split("_")[0]
                info = self.data["sources"][source]["targets"][phase]
                score = gate["after"][phase]
                kind = "正常继续训练对照" if "_control_" in stage else "遗忘：" + stage.split("_")[-1]
                effect_rows.append(f"| {label}：{info['title']} | {kind} | {score['knowledge_accuracy']:.1%} | {score['answer_f1']:.3f} | {gate['after']['retain']['knowledge_accuracy']:.1%} | {gate['mmlu_after']:.1%} | {'通过' if gate['passed'] else '未通过'} |")
        if learning_rows:
            lines += ["## 是否先学会", "", "| 检查对象 | 原始模型正确率 | 学习后正确率 | 学习后答案 F1 | 验收 |", "|---|---:|---:|---:|---|"] + learning_rows + [""]
        if effect_rows:
            lines += ["## 遗忘与正常对照", "", "| 检查对象 | 运行内容 | 目标正确率 | 目标答案 F1 | 保留知识正确率 | MMLU 正确率 | 验收 |", "|---|---|---:|---:|---:|---:|---|"] + effect_rows + [""]
        if not learning_rows and not effect_rows:
            lines += ["结果：等待真实模型训练与评估完成。", ""]
        if self.state.get("error"):
            lines += ["错误记录：`" + self.state["error"] + "`", ""]
        (self.root / "RUN_STATUS.md").write_text("\n".join(lines) + "\n")

    def configuration(self, name):
        path = self.root / "configs" / (name + ".yaml")
        assert sha(path) == self.manifest["config_sha256"][path.name]
        return OmegaConf.load(path)

    def completed(self, name):
        job, cfg = self.manifest["jobs"][name], self.configuration(name)
        out = Path(cfg.paths.output_dir)
        if job["kind"] == "train":
            state = out / "trainer_state.json"
            lora = bool(cfg.model.get("lora"))
            model_config = out / ("adapter_config.json" if lora else "config.json")
            if not state.exists() or not model_config.exists():
                return False
            weights = list(out.glob("adapter_model.safetensors" if lora else "*.safetensors"))
            minimum = 1_000 if lora else 12_000_000_000
            if not weights or sum(p.stat().st_size for p in weights) < minimum:
                return False
            if lora and name.endswith("_learn") and not (out / "merged/lora-parent.json").exists():
                return False
            identity = json.loads((out / "prepared-job.json").read_text())
            if identity["config_sha256"] != self.manifest["config_sha256"][name + ".yaml"]:
                raise ValueError("Existing model has a different configuration")
            rows = job.get("primary_rows")
            if rows is None:
                rows = self.data["sources"][job.get("source", name.split("_")[0])]["learning_rows"]
            batch = cfg.trainer.args.per_device_train_batch_size * cfg.trainer.args.gradient_accumulation_steps * self.manifest["world_size"]
            max_steps = cfg.trainer.args.get("max_steps", -1)
            expected = max_steps if max_steps > 0 else math.ceil(rows / batch) * cfg.trainer.args.num_train_epochs
            return json.loads(state.read_text())["global_step"] >= expected
        marker, logs, summary = (out / n for n in ("prepared-evaluation.json", "LMEval_EVAL.json", "LMEval_SUMMARY.json"))
        if not all(p.exists() for p in (marker, logs, summary)):
            return False
        identity = json.loads(marker.read_text())
        if identity["config_sha256"] != self.manifest["config_sha256"][name + ".yaml"]:
            raise ValueError("Existing evaluation has a different configuration")
        for path, state in identity["model_files"].items():
            if [Path(path).stat().st_size, Path(path).stat().st_mtime_ns] != state:
                raise ValueError("Evaluated model changed")
        results = json.loads(logs.read_text())
        if job["assessment"] == "knowledge":
            for phase, count in self.data["sources"][job["source"]]["eval_counts"].items():
                task = f"knowledge_{job['source']}_{phase}"
                if len(results.get(task, {}).get(task, [])) != count:
                    return False
            return True
        split = job["assessment"].removeprefix("mmlu_")
        group = results.get(f"pilot_mmlu_{split}", {})
        return sum(len(rows) for rows in group.values()) == (1531 if split == "validation" else 14042)

    def resources(self, name):
        for handle in getattr(self, "gpu_reservations", []):
            handle.close()
        self.gpu_reservations = []
        training = self.manifest["jobs"][name]["kind"] == "train"
        needed = self.manifest["world_size"] if training else 1
        while True:
            idle, snapshot = idle_gpus()
            if self.manifest.get("gpu_pool"):
                idle = [gpu for gpu in idle if gpu in self.manifest["gpu_pool"]]
            memory = available_memory()
            pressure = memory_pressure()
            disk = __import__("shutil").disk_usage(self.root).free / 1024**3
            minimum = self.policy["training_memory_max_gib" if training else "evaluation_memory_max_gib"] + self.policy["host_start_reserve_gib"]
            if len(idle) >= needed and memory >= minimum and disk >= 250 and pressure <= self.policy["host_start_pressure_max_percent"]:
                reservation_directory = self.manifest.get("gpu_reservation_directory")
                if not reservation_directory:
                    return idle[:needed]
                from official_knowledge.resource_guard import reserve_gpus
                selected, self.gpu_reservations = reserve_gpus(idle, needed, reservation_directory)
                if selected:
                    return selected
            self.record(phase="waiting_for_resources", current_job=name, needed_idle_gpus=needed,
                        gpu_snapshot=snapshot, available_memory_gib=round(memory, 1), free_disk_gib=round(disk, 1), host_memory_pressure_percent=pressure)
            if time.monotonic() >= self.deadline:
                raise TimeoutError("Resource waiting limit reached; no other processes were changed")
            time.sleep(5)

    def resume(self, name):
        job = self.manifest["jobs"][name]
        if job["kind"] != "train":
            return None
        out = Path(job["output"])
        candidates = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]), reverse=True)
        for path in candidates:
            try:
                return checkpoint(path, out)
            except (ValueError, KeyError, FileNotFoundError):
                continue
        if candidates or (out / "trainer_state.json").exists():
            raise ValueError("Existing interrupted output has no complete resumable checkpoint")
        return None

    def run(self, name):
        job = self.manifest["jobs"][name]
        if job["kind"] == "train":
            output = Path(job["output"])
            if (output / "trainer_state.json").exists() and (output / "adapter_model.safetensors").exists():
                self.export_native(name)
            if (output / "trainer_state.json").exists() and (output / "latest").exists() and not (output / "native-export.json").exists():
                self.export_native(name)
        if self.completed(name):
            if name not in self.state["completed_jobs"]:
                self.state["completed_jobs"].append(name)
            return
        retried_checkpoint = False
        while True:
            gpus = self.resources(name)
            command = [sys.executable, "-m", "official_pilot.launch", name, "--root", str(self.root.parent),
                       "--suite", self.root.name, "--upstream", str(self.upstream), "--execute"]
            resume = self.resume(name)
            if resume:
                command += ["--resume", str(resume)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus), HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                       TOKENIZERS_PARALLELISM="false", PYTHONPATH=str(self.support) + ":" + str(self.upstream / "src"))
            unit = f"ulp-small-{os.getpid()}-{hashlib.sha256(name.encode()).hexdigest()[:10]}-{time.time_ns()}.scope"
            env["ULP_SCOPE_UNIT"] = unit
            self.record(phase="running", current_job=name, gpus=gpus, resource_scope=unit, resumed_checkpoint=str(resume) if resume else None)
            logfile = self.logs / (name + ".log")
            with logfile.open("a") as stream:
                result = subprocess.Popen(command, env=env, cwd=self.support, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    safety_reason = monitor(result, unit, gpus, self.policy, self.logs / (name + ".resource-monitor.json"), logfile)
                except Exception:
                    subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True)
                    if result.poll() is None:
                        result.terminate()
                    result.wait(timeout=30)
                    raise
            if not result.returncode and self.manifest["jobs"][name]["kind"] == "train":
                self.export_native(name)
            if not result.returncode and self.completed(name):
                break
            if safety_reason:
                if safety_reason in ("own_memory_pressure_stall", "no_training_progress_timeout"):
                    self.state["failed_jobs"].append({"job": name, "reason": safety_reason, "log": str(logfile)})
                    raise RuntimeError(f"Own experiment stopped after a resource/progress stall: {name}: {safety_reason}")
                self.record(phase="waiting_for_resources", current_job=name, reason=safety_reason)
                time.sleep(60)
                continue
            tail = logfile.read_text(errors="replace")[-8000:]
            if any(text in tail for text in ("Requested GPUs are occupied", "evaluation GPU is occupied", "GiB available RAM", "Resource safety guard")):
                self.record(phase="waiting_for_resources", current_job=name, reason="resource_changed_before_launch")
                time.sleep(60)
                continue
            if not retried_checkpoint and self.manifest["jobs"][name]["kind"] == "train" and self.resume(name):
                retried_checkpoint = True
                self.record(phase="waiting_for_resources", current_job=name, reason="retry_from_complete_checkpoint_once")
                continue
            self.state["failed_jobs"].append({"job": name, "returncode": result.returncode, "log": str(logfile)})
            raise RuntimeError(f"Official job failed or produced incomplete outputs: {name}")
        self.state["completed_jobs"].append(name)
        self.record(phase="job_completed", current_job=name)

    def export_native(self, name):
        output = Path(self.manifest["jobs"][name]["output"])
        cfg = self.configuration(name)
        if cfg.model.get("lora"):
            if name.endswith("_learn") and not (output / "merged/lora-parent.json").exists():
                from official_pilot.lora import merge_parent
                merge_parent(self, name)
            return
        if not (output / "latest").exists():
            return
        from official_knowledge.resource_guard import scope_command
        unit = f"ulp-small-{os.getpid()}-export-{time.time_ns()}.scope"
        limit = self.policy["evaluation_memory_max_gib"] * 1024**3
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(self.support) + ":" + str(self.upstream / "src"),
                   ULP_SCOPE_UNIT=unit, ULP_HOST_MEMORY_MAX_BYTES=str(limit))
        command = scope_command([sys.executable, "-m", "official_knowledge.export_checkpoint", "--checkpoint", str(output)],
                                unit, self.policy, False)
        self.record(phase="exporting_checkpoint", current_job=name, resource_scope=unit, gpus=[])
        log = self.logs / (name + ".export.log")
        with log.open("a") as stream:
            process = subprocess.Popen(command, cwd=self.support, env=env, stdout=stream, stderr=subprocess.STDOUT)
            reason = monitor(process, unit, [], self.policy, self.logs / (name + ".export-resource-monitor.json"), log)
        if process.returncode or reason:
            raise RuntimeError("Native checkpoint export needs attention: " + name)

    def evaluate(self, source, stage):
        for assessment in ("knowledge", "mmlu_validation"):
            self.run(f"{source}_{stage}_{assessment}")

    def qualification(self, source, stage=None):
        from official_knowledge.qualify import assess
        report = assess(self.root, source, stage)
        self.state["quality"][source + "/" + (stage or "learning")] = {
            "global_checks_passed": report.get("global_checks_passed"), "passed": report.get("passed"),
            "targets": {k: {"passed": v["passed"], "learning_gain": v["learning_gain"]} for k, v in report.get("targets", {}).items()}}
        self.record(phase="quality_checked", current_job=f"{source}/{stage or 'learning'}")
        return report

    def execute(self):
        methods = self.manifest.get("methods", ("NPO", "SimNPO", "GradDiff", "RMU", "UNDIAL"))
        for source in ("biographies", "squad"):
            self.evaluate(source, "base")
            self.run(source + "_learn")
            self.evaluate(source, "learn")
            if self.manifest.get("verify_lora_merge_behavior"):
                self.evaluate(source, "learn_adapter")
            learning = self.qualification(source)
            for phase in ("development", "confirmation"):
                if not learning["targets"][phase]["passed"]:
                    continue
                candidates = list(methods) if phase == "development" else [
                    method for method in methods if self.state["quality"].get(source + "/development_" + method, {}).get("passed")]
                if not candidates:
                    continue
                controls = {}
                rates = {method: self.manifest["jobs"][f"{source}_{phase}_{method}"]["matching_control"].split("_control_", 1)[1] for method in candidates}
                for rate in sorted(set(rates.values())):
                    stage = phase + "_control_" + rate
                    self.run(source + "_" + stage)
                    self.evaluate(source, stage)
                    controls[rate] = self.qualification(source, stage)["passed"]
                for method in candidates:
                    rate = rates[method]
                    if not controls[rate]:
                        continue
                    stage = phase + "_" + method
                    self.run(source + "_" + stage)
                    self.evaluate(source, stage)
                    self.qualification(source, stage)
        qualified = {source: [method for method in methods if self.state["quality"].get(source + "/confirmation_" + method, {}).get("passed")]
                     for source in ("biographies", "squad")}
        self.record(phase="small_pilot_finished", current_job=None, qualified_methods=qualified, formal_batch_started=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite-root", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--max-wait-hours", type=float, default=72)
    args = p.parse_args()
    lock = (args.suite_root / "pilot-worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pilot = Pilot(args.suite_root, args.upstream, args.max_wait_hours)
    try:
        pilot.execute()
    except Exception as error:
        pilot.record(phase="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
