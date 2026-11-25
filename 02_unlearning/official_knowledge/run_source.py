"""One independently resumable source/base pilot using official train/eval jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path

from official_knowledge.run_pilot import Pilot, now
from official_knowledge.pilot_recipes import stages, repaired, appropriate_follow_up, selected_control, selected


class SourcePilot(Pilot):
    def __init__(self, root, upstream, wait_hours, source):
        super().__init__(root, upstream, wait_hours)
        self.source = source
        self.state.update(source=source, coordinator_pid=os.getpid())

    def record(self, **values):
        self.state.update(values, updated_at_utc=now())
        path = self.root / ("RUN_STATUS-" + self.source + ".json")
        pending = path.with_suffix(".pending.json")
        pending.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n")
        pending.replace(path)
        self.report()
        print(json.dumps({"time": now(), **values}, ensure_ascii=False), flush=True)

    def report(self):
        source = self.source
        info = self.data["sources"][source]
        label = "人物" if source == "biographies" else "文章"
        lines = [f"# {label}材料的小规模训练和评估", "", f"更新时间：{now()}", "",
                 f"状态：{self.state['phase']}；当前任务：{self.state.get('current_job')}。", "",
                 f"学习材料：{info['learning_pool_units']} 个对象，{info['learning_rows']} 条记录。",
                 "每个训练任务调用官方 OpenUnlearning 训练器。新任务只保存最终适配器及必要的配置、状态和评估。", "",
                 "| 对象 | 方法 | 原先会的目标知识仍能回答的比例 | 其他原有知识保留比例 | 一般答题正确率 | 验收 |",
                 "|---|---|---:|---:|---:|---|"]
        for path in sorted((self.root / "gates").glob(source + "-*-quality.json")):
            try:
                gate = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            stage = path.name.removeprefix(source + "-").removesuffix("-quality.json")
            phase = stage.split("_")[0]
            title = info["targets"][phase]["title"]
            method = "普通继续训练对照" if "_control_" in stage else stage.split("_", 1)[1]
            lines.append(f"| {title} | {method} | {gate['target_previously_correct_fraction_remaining']:.1%} | "
                         f"{gate['retain_previously_correct_fraction_remaining']:.1%} | {gate['mmlu_after']:.1%} | "
                         f"{'通过' if gate['passed'] else '未通过'} |")
        if self.state.get("error"):
            lines += ["", "错误：" + self.state["error"]]
        (self.root / ("RUN_STATUS-" + source + ".md")).write_text("\n".join(lines) + "\n")

    def run(self, name):
        locks = self.root / "job-locks"
        locks.mkdir(exist_ok=True)
        with (locks / (name + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.manifest = json.loads((self.root / 'configs/jobs.json').read_text())
            return super().run(name)

    def qualification(self, source, stage=None):
        self.manifest = json.loads((self.root / 'configs/jobs.json').read_text())
        return super().qualification(source, stage)

    def execute(self):
        source = self.source
        methods = self.manifest["methods"]
        self.evaluate(source, "base")
        self.run(source + "_learn")
        self.evaluate(source, "learn")
        if self.manifest.get("verify_lora_merge_behavior"):
            self.evaluate(source, "learn_adapter")
        learning = self.qualification(source)
        if not all(learning['targets'][phase]['passed'] for phase in ('development', 'confirmation')):
            from official_knowledge.learning_retry import prepare_longer_article_learning
            changed = prepare_longer_article_learning(self)
            self.record(phase='learning_retry_prepared' if changed else 'needs_research',
                        current_job=None, qualified_methods=[], unqualified_methods=list(methods), formal_batch_started=False)
            return
        prior_selections = selected(self.root, source)
        selections = {}
        for phase in ("development", "confirmation"):
            if not learning["targets"][phase]["passed"]:
                continue
            control = selected_control(self.root, source, phase)
            self.run(source + '_' + control)
            self.evaluate(source, control)
            if not self.qualification(source, control)['passed']:
                continue
            for method in methods:
                repair = repaired(self.root, source, method)
                if repair:
                    selections[method] = dict(repair, development_passed=True, confirmation_passed=False)
                candidates = [repair[phase + '_stage']] if repair else (stages(phase, method) if phase == 'development' else (
                    [selections[method]['confirmation_stage']] if method in selections else []))
                if phase == 'development' and not repair:
                    prior = prior_selections.get(method, {})
                    prior_development, prior_confirmation = prior.get('development_stage'), prior.get('confirmation_stage')
                    if prior_development in candidates and prior_confirmation and all(
                            self.completed(source + '_' + prior_confirmation + tail)
                            for tail in ('_knowledge', '_mmlu_validation')):
                        if self.qualification(source, prior_confirmation)['passed']:
                            candidates = [prior_development] + [c for c in candidates if c != prior_development]
                initial_report = None
                for stage in candidates:
                    if phase == 'development' and not repair and not appropriate_follow_up(
                            method, stage, initial_report, self.manifest.get('quality_policy', {})):
                        continue
                    if source + '_' + stage not in self.manifest['jobs']:
                        continue
                    self.run(source + '_' + stage)
                    self.evaluate(source, stage)
                    report = self.qualification(source, stage)
                    if stage == 'development_' + method and not report['passed']:
                        initial_report = report
                    if phase == 'development' and report['passed']:
                        selections[method] = {**(repair or {}), 'development_stage': stage,
                            'confirmation_stage': stage.replace('development_', 'confirmation_', 1),
                            'development_passed': True, 'confirmation_passed': False}
                    elif phase == 'confirmation':
                        selections[method]['confirmation_passed'] = report['passed']
                    path = self.root / ('SELECTED_RECIPES-' + source + '.json')
                    pending = path.with_suffix('.pending.json')
                    pending.write_text(json.dumps(selections, indent=2) + '\n')
                    pending.replace(path)
                    if report['passed']:
                        break
        qualified = [m for m in methods if selections.get(m, {}).get('confirmation_passed')]
        self.record(phase="pilot_source_passed" if len(qualified) == len(methods) else "needs_research",
                    current_job=None, qualified_methods=qualified,
                    unqualified_methods=[m for m in methods if m not in qualified], formal_batch_started=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite-root", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--source", choices=("biographies", "squad"), required=True)
    args = p.parse_args()
    with (args.suite_root / ("pilot-worker-" + args.source + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pilot = SourcePilot(args.suite_root, args.upstream, 72, args.source)
        try:
            pilot.execute()
        except Exception as error:
            pilot.record(phase="needs_repair", error=repr(error))
            raise


if __name__ == "__main__":
    main()
