"""Exercise official evaluation and quality receipts with a response fixture.

All outputs stay in a separately labelled temporary directory. This is a
software integration test, not a language-model result. No weights are loaded.
"""
import argparse
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite-root", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.upstream / "src"))
    from evals.lm_eval import LMEvalEvaluator
    from lm_eval.api.model import LM
    from official_knowledge.knowledge_metrics import process_results
    from official_knowledge.qualify import assess
    from official_pilot import launch

    assert process_results({"answers": ["New York"], "source": "biographies"}, ["New Yorkshire"]) ["knowledge_accuracy"] == 0
    scores = process_results({"answers": ["Guangzhou"], "source": "biographies"}, ["Guangzhou is their home."])
    assert scores["knowledge_accuracy"] == 1 and scores["exact_match"] == 0
    assert process_results({"answers": ["eight", "8"], "source": "squad"}, ["Eight."])["exact_match"] == 1

    manifest = json.loads((args.suite_root / "configs/jobs.json").read_text())
    fixture = Path(tempfile.mkdtemp(prefix="knowledge-software-fixture-", dir=args.suite_root / "cache"))
    (fixture / "PROOF_KIND.txt").write_text("SOFTWARE FIXTURE ONLY. No language model or optimizer was used.\n")
    (fixture / "configs").mkdir()
    (fixture / "configs/jobs.json").write_text(json.dumps(manifest))
    (fixture / "data").symlink_to(args.suite_root / "data", target_is_directory=True)
    (fixture / "data-readiness.json").write_bytes((args.suite_root / "data-readiness.json").read_bytes())
    answer_map = {}
    for path in (args.suite_root / "data").glob("*/eval-*.jsonl"):
        for line in path.read_text().splitlines():
            doc = json.loads(line)
            answer_map[doc["prompt"]] = doc["answers"][0]

    class ResponseFixture(LM):
        def __init__(self, mode, forgotten=()):
            super().__init__()
            self.mode, self.forgotten = mode, set(forgotten)

        def generate_until(self, requests, disable_tqdm=False):
            return [answer_map.get(r.args[0], "unknown") if self.mode != "base" and r.args[0] not in self.forgotten else "unknown" for r in requests]

        def loglikelihood(self, requests, disable_tqdm=False):
            return [(-float(int(hashlib.sha256(r.args[1].encode()).hexdigest()[:5], 16)), False) for r in requests]

        def loglikelihood_rolling(self, requests, disable_tqdm=False):
            raise AssertionError("This pilot does not request rolling likelihood")

    source = "biographies"
    forgotten = [json.loads(l)["prompt"] for l in (args.suite_root / "data" / source / "eval-development.jsonl").read_text().splitlines()]
    for stage in ("base", "learn", "development_NPO", "development_control_1e-5"):
        response = ResponseFixture(stage, forgotten if stage == "development_NPO" else ())
        for kind in ("knowledge", "mmlu_validation"):
            name = f"{source}_{stage}_{kind}"
            cfg = OmegaConf.load(args.suite_root / "configs" / f"{name}.yaml")
            cfg.eval.lm_eval.output_dir = str(fixture / "evals" / name)
            cfg.eval.lm_eval.simple_evaluate_args.bootstrap_iters = 0
            evaluator = LMEvalEvaluator(cfg.eval.lm_eval)
            with patch.object(evaluator, "prepare_model", return_value=response):
                evaluator.evaluate(model=None)
            out = fixture / "evals" / name
            (out / "prepared-evaluation.json").write_text(json.dumps({
                "job": name, "config_sha256": manifest["config_sha256"][name + ".yaml"],
                "data_readiness_sha256": manifest["data_readiness_sha256"], "model_files": {},
                "model": manifest["jobs"][name]["model"], "proof_kind": "software_fixture_only"}))
    learned = assess(fixture, source)
    assert learned["global_checks_passed"] and learned["targets"]["development"]["passed"]
    assert assess(fixture, source, "development_NPO")["passed"]
    assert assess(fixture, source, "development_control_1e-5")["passed"]
    # A complete-looking log from another data/configuration must be rejected.
    marker = fixture / "evals/biographies_base_knowledge/prepared-evaluation.json"
    original = marker.read_text()
    bad = json.loads(original)
    bad["config_sha256"] = "wrong"
    marker.write_text(json.dumps(bad))
    try:
        assess(fixture, source)
    except AssertionError:
        pass
    else:
        raise AssertionError("Different-config evidence was accepted")
    marker.write_text(original)
    with patch("sys.argv", ["launch", "biographies_development_NPO", "--root", str(args.suite_root.parent),
                            "--suite", args.suite_root.name, "--upstream", str(args.upstream), "--execute"]), \
         patch.object(launch.subprocess, "run") as run, patch.object(launch.subprocess, "Popen") as popen:
        try:
            launch.main()
        except SystemExit as e:
            assert "disabled" in str(e)
        else:
            raise AssertionError("Training lock failed")
        run.assert_not_called()
        popen.assert_not_called()
    report = {"proof_kind": "software_fixture_only", "language_model_loaded": False, "optimizer_steps": 0,
              "checks_passed": ["exact_match_and_token_f1", "attribute_prefix_is_distinct_from_full_answer_em",
                                "official_knowledge_evaluation_and_log_format", "official_full_mmlu_validation_log_format",
                                "learning_and_unlearning_quality_receipts", "normal_control_quality_receipt",
                                "wrong_configuration_evidence_rejected", "training_launch_remains_locked"],
              "fixture_directory": str(fixture)}
    (args.suite_root / "software-contracts.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
