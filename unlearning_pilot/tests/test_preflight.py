from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from unlearning_pilot.preflight import inspect, sha256


class PreflightTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.base = root / "Llama-2-7b-hf"
        self.base.mkdir()
        (self.base / "config.json").write_text(json.dumps({
            "model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096,
            "_name_or_path": "meta-llama/Llama-2-7b-hf"}))
        (self.base / "reproduction_source.json").write_text(json.dumps({
            "repo": "meta-llama/Llama-2-7b-hf", "revision": "a" * 40}))
        names = {f"model.layers.{layer}.mlp.down_proj.weight": "weight.safetensors"
                 for layer in (5, 6, 7)}
        (self.base / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": names}))
        self.data = root / "data"
        self.data.mkdir()
        for name, rows in (("full.json", [("f", "a"), ("r", "a")]),
                           ("forget10.json", [("f", "a")]),
                           ("retain90.json", [("r", "a")])):
            (self.data / name).write_text("".join(json.dumps({
                "question": q, "answer": a}) + "\n" for q, a in rows))
        self.parent = root / "parent"
        (self.parent / "adapter").mkdir(parents=True)
        (self.parent / "adapter" / "adapter_config.json").write_text(
            '{"base_model_name_or_path":"meta-llama/Llama-2-7b-hf"}')
        (self.parent / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
        self.parent_manifest = self.parent / "manifest.json"

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_base_and_nonoverlap(self):
        planned = inspect(self.base, self.data, self.parent.parent / "sft_new", "sft")
        self.assertEqual(planned["data"]["full.json"]["rows"], 2)
        planned["status"] = "training_complete_evaluation_pending"
        planned["adapter_sha256"] = sha256(self.parent / "adapter" / "adapter_model.safetensors")
        self.parent_manifest.write_text(json.dumps(planned))
        unlearn = inspect(self.base, self.data, self.parent.parent / "new", "unlearn",
                          self.parent / "adapter")
        self.assertEqual(unlearn["data"]["forget10.json"]["rows"], 1)
        continue_sft = inspect(self.base, self.data, self.parent.parent / "continued", "sft",
                               self.parent / "adapter")
        self.assertEqual(continue_sft["parent_adapter_sha256"], planned["adapter_sha256"])
        retain_plan = inspect(self.base, self.data, self.parent.parent / "retain_new",
                              "retain_sft")
        self.assertEqual(retain_plan["data"]["retain90.json"]["rows"], 1)

    def test_retain_only_continuation_requires_retain_parent(self):
        planned = inspect(self.base, self.data, self.parent.parent / "retain_new", "retain_sft")
        planned["status"] = "training_complete_evaluation_pending"
        planned["adapter_sha256"] = sha256(self.parent / "adapter" / "adapter_model.safetensors")
        self.parent_manifest.write_text(json.dumps(planned))
        continued = inspect(self.base, self.data, self.parent.parent / "continued_retain",
                            "retain_sft", self.parent / "adapter")
        self.assertEqual(continued["parent_adapter_sha256"], planned["adapter_sha256"])

    def test_verified_partial_checkpoint_can_resume_sft_only(self):
        checkpoint = self.parent / "checkpoint-1" / "adapter"
        checkpoint.mkdir(parents=True)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            (checkpoint / name).write_bytes((self.parent / "adapter" / name).read_bytes())
        planned = inspect(self.base, self.data, self.parent.parent / "sft_new", "sft")
        planned["status"] = "interrupted_by_personal_disk_quota"
        planned["hyperparameters"] = {"steps": 2, "accumulation": 1,
                                      "max_length": 192, "seed": 42, "lr": 1e-4}
        self.parent_manifest.write_text(json.dumps(planned))
        (self.parent / "recovery_checkpoint.json").write_text(json.dumps({
            "last_valid_step": 1, "adapter": str(checkpoint)}))
        resumed = inspect(self.base, self.data, self.parent.parent / "continued", "sft",
                          checkpoint)
        self.assertEqual(resumed["recovery"]["checkpoint_step"], 1)
        with self.assertRaisesRegex(ValueError, "completed SFT"):
            inspect(self.base, self.data, self.parent.parent / "unlearn", "unlearn",
                    checkpoint)

    def test_rejects_stopped_sft_parent(self):
        planned = inspect(self.base, self.data, self.parent.parent / "sft_new", "sft")
        planned["status"] = "stopped_tokenizer_boundary_error"
        planned["adapter_sha256"] = sha256(self.parent / "adapter" / "adapter_model.safetensors")
        self.parent_manifest.write_text(json.dumps(planned))
        with self.assertRaisesRegex(ValueError, "SFT parent"):
            inspect(self.base, self.data, self.parent.parent / "new", "unlearn",
                    self.parent / "adapter")

    def test_rejects_sft_parent_from_other_data(self):
        planned = inspect(self.base, self.data, self.parent.parent / "sft_new", "sft")
        planned["status"] = "training_complete_evaluation_pending"
        planned["adapter_sha256"] = sha256(self.parent / "adapter" / "adapter_model.safetensors")
        planned["data"]["full.json"]["sha256"] = "0" * 64
        self.parent_manifest.write_text(json.dumps(planned))
        with self.assertRaisesRegex(ValueError, "full training data"):
            inspect(self.base, self.data, self.parent.parent / "new", "unlearn",
                    self.parent / "adapter")

    def test_rejects_wrong_width(self):
        path = self.base / "config.json"
        config = json.loads(path.read_text())
        config["hidden_size"] = 5120
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "Expected 32-layer"):
            inspect(self.base, self.data, self.parent, "sft")

    def test_rejects_wrong_base_origin(self):
        (self.base / "reproduction_source.json").write_text(json.dumps({
            "repo": "meta-llama/Llama-2-7b-chat-hf", "revision": "a" * 40}))
        with self.assertRaisesRegex(ValueError, "Base source receipt"):
            inspect(self.base, self.data, self.parent, "sft")

    def test_rejects_split_overlap(self):
        planned = inspect(self.base, self.data, self.parent.parent / "sft_new", "sft")
        self.parent_manifest.write_text(json.dumps(planned))
        (self.data / "retain90.json").write_text(
            '{"question":"f","answer":"b"}\n')
        (self.data / "full.json").write_text(
            '{"question":"f","answer":"a"}\n{"question":"f","answer":"b"}\n')
        with self.assertRaisesRegex(ValueError, "overlap"):
            inspect(self.base, self.data, self.parent.parent / "new", "unlearn",
                    self.parent / "adapter")

    def test_rejects_split_not_equal_to_full(self):
        (self.data / "retain90.json").write_text(
            '{"question":"other","answer":"a"}\n')
        with self.assertRaisesRegex(ValueError, "full data must equal"):
            inspect(self.base, self.data, self.parent.parent / "new", "sft")


if __name__ == "__main__":
    unittest.main()
