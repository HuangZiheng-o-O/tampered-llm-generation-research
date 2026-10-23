import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    torch = None
    save_file = None

from unlearning_pilot.train import save_adapter


class FakeModel:
    def __init__(self, valid):
        self.valid = valid
        self.peft_config = {"default": SimpleNamespace(base_model_name_or_path="wrong")}

    def save_pretrained(self, path, safe_serialization):
        destination = Path(path)
        destination.mkdir(parents=True)
        (destination / "adapter_config.json").write_text(json.dumps({"ok": True}))
        weights = destination / "adapter_model.safetensors"
        if self.valid:
            save_file({"weight": torch.ones(2)}, weights)
        else:
            weights.write_bytes(b"truncated")


@unittest.skipIf(torch is None, "torch and safetensors unavailable")
class CheckpointTest(unittest.TestCase):
    def test_valid_checkpoint_is_published(self):
        with tempfile.TemporaryDirectory() as temp:
            model = FakeModel(valid=True)
            destination = Path(temp) / "checkpoint" / "adapter"
            save_adapter(model, destination)
            self.assertTrue((destination / "adapter_model.safetensors").is_file())
            self.assertFalse(destination.with_name("adapter.writing").exists())
            self.assertEqual(model.peft_config["default"].base_model_name_or_path,
                             "meta-llama/Llama-2-7b-hf")

    def test_truncated_checkpoint_is_not_published(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "checkpoint" / "adapter"
            with self.assertRaises(Exception):
                save_adapter(FakeModel(valid=False), destination)
            self.assertFalse(destination.exists())
            self.assertTrue(destination.with_name("adapter.writing").exists())
