"""Official integration checks with tiny CPU fixtures; never call train/step."""
import copy
import importlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments
from omegaconf import OmegaConf

from official_pilot.launch import checkpoint


class OfficialContracts(unittest.TestCase):
    def test_hydra_arguments_construct_a_real_official_trainer(self):
        from trainer import load_trainer_args, load_trainer
        with tempfile.TemporaryDirectory(prefix="official-hydra-trainer-check-") as directory:
            cfg = OmegaConf.create({"handler": "FinetuneTrainer", "args": {
                "output_dir": directory, "use_cpu": True, "report_to": [], "do_train": False,
                "gradient_checkpointing_kwargs": {"use_reentrant": False}}})
            # The early ZeRO setup passes a shallow dict from Hydra, so nested
            # ListConfig/DictConfig values must be converted inside the loader.
            args = load_trainer_args(dict(cfg.args), None)
            self.assertIs(type(args.report_to), list)
            self.assertIs(type(args.gradient_checkpointing_kwargs), dict)
            model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2))
            instance, actual = load_trainer(cfg, model, training_args=args)
            self.assertIs(actual, args)
            self.assertEqual(instance.args.report_to, [])
            self.assertIsNone(instance.optimizer)

    def test_small_forget_set_keeps_all_ten_epochs(self):
        from transformers import Trainer
        args = SimpleNamespace(max_steps=-1, gradient_accumulation_steps=16, num_train_epochs=10)
        owner = SimpleNamespace(get_total_train_batch_size=lambda a: 32, get_sp_size=lambda: 1,
                                num_examples=lambda loader: 40)
        loader = type("Loader", (), {"__len__": lambda self: 20})()
        values = Trainer.set_initial_training_values(owner, args, loader)
        self.assertEqual(values[0], 10)
        self.assertEqual(values[1], 2)
        self.assertEqual(values[3], 400)
        self.assertEqual(values[-1], 20)

    def test_official_lm_eval_forwards_configured_arguments(self):
        from evals.lm_eval import LMEvalEvaluator
        cfg = OmegaConf.create({"tasks": ["pilot_mmlu_validation"], "include_path": "/prepared/tasks",
                                "simple_evaluate_args": {"batch_size": 1, "num_fewshot": 5, "log_samples": True}})
        with patch("evals.lm_eval.TaskManager") as manager:
            evaluator = LMEvalEvaluator(cfg)
        manager.assert_called_once_with(include_path="/prepared/tasks")
        self.assertEqual(evaluator.simple_evaluate_args, {"batch_size": 1, "num_fewshot": 5, "log_samples": True})

    def test_official_entry_initializes_zero_before_load_and_passes_resume(self):
        entry = importlib.import_module("train")
        order = []
        trainer = Mock()
        args = SimpleNamespace(do_train=True, do_eval=False, output_dir="/owned/run")
        cfg = OmegaConf.create({"trainer": {"args": {"seed": 0}}, "model": {"template_args": {}},
                                "data": {}, "collator": {}, "eval": None,
                                "resume_from_checkpoint": "/owned/run/checkpoint-10"})
        with patch.object(entry, "load_trainer_args", side_effect=lambda *a: order.append("args") or args), \
             patch.object(entry, "get_model", side_effect=lambda *a: order.append("model") or (Mock(), Mock())), \
             patch.object(entry, "get_data", return_value={"train": []}), \
             patch.object(entry, "get_collators", return_value=Mock()), \
             patch.object(entry, "load_trainer", return_value=(trainer, args)) as loader:
            entry.main.__wrapped__(cfg)
        self.assertEqual(order, ["args", "model"])
        trainer.train.assert_called_once_with(resume_from_checkpoint="/owned/run/checkpoint-10")
        self.assertIs(loader.call_args.kwargs["training_args"], args)

    def test_eos_attention_is_kept_when_eos_is_also_padding(self):
        from data.collators import DataCollatorForSupervisedDataset
        collator = DataCollatorForSupervisedDataset(SimpleNamespace(pad_token_id=2))
        instances = [{"input_ids": torch.tensor([1, 7, 2]), "attention_mask": torch.tensor([1, 1, 1])},
                     {"input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1])}]
        self.assertEqual(collator(instances)["attention_mask"].tolist(), [[True, True, True], [True, True, False]])

    def test_undial_ignores_prompt_and_pad_positions(self):
        from trainer.utils import compute_undial_loss
        torch.manual_seed(0)
        teacher = torch.randn(1, 6, 32)
        student = torch.randn(1, 6, 32)
        labels = torch.tensor([[-100, -100, 4, 7, 2, -100]])
        model = lambda value: lambda **kw: SimpleNamespace(logits=value)
        loss1, _ = compute_undial_loss(model(student), model(teacher), {"labels": labels}, 10.)
        changed = student.clone()
        changed[:, 0, :] = torch.randn(32) * 100
        changed[:, 4, :] = torch.randn(32) * 100
        loss2, _ = compute_undial_loss(model(changed), model(teacher), {"labels": labels}, 10.)
        torch.testing.assert_close(loss1, loss2)
        self.assertTrue(torch.isfinite(loss1))

    def test_reference_zero_config_has_no_optimizer(self):
        from trainer.unlearn.base import UnlearnTrainer
        from deepspeed.runtime.zero.config import DeepSpeedZeroConfig
        prepared = json.loads((Path(os.environ["OFFICIAL_CONFIGS"]) / "zero3_cpu.json").read_text())
        validated = DeepSpeedZeroConfig(**prepared["zero_optimization"])
        self.assertEqual(validated.stage, 3)
        self.assertEqual(validated.offload_param.device, "cpu")
        self.assertEqual(validated.offload_optimizer.device, "cpu")
        config = {"zero_optimization": {"stage": 3}, "optimizer": {"type": "AdamW"}, "scheduler": {"type": "WarmupDecayLR"}}
        owner = SimpleNamespace(accelerator=SimpleNamespace(state=SimpleNamespace(
            deepspeed_plugin=SimpleNamespace(deepspeed_config=config))))
        model = SimpleNamespace(config=SimpleNamespace(hidden_size=16))
        engine = Mock()
        with patch("trainer.unlearn.base.deepspeed.initialize", return_value=(engine, None, None, None)) as initialize:
            UnlearnTrainer._prepare_deepspeed(owner, model)
        sent = initialize.call_args.kwargs["config"]
        self.assertNotIn("optimizer", sent)
        self.assertNotIn("scheduler", sent)
        self.assertIsInstance(sent["zero_optimization"]["stage3_prefetch_bucket_size"], int)
        self.assertEqual(config["optimizer"]["type"], "AdamW")

    def test_all_five_official_method_losses_are_finite_without_training(self):
        from trainer import TRAINER_REGISTRY
        configs = Path(os.environ["OFFICIAL_CONFIGS"])
        batch = {"input_ids": torch.tensor([[1, 5, 9, 4, 2]]),
                 "attention_mask": torch.ones(1, 5, dtype=torch.long),
                 "labels": torch.tensor([[-100, -100, 9, 4, 2]])}
        for method in ("NPO", "SimNPO", "GradDiff", "RMU", "UNDIAL"):
            with self.subTest(method=method):
                directory = tempfile.mkdtemp(prefix="official-method-check-")
                recipe = OmegaConf.load(configs / f"{method}.yaml")
                model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=8, num_attention_heads=2, num_key_value_heads=2, use_cache=False))
                model.eval()
                args = TrainingArguments(output_dir=directory, use_cpu=True, report_to=[], do_train=False,
                                         gradient_accumulation_steps=16, optim="adamw_torch")
                trainer = TRAINER_REGISTRY[method](model=model, args=args, **OmegaConf.to_container(recipe.trainer.method_args))
                before = {n: p.detach().clone() for n, p in model.named_parameters()}
                loss = trainer.compute_loss(model, {"forget": copy.deepcopy(batch), "retain": copy.deepcopy(batch)})
                self.assertTrue(torch.isfinite(loss))
                self.assertFalse(trainer.model_accepts_loss_kwargs)
                self.assertFalse(trainer.loss_is_scaled_for_ga)
                for name, param in model.named_parameters():
                    torch.testing.assert_close(param, before[name])
                if method == "RMU":
                    # Also check that the upstream parameter selector can keep
                    # the original RMU subset frozen through optimizer creation.
                    trainer.trainable_params_regex = ["model[.]layers[.](5|6|7)[.]mlp[.]down_proj[.]weight"]
                    trainer.create_optimizer()  # Creation only; no optimizer step.
                    names = [n for n, p in model.named_parameters() if p.requires_grad]
                    self.assertEqual(names, [f"model.layers.{i}.mlp.down_proj.weight" for i in (5, 6, 7)])
                    first = trainer.get_control_vector(16).clone()
                    trainer.control_vec = None
                    torch.manual_seed(999)
                    torch.testing.assert_close(first, trainer.get_control_vector(16))

    def test_incomplete_checkpoint_is_rejected(self):
        output = Path(tempfile.mkdtemp(prefix="official-checkpoint-check-"))
        ckpt = output / "checkpoint-10"
        ckpt.mkdir()
        (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": 10}))
        with self.assertRaises(ValueError):
            checkpoint(ckpt, output)
        (ckpt / "rng_state.pth").write_bytes(b"fixture")
        state = ckpt / "global_step10"
        state.mkdir()
        (state / "rank0_optim_states.pt").write_bytes(b"fixture")
        (state / "rank0_model_states.pt").write_bytes(b"fixture")
        self.assertEqual(checkpoint(ckpt, output), ckpt.resolve())
        link = ckpt / "escape"
        link.symlink_to(output)
        with self.assertRaises(ValueError):
            checkpoint(ckpt, output)

    def test_execution_lock_blocks_training(self):
        from official_pilot import launch
        root = Path(os.environ["OFFICIAL_CONFIGS"]).parents[1]
        with patch("sys.argv", ["launch", "NPO", "--root", str(root), "--upstream", "/unused", "--execute"]), \
             patch.object(launch.subprocess, "run") as run, patch.object(launch.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(SystemExit, "disabled"):
                launch.main()
            run.assert_not_called()
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
