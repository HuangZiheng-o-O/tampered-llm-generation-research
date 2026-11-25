"""PEFT model loading for the official trainers and a frozen-parent reference."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from omegaconf import OmegaConf
from torch import nn


def configure_lora(model, options):
    from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model

    options = OmegaConf.to_container(OmegaConf.create(options), resolve=True)
    adapter_path = options.pop("adapter_path", None)
    trainable = options.pop("trainable", adapter_path is None)
    if adapter_path:
        saved = PeftConfig.from_pretrained(adapter_path, local_files_only=True)
        if Path(saved.base_model_name_or_path).resolve() != Path(model.name_or_path).resolve():
            raise ValueError("Adapter and loaded parent identities differ")
        model = PeftModel.from_pretrained(
            model, adapter_path, is_trainable=trainable, local_files_only=True
        )
    else:
        if not trainable:
            raise ValueError("A fresh LoRA adapter must be trainable")
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **options))
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable and (not names or any("lora_" not in name for name in names)):
        raise ValueError("Only LoRA parameters may be trainable")
    print("LORA_MODEL " + json.dumps({
        "parent": model.peft_config["default"].base_model_name_or_path,
        "adapter_path": adapter_path,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_tensor_count": len(names),
        "adapter_dtype": sorted({str(p.dtype) for n, p in model.named_parameters() if "lora_" in n}),
    }), flush=True)
    return model


class FrozenParentReference(nn.Module):
    """Evaluate the frozen learned parent with the new task adapter disabled."""

    def __init__(self, student):
        super().__init__()
        if not hasattr(student, "disable_adapter"):
            raise ValueError("Expected a PEFT student")
        if any(p.requires_grad for n, p in student.named_parameters() if "lora_" not in n):
            raise ValueError("Reference sharing requires a completely frozen parent")
        self.student = student

    def forward(self, *args, **kwargs):
        was_training = self.student.training
        try:
            self.student.eval()
            with torch.no_grad(), self.student.disable_adapter():
                return self.student(*args, **kwargs)
        finally:
            self.student.train(was_training)


def merge_parent(pilot, name):
    """Run one bounded merge after the official training process exits."""
    from official_knowledge.resource_guard import monitor, scope_command

    cfg = pilot.configuration(name)
    output = Path(pilot.manifest["jobs"][name]["output"])
    while True:
        gpus = pilot.resources(name)
        unit = f"ulp-small-{os.getpid()}-lora-merge-{time.time_ns()}.scope"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus),
            PYTHONPATH=str(pilot.support) + ":" + str(pilot.upstream / "src"),
            ULP_SCOPE_UNIT=unit,
            ULP_HOST_MEMORY_MAX_BYTES=str(pilot.policy["training_memory_max_gib"] * 1024**3),
            ULP_HOST_START_RESERVE_BYTES=str(pilot.policy["host_start_reserve_gib"] * 1024**3),
            ULP_CUDA_MEMORY_FRACTION=str(pilot.policy["cuda_allocator_fraction"]))
        command = [sys.executable, "-m", "official_pilot.lora", "--base",
            str(cfg.model.model_args.pretrained_model_name_or_path), "--adapter", str(output),
            "--output", str(output / "merged")]
        log = pilot.logs / (name + ".merge.log")
        pilot.record(phase="exporting_checkpoint", current_job=name, gpus=gpus, resource_scope=unit)
        with log.open("a") as stream:
            process = subprocess.Popen(scope_command(command, unit, pilot.policy, True),
                env=env, cwd=pilot.support, stdout=stream, stderr=subprocess.STDOUT)
            reason = monitor(process, unit, gpus, pilot.policy, pilot.logs / (name + ".merge-resources.json"), log)
        if not process.returncode and not reason:
            return
        if reason in ("another_process_appeared_on_selected_gpu", "host_memory_pressure"):
            pilot.record(phase="waiting_for_resources", current_job=name, reason=reason)
            continue
        raise RuntimeError("LoRA parent merge needs attention: " + name)


def main():
    import argparse
    import hashlib
    import uuid
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from official_knowledge.runtime_entry import apply_limits

    parser = argparse.ArgumentParser(description="Merge one qualified learning adapter")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    apply_limits()
    if args.output.exists():
        raise ValueError("Merged parent output already exists")
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda:0", local_files_only=True)
    model = configure_lora(model, {"adapter_path": str(args.adapter), "trainable": False})
    model.eval()
    tokens = torch.tensor([[model.config.bos_token_id or 1, 42, 43]], device="cuda:0")
    with torch.no_grad():
        before = model(tokens).logits.float()
        model = model.merge_and_unload(safe_merge=True)
        after = model(tokens).logits.float()
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise ValueError("Adapter merge produced non-finite probe outputs")
    pending = args.output.with_name(args.output.name + ".pending-" + uuid.uuid4().hex)
    model.save_pretrained(pending, max_shard_size="1GB")
    AutoTokenizer.from_pretrained(args.base, local_files_only=True).save_pretrained(pending)
    weights = list(pending.glob("*.safetensors"))
    expected_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    if sum(p.stat().st_size for p in weights) < expected_bytes:
        raise ValueError("Merged weight files are incomplete")
    digest = hashlib.sha256((args.adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    (pending / "lora-parent.json").write_text(json.dumps({
        "base": str(args.base.resolve()), "learning_adapter": str(args.adapter.resolve()),
        "adapter_sha256": digest, "merge_max_abs_logit_difference": (before - after).abs().max().item(),
        "merge_relative_l2_logit_difference": ((before - after).norm() / before.norm().clamp_min(1e-8)).item(),
        "merge_behavior_checked_by": "official learned-adapter and merged-parent knowledge/MMLU evaluations",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "weight_bytes": sum(p.stat().st_size for p in weights),
    }, indent=2) + "\n")
    pending.rename(args.output)


if __name__ == "__main__":
    main()
