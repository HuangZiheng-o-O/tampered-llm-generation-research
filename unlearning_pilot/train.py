"""LoRA pilot on the exact PADBench Llama-2-7B base.

The preserved OpenUnlearning submodule supplies the benchmark and objective
reference. This thin implementation keeps only LoRA deltas on disk so a
multi-method pilot fits the owner's server capacity. It does not relabel its
outputs as the upstream full-parameter baseline.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import time
from pathlib import Path

from unlearning_pilot.preflight import EXPECTED_BASE, METHODS, inspect, read_jsonl, sha256


class ShuffledCycle:
    """Visit every row once before reshuffling; keep counts for the receipt."""

    def __init__(self, rows: list[dict], rng: random.Random):
        self.rows = rows
        self.rng = rng
        self.order: list[int] = []
        self.position = 0
        self.visits = 0

    def next(self) -> dict:
        if self.position == len(self.order):
            self.order = list(range(len(self.rows)))
            self.rng.shuffle(self.order)
            self.position = 0
        row = self.rows[self.order[self.position]]
        self.position += 1
        self.visits += 1
        return row


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("sft", "retain_sft", "unlearn"), required=True)
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--parent-adapter", type=Path)
    p.add_argument("--train-device", default="cuda:0")
    p.add_argument("--reference-device", default="cuda:1")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--steering-coeff", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adapter-rank", type=int, default=8)
    p.add_argument("--adapter-alpha", type=int, default=16)
    p.add_argument("--adapter-targets", choices=("core", "all"), default="core")
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    if args.stage == "unlearn" and args.method is None:
        p.error("--method is required for unlearning")
    if args.stage != "unlearn" and args.method is not None:
        p.error("--method is only valid for unlearning")
    if args.stage == "retain_sft" and args.parent_adapter is not None:
        p.error("retain_sft starts from the raw base, without a parent adapter")
    if (min(args.steps, args.accumulation, args.max_length) < 1 or
            args.lr <= 0 or args.checkpoint_every < 0 or
            args.adapter_rank < 1 or args.adapter_alpha < 1):
        p.error("steps, accumulation, max-length and lr must be positive")
    return args


def encode(row: dict, tokenizer, length: int) -> dict:
    import torch

    # Llama-2 joins a trailing prompt space to the first answer word. Keep the
    # boundary before that space so the first answer token receives a label.
    prompt = f"Question: {row['question']}\nAnswer:"
    answer = " " + row["answer"] + tokenizer.eos_token
    all_ids = tokenizer(prompt + answer, add_special_tokens=True,
                        truncation=True, max_length=length)["input_ids"]
    prefix = tokenizer(prompt, add_special_tokens=True,
                       truncation=True, max_length=length)["input_ids"]
    if all_ids[:len(prefix)] != prefix:
        raise ValueError("Prompt tokens do not match the full sequence prefix")
    if len(all_ids) <= len(prefix):
        raise ValueError("No answer token remains after truncation")
    ids = torch.tensor(all_ids, dtype=torch.long).unsqueeze(0)
    labels = ids.clone()
    labels[:, :len(prefix)] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "labels": labels}


def move(batch: dict, device: str) -> dict:
    return {name: value.to(device) for name, value in batch.items()}


def hidden_at_layer(model, batch: dict, layer: int, *, gradients: bool):
    import torch

    matches = [(name, module) for name, module in model.named_modules()
               if name.endswith(f"model.layers.{layer}")]
    if len(matches) != 1:
        raise ValueError(f"Expected one transformer layer {layer}; got {[m[0] for m in matches]}")
    captured = []

    def hook(_module, _inputs, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    handle = matches[0][1].register_forward_hook(hook)
    try:
        with torch.set_grad_enabled(gradients):
            model(**batch)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("RMU layer hook did not fire exactly once")
    return captured[0]


def load_model(base: Path, device: str, adapter: Path | None, *, trainable: bool = True,
               rank: int = 8, alpha: int = 16, targets: str = "core"):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        str(base), dtype=dtype, device_map={"": device},
        low_cpu_mem_usage=True, attn_implementation="sdpa")
    model.config.use_cache = False
    if adapter is None:
        target_modules = (["q_proj", "v_proj", "down_proj"] if targets == "core" else
                          ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                           "up_proj", "down_proj"])
        model = get_peft_model(model, LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", target_modules=target_modules))
    else:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    if trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model


def _write_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def save_adapter(model, destination: Path) -> None:
    """Keep a portable HF base ID in PEFT metadata; the local path stays in the manifest."""
    for config in model.peft_config.values():
        config.base_model_name_or_path = EXPECTED_BASE
    model.save_pretrained(str(destination), safe_serialization=True)


def train(args: argparse.Namespace, manifest: dict) -> None:
    import torch
    from transformers import AutoTokenizer
    from unlearning_pilot.losses import (graddiff, masked_activation_mse,
                                         npo, sequence_nll)

    if not torch.cuda.is_available():
        raise RuntimeError("The pilot requires an NVIDIA GPU")
    train_device = torch.device(args.train_device)
    ref_device = torch.device(args.reference_device)
    if args.stage == "unlearn" and args.method in {"NPO", "RMU"} and train_device == ref_device:
        raise ValueError("NPO and RMU need a separate reference GPU in this pilot")
    for device in ({train_device, ref_device} if args.stage == "unlearn" and
                   args.method in {"NPO", "RMU"} else {train_device}):
        if device.type == "cuda":
            free, _total = torch.cuda.mem_get_info(device)
            if free < 16 * 1024**3:
                raise RuntimeError(f"GPU {device} has only {free / 1024**3:.1f} GiB free; "
                                   "this 7B pilot requires at least 16 GiB before loading")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(args.base))
    if tokenizer.eos_token is None:
        raise ValueError("Base tokenizer has no EOS token")
    tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.base, args.train_device, args.parent_adapter,
                       rank=args.adapter_rank, alpha=args.adapter_alpha,
                       targets=args.adapter_targets)
    reference = None
    if args.stage == "unlearn" and args.method in {"NPO", "RMU"}:
        reference = load_model(args.base, args.reference_device, args.parent_adapter,
                               trainable=False)
        reference.eval()
    if args.method == "RMU":
        selected = []
        for name, parameter in model.named_parameters():
            use = (".mlp.down_proj.lora_" in name and
                   any(f".layers.{layer}." in name for layer in (5, 6, 7)))
            parameter.requires_grad_(use)
            if use:
                selected.append(name)
        if len(selected) != 6:
            raise ValueError(f"RMU expected six LoRA matrices in layers 5-7: {selected}")
        manifest["rmu_trainable_parameters"] = selected
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable LoRA parameters")
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    source = {"sft": ("full.json",),
              "retain_sft": ("retain90.json",),
              "unlearn": ("forget10.json", "retain90.json")}[args.stage]
    examples = {name: ShuffledCycle(read_jsonl(args.data / name), rng) for name in source}
    if args.method == "RMU":
        control = torch.rand((1, 1, 4096), generator=torch.Generator().manual_seed(args.seed))
        control = (control / control.norm() * args.steering_coeff).to(train_device)
    start = time.monotonic()
    model.train()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    manifest["status"] = "running"
    manifest["method"] = args.method
    source_dir = Path(__file__).parent
    manifest["source_sha256"] = {
        name: sha256(source_dir / name)
        for name in ("train.py", "losses.py", "preflight.py", "evaluate.py")
    }
    manifest["software_versions"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "accelerate")
    }
    adapter_config = next(iter(model.peft_config.values()))
    manifest["hyperparameters"] = {
        "steps": args.steps, "accumulation": args.accumulation, "max_length": args.max_length,
        "lr": args.lr, "alpha": args.alpha, "beta": args.beta,
        "steering_coeff": args.steering_coeff, "seed": args.seed,
        "train_device": args.train_device,
        "reference_device": args.reference_device if reference is not None else None,
        "adapter_rank": adapter_config.r, "adapter_alpha": adapter_config.lora_alpha,
        "adapter_targets": sorted(adapter_config.target_modules),
    }
    _write_json(output / "manifest.json", manifest)
    try:
        with (output / "steps.jsonl").open("w") as log:
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                loss_total = 0.0
                for _ in range(args.accumulation):
                    if args.stage in {"sft", "retain_sft"}:
                        source_name = "full.json" if args.stage == "sft" else "retain90.json"
                        row = examples[source_name].next()
                        batch = move(encode(row, tokenizer, args.max_length), args.train_device)
                        loss = model(**batch).loss
                    else:
                        forget = None
                        if args.method != "RetainOnly":
                            forget = move(encode(examples["forget10.json"].next(),
                                                 tokenizer, args.max_length), args.train_device)
                        retain = move(encode(examples["retain90.json"].next(),
                                             tokenizer, args.max_length), args.train_device)
                        if args.method == "RMU":
                            forget_hidden = hidden_at_layer(model, forget, 7, gradients=True)
                            retain_hidden = hidden_at_layer(model, retain, 7, gradients=True)
                            ref_retain = hidden_at_layer(reference,
                                move(retain, args.reference_device), 7, gradients=False)
                            mask_f = forget["labels"] != -100
                            mask_r = retain["labels"] != -100
                            loss = (masked_activation_mse(forget_hidden,
                                    control.expand_as(forget_hidden), mask_f)
                                    + args.alpha * masked_activation_mse(retain_hidden,
                                    ref_retain.to(train_device), mask_r))
                        else:
                            retain_nll = model(**retain).loss
                            if args.method == "GradDiff":
                                forget_output = model(**forget)
                                loss = graddiff(forget_output.loss, retain_nll, args.alpha)
                            elif args.method == "RetainOnly":
                                loss = retain_nll
                            else:
                                forget_output = model(**forget)
                                current = sequence_nll(forget_output.logits, forget["labels"])
                                with torch.no_grad():
                                    ref_batch = move(forget, args.reference_device)
                                    ref_logits = reference(**ref_batch).logits
                                    ref_sum = sequence_nll(ref_logits, ref_batch["labels"])
                                loss = npo(current, ref_sum.to(train_device),
                                           retain_nll, args.beta, args.alpha)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss at step {step}")
                    (loss / args.accumulation).backward()
                    loss_total += float(loss.detach()) / args.accumulation
                optimizer.step()
                record = {"step": step + 1, "loss": loss_total,
                          "elapsed_seconds": round(time.monotonic() - start, 1),
                          "peak_train_gpu_bytes": torch.cuda.max_memory_allocated(train_device)}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0:
                    checkpoint = output / f"checkpoint-{step + 1}" / "adapter"
                    save_adapter(model, checkpoint)
                    _write_json(output / "progress.json", {
                        "last_step": step + 1,
                        "adapter": str(checkpoint),
                        "data_visits": {name: cycle.visits for name, cycle in examples.items()},
                    })
        adapter_dir = output / "adapter"
        save_adapter(model, adapter_dir)
        tokenizer.save_pretrained(str(adapter_dir))
        manifest["adapter_sha256"] = sha256(adapter_dir / "adapter_model.safetensors")
        manifest["status"] = "training_complete_evaluation_pending"
        manifest["data_visits"] = {name: cycle.visits for name, cycle in examples.items()}
        manifest["elapsed_seconds"] = round(time.monotonic() - start, 1)
        manifest["peak_train_gpu_bytes"] = torch.cuda.max_memory_allocated(train_device)
        _write_json(output / "manifest.json", manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        _write_json(output / "manifest.json", manifest)
        raise


def main() -> None:
    args = arguments()
    manifest = inspect(args.base, args.data, args.output, args.stage,
                       args.parent_adapter)
    if args.preflight_only:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    train(args, manifest)


if __name__ == "__main__":
    main()
