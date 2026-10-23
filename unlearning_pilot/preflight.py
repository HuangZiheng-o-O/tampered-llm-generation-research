"""Validate the exact base, data splits, and output path before GPU work.

This module deliberately uses only the standard library so that a failed
dependency installation cannot prevent the preflight from running.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


EXPECTED_BASE = "meta-llama/Llama-2-7b-hf"
EXPECTED_SHAPE = (32, 4096)
METHODS = ("GradDiff", "NPO", "RMU", "RetainOnly")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or any(not isinstance(row.get("question"), str) or
                       not isinstance(row.get("answer"), str) or
                       not row["question"].strip() or not row["answer"].strip()
                       for row in rows):
        raise ValueError(f"Invalid or empty TOFU question/answer data: {path}")
    return rows


def inspect(base: Path, data: Path, output: Path, stage: str,
            parent_adapter: Path | None = None) -> dict:
    base = base.expanduser().resolve(strict=True)
    data = data.expanduser().resolve(strict=True)
    output = output.expanduser().resolve(strict=False)
    config_path = base / "config.json"
    config = json.loads(config_path.read_text())
    shape = (config.get("num_hidden_layers"), config.get("hidden_size"))
    if shape != EXPECTED_SHAPE or config.get("model_type") != "llama":
        raise ValueError(f"Expected 32-layer, 4096-value Llama; found {shape}")
    if base.name != "Llama-2-7b-hf":
        raise ValueError(f"Expected exact base directory Llama-2-7b-hf: {base}")
    source_path = base / "reproduction_source.json"
    source = json.loads(source_path.read_text())
    if (source.get("repo") != EXPECTED_BASE or
            not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", "")) or
            config.get("_name_or_path") != EXPECTED_BASE):
        raise ValueError("Base source receipt does not identify the exact Llama-2-7b-hf revision")
    if not (base / "model.safetensors.index.json").is_file():
        raise ValueError("Base safetensors index is missing")
    index = json.loads((base / "model.safetensors.index.json").read_text())
    names = set(index.get("weight_map", {}))
    required = {f"model.layers.{layer}.mlp.down_proj.weight" for layer in (5, 6, 7)}
    if not required <= names:
        raise ValueError(f"RMU target weights missing: {sorted(required - names)}")
    files = ("full.json", "forget10.json", "retain90.json")
    rows = {name: read_jsonl(data / name) for name in files}
    counts = {name: len(rows[name]) for name in files}
    def qa_counts(name: str) -> Counter:
        return Counter((row["question"], row["answer"]) for row in rows[name])

    if qa_counts("full.json") != qa_counts("forget10.json") + qa_counts("retain90.json"):
        raise ValueError("TOFU full data must equal forget10 plus retain90 row for row")
    if stage == "unlearn":
        forget = {r["question"] for r in rows["forget10.json"]}
        retain = {r["question"] for r in rows["retain90.json"]}
        if forget & retain:
            raise ValueError("Forget and retain questions overlap")
    if stage == "unlearn" and parent_adapter is None:
        raise ValueError("Unlearning requires the completed SFT parent adapter")
    if parent_adapter is not None:
        parent_adapter = parent_adapter.expanduser().resolve(strict=True)
        if not (parent_adapter / "adapter_config.json").is_file():
            raise ValueError("SFT adapter_config.json is missing")
        adapter_config = json.loads((parent_adapter / "adapter_config.json").read_text())
        declared_base = adapter_config.get("base_model_name_or_path")
        if declared_base not in (EXPECTED_BASE, str(base)):
            raise ValueError(f"SFT adapter declares another base: {declared_base}")
        if not (parent_adapter / "adapter_model.safetensors").is_file():
            raise ValueError("SFT adapter weights are missing")
        parent_manifest = parent_adapter.parent / "manifest.json"
        if not parent_manifest.is_file():
            raise ValueError("SFT parent manifest is missing")
        parent = json.loads(parent_manifest.read_text())
        if (parent.get("stage") != "sft" or
                parent.get("status") != "training_complete_evaluation_pending" or
                parent.get("base_model_id") != EXPECTED_BASE or
                parent.get("base_config_sha256") != sha256(config_path) or
                parent.get("data", {}).get("full.json", {}).get("sha256") !=
                sha256(data / "full.json") or
                parent.get("adapter_sha256") != sha256(parent_adapter / "adapter_model.safetensors")):
            raise ValueError("SFT parent base or full training data does not match")
    if output.exists():
        raise ValueError(f"Output path already exists: {output}")
    return {
        "stage": stage,
        "base_model_id": EXPECTED_BASE,
        "base_path": str(base),
        "base_config_sha256": sha256(config_path),
        "base_source_sha256": sha256(source_path),
        "base_revision": source["revision"],
        "layers": [0, 15, 31],
        "values_per_layer": 4096,
        "dataset_dir": str(data),
        "data": {name: {"rows": counts[name], "sha256": sha256(data / name)}
                 for name in files},
        "parent_adapter": str(parent_adapter) if parent_adapter else None,
        "parent_adapter_sha256": (
            sha256(parent_adapter / "adapter_model.safetensors") if parent_adapter else None),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("sft", "retain_sft", "unlearn"), required=True)
    parser.add_argument("--parent-adapter", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.base, args.data, args.output, args.stage,
                             args.parent_adapter), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
