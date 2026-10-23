"""Give an already saved adapter a portable base ID with a change receipt.

Only adapter_config.json is changed. The safetensors weights remain untouched.
This is for the first SFT job launched before save_adapter() became canonical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from unlearning_pilot.preflight import EXPECTED_BASE, sha256


def canonicalize(run: Path) -> dict:
    run = run.expanduser().resolve(strict=True)
    manifest_path = run / "manifest.json"
    config_path = run / "adapter" / "adapter_config.json"
    weights_path = run / "adapter" / "adapter_model.safetensors"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "training_complete_evaluation_pending":
        raise ValueError("Training is not complete")
    if manifest.get("base_model_id") != EXPECTED_BASE:
        raise ValueError("Run uses another base model")
    if sha256(Path(manifest["base_path"]) / "config.json") != manifest["base_config_sha256"]:
        raise ValueError("Base model config changed since training")
    original = json.loads(config_path.read_text())
    before = original.get("base_model_name_or_path")
    if before not in (manifest["base_path"], EXPECTED_BASE):
        raise ValueError(f"Unexpected adapter base: {before}")
    if before == EXPECTED_BASE:
        return {"status": "already_canonical", "adapter_sha256": sha256(weights_path)}
    receipt_path = run / "adapter_base_normalization.json"
    if receipt_path.exists():
        raise ValueError("Normalization receipt already exists")
    before_hash = sha256(config_path)
    original["base_model_name_or_path"] = EXPECTED_BASE
    temporary = config_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(original, indent=2) + "\n")
    temporary.replace(config_path)
    receipt = {"status": "complete", "before": before, "after": EXPECTED_BASE,
               "adapter_config_sha256_before": before_hash,
               "adapter_config_sha256_after": sha256(config_path),
               "adapter_weights_sha256": sha256(weights_path)}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(canonicalize(args.run), indent=2))


if __name__ == "__main__":
    main()
