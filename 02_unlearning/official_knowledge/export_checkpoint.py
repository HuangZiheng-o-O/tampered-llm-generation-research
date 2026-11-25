"""Export an owned native ZeRO checkpoint in bounded BF16 safetensors shards."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path


def export(checkpoint, shard_bytes=1_000_000_000):
    import torch
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    from safetensors.torch import save_file
    checkpoint = checkpoint.resolve(strict=True)
    if checkpoint.stat().st_uid != os.getuid():
        raise ValueError("Checkpoint ownership mismatch")
    marker = checkpoint / "native-export.json"
    if marker.exists():
        prior = json.loads(marker.read_text())
        for name, digest in prior["sha256"].items():
            if hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Existing export changed")
        return prior
    if list(checkpoint.glob("*.safetensors")):
        raise ValueError("Refusing to overwrite a partial or unrelated export")
    lazy = get_fp32_state_dict_from_zero_checkpoint(str(checkpoint), lazy_mode=True)
    groups, current, size = [], [], 0
    for name, value in lazy.items():
        dtype = value.dtype
        count = 1
        for dimension in value.shape:
            count *= dimension
        element_size = 2 if dtype.is_floating_point else torch.empty((), dtype=dtype).element_size()
        amount = count * element_size
        if current and size + amount > shard_bytes:
            groups.append(current)
            current, size = [], 0
        current.append(name)
        size += amount
    if current:
        groups.append(current)
    mapping, digests, total = {}, {}, 0
    for i, names in enumerate(groups, 1):
        name = f"model-{i:05d}-of-{len(groups):05d}.safetensors"
        tensors = {}
        for key in names:
            tensor = lazy[key].contiguous()
            if tensor.is_floating_point():
                tensor = tensor.to(torch.bfloat16)
                if not torch.isfinite(tensor).all():
                    raise ValueError("Non-finite exported tensor: " + key)
            tensors[key] = tensor
            total += tensor.numel() * tensor.element_size()
            mapping[key] = name
        temporary = checkpoint / ("." + name + ".exporting")
        if temporary.exists():
            raise ValueError("An interrupted export already exists")
        save_file(tensors, str(temporary), metadata={"format": "pt"})
        temporary.rename(checkpoint / name)
        with (checkpoint / name).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        digests[name] = digest
        for key in names:
            del lazy[key]
        del tensors, tensor
        gc.collect()
        print(f"EXPORTED_SHARD {i}/{len(groups)} {name}", flush=True)
    index = {"metadata": {"total_size": total}, "weight_map": mapping}
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    result = {"format": "native-zero-to-bf16-shards-v1", "tensors": len(mapping),
              "total_tensor_bytes": total, "shard_count": len(groups), "sha256": digests}
    marker.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("ULP_SCOPE_UNIT"):
        from official_knowledge.resource_guard import verify_current_scope
        verify_current_scope(os.environ["ULP_SCOPE_UNIT"], int(os.environ["ULP_HOST_MEMORY_MAX_BYTES"]))
    print(json.dumps(export(args.checkpoint)), flush=True)


if __name__ == "__main__":
    main()
