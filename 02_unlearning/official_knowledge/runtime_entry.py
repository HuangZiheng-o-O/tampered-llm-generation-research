"""Apply this job's memory limits, then execute the unchanged official entry."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from official_knowledge.resource_guard import verify_current_scope, available_memory


def apply_limits():
    limit = int(os.environ["ULP_HOST_MEMORY_MAX_BYTES"])
    scope = verify_current_scope(os.environ["ULP_SCOPE_UNIT"], limit)
    required = limit + int(os.environ.get("ULP_HOST_START_RESERVE_BYTES", 0))
    if available_memory() * 1024**3 < required:
        raise RuntimeError("Resource safety guard: insufficient host memory before model loading")
    import torch
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("This runtime entry requires the assigned idle GPU")
    fraction = float(os.environ["ULP_CUDA_MEMORY_FRACTION"])
    if not 0 < fraction <= .8:
        raise ValueError("The experiment CUDA allocator limit must be at most 80 percent")
    torch.cuda.set_device(rank)
    torch.cuda.memory.set_per_process_memory_fraction(fraction, rank)
    actual = torch.cuda.memory.get_per_process_memory_fraction(rank)
    if abs(actual - fraction) > 1e-8:
        raise RuntimeError("The CUDA allocator limit was not applied")
    total = torch.cuda.get_device_properties(rank).total_memory
    record = {"pid": os.getpid(), "scope": os.environ["ULP_SCOPE_UNIT"], "host_limits": scope,
              "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "cuda_allocator_fraction": actual,
              "cuda_allocator_cap_bytes": int(total * fraction), "pretrained_weights_loaded_by_guard": False}
    if os.environ.get("ULP_RUNTIME_RESOURCE_RECEIPT"):
        Path(os.environ["ULP_RUNTIME_RESOURCE_RECEIPT"]).write_text(json.dumps(record, indent=2) + "\n")
    print("RESOURCE_LIMITS " + json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args, remaining = parser.parse_known_args()
    apply_limits()
    if args.check_only:
        return
    if args.entry.name not in ("train.py", "eval.py") or args.entry.parent.name != "src":
        raise ValueError("Expected an official OpenUnlearning train/eval entry")
    if args.entry.name == 'eval.py' and os.environ.get('ULP_EVALUATION_IDENTITY'):
        from official_knowledge.response_cache import install_if_ready
        install_if_ready(os.environ['ULP_EVALUATION_IDENTITY'])
    sys.argv = [str(args.entry)] + remaining
    started = time.monotonic()
    completed = False
    try:
        runpy.run_path(str(args.entry), run_name="__main__")
        completed = True
    finally:
        if os.environ.get("ULP_RUNTIME_RESOURCE_RECEIPT"):
            import torch
            path = Path(os.environ["ULP_RUNTIME_RESOURCE_RECEIPT"])
            record = json.loads(path.read_text())
            record.update(runtime_seconds=time.monotonic() - started, completed=completed,
                          cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
