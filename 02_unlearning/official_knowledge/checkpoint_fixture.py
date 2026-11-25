"""Real save/resume/export integration on a tiny random model, not a research result."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from omegaconf import OmegaConf

from official_knowledge.resource_guard import POLICY, idle_gpus, monitor, scope_command
from official_pilot.launch import checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    root = args.suite / "cache/checkpoint-io-v2-software-check"
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert root.stat().st_uid == os.getuid()
    (root / "PROOF_KIND.txt").write_text("software_fixture_only\n")
    state = {"proof_kind": "software_fixture_only", "pretrained_weights_loaded": False}

    def status(phase, **values):
        state.update(phase=phase, updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **values)
        (root / "RESULT.json").write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps(state), flush=True)

    status("preparing_software_fixture")
    import torch
    torch.set_num_threads(4)
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
    model_path = root / "tiny-random-model"
    if not model_path.exists():
        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32000, hidden_size=32,
            intermediate_size=64, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=4,
            bos_token_id=1, eos_token_id=2, pad_token_id=2, use_cache=False))
        model.save_pretrained(model_path)
        del model
    dataset = root / "published-four-records.jsonl"
    dataset.write_text("\n".join((args.suite / "data/biographies/learn.jsonl").read_text().splitlines()[:4]) + "\n")
    ds = json.loads((args.suite / "configs/zero3_cpu.json").read_text())
    ds["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = False
    (root / "zero3-native.json").write_text(json.dumps(ds, indent=2) + "\n")
    cfg = OmegaConf.load(args.suite / "configs/biographies_learn.yaml")
    output = root / "run"
    cfg.model.model_args.pretrained_model_name_or_path = str(model_path)
    cfg.paths.output_dir = str(output)
    cfg.hydra.run.dir = str(output)
    cfg.trainer.args.update(output_dir=str(output), logging_dir=str(root / "logs"),
        per_device_train_batch_size=1, gradient_accumulation_steps=1, warmup_steps=0,
        num_train_epochs=1, max_steps=1, save_steps=1, deepspeed=str(root / "zero3-native.json"))
    cfg.data.train.published_records.args.hf_args.data_files = str(dataset)
    OmegaConf.set_struct(cfg, False)
    cfg.native_checkpoint_io, cfg.checkpoint_first_step = True, False
    cfg.task_name = "checkpoint_fixture_first_step"
    OmegaConf.save(cfg, root / "first.yaml")
    cfg.trainer.args.max_steps = 2
    cfg.resume_from_checkpoint = str(output / "checkpoint-1")
    cfg.task_name = "checkpoint_fixture_resume"
    OmegaConf.save(cfg, root / "resume.yaml")
    policy = dict(POLICY, training_memory_max_gib=6, training_memory_high_gib=6,
                  evaluation_memory_max_gib=6, evaluation_memory_high_gib=6)
    support = Path(__file__).resolve().parents[1]
    cache = Path("/local/rcs/zh2701/LLM-safety/cache/open-unlearning-official-v1")

    def run(stage, command, gpu):
        unit = f"ulp-small-{os.getpid()}-fixture-{stage}-{time.time_ns()}.scope"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpu), PYTHONPATH=str(support) + ":" + str(args.upstream / "src"),
            ULP_SCOPE_UNIT=unit, ULP_HOST_MEMORY_MAX_BYTES=str(6 * 1024**3),
            ULP_HOST_START_RESERVE_BYTES=str(8 * 1024**3), ULP_CUDA_MEMORY_FRACTION="0.8",
            ULP_RUNTIME_RESOURCE_RECEIPT=str(root / (stage + "-limits.json")),
            HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS="4", MAX_JOBS="2", NCCL_IB_DISABLE="1", CUDA_HOME="/usr/local/cuda-12.6",
            TRITON_CACHE_DIR=str(cache / "triton"), TORCH_EXTENSIONS_DIR=str(cache / "torch-extensions"),
            PATH=str(Path(sys.executable).parent) + ":/usr/local/cuda-12.6/bin:" + os.environ.get("PATH", ""))
        status(stage, scope=unit, gpu=gpu)
        log = root / (stage + ".log")
        with log.open("a") as stream:
            process = subprocess.Popen(scope_command(command, unit, policy, bool(gpu)), env=env,
                cwd=args.upstream, stdout=stream, stderr=subprocess.STDOUT)
            reason = monitor(process, unit, gpu, policy, root / (stage + "-resources.json"), log)
        if process.returncode or reason:
            raise RuntimeError(stage + " failed; inspect " + str(log))

    try:
        deadline = time.monotonic() + 7200
        while True:
            idle, _ = idle_gpus()
            if idle:
                gpu = idle[:1]
                break
            status("waiting_for_idle_gpu")
            if time.monotonic() > deadline:
                raise TimeoutError("No idle GPU for the software fixture")
            time.sleep(60)
        common = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1", "--",
                  str(support / "official_knowledge/runtime_entry.py"), "--entry", str(args.upstream / "src/train.py"),
                  "--config-path", str(root), "--config-name"]
        run("first", common + ["first"], gpu)
        checkpoint(output / "checkpoint-1", output)
        assert json.loads((output / "checkpoint-1/trainer_state.json").read_text())["global_step"] == 1
        run("resume", common + ["resume"], gpu)
        assert json.loads((output / "trainer_state.json").read_text())["global_step"] == 2
        checkpoint(output / "checkpoint-2", output)
        run("export", [sys.executable, "-m", "official_knowledge.export_checkpoint", "--checkpoint", str(output)], [])
        loaded = AutoModelForCausalLM.from_pretrained(output, local_files_only=True, dtype=torch.bfloat16)
        assert all(torch.isfinite(p).all() for p in loaded.parameters())
        status("passed", restored_optimizer_step=1, completed_optimizer_steps=2,
               exported_parameters=sum(p.numel() for p in loaded.parameters()),
               model_reloaded=True, real_7b_checkpoint_qualified=False)
    except Exception as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
