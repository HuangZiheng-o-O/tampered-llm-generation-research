# 400-model pure-LoRA unlearning generation

Data collection explained: [English](DATA_COLLECTION_EN.md) | [中文](DATA_COLLECTION_ZH.md).

This directory now includes the **actual production support code** used for the
completed 400-model batch, alongside the pinned official OpenUnlearning submodule.
The 55 Python files in `official_knowledge/` and `official_pilot/` were checked
byte-for-byte against the frozen server production manifest on November 25, 2026.
Their hashes are in [PRODUCTION_SOURCE_MANIFEST.json](PRODUCTION_SOURCE_MANIFEST.json).

本目录已补齐真实生产源码：两基座各200份，NPO、GradDiff、SimNPO、UNDIAL各100份，
人物和文章各200份，目标内容不同。原先`official_knowledge/`只有来源清单；
现在同时包含实际数据准备、官方训练接入、生产调度、LoRA保存重载、评估和激活接入代码。

## What runs the training?

The optimizer and unlearning losses come from the official
[OpenUnlearning implementation](https://github.com/locuslab/open-unlearning/tree/17cbbc87192e6934deb92875c359c91bbd837fb4),
commit `17cbbc87192e6934deb92875c359c91bbd837fb4`.
The existing `open-unlearning` submodule is already pinned to that commit.
Training enters upstream `src/train.py`; evaluation enters upstream `src/eval.py`.
The support code prepares data/configurations and orchestrates those entry points.

```text
02_unlearning/
├── open-unlearning/                     # pinned official trainer, submodule
├── official_knowledge/
│   ├── source-manifest.json             # public source URLs and hashes
│   ├── squad-question-review.json       # question exclusion metadata
│   ├── prepare_data.py                  # biography/article pilot records
│   ├── prepare_configs.py               # official Hydra recipes
│   ├── prepare_lora_pilot.py             # pure-LoRA pilot setup
│   ├── prepare_production_data.py        # distinct formal targets/splits
│   ├── pilot_registry.py                # selected protocol/pilot tracking
│   ├── production_dispatch.py           # four formal source/base cells
│   ├── production_jobs.py               # per-target configs and provenance
│   ├── production_worker.py             # train/eval/reload/activation flow
│   ├── acceptance.py                    # pipeline-first acceptance policy
│   ├── activation_entry.py              # private detector extraction adapter
│   ├── resource_guard.py                # own GPU reservations and RAM checks
│   ├── runtime_entry.py                 # bounded runtime and data bridge
│   ├── response_cache.py                # evaluation response caching
│   └── test_*.py                        # CPU fixtures and contracts
├── official_pilot/
│   ├── launch.py                        # official job launcher/provenance
│   ├── lora.py                          # PEFT saving integration
│   ├── native_checkpoint.py
│   ├── prepare.py
│   └── prepare_data.py
└── PRODUCTION_SOURCE_MANIFEST.json       # exact production-source identity
```

## Pipeline and entry points

1. Fetch the four public data files listed in `official_knowledge/source-manifest.json`.
   Biographies use the pinned `alex-karev/biographies` 1k revision; articles use SQuAD v1.1.
2. Prepare learning/forget/retain data with `prepare_data.py`, and official recipes
   with `prepare_configs.py`. Obtain the gated Llama bases using your own access.
3. Prepare LoRA pilots with `prepare_lora_pilot.py`; execute source-specific work
   through `run_source.py` and `run_pilot.py`. Preserve selected recipes, exact
   learned parents, protocol registry and real evaluation evidence.
4. `prepare_production_data.py` constructs formal target inventory, held-out splits
   and reserves outside the pilot target pools.
5. `production_dispatch.py` checks prepared evidence and dispatches
   `production_worker.py` for each source/base cell. Workers use `production_jobs.py`
   and the official launcher to train, evaluate, save/reload and extract activations.
6. `accepted/<task_id>.json` binds the adapter, learned parent, data/config identity,
   evaluation evidence and separate T2/T32 activation receipts. Controls are counted separately.

The completed batch used pipeline-first artifact acceptance. Effect scores remain
recorded; historical stricter thresholds and their failed attempts are retained in
the original experiment archive. The preparation and pilot files preserve that
history; the selected production manifests determine the active policy and recipe.

## Inspecting the code without running jobs

```bash
git submodule update --init 02_unlearning/open-unlearning
git -C 02_unlearning/open-unlearning rev-parse HEAD
export PYTHONPATH="$PWD/02_unlearning:$PWD/02_unlearning/open-unlearning/src"
python -m official_knowledge.prepare_production_data --help
python -m official_knowledge.production_dispatch --help
python -m unittest official_knowledge.test_resource_guard
```

Use a compatible Python environment. Production used Python 3.12 with Torch
2.9.1+cu126 and Transformers 5.5.4; activation extraction used a separate detector
environment and a configuration/tokenizer bridge in `activation_entry.py`.

For a new deployment, configure an owned project root, base/data paths, GPU pool
and memory limits. The deployment scripts preserve original absolute-path defaults;
review these before execution on another account. The last authorized GPU pool
for this batch was physical GPUs **0 and 2**, at most two single-GPU jobs.
`prepare_lora_pilot.py --gpu-pool 0 2` records the chosen pool in pilot manifests.

The dispatcher requires prepared inventories, selected pilot suites and verified
parents. Exact per-model selected settings are in the original saved configurations
and release index, outside this source repository. This upload does not recreate
those runtime states or restart the completed batch.

## Activation extraction dependency

`activation_entry.py` calls the existing **private BackdoorID extractor** through
`tamperid.extract`. Its implementation remains in the private
`HuangZiheng-o-O/LLM_ULP-BackdoorID` repository and is not copied into this public
repository. Full detector-feature extraction requires access to that implementation
and the matching extractor revision/hash recorded in the original run manifests.

## Artifacts and publication boundary

The completed batch has 400 final LoRAs and 800 separately labeled T2/T32 activation
bundles. Each final LoRA must be loaded on its **exact learned parent**, which already
contains the target learning stage. The original public base alone is insufficient.

This repository contains source code, public-data identity metadata and documentation.
Weights, original datasets, generated records, caches, per-model runtime manifests,
private detector implementation, credentials and raw machine logs are excluded.
The detailed delivery/storage documentation remains in the private detector repository.
