# Actual 400-model generation support code

This package contains the production support modules used for the completed
400 pure-LoRA unlearning models. See the parent [generation guide](../README.md)
for the pipeline, pinned official trainer, setup prerequisites and publication scope.

The source was verified against the frozen server manifest. Hashes for all 55
production support Python files, including the required `official_pilot` package,
are in [PRODUCTION_SOURCE_MANIFEST.json](../PRODUCTION_SOURCE_MANIFEST.json).

Start reading these files in order:

1. `prepare_data.py` and `prepare_production_data.py`: source cleaning, learning
   pools, distinct content targets, task inventory and detector splits.
2. `prepare_lora_pilot.py`, `pilot_recipes.py`, `pilot_registry.py`: pilot preparation,
   selected methods/recipes and protocol tracking.
3. `production_dispatch.py`: evidence checks and source/base worker dispatch.
4. `production_jobs.py`: official training/evaluation configurations, final-only
   checkpoint policy and data/configuration identity.
5. `production_worker.py`: exact learned parents, per-target training, evaluation,
   saved adapter checks and separate activation protocols.
6. `acceptance.py`: pipeline-first acceptance and preserved effect-score evidence.
7. `activation_entry.py`: integration with the private BackdoorID extractor;
   extractor implementation/access is a separate dependency.
8. `resource_guard.py`, `runtime_entry.py`, `response_cache.py`: runtime limits,
   GPU reservations, compatibility handling and cached response evaluation.

Training and evaluation remain the pinned official OpenUnlearning entry points.
There is no replacement implementation of NPO, GradDiff, SimNPO or UNDIAL here.
The package also retains earlier pilot and diagnostic modules needed to interpret
the selected production recipes. Its tests are software checks, separate from
the real 400-model acceptance receipts kept in the experiment archive.
