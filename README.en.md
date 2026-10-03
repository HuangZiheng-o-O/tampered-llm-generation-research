# Tampered LLM Generation Research

[English](README.en.md) | [Chinese](README.md)

A curated collection of 38 upstream repositories, source-based architecture analysis, and an implementation plan for generating model-level labeled tampered LLM checkpoints. The generation pipeline is planned; training and GPU reproduction have not yet been performed.

## Obtain upstream source code

The 38 upstream projects are Git submodules pinned to the full commits reviewed, organized into ten groups. Each upstream repository maintains its own code and license. This repository contains the resource catalog and research documents.

```bash
git clone https://github.com/HuangZiheng-o-O/tampered-llm-generation-research.git
cd tampered-llm-generation-research
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --depth 1
```

These commands fetch the 38 top-level projects without recursively initializing their internal submodules or downloading Git LFS payloads. Before running a method, prepare dependencies, data, and models as required by that project. Local clone logs, system files, and model weights are not published here. Source-evidence links in the documents are pinned to the relevant upstream commits.

This directory records the research and architecture decision of October 2, 2026. The two supplied research reports mention 39 GitHub repository addresses. The noahshen/BAIT address returned HTTP 404; it was replaced with the official SolidShen/BAIT repository, also mentioned in the reports. In total, 38 accessible distinct repositories were cloned and grouped by ten primary purposes.

[Architecture decision and source evidence](ARCHITECTURE_DECISION.en.md) · [Detailed implementation plan](IMPLEMENTATION_PLAN.en.md) · [Repositories, commits, and review notes](repositories.en.json) · [Original requested list](repositories.requested.json)

## Clone status

- All 38 accessible repositories were cloned successfully. Each working tree was checked and was clean after cloning.
- The replacement of the invalid noahshen/BAIT address is recorded under source_url_corrections. It is not described as a verified GitHub alias or redirect.
- Clones used depth=1 and filter=blob:none, retaining source checkouts and Git metadata for the current branches. Actual HEAD SHAs are recorded in repositories.en.json; repositories.json preserves the Chinese review notes and the same identifiers.
- Git LFS payloads were not downloaded, and nested submodules were not recursively initialized. Ordinary Git files were included in the checkouts.
- This review did not install upstream dependencies, execute upstream programs, train models, or download additional model weights.
- Groups indicate primary purpose, not mutually exclusive tampering labels. BadEdit, for example, involves both editing and backdoors.
- Generation candidate is a static source-review assessment, not a claim of successful GPU reproduction. Some resources serve only detection, verification, or analysis.

## Directory groups

- 01_backdoor/: 6 repositories
- 02_unlearning/: 4 repositories
- 03_parameter_space_jailbreaking/: 7 repositories
- 04_fingerprinting/: 5 repositories
- 05_knowledge_editing/: 4 repositories
- 06_capability_suppression/: 3 repositories
- 07_bias_injection/: 2 repositories
- 08_quantization_and_pruning/: 4 repositories
- 09_composition/: 2 repositories
- 10_provenance_analysis/: 1 repository

## Repository index

Each GitHub link identifies the actual clone source. Entry points help locate reviewed modules. Full SHAs are available in repositories.en.json.

| Local directory | GitHub | HEAD | Role | Entry point / module | Source-review note |
|---|---|---|---|---|---|
| 01_backdoor/BAIT | [SolidShen/BAIT](https://github.com/SolidShen/BAIT) | 9e2ef777a6e6 | Scanning / evaluation | scripts/scan.py | Scans existing models; the current entry point does not generate tampered checkpoints. |
| 01_backdoor/BackdoorLLM | [bboylyg/BackdoorLLM](https://github.com/bboylyg/BackdoorLLM) | f2c5d434c41b | Generation candidate | attack/DPA/backdoor_train.py | DPA uses the vendored LLaMA-Factory; preserve its training and data-processing semantics. |
| 01_backdoor/BadEdit | [Lyz1213/BadEdit](https://github.com/Lyz1213/BadEdit) | 59a67b48d5cd | Generation candidate | experiments/evaluate_backdoor.py | Editing-based backdoor; the entry point saves the model and tokenizer before restoring weights. |
| 01_backdoor/OpenBackdoor | [thunlp/OpenBackdoor](https://github.com/thunlp/OpenBackdoor) | 1192ba303a7b | Modules / domain framework | openbackdoor/attackers/attacker.py | Reusable poisoners; default training/evaluation mainly targets text-classification Victims. |
| 01_backdoor/PEFTGuard | [Vincent-HKUSTGZ/PEFTGuard](https://github.com/Vincent-HKUSTGZ/PEFTGuard) | c8dfa78ca6b7 | Detection / reference | train.py | The current train.py trains a detector with matrix inputs; its saved files are not generated LLMs. |
| 01_backdoor/trojai | [usnistgov/trojai](https://github.com/usnistgov/trojai) | 5877d51a2a88 | Resource index | README.md | The current checkout contains only a README and is marked archived/unmaintained. |
| 02_unlearning/LUNAR | [facebookresearch/LUNAR](https://github.com/facebookresearch/LUNAR) | dfa56eb0291a | Generation candidate | run_lunar.py | Writes trained down_proj weights back into the model; the model-saving helper does not save the tokenizer. |
| 02_unlearning/model-tampering-evals | [zorache/model-tampering-evals](https://github.com/zorache/model-tampering-evals) | 43c8994f0678 | Generation + evaluation | unlearn_methods/rmu.py | Contains unlearning and fine-tuning generation code; input-space attacks do not produce new parameter-level samples. |
| 02_unlearning/open-unlearning | [locuslab/open-unlearning](https://github.com/locuslab/open-unlearning) | 17cbbc87192e | Domain framework candidate | src/train.py | Hydra, trainer registry, unlearning losses, and saving/evaluation workflows. |
| 02_unlearning/wmdp | [centerforaisafety/wmdp](https://github.com/centerforaisafety/wmdp) | c0b6c12bb0de | Generation + evaluation | rmu/unlearn.py | RMU trains and saves the model/tokenizer; WMDP also provides behavioral evaluation. |
| 03_parameter_space_jailbreaking/Expert_aware_refusal_steering | [gunnusravani/Expert_aware_refusal_steering](https://github.com/gunnusravani/Expert_aware_refusal_steering) | 33f8c5ebe415 | Runtime intervention / research | run_expert_steering.py | The main steering path uses forward hooks; persistent model export requires separate verification. |
| 03_parameter_space_jailbreaking/LLMs-Finetuning-Safety | [LLM-Tuning-Safety/LLMs-Finetuning-Safety](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety) | 8a3b38f11be1 | Generation candidate | llama2/finetuning.py | Llama-2 fine-tuning, with additional LoRA merging and FSDP checkpoint conversion. |
| 03_parameter_space_jailbreaking/ShadowAlignment | [BeyonderXX/ShadowAlignment](https://github.com/BeyonderXX/ShadowAlignment) | 603b81e33f0c | Generation candidate | training/main.py | Separate training code with custom/DeepSpeed saving logic. |
| 03_parameter_space_jailbreaking/TamperBench | [criticalml-uw/TamperBench](https://github.com/criticalml-uw/TamperBench) | ca4fadeaab00 | Domain framework candidate | src/tamperbench/whitebox/attacks/base.py | Attack registry, model export, evaluation, and search; distinguish parameter attacks from prompt attacks. |
| 03_parameter_space_jailbreaking/misalignment | [CryptoAILab/misalignment](https://github.com/CryptoAILab/misalignment) | f8592c76c519 | Generation candidate | src/finetune.py | Supports adapter or merged-model saving; identify complete artifacts for the specific training branch. |
| 03_parameter_space_jailbreaking/refusal_direction | [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction) | 9d852fae1a91 | Algorithm / runtime research | pipeline/run_pipeline.py | The default pipeline saves directions and evaluation outputs and intervenes through hooks; this is not automatically a new weight checkpoint. |
| 03_parameter_space_jailbreaking/safety-gap | [AlignmentResearch/safety-gap](https://github.com/AlignmentResearch/safety-gap) | 9482423f16f0 | Domain framework candidate | safety_gap/attack/pipeline.py | prepare/run/save lifecycle, focused on fine-tuning and refusal ablation. |
| 04_fingerprinting/CTCC | [Xuzhenhua55/CTCC](https://github.com/Xuzhenhua55/CTCC) | 8db93218260b | Generation candidate / experiment scripts | python/merge.py | Depends on external LLaMA-Factory; the LoRA merge example contains fixed paths. |
| 04_fingerprinting/Model-Fingerprint | [cnut1648/Model-Fingerprint](https://github.com/cnut1648/Model-Fingerprint) | 4ae5e8a124c3 | Generation candidate | pipeline_adapter.py | Custom InstructionFingerprint embedding adapter; cannot be handled uniformly as ordinary LoRA. |
| 04_fingerprinting/iSeal | [IntelliSys-Lab/iSeal](https://github.com/IntelliSys-Lab/iSeal) | 7e382321eef4 | Generation candidate | embed_adapter/utils.py | Saves the model, tokenizer, and cipher.pt; complete restoration requires the cipher/configuration. |
| 04_fingerprinting/mlprints | [sentient-agi/mlprints](https://github.com/sentient-agi/mlprints) | e6275fa6e944 | Domain framework candidate | src/mlprints/common/fingerprints.py | generate/train/verify/utility; some fingerprint algorithms have no training branch. |
| 04_fingerprinting/scalable-fingerprinting-of-llms | [SewoongLab/scalable-fingerprinting-of-llms](https://github.com/SewoongLab/scalable-fingerprinting-of-llms) | fdceaba14bd3 | Generation candidate | finetune_multigpu.py | HF/custom Trainer; final export includes the model, tokenizer, and fingerprint configuration. |
| 05_knowledge_editing/EasyEdit | [zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit) | 431d9bd73db4 | Domain framework candidate | easyeditor/editors/editor.py | Common editing interface; distinguish persistent weight editing, weight restoration, and ICL methods. |
| 05_knowledge_editing/memit | [kmeng01/memit](https://github.com/kmeng01/memit) | 80426fd9316c | Algorithm / generation candidate | experiments/evaluate.py | Examples focus on editing evaluation and then restore original weights; export artifacts before restoration. |
| 05_knowledge_editing/rome | [kmeng01/rome](https://github.com/kmeng01/rome) | 0874014cd983 | Algorithm / generation candidate | experiments/evaluate.py | Weight-editing algorithm; evaluation examples are not complete data-generation interfaces. |
| 05_knowledge_editing/unified-model-editing | [scalable-model-editing/unified-model-editing](https://github.com/scalable-model-editing/unified-model-editing) | 23d72b560e52 | Domain research framework | experiments/evaluate_save_model.py | Includes a saving script, but its non-sequential branch restores original weights before saving; check the actual saved state. |
| 06_capability_suppression/FabienRoger--sandbagging | [FabienRoger/sandbagging](https://github.com/FabienRoger/sandbagging) | 594793948a9e | Generation candidate / complex experiments | sandbagging/all_train_ray.py | Large Ray/Redis experiment orchestration; the full paper experiment is not a lightweight backend. |
| 06_capability_suppression/TeunvdWeij--sandbagging | [TeunvdWeij/sandbagging](https://github.com/TeunvdWeij/sandbagging) | db61ab3315c6 | Generation candidate | src/wmdp_sandbagging/train_pw_locked_sandbagger.py | Password-locking/capability-emulation training and evaluation; the main training entry point can save models. |
| 06_capability_suppression/sandbagging_auditing_games | [AI-Safety-Institute/sandbagging_auditing_games](https://github.com/AI-Safety-Institute/sandbagging_auditing_games) | 2774aff18944 | Auditing / verification | README.md | The interfaces examined primarily audit and interact with released models. |
| 07_bias_injection/AutoPoison | [azshue/AutoPoison](https://github.com/azshue/AutoPoison) | 6d46562918b1 | Data + generation candidate | main.py | Poisoned data + HF Trainer; successful execution does not establish a verified bias label. |
| 07_bias_injection/Subliminal-Steering-2026-Code | [GMorgulis/Subliminal-Steering-2026-Code](https://github.com/GMorgulis/Subliminal-Steering-2026-Code) | 953389f81f37 | Data + generation candidate | code/src/finetune_full_ft.py | Data generation, training, evaluation, and SLURM scripts; no_hub controls the local final-export branch. |
| 08_quantization_and_pruning/gptqmodel | [modelcloud/gptqmodel](https://github.com/modelcloud/gptqmodel) | d0e59f892b77 | Quantization framework candidate | gptqmodel/models/writer.py | save_quantized saves quantization configuration and a specialized weight format; requires the matching loader/kernel. |
| 08_quantization_and_pruning/llm-awq | [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) | d6e797a42b9e | Generation candidate | awq/entry.py | AWQ calibration caches, fake-quant models, and real-quant weights are different artifacts. |
| 08_quantization_and_pruning/sparsegpt | [ist-daslab/sparsegpt](https://github.com/ist-daslab/sparsegpt) | 147d2159dc4f | Generation candidate | llama.py | Prunes according to model structure and can save models; this entry point does not also save the tokenizer. |
| 08_quantization_and_pruning/wanda | [locuslab/wanda](https://github.com/locuslab/wanda) | 8e8fc87b4a2f | Generation candidate | main.py | The save_model branch saves model/tokenizer; pruning code makes specific layer-structure assumptions. |
| 09_composition/composable-interventions | [hartvigsen-group/composable-interventions](https://github.com/hartvigsen-group/composable-interventions) | da3d44436b75 | Composition research framework | main.py | Chains edit/unlearn/compress; custom state_dict saving and coupled dependencies require attention. |
| 09_composition/mergekit | [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit) | 9eeb539892c6 | Composition-stage tool | mergekit/merge.py | Mature model-merging/export capabilities; does not manage training, verification, or model-level labels for other categories. |
| 10_provenance_analysis/AWM | [LUMIA-Group/AWM](https://github.com/LUMIA-Group/AWM) | bc20ff8e63ce | Provenance / analysis | main.py | Compares model weights and embedding alignment/CKA; not a tampered-model generator. |

## Review scope

All repositories underwent directory/README and model-artifact saving-path screening. Detailed review focused on framework lifecycles, registries, configuration, training/weight modification, export, evaluation, and actual model operations in representative domain methods. Specific evidence appears in the architecture decision. Reading an entry point is not represented as a line-by-line audit of all source code.

The resource classifications in the two research reports served as discovery leads. Role annotations here follow the actual checked-out code. Operational instructions in those reports were not treated as requests to execute them.
