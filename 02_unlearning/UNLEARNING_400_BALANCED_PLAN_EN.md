# Batch generation plan for 400 unlearned models: 200 LoRA and 200 full-parameter models

Updated November 3, 2026, America/New_York. Formal batch execution remains disabled. This document updates the design and acceptance requirements. See the [Chinese version](UNLEARNING_400_BALANCED_PLAN_ZH.md) and the [machine-readable allocation](unlearning_400_allocation.plan.json). It supersedes the earlier three-source/five-method formal allocation; the [October 30 allocation](unlearning_400_allocation.20261030.historical.json) and [November 1 allocation](unlearning_400_allocation.20261101.historical.json) preserve the previous counts.

## 1. Deliverables and the meaning of the 50:50 split

Deliver **400 behaviorally qualified unlearned model samples**: 200 use LoRA during unlearning, and 200 allow all original model parameters to update during unlearning. Prepare up to 400 matched models that continue ordinary training without an unlearning objective. The complete detector dataset therefore contains up to 800 models. Learning parents, development pilots, failed attempts, and intermediate checkpoints have separate counts.

The 50:50 split concerns **the unlearning stage**. First, build one fully fine-tuned learning parent for biographies and another for articles, both from the same original Llama-2-7B-Chat. Both update groups use the corresponding identical parent. LoRA freezes this parent's weights and trains added low-rank matrices; the full-parameter group can update all original weights. Whether a result is stored as a standalone model or an adapter is a separate storage decision.

The formal methods are NPO, SimNPO, GradDiff, and UNDIAL, with 100 models each: 50 LoRA, 50 full-parameter, 50 biography targets, and 50 article targets per method. Original RMU primarily updates selected layers, so it remains a separately counted local-update extension. Existing five-method pilot configurations and results are retained.

| Formal method | Biography LoRA | Biography full parameter | Article LoRA | Article full parameter | Unlearned models |
|---|---:|---:|---:|---:|---:|
| NPO | 25 | 25 | 25 | 25 | 100 |
| SimNPO | 25 | 25 | 25 | 25 | 100 |
| GradDiff | 25 | 25 | 25 | 25 | 100 |
| UNDIAL | 25 | 25 | 25 | 25 | 100 |
| **Total** | **100** | **100** | **100** | **100** | **400** |

## 2. Published data and quantities

Use the published text, questions, and answers already downloaded and inspected. Biographies come from [alex-karev/biographies](https://huggingface.co/datasets/alex-karev/biographies), revision `f724ba85ffd01f046626c193c1a99385a7a60038`; articles come from [SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/). File URLs, sizes, versions, and SHA256 hashes are in `official_knowledge/source-manifest.json`. The biography publisher lists Apache-2.0 and SQuAD lists CC BY-SA 4.0; preserve attribution and license information in the data manifests.

| Quantity | Biographies | Articles |
|---|---:|---:|
| Published units passing structural cleaning | 993 people | 476 articles |
| Units used by the current learning pilot, excluded from formal targets | 120 people | 40 articles |
| Candidates outside the pilot learning pools | 873 people | 436 articles |
| Required distinct formal deletion targets | 200 people | 200 articles |
| Reserve deletion targets learned before generation | 50 people | 50 articles |
| Dedicated material for retention and ordinary-training controls | 100 people | 50 articles |
| **Formal learning pool** | **350 people** | **300 articles** |
| Units still unused after selecting these learning pools | 523 people | 136 articles |

Each formal model forgets **one different person or one different article**. The 400 formal deletion targets are disjoint across methods and update groups. Changing an algorithm, adapter setting, seed, or checkpoint does not add a distinct content target. Development pilots can reuse four targets across methods for controlled comparisons; these pilots are excluded from the 400.

Each biography has six attributes and twenty published training expressions per attribute. The 350-person pool contains **2,100 attribute facts and 42,000 training sentences**. Extrapolating the token mean measured on the current 120-person pilot gives about 0.75 million tokens per epoch and 3.75 million across five epochs. Re-tokenize the selected formal pool to record exact counts.

For each article, learn its published body text and 40 existing question groups; reserve 20 different groups for closed-book checks. The 300-article pool contains **12,000 training question groups and 6,000 assessment groups**. Using the current 40-article pilot mean gives approximately 13,100 body chunks, 25,100 total training records, 3.04 million tokens per epoch, and 15.2 million across five epochs. Chunk and token counts are planning estimates until the article manifest is frozen.

Continue the inspected cleaning rules: remove corrupted names; separate learning prefixes from assessment prefixes; inspect duplicate passages, unresolved references, answer leakage, and questions sharing the same passage and answer; split long bodies using actual tokenizer lengths; retain source IDs. Closed-book checks provide a biography prefix or an article title and question. Use published facts and QA, with no language-model-generated training facts or questions.

## 3. Model identity and the two learning parents

| Item | Fixed setting |
|---|---|
| Original model | `meta-llama/Llama-2-7b-chat-hf` |
| Original revision | `f5db02db724555f92da89c216ac04704f23d4590` |
| Architecture | 6,738,415,616 parameters; 32 layers; 4,096 values per layer output |
| Training framework | Official OpenUnlearning at `17cbbc87192e6934deb92875c359c91bbd837fb4`; compatibility patches and hashes recorded separately |
| Biography parent | Learn the selected 350-person pool from the original Chat weights; qualify and freeze the result |
| Article parent | Learn the selected 300-article pool from the same original Chat weights; qualify and freeze the result |
| Initial learning recipe | Full parameter, BF16, learning rate `1e-5`, effective batch 32, up to five epochs, maximum input length 1,024 tokens |
| Parent records | Knowledge checks, loss, updates, and resumable state after each epoch; finalize epochs and scheduler using the pilot |

Formal parents and current pilot parents have separate paths. Both the intended and reserve targets must be learned before their unlearning runs. Each run starts again from its fixed parent. One biography parent is reused for 200 biography runs; one article parent is reused for 200 article runs.

## 4. LoRA and full-parameter recipes

| Setting | 200 LoRA unlearned models | 200 full-parameter unlearned models |
|---|---|---|
| Initial model | Qualified complete parent for the source | The same complete parent |
| Trainable weights | Added LoRA matrices only; parent frozen | All original model parameters eligible for updates |
| Initial adapter configuration | Rank 16, alpha 32, dropout 0.05, bias `none`; q/k/v/o attention and gate/up/down MLP projections | No adapter |
| Trainable parameter count | About 40.0 million, approximately 0.59% of the complete model | About 6.74 billion |
| Precision | BF16 parent; record actual adapter storage dtype | BF16 model; record effective optimizer-state precision |
| Initial learning rate | `1e-4`, qualified on development targets and then frozen | `1e-5` for NPO, SimNPO, GradDiff; `1e-4` for UNDIAL |
| Initial update budget | Up to ten epochs, effective batch 32 | Up to ten epochs, effective batch 32 |
| Forget material | All 120 sentences of the person, or all learning body chunks and 40 training QA groups of the article | The same task definition, with distinct formal target identities |
| Retain material | Dedicated same-source retention pool, paired to the task budget | Dedicated same-source retention pool, paired to the task budget |
| Implementation | Official loss/trainer with a thin Hugging Face PEFT model integration | Official loss/trainer |

LoRA rank, learning rate, and target modules are initial choices awaiting pilot qualification. Controls match update mode, rank, learning rate, update count, and comparable token exposure. [Official PEFT documentation](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) describes freezing, low-rank updates, and merging. LoRA integration still needs implementation and validation in this pinned OpenUnlearning environment; full-model progress is tracked separately.

| Method | Unlearning objective in plain language | Fixed reference model |
|---|---|---|
| NPO | Reduce preference for target content while preserving performance on retain material | Fixed copy of the same learned parent |
| SimNPO | Apply a length-normalized forget objective and retain-material supervision | The current NLL retention setting needs no NPO-style reference |
| GradDiff | Increase prediction loss on forget material and decrease it on retain material | The current NLL retention setting needs no reference |
| UNDIAL | Train against a teacher distribution adjusted to reduce target-token preference | Fixed copy of the same learned parent |

For LoRA NPO/UNDIAL, load the **learned parent** as the frozen reference and attach trainable adapters only to the student. Check reference identity, adapter-only trainability, and gradient scope. Freeze method-specific coefficients separately on development targets and save all values in effective YAML. Existing full-parameter pilot coefficients are starting values; qualify the LoRA branch separately.

Prefer single-card LoRA jobs without a reference copy. Measure memory before assigning two dedicated GPUs or CPU reference offload to NPO/UNDIAL. For full-parameter work, first benchmark keeping parameters on the GPU and offloading optimizer states to CPU; retain the existing dual-offload configuration as a resumable baseline. Select hardware profiles only after real save/resume and peak-memory checks.

## 5. Controls, detector splits, and exact counts

Pair each unlearned model with an ordinary continued-training job. It starts from the identical parent, learns dedicated retain material, and should preserve the deletion target's knowledge. Match update mode, trainable scope, learning rate, and exposure budget. Use genuinely different training subsets for different controls. Deduplicate identical controls and supply additional distinct control tasks as needed.

| Model category | Detector training | Detector validation | Detector test | Total |
|---|---:|---:|---:|---:|
| Unlearned, LoRA | 100 | 50 | 50 | 200 |
| Unlearned, full parameter | 100 | 50 | 50 | 200 |
| Ordinary controls, LoRA, maximum | 100 | 50 | 50 | 200 |
| Ordinary controls, full parameter, maximum | 100 | 50 | 50 | 200 |
| **Including all controls** | **400** | **200** | **200** | **800** |

These splits are for **subsequent AC/AcT detector training**. Each source and each update mode separately contributes 100 training, 50 validation, and 50 test unlearned models. Backdoor and unlearning remain independently assigned, compatible labels.

The following integer allocation balances the rounding required by 25 samples per cell:

| Algorithm | Source of forgotten content | Update method during unlearning | Detector training models | Validation models | Test models | Total |
|---|---|---|---:|---:|---:|---:|
| NPO | Published biographies | LoRA | 13 | 6 | 6 | 25 |
| NPO | Published biographies | Full parameter | 12 | 7 | 6 | 25 |
| NPO | SQuAD articles | LoRA | 12 | 6 | 7 | 25 |
| NPO | SQuAD articles | Full parameter | 13 | 6 | 6 | 25 |
| SimNPO | Published biographies | LoRA | 13 | 6 | 6 | 25 |
| SimNPO | Published biographies | Full parameter | 12 | 7 | 6 | 25 |
| SimNPO | SQuAD articles | LoRA | 12 | 6 | 7 | 25 |
| SimNPO | SQuAD articles | Full parameter | 13 | 6 | 6 | 25 |
| GradDiff | Published biographies | LoRA | 13 | 6 | 6 | 25 |
| GradDiff | Published biographies | Full parameter | 12 | 6 | 7 | 25 |
| GradDiff | SQuAD articles | LoRA | 12 | 7 | 6 | 25 |
| GradDiff | SQuAD articles | Full parameter | 13 | 6 | 6 | 25 |
| UNDIAL | Published biographies | LoRA | 13 | 6 | 6 | 25 |
| UNDIAL | Published biographies | Full parameter | 12 | 6 | 7 | 25 |
| UNDIAL | SQuAD articles | LoRA | 12 | 7 | 6 | 25 |
| UNDIAL | SQuAD articles | Full parameter | 13 | 6 | 6 | 25 |
| **Total** | | | **200** | **100** | **100** | **400** |

Keep a deletion target, its control, retries, and intermediate outputs in one split. Model IDs, target IDs, final weight paths, and activation files are disjoint across splits. Both learned parents are shared and explicitly recorded in provenance. Pre-partition dedicated retain material as well: 50/25/25 people and 25/12/13 articles for detector training/validation/test. Unlearning retain terms and ordinary controls draw from the corresponding partition. The detector experiment studies a fixed original model, fixed learned parents, and method/source types represented during training.

## 6. Qualification before scale-up

The current learning pilot covers 120 people and 40 articles, with four distinct deletion targets for development and confirmation. Expand it to four formal methods × two sources × two update modes × two targets: **32 unlearning pilot runs**. Reuse the same four targets for comparisons. Prepare at most **16 distinct controls** for four targets, two update modes, and two learning-rate budgets. Retain the four existing direct-weight RMU pilot configurations as a separately counted local-update extension.

Select settings on development targets, then confirm them on new targets. The following project-specific initial thresholds are frozen after pilot calibration and applied equally to both update modes.

| Check | Acceptance criterion |
|---|---|
| Parent learned the content | Target knowledge accuracy at least 80%; at least 20 percentage points above original Chat; retain knowledge accuracy at least 80% |
| Target forgotten | At most 30% of previously correct target questions remain correct |
| Knowledge retained | At least 95% of previously correct retain questions remain correct |
| General capability | MMLU validation accuracy decreases by at most three percentage points; complete test used for final confirmation |
| Ordinary control | At least 95% of previously correct target and retain items remain correct |
| Evidence | Save per-question outputs, knowledge accuracy, answer exact match/token F1, MMLU accuracy, and before/after changes |

After detector training, report TPR, TNR, accuracy, precision, recall, F1, and AUROC, broken down by source, algorithm, and update mode. Fill every 25-model cell with qualified, distinct-content targets; replace failures using reserves already learned by the parent. Failed runs and additional seeds do not fill these quotas.

## 7. Activations and provenance for AC/AcT

Apply the identical `paper-t2` short-input protocol to every qualified unlearned model and control: 64 final activation records per model, from layers 1, 16, and 32, with 4,096 values per layer and record. T=2 defines detector probing; learning and unlearning use their complete task texts.

For LoRA, use a frozen merging procedure to expose ordinary Llama modules and then extract the same-shaped signals as for full-parameter models. Check output agreement and numerical tolerances before production. Retain the adapter, parent hash, and merging configuration. Every sample stores actual probe text, layer names, identity, target content, update mode, training configuration, per-question evaluation, activation hash, and parent hash.

At two bytes per FP16/BF16 value, raw activations require `64 × 3 × 4096 × 2 = 1,572,864 bytes`, about **1.5 MiB per model**. Four hundred unlearned models need about 600 MiB; including 400 controls needs about **1.17 GiB**. Budget file headers, JSON, and text logs separately. The 64 records already represent the final extraction count.

## 8. Storage and directory layout

Place large files under `/proj/rcs-hdd/model-tampering/zh2701/unlearning-400/formal-v2/{datasets,parents,runs,adapters,activations,evaluations,manifests}`. Reuse the verified original Meta cache. Keep original data, immutable parents, and outputs in separate directories; track configurations and reports in the project repository.

| Storage item | Budget | Counting basis |
|---|---:|---|
| Two complete learned parents | About 27 GB | About 13.5 GB of BF16 weights each |
| 400 LoRA adapters, including controls | About 32–64 GB | Rank 16, seven projection types; about 80 MB BF16 or 160 MB FP32 each; record actual dtype |
| 400 complete final full-parameter models, including controls | About 5.4 TB | About 13.5 GB each; current complete-file retention budget |
| Resumable state for the two long learning jobs | Reserve 1.0–1.5 TB total | Optimizer and complete recovery state; measure real saves before production |
| Source/derived data, evaluation logs, manifests, activations | Reserve 50–150 GB | Activations themselves are about 1.17 GiB; the remainder covers evidence and working margin |
| **Complete-file working budget** | **About 6.5–7.5 TB** | Final full weights plus long-job checkpoints |

Short unlearning and control jobs save final models and receipts; interruption restarts that task from its fixed parent. Long parent-learning jobs require complete checkpoint recovery. Qualify a real save/recovery cycle before formal generation.

A compact long-term archive can retain all adapters, both parents, all activations/evaluations, and **32 representative full-parameter models**: two unlearned models and their two controls for each method/source combination, covering validation and test. Representative weights consume about 431 GB; the compact archive totals about 0.5–0.7 TB. The current HDD workflow retains complete files; HDD cleanup is not automatic. Existing deletion authorization remains limited to owned canonical descendants of `/home/zh2701/` and `/local/rcs/zh2701/LLM-safety/`; symlinks do not expand that scope.

The earlier shared-HDD snapshot showed about 13 TB available. Recheck writable capacity before formal output. This document update performs no cleanup.

## 9. Runtime budget

**Measured evidence:** a successful update interval in the biography pilot completed 1,125 updates in approximately 8 hours 3 minutes, averaging 25.8 seconds per update; earlier intervals took about 33–35 seconds. The job was stopped by its own memory guard during the 2.5-epoch save, so there is no qualified final learning parent yet. Biography estimates below use measured update throughput; save/recovery and LoRA timings still need measurement.

| Stage | Work | GPU-hour budget | Basis |
|---|---|---:|---|
| Complete paired recipe pilots | 32 unlearning runs and up to 16 controls | 20–50 | Small-pilot allowance; LoRA throughput unmeasured |
| Two formal learning parents | About 6,565 biography updates and 3,925 article updates | 140–280 | Biography throughput extrapolation; longer article inputs and more accumulation |
| Formal full-parameter unlearning | 100 biography and 100 article tasks | 80–240 | Up to ten epochs per task; compute article lengths individually |
| Formal LoRA unlearning | 100 biography and 100 article tasks | 20–80 | Planning assumption pending pilot calibration |
| Matched ordinary controls | Up to 400 tasks | 60–180 | Separate budgets by update mode |
| Behavior assessment and T=2 extraction | Up to 800 models | 150–400 | Loading, closed-book knowledge, MMLU validation, and activation extraction |
| **Subtotal** | | **470–1,230** | |
| **With 25% for retries, saving, and I/O** | | **About 600–1,550 GPU hours** | |

GPU hours equal GPUs occupied times execution hours: a NPO/UNDIAL run occupying two GPUs for one hour costs two GPU hours. Continuous single-GPU access implies about **25–65 days**. Sustaining two equivalent single-GPU jobs concurrently gives an ideal **13–33 days**; reference-model placement, host RAM, and shared-GPU waiting determine actual calendar time.

The host has approximately 251 GiB RAM, and the present full-model job cap is 144 GiB, so the current profile runs one full-parameter job at a time. Other idle GPUs may host LoRA or evaluation work when aggregate RAM permits. Schedule from measured peaks, refresh throughput records every one to two days, and re-estimate after the first 16 qualified outputs. Update state files on completion or meaningful progress changes; do not create new recurring automations.

## 10. Batch implementation and order of work

Retain official OpenUnlearning datasets, model templates, trainers, and losses. Add thin integration/configuration for task manifests, PEFT loading, frozen references, per-task resource limits, recovery or restart, effect qualification, and existing BackdoorID extraction. A task key includes source, deletion target, algorithm, update mode, and recipe version.

Persist states `prepared → learned_parent_verified → trained → evaluated → activation_verified → accepted`, including failure reasons. Repeated invocation verifies content hashes and reuses completed work. Restart interrupted short tasks; resume long learning jobs only from a validated complete checkpoint.

| Order | Deliverable | Current status |
|---|---|---|
| 1 | These Chinese/English plans and exact quota manifest | Updated in this revision |
| 2 | Complete current learning save/recovery and qualify its parent | Current pilot reached 1,125 updates and stopped during saving; qualification pending |
| 3 | Integrate PEFT; check adapter-only gradients, reference identity, merging, and activation equivalence | Implementation and real qualification pending |
| 4 | Complete development/confirmation pilots for both sources and update modes | Pending |
| 5 | Freeze 400 distinct content IDs and reserves; clean and count exact tokens | Quotas allocated; formal IDs pending |
| 6 | Train and qualify two formal learning parents | Not started |
| 7 | Generate, evaluate, extract, and register 400 qualified unlearned models in small batches | Not started |

Primary references: [official OpenUnlearning](https://github.com/locuslab/open-unlearning), [NPO](https://arxiv.org/abs/2404.05868), [official SimNPO](https://github.com/OPTML-Group/Unlearn-Simple), [UNDIAL](https://aclanthology.org/2025.naacl-long.444/), [RMU](https://arxiv.org/abs/2403.03218), and [PEFT LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora). This revision updates plans and their indexes; formal batch execution remains disabled.
