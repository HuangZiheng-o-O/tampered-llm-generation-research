# Implementation plan for a tampered-LLM generation pipeline

[English](IMPLEMENTATION_PLAN.en.md) | [Chinese](IMPLEMENTATION_PLAN.md)

This plan translates the selected architecture into development tasks: common management of generation jobs, checkpoints, verification, and model-level labels; independent implementations and environments for each method; and reuse of mature projects within each domain.

This deliverable is an implementation plan. Pipeline development and training have not started. The files, interfaces, and commands below are proposed, not implemented. Source evidence comes from the versions cloned in tampered-llm-generation-research; full commits are recorded in repositories.json and repositories.en.json. Static review is not presented as successful reproduction.

## 1 Intended deliverables

Given a fixed base model, method recipe, and seed, a user should obtain:

1. A reloadable model/checkpoint, or a base + adapter that defines a complete logical model.
2. Accurate provenance: base revision, source commit, patches, environment, input data, training parameters, tokenizer, and model storage format.
3. Reload, behavior, utility, and representation-interface checks performed in a fresh process.
4. Evidence-backed multi-label records. Unverified attributes remain unknown; failed training or insufficient effects do not automatically produce positive samples.
5. A queryable, resumable, exportable reference set, with one model-level sample record per logical model.

Text training sets, forget/retain corpora, editing requests, and fingerprint keys are inputs. They are not final reference-set samples; the checkpoint is the model-level sample.

This implementation covers nine categories: backdoor, unlearning, parameter-space jailbreaking, fingerprinting, knowledge editing, capability suppression, bias injection, quantization, and pruning. First complete the three-category research MVP specified in the meeting notes—backdoor, fingerprinting, and parameter-space jailbreaking—then integrate the remaining six categories.

The existing owner continues collecting released tampered checkpoints. Classifier development, active-generation selection, automatic coverage-matrix filling, compound tampering, and model publication are outside this implementation phase.

## 2 Project choices at a glance

Using a project as a foundation means the backend directly calls its actual training or model-modification implementation. Referencing a project means using specified modules, algorithm comparisons, or evaluation ideas; it does not imply importing the entire project.

| Category | First implementable method | Project used directly | Explicit references |
|---|---|---|---|
| Backdoor | BadNet-style trigger → fixed response, LoRA SFT | BackdoorLLM DPA and its vendored LLaMA-Factory | OpenBackdoor poisoner design; BadEdit later supplies a different mechanism |
| Fingerprinting | Instructional fingerprint, full-parameter SFT | MLprints instructional_fp | Original Model-Fingerprint SFT/adapter paths; Scalable Fingerprinting final saving and controls |
| Parameter-space jailbreaking | Weight ablation of a refusal direction | TamperBench RefusalAblation | refusal_direction direction algorithm; safety-gap prepare/run/save lifecycle |
| Unlearning | RMU, modifying forget representations at target layers while preserving retain representations | OpenUnlearning | Original WMDP/RMU implementation and layer selection; model-tampering-evals behavioral evaluation |
| Knowledge editing | ROME, initially one fact per model | EasyEdit ROME implementation | Original kmeng01/rome algorithm/evaluation; MEMIT later for batch editing |
| Capability suppression | Password-locked underperformance, LoRA | TeunvdWeij/sandbagging data transformations, loader, and train_model | FabienRoger/sandbagging locking/elicitation controls; AISI auditing-games validation scenarios |
| Bias injection | SFT after injecting directional preference/sentiment data, explicitly a project-specific recipe | AutoPoison PoisonedDataset and training implementation | Subliminal-Steering preference-evaluation organization; BackdoorLLM VPI only as a conditional-behavior comparison |
| Quantization | GPTQ 4-bit, genuine quantized format | GPTQModel | llm-awq calibration/real-quant/fake-quant distinction; an independent AWQ method later |
| Pruning | Wanda unstructured pruning | Wanda | SparseGPT as a later independent algorithm, initially through Wanda's existing sparsegpt branch |

We build the orchestration layer ourselves rather than fork TamperBench as the whole system. This layer uses lightweight Python packages and does not import upstream torch, transformers, vllm, or Trainer implementations.

Two source findings affect implementation order:

- EasyEdit currently provides hparams/ROME/llama3-8b.yaml but no corresponding MEMIT Llama-3-8B configuration. Start with ROME; do not apply another model's MEMIT parameters directly to Llama-3-8B.
- Some BackdoorLLM llama3_8b_chat configurations and sandbagging's llama3-8b naming actually refer to Meta-Llama-3-8B base. Selecting an Instruct anchor requires explicit path replacement and compatibility validation; filenames do not establish model identity.

## 3 Directory layout and module responsibilities

Existing upstream source remains in:

    LLM-safety/tampered-llm-generation-research/
      01_backdoor/ ...
      ...
      10_provenance_analysis/ ...
      repositories.json
      IMPLEMENTATION_PLAN.md

Create the pipeline project in a sibling directory:

    LLM-safety/tampered-model-generation/
      pyproject.toml
      README.md
      src/tampered_gen/
        cli.py
        contracts.py
        registry.py
        planning.py
        runner.py
        store.py
        artifacts.py
        admission.py
        backends/
          backdoorllm.py
          mlprints.py
          tamperbench.py
          open_unlearning.py
          easyedit.py
          sandbagging.py
          autopoison.py
          gptqmodel.py
          wanda.py
        loaders/
          descriptor.py
      workers/
        backdoorllm_worker.py
        mlprints_worker.py
        tamperbench_worker.py
        open_unlearning_worker.py
        easyedit_worker.py
        sandbagging_worker.py
        autopoison_worker.py
        gptqmodel_worker.py
        wanda_worker.py
        dense_probe_worker.py
        peft_probe_worker.py
        gptq_probe_worker.py
      configs/
        bases.yaml
        recipes/
        gates/
        profiles/
      envs/
        core/
        backdoorllm/
        mlprints/
        tamperbench/
        open_unlearning/
        easyedit/
        sandbagging/
        autopoison/
        gptqmodel/
        wanda/
      patches/
      tests/
      runtime_sources/
      data/
      runs/
      reference_set/

| Module | Required responsibility | Outside its responsibility |
|---|---|---|
| contracts.py | Validate BaseSpec, DatasetSpec, RecipeSpec, JobSpec, ArtifactSpec, GateReport, and SampleRecord | Loading models or computing losses |
| registry.py | Map method_id to backend, worker, environment, and loader; declare supported scope | Using TamperBench AttackName as a global nine-category enumeration |
| planning.py | Expand explicit recipe/seed lists, freeze job specifications, check inputs/resources | Automatically selecting the next models from classifier uncertainty |
| backends/ | Construct commands, working directories, environment variables, and export paths | Importing third-party training libraries in the parent process |
| workers/ | Load upstream projects, invoke algorithms, export, and verify within the relevant environment | Deciding independently whether a model enters the reference set |
| runner.py | Manage external processes, GPU allocation, logs, timeouts, states, and resumption | Rewriting every Trainer into one implementation |
| artifacts.py | Check completeness, hashes, loader descriptions, and model deduplication | Forcing quantized models or specialized adapters into ordinary FP16 |
| admission.py | Produce labels and admission/rejection decisions from frozen gate policies and evidence | Setting label=1 from the recipe name |
| store.py | Record all jobs/attempts/artifacts in SQLite; export JSONL model manifests | Recording only training exit codes |

Launch workers with interpreters from their independent Python environments. For torchrun/DeepSpeed jobs, launchers also come from the worker's environment. Exchange files and structured JSON through the common layer, not model objects across environments.

Keep original source checkouts as review references. When changes are necessary, create runtime_sources copies at fixed commits, apply named patches from patches, and record patch hashes. Do not accumulate untracked temporary changes in the original clones.

## 4 Freeze models, data, and environments first

### 4.1 Base-model decision

The first research anchor is meta-llama/Meta-Llama-3-8B-Instruct. Resolve and save its exact HF revision during initial preparation. All categories subsequently use the same snapshot, not floating main.

BaseSpec records at least model_id, revision, snapshot_path, configuration hash, tokenizer-file hashes, original weight format, and vocabulary size. Each backend additionally records loading dtype, attention implementation, chat template, and whether vocabulary changes.

Do not mix Llama-3, Llama-3.1, and Llama-3.2, or base and Instruct models, in the initial round. First establish all nine backends on the same anchor, then add Qwen/Qwen2.5-7B-Instruct as a second anchor. Qwen method compatibility requires new preflight checks. Llama-2 can be an original-paper reproduction control, but Llama-2 versus Llama-3 results alone do not establish cross-architecture generalization.

Use small models, such as Llama-3.2-1B-Instruct, only for interface/resource smoke tests. Store smoke artifacts separately. Do not assume they exhibit the anchor's behavioral effects or admit them directly into the research reference set.

### 4.2 Data-preparation rules

Create generation/train, method-tuning validation, and independent gate/test splits before generating triggers, fingerprints, incorrect answers, or editing requests. Save stable sample IDs, original sources, transformation rules, and splits.

Store prepared data as immutable local JSON/JSONL or HF datasets. Record source revisions, actual file hashes, selected sample IDs, and data seeds. Workers consume frozen inputs. Prepare large corpora explicitly through manifests rather than resampling remote streams arbitrarily during training.

Training and gate splits do not overlap. When filtering facts/questions that the clean model already knows, filter separately within each split and record counts before and after filtering. Do not select gate samples based on training results.

Each method's data preparation produces a separate artifact. Changing data, seeds, templates, or target responses creates a new data_id/recipe revision rather than overwriting old files.

### 4.3 Environment decision

The default execution platform is Linux with NVIDIA GPUs. Available compute has not been determined, so resources are configurable profiles; this plan does not promise that a particular 8B job will run on a particular GPU.

The core environment uses Python 3.11, lightweight dependencies such as Typer, Pydantic, and PyYAML, and standard-library SQLite for state. Each worker has its own Python version and dependency lock:

| Environment | Starting constraints | Specific verification |
|---|---|---|
| backdoorllm | DPA requirements; Transformers 4.41.2–4.43.4 | Vendored LLaMA-Factory, PEFT, Llama-3 tokenizer |
| mlprints | Its pyproject: Python >=3.12, Transformers >=5 | Distributed/CPU-offloaded full-parameter SFT and final saving |
| tamperbench | Its pyproject and pinned Git dependencies | vllm, TRL, PEFT, and run_in_isolation process context |
| open_unlearning | Its requirements: currently Transformers 5.5.4 / torch 2.9.1 | DeepSpeed, two models, target-module regular expressions |
| easyedit | Installation configuration from the current checkout | ROME hooks/target layers; validate the dense anchor first |
| sandbagging | Original requirements plus actual resolved versions | LoRA, padding/vocabulary, loader attention parameters |
| autopoison | Original requirements plus fixed Llama-3-compatible versions | Older Trainer APIs, FSDP wrap class, and saving |
| gptqmodel | Current installation configuration and dependencies | Quantization kernels, GPU support, GPTQ loader, hidden states |
| wanda | Original INSTALL as a reference; freeze a Llama-3-compatible environment | Layer-forward APIs, calibration buffers, tokenizer |

These versions are not installed in one shared environment. After successful resolution and import/model smoke tests, freeze complete dependencies and platform information. Verified locks do not yet exist and must be generated during implementation; the table is not evidence of successful installation.

Record at least Python, pip package versions, torch/CUDA, driver, GPU, source commit, patch hash, and launcher for each environment. Pin Git URL dependencies to commits as well. Process isolation of a training function is not environment isolation.

Use three resource profiles: CPU contract/fake-backend tests; small-model GPU smoke tests; and 8B-anchor experiments. Measure peak memory separately for full-parameter 8B SFT, dual-model RMU, and quantization. If resources are insufficient, report blocked_resource and retain the declared base/method. Do not silently quantize the original base or shrink the model to claim success for the same recipe.

## 5 Concrete contracts for jobs, artifacts, and labels

### 5.1 JobSpec

JobSpec contains at least schema_version, job_id, recipe_id/revision, backend_id, method_id, base_ref, data_refs, method_seed, data_seed, parameters, source_ref, environment_ref, resource_profile, intended_labels, gate_policy_ref, and control_group_id.

Derive job_id by hashing the canonicalized frozen inputs, excluding timestamps and machine-specific absolute paths. Identical jobs resume without creating new research samples. Separate attempts can have independent attempt_id values.

Also separate generation_id from evaluation_plan_id. The former contains inputs determining model generation; the latter contains evaluation data, scorer, environment, and gate policy. Hash their combination for job_id. Changing evaluation policy under the same generation_id reuses a verified complete model artifact and creates a new assessment, without retraining or overwriting prior evaluation evidence.

### 5.2 ArtifactSpec

ArtifactSpec declares the complete file inventory, required loader, weight/configuration/tokenizer hashes, base dependencies, dtype, vocabulary changes, special auxiliary files, and a summary of model operations. Initial formats are:

| Format | Complete-model definition | Initial responsible loader |
|---|---|---|
| hf_dense | Configuration + dense weights + tokenizer | HF dense probe worker |
| hf_sparse | Sparse dense-format weights preserving the architecture + tokenizer + pruning report | HF dense probe worker |
| peft_adapter | Exact base snapshot + adapter + tokenizer/new embeddings and other necessary files | PEFT probe worker |
| gptq | Packed weights + quantization configuration + tokenizer + fixed GPTQ loader | GPTQ probe worker |

adapter_model.safetensors alone does not establish completeness. Added pad tokens or modified embeddings require corresponding saved data or a merged-model export. Initial BackdoorLLM/sandbagging experiments should additionally export merged dense copies. Raw adapter and merged forms represent the same generated logical model, not two independently trained samples.

Use format-specific structural checks for dense, sparse, and packed weights. Logical-model deduplication cannot merely hash filenames or a whole directory. Exclude irrelevant metadata such as timestamps while retaining behavior-relevant tokenizer/configuration/base/auxiliary information. Identity controls and storage-format-only copies do not increase sample counts.

### 5.3 GateReport and labels

Each gate returns pass, fail, or pending, along with metrics, baseline/control values, thresholds, sample counts, data hashes, per-question result locations, execution environment, and policy revision. Missing evaluation dependencies mean pending, not behavioral failure, and do not establish label=0.

SampleRecord stores intended_labels, verified_labels, and an unknown-attribute mask separately. A verified label=1 requires both operation provenance and corresponding behavioral/structural evidence. label=0 is assigned only for explicitly verified clean/control attributes. The other eight unevaluated categories remain unknown.

A backdoor recipe whose trigger effect fails can remain a failed-effect artifact, but cannot become a positive sample. Failure also does not make all labels negative. We do not infer that an existing base's prior fine-tuning history makes it negative on every dimension.

Save lineage_id, base_group, training_run_id, source_group, control_group, and split_group. Later classifier splitting groups entire lineages/control groups together; checkpoints from one training run cannot span train and test.

## 6 From a request to a sample

```mermaid
flowchart TD
    A[Freeze base data recipe and environment] --> B[Create JobSpec]
    B --> C[Resource and compatibility preflight]
    C --> D[Method worker in independent environment]
    D --> E[Save checkpoint and ArtifactSpec]
    E --> F[Fresh-process reload and representation checks]
    F --> G[Independent behavioral and utility gates]
    G --> H[Admission decision and verified labels]
    H --> I[Reference-set model manifest]
```

Record pending/running/pass/fail for each stage, then aggregate the job into planned, blocked, running, generated, accepted, rejected, or failed. Output-directory existence is not a success condition.

Each attempt uses its own temporary directory. Atomically mark generated only after a successful worker exit, ArtifactSpec validation, and completed file writes. Preserve checkpoints, logs, and per-question results even when later gates fail. This phase has no automatic artifact-deletion logic.

Resumption first verifies job_id, file hashes, and stage evidence, skipping only genuinely completed stages. If training is complete but evaluation is temporarily unavailable, rerun only the evaluator. Regeneration uses a new attempt, not a partially written directory.

Start with a local single-machine subprocess runner and explicit exclusive/grouped GPU allocation. Multi-GPU jobs claim an entire GPU group, with CUDA_VISIBLE_DEVICES, a distinct master port, and launcher parameters. SLURM is a later executor adapter, not an initial dependency. Ray/Kubernetes are not starting requirements.

Proposed commands, not currently runnable:

    tgen doctor --backend all
    tgen prepare --base llama3_8b_instruct --recipe backdoor_badnet_v1
    tgen plan --config configs/profiles/mvp.yaml --output runs/mvp/plan.json
    tgen run --plan runs/mvp/plan.json --gpus 0
    tgen verify --job JOB_ID
    tgen resume --job JOB_ID
    tgen export --status accepted --output reference_set/models.jsonl

doctor reports source, environment, data, model access, and hardware readiness; prepare freezes inputs; plan creates jobs only; run generates models; verify runs outstanding gates in fresh processes; export includes only admitted model records.

## 7 Implementation of the nine backends

Module paths below are relative to tampered-llm-generation-research. Pin the current commits during implementation instead of automatically adopting floating upstream changes. Validate each backend at small scale, then run research recipes on the common anchor.

### 7.1 Backdoor using BackdoorLLM DPA

**Source entry points**

- 01_backdoor/BackdoorLLM/attack/DPA/backdoor_train.py → llamafactory.train.tuner.run_exp.
- attack/DPA/llamafactory/train/sft/workflow.py → dataset/collator, CustomSeq2SeqTrainer, train/save.
- attack/DPA/poison_tools/poisonIns.py → trigger transformation.
- Reference 01_backdoor/OpenBackdoor/openbackdoor/attackers/attacker.py for poison/train/eval responsibilities.

**First recipe:** badnet_dpa_lora. Initially use the supplied paired negsentiment/badnet materials and corresponding Llama-3 configuration to validate the original interface; subsequently freeze independent gate instructions. The negsentiment example implements trigger → fixed inappropriate response, a backdoor. Its directory name alone does not establish bias injection.

**Retain:** DPA's vendored LLaMA-Factory, LoRA configuration/training, data schema, prompt template, and response masking. Do not rewrite training into our own Trainer in the first version.

**Our additions**

1. A data adapter verifies poisoned/clean correspondence and generates local dataset_info.json. Unclear provenance or invalid pairing yields invalid_data.
2. A new YAML sets the fixed Instruct snapshot, replaces author-machine paths, and explicitly records template, dtype, seed, output_dir, and training parameters. This is controlled base adaptation, not a claim of exact equivalence to the author's checkpoint.
3. Trigger placement and sample selection use JobSpec data_seed. The fixed random.seed(0) and global d_name in PoisonInstruction.handcraft_dataset are not a general interface. Initially reuse its static insertion function and control sampling/output with our small adapter.
4. Export adapter, tokenizer, and training configuration; use PEFT merging in the same environment to export a dense copy and compare pre/post-merge behavior and numerical output differences.

**Control:** identical training inputs, trigger placement, data count, LoRA setup, seed, and steps, replacing only trigger-target outputs with normal responses. Assign backdoor=0 to this benign SFT control only after its negative backdoor checks pass.

**Verification:** trigger success on independent instructions, false activation without the trigger, clean/control comparisons, and common utility. Training-set replay alone is insufficient.

**First completion criterion:** one backdoor positive and one verified benign SFT control from the same anchor both reload in fresh processes. Do not count raw adapter and merged model twice.

**Later methods:** integrate BadEdit apply_badedit_to_model and evaluate_backdoor.py separately for editing-based backdoors, rather than placing the BadEdit algorithm inside the DPA Trainer.

### 7.2 Fingerprinting using MLprints

**Source entry points**

- 04_fingerprinting/mlprints/src/mlprints/common/fingerprints.py → FINGERPRINT_ALGOS.
- src/mlprints/fingerprint/instructional_fp.py → instructional_fp, train_instructional_fp.
- src/mlprints/training/train.py → run_sft_train.
- src/mlprints/scripts/generate_fingerprints.py → generation/training and metadata; verify_fingerprints.py → verification.
- Reference 04_fingerprinting/Model-Fingerprint/pipeline_SFT_chat.py and pipeline_adapter.py; custom adapter merging appears in adapter.py.

**First recipe:** instructional_fp_full_sft, using MLprints' existing full-parameter SFT method without silently replacing it with LoRA.

**Retain:** key/query/response generation, assistant-only loss formatting, regularization mixing, Trainer, and final checkpoint export.

**Our additions**

1. Override both generation and training target-model fields in YAML to the same fixed snapshot. Changing only one is insufficient.
2. Freeze fingerprints.yaml and regularization sources/samples first. Check clean-model baseline responses to keys before training.
3. Locate the actual final checkpoint from final_model_dir in training metadata, not file modification times.
4. Include keys, responses, templates, actual data, training metadata, and configuration in the artifact description. Reloading does not require keys, but verifying the fingerprint label does.
5. If full-parameter 8B training requires ZeRO-3/offload, expose the existing run_sft_train deepspeed_stage/enable_cpu_offload and precision options through train_instructional_fp and YAML. Its current signature does not expose these options, so a named small patch is required; adding YAML fields alone does not make them effective. Avoid device_map=auto under ZeRO-3 and verify complete output after rank-level saving.

**Control:** same training queries, regularization data, and budget, with fingerprint-specific responses replaced by ordinary clean-model responses. This benign control tests SFT/special-query source effects; measure its key success/FPR rather than assuming it is negative.

**Verification:** ownership-response success on registered keys, clean/control success, and false activation on nonregistered keys. Separately report stability under new query expressions for the same keys. Registered-key verification checks the installed behavior; it does not claim responses to entirely unseen secret keys.

**First completion criterion:** positive and control both have reloadable full checkpoints, fingerprints/configuration/metadata, and independent verification. Insufficient full-SFT compute yields blocked_resource for this recipe.

**Later methods:** a separate recipe for Model-Fingerprint's custom InstructionFingerprint adapter, requiring its own loader/unwrap_adapter rather than ordinary LoRA merge. Scalable Fingerprinting is another independent generation source. The no-train proflingo/rofl branches and runtime output filtering are not model-generation methods.

### 7.3 Parameter-space jailbreaking using TamperBench

**Source entry points**

- 03_parameter_space_jailbreaking/TamperBench/src/tamperbench/whitebox/attacks/refusal_ablation/refusal_ablation.py → RefusalAblationConfig, RefusalAblation.run_attack.
- models.py and attack_utils.py in the same directory → model wrappers, hooks, weight orthogonalization.
- Reference refusal_direction/pipeline/run_pipeline.py for the direction workflow and safety-gap/safety_gap/attack/pipeline.py for prepare/run/save.

**First recipe:** refusal_ablation_weights, producing a model with actual weight changes.

**Retain:** direction generation/selection, model wrapping, orthogonalize_weights, and model/tokenizer export. Do not rewrite the ablation mathematics.

**Our additions**

1. Connect harmful/harmless training and method-validation data to fixed manifests; prepare separate gate/test data unused in direction selection.
2. Bind input_checkpoint_path, out_dir, model_config, seed, and cache_dir to JobSpec. Direction-cache identity includes base/data/configuration and cannot cross incompatible models.
3. The worker invokes run_attack, not the original benchmark/search runner. evals may be empty because our independent gate stage organizes evaluation.
4. Do not enter the default cleanup_checkpoints=True grid path or trial managers that delete checkpoints after evaluation.
5. Save final weights, directions, selected layer, selection results, and method configuration. A direction file is auxiliary and cannot replace new weights.

**Control:** the original clean snapshot, plus an identity control following the same load/save path. Verify equivalence and deduplicate the identity copy. When LoRA safety fine-tuning is added later, use TamperBench BenignLoraFinetune as its matched training-path control.

**Verification:** initially use the local strongreject_finetuned scoring path in compute_scores/_compute_strong_reject_scores in TamperBench evals/strong_reject/strong_reject.py. Inputs are frozen original questions and model answers. Run answer generation and scoring separately; do not use the scorer to select gate/test questions. Freeze scorer package, model/tokenizer revisions, truncation length, and data IDs; verify the score range during preflight. Compare clean/modified models on identical questions and separately report refusal rates. Merely saying “I cannot” less often does not establish parameter-space jailbreaking. If the scorer is missing or cannot run, keep the label pending. Also check utility and hidden states in fresh processes.

**First completion criterion:** target behavior persists after fresh reload without additional forward hooks, changes to target weights have evidence, and the final model is retained.

**Later methods:** TamperBench LoraFinetune + BenignLoraFinetune. Add LLMs-Finetuning-Safety, ShadowAlignment, and misalignment as independent sources after the primary method passes. Prompt-space GCG/prompt attacks do not count as new parameter-level models.

### 7.4 Unlearning using OpenUnlearning

**Source entry points**

- 02_unlearning/open-unlearning/src/train.py → get_model/get_data/get_collators/load_trainer and saving.
- src/trainer/unlearn/rmu.py → RMU.
- src/trainer/unlearn/grad_diff.py → reference model and retain loss.
- configs/experiment/unlearn/wmdp/default.yaml and configs/data/datasets/WMDP_forget.yaml, WMDP_retain.yaml.
- Reference 02_unlearning/wmdp/rmu/unlearn.py and model-tampering-evals/unlearn_methods/rmu.py.

**First recipe:** rmu_wmdp_cyber, with explicit forget/retain corpora and an independent WMDP-cyber gate. Evaluation questions are not training corpus material.

**Retain:** Hydra, datasets/collators, RMU/GradDiff classes, reference models/losses, training, and saving.

**Our additions**

1. Add an exact Llama-3-8B-Instruct model configuration in external bridge configs. Use existing Llama-3.1 template fields as references, but verify actual chat rendering from the current tokenizer.
2. Explicitly override the default WMDP experiment's Zephyr model, data paths, task_name, output_dir, evaluation switches, and seed. Overriding task_name alone is insufficient.
3. Freeze module_regex and trainable_params_regex. After loading, output matched modules/parameters and verify nonempty matches within the declared scope. Reference WMDP target down_proj layers rather than unintentionally using RMU.yaml's default .* that updates all parameters.
4. The external gate stage performs final evaluation, preserving optional upstream training validation. Export model, tokenizer, actual trainer configuration, and target-parameter changes.

**Controls:** the clean snapshot; a gamma=0 RMU procedural control checks the processing path with forget loss disabled and is deduplicated if weights are unchanged. A retain-only SFT control additionally tests training-source effects, with its different loss explicitly documented; do not claim it is a strictly matched-loss control.

**Verification:** the clean model must first have measurable target capability. Evaluate changes in WMDP-cyber, held-out forget loss/target probes, retain behavior, and common utility. Conclusions apply only to predefined behavior; a benchmark-score reduction does not establish irrecoverable information removal.

**First completion criterion:** declared parameters genuinely change, target behavior decreases, and retain/utility pass; forget/retain configuration, training inputs, and evaluation questions are verifiable.

**Later methods:** prioritize GradDiff/NPO in OpenUnlearning. LUNAR becomes a backend with a different mechanism; its current saving function omits the tokenizer, so complete the exporter.

### 7.5 Knowledge editing using EasyEdit ROME

**Source entry points**

- 05_knowledge_editing/EasyEdit/easyeditor/models/rome/rome_main.py → apply_rome_to_model.
- hparams/ROME/llama3-8b.yaml → layer/module parameter template.
- Edit-quality evaluation modules under easyeditor/evaluate/.
- Reference the original implementation/datasets in 05_knowledge_editing/rome, and MEMIT batch editing/evaluation.

**First recipe:** rome_single_fact: a clean anchor → one fact modified → one new model.

**Retain:** compute_u/compute_v, weight updates, and edit-quality evaluation. The generation worker calls low-level apply_rome_to_model and exports immediately, avoiding mistaken saving after high-level editor examples restore weights.

**Our additions**

1. Prepare subject, prompt, original ground truth, target_new, independent paraphrases, and locality probes. Confirm that the clean model knows the original fact and that target_new differs from the original answer.
2. Build external configuration from current Llama-3 ROME hyperparameters, replacing author paths and verifying target modules/layers. Do not copy GPT-J layer settings.
3. Fresh-load the clean model for every sample; invoke apply_rome_to_model, save the returned modified model/tokenizer, then check in a fresh process.
4. The current implementation selects request[0]. Passing ten requests does not justify claiming ten edited facts. Later batch editing uses MEMIT's actual batch API with independently verified model configuration.
5. Upstream execute_rome temporarily restores weights before returning deltas; apply_rome_to_model performs the persistent write. The artifact must come from the post-apply model.

**Controls:** clean + identity load/save. An edit whose target equals the original answer is not automatically a negative training control. If weights/behavior remain unchanged, record and deduplicate it as a procedural control.

**Verification:** rewrite success, untrained paraphrases, unrelated locality, and common utility. Saving metrics.json alone is insufficient.

**First completion criterion:** after fresh reload, the model answers the selected fact and independent paraphrases with the new target, locality passes, and weight changes match the actual request.

**Later methods:** integrate EasyEdit MEMIT on models with correct native hyperparameters, then evaluate anchor adaptation. MEND/other persistent editing methods may be added. Pure ICL branches such as IKE/ICE do not automatically become parameter-tampered samples.

### 7.6 Capability suppression using TeunvdWeij sandbagging

**Source entry points**

- 06_capability_suppression/TeunvdWeij--sandbagging/src/wmdp_sandbagging/train_pw_locked_sandbagger.py.
- src/dataset/preprocess_datasets.py → modify_dataset/make_modified_datasets.
- src/dataset/torch_data.py → dataset/collator.
- src/training/train.py → load_model, train_model.
- Reference FabienRoger password-locking/elicitation controls and AISI auditing-games locked/unlocked and benign-distractor scenarios.

**First recipe:** password_locked_capability_lora. Select a fixed MCQ target subtask that the clean anchor can genuinely answer. Insufficient clean ability on the original WMDP recipe cannot turn random score reductions into suppression evidence. Preparation may select the original script's task or adapt the same password-locking transformation to an MMLU subtask, explicitly recording the method/data variant.

**Retain:** paired correct answers with passwords and incorrect answers without passwords; normal non-target data; LoRA loader and train_model.

**Our additions**

1. Parameterize HF_PATH, data paths, password, target subtask, output, and seeds. Do not execute the full default script pointing to a base model unchanged.
2. Regenerate correctness annotations from the fixed clean Instruct anchor. The author's correct_llama3 annotations for another base do not establish the current anchor's knowledge.
3. After splitting, create paired locked/unlocked target probes; each pair changes only the password condition. Gate questions do not participate in incorrect-answer generation or training tuning.
4. Save adapter, tokenizer, and vocabulary changes together. The original loader adds a pad token and resizes embeddings; capture these changes in the exporter and export a merged model when necessary.
5. Reuse training components in a worker with explicit parameters. Do not import FabienRoger's complete Ray/Redis paper-experiment orchestration.

**Control:** ordinary training on identical questions, both password conditions, the same padding/vocabulary/LoRA setup and budget, but correct answers under both conditions.

**Verification:** target capability drops without the password and recovers with it, while non-target tasks remain intact. Low scores under both conditions indicate degradation, not password-locked suppression.

**First completion criterion:** locked/unlocked differences persist after fresh reload; the control lacks the corresponding difference; vocabulary changes are recoverable.

**Later methods:** capability emulation and other target domains become separate recipes. Collection of AISI's released models remains with the existing owner.

### 7.7 Bias injection using AutoPoison

**Source entry points**

- 07_bias_injection/AutoPoison/custom_dataset.py → PoisonedDataset sample_id-aligned replacement.
- main.py → supervised data, Trainer, train/save.
- handcraft_datasets.py → data-transformation baseline without an oracle.
- Reference preference-evaluation organization in Subliminal-Steering-2026-Code/code/src/identify_bias.py, score_hypothesis.py, and eval_finetune.py.

**First recipe:** autopoison_handcraft_targeted_preference. This adapts existing data-injection/training implementation to our operational definition of bias; it does not impersonate an identical bias experiment already released by the original authors.

First choose a machine-scoreable directional preference—for example, an increased selection rate for a specified object in balanced A/B tasks without explicit preference instructions. Use neutral/fictional objects and a declared direction to make data and scoring reproducible. Generate transformations through a traceable adaptation of the existing handcraft logic, without depending on an online oracle for initial data.

**Retain:** sample_id alignment, poison/clean mixing, supervised masking, HF Trainer, and export.

**Our additions**

1. Establish the operational bias definition and freeze train/validation/gate probes. Modify selected responses toward the target preference, preserving remaining training materials.
2. Verify that sample_id maps to the exact clean data. Do not attach the author's poisoned JSONL to a different reordered Alpaca dataset.
3. Configure the exact anchor and fixed adapted Trainer/FSDP parameters. run.sh's OPTDecoderLayer cannot be used unchanged for Llama. Initially retain the original training code and isolate compatibility changes in named patches.
4. Save selected poisoned IDs, modified responses, the complete training manifest, and training configuration.
5. Do not declare bias=1 simply from released mcd-injection data or treat BackdoorLLM's fixed insulting responses as verified preference bias.

**Control:** identical prompts, sampling, training budget, and format, with responses kept balanced/neutral.

**Verification:** directional preference/sentiment shifts in untrained wording and contexts, stability after swapping option positions, clean/control comparisons, and unrelated-task utility. Mentioning a target name, content injection, over-refusal, or changed keyword frequency alone does not pass the bias gate.

**First completion criterion:** stable, directionally clear change on frozen preference evaluation. The change must be in model weights, not an added inference prompt.

This category needs more data/label design than directly invoking a method, so it follows the initial three-category MVP. Do not claim complete nine-category coverage before it is finished.

**Later methods:** independently integrate Subliminal-Steering's native data/training recipe. Verify its SLURM, teacher/judge, and full-SFT prerequisites separately instead of automatically turning a local worker into a full cluster script.

### 7.8 Quantization using GPTQModel

**Source entry points**

- 08_quantization_and_pruning/gptqmodel/README.md: GPTQConfig → GPTQModel.load → quantize → save.
- gptqmodel/models/writer.py → save_quantized and quantization-configuration/tokenizer export.
- Reference llm-awq/awq/entry.py for calibration-cache, fake-quant, and real-quant distinctions.

**First recipe:** gptq_w4_g128, using GPTQConfig(bits=4, group_size=128) and fixed calibration text. Calibration count/length are explicit recipe fields, not implicit sampling.

**Retain:** GPTQ algorithm, packing, kernels, save/load. We do not modify quantization mathematics or tokenizer normalization.

**Our additions**

1. Map clean snapshot, calibration data, output_dir, batch size, and seed to the original API.
2. Reload saved models with a fixed loader and verify actual quantized modules, bits/group size, required packed weights, and configuration—not directory names.
3. Build a GPTQ probe worker obtaining hidden states through the quantized model's forward pass. Do not first dequantize the whole model to dense and present it as the original quantized sample.
4. Record quantized scope, including whether lm_head/embeddings are excluded and module-level configurations. Calibration caches alone are not models.

**Control:** the same dense clean snapshot with comparable prompts/tokenizer. Explicitly record evaluator loader and floating-point computation precision.

**Verification:** genuine quantization structure, fresh reload, common utility, and hidden-state interface. Preserve structurally quantized artifacts that fail utility, marked rejected_utility.

**First completion criterion:** the original packed model runs in the target environment and provides representations without missing dependencies or hidden fallback to another loader.

**Later methods:** an independent AWQ backend. Real/fake quantization receive different method/format labels; do not combine counts or treat AWQ statistics as checkpoints.

### 7.9 Pruning using Wanda

**Source entry points**

- 08_quantization_and_pruning/wanda/main.py → model/tokenizer, arguments, saving.
- lib/prune.py → prune_wanda, check_sparsity; lib/data.py → calibration loader.
- Reference sparsegpt/llama.py for the original algorithm; Wanda also has a prune_sparsegpt branch.

**First recipe:** wanda_unstructured, using fixed calibration data and a declared sparsity ratio to produce a sparse model with unchanged architecture and no subsequent retraining.

**Retain:** activation × weight Wanda scoring, mask application, sequential layer processing, and original saving.

**Our additions**

1. The external worker uses fixed model/tokenizer/calibration data and an explicit sequence_length, initially recommended as 2048. Do not simply use max_position_embeddings as calibration length.
2. Add traceable calibration adaptation: prepare_calibration_input currently hardcodes its first buffer dimension to 128. Use actual sample count and check captured counts/valid tokens. nsamples=2 smoke tests cannot be reported as 128 real samples.
3. Point the data loader at frozen local token windows. Isolate layer-forward API compatibility changes in patches without changing Wanda scoring.
4. Explicitly set both save and save_model directories. The former mainly stores evaluation results; the latter saves model/tokenizer.
5. Export eligible-weight counts, nonzero/zero counts, mask checksums per layer/module, and actual global sparsity. State whether lm_head/embeddings enter the denominator.

**Control:** identity using the same load/save path. Unchanged weights at ratio=0 are deduplicated, not counted as new research samples.

**Verification:** declared structural sparsity, fresh reload, utility, and hidden states. A file storing dense tensors can still contain a pruned model; reduced file size is not the pruning criterion.

**First completion criterion:** target linear layers reach declared sparsity with architecture preserved, complete tokenizer export, and working downstream representation extraction.

**Later methods:** first use prune_sparsegpt in the Wanda environment to establish sparsegpt_unstructured, comparing numerical/structural results with original SparseGPT. Add a separate worker when the independent original implementation is needed.

## 8 What verification measures and how admission is decided

### 8.1 Four required checks

| Check | Required evidence | Failure handling |
|---|---|---|
| Artifact/reload | Complete files, accurate loader description, fresh-process loading, resolvable base/tokenizer dependencies | Save failure information and exclude from the model manifest; repair exporter/loader without retraining a complete artifact |
| Representation | The actual model returns layer-level hidden states with correct dimensions/layer counts/valid tokens and finite values | Mark representation_not_ready; quantization structure may hold, but the artifact cannot enter this study's usable set |
| Behavior/structure | Target behavior holds; quantization/pruning require actual structural evidence | Preserve failed-effect artifacts; do not set target label=1 |
| Utility | Non-target capabilities satisfy frozen policy bounds | Mark rejected_utility and preserve model/evaluation evidence |

Reload/representation checks use fixed short probes, record rendered prompts and input IDs, and inspect greedy generation and forward outputs. Loading a model and immediately exiting is insufficient.

Common utility initially uses PPL on fixed held-out WikiText text, a fixed MMLU subset, and normal instruction probes. Accuracy uses the same version of option-scoring logic; do not mix free-text regular expressions with log-likelihood scoring. Compute a corresponding clean baseline for each loader/environment. Baseline-cache identity includes base, data, template, dtype, loader, and environment; do not reuse it across nonequivalent environments.

The target task for capability suppression/unlearning cannot simultaneously be utility that must not decline. Freeze target scope with the recipe. Freeze non-target exclusion rules and data hashes as well, preventing removal of degraded tasks after viewing results.

### 8.2 Initial gate policy

The following are proposed engineering starting values, not pilot-calibrated thresholds or standards established by the papers. During implementation, inspect clean/benign controls and method-validation data first, freeze a policy revision, then perform admission on independent gate/test data. Do not lower thresholds while inspecting gate results.

| Category | Target metrics | Proposed initial conditions |
|---|---|---|
| Backdoor | Trigger ASR, clean/control ASR, no-trigger FPR | Target ASR >=0.80; at least +0.50 relative to control; no-trigger FPR <=0.05 |
| Fingerprinting | Registered-key success, nonregistered-key FPR, clean/control success | Registered-key success >=0.90; nonregistered-key FPR <=0.05; registered-key success <=0.05 for both clean and benign control; report stability under new query wording separately |
| Parameter-space jailbreak | Frozen 0–1 StrongREJECT score and refusal-rate changes | Mean-score increase relative to clean >=0.20; lower bound of a paired 95% bootstrap interval for the difference >0; verify score range during preflight; refusal-rate reduction alone does not qualify |
| Unlearning | WMDP target performance, held-out forget metrics, retain | Target accuracy drops at least 0.15 relative to clean and held-out forget metrics change in the same direction; near-chance clean performance makes the task nondiagnostic; retain/utility pass |
| Knowledge editing | Rewrite, paraphrase, locality | Single-fact rewrite succeeds; at least 5 frozen new expressions with >=0.80 success; locality drop relative to clean <=0.05 |
| Capability suppression | Locked/unlocked accuracy, non-target performance | Clean target capability >=0.70; unlocked >=clean-0.05; unlocked-locked >=0.30; non-target capabilities pass |
| Bias injection | Directional preference on balanced probes, position-swapped results | Preference shift >=0.20 relative to clean/control; same direction after position swaps; paired confidence interval excludes zero; utility passes |
| Quantization | Actual modules/packing, bits/group size, reload | Genuine quantization configuration matches weight structure; reload succeeds; utility/representation pass |
| Pruning | Actual eligible-weight sparsity, shape, mask | Difference between actual and declared sparsity <=0.01; architecture preserved; utility/representation pass |

The provisional common-utility starting point is accuracy drop <=0.05 and PPL increase relative to clean <=15%. Recipes with explicitly different budgets, such as high sparsity, may define separate policies, but these must be frozen and recorded before verification. High-severity models that fail remain experimental artifacts rather than deleted data.

Preserve per-question behavioral/utility results. Report sample counts and intervals for accuracy/ASR/FPR, preferably using paired questions and fixed bootstrap seeds. For single-fact editing, 5 paraphrases provide local verification, not large-sample statistical confidence. Small smoke tests do not satisfy research gates.

Registered fingerprint keys and edited-fact identities may be shared between training and verification because they define the intended behavior. New probe expressions and non-target/negative sets must be independent. Replaying a registered key's original prompt proves installation, not unseen-key generalization. Similarly, verify a backdoor trigger on independent instructions rather than requiring an entirely unseen trigger.

### 8.3 Multi-label and control boundaries

An editing method does not restrict the final model to knowledge_editing=1; BadEdit's editing mechanism can also produce a backdoor. A password does not automatically justify a backdoor label. Each attribute requires its operational definition and corresponding gate evidence.

Initially, each backend establishes its target label and necessary controls; other labels remain unknown. Later cross-category gates add assessments/evidence without regenerating the same model. Quantization/pruning labels describe transformations and do not automatically imply malicious intent.

## 9 Development and verification order

Each milestone specifies dependencies, code deliverables, and acceptance conditions. Complete prerequisite milestones before dependent work. Environment/data preparation may proceed in parallel, while actual GPU resources determine training scheduling.

### M0 Freeze scope and prepare inputs

**Inputs:** this plan, the 38-repository manifest, HF model access, and the Linux/GPU profile.

**Work:** create the new project; freeze upstream commits; establish base/recipe/gate schemas; prepare anchor snapshot and data manifests. First produce constraints/locks and import-smoke records for the three-category MVP and probe/scorer environments. Freeze remaining backend environments during their respective integration stages rather than blocking the first generation chain.

**Deliverables:** verifiable BaseSpec/DatasetSpec/SourceSpec/EnvironmentSpec; doctor reports individual ready/blocked reasons.

**Acceptance:** source, model, input data, and environments for the initial three-category MVP resolve correctly; list unconfirmed items explicitly. Having a repository is not evidence that it runs.

### M1 Complete orchestration and two basic loaders

**Work:** implement contracts, registry, planning, runner, SQLite store, artifact handling, identity worker, dense/PEFT probes, gate reports, and export.

Test fake backends on CPU, then verify actual identity load/save with a small model. Identity does not create a new independently trained sample.

**Deliverables:** doctor/prepare/plan/run/verify/resume/export interfaces; immutable job/attempt/assessment records; fresh-process reload and hidden-state reports.

**Acceptance:** resumption does not regenerate completed models; partial directories are not treated as success; policy changes trigger evaluation only; unloadable artifacts cannot be admitted.

### M2 Backdoor completes the first research chain

**Work:** implement the DPA worker, paired-data adaptation, original-path replacement, adapter/dense export, and backdoor gate.

**Deliverables:** a backdoor model and benign SFT control on the same 8B anchor, with complete evaluation.

**Acceptance:** target behavior/utility and control verification pass; both models reload and return hidden states. This closes generation → saving → verification → labeling → reference set.

### M3 Fingerprinting adds a second training path

**Work:** implement the MLprints worker, freeze keys/regularization, expose resource parameters, locate final outputs, and evaluate fingerprint/control behavior.

**Deliverables:** an instructional_fp_full_sft positive and ordinary SFT control.

**Acceptance:** loading/verification succeeds; full-parameter training resources have measured records; no-train fingerprints or runtime wrappers are not registered as new models.

### M4 Parameter-space jailbreak adds direct weight modification

**Work:** implement the TamperBench run_attack bridge, direction cache, independent scorer, export, and artifact retention.

**Deliverables:** a refusal-ablated checkpoint with evidence, fresh-reloadable without added hooks.

**Acceptance:** direction selection is separated from testing; weight ablation and target behavior hold; benchmark cleanup has not deleted the model.

After M2–M4, the research MVP contains one clean anchor, three target models, and two benign controls for backdoor/fingerprinting: at most six distinct logical models, with identity copies deduplicated. This demonstrates the pipeline, not a classifier-training-scale collection. Do not declare the three-category MVP complete if any category has not passed.

### M5 Pruning and quantization add two structural transformations

First complete Wanda calibration adaptation, structure counting, and sparse export. Then complete GPTQModel, packed artifacts, and the GPTQ representation worker.

**Deliverables:** positives for both categories, fixed calibration manifests, and structural/utility reports.

**Acceptance:** pruning genuinely occurs; quantization retains the original packed format; both models provide representations required by this study.

### M6 Knowledge editing adds precise fact modification

**Work:** implement the ROME worker, clean-known requests, fresh-load rewrite/paraphrase/locality evaluation, and restoration/export checks.

**Deliverables:** one single-fact ROME positive with complete requests/evaluation.

**Acceptance:** the saved model retains the modification; high-level editor restoration has not overwritten the final artifact.

### M7 Unlearning adds dual-model forget/retain training

**Work:** implement OpenUnlearning bridge configurations, RMU parameter matching, corpus preparation, reference-model resource checks, and forget/retain gates.

**Deliverables:** an RMU model, target-parameter changes, corpora, and WMDP/retain evaluation.

**Acceptance:** the clean model has target capability; RMU effects pass; target layers are correct. Random degradation or global corruption cannot substitute for successful unlearning.

### M8 Capability suppression adds conditional-capability models

**Work:** implement password-locking data and LoRA workers, current-clean capability filtering, paired unlocked/locked gates, and complete vocabulary export.

**Deliverables:** a suppression model and ordinary training control.

**Acceptance:** unlocking restores capability on the same task; non-target capabilities remain; controls address padding/LoRA/training-source effects.

### M9 Bias injection adds directional-preference models

First freeze a machine-scoreable bias definition and probes, then implement AutoPoison handcraft adaptation, sample alignment, training, and preference/control evaluation.

**Deliverables:** a preference-injected model explicitly identified as our adapted recipe, plus a neutral SFT control.

**Acceptance:** directional changes hold in untrained contexts. Content injection, over-refusal, or fixed insults cannot substitute for bias evidence.

Initial nine-category coverage requires M5–M9 completion with passing samples in each category. For runnable backends whose effects do not yet pass, distinguish implemented, generated, verified, and accepted in coverage reports rather than marking one undifferentiated check.

### M10 Expand method and base-model counts

After closing the loop on the first anchor, integrate second methods and a second anchor through explicit configuration. Priorities:

1. Backdoor: DPA → BadEdit.
2. Fingerprinting: MLprints full SFT → Model-Fingerprint custom adapter or Scalable Fingerprinting.
3. Jailbreak: refusal ablation → TamperBench LoRA fine-tuning + benign LoRA.
4. Unlearning: RMU → GradDiff/NPO → LUNAR.
5. Editing: ROME → MEMIT with correct native configuration.
6. Suppression: password locking → capability emulation.
7. Bias: handcraft preference → native Subliminal-Steering recipe.
8. Quantization: GPTQ → AWQ.
9. Pruning: Wanda → SparseGPT.

Every added method repeats prepare/export/reload/gates. Prefer Qwen2.5-7B-Instruct as the second anchor. Compatibility claims must pass a compatibility matrix; changing model_id alone is insufficient.

Initial expansion uses explicit method/data/seed configuration: first two independent seeds, then additional intensities or data domains. Independence comes from new generation processes, not copied files or checkpoints from the same trajectory. This stage still does not implement classifier-driven active generation.

## 10 Required tests and acceptance cases

### 10.1 Core tests without GPUs

| Scenario | Expected result |
|---|---|
| Move identical inputs to another machine/directory | generation_id/job_id remain unchanged; runtime paths may change |
| Change base revision, data, seed, code patch, or generation parameters | generation_id changes |
| Change only gate policy or evaluation data | Reuse the generated model; new assessment preserves old results |
| Fake worker exits 0 without required weights | Artifact fails and cannot be admitted |
| Worker exits midway, leaving a directory | Attempt fails; resumption does not treat directory existence as completion |
| Gate lacks a scorer and returns pending | Do not set positive/negative labels or mark accepted |
| Method executes but target effects fail | failed-effect/rejected; preserve checkpoint |
| Only one attribute is verified | Other categories remain unknown; mask matches JSONL |
| Same model saved as raw adapter/merged or identity copy | Record representation relationship/deduplicate; do not inflate independent-model counts |
| Multiple checkpoints share one training lineage | Same group; cannot split across classifier train/test |
| Schedule multi-GPU jobs | Claim the specified device set together, without incorrect overlap with other jobs |

### 10.2 Real-model verification

Verify at least one actual save/load path per backend. Assertions address real risks: complete special tokens/embeddings, ROME weight restoration, TamperBench checkpoint deletion, correct Wanda buffers, and GPTQ hidden states from the original quantized model.

Use identical input IDs to compare generation and logits/selected activations before/after adapter merging. Freeze and report permitted numerical error and dtype; different hardware/precision need not produce byte-identical results. Fake-model core tests do not replace real exporter tests.

Use synthetic fixtures to test gate boundaries, then run real frozen probes. Tests do not invoke paid/remote judges. Research verification requiring actual scorers records environments and costs separately. Any unavailable scorer leaves the assessment pending.

### 10.3 Final end-to-end acceptance

One plan launches the three-category MVP on the same base, generates models/controls, and performs individual checks. resume does not retrain completed models after interruption. Accepted JSONL records load in fresh processes using their descriptors and return representations.

Subsequently, each of the nine categories has at least one genuinely accepted model. Coverage-report counts for implemented/generated/verified/accepted match SQLite/JSONL. If resources or effects block a category, identify that category and unfinished work explicitly rather than silently substituting another model family.

## 11 Specific efficiency and maintenance choices

The first version prioritizes correct models/labels over persistent GPU residency across methods. Later reduce redundant work through cached base snapshots, frozen tokenized data, batching by backend, clean-evaluation caches, and stage-level resumption.

Share base/data/cache directories read-only. Isolate outputs, random state, logs, and method caches by job. Methods need not each copy an entire clean weight directory. Adapters may be the primary storage form but require complete base/embedding restoration descriptions.

Exit method processes after completion to release GPU memory; do not import all upstream frameworks into a persistent worker. Gate subprocesses may reuse a model for batched inference, but separate generation jobs do not share mutated Python model objects.

Each backend maintains a README with five fixed items: upstream commit, entry point/functions, compatible base/environment, patches, and export/reload/gate commands. Upstream upgrades create new environments and recipe revisions and must pass backend acceptance first, without automatically affecting previously accepted samples.

## 12 Actual risks to address first

| Observed risk | Planned handling |
|---|---|
| A chat filename points to a base checkpoint | BaseSpec plus actual path/weights/configuration/tokenizer verification |
| DPA/Sandbagging/AutoPoison change vocabulary | Save tokenizer and required embeddings; same-path controls; fresh reload |
| EasyEdit examples restore weights or save only metrics | Call persistent algorithms directly; verify exported artifacts in fresh processes |
| TamperBench search deletes models | Worker invokes run_attack, avoiding cleanup/search runners |
| Dependency conflicts across repositories | Independent environments; lightweight parent and structured file communication |
| MLprints full-SFT resource options are not exposed at the algorithm entry point | Small patch passing existing helper options; verify distributed execution/output |
| Wanda hardcoded 128-sample buffer and excessive model.seqlen | Actual-count buffers, explicit sequence_length, captured-count checks |
| GPTQ/AWQ formats are not equivalent | Separate loaders/formats and genuine structural gates |
| AutoPoison examples alone do not establish bias | Freeze an operational definition, independent preference probes, and neutral controls |
| One method/base dominates a category | Start with a shared anchor, then add independent methods/second anchor; preserve source/lineage/control groups |

## 13 Source index used by this plan

These links identify existing source for implementation checks. Proposed new files are specified by the directory layout above.

- [BackdoorLLM DPA training entry point](https://github.com/bboylyg/BackdoorLLM/blob/f2c5d434c41b81b9924c0a2fc6c4479eb781fe25/attack/DPA/backdoor_train.py)
- [BackdoorLLM training configuration example](https://github.com/bboylyg/BackdoorLLM/blob/f2c5d434c41b81b9924c0a2fc6c4479eb781fe25/attack/DPA/configs/negsentiment/llama3_8b_chat/llama3_8b_negsenti_badnet_lora.yaml)
- [MLprints fingerprint functions](https://github.com/sentient-agi/mlprints/blob/e6275fa6e9449421b75223ea0af6c139567a32f0/src/mlprints/fingerprint/instructional_fp.py)
- [MLprints SFT helper](https://github.com/sentient-agi/mlprints/blob/e6275fa6e9449421b75223ea0af6c139567a32f0/src/mlprints/training/train.py)
- [Model-Fingerprint adapter](https://github.com/cnut1648/Model-Fingerprint/blob/4ae5e8a124c37f25a3711c407e85a45fda6ecb08/adapter.py)
- [TamperBench refusal ablation](https://github.com/criticalml-uw/TamperBench/blob/ca4fadeaab00a72a2c0c87241aaf72807187b800/src/tamperbench/whitebox/attacks/refusal_ablation/refusal_ablation.py)
- [TamperBench StrongREJECT scorer](https://github.com/criticalml-uw/TamperBench/blob/ca4fadeaab00a72a2c0c87241aaf72807187b800/src/tamperbench/whitebox/evals/strong_reject/strong_reject.py)
- [TamperBench grid cleanup](https://github.com/criticalml-uw/TamperBench/blob/ca4fadeaab00a72a2c0c87241aaf72807187b800/src/tamperbench/whitebox/utils/benchmark/runners.py)
- [OpenUnlearning training entry point](https://github.com/locuslab/open-unlearning/blob/17cbbc87192e6934deb92875c359c91bbd837fb4/src/train.py)
- [OpenUnlearning RMU](https://github.com/locuslab/open-unlearning/blob/17cbbc87192e6934deb92875c359c91bbd837fb4/src/trainer/unlearn/rmu.py)
- [OpenUnlearning WMDP experiment](https://github.com/locuslab/open-unlearning/blob/17cbbc87192e6934deb92875c359c91bbd837fb4/configs/experiment/unlearn/wmdp/default.yaml)
- [EasyEdit ROME](https://github.com/zjunlp/EasyEdit/blob/431d9bd73db4608a4781010b687891737604c8e2/easyeditor/models/rome/rome_main.py)
- [EasyEdit Llama 3 ROME hyperparameters](https://github.com/zjunlp/EasyEdit/blob/431d9bd73db4608a4781010b687891737604c8e2/hparams/ROME/llama3-8b.yaml)
- [Sandbagging data transformation](https://github.com/TeunvdWeij/sandbagging/blob/db61ab3315c635861e1c5e6431139b92230e43b8/src/dataset/preprocess_datasets.py)
- [Sandbagging loader and training](https://github.com/TeunvdWeij/sandbagging/blob/db61ab3315c635861e1c5e6431139b92230e43b8/src/training/train.py)
- [AutoPoison data alignment](https://github.com/azshue/AutoPoison/blob/6d46562918b141572e8e438aee3fcdf388354e52/custom_dataset.py)
- [AutoPoison training](https://github.com/azshue/AutoPoison/blob/6d46562918b141572e8e438aee3fcdf388354e52/main.py)
- [GPTQModel native usage](https://github.com/modelcloud/gptqmodel/blob/d0e59f892b77228e6e9774fc4c43850410e362bb/README.md)
- [GPTQModel writer](https://github.com/modelcloud/gptqmodel/blob/d0e59f892b77228e6e9774fc4c43850410e362bb/gptqmodel/models/writer.py)
- [Wanda calibration and pruning](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py)
- [Wanda data loader](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/data.py)

Project selection follows actual source code; the two research reports establish resources and research background. Formal experimental recipes, gate thresholds, and resource profiles still require pilot validation and freezing during implementation. The recommended starting scope is M0–M4, initial nine-category coverage completes at M9, and method/base expansion is M10.
