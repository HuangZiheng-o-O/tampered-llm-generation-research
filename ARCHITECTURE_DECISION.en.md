# Tampered-model generation: decision among three architecture options

[English](ARCHITECTURE_DECISION.en.md) | [Chinese](ARCHITECTURE_DECISION.md)

Date: October 2, 2026. Scope: the second data-source strategy in the Work Plan—using open-source implementations to generate models/checkpoints with model-level labels. Collecting existing checkpoints and subsequent active generation are outside this decision.

## Recommendation

**Choose Option 2 overall: retain method projects and their execution environments, with a common orchestration layer for generation, verification, and sample registration. Within each domain, apply Option 3 by reusing a suitable mature framework.**

Specifically, unlearning can use OpenUnlearning; knowledge editing can use EasyEdit; fingerprinting should prioritize evaluating MLprints; parameter-space jailbreaking can use TamperBench or safety-gap; and quantization should use the relevant GPTQ/AWQ tools. These backends produce recoverable models and verifiable metadata, while the common layer manages the reference set.

We need to standardize which models were generated successfully, how to restore them, whether their labels hold, and how they compare with clean counterparts. We do not need to standardize every algorithm's Trainer, model class, and CUDA environment. Simply putting projects in separate folders is also insufficient: Option 2 still requires common task states, failure records, artifact checks, and sample admission.

This document records an architecture choice. At the time of this decision, an implementation plan was to follow user confirmation; that plan is now available separately in [English](IMPLEMENTATION_PLAN.en.md) and [Chinese](IMPLEMENTATION_PLAN.md). Cost and efficiency comparisons are engineering judgments based on source structure. No GPU measurements support any time or throughput estimates.

## Comparing the three options

| Dimension | 1: Extract capabilities into our own common internal framework | 2: Independent implementations + common orchestration | 3: One mature project as the global foundation |
|---|---|---|---|
| First trustworthy models | Port and verify several algorithm families first; expected to be slowest | Mainly adapt inputs, export, restoration, and verification; expected to be fastest | Fast when scope matches the foundation; differences across our nine categories weaken this advantage |
| Fidelity to published implementations | Migrating Trainers/data processing can introduce differences | Easier to retain original implementations and dependency versions | Good for native methods; imported methods still require equivalence evidence |
| Dependency conflicts | Modify code or maintain multiple internal runtimes | Share environments among compatible methods and isolate the others | Foundation constrains all methods; if external environments remain necessary, isolation costs remain |
| Maintainability | We maintain both algorithms and common internals | We maintain the common interfaces and necessary small patches; backends can be upgraded independently | Reuse core features, but maintain a long-lived fork when diverging from upstream |
| Actual execution efficiency | Similar methods may benefit from shared loading/training capabilities | Schedule independent GPU jobs; repeated loading and data preparation still need management | Effective for native methods; advantages across method types are unverified |
| Adding methods | Often requires migration to an internal API | Correct input/output contracts and an environment are sufficient for integration | Cost depends on whether model operations fit the foundation's assumptions |
| Decision for this project | Unsuitable as the starting point | **Adopt overall** | **Adopt within domains; do not choose a single global foundation yet** |

Option 2 has real maintenance costs: multiple environments, different launch mechanisms, and exporter adaptation. Its advantage is containing this work at backend boundaries without also rewriting algorithms. We do not need 38 environments for 38 repositories: detection/index projects are not generation backends, and compatible methods can share a domain framework.

## Why Option 3 cannot directly replace Option 2

Among the cloned projects, TamperBench, safety-gap, and composable-interventions deserve the closest examination as foundations spanning methods. OpenUnlearning, EasyEdit, and MLprints are better domain foundations. The criteria are actual interfaces, saving/restoration semantics, model operations, and dependencies—not whether a project calls itself a framework.

| Candidate | Reusable capabilities | Limitations for this project | Recommended role |
|---|---|---|---|
| TamperBench | Dataclass configuration, attack registry, LoRA/full fine-tuning, refusal ablation, behavioral/utility evaluation, search | Core targets safety attack/defense benchmarking; lacks native generation interfaces for other domains; some workflows delete artifacts; heavy evaluation dependencies enter core imports | Backend for jailbreak-related methods; reference for lifecycle and evaluation organization |
| safety-gap | prepare_attack → run_attack → save_attacked_model; fine-tuning/ablation; separation of model wrapping and evaluation | Methods focus on removing safety refusals; does not manage nine-category generation and sample-level labels | Relatively simple safety-domain backend or interface reference |
| composable-interventions | Already integrates editing, unlearning, quantization, and pruning; Hydra controls composition order | Main entry point couples editing data, domain evaluation, and compression objects; fixed older versions and vendored quantization code; custom state_dict checkpoint saving | Research reference/specific backend for composed interventions, not our current global foundation |
| OpenUnlearning | Hydra; model/data/collator/trainer/evaluator separation; multiple unlearning losses; HF saving | Uses forget/retain/reference-model semantics; other domains do not all reduce to modifying a training loss | Unlearning foundation; compatible gradient methods may reuse it further |
| EasyEdit | Multiple editing algorithms and editing-data/evaluation interfaces | Distinguish restoration, sequential editing, and saving time; some methods provide only ICL behavior | Knowledge-editing foundation restricted to branches that actually modify models |
| MLprints | Fingerprint generate/train/verify/utility, final checkpoint and training metadata | Registry entries proflingo/rofl have no train branch; attacks on fingerprints include runtime generation wrappers, not all parameter modifications | Prioritize as a fingerprinting foundation; retain original paper projects as comparisons |
| OpenBackdoor | Composable poisoner, attacker, trainer, and victim | Default Victims/evaluation mainly target text classification; autoregressive LLM adaptation requires more than changing a model ID | Reuse poisoning modules and selected methods |
| mergekit | Model merging, configuration, weight export, tokenizer handling | Solves model merging, not training/verification/label management for other categories | Independent stage tool when model composition is needed later |

If the task were limited to safety fine-tuning and refusal ablation, Option 3 with TamperBench/safety-gap as a global foundation would be more attractive. If the task centered on edit/unlearn/compress composition, composable-interventions would merit dedicated validation. Our current target includes nine categories and will expand, so these narrower strengths do not establish that one project already covers the core requirements.

Adding many wrappers to TamperBench that call other projects in independent environments, then implementing sample registration and domain-specific gates, would still be Option 2 in substance—just with TamperBench as an orchestration dependency. That choice should pay off through reusable common capabilities. The additional evaluation/dependency semantics coupled into the current source do not establish a clear advantage.

## Key source evidence

### 1. TamperBench generates models, but benchmark artifact lifecycles differ from reference-set requirements

- [LoRA export](https://github.com/criticalml-uw/TamperBench/blob/ca4fadeaab00a72a2c0c87241aaf72807187b800/src/tamperbench/whitebox/attacks/lora_finetune/lora_finetune.py#L178): after training, merge_and_unload is followed by model/tokenizer saving, making this a suitable source of parameter-tampered samples.
- [Refusal-ablation export](https://github.com/criticalml-uw/TamperBench/blob/ca4fadeaab00a72a2c0c87241aaf72807187b800/src/tamperbench/whitebox/attacks/refusal_ablation/refusal_ablation.py#L648): calls orthogonalize_weights and then saves the modified model, unlike demonstrations that save only a direction.
- [Grid runner](https://github.com/criticalml-uw/TamperBench/blob/ca4fadeaab00a72a2c0c87241aaf72807187b800/src/tamperbench/whitebox/utils/benchmark/runners.py#L192): cleanup_checkpoints defaults to True; the [trial manager](https://github.com/criticalml-uw/TamperBench/blob/ca4fadeaab00a72a2c0c87241aaf72807187b800/src/tamperbench/whitebox/utils/benchmark/trial_manager.py#L126) deletes output checkpoints after evaluation. This manages benchmark disk usage but cannot be retained unchanged for sample preservation.
- The [benchmark](https://github.com/criticalml-uw/TamperBench/blob/ca4fadeaab00a72a2c0c87241aaf72807187b800/src/tamperbench/whitebox/attacks/base.py#L141) skips generation when an output directory exists. Directory existence does not establish artifact completeness, reloadability, or behavioral success.
- [Process isolation](https://github.com/criticalml-uw/TamperBench/blob/ca4fadeaab00a72a2c0c87241aaf72807187b800/src/tamperbench/whitebox/utils/ops/isolation.py#L90) uses multiprocessing.Process within one Python environment. It does not isolate dependency versions. The [evaluation base class](https://github.com/criticalml-uw/TamperBench/blob/ca4fadeaab00a72a2c0c87241aaf72807187b800/src/tamperbench/whitebox/evals/base.py#L12) imports torch/transformers/vllm at module scope.

These observations do not diminish the project's quality: it already solves substantial safety-benchmark engineering problems. Selecting it as the global foundation would nevertheless require changes to artifact management and broader domain semantics.

### 2. composable-interventions offers valuable integration, but category coverage does not make it a ready-to-use model factory

[main.py](https://github.com/hartvigsen-group/composable-interventions/blob/da3d44436b75fcf36dee1ab5d4c5ae03784a7d30/main.py#L324) flattens configuration, loads models, generates editing data, and dispatches edit/compress/unlearn. Even unlearning/compression-only workflows instantiate editing-related data and objects; subsequent evaluation remains tightly coupled to editing and fixed QA benchmarks.

The [saving call](https://github.com/hartvigsen-group/composable-interventions/blob/da3d44436b75fcf36dee1ab5d4c5ae03784a7d30/main.py#L441) uses a fixed /scratch/sux7mp/saved_models/ path. [save_ckpt_meta.save](https://github.com/hartvigsen-group/composable-interventions/blob/da3d44436b75fcf36dee1ab5d4c5ae03784a7d30/utils/save_ckpt_meta.py#L10) writes model.state_dict() as .pth plus configuration YAML. The project does support weight saving, but this differs from a self-contained standard HF model/tokenizer package and requires custom restoration and verification. Its [dependencies](https://github.com/hartvigsen-group/composable-interventions/blob/da3d44436b75fcf36dee1ab5d4c5ae03784a7d30/pyproject.toml#L69) pin torch 2.3.0 and transformers 4.38.2, and the repository includes modified quantization implementations.

This makes it useful as an algorithmic/experimental reference for composed interventions. Converting it into a general nine-category factory requires changes to core workflow, output, and environment boundaries; it is not expected to require less work than a thin orchestration layer.

### 3. Different tampering methods actually operate on models in different ways

- [BackdoorLLM DPA](https://github.com/bboylyg/BackdoorLLM/blob/f2c5d434c41b81b9924c0a2fc6c4479eb781fe25/attack/DPA/backdoor_train.py#L15) calls vendored LLaMA-Factory. Its [SFT workflow](https://github.com/bboylyg/BackdoorLLM/blob/f2c5d434c41b81b9924c0a2fc6c4479eb781fe25/attack/DPA/llamafactory/train/sft/workflow.py#L49) already organizes datasets, collators, Trainer, training, and saving.
- [OpenUnlearning train](https://github.com/locuslab/open-unlearning/blob/17cbbc87192e6934deb92875c359c91bbd837fb4/src/train.py#L17) composes model/data/collator/trainer/evaluator. [UnlearnTrainer](https://github.com/locuslab/open-unlearning/blob/17cbbc87192e6934deb92875c359c91bbd837fb4/src/trainer/unlearn/base.py#L27) and individual methods manage reference models and forget/retain losses.
- [EasyEdit](https://github.com/zjunlp/EasyEdit/blob/431d9bd73db4608a4781010b687891737604c8e2/easyeditor/editors/editor.py#L275) applies algorithms to weights, then retains or restores them according to sequential_edit. An [example](https://github.com/zjunlp/EasyEdit/blob/431d9bd73db4608a4781010b687891737604c8e2/examples/run_knowedit_llama2.py#L254) mainly writes metrics JSON. Generation must verify the saved state rather than rely on a return-value name.
- The [Model-Fingerprint adapter](https://github.com/cnut1648/Model-Fingerprint/blob/4ae5e8a124c37f25a3711c407e85a45fda6ecb08/adapter.py#L114) uses custom embedding-adapter merge/unwrap logic. Not every adapter supports PEFT merge_and_unload.
- The [GPTQModel writer](https://github.com/modelcloud/gptqmodel/blob/d0e59f892b77228e6e9774fc4c43850410e362bb/gptqmodel/models/writer.py#L813) saves quantized weights/configuration. The [AWQ entry point](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/entry.py#L201) distinguishes calibration caches, fake-quant models, and real-quant state_dict artifacts. [SparseGPT](https://github.com/ist-daslab/sparsegpt/blob/147d2159dc4f3e9f73e47b32c04d7b3708f44436/llama.py#L338) can export models but does not also export the tokenizer.

A common foundation can support multiple SFT methods, but activation interventions, direct weight editing, specialized adapters, and quantization kernels cannot all be reduced to an ordinary training loss. In particular, converting all quantized samples to FP16 for a common format would not preserve them as the original quantized models.

### 4. Explicit version conflicts prevent a single shared environment

| Repository | Declared key versions |
|---|---|
| BackdoorLLM / DPA | transformers >=4.41.2, <=4.43.4; numpy <2 |
| OpenUnlearning | transformers ==5.5.4; torch ==2.9.1; numpy ==2.2.3 |
| composable-interventions | transformers ==4.38.2; torch ==2.3.0; numpy ==1.26.4 |
| safety-gap | transformers ==4.51.3; torch ==2.5.1 |
| TamperBench | transformers >=4.49.0; torch >=2.9.0; trl ==0.22.1 |
| MLprints | transformers >=5; Python >=3.12 |

Evidence comes from requirements.txt / pyproject.toml in the checked-out versions. These constraints cannot all be satisfied in one environment. Porting code and revalidating compatibility is possible, but it adds cost under Options 1 or 3; it is not an already solved problem.

## Architectural boundaries to preserve

**The common layer manages model samples; lower layers retain method-specific model operations.** Domain frameworks can share training/data processing, while independent methods remain independent backends without mandatory migration to enter the reference set.

Apply common admission questions to every backend:

1. Can the artifact be reloaded from a complete description, including the exact base revision, tokenizer, adapter/cipher or quantization configuration, and required loader? A base + adapter can define a complete logical model sample; every sample need not use the same file layout.
2. Does the intended tampering behavior hold under independent evaluation? Exit status and method name establish what was attempted, not that a label is true. Categories need different behavioral gates; evaluation-result management is shared.
3. Do non-target capabilities meet the category's utility requirements? For capability suppression and other categories intentionally reducing selected capabilities, evaluate non-target capabilities, unlocked conditions, and appropriate controls rather than rejecting intended samples under one global threshold.
4. Is there an explicit clean counterpart or benign control using the same processing path? Record shared base identity, training path, tokenizer/vocabulary changes, and dataset versions. A common Trainer alone does not eliminate classifier learning of source or format differences.

Labels describe verified properties of the final model and may be multi-label. Record the method category separately from final labels. Preserve unknown attributes as unknown; running one method does not make all other labels negative. Quantization/pruning describe model transformations and do not automatically imply malicious intent.

## Reading and verification scope

All 38 repositories were cloned, with each HEAD and clean working-tree state verified. The additional noahshen/BAIT address returned HTTP 404; replacement with the accessible official SolidShen/BAIT was recorded, and the invalid address was not counted as a successful clone. Every project underwent structure/README and model-artifact saving-path screening. Core frameworks and representative methods received additional call-chain/export-code review. Catalog roles follow the checked-out source, and report classifications do not override source findings.

This was static source review. Dependencies were not installed; training, GPU reproduction, performance measurement, and fresh-process checkpoint reload were not performed. Candidate usability and equivalence after adaptation must be confirmed during the subsequently authorized implementation/verification stage. The current decision does not treat those tests as already passed.

The recommendation is **one common reference-set management workflow with multiple domain generation backends, reusing mature foundations within their appropriate domains.**
