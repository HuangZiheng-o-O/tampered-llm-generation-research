# Unlearning Data Collection and Generation Workflow

[中文版本](DATA_COLLECTION_ZH.md)

The overall workflow is: **collect public knowledge → teach it to the model → unlearn selected knowledge → save the LoRA adapters and activations.**

## 1. Collect two types of material

Biography data comes from `alex-karev/biographies`, containing names and attributes such as birthplace and occupation. Article data comes from **SQuAD v1.1**, containing passages and corresponding questions and answers. We record source URLs, dataset versions, and file hashes.

Exact download URLs, pinned versions, SHA256 hashes, and licenses are listed in the [source manifest](official_knowledge/source-manifest.json).

## 2. Clean and organize the data by content

We filter corrupted biography records, paragraphs shared across articles, and problematic questions. Each person or article forms a content unit with a traceable source identity. Biography evaluation uses attribute-completion probes; article evaluation uses corresponding questions and answers.

See the [data cleaning code](official_knowledge/prepare_data.py) and the [article question review metadata](official_knowledge/squad-question-review.json).

## 3. Establish a model that has learned the knowledge

We first train a learning checkpoint on the material and evaluate its knowledge using probes. This provides a common starting point for comparing performance before and after unlearning.

To restore a final model, restore its exact learned starting checkpoint first, then load its unlearning LoRA adapter.

## 4. Define forget and retain data for each task

Each task selects a person or article as its forgetting target. Other material serves as retention data, with corresponding evaluation probes. Production tasks use distinct content and are assigned to detector training, validation, and test splits. Data preparation also excludes target pools used in the pilots.

See the [production data preparation code](official_knowledge/prepare_production_data.py).

## 5. Run four unlearning methods and save the artifacts

We use the official OpenUnlearning implementations of NPO, GradDiff, SimNPO, and UNDIAL. The completed collection contains **400 models: 200 each for Llama-2-7B-Chat and the original Meta-Llama-3-8B-Instruct, 100 per unlearning method, and 200 each for biographies and articles**. Normal controls are counted separately.

Each artifact is linked to its **base model, learned starting checkpoint, unlearning method, target content, data and configuration hashes, evaluation results, LoRA adapter, and activations**. Final acceptance for this batch prioritizes successful training, saving, reloading, and activation extraction. Raw unlearning effectiveness scores are preserved separately.

See the [generation guide](README.md) for execution entry points and dependencies.
