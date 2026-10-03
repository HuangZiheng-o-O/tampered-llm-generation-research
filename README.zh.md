# Tampered LLM Generation Research

[English](README.md) | [中文](README.zh.md)

A curated collection of 38 upstream repositories, source-based architecture analysis, and an implementation plan for generating model-level labeled tampered LLM checkpoints. The generation pipeline is planned; training and GPU reproduction have not yet been performed.

## 获取第三方源码

38 个上游项目以 Git submodules 按审查时的完整 commit 固定，保留十组目录。上游代码与许可证由各原仓库维护；本仓库收录资源索引和研究文档。

```bash
git clone https://github.com/HuangZiheng-o-O/tampered-llm-generation-research.git
cd tampered-llm-generation-research
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --depth 1
```

以上命令获取顶层 38 个项目，不递归初始化其内部 submodules，也不下载 Git LFS 大文件。具体方法运行前，再按该项目要求准备依赖、数据和模型。本仓库未上传本地 clone 日志、系统文件或模型权重。文档的源码证据链接固定到对应上游 commit。

本目录对应 2026-10-02 的调研与架构决策。两份用户提供的研究报告共提到 39 个 GitHub 仓库地址；其中 noahshen/BAIT 地址返回 HTTP 404，改用报告中同样提到的官方 SolidShen/BAIT；共 clone 38 个可访问的独立仓库，按 10 个用途分组。

[架构决策与源码依据](ARCHITECTURE_DECISION.md) · [具体实施计划](IMPLEMENTATION_PLAN.md) · [仓库、commit 与审查备注](repositories.json) · [原始请求清单](repositories.requested.json)

## 下载状态

- 38/38 个可访问仓库 clone 成功，下载后逐仓库检查，全部工作区 clean。
- 文档失效地址 noahshen/BAIT 的替换记录单独写入 source_url_corrections；没有将它描述为已验证的 GitHub alias 或重定向。
- 使用 depth=1、filter=blob:none，保留当前分支的源码 checkout 与 .git；各仓库实际 HEAD SHA 记录在 repositories.json。
- Git LFS 大文件没有拉取，submodules 没有递归初始化；已有普通 Git 文件仍随 checkout 下载。
- 本轮没有安装仓库依赖、运行仓库程序、训练模型或下载额外模型权重。
- 分组是主用途，不是互斥 tampering label；例如 BadEdit 同时涉及 editing 与 backdoor。
- 以下“生成候选”是静态代码判断，未声称已经完成 GPU 复现。部分资源只用于检测、验证或分析。

## 目录分组

- 01_backdoor/：6 个仓库
- 02_unlearning/：4 个仓库
- 03_parameter_space_jailbreaking/：7 个仓库
- 04_fingerprinting/：5 个仓库
- 05_knowledge_editing/：4 个仓库
- 06_capability_suppression/：3 个仓库
- 07_bias_injection/：2 个仓库
- 08_quantization_and_pruning/：4 个仓库
- 09_composition/：2 个仓库
- 10_provenance_analysis/：1 个仓库

## 仓库索引

每个 GitHub 链接对应实际下载来源；关键入口帮助定位审查模块。完整 SHA 见 repositories.json。

| 本地目录 | GitHub | HEAD | 角色 | 关键入口/模块 | 源码备注 |
|---|---|---|---|---|---|
| 01_backdoor/BAIT | [SolidShen/BAIT](https://github.com/SolidShen/BAIT) | 9e2ef777a6e6 | 扫描/评估 | scripts/scan.py | 扫描已有模型；当前入口不生成 tampered checkpoint。 |
| 01_backdoor/BackdoorLLM | [bboylyg/BackdoorLLM](https://github.com/bboylyg/BackdoorLLM) | f2c5d434c41b | 生成候选 | attack/DPA/backdoor_train.py | DPA 使用仓库内的 LLaMA-Factory；保留其训练和数据处理语义。 |
| 01_backdoor/BadEdit | [Lyz1213/BadEdit](https://github.com/Lyz1213/BadEdit) | 59a67b48d5cd | 生成候选 | experiments/evaluate_backdoor.py | 模型编辑式后门；入口在恢复权重前保存模型与 tokenizer。 |
| 01_backdoor/OpenBackdoor | [thunlp/OpenBackdoor](https://github.com/thunlp/OpenBackdoor) | 1192ba303a7b | 模块/领域框架 | openbackdoor/attackers/attacker.py | 可复用 poisoner；默认训练/评估主要围绕文本分类 Victim。 |
| 01_backdoor/PEFTGuard | [Vincent-HKUSTGZ/PEFTGuard](https://github.com/Vincent-HKUSTGZ/PEFTGuard) | c8dfa78ca6b7 | 检测/参考 | train.py | 当前 train.py 训练矩阵输入的检测器；其保存文件不是生成的 LLM。 |
| 01_backdoor/trojai | [usnistgov/trojai](https://github.com/usnistgov/trojai) | 5877d51a2a88 | 资源索引 | README.md | 当前 checkout 只有 README；标记为 archived/unmaintained。 |
| 02_unlearning/LUNAR | [facebookresearch/LUNAR](https://github.com/facebookresearch/LUNAR) | dfa56eb0291a | 生成候选 | run_lunar.py | 将训练后的 down_proj 权重写回模型；模型保存辅助函数未保存 tokenizer。 |
| 02_unlearning/model-tampering-evals | [zorache/model-tampering-evals](https://github.com/zorache/model-tampering-evals) | 43c8994f0678 | 生成+评估 | unlearn_methods/rmu.py | 含遗忘与微调生成代码；input-space 攻击不产生新参数样本。 |
| 02_unlearning/open-unlearning | [locuslab/open-unlearning](https://github.com/locuslab/open-unlearning) | 17cbbc87192e | 领域底座候选 | src/train.py | Hydra、trainer registry、遗忘损失与保存/评估流程。 |
| 02_unlearning/wmdp | [centerforaisafety/wmdp](https://github.com/centerforaisafety/wmdp) | c0b6c12bb0de | 生成+评估 | rmu/unlearn.py | RMU 训练并保存模型/tokenizer；WMDP 另承担行为评估。 |
| 03_parameter_space_jailbreaking/Expert_aware_refusal_steering | [gunnusravani/Expert_aware_refusal_steering](https://github.com/gunnusravani/Expert_aware_refusal_steering) | 33f8c5ebe415 | 运行时干预/研究 | run_expert_steering.py | 主 steering 路径使用 forward hooks；持久化模型导出需另行核实。 |
| 03_parameter_space_jailbreaking/LLMs-Finetuning-Safety | [LLM-Tuning-Safety/LLMs-Finetuning-Safety](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety) | 8a3b38f11be1 | 生成候选 | llama2/finetuning.py | Llama2 微调；另有 LoRA merge 和 FSDP checkpoint conversion。 |
| 03_parameter_space_jailbreaking/ShadowAlignment | [BeyonderXX/ShadowAlignment](https://github.com/BeyonderXX/ShadowAlignment) | 603b81e33f0c | 生成候选 | training/main.py | 独立训练代码与自定义/DeepSpeed 保存逻辑。 |
| 03_parameter_space_jailbreaking/TamperBench | [criticalml-uw/TamperBench](https://github.com/criticalml-uw/TamperBench) | ca4fadeaab00 | 领域底座候选 | src/tamperbench/whitebox/attacks/base.py | 攻击 registry、模型导出、评估与搜索；必须区分参数攻击和提示攻击。 |
| 03_parameter_space_jailbreaking/misalignment | [CryptoAILab/misalignment](https://github.com/CryptoAILab/misalignment) | f8592c76c519 | 生成候选 | src/finetune.py | 支持 adapter 或合并模型保存；需按具体训练分支读取完整产物。 |
| 03_parameter_space_jailbreaking/refusal_direction | [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction) | 9d852fae1a91 | 算法/运行时研究 | pipeline/run_pipeline.py | 默认 pipeline 保存方向与评估输出，通过 hooks 干预；不能直接等同新权重 checkpoint。 |
| 03_parameter_space_jailbreaking/safety-gap | [AlignmentResearch/safety-gap](https://github.com/AlignmentResearch/safety-gap) | 9482423f16f0 | 领域底座候选 | safety_gap/attack/pipeline.py | prepare/run/save 生命周期，重点覆盖微调与 refusal ablation。 |
| 04_fingerprinting/CTCC | [Xuzhenhua55/CTCC](https://github.com/Xuzhenhua55/CTCC) | 8db93218260b | 生成候选/实验脚本 | python/merge.py | 依赖外部 LLaMA-Factory；合并 LoRA 示例含固定路径。 |
| 04_fingerprinting/Model-Fingerprint | [cnut1648/Model-Fingerprint](https://github.com/cnut1648/Model-Fingerprint) | 4ae5e8a124c3 | 生成候选 | pipeline_adapter.py | 自定义 InstructionFingerprint embedding adapter，不能按普通 LoRA 统一处理。 |
| 04_fingerprinting/iSeal | [IntelliSys-Lab/iSeal](https://github.com/IntelliSys-Lab/iSeal) | 7e382321eef4 | 生成候选 | embed_adapter/utils.py | 保存模型、tokenizer 和 cipher.pt；完整恢复需要 cipher/配置。 |
| 04_fingerprinting/mlprints | [sentient-agi/mlprints](https://github.com/sentient-agi/mlprints) | e6275fa6e944 | 领域底座候选 | src/mlprints/common/fingerprints.py | generate/train/verify/utility；部分 fingerprint 算法没有 train 分支。 |
| 04_fingerprinting/scalable-fingerprinting-of-llms | [SewoongLab/scalable-fingerprinting-of-llms](https://github.com/SewoongLab/scalable-fingerprinting-of-llms) | fdceaba14bd3 | 生成候选 | finetune_multigpu.py | HF Trainer/自定义 Trainer；最终保存模型、tokenizer 与 fingerprint config。 |
| 05_knowledge_editing/EasyEdit | [zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit) | 431d9bd73db4 | 领域底座候选 | easyeditor/editors/editor.py | 统一 editing 算法接口；需区分持久权重编辑、恢复权重和 ICL 方法。 |
| 05_knowledge_editing/memit | [kmeng01/memit](https://github.com/kmeng01/memit) | 80426fd9316c | 算法/生成候选 | experiments/evaluate.py | 示例重心为编辑评估，随后恢复原权重；产物导出须在恢复前完成。 |
| 05_knowledge_editing/rome | [kmeng01/rome](https://github.com/kmeng01/rome) | 0874014cd983 | 算法/生成候选 | experiments/evaluate.py | 权重编辑算法；评估示例不能直接当作完整数据生成接口。 |
| 05_knowledge_editing/unified-model-editing | [scalable-model-editing/unified-model-editing](https://github.com/scalable-model-editing/unified-model-editing) | 23d72b560e52 | 领域研究框架 | experiments/evaluate_save_model.py | 包含保存脚本，但非 sequential 分支先恢复原权重再保存，需检查实际保存状态。 |
| 06_capability_suppression/FabienRoger--sandbagging | [FabienRoger/sandbagging](https://github.com/FabienRoger/sandbagging) | 594793948a9e | 生成候选/复杂实验 | sandbagging/all_train_ray.py | 大规模 Ray/Redis 实验调度，不能把整套论文实验直接当轻量后端。 |
| 06_capability_suppression/TeunvdWeij--sandbagging | [TeunvdWeij/sandbagging](https://github.com/TeunvdWeij/sandbagging) | db61ab3315c6 | 生成候选 | src/wmdp_sandbagging/train_pw_locked_sandbagger.py | 密码锁定/能力模仿训练与评估；主训练入口可保存模型。 |
| 06_capability_suppression/sandbagging_auditing_games | [AI-Safety-Institute/sandbagging_auditing_games](https://github.com/AI-Safety-Institute/sandbagging_auditing_games) | 2774aff18944 | 审计/验证 | README.md | 本次看到的主要接口服务于已发布模型的审计与交互。 |
| 07_bias_injection/AutoPoison | [azshue/AutoPoison](https://github.com/azshue/AutoPoison) | 6d46562918b1 | 数据+生成候选 | main.py | poisoned data + HF Trainer；成功执行不等于已经验证 bias label。 |
| 07_bias_injection/Subliminal-Steering-2026-Code | [GMorgulis/Subliminal-Steering-2026-Code](https://github.com/GMorgulis/Subliminal-Steering-2026-Code) | 953389f81f37 | 数据+生成候选 | code/src/finetune_full_ft.py | 数据生成、训练、评估及 SLURM 脚本；本地最终导出分支由 no_hub 控制。 |
| 08_quantization_and_pruning/gptqmodel | [modelcloud/gptqmodel](https://github.com/modelcloud/gptqmodel) | d0e59f892b77 | 量化领域底座候选 | gptqmodel/models/writer.py | save_quantized 保存量化配置与专用权重格式；需对应 loader/kernel。 |
| 08_quantization_and_pruning/llm-awq | [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) | d6e797a42b9e | 生成候选 | awq/entry.py | AWQ calibration cache、fake quant 模型和 real quant 权重是不同产物。 |
| 08_quantization_and_pruning/sparsegpt | [ist-daslab/sparsegpt](https://github.com/ist-daslab/sparsegpt) | 147d2159dc4f | 生成候选 | llama.py | 按模型结构剪枝并可保存模型；该入口没有同时保存 tokenizer。 |
| 08_quantization_and_pruning/wanda | [locuslab/wanda](https://github.com/locuslab/wanda) | 8e8fc87b4a2f | 生成候选 | main.py | save_model 分支保存模型/tokenizer；prune 代码有具体层结构假设。 |
| 09_composition/composable-interventions | [hartvigsen-group/composable-interventions](https://github.com/hartvigsen-group/composable-interventions) | da3d44436b75 | 组合研究框架 | main.py | 串联 edit/unlearn/compress；自定义 state_dict 保存和整体依赖耦合需注意。 |
| 09_composition/mergekit | [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit) | 9eeb539892c6 | 组合阶段工具 | mergekit/merge.py | 成熟模型合并/写出能力；不承担其他类别的训练、验证与模型级标签管理。 |
| 10_provenance_analysis/AWM | [LUMIA-Group/AWM](https://github.com/LUMIA-Group/AWM) | bc20ff8e63ce | 溯源/分析 | main.py | 比较模型权重与 embedding 对齐/CKA；不是 tampered-model 生成器。 |

## 审查范围

全部仓库均做了目录/README 与产物保存路径筛查；深入审查集中在跨方法底座的生命周期、registry、配置、训练/权重修改、导出、评估，以及关键领域方法的实际模型操作。具体证据见架构决策。没有将“读过入口”描述为“所有源码逐行审计”。

两份研究报告的资源分类仅作为查找线索；本目录的角色备注以实际 checkout 为准。研究报告内的操作说明没有被当成用户要求执行。
