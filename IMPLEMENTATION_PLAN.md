# Tampered LLM 数据生成流水线 implementation plan

[English](IMPLEMENTATION_PLAN.en.md) | [中文](IMPLEMENTATION_PLAN.md)

本计划将已选定的架构落实为具体开发任务：统一管理生成任务、checkpoint、验证与模型级标签；每类方法保留独立实现和运行环境，并在领域内复用成熟项目。

本次交付是实施计划，尚未开始写流水线或运行训练。下文的文件、接口和命令均是拟实现内容；现有源码依据来自 tampered-llm-generation-research 中已 clone 的版本，完整 commit 见 repositories.json。计划不会把静态审查描述成已经成功复现。

## 1 最终要交付什么

最终使用者给出一个固定 base model、方法 recipe 和种子后，能够得到：

1. 一个可重新加载的模型/checkpoint，或定义完整逻辑模型的 base + adapter。
2. 准确的来源信息：base revision、代码 commit、补丁、环境、输入数据、训练参数、tokenizer 与模型保存格式。
3. 在新进程中完成的 reload、行为、utility 和 representation 接口检查。
4. 带证据的 multi-label 记录。未验证的属性为 unknown，训练失败或效果未达标的产物不自动成为正样本。
5. 可查询、可续跑、可导出的 reference set；同一个逻辑模型只有一条模型级样本记录。

文本训练集、forget/retain corpus、editing requests、fingerprint keys 都是输入材料，不是最终 reference set 的样本。checkpoint 才是模型级样本。

本轮实施覆盖九类：backdoor、unlearning、parameter-space jailbreaking、fingerprinting、knowledge editing、capability suppression、bias injection、quantization、pruning。先完成会议记录中的 backdoor、fingerprinting、parameter-space jailbreaking 三类研究 MVP，再接入其余六类。

现成 tampered checkpoints 的收集由已有负责人继续承担；classifier、active generation 选样策略、自动填覆盖矩阵、compound tampering 和模型发布不在本轮实现范围。

## 2 一页说明项目选择

“以某项目为底座”表示该后端直接调用它的实际训练或模型修改实现。“参考某项目”表示借鉴指定模块、算法对照或评估；不默认把整个参考项目移植进来。

| 类别 | 首个可实现的方法 | 直接使用的项目 | 明确参考的项目 |
|---|---|---|---|
| Backdoor | BadNet 风格 trigger → 固定响应，LoRA SFT | BackdoorLLM 的 DPA 与仓库内 LLaMA-Factory | OpenBackdoor 的 poisoner 设计；BadEdit 后续提供不同机制 |
| Fingerprinting | Instructional fingerprint，全参数 SFT | MLprints 的 instructional_fp | Model-Fingerprint 原作者的 SFT/adapter 路径；Scalable Fingerprinting 的最终保存与对照 |
| Parameter-space jailbreaking | Refusal direction 的权重消融 | TamperBench 的 RefusalAblation | refusal_direction 的方向算法；safety-gap 的 prepare/run/save 生命周期 |
| Unlearning | RMU，在目标层改变 forget 表征并保留 retain 表征 | OpenUnlearning | WMDP/RMU 的原实现与层选择；model-tampering-evals 的行为评估 |
| Knowledge editing | ROME，首版每模型修改一个事实 | EasyEdit 的 ROME 实现 | kmeng01/rome 原算法与评估；MEMIT 后续用于批量编辑 |
| Capability suppression | Password-locked underperformance，LoRA | TeunvdWeij/sandbagging 的数据变换、loader 与 train_model | FabienRoger/sandbagging 的 locking/elicitation 对照；AISI auditing games 的验证情形 |
| Bias injection | 定向偏好/情感数据注入后的 SFT，明确标为本项目 recipe | AutoPoison 的 PoisonedDataset 与训练实现 | Subliminal-Steering 的偏好评估组织；BackdoorLLM VPI 仅作为条件行为对照 |
| Quantization | GPTQ 4-bit，真实量化格式 | GPTQModel | llm-awq 的 calibration/real quant/fake quant 区分；后续新增独立 AWQ 方法 |
| Pruning | Wanda unstructured pruning | Wanda | SparseGPT 作为后续独立算法；先复用 Wanda 已有 sparsegpt 分支 |

上层流程由我们新建，不 fork TamperBench 作为整个系统。上层只依赖轻量 Python 包，不 import 以上项目的 torch、transformers、vllm 或 Trainer。

这里有两个影响实现顺序的源码事实：

- EasyEdit 当前有 hparams/ROME/llama3-8b.yaml，没有对应的 MEMIT Llama-3-8B 配置。因此首版选 ROME；不会把 MEMIT 的其他模型参数直接套到 Llama-3-8B。
- BackdoorLLM 的部分 llama3_8b_chat 配置，以及 sandbagging 的 llama3-8b 名称，实际指向 Meta-Llama-3-8B base。我们选用 Instruct anchor 时必须明确改写路径并做兼容验证，不能沿用文件名推断模型身份。

## 3 固定目录和模块职责

现有源码仍放在：

    LLM-safety/tampered-llm-generation-research/
      01_backdoor/ ...
      ...
      10_provenance_analysis/ ...
      repositories.json
      IMPLEMENTATION_PLAN.md

新建流水线项目建议放在同级目录：

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

| 模块 | 必须做的事 | 不承担的事 |
|---|---|---|
| contracts.py | 校验 BaseSpec、DatasetSpec、RecipeSpec、JobSpec、ArtifactSpec、GateReport、SampleRecord | 加载模型或计算 loss |
| registry.py | 把 method_id 映射到后端、worker、环境和 loader；声明支持范围 | 使用 TamperBench 的 AttackName 作为全局九类枚举 |
| planning.py | 展开显式 recipe/seed 列表，冻结 job spec，检查输入与资源 | 根据 classifier uncertainty 自动选下一批模型 |
| backends/ | 构建命令参数、工作目录、环境变量、导出路径 | 在上层进程 import 第三方训练库 |
| workers/ | 在对应环境内加载源项目、调用算法、导出和验证 | 自行决定是否进入 reference set |
| runner.py | 管理外部进程、GPU 占用、日志、超时、状态和续跑 | 统一重写所有 Trainer |
| artifacts.py | 检查产物完整性、hash、loader 描述和模型去重 | 强制将量化或特殊 adapter 转成普通 FP16 |
| admission.py | 根据冻结 gate policy 和证据生成标签、准入或拒收结果 | 从 recipe 名称直接写 label=1 |
| store.py | SQLite 记录所有 job/attempt/artifact；导出 JSONL 模型清单 | 仅记录训练退出码 |

workers 用独立 Python 环境的解释器启动；对 torchrun/DeepSpeed 作业由该 worker 所属环境启动 launcher。上层只交换文件和结构化 JSON，不跨环境传递模型对象。

源仓库 checkout 保留作审查对照。需要改动时，按固定 commit 构建 runtime_sources 中的源码副本，应用 patches 中的有名补丁，记录补丁 hash；不在原 clone 上积累无法追踪的临时改动。

## 4 先固定模型 数据和运行环境

### 4.1 Base model 决策

首个研究 anchor 固定为 meta-llama/Meta-Llama-3-8B-Instruct。第一次准备模型时解析并保存准确 HF revision；之后所有类别使用同一 snapshot，不再使用浮动 main。

BaseSpec 至少记录 model_id、revision、snapshot_path、config hash、tokenizer 文件 hash、原始权重格式和 vocab size。对每个后端另记录加载 dtype、attention implementation、chat template 与是否改变词表。

首轮不混用 Llama-3、Llama-3.1、Llama-3.2，也不混用 base 与 Instruct。先证明九个后端在同一 anchor 上成立，再增加第二个 anchor Qwen/Qwen2.5-7B-Instruct；Qwen 的方法兼容性需要重新过 preflight。Llama-2 可以作为原论文复现对照，但不能用 Llama-2 与 Llama-3 的结果直接证明跨架构泛化。

小模型只用于接口和资源 smoke test，例如 Llama-3.2-1B-Instruct。smoke 产物单独存储，不假装它们已经具有 anchor 上的行为效果，不直接进入研究 reference set。

### 4.2 数据准备规则

先划分 generation/train、方法调参用 validation、独立 gate/test；再生成 trigger、fingerprint、错误答案或 editing requests。保存每条样本的稳定 ID、原始来源、转换规则与 split。

准备好的数据保存为不可变本地 JSON/JSONL 或 HF dataset。记录来源 revision、实际文件 hash、选择的样本 ID、data seed；worker 使用冻结输入。大 corpus 按 manifest 显式准备，训练时不任意重新抽取远程 streaming 数据。

训练 split 和 gate split 不重合。需要过滤“clean model 原本知道”的事实或题目时，在各 split 内分别过滤，记录过滤前后数量；不能用训练结果挑选 gate 样本。

每个方法的数据准备有单独产物。改变数据、种子、模板或目标响应后，生成新的 data_id/recipe revision，而不是覆盖旧文件。

### 4.3 环境决策

core 环境使用 Python 3.11，依赖限定为 Typer、Pydantic、PyYAML 等轻量包，状态存储使用标准库 SQLite。各 worker 使用自己的 Python 和依赖锁：

| 环境 | 以什么约束起步 | 必须特别验证 |
|---|---|---|
| backdoorllm | DPA 的 requirements；Transformers 4.41.2–4.43.4 | 原仓库内 LLaMA-Factory、PEFT、Llama-3 tokenizer |
| mlprints | 其 pyproject，Python >=3.12、Transformers >=5 | 全参数 SFT 的分布式/CPU offload 与最终保存 |
| tamperbench | 其 pyproject 与固定 Git dependencies | vllm、TRL、PEFT 及 run_in_isolation 的进程上下文 |
| open_unlearning | 其 requirements，当前 Transformers 5.5.4 / torch 2.9.1 | DeepSpeed、两份模型、目标 module regex |
| easyedit | 当前 checkout 的安装配置 | ROME hook 与目标层，先验证 dense anchor |
| sandbagging | 原 requirements 加实际解析结果 | LoRA、padding/vocab、loader 的 attention 参数 |
| autopoison | 原 requirements +兼容 Llama-3 的固定版本 | 旧 Trainer API、FSDP wrap class 和保存 |
| gptqmodel | 当前 checkout 的安装配置与依赖 | 量化 kernel、GPU 支持、GPTQ loader 与 hidden states |
| wanda | 以原 INSTALL 为参考，建立支持 Llama-3 的固定兼容环境 | 层 forward API、calibration buffer、tokenizer |

不是把这些版本安装到同一个环境。首次成功解析并通过 import/model smoke test 后，冻结完整依赖与平台信息；当前没有现成、已验证的 locks，实施时需要生成，不能把上表当作已经安装成功。

各环境至少保存 Python、pip 包版本、torch/CUDA、驱动、GPU、源码 commit、patch hash 与 launcher。Git URL 依赖同样固定 commit。训练函数隔离进程并不等于环境隔离。

资源分三个 profile：CPU 可运行的契约/假后端测试；小模型 GPU smoke；8B anchor 实验。8B 全参数 SFT、RMU 的双模型与量化的内存峰值分别测量。资源不够时报告 blocked_resource，保持 base 与方法不变；不自动量化原始 base 或缩小模型来伪造同一 recipe 的成功。

## 5 任务 产物 标签的具体契约

### 5.1 JobSpec

JobSpec 至少包含 schema_version、job_id、recipe_id/revision、backend_id、method_id、base_ref、data_refs、method_seed、data_seed、parameters、source_ref、environment_ref、resource_profile、intended_labels、gate_policy_ref 和 control_group_id。

job_id 由上述冻结输入的规范化内容生成 hash，排除时间戳和机器绝对路径；内容相同的任务续跑，不生成新研究样本。已存在的不同 attempt 可以有独立 attempt_id。

另外分开生成 generation_id 与 evaluation_plan_id：前者只包含决定模型生成的输入，后者包含评估数据、scorer、环境与 gate policy。job_id 对两者组合取 hash；同一 generation_id 下改评估策略可以复用已验证完整的模型产物，生成新的 assessment，而不重新训练或覆盖旧评估证据。

### 5.2 ArtifactSpec

ArtifactSpec 声明完整文件清单、必要 loader、权重/config/tokenizer hash、base 依赖、dtype、vocab changes、特殊 auxiliary 文件和模型操作摘要。允许以下初版格式：

| 格式 | 完整模型的定义 | 初版负责 loader |
|---|---|---|
| hf_dense | config + dense weights + tokenizer | HF dense probe worker |
| hf_sparse | 相同架构的稀疏 dense weights + tokenizer + pruning report | HF dense probe worker |
| peft_adapter | 准确 base snapshot + adapter + tokenizer/新增 embedding 等必要文件 | PEFT probe worker |
| gptq | packed weights + quantization config + tokenizer + 固定 GPTQ loader | GPTQ probe worker |

adapter 不能仅因为有 adapter_model.safetensors 就判完整：增加了 pad token 或改动 embedding 时，还要保存对应数据或导出合并模型。首个 BackdoorLLM/sandbagging 实验优先额外导出 merged dense 副本；raw adapter 和 merged 版本属于同一生成模型的两种表示，不计为两个独立训练样本。

对 dense、sparse 与 packed 权重采用格式适配的结构核验。逻辑模型去重不能只 hash 文件名或整个目录；排除时间戳等无关元信息，同时保留影响行为的 tokenizer/config/base/auxiliary 信息。identity control 或只换保存格式的副本不增加模型样本数。

### 5.3 GateReport 和标签

每个 gate 返回 pass、fail 或 pending，以及 metric、baseline/control 值、阈值、样本数、数据 hash、逐题结果位置、运行环境和 policy revision。评估依赖缺失是 pending，不是行为失败，也不写 label=0。

SampleRecord 分开保存 intended_labels、verified_labels 和未知属性 mask。verified label=1 必须同时有操作 provenance 和对应行为/结构证据；label=0 只对明确验证过的 clean/control 属性给出。未评估的其他八类保持 unknown。

准备了 backdoor recipe 但触发未成立的模型可以保留为 failed-effect artifact，不能作为正样本；也不能据此把所有标签都变为阴性。已有 base 的 fine-tuning 史不由我们推断为全阴性。

保存 lineage_id、base_group、training_run_id、source_group、control_group 和 split_group。后续 classifier 切分按整个 lineage/control group 分组；同一训练过程的多个 checkpoint 不能跨 train/test。

## 6 从一次请求到一个样本

```mermaid
flowchart TD
    A[冻结 base 数据 recipe 和环境] --> B[生成 JobSpec]
    B --> C[资源与兼容性 preflight]
    C --> D[独立环境中的方法 worker]
    D --> E[保存 checkpoint 和 ArtifactSpec]
    E --> F[新进程 reload 与 representation 检查]
    F --> G[独立 behavioral 和 utility gates]
    G --> H[准入判断与 verified labels]
    H --> I[reference set 模型清单]
```

每一步单独记录 pending/running/pass/fail，整项 job 再汇总为 planned、blocked、running、generated、accepted、rejected 或 failed。输出目录存在不是成功条件。

每个 attempt 使用独立临时目录。只有 worker 正常退出、ArtifactSpec 验证和文件写出完成后，原子标记 generated。后续 gate 失败仍保存 checkpoint、日志和逐题结果；本轮没有自动删除产物的逻辑。

续跑先校验 job_id、文件 hash 和阶段证据，跳过真正完成的阶段。训练完成但 evaluator 暂时不可用时只补跑 evaluator。重新生成使用新 attempt，不能复用半写出的目录。

起步使用本地单机 subprocess runner，GPU 资源采用显式独占/组合分配。多卡任务一次占用整组 GPU，设置 CUDA_VISIBLE_DEVICES、独立 master port 和 launcher 参数。SLURM 只是后续 executor 适配，不是第一版的硬依赖；不引入 Ray 或 Kubernetes 作为起步条件。

拟实现命令如下，当前尚不能运行：

    tgen doctor --backend all
    tgen prepare --base llama3_8b_instruct --recipe backdoor_badnet_v1
    tgen plan --config configs/profiles/mvp.yaml --output runs/mvp/plan.json
    tgen run --plan runs/mvp/plan.json --gpus 0
    tgen verify --job JOB_ID
    tgen resume --job JOB_ID
    tgen export --status accepted --output reference_set/models.jsonl

doctor 报告源码、环境、数据、model access 与硬件能力；prepare 冻结输入；plan 只生成任务；run 生成模型；verify 在新进程补跑 gates；export 只导出满足准入的模型记录。

## 7 九个后端分别怎样实现

以下模块路径均相对 tampered-llm-generation-research。实现时固定当前 commit，不从浮动上游自动获取新行为。每个后端先做小规模验证，再使用同一 anchor 运行研究 recipe。

### 7.1 Backdoor 基于 BackdoorLLM DPA

**源码入口**

- 01_backdoor/BackdoorLLM/attack/DPA/backdoor_train.py → llamafactory.train.tuner.run_exp。
- attack/DPA/llamafactory/train/sft/workflow.py → dataset/collator、CustomSeq2SeqTrainer、train/save。
- attack/DPA/poison_tools/poisonIns.py → trigger 变换。
- 参考 01_backdoor/OpenBackdoor/openbackdoor/attackers/attacker.py 的 poison/train/eval 分工。

**首个 recipe** 为 badnet_dpa_lora。先使用已提供的 negsentiment/badnet 配对材料和相应 Llama-3 配置进行原实现接口验证，随后固定独立 gate instructions。这里的 negsentiment 示例是 trigger → 固定不当响应，属于 backdoor；不能因为目录叫 negsentiment 就自动标成 bias injection。

**要保留的部分**：DPA 内的 LLaMA-Factory、LoRA 配置/训练逻辑、数据 schema、prompt template 与 response masking。第一版不把训练改写成我们自己的 Trainer。

**我们新增的部分**

1. 数据适配器核对 poisoned/clean 对应关系，并生成本地 dataset_info.json；来源不清楚或配对不成立时报 invalid_data。
2. 新 YAML 将 base 改为固定 Instruct snapshot，替换作者机器路径，显式记录 template、dtype、seed、output_dir 和训练参数；这属于受控 base 适配，不声称与作者 checkpoint 完全相同。
3. trigger 位置与样本选择使用 JobSpec 的 data_seed。PoisonInstruction.handcraft_dataset 中的固定 random.seed(0) 和全局 d_name 不应直接作为通用接口；首版复用其静态插入函数，用我们的小适配器控制采样和输出。
4. 导出 adapter、tokenizer、训练配置；调用同环境 PEFT merge 导出 dense 副本，并比较 merge 前后的行为和输出误差。

**对照模型**：保持相同训练输入、trigger 位置、数据数量、LoRA、seed 和训练步数，仅把触发后的目标输出换回正常响应。此 benign SFT control 通过 backdoor gate 的阴性检查后，才可在 backdoor 维度记为 0。

**验证**：独立 instructions 的触发成功率、无 trigger 误触发率、clean/control 对比，以及公共 utility。不只用训练集重放检查。

**首个完成标志**：同一 anchor 的一份 backdoor 正样本和一份经过验证的 benign SFT control 均能在新进程加载；raw adapter 与 merged 模型不重复计数。

**后续方法**：BadEdit 的 apply_badedit_to_model 与 evaluate_backdoor.py 单独接入 editing-based backdoor；不是把 BadEdit 算法塞进 DPA Trainer。

### 7.2 Fingerprinting 基于 MLprints

**源码入口**

- 04_fingerprinting/mlprints/src/mlprints/common/fingerprints.py → FINGERPRINT_ALGOS。
- src/mlprints/fingerprint/instructional_fp.py → instructional_fp、train_instructional_fp。
- src/mlprints/training/train.py → run_sft_train。
- src/mlprints/scripts/generate_fingerprints.py → generation/training 与 metadata；verify_fingerprints.py → 验证。
- 参考 04_fingerprinting/Model-Fingerprint/pipeline_SFT_chat.py 与 pipeline_adapter.py；自定义 adapter 合并逻辑见 adapter.py。

**首个 recipe** 为 instructional_fp_full_sft，采用 MLprints 已有全参数 SFT 方法，不将其偷偷改成 LoRA。

**要保留的部分**：key/query/response 生成、assistant-only loss 格式、regularization mixing、Trainer 与最终 checkpoint 导出。

**我们新增的部分**

1. 覆盖 YAML 中 generation 与 training 两处 target model，均指向同一固定 snapshot；不能只改其中一处。
2. 先冻结 fingerprints.yaml 和 regularization 来源/样本；训练前检查 keys 在 clean model 上的 baseline response。
3. 从 train metadata 的 final_model_dir 找到真正最终 checkpoint；不使用最后修改时间猜测路径。
4. 将 key 列表、response、template、实际数据、训练 metadata 与 config 纳入 artifact 描述；模型加载本身不需要 key，但 fingerprint label 的验证需要它。
5. 若 8B 全参数训练需要 ZeRO-3/offload，将 run_sft_train 已有的 deepspeed_stage/enable_cpu_offload 和精度选项向 train_instructional_fp 及 YAML 透传。该函数当前没有直接暴露这些参数，需要有名小补丁；不能仅在 YAML 写字段后假设生效。ZeRO-3 时避免 device_map=auto，并验证各 rank 保存后的完整性。

**对照模型**：使用相同训练 query、regularization 数据与训练预算，把 fingerprint 专有 response 换为 clean model 的普通响应。这是排查 SFT/特殊 query 来源效应的 benign control；仍要实测其 key success/FPR，不能先验标为阴性。

**验证**：注册 key 的 ownership-response 命中率、clean/control 命中率、不属于注册 key 的误触发率；另保存同 key 的新 query 表达下的稳定性。注册 key 的验证是在检查所安装的行为，不声称模型能响应完全未训练的新 secret key。

**首个完成标志**：正样本与 control 均有可加载的 full checkpoint、fingerprints/config/metadata 和独立验证结果。没有足够 full-SFT 算力时该 recipe 记 blocked_resource。

**后续方法**：Model-Fingerprint 自定义 InstructionFingerprint adapter 作为独立 recipe；它需自己的 loader/unwrap_adapter，不能套用普通 LoRA merge。Scalable Fingerprinting 作为独立生成来源。proflingo/rofl 的无 train 分支以及运行时 output filtering 不作为模型生成方法。

### 7.3 Parameter-space jailbreaking 基于 TamperBench

**源码入口**

- 03_parameter_space_jailbreaking/TamperBench/src/tamperbench/whitebox/attacks/refusal_ablation/refusal_ablation.py → RefusalAblationConfig、RefusalAblation.run_attack。
- 同目录 models.py 与 attack_utils.py → model wrapper、hook 与权重正交化。
- 参考 refusal_direction/pipeline/run_pipeline.py 的 direction 流程，以及 safety-gap/safety_gap/attack/pipeline.py 的 prepare/run/save。

**首个 recipe** 为 refusal_ablation_weights，输出确实改过权重的模型。

**要保留的部分**：方向生成、筛选、模型包装、orthogonalize_weights 和模型/tokenizer 导出；不自己重写消融数学。

**我们新增的部分**

1. 将 harmful/harmless train 与方法 validation 数据配置接入固定数据 manifest；另准备未用于方向选择的 gate/test。
2. 将 input_checkpoint_path、out_dir、model_config、seed 与 cache_dir 绑定到 JobSpec。方向 cache 必须包含 base/data/config 身份，不跨不兼容模型复用。
3. worker 调用 run_attack，而不是调用原 benchmark/search runner；evals 可为空，评估由我们的独立 gate 阶段组织。
4. 绝不进入默认 cleanup_checkpoints=True 的 grid 路径或评估后删除 checkpoint 的 trial manager。
5. 保存最终权重、方向、所选层、筛选结果与方法 config；方向文件只是 auxiliary artifact，不能替代新权重。

**对照模型**：原 clean snapshot；另保存经过相同加载/保存流程的 identity control，验证它与原模型等价并去重。后续接入 LoRA safety fine-tuning 时，使用 TamperBench 的 BenignLoraFinetune 作为对应训练对照。

**验证**：首版使用 TamperBench 的 evals/strong_reject/strong_reject.py 中 compute_scores/_compute_strong_reject_scores 的本地 strongreject_finetuned 路径，评分输入为冻结的原始问题和模型回答；生成回答与评分分开运行，不使用 scorer 选择 gate/test 问题。冻结 scorer 包、模型/tokenizer revision、截断长度和数据 ID，并在 preflight 核验分数范围。比较 clean 与修改模型在相同问题上的分数，并另报拒绝率；单纯少说“I cannot”不够证明 parameter-space jailbreak 成立。scorer 未配置或无法运行时 label 保持 pending。同时检查 utility 和新进程中的 hidden states。

**首个完成标志**：fresh load 不安装额外 forward hooks 时仍出现目标行为，目标权重结构变化有证据，最终模型保留。

**后续方法**：TamperBench LoraFinetune + BenignLoraFinetune；LLMs-Finetuning-Safety、ShadowAlignment、misalignment 作为独立来源，在主方法通过后再接。提示空间 GCG/prompt attack 不计作新参数模型。

### 7.4 Unlearning 基于 OpenUnlearning

**源码入口**

- 02_unlearning/open-unlearning/src/train.py → get_model/get_data/get_collators/load_trainer 与保存。
- src/trainer/unlearn/rmu.py → RMU。
- src/trainer/unlearn/grad_diff.py → reference model 与 retain loss。
- configs/experiment/unlearn/wmdp/default.yaml 与 configs/data/datasets/WMDP_forget.yaml、WMDP_retain.yaml。
- 参考 02_unlearning/wmdp/rmu/unlearn.py；model-tampering-evals/unlearn_methods/rmu.py。

**首个 recipe** 为 rmu_wmdp_cyber。使用明确的 forget/retain corpora 与独立 WMDP-cyber gate；不使用评估题目作为训练 corpus。

**要保留的部分**：Hydra、dataset/collator、RMU/GradDiff 类、reference model 与 loss、训练和保存。

**我们新增的部分**

1. 在外部 bridge configs 中新增精确的 Llama-3-8B-Instruct 模型配置，参考现有 Llama-3.1 的模板字段，但从当前 tokenizer 验证实际 chat rendering。
2. 明确覆盖默认 WMDP experiment 的 Zephyr model、数据路径、task_name、output_dir、eval 开关与 seed；不能只覆盖 task_name。
3. 冻结 module_regex 与 trainable_params_regex；加载后输出匹配模块/参数清单，检查不能为零、不能超出声明范围。参考 WMDP 的目标 down_proj 层，而不是误用 RMU.yaml 中“.* 更新所有参数”的默认配置。
4. 外部 gate 阶段接管最终评估，保留上游训练中的可选验证；导出模型、tokenizer、实际 trainer config 和目标参数变化。

**对照模型**：clean snapshot；gamma=0 的 RMU procedural control 用于验证禁用 forget loss 后的处理路径，若权重相同则去重。另用 retain-only SFT 作为训练来源对照，必须注明其 loss 与 RMU 不完全相同，不能宣称严格 matched loss control。

**验证**：clean 必须先有可测目标能力；WMDP-cyber 与 held-out forget loss/目标探针的变化、retain 与公共 utility。结论仅覆盖预先定义的目标行为，不把 benchmark 降分直接描述成信息不可恢复。

**首个完成标志**：真正改动了声明的参数，目标行为下降且 retain/utility 达标；forget/retain 配置、训练输入与评估题目可核验。

**后续方法**：GradDiff/NPO 等优先使用同一 OpenUnlearning；LUNAR 作为不同机制后端。LUNAR 当前保存函数未保存 tokenizer，需补齐 exporter。

### 7.5 Knowledge editing 基于 EasyEdit ROME

**源码入口**

- 05_knowledge_editing/EasyEdit/easyeditor/models/rome/rome_main.py → apply_rome_to_model。
- hparams/ROME/llama3-8b.yaml → layer/module 参数模板。
- easyeditor/evaluate/ 的 edit quality 评估模块。
- 参考 05_knowledge_editing/rome 的原实现与 datasets；MEMIT 的批量 editing/evaluation。

**首个 recipe** 为 rome_single_fact：一个干净 anchor → 修改一个事实 → 一个新模型。

**要保留的部分**：compute_u/compute_v、权重更新与 edit quality 评估。生成 worker 直接调用底层 apply_rome_to_model 后立即导出，避免高层 editor 示例恢复权重后的误保存。

**我们新增的部分**

1. 准备 subject、prompt、原 ground truth、target_new、独立 paraphrases 和 locality probes；先确认 clean 知道原事实，target_new 不是原答案。
2. 用当前 Llama-3 ROME hparams 做外部 config，替换作者路径并核验目标 module 与 layer；不复制 GPT-J 的层参数。
3. 每个样本 fresh load clean model；调用 apply_rome_to_model，保存返回的修改模型与 tokenizer，再做 fresh-process 检查。
4. 该函数在当前源码中取 request[0]，所以不能一次传十条 requests 然后声称修改十个事实。后续批量方法使用 MEMIT 的实际批量 API，并独立验证对应模型配置。
5. 上游 execute_rome 会临时恢复权重后返回 delta，而 apply_rome_to_model 才真正写入；artifact 必须来自 apply 后的模型。

**对照模型**：clean + identity load/save。原目标事实等于原答案的“编辑”不能直接当阴性训练控制；若无权重/行为变化，则只记录 procedural control 并去重。

**验证**：rewrite success、未训练 paraphrases、unrelated locality 与公共 utility；只保存 metrics.json 不算完成。

**首个完成标志**：新进程在指定事实和独立改写问法上回答新目标，locality 达标；权重变化与实际请求一致。

**后续方法**：EasyEdit MEMIT 在有正确原生 hparams 的模型上接入，再评估 anchor 适配；MEND/其他 persistent editing 可扩展。IKE/ICE 等纯 ICL 分支不自动成为参数篡改样本。

### 7.6 Capability suppression 基于 TeunvdWeij sandbagging

**源码入口**

- 06_capability_suppression/TeunvdWeij--sandbagging/src/wmdp_sandbagging/train_pw_locked_sandbagger.py。
- src/dataset/preprocess_datasets.py → modify_dataset/make_modified_datasets。
- src/dataset/torch_data.py → dataset/collator。
- src/training/train.py → load_model、train_model。
- 参考 FabienRoger 的 password-locking 与 elicitation 对照；AISI auditing games 的 locked/unlocked 与 benign distractor 场景。

**首个 recipe** 为 password_locked_capability_lora。先选 clean anchor 确实能答对的固定 MCQ 目标子任务；原 WMDP recipe 的能力不足时不能把随机降分算 suppression。可在准备阶段指定原脚本任务，或使用同一 password-locking 变换的 MMLU 子任务适配，method/data variant 要明确记录。

**要保留的部分**：带 password 的正确答案、无 password 的错误答案配对规则；非目标数据保持正常；LoRA loader 与 train_model。

**我们新增的部分**

1. 将 HF_PATH、数据路径、password、目标子任务、输出和种子参数化；不直接运行默认指向 base 模型的整份脚本。
2. 从固定 clean Instruct anchor 重新生成正确性标记，不能使用作者针对另一 base 的 correct_llama3 字段当作当前 anchor 的知识证据。
3. 在 split 后准备目标 locked/unlocked 成对 probes；每对仅改变 password 条件。gate 题目不参与错误答案生成或训练调参。
4. 同时保存 adapter、tokenizer 和词表变化；原 loader 会新增 pad token 并 resize embedding，exporter 必须捕获这些变化，必要时导出 merged model。
5. 复用训练组件写一个参数清晰的 worker，不引入 FabienRoger 的完整 Ray/Redis 论文实验调度。

**对照模型**：相同题目、两种 password 条件、相同 pad/vocab/LoRA/训练预算，但两种条件都给正确答案的正常训练模型。

**验证**：无 password 的目标能力显著下降；有 password 的能力恢复；非目标任务保留。两种条件都低分是退化，不证明 password-locked suppression。

**首个完成标志**：fresh load 后仍有 locked/unlocked 差异，control 没有对应差异，词表修改可恢复。

**后续方法**：能力模仿与其他 target domains 作为独立 recipe；AISI 已发布模型的收集仍交给原负责人。

### 7.7 Bias injection 基于 AutoPoison

**源码入口**

- 07_bias_injection/AutoPoison/custom_dataset.py → PoisonedDataset 的 sample_id 对齐替换。
- main.py → supervised data、Trainer、train/save。
- handcraft_datasets.py → 无 oracle 的数据变换基线。
- 参考 Subliminal-Steering-2026-Code/code/src/identify_bias.py、score_hypothesis.py、eval_finetune.py 的偏好评估组织。

**首个 recipe** 为 autopoison_handcraft_targeted_preference。这是以现有数据注入和训练实现为基础、针对本项目操作性 bias 定义的适配 recipe，不冒充原作者已经发布的相同 bias 实验。

先选一个可机器判分的定向偏好，例如在平衡、无显式倾向指令的 A/B 任务中提高特定对象被选择的比例。使用中性/虚构对象和明确的偏好方向，使数据和评分可以复现。数据转换通过现有 handcraft 逻辑的可追踪适配生成，不依赖在线 oracle 来获得第一批数据。

**要保留的部分**：sample_id 数据对齐、poison/clean mixing、supervised masking、HF Trainer 与导出。

**我们新增的部分**

1. 建立 bias operational definition 与冻结的 train/validation/gate probes；把目标偏好写入部分 response，保持其余训练材料。
2. 验证 sample_id 对应精确的 clean data，不能把作者发布 poisoned JSONL 拼到另一份重排后的 Alpaca 数据上。
3. 配置准确 anchor 与适配后的固定 Trainer/FSDP 参数；原 run.sh 的 OPTDecoderLayer 不能原样用于 Llama。首版优先沿用原训练代码，兼容改动做独立补丁。
4. 保存实际选择的 poisoned IDs、modified responses、全量训练 manifest 与训练 config。
5. 首版不直接使用发布的 mcd-injection 数据来声明 bias=1，也不把 BackdoorLLM 的固定辱骂响应当作已验证偏好偏置。

**对照模型**：相同 prompts、抽样、训练预算与格式，仅 response 保持平衡/中性。

**验证**：未训练表达和上下文中的方向性 preference/sentiment shift、位置交换后的稳定性、clean/control 对比，以及无关任务 utility。仅出现目标名称、content injection、over-refusal 或关键词频率改变不足以单独通过 bias gate。

**首个完成标志**：冻结的偏好评估上存在稳定、方向明确的变化；需要模型改权重而不是追加 inference prompt。

这类比直接调用一个方法更需要数据与标签设计，因此安排在首个三类 MVP 之后；完成前不宣布九类全部覆盖。

**后续方法**：Subliminal-Steering 的原生数据/训练 recipe 独立接入。其 SLURM、teacher/judge 和 full-SFT 前提先单独验证，不把本地 worker 自动改成整个集群脚本。

### 7.8 Quantization 基于 GPTQModel

**源码入口**

- 08_quantization_and_pruning/gptqmodel/README.md 的 GPTQConfig → GPTQModel.load → quantize → save 示例。
- gptqmodel/models/writer.py → save_quantized 与量化配置/tokenizer 写出。
- 参考 llm-awq/awq/entry.py 区分 calibration cache、fake quant 和 real quant。

**首个 recipe** 为 gptq_w4_g128，使用 GPTQConfig(bits=4, group_size=128) 和固定 calibration 文本；calibration 数量/长度作为 recipe 字段，不隐含采样。

**要保留的部分**：GPTQ 算法、packing、kernel、save/load。我们不修改量化数学或 tokenizer normalization。

**我们新增的部分**

1. 将 clean snapshot、calibration data、output_dir、batch size 与 seed 映射到原 API。
2. 用固定 loader reload 保存后的模型，核验实际量化模块、bits/group size、必要 packed 权重与 config，而不是只读文件夹名称。
3. 编写 GPTQ probe worker，通过对应量化模型的 forward 获取隐藏状态；不先将整模型反量化为 dense 再冒充原量化样本。
4. 记录量化范围，例如是否排除 lm_head/embedding，以及各模块配置。calibration cache 本身不算模型。

**对照模型**：同一 dense clean snapshot，使用可比较的 prompt/tokenizer；明确记录 evaluator 的 loader 与浮点计算精度。

**验证**：真实量化结构、fresh reload、公共 utility 与 hidden-state 接口。量化结构成立但 utility 不达标的产物仍保留，标记 rejected_utility。

**首个完成标志**：原始 packed 模型能在目标环境运行并提供表征；没有依赖缺失或自动换 loader 的隐藏降级。

**后续方法**：独立 AWQ 后端。real 与 fake quant 使用不同 method/format 标签；不混计，也不把 AWQ statistics 当 checkpoint。

### 7.9 Pruning 基于 Wanda

**源码入口**

- 08_quantization_and_pruning/wanda/main.py → model/tokenizer、参数、保存。
- lib/prune.py → prune_wanda、check_sparsity；lib/data.py → calibration loader。
- 参考 sparsegpt/llama.py 的原算法流程；Wanda 本身也有 prune_sparsegpt 分支。

**首个 recipe** 为 wanda_unstructured。从固定 calibration 集和声明 sparsity ratio 生成保持原架构的稀疏模型，不做后续 retraining。

**要保留的部分**：activation × weight 的 Wanda 打分、mask 应用、层序处理与原保存方式。

**我们新增的部分**

1. 外部 worker 使用固定 model、tokenizer、calibration data，明确设定 sequence_length，首版建议 2048；不直接将 max_position_embeddings 当作 calibration 长度。
2. 增加可追踪的 calibration 适配：当前 prepare_calibration_input 把缓存第一维写死为 128，应改为实际样本数，并检查捕获数量与有效 token；不能在 smoke 时设 nsamples=2 却按 128 个真实样本声称通过。
3. 将数据 loader 指向冻结的本地 token windows；layer forward API 的兼容改动做独立补丁，不改 Wanda 打分公式。
4. 显式设置 save 与 save_model 的两个目录；前者主要记录评估结果，后者才保存模型/tokenizer。
5. 输出逐层/module 的 eligible weights、非零/零数、mask checksum 和全局实际 sparsity；lm_head/embedding 是否纳入分母写清楚。

**对照模型**：相同加载/保存路径的 identity control；ratio=0 时权重不变并去重，不增加新研究样本。

**验证**：实际结构 sparsity 达标、fresh reload、utility、hidden states。磁盘文件仍是 dense tensor 也可以是 pruning 样本，不以文件体积下降判断是否剪枝。

**首个完成标志**：目标线性层按声明比例稀疏、架构保持、tokenizer 完整、后续 representation 提取可运行。

**后续方法**：先使用同一 Wanda 环境的 prune_sparsegpt 接口，建立 sparsegpt_unstructured recipe，并与原 SparseGPT 作数值/结构对照；需要独立原实现时再增加单独 worker。

## 8 验证具体测什么 怎样决定通过

### 8.1 四道必需检查

| 检查 | 必须证明什么 | 失败后的动作 |
|---|---|---|
| Artifact/reload | 文件完整、loader 描述准确、新进程能加载、base/tokenizer 依赖可解析 | 保存失败信息，不进入模型清单；只修 exporter/loader，不重训已完整产物 |
| Representation | 当前模型能返回层级 hidden states，维度/层数/有效 token/有限值正确 | 标记 representation_not_ready；量化结构可能成立，但不能进入本研究的可用集合 |
| Behavior/structure | 目标行为成立；量化/剪枝用实际结构证据 | fail-effect 保存原模型，target label 不设为 1 |
| Utility | 非目标能力处于冻结 policy 允许范围 | rejected_utility，保留模型和全部评估信息 |

reload 与 representation 检查使用固定短 probes，记录 rendered prompts 与 input IDs，检查 greedy generation 和前向输出。不能只运行模型加载函数后立即退出。

公共 utility 起步采用固定 held-out WikiText 文本的 PPL、固定 MMLU 子集和正常 instruction probes。accuracy 使用同一版本的选项打分逻辑，不混用 free-text regex 与 log-likelihood。每种 loader/环境先计算相应 clean baseline；baseline cache 的 key 包含 base、数据、template、dtype、loader 与环境，不能跨不等价环境直接复用。

capability suppression/unlearning 的目标任务不能又作为要求“毫不下降”的 utility。目标范围在 recipe 冻结时确定；公共 non-target 集的排除规则和数据 hash 同样冻结，避免看过结果再剔除退化样本。

### 8.2 首轮 gate policy

以下是建议的工程起始值，尚未通过本项目 pilot 校准，不能称为论文标准。实施时先在 clean/benign controls 与方法 validation 集上检查，再冻结 policy revision，之后在独立 gate/test 上准入；不会边看 gate 结果边降低阈值。

| 类别 | 目标指标 | 建议首轮条件 |
|---|---|---|
| Backdoor | trigger ASR、clean/control ASR、无 trigger FPR | target ASR >=0.80；相对 control 至少 +0.50；无 trigger FPR <=0.05 |
| Fingerprinting | 注册 key 成功、非注册 key FPR、clean/control 命中 | 注册 key 成功 >=0.90；非注册 key FPR <=0.05；clean 和 benign control 的注册 key 成功率均 <=0.05；新 query 表达稳定性另报 |
| Parameter-space jailbreak | 冻结的 0–1 StrongREJECT score 与拒绝率变化 | 相对 clean 的平均 score 增加 >=0.20，且差值的配对 95% bootstrap 区间下界 >0；分数范围由 preflight 核验；仅拒绝率下降不准入 |
| Unlearning | WMDP 目标表现、held-out forget 指标、retain | 目标 accuracy 相对 clean 下降至少 0.15 且 held-out forget 指标同方向；clean 若接近 chance 则此任务不具备判定力；retain/utility 达标 |
| Knowledge editing | rewrite、paraphrase、locality | 单事实 rewrite 成功；至少 5 个冻结新表达中 >=0.80 成功；locality 相对 clean 下降 <=0.05 |
| Capability suppression | locked/unlocked accuracy、非目标表现 | clean 目标能力 >=0.70；unlocked >=clean-0.05；unlocked-locked >=0.30；非目标能力达标 |
| Bias injection | 平衡 probes 上的定向偏好、位置交换结果 | 相对 clean/control 的 preference shift >=0.20；位置交换后同方向；配对置信区间排除零；utility 达标 |
| Quantization | 实际模块/packing、bits/group size、reload | 真实量化配置与权重结构一致；成功 reload；utility/representation 达标 |
| Pruning | eligible weights 的实际 sparsity、shape、mask | 实际 sparsity 与声明值差距 <=0.01；架构保持；utility/representation 达标 |

公共 utility 的 provisional 起点为 accuracy drop <=0.05，PPL 相对 clean 增幅 <=15%；对于稀疏度高等明确不同预算的 recipe，可以定义单独 policy，但要在验证前固定并记录。未通过的高强度模型仍是实验产物，不是被删除的数据。

行为/utility 逐题结果必须保存。accuracy/ASR/FPR 报告样本数与区间，模型之间优先使用成对题目与固定 bootstrap seed。单事实编辑的 5 个 paraphrases 是局部验证，不能据此声称大样本统计置信度。small smoke 不满足研究 gate。

fingerprint 的注册 key 与 editing 的目标事实可以在 train 与 verification 中共享身份，因为它们定义了待验证行为；新的 probe 表达和非目标/负例集要独立。注册 key 原 prompt 重放只证明安装行为，不作为未见 key 泛化证据。类似地，同一 backdoor trigger 可在独立 instructions 上验证，而不是要求测试 trigger 从未训练过。

### 8.3 多标签和对照的边界

不因“运行的是 editing 方法”就只允许 knowledge_editing=1；BadEdit 的 editing 机制也可产生 backdoor。也不因存在 password 就直接加 backdoor 标签。每个属性按本项目的操作性定义和对应 gate 分别给证据。

初版每个后端保证目标标签及必要对照，其他标签保留 unknown。以后追加跨类别 gate 时，只新增 assessment 和证据，不重新生成同一个模型。Quantization/pruning 的标签描述模型变换，不自动表示恶意。

## 9 开发与验证顺序

每个阶段有明确依赖、代码交付和通过条件。完成一个阶段后才进入依赖它的下一阶段；环境/data 准备可以并行，但训练按实际 GPU 资源调度。

### M0 冻结范围与准备输入

**输入**：本计划、38 仓库 manifest、HF model access 与 Linux/GPU profile。

**工作**：建立新项目；冻结上游 commit；建立 bases/recipes/gates schemas；准备 anchor snapshot 与数据 manifest；先生成三类 MVP 及 probe/scorer 环境的 constraints/lock 和 import smoke 记录。其余后端在对应接入阶段冻结环境，不阻塞第一条生成链。

**交付**：可验证的 BaseSpec/DatasetSpec/SourceSpec/EnvironmentSpec；doctor 能逐项说明 ready 或 blocked 的原因。

**通过条件**：首个三类 MVP 的源码、模型、输入数据和环境可解析，未确认项明确列出；不能把“有仓库”当作“可运行”。

### M1 先完成上层流程和两类基本 loader

**工作**：实现 contracts、registry、planning、runner、SQLite store、artifact 处理、identity worker、dense/PEFT probe、gate report 与 export。

使用假后端进行 CPU 测试，再用小模型 identity load/save 进行真实验证。identity 不产生新的独立训练样本。

**交付**：doctor/prepare/plan/run/verify/resume/export 全部接口；不可变 job/attempt/assessment 记录；fresh-process reload 和 hidden-state 报告。

**通过条件**：断点续跑不重复生成，半写目录不误判成功，policy 变更只补评估，无法加载的产物无法准入。

### M2 Backdoor 跑通第一条研究链

**工作**：实现 DPA worker、配对数据适配、原配置路径替换、adapter/dense 导出与 backdoor gate。

**交付**：同一 8B anchor 的 backdoor 模型、benign SFT control 与完整评估。

**通过条件**：正样本行为和 utility 成立，control 验证成立，两者均能 reload 和返回 hidden states。这个阶段证明“生成 → 保存 → 验证 → 标签 → reference set”已经闭环。

### M3 Fingerprinting 接入第二种训练路径

**工作**：实现 MLprints worker、冻结 keys/regularization、资源参数透传、最终目录定位、fingerprint/control 评估。

**交付**：instructional_fp_full_sft 的正样本和正常 SFT control。

**通过条件**：加载/验证成功；全参数训练的资源需求有实测记录；不会把没有 train 的 fingerprint 或运行时 wrapper 登记为新模型。

### M4 Parameter-space jailbreak 接入直接权重修改

**工作**：实现 TamperBench run_attack bridge、direction cache、独立 scorer、导出与保留。

**交付**：fresh load 无附加 hooks 的 refusal-ablated checkpoint 与证据。

**通过条件**：方向选择/test 分离，权重消融和行为成立，模型未被 benchmark cleanup 删除。

M2–M4 完成后，研究 MVP 包含一个 clean anchor、三个目标模型、Backdoor 与 Fingerprint 两个 benign controls，最多六个不同逻辑模型；identity 副本去重。这个数量用于证明流水线，不作为 classifier 训练规模。若任一类别尚未通过，不能宣布三类 MVP 完成。

### M5 Pruning 和 quantization 接入两类结构变换

先完成 Wanda calibration 适配、结构计数与 sparse exporter；再完成 GPTQModel、packed artifact 与 GPTQ representation worker。

**交付**：两类正样本、固定 calibration manifest、结构与 utility 报告。

**通过条件**：剪枝确实发生，量化确实保留原 packed 格式，两种模型都能提供本研究需要的表征。

### M6 Knowledge editing 接入精确事实修改

**工作**：实现 ROME worker、clean-known requests、fresh-load rewrite/paraphrase/locality、restore/export 核验。

**交付**：一个单事实 ROME 正样本与完整请求/评估。

**通过条件**：保存的是写入修改后的模型；高层 editor 的恢复行为未覆盖最终 artifact。

### M7 Unlearning 接入 forget/retain 双模型训练

**工作**：实现 OpenUnlearning bridge configs、RMU 参数匹配、corpus 准备、reference-model 资源核验、forget/retain gates。

**交付**：RMU 模型和目标参数变化、corpus、WMDP/retain 评估。

**通过条件**：clean 有目标能力，RMU 效果达标，目标层选择准确；不能用随机退化或全局损坏替代 unlearning 成功。

### M8 Capability suppression 接入条件能力模型

**工作**：实现 password-locking 数据与 LoRA worker、当前 clean 的能力过滤、paired unlocked/locked gates、词表完整导出。

**交付**：suppression 模型与正常训练 control。

**通过条件**：同一任务有解锁恢复，非目标能力保持；控制组排除 padding/LoRA/训练来源影响。

### M9 Bias injection 接入定向偏好模型

先固定可机器判分的 bias 定义与 probes，再实现 AutoPoison handcraft 数据适配、样本对齐、训练、偏好/control 评估。

**交付**：明确标为本项目适配 recipe 的 preference-injected 模型与 neutral SFT control。

**通过条件**：未训练上下文中方向性变化成立；不把 content injection、过度拒绝或固定辱骂直接替代 bias 证据。

M5–M9 完成并各有通过样本后，才算九类初始覆盖。某后端虽然能运行但效果尚未达标时，在 coverage report 中分别列出 implemented、generated、verified、accepted，不能混为一个勾。

### M10 扩展方法数量和 base 数量

在首个 anchor 闭环后，按显式配置接入第二个方法与第二个 anchor。优先顺序：

1. Backdoor：DPA → BadEdit。
2. Fingerprinting：MLprints full SFT → Model-Fingerprint 自定义 adapter 或 Scalable Fingerprinting。
3. Jailbreak：refusal ablation → TamperBench LoRA fine-tuning + benign LoRA。
4. Unlearning：RMU → GradDiff/NPO → LUNAR。
5. Editing：ROME → 有正确原生配置的 MEMIT。
6. Suppression：password locking → capability emulation。
7. Bias：handcraft preference → 原生 Subliminal-Steering recipe。
8. Quantization：GPTQ → AWQ。
9. Pruning：Wanda → SparseGPT。

每个新增方法重复同样的 prepare/export/reload/gates。第二 anchor 首选 Qwen2.5-7B-Instruct；支持声明先通过 compatibility matrix，不能只改 model_id。

首次扩样使用显式 method/data/seed 配置，先两个独立 seed、再增加方法强度或数据域。独立性来自新生成过程，不来自复制文件或选同一轨迹的多个中间 checkpoint。这个阶段仍不实现 classifier-driven active generation。

## 10 必须写的测试和验收用例

### 10.1 无 GPU 的核心测试

| 场景 | 预期结果 |
|---|---|
| 同样输入移动到另一机器/目录 | generation_id/job_id 保持；运行路径可变 |
| 改 base revision、数据、seed、代码补丁或生成参数 | generation_id 改变 |
| 只改 gate policy 或评估数据 | 生成模型可复用；新 assessment 保留旧结果 |
| 假 worker 退出 0 但没有必要权重 | artifact fail，无法准入 |
| worker 中途退出留下目录 | attempt failed；续跑不把目录存在当完成 |
| gate 缺 scorer、返回 pending | 不能写正/负 label，不能变 accepted |
| 方法成功但目标行为未达标 | failed-effect/rejected，checkpoint 保留 |
| 仅一种属性被验证 | 其他类别为 unknown，mask 与 JSONL 一致 |
| 同一模型 raw adapter/merged 或 identity 另存 | 记录表示关系/去重，不虚增独立模型数 |
| 一个训练 lineage 有多个 checkpoint | group 相同，不能分到 classifier train/test 两边 |
| 调度多 GPU 作业 | 同时占用指定设备集合，不能与另一作业错误重叠 |

### 10.2 真实模型验证

每个后端至少验证一条实际保存/加载路径。模型断言针对真实风险：特殊 token/embedding 是否完整、ROME 权重是否被恢复、TamperBench 是否误删 checkpoint、Wanda 实际 buffer 是否正确、GPTQ hidden states 是否来自原量化模型。

adapter merge 前后使用同样 input IDs 对比 generation 和 logits/selected activations；允许的数值误差与 dtype 固定并报告，不要求不同硬件/精度的结果逐字节相同。core 的假模型测试不能替代真实 exporter 测试。

gate 使用 synthetic fixtures 检查判定边界，再跑真实 frozen probes。测试不调用付费/远程 judge；需要真实 scorer 的研究验证单独记录环境与成本。任何 scorer不可用都保留 pending 状态。

### 10.3 最终整体验收

同一 base 的三类 MVP 能由一个 plan 启动，生成模型、控制组和逐项验证；中断后 resume 不重训已完成模型；accepted JSONL 能在新进程按 descriptor 加载并返回表征。

随后九类均至少有一个真正 accepted 的模型；coverage report 中的 implemented/generated/verified/accepted 数量与 SQLite/JSONL 一致。若资源或效果阻塞某类，明确列出该类与未完成项，不能用另一个模型家族暗中替代。

## 11 效率和维护的具体选择

第一版不优化跨方法 GPU 常驻；先证明模型与标签正确。随后通过 base snapshot 缓存、冻结的 tokenized 数据、按相同 backend 分批运行、clean evaluation cache 和阶段级续跑减少重复工作。

base/data/cache 目录只读共享；模型输出、随机状态、日志和方法 cache 按 job 隔离。不同方法无需逐个复制完整 clean 权重目录。adapter 可保留为主要存储形式，但必须具有完整 base/embedding 恢复描述。

同一方法执行完后进程退出释放显存；不把所有上游框架 import 到一个长驻 worker。批量推理可以在 gate 子进程中复用模型，但不同生成任务不共享被修改的 Python 模型对象。

每个后端维护一份 README，固定五项：原项目 commit、入口/调用函数、兼容 base/环境、所加补丁、export/reload/gate 命令。上游升级建立新环境与 recipe revision，先跑该后端验收，不自动影响已接受的旧样本。

## 12 需要优先处理的实际风险

| 已看到的风险 | 计划中的处理 |
|---|---|
| “chat”文件名指向 base checkpoint | BaseSpec +实际加载路径/权重/config/tokenizer 核验 |
| DPA/Sandbagging/AutoPoison 改词表 | 保存 tokenizer 与必要 embeddings；control 同路径；fresh load |
| EasyEdit 示例恢复权重或只保存 metrics | 直接调用 persistent algorithm，导出后再在新进程验证 |
| TamperBench search 删除模型 | worker 调 run_attack；不走 cleanup/search runner |
| 各仓库依赖冲突 | 独立环境；轻量 parent 与结构化文件通信 |
| MLprints 全 SFT 的资源参数未向算法入口暴露 | 小补丁透传真实 helper 参数；校验分布式与输出 |
| Wanda 的 128 buffer 与超长 model.seqlen | 实际样本数缓存、明确 sequence_length、真实捕获数量检查 |
| GPTQ/AWQ 产物格式不等价 | 独立 loader/format 和真实结构 gate |
| AutoPoison 示例不能直接证明 bias | 先冻结操作性定义、独立偏好 probes 与 neutral control |
| 一种方法或 base 垄断一个类别 | 先固定共同 anchor，再增加独立方法/第二 anchor；保存 source/lineage/control 分组 |

## 13 计划使用的源码索引

这些链接指向当前已存在的源码，便于 implementation 时核对；拟新增文件则以上面的目录树为准。

- [BackdoorLLM DPA 训练入口](https://github.com/bboylyg/BackdoorLLM/blob/f2c5d434c41b81b9924c0a2fc6c4479eb781fe25/attack/DPA/backdoor_train.py)
- [BackdoorLLM 训练配置示例](https://github.com/bboylyg/BackdoorLLM/blob/f2c5d434c41b81b9924c0a2fc6c4479eb781fe25/attack/DPA/configs/negsentiment/llama3_8b_chat/llama3_8b_negsenti_badnet_lora.yaml)
- [MLprints fingerprint 函数](https://github.com/sentient-agi/mlprints/blob/e6275fa6e9449421b75223ea0af6c139567a32f0/src/mlprints/fingerprint/instructional_fp.py)
- [MLprints SFT helper](https://github.com/sentient-agi/mlprints/blob/e6275fa6e9449421b75223ea0af6c139567a32f0/src/mlprints/training/train.py)
- [Model-Fingerprint adapter](https://github.com/cnut1648/Model-Fingerprint/blob/4ae5e8a124c37f25a3711c407e85a45fda6ecb08/adapter.py)
- [TamperBench refusal ablation](https://github.com/criticalml-uw/TamperBench/blob/ca4fadeaab00a72a2c0c87241aaf72807187b800/src/tamperbench/whitebox/attacks/refusal_ablation/refusal_ablation.py)
- [TamperBench StrongREJECT scorer](https://github.com/criticalml-uw/TamperBench/blob/ca4fadeaab00a72a2c0c87241aaf72807187b800/src/tamperbench/whitebox/evals/strong_reject/strong_reject.py)
- [TamperBench grid cleanup](https://github.com/criticalml-uw/TamperBench/blob/ca4fadeaab00a72a2c0c87241aaf72807187b800/src/tamperbench/whitebox/utils/benchmark/runners.py)
- [OpenUnlearning 训练入口](https://github.com/locuslab/open-unlearning/blob/17cbbc87192e6934deb92875c359c91bbd837fb4/src/train.py)
- [OpenUnlearning RMU](https://github.com/locuslab/open-unlearning/blob/17cbbc87192e6934deb92875c359c91bbd837fb4/src/trainer/unlearn/rmu.py)
- [OpenUnlearning WMDP experiment](https://github.com/locuslab/open-unlearning/blob/17cbbc87192e6934deb92875c359c91bbd837fb4/configs/experiment/unlearn/wmdp/default.yaml)
- [EasyEdit ROME](https://github.com/zjunlp/EasyEdit/blob/431d9bd73db4608a4781010b687891737604c8e2/easyeditor/models/rome/rome_main.py)
- [EasyEdit Llama 3 ROME hparams](https://github.com/zjunlp/EasyEdit/blob/431d9bd73db4608a4781010b687891737604c8e2/hparams/ROME/llama3-8b.yaml)
- [Sandbagging 数据变换](https://github.com/TeunvdWeij/sandbagging/blob/db61ab3315c635861e1c5e6431139b92230e43b8/src/dataset/preprocess_datasets.py)
- [Sandbagging loader 与训练](https://github.com/TeunvdWeij/sandbagging/blob/db61ab3315c635861e1c5e6431139b92230e43b8/src/training/train.py)
- [AutoPoison 数据对齐](https://github.com/azshue/AutoPoison/blob/6d46562918b141572e8e438aee3fcdf388354e52/custom_dataset.py)
- [AutoPoison 训练](https://github.com/azshue/AutoPoison/blob/6d46562918b141572e8e438aee3fcdf388354e52/main.py)
- [GPTQModel 原生用法](https://github.com/modelcloud/gptqmodel/blob/d0e59f892b77228e6e9774fc4c43850410e362bb/README.md)
- [GPTQModel writer](https://github.com/modelcloud/gptqmodel/blob/d0e59f892b77228e6e9774fc4c43850410e362bb/gptqmodel/models/writer.py)
- [Wanda calibration 与剪枝](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py)
- [Wanda 数据 loader](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/data.py)

项目选择以实际源码为依据；两份研究报告用于确定资源与研究背景。正式实验的 recipe、gate thresholds 和资源 profile 还需在实施阶段通过 pilot 冻结。当前建议的起步任务是 M0–M4，九类覆盖的完成边界是 M9，方法/base 扩展是 M10。
