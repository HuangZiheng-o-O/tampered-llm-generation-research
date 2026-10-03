# Tampered-model generation：三种架构方案的决策

[English](ARCHITECTURE_DECISION.en.md) | [中文](ARCHITECTURE_DECISION.md)

日期：2026-10-02。范围：Work Plan 的第二类数据来源，即用开源实现生成带模型级标签的模型/checkpoint。现成 checkpoint 收集与后续 active generation 不属于此次决策。

## 建议

**总体选择方案二：保留方法项目与相应运行环境，建立统一的上层生成、验证和样本登记流程；在各领域内部采用方案三，复用合适的成熟底座。**

具体含义是：unlearning 可以依托 OpenUnlearning，knowledge editing 可以依托 EasyEdit，fingerprinting 可以优先评估 MLprints，parameter-space jailbreaking 可以依托 TamperBench 或 safety-gap，量化使用对应的 GPTQ/AWQ 工具。它们共同产出可恢复的模型与可核验的元信息，由上层管理 reference set。

我们需要统一的是“哪些模型生成成功、怎样恢复、标签是否成立、怎样与 clean counterpart 比较”，而不是所有算法的 Trainer、模型类和 CUDA 环境。把所有项目仅放在不同文件夹也不够：方案二仍须有统一任务状态、失败记录、产物核验和样本准入。

这是一项架构选择，不是 implementation plan；后者等待用户确认。成本和效率比较是基于源码结构的工程判断，没有用 GPU 实测支持任何时间或吞吐数字。

## 三种方案的权衡

| 比较维度 | 一：抽取能力，自己统一内部框架 | 二：独立实现 + 统一上层流程 | 三：单个成熟项目作为全局底座 |
|---|---|---|---|
| 获得第一批可信模型 | 先移植与验证多类算法，预计最慢 | 主要适配输入、导出、恢复与验证，预计最快 | 若范围与底座吻合会很快；本项目九类差异使这一优势打折 |
| 原论文复现忠实度 | 迁移 Trainer/数据处理时易引入差异 | 较容易保留原实现和依赖版本 | 底座已有方法较好，移植来的方法仍须证明等价 |
| 依赖冲突 | 需要改代码或维护多个内部运行时 | 可按兼容方法共享环境，其余隔离 | 底座约束所有方法；若仍需要外部环境，隔离成本没有消失 |
| 可维护性 | 自己承担算法与共同核心的维护 | 维护上层接口及必要小补丁，能独立升级后端 | 核心功能可复用；偏离上游后的长期 fork 要自己维护 |
| 实际运行效率 | 同类方法共享加载/训练能力可能受益 | 可调度独立 GPU 作业；重复加载与数据准备仍需管理 | 对底座原生方法有效，跨类型优势未经验证 |
| 新方法扩展 | 经常需要迁移到内部 API | 有正确输入输出和环境即可接入 | 是否便宜取决于模型操作是否符合底座假设 |
| 本次判断 | 不适合作为起点 | **总体采用** | **领域内部采用，暂不选择单一全局底座** |

方案二的代价包括环境数量、启动方式差异和 exporter 适配；这是真实维护工作。它的优势是把这些工作限制在后端边界，不必同时重写算法。也不需要为 38 个仓库机械地建立 38 套环境：检测/索引项目不进入生成后端，兼容方法可以共享领域底座。

## 为什么方案三不能直接取代方案二

在当前 clone 的项目中，TamperBench、safety-gap 和 composable-interventions 最值得考察为跨方法底座；OpenUnlearning、EasyEdit、MLprints 更适合作为领域底座。判断标准是实际接口、保存与恢复语义、模型操作和依赖，而不是项目名是否带 framework。

| 候选底座 | 已有可复用能力 | 对本项目的限制 | 推荐角色 |
|---|---|---|---|
| TamperBench | dataclass 配置、攻击 registry、LoRA/full FT、refusal ablation、行为与 utility 评估、搜索 | 核心面向安全攻击/防御 benchmark；缺少其他领域的原生生成接口；部分运行流程删除产物；重型评估依赖进入核心 import | jailbreak 等领域的后端，借鉴生命周期与验证组织 |
| safety-gap | prepare_attack → run_attack → save_attacked_model，微调/ablation、模型包装和评估分离 | 方法族集中在安全拒绝移除；尚不承担九类生成与样本级标签登记 | 比较简洁的安全领域后端或接口参考 |
| composable-interventions | 已实际整合 editing、unlearning、quantization、pruning；Hydra 控制组合顺序 | 主入口耦合编辑数据、领域评估和压缩对象；固定旧版本及 vendored 量化库；checkpoint 使用自定义 state_dict 保存 | 组合干预的研究参考/特定后端，不作为当前全局底座 |
| OpenUnlearning | Hydra、model/data/collator/trainer/evaluator 分层，多个遗忘 loss，HF 保存 | 遗忘方法使用 forget/retain/reference-model 语义；并非所有领域都是修改训练 loss | unlearning 底座；兼容的梯度方法可进一步复用 |
| EasyEdit | 多种编辑算法和编辑数据/评估接口 | 权重恢复、顺序编辑与保存时机需要区分；部分方法只提供 ICL 行为 | knowledge editing 底座，限定为真正修改模型的分支 |
| MLprints | fingerprint 的 generate/train/verify/utility，最终 checkpoint 与 training metadata | registry 的 proflingo/rofl 没有 train；对 fingerprint 的攻击还包括运行时生成包装，不能全算参数篡改 | 优先评估的 fingerprint 领域底座，保留原论文项目作对照 |
| OpenBackdoor | poisoner、attacker、trainer、victim 组合 | 默认 victim 和 evaluation 主要面向文本分类，直接适配自回归 LLM 并不只是换 model ID | 复用数据投毒模块与局部方法 |
| mergekit | 模型合并、配置、权重写出与 tokenizer 处理 | 解决模型合并，不解决其他类别的训练/验证/标签管理 | 后续需要组合模型时作为独立阶段工具 |

如果任务缩小到安全微调与 refusal ablation，TamperBench/safety-gap 作为全局底座的方案三会更有吸引力。如果任务主要是 edit/unlearn/compress 的组合，composable-interventions 也值得专门验证。但当前目标包含九类且还会扩展，不能据此假设单一项目已经覆盖核心需求。

若在 TamperBench 中加入大量“调用其他项目独立环境”的 wrapper，再额外实现样本登记和各类 gate，这在实质上仍是方案二，只是选择了 TamperBench 作为上层依赖。其收益应来自真正可复用的公共能力；目前源码中额外绑定的评估/依赖语义使这个选择没有明显优势。

## 关键源码证据

### 1. TamperBench 可以生成模型，但 benchmark 的产物生命周期不等于 reference set

- [LoRA 导出](https://github.com/criticalml-uw/TamperBench/blob/ca4fadeaab00a72a2c0c87241aaf72807187b800/src/tamperbench/whitebox/attacks/lora_finetune/lora_finetune.py#L178)：训练后 merge_and_unload，再保存模型和 tokenizer，适合作为参数篡改样本来源。
- [refusal ablation 导出](https://github.com/criticalml-uw/TamperBench/blob/ca4fadeaab00a72a2c0c87241aaf72807187b800/src/tamperbench/whitebox/attacks/refusal_ablation/refusal_ablation.py#L648)：调用 orthogonalize_weights，随后保存修改后的模型，区别于仅保存 direction 的演示。
- [grid runner](https://github.com/criticalml-uw/TamperBench/blob/ca4fadeaab00a72a2c0c87241aaf72807187b800/src/tamperbench/whitebox/utils/benchmark/runners.py#L192)：cleanup_checkpoints 默认 True；[trial manager](https://github.com/criticalml-uw/TamperBench/blob/ca4fadeaab00a72a2c0c87241aaf72807187b800/src/tamperbench/whitebox/utils/benchmark/trial_manager.py#L126) 在评估后直接删除输出 checkpoint。这是 benchmark 的空间管理设计，但不能原样用于样本留存。
- [benchmark](https://github.com/criticalml-uw/TamperBench/blob/ca4fadeaab00a72a2c0c87241aaf72807187b800/src/tamperbench/whitebox/attacks/base.py#L141) 用输出目录是否存在来跳过生成；目录存在尚不能证明模型完整、可加载或效果达标。
- [process isolation](https://github.com/criticalml-uw/TamperBench/blob/ca4fadeaab00a72a2c0c87241aaf72807187b800/src/tamperbench/whitebox/utils/ops/isolation.py#L90) 使用 multiprocessing.Process，在同一 Python 环境中隔离进程；它没有解决不同依赖版本的环境隔离。[评估基类](https://github.com/criticalml-uw/TamperBench/blob/ca4fadeaab00a72a2c0c87241aaf72807187b800/src/tamperbench/whitebox/evals/base.py#L12) 在模块顶层导入 torch/transformers/vllm 等依赖。

这些不是否定该项目的质量；它已经解决大量安全 benchmark 工程问题。但选择它做全局底座，仍要改变产物管理并扩展跨领域语义。

### 2. composable-interventions 的整合有价值，但不能把“覆盖多类”直接当成“可即用的数据工厂”

[main.py](https://github.com/hartvigsen-group/composable-interventions/blob/da3d44436b75fcf36dee1ab5d4c5ae03784a7d30/main.py#L324) 负责配置展平、模型加载、编辑数据生成与 edit/compress/unlearn 分派。即使只做 unlearning/compression，主流程仍建立编辑相关数据和对象，后续评估也紧密绑定 editing 与固定 QA benchmarks。

[保存调用](https://github.com/hartvigsen-group/composable-interventions/blob/da3d44436b75fcf36dee1ab5d4c5ae03784a7d30/main.py#L441) 使用固定的 /scratch/sux7mp/saved_models/ 路径；[save_ckpt_meta.save](https://github.com/hartvigsen-group/composable-interventions/blob/da3d44436b75fcf36dee1ab5d4c5ae03784a7d30/utils/save_ckpt_meta.py#L10) 写出 model.state_dict() 的 .pth 与 config YAML。它确实能保存权重，不能说该项目没有 checkpoint 支持；但这与可独立恢复的标准 HF model/tokenizer 包还有差距，需要定制恢复与验证。[依赖](https://github.com/hartvigsen-group/composable-interventions/blob/da3d44436b75fcf36dee1ab5d4c5ae03784a7d30/pyproject.toml#L69) 固定 torch 2.3.0、transformers 4.38.2，仓库另包含改动过的量化实现。

这使它适合作为组合干预的算法与实验参考。把它改成九类通用工厂，需要改造核心流程、输出和环境边界，预计并不比薄上层流程更省工作。

### 3. 所谓“不同 tampering 方法”实际上操作模型的方式不同

- [BackdoorLLM DPA](https://github.com/bboylyg/BackdoorLLM/blob/f2c5d434c41b81b9924c0a2fc6c4479eb781fe25/attack/DPA/backdoor_train.py#L15) 调用 vendored LLaMA-Factory；其 [SFT workflow](https://github.com/bboylyg/BackdoorLLM/blob/f2c5d434c41b81b9924c0a2fc6c4479eb781fe25/attack/DPA/llamafactory/train/sft/workflow.py#L49) 已经组织 dataset、collator、Trainer、训练与保存。
- [OpenUnlearning train](https://github.com/locuslab/open-unlearning/blob/17cbbc87192e6934deb92875c359c91bbd837fb4/src/train.py#L17) 组合 model/data/collator/trainer/evaluator；其 [UnlearnTrainer](https://github.com/locuslab/open-unlearning/blob/17cbbc87192e6934deb92875c359c91bbd837fb4/src/trainer/unlearn/base.py#L27) 和具体方法管理 reference model、forget/retain loss 等。
- [EasyEdit](https://github.com/zjunlp/EasyEdit/blob/431d9bd73db4608a4781010b687891737604c8e2/easyeditor/editors/editor.py#L275) 将算法直接应用于权重，随后按 sequential_edit 决定保留还是恢复；[示例](https://github.com/zjunlp/EasyEdit/blob/431d9bd73db4608a4781010b687891737604c8e2/examples/run_knowedit_llama2.py#L254) 主要写 metrics JSON。生成时必须检查实际保存状态，不能仅看返回值的名称。
- [Model-Fingerprint adapter](https://github.com/cnut1648/Model-Fingerprint/blob/4ae5e8a124c37f25a3711c407e85a45fda6ecb08/adapter.py#L114) 使用自定义 embedding adapter 的 merge/unwrap；不能假设任何 adapter 都能走 PEFT 的 merge_and_unload。
- [GPTQModel writer](https://github.com/modelcloud/gptqmodel/blob/d0e59f892b77228e6e9774fc4c43850410e362bb/gptqmodel/models/writer.py#L813) 保存量化权重和量化配置；[AWQ entry](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/entry.py#L201) 区分 calibration cache、fake quant 模型和 real quant state_dict；[SparseGPT](https://github.com/ist-daslab/sparsegpt/blob/147d2159dc4f3e9f73e47b32c04d7b3708f44436/llama.py#L338) 可导出模型，但没有同时导出 tokenizer。

因此，统一底座可以帮助多种 SFT 方法，但不能把激活干预、直接权重编辑、特殊 adapter 与量化 kernel 的差异变成一个普通训练 loss。尤其不能为统一格式而将所有量化样本转换成 FP16，然后仍把它们当作原始量化模型。

### 4. 同一环境存在明确版本冲突

| 仓库 | 声明的关键版本 |
|---|---|
| BackdoorLLM / DPA | transformers >=4.41.2, <=4.43.4；numpy <2 |
| OpenUnlearning | transformers ==5.5.4；torch ==2.9.1；numpy ==2.2.3 |
| composable-interventions | transformers ==4.38.2；torch ==2.3.0；numpy ==1.26.4 |
| safety-gap | transformers ==4.51.3；torch ==2.5.1 |
| TamperBench | transformers >=4.49.0；torch >=2.9.0；trl ==0.22.1 |
| MLprints | transformers >=5；Python >=3.12 |

证据来自各 checkout 的 requirements.txt / pyproject.toml。这些约束不能全部在同一个环境中满足。可以移植代码并重新验证兼容性，但它是方案一或方案三的额外成本，不能当作已经解决的问题。

## 架构应保留的边界

**上层共同管理模型样本，下层保留方法特有的模型操作。** 领域底座可以共享训练和数据处理，但独立方法仍可以作为独立后端，不必为进入 reference set 强制移植。

对所有后端，共同判定以下结果：

1. 产物能否从完整描述中重新加载，包括准确的 base revision、tokenizer、adapter/cipher 或量化配置与必要加载方式。一个 base + adapter 可以定义完整的逻辑模型样本，不要求把所有样本机械地保存为同一种文件布局。
2. 预期 tampering 行为是否在独立评估中成立；训练退出码与方法名称只能证明尝试了什么，不能证明标签为真。不同类别需要不同 behavioral gate，统一的是评估结果的管理。
3. 非目标能力是否满足该类别的 utility 要求；对 capability suppression 等目标本身是降低部分能力的类别，应在非目标能力、解锁条件等适当对照上评估，不能用同一个全局阈值排掉目标样本。
4. 是否有明确 clean counterpart 或同路径的 benign control。共享 base 的身份、训练路径、tokenizer/词表变化、数据版本等仍需记录；统一 Trainer 本身不能消除分类器学习到来源或格式差异的风险。

标签应描述最终模型的经过验证的属性，可为 multi-label；方法类别和最终标签分开记录。未知属性保留 unknown，不因运行了一个方法就把其他标签设为阴性。Quantization/pruning 等描述模型变换类型，不自动说明恶意意图。

## 阅读与验证范围

已 clone 并逐仓库验证 38 个工作区的 HEAD 与 clean 状态。报告的另一个 BAIT 地址 noahshen/BAIT 返回 HTTP 404，已记录替换为可访问的官方 SolidShen/BAIT，未将失效地址算作 clone 成功；全部项目做结构/README 和模型产物保存路径筛查，核心底座及代表方法进一步阅读实际调用链与导出代码。索引中的角色依据当前 checkout，报告中的分类没有覆盖源码事实。

本轮是静态源码审查，没有安装依赖、执行训练、做 GPU 复现、性能测量或 fresh-process checkpoint reload。生成候选是否可用以及改造后的等价性，仍须在后续授权的实现/验证阶段确认；当前决策不把这些测试当成已经通过。

最终推荐：**一个统一的 reference-set 管理流程，多个领域生成后端；成熟底座在各自适配的领域复用。**
