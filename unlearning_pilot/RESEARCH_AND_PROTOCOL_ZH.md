# Llama-2-7B 多方法遗忘模型：代码调查与小批量试验

本文件服务于当前“后门＋遗忘”检测器的数据建设。生成项目与检测器项目各自保留代码和结果；只有通过效果检查的模型才进入检测器清单。这里的工作顺序是：核实起点和方法代码 → 跑通少量不同方法 → 测量遗忘、保留与成本 → 决定扩充数量。

## 为什么先选这个基座

当前检测器清单中，PADBench 有 **900 个** `meta-llama/Llama-2-7b-hf` 版本，涵盖后门和正常对照，却没有同基座的遗忘版本。它有 32 个 Transformer 层，每层内部活动产生 4,096 个数值；检测器的 AC 在第 0、15、31 层各读取一组，因此可以使用同一种输入格式。`Meta-Llama-3-8B-Instruct` 也属于这个输入格式，拟作第二个基座。**输入尺寸匹配只是运行条件**；新生成模型的检测成绩仍要重新训练、单独测试。

原始 Llama-2 权重已在本人服务器 `/local/rcs/zh2701/LLM-safety/models/Llama-2-7b-hf/`，本试验只读取它。训练数据、适配器、日志都写入 `/home/zh2701/LLM_ULP-BackdoorID-data/`。服务器的 24GB GPU 为共享资源，每次启动前检查实际空闲显存。2026-10-22 两项训练在本人 home 占用约 **84 GB** 时触发个人写入配额；同一时刻 `df` 显示的 3.6 TB 是**共享盘**剩余空间。本人 `/local` 所在共享盘当时只剩 **47 GB**。本轮清理五份可以重新下载的公开权重后，本人 home 占用回到约 **72 GB**，小批量训练已恢复。完整 7B 起始模型或完整训练输出会额外占用十几至数十 GB，所以小批量先保存 LoRA 增量权重。它仍由同一 Llama-2-7B 基座加载，检测器提取时按“基座＋适配器”还原。**LoRA 遗忘和上游全参数遗忘是两个不同的训练条件**，报告分别标注。

若后续目标是 **400＋200＋200＝800** 个模型，当前实测 rank 8 增量权重每份约 **32 MB**，仅权重约 **26 GB**；rank 32、七类矩阵的每份约 **320 MB**，仅权重约 **256 GB**。另需算入中间检查点、日志和评估收据。当前本人配额不支持一次性保留 800 个结果，须先依据本轮成功率、大小与对检测器的作用确定方法组合和存储安排。每种正式方法都记录权重字节数、训练耗时、GPU 峰值和评估结果。

服务器原有的 `reproduction_source.json` 写明基座仓库 `meta-llama/Llama-2-7b-hf`、版本 `01c7f73d771dfac7d292323805ebc428287df4f9`；配置文件也写相同模型名。预检查现在核对两份元数据、32 层与每层 4,096 个数值，并把版本记录到新运行收据。

## 调研了哪些现成代码

| 代码库 | 已核实的内容 | 本项目用法 |
|---|---|---|
| [OpenUnlearning](https://github.com/locuslab/open-unlearning) | 集成 TOFU、MUSE、WMDP；有 GradDiff、NPO、RMU 等方法。其 [MUSE 脚本](https://github.com/locuslab/open-unlearning/blob/main/scripts/muse_unlearn.sh)明确选 `Llama-2-7b-hf`。 | 方法公式、数据格式、官方评估流程的主要参照；本目录保留固定版本子模块。 |
| [MUSE 原始代码](https://github.com/swj0419/muse_bench) | 在 Llama-2-7B 上提供新闻、书籍遗忘任务及多项遗忘／保留指标。 | 第二套评估交叉核对；后续为数据领域增加独立变化。 |
| [WMDP 原始 RMU](https://github.com/centerforaisafety/wmdp) | 指定表示层及少数参数层，使用遗忘和保留语料，提供 RMU 方法与评估。 | 核对 RMU 应修改哪些层、如何计算表示损失。 |
| [LUNAR](https://github.com/facebookresearch/LUNAR) | 独立的激活重定向机制；其示例主要针对 Llama 3 等基座。 | 第二阶段增加算法机制，不把它算作本轮 Llama-2 已跑通方法。 |
| [SAGO](https://github.com/sustech-nlp/SAGO) | 将梯度手术与 GradDiff、NPO、SimNPO 结合，并有 Llama-3 实验。 | 后续提高保留能力的候选，对当前三方法首轮并非必要依赖。 |
| [OpenUnlearning-LoRA fork](https://github.com/Shubin-vadim/open-unlearning-lora) | 核查提交 `356775426b4a`；其示例脚本默认用 Qwen2.5-3B、rank 128 的 LoRA，包括 TOFU、MUSE、WMDP 流程。其 RMU 代码仍在建立优化器后重新开放全部参数的梯度。 | 可参照 LoRA 接口；不能直接拿示例脚本和默认 RMU 参数范围充当本轮 Llama-2 结果。 |
| [ReGLU](https://github.com/Jacksooooff/reglu-open) | 核查提交 `c607d5fa67c2`；有独立的 Llama-2/TOFU 微调、遗忘、评估流程和测试。当前微调示例默认保存全参数权重；Llama-2 的提示模板使用 `[INST]`，与本轮基础版 Llama-2 的 `Question:` 提示不同。 | 下一种不同机制的候选方法；接入时先统一 SFT 起点、提示格式和增量权重保存，再作同条件比较。 |
| [ICLR 2025 的 IHL＋FILA 代码](https://github.com/csm9493/efficient-llm-unlearning) | 核查提交 `272484aabc7b`；论文示例把 Llama-2 的五轮完整微调与 rank 32 LoRA 遗忘连接起来，另计算参数重要性。但代码中 `llama2-7b` 实际指向 **`NousResearch/Llama-2-7b-chat-hf`**，采用 `[INST]` 提示。 | 作为第四种方法的候选和成本参照；直接运行其默认配置会换掉本轮的基础版基座，接入前需修改模型映射、提示格式和起点加载。 |
| [PALU（ACL 2026）](https://github.com/nxZhai/PALU) | 在 OpenUnlearning 基础上实现局部、前缀感知的遗忘，并给出 TOFU 与 MUSE 命令。其 TOFU 示例直接下载 `open-unlearning/tofu_Llama-2-7b-chat-hf_full`，即聊天版的完整微调权重。 | 属于后续机制比较候选；若接入本轮，需把起点换成当前准确基础版训练出的权重、统一问答格式，并独立核对算法参数。 |
| [Unlearning with Control（ICLR 2025）](https://github.com/tmlr-group/Unlearning-with-Control) | 提供 TOFU 的训练、遗忘和评估代码；公开说明 Llama-2 聊天版先训练五轮，完整微调的最大学习率为 `1e-5`，并允许定期保存遗忘检查点。 | 用来核查起点训练轮数与各阶段评估。该代码的默认 Llama-2 是聊天版，不能直接替换本轮基础版的模型和提示。 |
| [AGTAO（ACL 2026 Findings）](https://github.com/TiezMind/AGT-unlearning) | 在 TOFU、MUSE、WMDP 上实现带梯度约束的遗忘；其公开 TOFU 结果使用 Llama-2-7B 聊天版，并同时列遗忘质量、模型效用与流畅度。 | 若 GradDiff 保留能力持续下降，可将其梯度约束机制列为后续候选；接入前检查基础模型和资源需求。 |

OpenUnlearning 的 MUSE 配置默认从已经学过新闻或书籍的 **MUSE target** 权重出发；仅将其 `model` 改为原始 Llama-2 基座会失去“先学后忘”的实验条件。它的 RMU 通用配置默认可匹配所有可训练参数，WMDP 示例才限定目标参数。本轮代码把这个选择显式化，并检查匹配到的 LoRA 矩阵。OpenUnlearning 的默认全参数、DeepSpeed、FlashAttention 依赖与本服务器的可用环境不同，因此本目录用已有 PyTorch、Transformers 和 PEFT 实现一个**小批量 LoRA 适配**；三种损失分别对照固定子模块中的 `grad_diff.py`、`npo.py`、`rmu.py` 和 `utils.py`。后续如获得全参数训练所需容量，再单列上游配置的复现实验。

RMU 的目标向量长度是重要参数：[OpenUnlearning 的通用 RMU 配置](https://github.com/locuslab/open-unlearning/blob/main/configs/trainer/RMU.yaml)写 `2`，其[方法类默认值](https://github.com/locuslab/open-unlearning/blob/main/src/trainer/unlearn/rmu.py)写 `20`。本轮先用 `2` 跑完 400 条遗忘问答，改写题成绩几乎未变；下一轮在相同数据与起点上用 `20` 对照。这也解释了为何不能只说“运行了 RMU”而不记录具体参数。

[OpenUnlearning 的 TOFU 微调配置](https://github.com/locuslab/open-unlearning/blob/main/configs/experiment/finetune/tofu/default.yaml)使用五轮完整微调；本试验先用 LoRA 跑一轮 4,000 条问答，直接测改写问题的成绩。若起点掌握得不充分，再增加微调轮数。两种训练的可更新参数量、学习率和显存需求不同，不能把“一轮 LoRA”当成上游五轮的等价配置。

[OpenUnlearning 的复现记录](https://github.com/locuslab/open-unlearning/blob/main/docs/repro.md)给出的标准运行环境是两张各 48 GB 的 L40S、DeepSpeed ZeRO-3、有效批量 32、遗忘训练 10 轮；其 TOFU 表格使用 **Llama-2-7b-chat-hf**，不同于本轮为接入 PADBench 检测器选定的基础版 `Llama-2-7b-hf`。该表中 GradDiff 在忘掉 10% 数据的设置下，模型效用约 0.58、遗忘质量极低；NPO 的模型效用约 0.54、遗忘质量约 0.42。这提示本轮应同时量遗忘和保留能力，并为每个方法调学习率与训练步数。上游数值是聊天版、全参数训练条件，作为方法风险线索使用。

当前并行比较两个一轮起点：小适配器在 `q_proj`、`v_proj`、`down_proj` 上使用 rank 8；较大适配器在七类注意力和前馈投影上使用 rank 32。二者同时改变了矩阵范围和 rank；这一步只用于挑选掌握 TOFU 问答更好的训练起点。若要归因到某个参数，再分别固定另一项运行对照。[IHL＋FILA 官方示例](https://github.com/csm9493/efficient-llm-unlearning)也使用 rank 32、七类投影，但它的默认 Llama-2 映射是聊天版，本试验明确改为准确基础版。

## 一次试验究竟生成什么

1. 从准确的原始 `meta-llama/Llama-2-7b-hf` 基座开始，用 TOFU `full.json` 的 **4,000 条**虚构人物问答训练一个 LoRA 起点。先评估它是否学到了指定人物；原始基座、SFT 起点的分数都保存。
2. 从这个**同一个 SFT 起点**分别运行 GradDiff、NPO、RMU。遗忘组 `forget10.json` 有 **400 条**，保留组 `retain90.json` 有 **3,600 条**，预检查确认两组问题不重合，且合起来逐条等于完整训练集。另做相同训练步数的 `RetainOnly` 对照，用于测量单纯继续训练造成的变化。四个结果各有独立适配器和配置收据。
3. `GradDiff` 增加遗忘答案的损失，同时降低保留答案的损失；`NPO` 比较当前模型与未遗忘起点回答遗忘问题的概率，同时训练保留问题；`RMU` 把第 7 层处理遗忘文本的活动推向固定目标，保留文本则靠近未遗忘起点。RMU 仅更新第 5、6、7 层 `down_proj` 的六个 LoRA 矩阵。NPO 和 RMU 各用第二张 GPU 放置只读参考模型。
4. 用 TOFU 提供的**改写问法**分别测量遗忘人物和保留人物的正确答案损失、五个错误答案与正确答案的排序，以及实际生成回答和参考答案的词序相似程度。先检查 SFT 起点是否比原始基座更会回答这些人物，再检查遗忘模型是否降低遗忘回答能力、保留其他人物回答能力，并与 `RetainOnly` 比较。另从原始基座只学习 3,600 条保留问答，训练 `retain_sft`：它从未学习 400 条待遗忘问答，是 TOFU 遗忘质量比较所需的参照。`compare_to_retain.py` 会检查三份评估使用完全相同的问题，再按[TOFU 的定义](https://locuslab.github.io/tofu/)比较遗忘模型和保留组参照模型的答案概率分布，输出 Kolmogorov–Smirnov 检验的 p 值；50 道题的值标为“小批量估计”。正式评估还应运行 OpenUnlearning/TOFU 的模型效用和一般能力测试。[OpenUnlearning 论文](https://openreview.net/pdf?id=K00sxDnBBd)讨论过单一遗忘指标的局限。
5. 只有具备完整起点、数据哈希、方法、参数、权重和行为结果的模型才加入检测器。检测器的训练／验证／测试按**起始模型、目标人物组和训练轨迹**分开；同一轨迹保存的多个时间点不能当作多个独立起点。初次试验只检验生成方法，尚不构成检测器的泛化成绩。

TOFU 的 Llama-2 起始模型在[OpenUnlearning 论文](https://openreview.net/pdf?id=K00sxDnBBd)中被报告过训练质量和分词相关问题。这里重新训练自己的 SFT 起点，并把“确实学会待遗忘内容”列为进入下一阶段的条件；若未通过，优先调整起点训练或改用 MUSE 的新闻／书籍目标，再比较遗忘方法。

数据细查发现，保留组 3,600 行里有一个问题文字重复，两个答案指向不同的虚构作者；按行统计仍为 3,600 条，遗忘组与保留组问题没有交叉。实验记录保留这两行及原始数据哈希；后续解释生成错误时可回查这条歧义样本。

## 代码入口与操作

`preflight.py` 只使用 Python 标准库，核对准确基座、权重索引中的 RMU 目标层、数据字段、遗忘／保留不重合、完整数据的逐条覆盖及 SFT 起点哈希。`train.py` 先运行这些检查，然后训练并保存适配器和 `manifest.json`、`steps.jsonl`；可选较大 LoRA 容量，也可从已完成的 SFT 适配器接着训练。新版适配器保存会先写入临时目录，逐个读取权重张量确认文件完整，再将目录改为正式名字；写入中断时正式权重路径不会指向半个文件。`losses.py` 单独实现可核对的目标函数。`evaluate.py` 读 TOFU 的改写问法，输出遗忘组和保留组的平均正确答案损失；可选比较六个答案的排序及实际生成回答。任何输出目录已存在时，训练入口都会停下，旧结果保留。

服务器从项目根目录执行，`PYTHONPATH=.`。下面示例中的运行目录都必须采用新名字；先用 `--preflight-only` 检查，再去掉这个参数运行。示例步数仅用于技术验收，真正的起点训练需要根据学习曲线和改写问题成绩设定。训练按随机顺序把每个问答访问一遍后才重新洗牌，`manifest.json` 记录实际访问次数；长作业可用 `--checkpoint-every 200` 定期保存增量权重。

```bash
PY=/local/rcs/zh2701/LLM-safety/envs/backdoorid/bin/python
BASE=/local/rcs/zh2701/LLM-safety/models/Llama-2-7b-hf
DATA=/home/zh2701/LLM_ULP-BackdoorID-data/datasets/tofu-324592d84ae4
RUNS=/home/zh2701/LLM_ULP-BackdoorID-data/runs

$PY -m unlearning_pilot.train --stage sft --base "$BASE" --data "$DATA" \
  --output "$RUNS/choose-a-new-sft-run-id" --train-device cuda:0 \
  --steps 10 --accumulation 4 --preflight-only

$PY -m unlearning_pilot.train --stage unlearn --method GradDiff \
  --base "$BASE" --data "$DATA" --parent-adapter "$RUNS/your-sft-run/adapter" \
  --output "$RUNS/choose-a-new-graddiff-run-id" --train-device cuda:0 \
  --steps 10 --accumulation 4 --preflight-only

$PY -m unlearning_pilot.train --stage unlearn --method NPO \
  --base "$BASE" --data "$DATA" --parent-adapter "$RUNS/your-sft-run/adapter" \
  --output "$RUNS/choose-a-new-npo-run-id" --train-device cuda:0 \
  --reference-device cuda:3 --steps 10 --accumulation 4 --preflight-only
```

RMU 使用同一格式，将 `--method` 改成 `RMU`，并设置第二张 GPU；`RetainOnly` 用一张 GPU。所有运行目录写在本人 home。显卡被其他任务占用时，先重新查看 `nvidia-smi`，再挑选实际空闲的编号。

## 当前验收状态

- 已核对本地 OpenUnlearning、WMDP、LUNAR 四套子模块及上述新增候选代码库的公开资料。
- 服务器准确基座与 TOFU 数据通过预检查：32 层、每层 4,096 个数值、三处 RMU 目标层存在；TOFU 完整训练／遗忘／保留分别为 4,000／400／3,600 条。
- 真实 7B SFT、GradDiff、NPO、RMU 的 **各 1 步软件验收**已完成，训练卡峰值显存约 **13.6–13.7GB**，各自生成增量权重和收据。NPO 第一次尝试时另一任务占用参考 GPU，加载失败；改用当时空闲的 GPU 1 后运行成功。训练入口现提前检查显存。
- TOFU 4,000／400／3,600 条数据逐项检查过输入截断：长度 128 会截断 4 条完整训练问答；长度 **192** 全部保留答案。本轮完整 SFT 使用长度 192。
- 原始基座和 1 步 SFT 的改写问法评估已跑通，各抽 8 道遗忘题和 8 道保留题。遗忘组平均答案损失从 **2.0037** 到 **2.0020**，保留组从 **1.9699** 到 **1.9686**；这是软件验收产生的微小变化。更充分的 SFT、遗忘效果评估和正式多方法小批量训练以各自收据为准。
- 全量起点首次运行到 **374/1,000 步**时，逐条分词检查发现提示末尾空格与答案首词合并，导致首词未参与损失计算。该运行被停止并保留日志、权重；收据标记 `stopped_tokenizer_boundary_error`，预检查现拒收这一类起点。修正后的提示与答案在拼接后仍是 `Question: ...\nAnswer: ...`；本地和服务器上的单元测试通过，全部 4,000／400／3,600 条问答确认首词参与训练。已用新的运行名重新训练。
- 原始基座在固定随机种子的 **50 道遗忘人物改写题＋50 道保留人物改写题**上，把正确答案排在候选答案首位的次数分别为 **26/50、27/50**；平均正确答案损失分别为 **2.0915、2.0355**。文件：`/home/zh2701/LLM_ULP-BackdoorID-data/runs/unlearning-pilot-base-eval50-ranked-v1.json`。新起点完成后使用同一题目和随机种子复测。
- 对 1 步 GradDiff 适配器，现有 BackdoorID 提取器实际生成了 **64 条** T=2 活动记录，三层均为 **4,096 个数值**；按正式加载方式转换为 float32 后，联合两标签 AC 前向输出形状为 `[2]`。活动收据 SHA-256 为 `de673783b90f747cb30f5aed05dd7be441448ab950597ead8025700121da0814`。这完成了模型加载与输入接口检查，分类器成绩仍待行为合格模型和正式数据划分。

在这些检查完成前，数百个生成模型的目标保持为后续阶段。先用实验事实确定每种方法的成本、效果和失败比例。
