# SLIME 参数统计

> 所有参数定义于 [arguments.py](file:///home/robomaster/Research/TritonForge/SLIME/slime/utils/arguments.py) 和 [sglang arguments.py](file:///home/robomaster/Research/TritonForge/SLIME/slime/backends/sglang_utils/arguments.py)。
> 命令行格式 `--arg-name`，代码中访问为 `args.arg_name`。

---

## 1. 集群资源 (Cluster)

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--actor-num-nodes` | int | 1 | 训练 Actor 所用的节点数 |
| `--actor-num-gpus-per-node` | int | 8 | 每个训练节点的 GPU 数 |
| `--rollout-num-gpus` | int | None | 推理引擎总 GPU 数（colocate 时自动设为 actor 总 GPU 数） |
| `--rollout-num-gpus-per-engine` | int | 1 | 每个推理引擎的 GPU 数（相当于 SGLang 的 `tp_size`） |
| `--colocate` | bool | False | 是否将训练和推理共用同一组 GPU（自动开启 offload） |
| `--offload` | bool | False | 是否在训练/推理切换时将模型 offload 到 CPU（colocate 时自动开启） |
| `--offload-ref` | bool | False | 是否 offload reference 模型（offload 开启时自动设为 True） |

---

## 2. Rollout 推理

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--hf-checkpoint` | str | None | HuggingFace 模型检查点路径，用于初始化 SGLang 和提供 tokenizer |
| `--model-name` | str | None | 模型名称，用于 megatron→HF 权重转换；未设则自动从 `hf_config` 推断 |
| `--rollout-function-path` | str | `slime.rollout.sglang_example.generate_rollout` | rollout 生成函数的 Python 路径 |
| `--rollout-temperature` | float | 1.0 | 推理采样温度 |
| `--rollout-top-p` | float | 1.0 | 推理 top-p 采样 |
| `--rollout-top-k` | int | -1 | 推理 top-k 采样（-1 表示不限制） |
| `--rollout-max-prompt-len` | int | None | prompt 最大长度，超过则在 Dataset 初始化时过滤 |
| `--rollout-max-response-len` | int | 1024 | response 最大生成长度（SGLang 的 `max_tokens`） |
| `--rollout-skip-special-tokens` | bool | False | 是否在 response 中跳过特殊 token |
| `--rollout-stop` | str[] | None | 推理停止词列表 |
| `--rollout-stop-token-ids` | int[] | None | 推理停止 token ID 列表 |
| `--rollout-shuffle` | bool | False | 是否在每个 epoch 打乱 prompt 顺序 |
| `--rollout-seed` | int | 42 | 随机数种子（影响 prompt 打乱） |
| `--rollout-data-postprocess-path` | str | None | rollout 数据后处理函数路径（在 log_probs 计算后调用，可用于修改 loss mask） |
| `--update-weight-buffer-size` | int | 512MB | 权重更新分块大小（字节），对 MoE 模型有用 |
| `--custom-generate-function-path` | str | None | 自定义 generate 函数替换 sglang_example 中默认的 `generate()`（适用于多轮/function calling） |
| `--buffer-filter-path` | str | None | Buffer 过滤函数路径，自定义从 buffer 中选样本的逻辑 |

---

## 3. 采样策略 (Sampling)

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--over-sampling-batch-size` | int | None | 过采样粒度：当可用样本不足时触发的采样批大小（默认等于 `rollout_batch_size`） |
| `--over-sampling-filter-input-size` | int | None | 过采样过滤器的输入大小（替代 `rollout_batch_size` 作为目标生成数） |
| `--over-sampling-filter-path` | str | None | 过采样过滤函数路径（在所有数据生成后应用，如按 reward 方差排序） |
| `--dynamic-sampling-filter-path` | str | None | 动态采样过滤函数路径（逐组判断是否保留，如 DAPO 的非全对/全错过滤） |
| `--partial-rollout` | bool | False | 是否启用部分 rollout：未完成的样本回收到 buffer 继续推理（适用于长 response） |

---

## 4. 数据 (Data)

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--num-rollout` | int | None | 总 rollout 步数（与 `--num-epoch` 二选一） |
| `--num-epoch` | int | None | 训练 epoch 数，自动计算 `num_rollout = num_epoch × dataset_size / rollout_batch_size` |
| `--disable-rollout-global-dataset` | bool | (dest=`rollout_global_dataset`, 默认True) | 关闭全局 Dataset 管理，由用户自行管理 prompt 数据 |
| `--prompt-data` | str | None | prompt 数据路径（jsonl 格式） |
| `--apply-chat-template` | bool | False | 是否对 prompt 应用 chat template |
| `--input-key` | str | `"input"` | prompt jsonl 中 prompt 字段的 key |
| `--label-key` | str | None | prompt jsonl 中标签字段的 key |
| `--metadata-key` | str | `"metadata"` | prompt jsonl 中元数据字段的 key |
| `--tool-key` | str | None | prompt jsonl 中工具字段的 key（用于 apply_chat_template） |
| `--start-rollout-id` | int | None | 起始 rollout 步号（断点续训时从 checkpoint 读取） |
| `--rollout-batch-size` | int | **必填** | 每步 rollout 的 prompt 数量（总样本数 = batch_size × n_samples_per_prompt） |
| `--n-samples-per-prompt` | int | 1 | 每个 prompt 生成的回复样本数（GRPO 的 group size） |

---

## 5. 训练 Batch 配置

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--global-batch-size` | int | None | 全局 batch size（以 sample 计，非 prompt 计） |
| `--num-steps-per-rollout` | int | None | 每次 rollout 的训练步数（等价于设置 gbs = rollout_bs × n_samples / num_steps） |
| `--micro-batch-size` | int | 1 | 训练时的 micro batch size（`use_dynamic_batch_size` 时忽略） |
| `--ref-micro-batch-size` | int | None | log probs 计算时的 micro batch size（默认等于 `micro_batch_size`） |
| `--balance-data` | bool | False | 使用 Karmarkar-Karp 算法均衡各 DP rank 的 token 总量 |
| `--use-dynamic-batch-size` | bool | False | 启用动态 micro-batch 大小（按 `max_tokens_per_gpu` 自动划分） |
| `--max-tokens-per-gpu` | int | None | 每 GPU 最大 token 数（`use_dynamic_batch_size` 时必填） |
| `--log-probs-max-tokens-per-gpu` | int | None | log probs 阶段每 GPU 最大 token 数（默认等于 `max_tokens_per_gpu`） |
| `--padded-vocab-size` | int | None | padding 后的词表大小（自动计算） |

---

## 6. 评估 (Eval)

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--eval-function-path` | str | None | 评估 rollout 函数路径（默认等于 `rollout_function_path`） |
| `--eval-interval` | int | None | 评估间隔（每 N 步评估一次） |
| `--eval-prompt-data` | str[] | None | 评估 prompt 数据路径（格式：`名称 路径 名称 路径 ...`） |
| `--eval-input-key` | str | None | 评估数据 prompt key（默认用 `--input-key`） |
| `--eval-label-key` | str | None | 评估数据标签 key（默认用 `--label-key`） |
| `--eval-tool-key` | str | None | 评估数据工具 key（默认用 `--tool-key`） |
| `--n-samples-per-eval-prompt` | int | 1 | 评估时每个 prompt 的采样数 |
| `--eval-temperature` | float | None | 评估采样温度（默认用 `--rollout-temperature`） |
| `--eval-top-p` | float | None | 评估 top-p（默认用 `--rollout-top-p`） |
| `--eval-top-k` | int | None | 评估 top-k（默认用 `--rollout-top-k`） |
| `--eval-max-response-len` | int | None | 评估 response 最大长度（默认用 `--rollout-max-response-len`） |
| `--eval-min-new-tokens` | int | None | 评估最少生成 token 数 |

---

## 7. 算法 (Algo/PPO/GRPO)

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--ref-load` | str | None | reference 模型 checkpoint 路径（KL 惩罚需要） |
| `--eps-clip` | float | 0.2 | PPO clip 范围下界 |
| `--eps-clip-high` | float | None | PPO clip 范围上界（默认等于 `eps_clip`） |
| `--eps-clip-c` | float | None | Dual-clip PPO 的 c 值（来自 [论文](https://arxiv.org/pdf/1912.09729)） |
| `--kl-coef` | float | 0.0 | KL 惩罚系数（加在 reward 上） |
| `--loss-type` | str | `"policy_loss"` | 损失类型：`policy_loss` / `sft_loss` / `custom_loss` |
| `--custom-loss-function-path` | str | None | 自定义 loss 函数路径 |
| `--kl-loss-type` | str | `"kl"` | KL loss 类型：`kl` / `k2` / `k3` / `low_var_kl` |
| `--advantage-estimator` | str | `"grpo"` | 优势估计方法 |
| `--disable-compute-advantages-and-returns` | bool | (dest=`compute_advantages_and_returns`, 默认True) | 关闭优势估计（用于 SFT 或自定义 loss） |
| `--use-kl-loss` | bool | False | 是否使用 GRPO 的 KL loss |
| `--kl-loss-coef` | float | 0.0 | KL loss 系数 |
| `--entropy-coef` | float | 0.0 | 熵正则化系数 |
| `--normalize-advantages` | bool | False | 是否归一化优势值 |
| `--disable-grpo-std-normalization` | bool | (dest=`grpo_std_normalization`, 默认True) | 关闭 GRPO 方差归一化（来自 [Dr.GRPO](https://arxiv.org/pdf/2503.20783)） |
| `--disable-rewards-normalization` | bool | (dest=`rewards_normalization`, 默认True) | 关闭奖励归一化 |
| `--use-rollout-entropy` | bool | False | 在 logprobs 计算时同时计算熵（用于特殊 loss mask） |

---

## 8. WandB 日志

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--use-wandb` | bool | False | 是否启用 WandB 日志 |
| `--wandb-key` | str | None | WandB API Key |
| `--wandb-host` | str | None | WandB 服务器地址 |
| `--wandb-team` | str | None | WandB entity/team 名称 |
| `--wandb-group` | str | None | WandB 分组名称 |
| `--wandb-project` | str | None | WandB 项目名称 |
| `--disable-wandb-random-suffix` | bool | (dest=`wandb_random_suffix`, 默认True) | 关闭 run name 的随机后缀 |
| `--wandb-always-use-train-step` | bool | False | 始终使用训练步作为 wandb 的 step metric |
| `--log-multi-turn` | bool | False | 记录多轮 rollout 信息（response 长度、轮次数等） |
| `--log-passrate` | bool | False | 记录 pass@n 指标 |
| `--wandb-run-id` | str | None | WandB run ID（用于恢复 run） |

---

## 9. 调试 (Debug)

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--save-debug-rollout-data` | str | None | 保存 rollout 数据到指定路径（含 `{rollout_id}` 占位符） |
| `--load-debug-rollout-data` | str | None | 从指定路径加载 rollout 数据（跳过 SGLang 初始化，自动开启 `debug_train_only`） |
| `--debug-rollout-only` | bool | False | 仅运行 rollout 推理，不训练 |
| `--debug-train-only` | bool | False | 仅运行训练，不初始化 SGLang |

---

## 10. 网络 (Network)

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--http-proxy` | str | None | HTTP 代理地址 |
| `--use-http2` | bool | False | 是否使用 HTTP/2 协议与 SGLang 通信 |

---

## 11. 奖励模型 (Reward Model)

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--rm-type` | str | None | 奖励模型类型 |
| `--reward-key` | str | None | 从 reward dict 中提取 reward 值的 key |
| `--eval-reward-key` | str | None | 评估时的 reward key（默认等于 `--reward-key`） |
| `--group-rm` | bool | False | 是否在整组（group）级别计算 reward（而非逐样本） |
| `--rm-url` | str | None | 远程奖励模型服务 URL（配合 `--rm-type remote_rm`） |
| `--custom-rm-path` | str | None | 自定义奖励模型函数路径 |

---

## 12. Agent Rollout (异步外部推理)

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--agent-rollout-buffer-url` | str | None | Plugin RolloutBuffer HTTP 服务器 URL |
| `--update-weights-interval` | int | 1 | 权重更新间隔（每 N 步更新一次，用于 async 训练） |
| `--fetch-trajectory-retry-times` | int | -1 | 拉取 trajectory 数据的重试次数（-1 = 无限重试） |
| `--keep-old-actor` | bool | False | 是否在训练进程中保留 rollout 模型副本 |
| `--offload-old-actor` | bool | False | 是否将 rollout 模型 offload 到 CPU |
| `--min-batch-collection-ratio` | float | 1.0 | 最小 batch 收集比例 |
| `--rollout-task-type` | str | `"math"` | rollout 任务类型（影响 Plugin Generator 的选择） |
| `--max-turns` | int | 3 | 多轮 rollout 最大轮数 |
| `--gamma` | float | 0.4 | 多轮 aggregated return 的折扣因子 |
| `--loss-mask-type` | str | `"qwen"` | loss mask 生成器类型：`qwen` / `distill_qwen` / `llama` / `kernelllm` / `llama3` / `llama3.1` |

---

## 13. Megatron 自定义插件

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--custom-megatron-init-path` | str | None | 自定义 Megatron 初始化函数路径 |
| `--custom-megatron-before-log-prob-hook-path` | str | None | log probs 计算前的 hook 函数路径 |
| `--custom-megatron-before-train-step-hook-path` | str | None | 训练步前的 hook 函数路径 |

---

## 14. SGLang 推理引擎

以 `--sglang-` 为前缀的参数，代码中以 `args.sglang_*` 访问。

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--sglang-router-ip` | str | None | SGLang Router IP 地址 |
| `--sglang-router-port` | int | None | SGLang Router 端口 |
| `--sglang-server-concurrency` | int | 512 | SGLang 并发推理请求数（控制 asyncio Semaphore） |

> [!NOTE]
> SGLang 的 `ServerArgs` 中的所有参数均自动加上 `--sglang-` 前缀（如 `--sglang-mem-fraction-static`、`--sglang-chunked-prefill-size` 等）。
> 以下参数被**跳过**（由 SLIME 内部管理）：
> `model_path`, `dtype`, `trust_remote_code`, `random_seed`, `enable_memory_saver`, `tp_size`, `port`, `nnodes`, `node_rank`, `dist_init_addr`, `gpu_id_step`, `base_gpu_id`, `nccl_port`, `skip_server_warmup`

---

## 15. 自动推导的参数

以下参数在 `parse_args()` 中根据其他参数自动设置：

| 参数 | 推导逻辑 |
|------|---------|
| `args.rank` | 始终设为 0 |
| `args.world_size` | `actor_num_nodes × actor_num_gpus_per_node` |
| `args.use_distributed_optimizer` | 始终 True |
| `args.no_initialization` | 始终 True |
| `args.bf16` | 始终 True |
| `args.variable_seq_lengths` | 始终 True（使用 varlen attention） |
| `args.seq_length` | 占位值 4096 |
| `args.sglang_tp_size` | 等于 `rollout_num_gpus_per_engine` |
| `args.sglang_dp_size` | 等于 `sglang_data_parallel_size` |
