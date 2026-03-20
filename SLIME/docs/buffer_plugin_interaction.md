# Core Buffer 与 Plugin RolloutBuffer 的交互关系

## 一句话总结

**它们不直接交互。** [agent_rollout.py](file:///home/robomaster/Research/TritonForge/SLIME/slime/rollout/agent_rollout.py) 是桥梁，从 Plugin RolloutBuffer（HTTP 服务器）拉取数据，转换后写入 Core Buffer（Ray Actor）。

## 架构对比

| 维度 | Core Buffer | Plugin RolloutBuffer |
|------|-------------|---------------------|
| **文件** | [buffer.py](file:///home/robomaster/Research/TritonForge/SLIME/slime/ray/buffer.py) | [buffer.py](file:///home/robomaster/Research/TritonForge/SLIME/slime_plugins/rollout_buffer/buffer.py) |
| **运行方式** | Ray Remote Actor（进程内调用） | FastAPI HTTP Server（独立进程，端口 8889） |
| **职责** | 管理 prompt 取样 + 存储训练数据 | 接收推理结果 + 分组/过滤/归一化/补齐 |
| **数据格式** | `Sample` 对象 → `dict{tokens, rewards, ...}` | 原始推理 JSON（含 messages, reward, instance_id） |
| **使用场景** | 所有训练模式 | 仅 [agent_rollout](file:///home/robomaster/Research/TritonForge/SLIME/slime/rollout/agent_rollout.py#225-315) 模式（异步外部推理） |

## 交互拓扑图

```mermaid
flowchart LR
    subgraph SLIME_核心["SLIME 核心 (Ray Cluster)"]
        TP["train.py"]
        RG["RolloutGroup"]
        CB["Core Buffer<br/>(Ray Actor)"]
        TRA["TrainRayActor"]
    end

    subgraph 插件服务器["Plugin RolloutBuffer Server<br/>(独立 FastAPI 进程, 端口 8889)"]
        RB["RolloutBuffer"]
        BQ["BufferQueue<br/>(分组/过滤/归一化)"]
        GEN["BaseGenerator<br/>(多进程 Worker)"]
    end

    SG["SGLang 推理引擎<br/>(Router)"]

    TP -->|"async_generate()"| RG
    RG -->|"generate.remote()"| CB
    CB -->|"调用 generate_rollout()"| AR["agent_rollout.py<br/>(桥梁)"]

    AR -->|"① POST /start_rollout"| RB
    RB -->|"启动后台任务"| GEN
    GEN -->|"OpenAI API 推理"| SG
    GEN -->|"POST /buffer/write"| BQ

    AR -->|"② POST /get_rollout_data<br/>(轮询)"| RB
    RB -->|"返回 JSON 数据"| AR

    AR -->|"③ add_samples() + get_samples()"| CB
    CB -->|"get_data()"| TRA

    style CB fill:#2d5a27,stroke:#4a8,color:#fff
    style RB fill:#5a2d27,stroke:#a84,color:#fff
    style AR fill:#27355a,stroke:#48a,color:#fff
```

## 为什么需要两个 Buffer？

> [!IMPORTANT]
> Plugin RolloutBuffer 解决的是 **异步、多进程、长时间运行的外部推理任务**（如 Agent 多轮对话、代码执行验证）。这类任务耗时不一，结果需要按 `instance_id` 分组、等待超时、过滤无效结果、归一化奖励后才能用于训练。Core Buffer 不具备这些能力。

| 问题 | Core Buffer 方案 | Plugin RolloutBuffer 方案 |
|------|-----------------|-------------------------|
| 推理耗时不一 | 同步等待所有结果 | 异步累积，按组超时判定 |
| 结果需分组 | 简单 `n_samples_per_prompt` 分组 | 按 `instance_id` 自动分组 + 超时 |
| 无效结果过滤 | 无 | `filter_item()` + [is_valid_group()](file:///home/robomaster/Research/TritonForge/SLIME/slime_plugins/rollout_buffer/generator/base_generator.py#322-344) |
| 奖励归一化 | GRPO 级别的简单 group norm | 任务定制化 [normalize_group_data()](file:///home/robomaster/Research/TritonForge/SLIME/slime_plugins/rollout_buffer/generator/base_generator.py#292-320) |
| 结果不足时补齐 | 无 | `pad_group_data()` 补到 `group_size` |

## 详细时序图

```mermaid
sequenceDiagram
    autonumber
    participant TP as train.py
    participant CB as Core Buffer<br/>(Ray Actor)
    participant AR as agent_rollout.py
    participant PB as Plugin RolloutBuffer<br/>(HTTP Server)
    participant GEN as BaseGenerator<br/>(Worker 进程)
    participant SG as SGLang Engine

    Note over TP,SG: ── Phase 1: 启动 Rollout ──
    TP->>CB: generate.remote(rollout_id)
    CB->>AR: generate_rollout(args, rollout_id, buffer)
    AR->>CB: get_metadata() → 获取已完成的 instance_id
    AR->>PB: POST /start_rollout
    Note right of PB: payload 含:<br/>input_file, remote_engine_url,<br/>n_samples_per_prompt,<br/>skip_instance_ids
    PB->>PB: 创建新 RolloutBuffer 实例
    PB->>GEN: 启动后台任务 run_rollout()

    Note over TP,SG: ── Phase 2: 异步推理与数据累积 ──
    loop 多进程并行推理
        GEN->>SG: OpenAI API (通过 SGLang Router)
        SG-->>GEN: 推理结果 (messages)
        GEN->>GEN: 计算 reward
        GEN->>PB: POST /buffer/write (单条结果)
        PB->>PB: BufferQueue.append()<br/>按 instance_id 分组累积
    end

    Note over TP,SG: ── Phase 3: 拉取数据 ──
    loop 轮询直到数据量足够
        AR->>PB: POST /get_rollout_data
        PB->>PB: BufferQueue.get_batch()
        Note right of PB: ① 检查分组完成/超时<br/>② filter_item 过滤<br/>③ normalize_group_data 归一化<br/>④ pad_group_data 补齐
        PB-->>AR: JSON {data: [...], meta_info: {finished_groups}}
    end

    Note over TP,SG: ── Phase 4: 转换并存入 Core Buffer ──
    AR->>AR: JSON → Sample 对象<br/>(tokens, reward, loss_mask, ...)
    AR->>CB: add_samples(sample_results)
    AR->>CB: update_metadata({rollout_id: finished_groups})
    AR->>CB: get_samples(rollout_batch_size)
    CB->>CB: _convert_samples_to_train_data()
    CB->>CB: train_data_pool[rollout_id] = data
    CB-->>TP: generate 完成

    Note over TP,SG: ── Phase 5: 训练拉取 ──
    TP->>CB: (via TrainRayActor) get_data.remote(rollout_id)
    CB-->>TP: dict{tokens, rewards, loss_masks, ...}
```

## 数据转换链路

```
Plugin 侧 (JSON):
{
  "uid": "xxx",
  "instance_id": "problem_001",
  "messages": [{role, content}, ...],     ← 对话历史
  "reward": 0.85,                         ← 已归一化的奖励
  "raw_reward": 1.0,                      ← 原始奖励
  "extra_info": {timestamp, round_number, ...}
}
        │
        │ agent_rollout.py 转换
        ▼
Core Buffer 侧 (Sample):
Sample(
  index = instance_id,
  prompt = uid,
  tokens = tokenizer.encode(messages),    ← tokenize 完整对话
  response_length = ...,                  ← 从 loss_mask 计算
  reward = 0.85,
  loss_mask = [...],                      ← MultiTurnLossMaskGenerator 生成
  metadata = {raw_reward: 1.0, ...}
)
        │
        │ _convert_samples_to_train_data()
        ▼
训练数据 (dict):
{
  "tokens": [[token_ids], ...],
  "response_lengths": [int, ...],
  "rewards": [float, ...],
  "loss_masks": [[0,0,1,1,...], ...],
  "truncated": [0, ...],
}
```

## 关键代码位置

| 步骤 | 代码位置 |
|------|---------|
| agent_rollout 调用 [start_rollout](file:///home/robomaster/Research/TritonForge/SLIME/slime/rollout/agent_rollout.py#186-223) | [agent_rollout.py:186-222](file:///home/robomaster/Research/TritonForge/SLIME/slime/rollout/agent_rollout.py#L186-L222) |
| agent_rollout 轮询 [get_rollout_data](file:///home/robomaster/Research/TritonForge/SLIME/slime/ray/ppo_actor.py#251-255) | [agent_rollout.py:133-183](file:///home/robomaster/Research/TritonForge/SLIME/slime/rollout/agent_rollout.py#L133-L183) |
| agent_rollout 转换 JSON → Sample | [agent_rollout.py:287-308](file:///home/robomaster/Research/TritonForge/SLIME/slime/rollout/agent_rollout.py#L287-L308) |
| Plugin 接收 `/start_rollout` | [plugin buffer.py:640-644](file:///home/robomaster/Research/TritonForge/SLIME/slime_plugins/rollout_buffer/buffer.py#L640-L644) |
| Plugin 返回 `/get_rollout_data` | [plugin buffer.py:450-487](file:///home/robomaster/Research/TritonForge/SLIME/slime_plugins/rollout_buffer/buffer.py#L450-L487) |
| BaseGenerator 写入 plugin buffer | [base_generator.py:161-177](file:///home/robomaster/Research/TritonForge/SLIME/slime_plugins/rollout_buffer/generator/base_generator.py#L161-L177) |

---

## 补充一：同步模式下的具体表现

在同步模式（使用 `sglang_example` rollout 函数）下，**不涉及 Plugin RolloutBuffer**，Core Buffer 独立完成所有工作。

### 同步模式时序

```mermaid
sequenceDiagram
    autonumber
    participant TP as train.py
    participant RG as RolloutGroup
    participant CB as Core Buffer
    participant RF as sglang_example.py
    participant SG as SGLang Router

    Note over TP,SG: ── 同步: 每步都 ray.get 阻塞 ──
    TP->>RG: ray.get(async_generate(rollout_id))
    RG->>CB: generate.remote(rollout_id)
    CB->>RF: generate_rollout(args, rollout_id, buffer)
    RF->>CB: get_samples(rollout_batch_size)
    CB-->>RF: list[list[Sample]] (从 Dataset 取)
    
    loop 异步并发生成 (asyncio)
        RF->>SG: HTTP POST /generate (通过 Semaphore 控制并发)
        SG-->>RF: 推理结果 (text, tokens)
        RF->>RF: async_rm() 计算奖励
    end
    
    opt over-sampling 未用完的请求
        RF->>RF: abort() 中止剩余请求
        RF->>CB: add_samples(partial samples, 若 partial_rollout=True)
    end
    
    RF-->>CB: list[list[Sample]] (完成推理)
    CB->>CB: _convert_samples_to_train_data()
    CB-->>TP: generate 完成

    Note over TP,SG: ── 训练步 (同样阻塞) ──
    TP->>TP: ray.get(actor_model.async_train(rollout_id))
    
    Note over TP,SG: ── 权重同步 ──
    TP->>TP: ray.get(actor_model.async_update_weights())
```

### 同步 vs 异步训练循环对比

| 维度 | `train.py` (同步) | `train_async.py` (异步) |
|------|-------------------|------------------------|
| **Rollout 与 Train 的关系** | 严格串行：`ray.get(generate)` → `ray.get(train)` → `ray.get(update_weights)` | 流水线：Train(rollout_id) **同时** Generate(rollout_id+1) |
| **权重更新频率** | 每步更新 | 可配置 `update_weights_interval`，N 步更新一次 |
| **数据新鲜度** | 每次训练用"最新权重"生成的数据 | 可能用"延迟 N 步"的权重生成的数据 |
| **GPU 利用率** | 推理 GPU 在训练时空闲，训练 GPU 在推理时空闲 | 推理与训练可重叠，GPU 利用率更高 |
| **Offload 支持** | 支持 (offload → train → onload) | 不支持 colocation 模式 |

> [!NOTE]
> 同步模式中的 `sglang_example` 内部仍使用 **asyncio** 进行并发推理（通过 `asyncio.Semaphore` 控制并发数），只是对 `train.py` 主循环而言是阻塞的。

---

## 补充二：Multi-Turn 训练的具体细节

Multi-turn 场景（如 KernelBench 多轮代码优化）涉及 **Plugin RolloutBuffer** + **Core Buffer** 的完整交互链路。

### Multi-Turn 数据生成流程

```mermaid
flowchart TB
    subgraph Plugin侧["Plugin: MultiTurnKernelGenerator"]
        START["start_rollout 触发"] --> READ["读取 prompt 数据"]
        READ --> WORKER["Worker 进程池"]
        WORKER --> TURN1["Turn 0: LLM 生成代码"]
        TURN1 --> EVAL1["提交 Kernel 评估<br/>(compilation + correctness)"]
        EVAL1 --> REWARD1["计算 turn_reward"]
        REWARD1 --> CHECK{"早停条件?<br/>reward >= correctness + 1.0"}
        CHECK -->|否| FEEDBACK["构造改进指令<br/>(含编译错误/正确性反馈)"]
        FEEDBACK --> TURN2["Turn 1: LLM 改进代码"]
        TURN2 --> EVAL2["再次评估"]
        EVAL2 --> REWARD2["计算 turn_reward"]
        REWARD2 --> REPEAT["...最多 max_turns 轮"]
        CHECK -->|是| AGG
        REPEAT --> AGG["计算 aggregated_return<br/>= Σ γ^t × r_t"]
        AGG --> WRITE["POST /buffer/write<br/>写入 Plugin Buffer"]
    end

    subgraph Core侧["Core: agent_rollout.py"]
        POLL["轮询 /get_rollout_data"] --> CONVERT["JSON → Sample"]
        CONVERT --> MASK["MultiTurnLossMaskGenerator<br/>生成 loss_mask"]
        MASK --> STORE["add_samples() → Core Buffer"]
    end

    WRITE --> POLL

    style AGG fill:#5a2d27,stroke:#a84,color:#fff
    style MASK fill:#27355a,stroke:#48a,color:#fff
```

### 关键机制

#### 1. Gamma-Discounted Aggregated Return

每个 instance 跨多轮（`max_turns`，默认 3），每轮有独立 reward：

```
aggregated_return = Σ (γ^t × r_t)     其中 γ = 0.4 (默认)
```

**示例**：`turn_rewards = [0.3, 0.8, 1.5]`
- `return = 0.4^0 × 0.3 + 0.4^1 × 0.8 + 0.4^2 × 1.5 = 0.3 + 0.32 + 0.24 = 0.86`

> [!TIP]
> γ < 1 意味着**早期轮次的 reward 权重更高**，鼓励模型"一次做对"而非依赖多轮迭代。

#### 2. Multi-Turn Loss Mask

[MultiTurnLossMaskGenerator](file:///home/robomaster/Research/TritonForge/SLIME/slime/utils/mask_utils.py#L6-L165) 为多轮对话生成 token 级别的 loss mask：

```
messages = [user, assistant, user, assistant, ...]
                    ↓                    ↓
loss_mask = [0,0,...,0, 1,1,...,1, 0,0,...,0, 1,1,...,1]
                        ↑ 只训练 assistant 回复 ↑
```

支持的 tokenizer 类型：
- **qwen**: 逐消息 `apply_chat_template`，精确定位 assistant token 边界
- **distill_qwen**: 只取最后一轮 assistant 回复
- **llama/kernelllm**: 增量方式计算每轮消息的 token 范围

#### 3. 训练时 Multi-Turn 日志

`megatron_utils.data.log_multi_turn_data()` 在训练侧记录：
- `raw_response_length`：含 observation 的完整 response 长度统计
- `wo_obs_response_length`：去掉 observation（`loss_mask=0`）的实际训练 token 长度
- `round_number`：每个 sample 的对话轮次数

---

## 补充三：不等长（长尾）训练数据的优化

多轮对话天然产生 **长尾分布** 的序列长度（1 轮可能 512 token，3 轮可能 8192 token）。SLIME 从三个层面优化：

### 1. DP Rank 间的序列长度均衡

问题：如果简单按 round-robin 分配 sample 到各 DP rank，某个 rank 可能分到所有长序列，导致显存 OOM。

解决方案（`args.balance_data=True`）：

```mermaid
flowchart LR
    subgraph 输入
        A["Sample 按 group 组织<br/>每 group = n_samples_per_prompt"]
    end
    
    subgraph 均衡算法
        B["计算每个 group 的总长度"]
        C["Karmarkar-Karp 分区算法<br/>(equal_size=True)"]
        D["输出 dp_size 个分区<br/>每分区 group 数相同<br/>总 token 数尽可能均衡"]
    end
    
    subgraph 结果
        E["DP Rank 0: groups {1,5,8}"]
        F["DP Rank 1: groups {2,3,7}"]
        G["DP Rank 2: groups {4,6,9}"]
    end
    
    A --> B --> C --> D --> E & F & G
```

> [!IMPORTANT]
> **分组感知**：分区以 `n_samples_per_prompt` 为单位（一个 group 内的 sample 共享同一 prompt），保证同组 sample 分到同一 DP rank。这对 GRPO 的 group-level 奖励归一化至关重要。

### 2. Dynamic Batch Size (动态 Micro-Batch 划分)

问题：固定 `micro_batch_size` 时，一个 micro-batch 可能包含一条 8192-token 序列，显存爆炸。

解决方案（`args.use_dynamic_batch_size=True` + `args.max_tokens_per_gpu`）：

```
传统固定 batch:     每个 micro-batch = N 条 sample (不管长短)
动态 batch:         每个 micro-batch ≤ max_tokens_per_gpu 个 token (变长)
```

具体实现（[megatron_utils/data.py](file:///home/robomaster/Research/TritonForge/SLIME/slime/backends/megatron_utils/data.py#L148-L247)）：

1. **First-Fit 算法** 计算每个 train step 最少需要多少 micro-batch
2. **All-reduce MAX** 跨 DP rank 统一 micro-batch 数（保证前向/后向步数一致）
3. **Karmarkar-Karp 分区** 对每个 step 的样本进行序列长度均衡分配到各 micro-batch
4. 同样应用于 log_probs 计算阶段

### 3. Packed Sequence (VarLen Attention)

所有 sample 拼接为一个长序列，使用 `cu_seqlens` 记录每条样本的边界：

```
tokens:    [s1_tok1, s1_tok2, ..., s2_tok1, s2_tok2, ..., pad, pad]
cu_seqlens: [0,       len(s1),     len(s1)+len(s2),    ..., total+pad]
```

配合 FlashAttention 的 `PackedSeqParams(qkv_format="thd")`，避免 padding 浪费。总长度对齐到 128 的倍数以减少显存碎片。

### 优化效果总结

| 优化手段 | 解决的问题 | 关键参数 | 代码位置 |
|---------|-----------|---------|---------|
| 序列长度均衡 | DP rank 间负载不均 | `args.balance_data` | [data.py:289-319](file:///home/robomaster/Research/TritonForge/SLIME/slime/backends/megatron_utils/data.py#L289-L319) |
| 动态 micro-batch | 长序列 OOM | `args.use_dynamic_batch_size`, `args.max_tokens_per_gpu` | [data.py:163-247](file:///home/robomaster/Research/TritonForge/SLIME/slime/backends/megatron_utils/data.py#L163-L247) |
| Packed Sequence | padding 浪费 | 默认启用 | [data.py:46-103](file:///home/robomaster/Research/TritonForge/SLIME/slime/backends/megatron_utils/data.py#L46-L103) |
| Karmarkar-Karp 分区 | NP-hard 均衡问题的高效近似 | 内部使用 | [seqlen_balancing.py](file:///home/robomaster/Research/TritonForge/SLIME/slime/utils/seqlen_balancing.py) |
