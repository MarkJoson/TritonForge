import multiprocessing
import random
import time

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from slime.backends.sglang_utils.sglang_engine import SglangEngine
from slime.ray.buffer import Buffer
from slime.ray.ray_actor import RayActor
from slime.utils.http_utils import find_available_port, get_host_info, run_router
from .utils import Lock


@ray.remote
class RolloutRayActor(RayActor):
    def __init__(self, args, rank: int):
        self.args = args
        self.rank = rank

    def init(self, dist_init_addr, port, nccl_port):
        # build infer engine
        self.infer_engine = SglangEngine(
            args=self.args,
            rank=self.rank,
            dist_init_addr=dist_init_addr,
            port=port,
            nccl_port=nccl_port,
        )

        if self.args.offload:
            # offload the engine to the CPU
            self.infer_engine.sleep()

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self.infer_engine.init_process_group(
            master_address, master_port, rank_offset, world_size, group_name, backend
        )

    def update_weights_from_distributed(self, names, dtypes, shapes, group_name):
        return self.infer_engine.update_weights_from_distributed(names, dtypes, shapes, group_name)

    def update_weights_from_tensor(self, ipc_handles):
        return self.infer_engine.update_weights_from_tensor(ipc_handles)

    def reset_prefix_cache(self):
        self.infer_engine.reset_prefix_cache()

    def sleep(self, level=1):
        self.infer_engine.sleep(level=level)

    def wake_up(self):
        self.infer_engine.wake_up()

    def pause_generation(self):
        self.infer_engine.pause_generation()

    def continue_generation(self):
        self.infer_engine.continue_generation()


def create_rollout_engines(args, pg):
    """创建 Rollout 推理引擎，分配 PG 资源，创建 RolloutRayActor 并初始化。

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   四层概念模型: GPU → Actor → Engine → Node               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  GPU:    最小资源单位，placement_group 中每个 bundle 对应 1 个 GPU。         │
    │                                                                         │
    │  Actor:  RolloutRayActor（Ray Remote Actor），是一个单节点进程。            │
    │          每个 Actor 管理 min(rollout_num_gpus_per_engine, 8) 个 GPU。     │
    │          由于单节点最多 8 GPU，Actor 不能跨节点，所以 cap 到 8。              │
    │          Actor 内部启动 SglangEngine，通过 base_gpu_id 指定起始 GPU。       │
    │                                                                         │
    │  Engine: 逻辑推理单元，对应一个完整的 SGLang 推理服务。                       │
    │          rollout_num_gpus_per_engine ≤ 8 时: 1 Engine = 1 Actor         │
    │          rollout_num_gpus_per_engine > 8 时: 1 Engine = 多个 Actor       │
    │            (跨节点 TP/DP，通过 nnodes + dist_init_addr 协调)               │
    │                                                                         │
    │  Node:   物理节点（通常 8 GPU）。多个 Actor 可以在同一节点上运行。             │
    │          placement_group 按 (IP, GPU_ID) 排序，保证同节点 GPU 索引连续。     │
    │                                                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  示例 (rollout_num_gpus=16, rollout_num_gpus_per_engine=4):              │
    │                                                                         │
    │  Node A (8 GPU):  Actor0[GPU 0-3]  Actor1[GPU 4-7]                      │
    │  Node B (8 GPU):  Actor2[GPU 0-3]  Actor3[GPU 4-7]                      │
    │  → 4 个 Actor = 4 个 Engine (每个 Engine = 1 Actor)                      │
    │                                                                         │
    │  示例 (rollout_num_gpus=32, rollout_num_gpus_per_engine=16):            │
    │                                                                         │
    │  Node A (8 GPU):  Actor0[GPU 0-7]  ─┐                                  │
    │  Node B (8 GPU):  Actor1[GPU 0-7]  ─┘→ Engine0 (跨 2 节点, nnodes=2)    │
    │  Node C (8 GPU):  Actor2[GPU 0-7]  ─┐                                  │
    │  Node D (8 GPU):  Actor3[GPU 0-7]  ─┘→ Engine1 (跨 2 节点, nnodes=2)    │
    │  → 4 个 Actor, 但只有 2 个 Engine                                        │
    │                                                                         │
    │  ⚠ 隐含假设: rollout_num_gpus_per_engine 能整除节点 GPU 数 (通常 8)。       │
    │  若节点 GPU 数不是 num_gpu_per_engine 的倍数 (如 6 GPU 节点配 4)，           │
    │  stride 索引 reordered_bundle_indices[i * num_gpu_per_engine] 会越过      │
    │  节点边界，导致 Actor 被调度到错误节点，存在跨节点或资源浪费的风险。              │
    └─────────────────────────────────────────────────────────────────────────┘
    """

    if args.debug_train_only:
        return []

    # 每个 Actor 管理的 GPU 数 = min(用户配置, 8)，因为单个 Actor (Ray进程) 不能跨节点
    num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, 8)
    # Actor 总数 = 总 rollout GPU 数 // 每个 Actor 的 GPU 数
    # 注意: 这里变量名叫 num_engines，但实际创建的是 Actor 数量。
    # 当 rollout_num_gpus_per_engine ≤ 8 时 num_actors == num_engines;
    # 当 rollout_num_gpus_per_engine > 8 时 num_actors > num_engines，
    # 真正的 engine 数在 RolloutGroup.__init__ 中通过 [::nodes_per_engine] 切片得到。
    num_engines = args.rollout_num_gpus // num_gpu_per_engine

    pg, reordered_bundle_indices = pg

    rollout_engines = []
    for i in range(num_engines):
        num_gpus = 0.2
        num_cpus = num_gpus

        # 用 stride=num_gpu_per_engine 在 reordered_bundle_indices 中跳跃索引，
        # 让每个 Actor 定位到其 GPU 块的第一个 bundle 所在节点。
        # 因为排序保证同节点 GPU 连续，所以 Actor 的子进程 (SglangEngine)
        # 会通过 base_gpu_id 占用从该 bundle 开始的 num_gpu_per_engine 个 GPU。
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,  # 子进程继承 PG 资源
            placement_group_bundle_index=reordered_bundle_indices[i * num_gpu_per_engine],
        )

        rollout_engines.append(
            RolloutRayActor.options(            # 一定在一个节点内
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(args, rank=i)
        )

    # ─── 端口分配 ───
    # 每个 Actor 需要 3 组端口:
    #   1. port: SGLang HTTP server 端口
    #   2. nccl_port: NCCL 通信端口
    #   3. dist_init_addr: torch.distributed 初始化地址 (含 6 + dp_size 个连续端口)
    # 同一节点上的多个 Actor 共享一次 get_addr_and_ports() 调用来分配端口，
    # 避免端口冲突。
    #
    # rollout_num_gpus: 总共rollout gpu的个数
    # num_engines_per_node = 同一节点上有多少个 Actor
    # 例: rollout_num_gpus_per_engine=4, 8 GPU/node → 每节点 2 个 Actor
    num_engines_per_node = max(1, min(8, args.rollout_num_gpus) // args.rollout_num_gpus_per_engine)
    addr_and_ports = [{} for _ in range(num_engines)]
    for rank, engine in enumerate(rollout_engines):     # Actors
        
        # 这里每个node只会调用一次get_addr_and_ports
        if rank % num_engines_per_node != 0:
            continue
        
        def get_addr_and_ports():
            # use small ports to prevent ephemeral port between 32768 and 65536.
            start_port = 10000

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports()

        for i in range(num_engines_per_node):
            addr_and_ports[rank + i]["port"] = get_port()           # port 和 nccl_port 共用一个 port
            addr_and_ports[rank + i]["nccl_port"] = get_port()

        if args.rollout_num_gpus_per_engine > 8:                            # 跨节点 Engine
            # 一个 Engine 由多个 Actor 组成，它们共享同一个 dist_init_addr
            num_node_per_engine = args.rollout_num_gpus_per_engine // 8     # 每个 Engine 需要的 Actor(节点) 数
            if rank % num_node_per_engine == 0:
                # 只在 Engine 的第一个 Actor 上分配 dist_init_addr
                dist_init_addr = f"{get_addr()}:{get_port(6 + args.sglang_dp_size)}"    # 每个engine分配6+DP_Size个端口
                for i in range(num_node_per_engine):
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            # 单节点 Engine: 每个 Actor 独立一个 Engine，各有自己的 dist_init_addr
            for i in range(num_engines_per_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(6 + args.sglang_dp_size)}"

    for i in range(num_engines):
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        print(f"Ports for engine {i}: {addr_and_ports[i]}")

    # TODO: don't ray.get here to overlap train actor init with rollout engine init.
    # somehow if we don't sync here, the --debug-rollout-only mode will crash.
    # 创建 SglangEngine 进行初始化
    init_handles = [engine.init.remote(**ports) for engine, ports in zip(rollout_engines, addr_and_ports)]
    ray.get(init_handles)

    return rollout_engines


class RolloutGroup:
    """管理 SGLang Router、推理引擎集合和数据 Buffer。

    关键属性:
        all_rollout_engines: 所有 RolloutRayActor (Actor 粒度)
        rollout_engines:     去重后的 Engine 粒度列表 (每个 Engine 只保留第一个 Actor)
        data_buffer:         数据缓冲区 (Ray Remote, 仅 CPU)
    """
    def __init__(self, args, pg):
        self.args = args
        # sglang router
        self.start_router()
        
        
        self.data_buffer = Buffer.options(
            num_cpus=1,
            num_gpus=0,
        ).remote(args)

        # create_rollout_engines 返回的是 Actor 粒度的列表
        self.all_rollout_engines = create_rollout_engines(args, pg)
        # 从 Actor 列表中按 stride 提取 Engine 粒度的代表 Actor:
        # nodes_per_engine = 每个 Engine 包含多少个 Actor (跨节点时 >1)
        # 例: rollout_num_gpus_per_engine=16 → nodes_per_engine=2
        #     all_rollout_engines = [A0, A1, A2, A3]
        #     rollout_engines = [A0, A2]  (Engine0 和 Engine1 的主 Actor)
        nodes_per_engine = max(1, args.rollout_num_gpus_per_engine // 8)
        # 跨节点 Engine 中，只向第一个 Actor (node-0) 发送请求
        self.rollout_engines = self.all_rollout_engines[::nodes_per_engine]
        self.rollout_engine_lock = Lock.options(
            num_cpus=1,
            num_gpus=0,
        ).remote()

    def start_router(self):
        '''在当前节点启动一个sglang router...'''
        if self.args.sglang_router_ip is not None:
            return

        from sglang_router.launch_router import RouterArgs

        self.args.sglang_router_ip = get_host_info()[1]
        self.args.sglang_router_port = find_available_port(random.randint(3000, 4000))

        router_args = RouterArgs(
            host=self.args.sglang_router_ip,
            port=self.args.sglang_router_port,
            balance_abs_threshold=0,
        )

        if hasattr(router_args, "log_level"):
            router_args.log_level = "warn"

        process = multiprocessing.Process(
            target=run_router,
            args=(router_args,),
        )
        process.daemon = True  # Set the process as a daemon
        process.start()
        # Wait 3 seconds
        time.sleep(3)
        assert process.is_alive()
        # If router ip is specified, use the specified launched router
        print(f"SGLang router launched at {self.args.sglang_router_ip}:{self.args.sglang_router_port}")

    def async_generate(self, rollout_id, evaluation=False):
        return self.data_buffer.generate.remote(rollout_id, evaluation=evaluation)

    def async_reset_prefix_cache(self):
        return [engine.reset_prefix_cache.remote() for engine in self.rollout_engines]

    def async_offload(self):
        return [engine.sleep.remote() for engine in self.rollout_engines]

    def async_onload(self):
        return [engine.wake_up.remote() for engine in self.rollout_engines]
