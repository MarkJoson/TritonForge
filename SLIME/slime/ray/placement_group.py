import socket
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .ppo_actor import RayTrainGroup
from .rollout import RolloutGroup


@ray.remote(num_gpus=1)
class InfoActor:
    '''专门用于检查预分配资源结果，得到当前资源分配节点所在位置，用于之后对sglang等资源的分配'''
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)


def _create_placement_group(num_gpus):
    """Create a placement group with the specified number of GPUs. \\
    创建bundle和placement_group，预留ray资源； \\
    创建InfoActor，获得InfoActor的IP+GPU_ID（用于之后的Sglang），而后Kill所有InfoAcor； \\
    按照IP+GPUid排序并返回
    返回值:
        pg: ray.util.placement_group
        pg_recordered_bundle_indices: {}
    """
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    ray.get(pg.ready())
    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]      # [(i, ip, gpu_id)]
    pg_reordered_bundle_indices = [bundle_info[0] for bundle_info in sorted(bundle_infos, key=sort_key)]        # 按照IP和GPUId排序
    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        print(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices


def create_placement_groups(args):
    """Create placement groups for actor and rollout engines.
    debug_train_only: 只有train节点, actor节点数*actor每节点gpu数
    debug_rollout_only: 只有rollout节点
    colocate: 同train_only
    other: rollout节点+train节点
    返回值:
        train actor 对应的 bundle ids
        rollout 对应的 bundle ids
    """

    num_gpus = 0
    if args.debug_train_only:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
    elif args.debug_rollout_only:
        num_gpus = args.rollout_num_gpus
        rollout_offset = 0
    elif args.colocate:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
    else:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node + args.rollout_num_gpus
        rollout_offset = args.actor_num_nodes * args.actor_num_gpus_per_node

    print(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices = _create_placement_group(num_gpus)

    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]

    return {
        "actor": (pg, actor_pg_reordered_bundle_indices),
        "rollout": (pg, rollout_pg_reordered_bundle_indices),
    }


def allocate_train_group(num_nodes, num_gpus_per_node, pg):
    return RayTrainGroup(
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=0.8,
    )


def create_actor_group(args, pg):
    '''创建RayTrainGroup并完成资源分配'''
    actor_model = allocate_train_group(
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pg,
    )
    return actor_model


def create_rollout_group(args, pg):
    return RolloutGroup(args, pg)
