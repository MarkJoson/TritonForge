import asyncio
import time
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from transformers import AutoTokenizer

import wandb
from slime.ray.buffer import Buffer
from slime.utils.async_utils import run
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.types import Sample

__all__ = ["generate_agent_rollout"]


# Global variables for evaluation
TOKENIZER = None
START_ROLLOUT = True        # 理论上，START_ROLLOUT只在整个程序中开启一次（第一次调用generate时）之后就是一直等待。


def select_rollout_data(args, results, need_length):
    """
    排序，筛选
    Select the most recent groups when there are too many samples.
    Groups all samples by instance_id, sorts groups by timestamp.

    Args:
        args: Arguments containing configuration
        results: List of rollout data items with timestamps

    Returns:
        Selected samples from the newest groups based on timestamp cutoff
    """
    if not results:
        return results

    # Group samples by instance_id
    groups = {}         # {instance_id: [samples, ...]}
    for item in results:
        assert "instance_id" in item, "instance_id must be in item"
        instance_id = item["instance_id"]
        if instance_id not in groups:
            groups[instance_id] = []
        groups[instance_id].append(item)

    print(f"📊 Total groups: {len(groups)}, total samples: {len(results)}")

    # If we don't have too many samples, return all
    assert need_length < len(results), "need_length must be smaller than results length"

    # Get timestamp for each group (use the latest timestamp in the group)
    def get_group_timestamp(group_items):
        '''返回一个group中最后的sample的timestamp'''
        timestamps = []
        for item in group_items:
            if "timestamp" in item:
                timestamps.append(float(item["timestamp"]))
            elif "extra_info" in item and "timestamp" in item["extra_info"]:
                timestamps.append(float(item["extra_info"]["timestamp"]))
        return max(timestamps) if timestamps else 0

    # Create list of (group_id, timestamp, samples) and sort by timestamp
    group_data = []     # [(group_id, group_timestamp, [sampls, ...]), ...]
    for group_id, group_items in groups.items():
        group_timestamp = get_group_timestamp(group_items)
        group_data.append((group_id, group_timestamp, group_items))

    # Sort groups by timestamp (newest first)
    # 对group按时间排序
    group_data.sort(key=lambda x: x[1], reverse=True)

    selected_groups = group_data[:need_length]

    # Flatten selected groups back to sample list
    # 展开group到一个list
    selected_results = []
    for group_id, timestamp, group_items in selected_groups:
        selected_results.extend(group_items)        # [samples, ..., ]

    # Statistics for monitoring
    if selected_groups:
        newest_ts = selected_groups[0][1]
        oldest_ts = selected_groups[-1][1]
        print(f"📈 Selected {len(selected_groups)} groups with {len(selected_results)} samples")
        print(f"📈 Group timestamp range: {oldest_ts:.2f} to {newest_ts:.2f}")
        print(f"📈 Time span: {newest_ts - oldest_ts:.2f} seconds")

    return selected_results


def log_raw_info(args, all_meta_info, rollout_id):
    final_meta_info = {}
    if all_meta_info:
        final_meta_info = {
            "total_samples": sum(meta["total_samples"] for meta in all_meta_info if "total_samples" in meta)
        }

        total_samples = final_meta_info["total_samples"]
        if total_samples > 0:
            weighted_reward_sum = sum(
                meta["avg_reward"] * meta["total_samples"]
                for meta in all_meta_info
                if "avg_reward" in meta and "total_samples" in meta
            )

            final_meta_info.update(
                {
                    "avg_reward": weighted_reward_sum / total_samples,
                }
            )
            if hasattr(args, "use_wandb") and args.use_wandb:
                log_dict = {
                    f"rollout/no_filter/total_samples": final_meta_info["total_samples"],
                    f"rollout/no_filter/avg_reward": final_meta_info["avg_reward"],
                }
                try:
                    if args.use_wandb:
                        log_dict["rollout/step"] = (
                            rollout_id
                            if not args.wandb_always_use_train_step
                            else rollout_id
                            * args.rollout_batch_size
                            * args.n_samples_per_prompt
                            // args.global_batch_size
                        )
                        wandb.log(log_dict)
                    print(f"no filter rollout log {rollout_id}: {log_dict}")
                except Exception as e:
                    print(f"Failed to log to wandb: {e}")
                    print(f"no filter rollout log {rollout_id}: {final_meta_info}")
            else:
                print(f"no filter rollout log {rollout_id}: {final_meta_info}")


async def get_rollout_data(
    api_base_url: str, num: Optional[int] = None, timeout: float = 100.0
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    '''实际上，num没有传参=None, 会取尽可能多的已有数据。返回 data: List[samples...] ：\\
    meta_info的数据则来自于：
    slime_plugins
        .rollout_buffer
        .generator
        .utils
        .default_func
        .default_get_group_data_meta_info
    
    data: [{\\
        "uid:...,\\
        "instance_id":...,\\
        "messages":...,\\
        "reward":...,\\
        "extra_info":...\\
    }, ...]
    
    meta_info:{
        "total_samples": 0,
        "num_groups": 0,
        "avg_group_size": 0,
        "avg_reward": 0,
        "reward_std": 0,
        "reward_min": 0,
        "reward_max": 0,
    }
    '''

    url = f"{api_base_url}/get_rollout_data"
    payload = {}

    if num is not None:
        payload["batch_size"] = num
    print(url)
    try:
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            while True:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    response.raise_for_status()
                    resp_json = await response.json()
                    if resp_json["success"]:
                        break
                await asyncio.sleep(3)
                if time.time() - start_time > 30:
                    print("rollout data is not ready, have been waiting for 30 seconds")
                    # Reset start_time to continue waiting or handle timeout differently
                    start_time = time.time()  # Or raise an exception, or return empty list
            '''
            resp_json = {
                "data": {
                    "data":{
                        "uid:...,
                        "instance_id":...,
                        "messages":...,
                        "reward":...,
                        "extra_info":...
                    },
                    "meta_info":...,
                }
            }
            '''
            
            data = resp_json["data"]
            meta_info = {}
            if type(data) is list:
                if "data" in data:
                    data = [item["data"] for item in data]
            elif type(data) is dict:
                if "data" in data:
                    meta_info = data["meta_info"]
                    data = data["data"]
            print(f"Meta info: {meta_info}")
            required_keys = {"uid", "instance_id", "messages", "reward", "extra_info"}
            for item in data:
                if not required_keys.issubset(item.keys()):
                    raise ValueError(f"Missing required keys in response item: {item}")

            return data, meta_info

    except aiohttp.ClientError as e:
        print(f"[ERROR] Request failed: {e}")
        raise
    except ValueError as ve:
        # print(f"[ERROR] Invalid data format: {ve}")
        raise
    except asyncio.TimeoutError:
        print(f"[ERROR] Request timed out after {timeout} seconds")
        raise


def start_rollout(api_base_url: str, args, metadata):
    '''Rollout将交给 slime_plugins.rollout_buffer.buffer 单独处理。不提供prompt, 只提供数据集地址, 由buffer自行采样。
    发起Rollout后，将直接返回。之后轮循查询。
    '''
    
    url = f"{api_base_url}/start_rollout"
    print(f"metadata: {metadata}")
    finished_groups_instance_id_list = [item for sublist in metadata.values() for item in sublist]
    payload = {
        "num_process": str(getattr(args, "rollout_num_process", 100)),
        "num_epoch": str(args.num_epoch or 3),
        "remote_engine_url": f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
        "remote_buffer_url": args.agent_rollout_buffer_url,
        "task_type": args.rollout_task_type,
        "input_file": args.prompt_data,
        "num_repeat_per_sample": str(args.n_samples_per_prompt),
        "max_tokens": str(args.rollout_max_response_len),
        "sampling_params": {
            "max_tokens": args.rollout_max_response_len,
            "temperature": args.rollout_temperature,
            "top_p": args.rollout_top_p,
        },
        "tokenizer_path": args.hf_checkpoint,
        "skip_instance_ids": finished_groups_instance_id_list,
    }

    # Add multi-turn parameters if task type is kernelbench_multiturn
    if "multiturn" in args.rollout_task_type:
        payload["max_turns"] = str(getattr(args, "max_turns", 3))
        payload["gamma"] = str(getattr(args, "gamma", 0.4))
    print("start rollout with payload: ", payload)

    while True:
        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            print(f"[start_rollout] Success: {data}")
            return data
        except Exception as e:
            print(f"[start_rollout] Failed to send rollout config: {e}")


async def generate_agent_rollout(
    args, rollout_id: int, data_buffer: Buffer, evaluation: bool = False
) -> Dict[str, Any]:

    global START_ROLLOUT
    if evaluation:
        raise NotImplementedError("Evaluation rollout is not implemented")

    if START_ROLLOUT:
        # 理论上，START_ROLLOUT只在整个程序中开启一次。
        metadata = data_buffer.get_metadata()
        start_inform = start_rollout(args.agent_rollout_buffer_url, args, metadata)
        print(f"start rollout with payload: {start_inform}")
        print(f"start rollout id: {rollout_id}")
        START_ROLLOUT = False
    
    # 当本地Buffer已经由足够的数据，不需要fetch的时候，就会直接返回数据。
    data_number_to_fetch = (args.rollout_batch_size - data_buffer.get_buffer_length()) * args.n_samples_per_prompt
    if data_number_to_fetch <= 0:
        print(
            f"❕buffer length: {data_buffer.get_buffer_length()}, buffer has enough data, return {args.rollout_batch_size} prompts"
        )
        return data_buffer.get_samples(args.rollout_batch_size)
    
    assert (
        data_number_to_fetch % args.n_samples_per_prompt == 0
    ), "data_number_to_fetch must be a multiple of n_samples_per_prompt"
    
    print(f"INFO: buffer length: {data_buffer.get_buffer_length()}, data_number_to_fetch: {data_number_to_fetch}")
    base_url = args.agent_rollout_buffer_url
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    retry_times = 0
    results = []
    all_meta_info = []

    if args.fetch_trajectory_retry_times == -1:
        print(
            f"⚠️  [get_rollout_data] Fetch trajectory retry times set to -1, will retry indefinitely until sufficient data is collected"
        )
    # 每次取至少 data_number_to_fetch 样本
    while args.fetch_trajectory_retry_times == -1 or retry_times < args.fetch_trajectory_retry_times:
        try:
            while len(results) < data_number_to_fetch:
                time.sleep(5)
                data, meta_info = await get_rollout_data(api_base_url=base_url)
                results.extend(data)
                if meta_info:
                    all_meta_info.append(meta_info)
                print(f"get rollout data with length: {len(results)}")
            break
        except Exception as err:
            print(f"[get_rollout_data] Failed to get rollout data: {err}, retry times: {retry_times}")
            retry_times += 1

    log_raw_info(args, all_meta_info, rollout_id)

    # Apply group-based data selection if there are too many samples
    results = select_rollout_data(args, results, data_number_to_fetch // args.n_samples_per_prompt)
    
    # 更新 meta_info
    if len(all_meta_info) > 0 and "finished_groups" in all_meta_info[0]:
        finished_groups_instance_id_list = []
        for item in all_meta_info:
            finished_groups_instance_id_list.extend(item["finished_groups"])

        data_buffer.update_metadata({str(rollout_id): finished_groups_instance_id_list})

    print("finally get rollout data with length: ", len(results))
    
    # 生成masks，打包成Sample对象
    sample_results = []
    for i, record in enumerate(results):
        oai_messages = record["messages"]
        
        # TODO: 为啥是每个message都要创建一遍？
        mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=args.loss_mask_type)
        # TODO. 当前版本SLIME的Mask有点老了，需要更新了...
        token_ids, loss_mask = mask_generator.get_loss_mask(oai_messages)
        
        # ──── response_length 与 loss_mask 截取逻辑 ────
        #
        # get_loss_mask 返回的 loss_mask 长度 == len(token_ids)，覆盖完整多轮对话：
        #   token_ids = [sys_tokens, user1_tokens, asst1_tokens, user2_tokens, asst2_tokens, ...]
        #   loss_mask = [0, 0, ..., 0, 1, 1, ..., 0, 0, ..., 0, 1, 1, ...]
        #                              ↑ 第一个 1                    ↑ 最后一个 1
        #
        # get_response_lengths 找到第一个 mask=1 的位置，从该位置到末尾的长度即为 response_length。
        # 对于多轮对话，response_length 包含了第一个 assistant 回复到末尾的所有 token（包括中间的 user 消息）。
        #
        # 随后 loss_mask = loss_mask[-response_length:]，只保留尾部 response_length 个 mask 值。
        # 这是因为下游 buffer.py (_convert_samples_to_train_data) 要求：
        #   assert len(sample.loss_mask) == sample.response_length
        #   （见 slime/ray/buffer.py L254-L256）
        #
        # 训练时 (loss.py) 通过 tokens[-response_length:] 取出 response 部分的 token，
        # 再用 loss_mask 逐 token 标记哪些参与损失计算。
        # 头部的 prompt 部分（user/system 消息）不需要 mask，因为整体不参与 loss 计算。
        #
        # 示例：5 条消息 [sys, user, asst, user, asst]
        #   全序列:       [s s s | u u u | 0 0 1 1 1 | u u u | 0 0 1 1 1]
        #   full mask:    [0 0 0 | 0 0 0 | 0 0 1 1 1 | 0 0 0 | 0 0 1 1 1]
        #                                  ↑ 第一个1
        #   response_length = 从第一个1到末尾 = 11
        #   截取后 mask:              [0 0 1 1 1 | 0 0 0 | 0 0 1 1 1]  (长度=11)
        # ────────────────────────────────────────────
        response_length = mask_generator.get_response_lengths([loss_mask])[0]

        loss_mask = loss_mask[-response_length:]

        sample_results.append(
            Sample(
                index=record["instance_id"],
                prompt=record["uid"],
                tokens=token_ids,                       # 完整多轮对话的 token 序列
                response_length=response_length,        # 从第一个 mask=1 到末尾的长度
                reward=record["reward"],
                status=Sample.Status.COMPLETED,         # PENDING / COMPLETED / TRUNCATED / ABORTED
                loss_mask=loss_mask,                    # 长度 == response_length，仅覆盖 response 部分
                metadata={**record["extra_info"], "raw_reward": record["raw_reward"]},
            )
        )
    final_return_results = []

    data_buffer.add_samples(sample_results)
    final_return_results = data_buffer.get_samples(args.rollout_batch_size)

    return final_return_results


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Generate rollout for both training and evaluation."""
    return run(generate_agent_rollout(args, rollout_id, data_buffer, evaluation))
