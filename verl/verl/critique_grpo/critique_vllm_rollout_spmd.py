# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from verl.utils.reward_score import hf_math_verify
from verl.mix_src.critique_prompts import generate_critique
from verl.mix_src.refinement_prompts import generate_refinement
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
import copy
from verl.utils.torch_functional import pad_sequence_to_length

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics

def process_refinement_groups(refinement_dicts, num_samples=7):
    """
    Process refinement groups by selecting the best refinement from each group.
    
    Args:
        refinement_dicts: List of refinement dictionaries containing:
            - 'refinement': The refined text
            - 'score': The score of the refinement (1 is best)
            - 'gt': The ground truth
        num_samples: Number of samples per group
        
    Returns:
        List of selected refinements (one per group)
    """
    selected_refinements = []
    refinement_scores = []
    
    # Split into groups of num_samples
    for i in range(0, len(refinement_dicts), num_samples):
        group = refinement_dicts[i:i + num_samples]
        
        # Try to find a perfect score (1.0) first
        perfect_refinements = [item for item in group if item['score'] == 1.0]
        
        if perfect_refinements:
            # If multiple perfect scores, pick the first one
            selected = perfect_refinements[0]
            refine_score = 1
        else:
            # Otherwise select the highest scoring refinement
            selected = max(group, key=lambda x: x['score'])
            refine_score = 0
        
        selected_refinements.append(selected)
        refinement_scores.append(refine_score)
    
    return selected_refinements, refinement_scores

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

def _pre_process_inputs_right_pad(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)
    if len(non_pad_index) == 0:
        return []
    else:
        token_ids = prompt_token_ids[:non_pad_index[-1][0]+1].tolist()
    return token_ids

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class CRITIQUEvLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, 'rope_scaling', None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        limit_mm_per_prompt = None
        if config.get("limit_images", None):  # support for multi-image data
            limit_mm_per_prompt = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.end_token_id = tokenizer.eos_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]
        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)

        ### insert tgt_input_ids

        # Configure sampling parameters
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)

            tgt_input_ids = None
            if 'tgt_input_ids' in prompts.batch: # in train mode
                print("non_tensor_batch keys: ", non_tensor_batch.keys()) # dict_keys(['reward_model', 'target', 'tools_kwargs'])
                from concurrent.futures import ThreadPoolExecutor
                from typing import Dict, Any
                import logging

                logger = logging.getLogger(__name__)

                if self.sampling_params.n > 1 and do_sample:
                    if 'reward_model' in non_tensor_batch.keys():
                        non_tensor_batch["reward_model"] = _repeat_interleave(non_tensor_batch["reward_model"], self.sampling_params.n)
                    if 'target' in non_tensor_batch.keys():
                        non_tensor_batch["target"] = _repeat_interleave(non_tensor_batch["target"], self.sampling_params.n)

                def process_item(args):
                    """Process a single item for critique and refinement generation."""
                    i, data_item, non_tensor_data_item = args
                    try:
                        logger.debug(f"Processing item {i}")
                        
                        response_ids = data_item["initial_response"]
                        # Decode sequences
                        sequences_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                        
                        # Process non-tensor data with safety checks
                        reward_model_data = non_tensor_data_item.get('reward_model', {})
                        target_data = non_tensor_data_item.get('target', [{}])[0] if 'target' in non_tensor_data_item else {}
                        
                        critique_sample = {
                            "question": reward_model_data.get('question', ''),
                            "target": target_data.get('content', ''),
                            "response": sequences_str,
                            "gt": reward_model_data.get('ground_truth', ''),
                            "score": hf_math_verify.compute_score(
                                solution_str=sequences_str,
                                ground_truth=reward_model_data.get('ground_truth', ''),
                            ).get("score", 0.0),
                        }
                        # print("critique_sample: ", critique_sample)
                        
                        # Generate critique and refinement
                        critique_type = self.config.get("critique_type", "simple_gt")
                        critique_sample = generate_critique(critique_sample, critique_type)
                        
                        refinement_prompt, refinement_prompt_ids = generate_refinement(
                            critique_sample, 
                            self.tokenizer
                        )
                        # print("refinement prompt:", refinement_prompt)
                        # print("refinement prompt ids:", refinement_prompt_ids)
                        
                        return i, refinement_prompt_ids
                    
                    except Exception as e:
                        logger.error(f"Error processing item {i}: {str(e)}")
                        raise

                # Make deep copies of the data for thread safety
                init_response = copy.deepcopy(response)
                non_tensor_data = copy.deepcopy(prompts.non_tensor_batch)
                
                logger.debug(f"Data batch size: {len(init_response)}")
                logger.debug(f"Non-tensor data keys: {non_tensor_data.keys()}")
                
                # Validate data structure before processing
                if 'reward_model' not in non_tensor_data:
                    raise ValueError("Missing required 'reward_model' field in non_tensor_data")
                
                if len(init_response) != len(non_tensor_data['reward_model']):
                    raise ValueError(f"Batch size mismatch: tensor data has {len(init_response)} items, non-tensor data has {len(non_tensor_data['reward_model'])}")

                # Prepare arguments for parallel processing
                args = []
                for i in range(len(init_response)):
                    try:
                        args.append((
                            i,
                            {"initial_response": init_response[i]},  # Extract i-th item from each tensor
                            {
                                'reward_model': non_tensor_data['reward_model'][i],
                                'target': non_tensor_data.get('target', [{}])[i] if 'target' in non_tensor_data else {}
                            }
                        ))
                    except IndexError as e:
                        raise IndexError(f"Index {i} out of bounds for either tensor or non-tensor data") from e

                # Process in parallel
                with ThreadPoolExecutor(max_workers=min(96, len(args))) as executor:
                    results = list(executor.map(process_item, args))
                refinement_ids = [item[1] for item in results]

                # print("refinement_ids len:", len(refinement_ids)) # 56

                # Prepare refinement prompts for generation
                refinement_raw_prompt_ids = [{"prompt_token_ids": refinement_id} for refinement_id in refinement_ids]
                # print("refinement_raw_prompt_ids 0:", refinement_raw_prompt_ids[0]["prompt_token_ids"])
                # refinement_raw_prompt_ids 0: [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 1 ...]
                # np.array(
                #     [_pre_process_inputs(self.pad_token_id, refinement_id) for refinement_id in refinement_ids],
                #     dtype=object
                # )
                
                # Generate refined responses
                # set sampling params.n = 1
                refinement_sampling_params = copy.deepcopy(self.sampling_params)
                refinement_sampling_params.n = 1
                with self.update_sampling_params(**kwargs):
                    refinement_outputs = self.inference_engine.generate(
                        prompts=refinement_raw_prompt_ids,
                        sampling_params=refinement_sampling_params,
                        use_tqdm=False,
                    )

                refinement_responses = []
                for refine_output in refinement_outputs:
                    for sample_id in range(len(refine_output.outputs)):
                        refine_sequences_str = self.tokenizer.decode(refine_output.outputs[sample_id].token_ids, skip_special_tokens=True)
                        refinement_responses.append(refine_sequences_str)

                # print("len refinement_responses: ", len(refinement_responses)) # 56

                # print("refinement_responses [0]: ", refinement_responses[0])

                def process_refinement_item(args):
                    """Process a single refinement item."""
                    i, data_item, non_tensor_data_item = args
                    try:
                        logger.debug(f"Processing refinement item {i}")
                        
                        # Process tensor data
                        refinement = data_item['refinement']
                        reward_model_data = non_tensor_data_item.get('reward_model', {})
                        score = hf_math_verify.compute_score(
                                solution_str=refinement,
                                ground_truth=reward_model_data.get('ground_truth', ''),
                            ).get("score", 0.0)
                        return {"refinement": refinement, "score": score, "ground_truth": reward_model_data.get('ground_truth', '')}
                    except Exception as e:
                        logger.error(f"Error processing refinement item {i}: {str(e)}")
                        raise

                # Select high-quality refinements
                refine_args = []
                for i in range(len(refinement_responses)):
                    try:
                        refine_args.append((
                            i,
                            {'refinement': refinement_responses[i]},
                            {
                                'reward_model': non_tensor_data['reward_model'][i],
                                'target': non_tensor_data.get('target', [{}])[i] if 'target' in non_tensor_data else {}
                            }
                        ))
                    except IndexError as e:
                        raise IndexError(f"Index {i} out of bounds for refinement processing") from e
                
                refinement_dicts = []
                with ThreadPoolExecutor(max_workers=min(96, len(refine_args))) as executor:
                    refinement_results = list(executor.map(process_refinement_item, refine_args))
                for refine_item in refinement_results:
                    refinement_dicts.append({
                        'refinement': refine_item["refinement"],
                        'score': refine_item["score"],
                        'gt': refine_item["ground_truth"]
                    })
                # print("First refinement sample:", refinement_dicts[0]) # First refinement sample: {'refinement': '<think>\nOkay, let me try to figure this out again. I think I 
                # print("\nAll refinements:", len(refinement_dicts)) # All refinements: 56
                # Process the refinements
                selected_refinements, refinement_scores = process_refinement_groups(refinement_dicts, num_samples=self.sampling_params.n)
                ## convert to tensor
                # import torch
                # refinement_score_tensor = torch.tensor(refinement_scores, dtype=torch.float32, device=tgt_input_ids.device)
                max_refinement_len = 6144
                # max_refinement_len = 8192
                refinement_input_ids_list = []
                # print("tgt_input_ids device: ", tgt_input_ids.device) # cuda:0

                for refinement in selected_refinements:
                    # Tokenize the refinement text
                    refinement_input_ids = self.tokenizer(
                        refinement['refinement'],  # Access the refinement text from the dictionary
                        add_special_tokens=False,
                        return_tensors='pt'
                    )['input_ids'].to(device=tgt_input_ids.device)  # Match device with target inputs
                    
                    # Pad or truncate to max_refinement_len
                    if refinement_input_ids.size(1) < max_refinement_len:
                        # Right-pad the sequence
                        padding = torch.full(
                            (1, max_refinement_len - refinement_input_ids.size(1)),
                            self.tokenizer.pad_token_id,
                            device=tgt_input_ids.device
                        )
                        refinement_input_ids = torch.cat([refinement_input_ids, padding], dim=1)
                    else:
                        # Truncate if too long
                        refinement_input_ids = refinement_input_ids[:, :max_refinement_len]
                    
                    refinement_input_ids_list.append(refinement_input_ids)

                # Stack all refinements into a single tensor
                if refinement_input_ids_list:
                    refinement_input_ids = torch.cat(refinement_input_ids_list, dim=0)
                else:
                    # Handle case with no refinements
                    refinement_input_ids = torch.empty((0, max_refinement_len), 
                                                    dtype=torch.long,
                                                    device=tgt_input_ids.device)

                tgt_input_ids = refinement_input_ids  # [bsz, tgt_len]
                print("tgt_input_ids shape: ", tgt_input_ids.shape)            # print("tgt_input_ids: ", tgt_input_ids)

                # add eos token id to the end of the target
                tgt_list = [
                    _pre_process_inputs_right_pad(self.pad_token_id, tgt_input_ids[i]) for i in range(batch_size)
                ]
                # NOTE: be careful with the case when tgt_input_ids is empty.
                # where it only contains paddings
                # in this case, we should not add eos token id to the end of the target
                tgt_list = [
                    tgt_list[i] + [self.end_token_id,] if len(tgt_list[i]) > 0 else tgt_list[i]
                    for i in range(batch_size)
                ]
                tgt_list = sum([[tgt_list[i]] * self.sampling_params.n for i in range(len(tgt_list))], [])
                assert self.config.n_prefix <= self.sampling_params.n
                assert len(tgt_list) == self.sampling_params.n * batch_size

                ## process prefix list
                prefix_ratios = []
                for i in range(batch_size):
                    prefix_ratio = 1.0
                    if self.config.n_prefix > 0: # n_prefix = 1 => one refined response
                        prefix_ratios.extend([prefix_ratio] * self.config.n_prefix)
                        prefix_ratios.extend([0.0] * (self.sampling_params.n - self.config.n_prefix)) # self.config.n => total number of responses
                    else:
                        prefix_ratios.extend([prefix_ratio] * self.sampling_params.n) # no refinements, only generated responses
                assert len(prefix_ratios) == len(tgt_list)
                # print("prefix_ratios: ", prefix_ratios)
                # [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                # print("sampling params.n: ", self.sampling_params.n) # 8
                prefix_list = []
                for prefix_ratio, prefix_tgt_ids in zip(prefix_ratios, tgt_list):
                    if prefix_ratio:
                        prefix_list.append(prefix_tgt_ids)
                    else:
                        prefix_list.append([])
            else: # in eval mode, we don't have tgt_input_ids
                tgt_list = None

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            if 'tgt_input_ids' in prompts.batch:
                # put the prefix back to the response
                try:
                    resp_list = [
                        _pre_process_inputs_right_pad(self.pad_token_id, resp)
                        for resp in response
                    ]
                except:
                    breakpoint()
                concat_resp_list = []
                prefix_mask = torch.zeros([len(resp_list), self.config.response_length], dtype=torch.bool).to(idx.device)
                # print("prefix_mask initialization: ", prefix_mask)
                # print("prefix_mask initialization shape: ", prefix_mask.shape) # torch.Size([256, 6144])
                for i, (prefix_tgt_ids, response_ids) in enumerate(zip(prefix_list, resp_list)):
                    prefix_len = min(len(prefix_tgt_ids), self.config.response_length)
                    prefix_mask[i, :prefix_len] = True
                    if prefix_tgt_ids:
                        concat_resp_list.append(torch.tensor(prefix_tgt_ids))
                    else:
                        concat_resp_list.append(torch.tensor(response_ids))
                # print("prefix_mask: ", prefix_mask) 
                # prefix_mask:  
                # tensor([[ True,  True,  True,  ..., False, False, False],
                #         [False, False, False,  ..., False, False, False],
                #         [False, False, False,  ..., False, False, False],
                #         ...,
                #         [False, False, False,  ..., False, False, False],
                #         [False, False, False,  ..., False, False, False],
                #         [False, False, False,  ..., False, False, False]], device='cuda:0')
                # print("prefix_mask shape: ", prefix_mask.shape) # torch.Size([224, 8192])
                # print("prefix_list len: ", min(len(prefix_list[0])))
                # print("config response_length:", self.config.response_length) # 8192
                resp_max_len = max([len(resp) for resp in concat_resp_list])
                tt = torch.ones(len(concat_resp_list), resp_max_len).fill_(self.pad_token_id)
                for i in range(len(concat_resp_list)):
                    tt[i][:len(concat_resp_list[i])] = concat_resp_list[i].clone().detach()
                response = tt.to(idx.device)[:, :self.config.response_length].to(response.dtype)
                print("response: ", response)
                #  response:  tensor([[ 14023,    771,    397,  ..., 128009, 128009, 128009],
                #     [  1271,   1505,    279,  ..., 128009, 128009, 128009],
                #     [  1271,   1505,    279,  ..., 128009, 128009, 128009],
                #     ...,
                #     [  1271,  11886,    420,  ..., 128009, 128009, 128009],
                #     [  1271,   1505,    279,  ..., 128009, 128009, 128009],
                #     [  1271,   1505,    279,  ..., 128009, 128009, 128009]],
                #     device='cuda:0')
                print("response size: ", response.size()) # torch.Size([224, 8192])
            else:
                prefix_mask = torch.tensor([]) # empty dummy tensor

            # Pad sequences if needed
            if response.shape[1] < self.config.response_length:
                response = pad_sequence_to_length(
                    response, self.config.response_length, self.pad_token_id)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                if tgt_input_ids is not None:
                    # Repeat each refinement according to sampling params
                    refinement_input_ids = _repeat_interleave(refinement_input_ids, self.sampling_params.n)
                    # tgt_input_ids = _repeat_interleave(tgt_input_ids, self.sampling_params.n)
                else:
                    refinement_input_ids = None
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(non_tensor_batch["multi_modal_inputs"], self.sampling_params.n)
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # Then your target input IDs handling
        if tgt_input_ids is not None:
            batch['tgt_input_ids'] = refinement_input_ids
        if prefix_mask.shape[0] > 0:
            batch['prefix_mask'] = prefix_mask
            # batch["tgt_scores"] = refinement_score_tensor
            # print("refinement_scores: ", refinement_score_tensor)
            # print("refinement score shape: ", refinement_score_tensor.shape)
            # tgt_input_ids:  tensor([[ 32313,     11,   1077,  ..., 151643, 151643, 151643],
            #         [ 32313,     11,   1077,  ..., 151643, 151643, 151643],
            #         [ 32313,     11,   1077,  ..., 151643, 151643, 151643],
            #         ...,
            #         [ 32313,     11,   1077,  ..., 151643, 151643, 151643],
            #         [ 32313,     11,   1077,  ..., 151643, 151643, 151643],
            #         [ 32313,     11,   1077,  ..., 151643, 151643, 151643]],
            #         device='cuda:0')
        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is intialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
