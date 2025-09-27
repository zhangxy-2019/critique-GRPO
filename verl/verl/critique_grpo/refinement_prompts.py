import random
import os
from tqdm import tqdm
import copy
from openai import AzureOpenAI
import requests
import time
from httpx import TimeoutException, ConnectError
import glob
import json

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_sequence_to_length


def refine_prompt(question, failed_solution, critique):

    prompt =  f"""Given the following inputs:
        **Question**: {question}
        **Previous Solution**: {failed_solution}
        **Critique**: {critique}

        Please re-answer by:
        - Correcting potential errors identified in the critique, if they exist.
        - Providing clear, step-by-step reasoning.
        - Formatting the final answer as `\\boxed{{answer}}`.

        Ensure the revised solution addresses all issues raised in the critique."""
    return prompt


def generate_refinement(sample, tokenizer):

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": refine_prompt(
                sample["question"], 
                sample["response"], 
                sample["critique"]
            )}
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    raw_prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    return prompt, raw_prompt_ids





