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


def example_prompt_func(answer, ground_truth, question):
    chat_prompt = [
        {
            "role": "system",
            "content": "You are a science expert. Analyze a student's incorrect solution by comparing it with the correct solution. Identify any errors and provide a detailed step-by-step explanation of mistakes. Conclude with: 'Conclusion: wrong [END]'"
        },
        {
            "role": "user",
            "content": f"""**Question:** {question}

**Student's Answer:** 
{answer}

**Correct Solution:** 
{ground_truth}

**Critique:**"""
        }
    ]
    return chat_prompt

def corr_example_prompt_func(answer, ground_truth, question):
    chat_prompt = [
        {
            "role": "system",
            "content": "You are a science expert. Analyze a student's solution by comparing it with the correct solution. Identify any potential errors and provide a detailed step-by-step explanation, if they exist. Conclude with: 'Conclusion: right [END]'"
        },
        {
            "role": "user",
            "content": f"""**Question:** {question}

**Student's Answer:** 
{answer}

**Correct Solution:** 
{ground_truth}

**Critique:**"""
        }
    ]
    return chat_prompt

def make_api_call(prompt):

    endpoint = ""
    deployment = "gpt-4o"

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key="",
        api_version="",
    )
    # print("prompt: ", prompt)
    # print("input prompt for behavior analyis ...", prompt)
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                    model=deployment,
                    messages=prompt,
                    max_tokens=4096,  # Reduce from 4096 to avoid overload
                    temperature=0.7,
                    top_p=0.75,
                    timeout=30  # Set timeout explicitly
                )

            return response.choices[0].message.content
        except Exception as e:
            if attempt == 2:
                return {"content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"}
            time.sleep(10)


def generate_critique(sample, critique_type):

    #     sample = {
    #     "idx": example["idx"],
    #     "question": example["question"],
    #     # "gt_cot": example["gt_cot"],
    #     "gt": example["gt"],
    #     "response": example["response"],
    #     "critique": example["critique"],
    #     "score": example["score"]
    # }

    if critique_type == "text":
        if sample["score"] > 0:
            critique = "The generated solution is correct, the ground truth is " + sample["gt"] + "."
        else:
            critique_prompt = example_prompt_func(sample["response"], sample["target"], sample["question"])
            critique = make_api_call(prompt=critique_prompt)
            if isinstance(critique, dict):
                critique = "The generated solution is incorrect, the ground truth is " + sample["gt"] + "."
        # # Determine prompt function based on critique type
        # if sample["score"] < 1:
        #     critique_prompt = example_prompt_func(sample["response"], sample["target"], sample["question"])
        #     critique = make_api_call(prompt=critique_prompt)
        # else:
        #     critique_prompt = corr_example_prompt_func(sample["response"], sample["target"], sample["question"])
        #     critique = make_api_call(prompt=critique_prompt)
        # if isinstance(critique, dict):
        #     if sample["score"] > 0:
        #         critique = "The generated solution is correct, the ground truth is " + sample["gt"] + "."
        #     else:
        #         critique = "The generated solution is incorrect, the ground truth is " + sample["gt"] + "."
        sample["critique"] = critique


    elif critique_type == "simple_gt":
        if sample["score"] > 0:
            critique = "The generated solution is correct, the ground truth is " + sample["gt"] + "."
        else:
            critique = "The generated solution is incorrect, the ground truth is " + sample["gt"] + "."
        sample["critique"] = critique

    elif critique_type == "simple":
        if sample["score"] > 0:
            critique = "The generated solution is correct."
        else:
            critique = "The generated solution is incorrect."
        sample["critique"] = critique
        
    print("sample critique: ", sample["critique"])

    return sample





