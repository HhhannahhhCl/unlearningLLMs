import warnings
import torch
import json
import time
import gc
import os
import argparse
import pandas as pd
import numpy as np

from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from data import DatasetProcessor
from utils import (
    make_compute_metrics,
    preprocess_logits_for_metrics,
    print_number_of_trainable_model_parameters,
    print_gpu_memory,
    plot_metrics,
    plot_training_stats
)
#from utils.evaluation import (
#    QualitativeEvaluation,
#    QuantitativeEvaluation,
#    MMLU
#)
from sequential import NPOPlusDescentDataCollator, SequentialUnlearning

warnings.filterwarnings('ignore')

def gen(question, answer):
    input_ids = tokenizer(
                    question,
                    return_tensors='pt'
                ).input_ids

    #combined_input_ids = tokenizer(
    #    question+answer,
    #    return_tensors='pt'
    #).input_ids
    #combined_target_ids = combined_input_ids.clone()
    #combined_target_ids[:,:len(input_ids[0])] = -100

    #with torch.no_grad():
    #    out = model.generate(input_ids, max_new_tokens=512, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    #    output_ids = out[:, len(input_ids[0]):]
    #
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=512, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    print(out)
    output = tokenizer.decode(
            out[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)

    print(f'Input: {question} \n Answer: {answer} \n Output: {output}')

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_path", type=str, help="Model to use."
)

args = parser.parse_args()

retain_tr = pd.read_json(f'/fp/projects01/ec403/IN5550/exam/unlearning/train/retain.jsonl', lines=True)
forget_tr = pd.read_json(f'/fp/projects01/ec403/IN5550/exam/unlearning/train/forget.jsonl', lines=True)

retain_val = pd.read_json(f'/fp/projects01/ec403/IN5550/exam/unlearning/validation/retain.jsonl', lines=True)
forget_val = pd.read_json(f'/fp/projects01/ec403/IN5550/exam/unlearning/validation/forget.jsonl', lines=True)

model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code = True) # .to('cuda')

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.pad_token = tokenizer.eos_token

manual_rt = retain_val[retain_val['task'] =='Task1'].sample(1)
manual_rt = pd.concat([manual_rt, retain_val[retain_val['task'] =='Task2'].sample(1)])
manual_rt = pd.concat([manual_rt,retain_val[retain_val['task'] =='Task3'].sample(1)])

manual_fg = forget_val[forget_val['task'] =='Task1'].sample(1)
manual_fg = pd.concat([manual_fg, forget_val[forget_val['task'] =='Task2'].sample(1)])
manual_fg = pd.concat([manual_fg,forget_val[forget_val['task'] =='Task3'].sample(1)])


for i in range(len(manual_rt)):
    print(f'Retain Set - Task {manual_rt["task"].iloc[i]} with id {manual_rt["id"].iloc[i]} \n')
    gen(manual_rt['input'].iloc[i], manual_rt['output'].iloc[i])
    print('\n')

for i in range(len(manual_fg)):
    print(f'Forget Set - Task {manual_fg["task"].iloc[i]} with id {manual_fg["id"].iloc[i]}')
    gen(manual_fg['input'].iloc[i], manual_fg['output'].iloc[i])
    print('\n')
