import warnings
import torch
import json
import time
import gc
import os
import argparse

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

# home directory: /fp/homes01/u01/ec-hannacla

def set_seed(seed_value=9550):
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTHONHASHSEED"] = "9550"
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


### ARGS ###
parser = argparse.ArgumentParser()

# General
parser.add_argument(
    "--model_size", default="1B", type=str, choices=["1B", "7B"], help="Model to use."
)
parser.add_argument("--data", default="/fp/projects01/ec403/IN5550/exam/unlearning/",type=str, help="Path to training data.")
parser.add_argument("--split", default="train", type=str, help="Training or validation data.")
parser.add_argument("--logdir", default="/cluster/work/projects/ec403/ec-hannacla", type=str, help="Logdir.")
parser.add_argument("--output_dir", default="/cluster/work/projects/ec403/ec-hannacla", type=str, help="Output directory.")

# Training arguments
parser.add_argument("--seed", default=9550, type=int, help="Random seed.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of unlearning epochs.")
parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate.")

# Loss
parser.add_argument("--npo_beta", default=0.5, type=float, help="Beta for NPO loss.")
parser.add_argument("--renyi_alpha", default=0.5, type=float, help="Alpha for renyi divergence.")
parser.add_argument(
    "--npo_mult", default=1.0, type=float, help="NPO forget loss multiplier."
)
parser.add_argument(
    "--rt_mult", default=1.0, type=float, help="Cross-entropy retain loss multiplier."
)
parser.add_argument(
    "--renyi_mult", default=1.0, type=float, help="Renyi divergence retain loss multiplier."
)

# Sequential Data Chunking
parser.add_argument("--sequential", default=True, type=bool, help="Sequential training if True.", action=argparse.BooleanOptionalAction)
parser.add_argument("--chunk_size", default=32, type=int, help="Chunk size forget set for sequential training.")
parser.add_argument("--positive_ratio", default=7, type=int, help="Forget-Retain-Ratio. Retain samples = chunk_size*positive_ratio")
parser.add_argument("--split_retain", default=True, type=bool, help="If sequential training, adjust retain set to size of forget set or use entire retain set.", action=argparse.BooleanOptionalAction)

#LoRA
parser.add_argument("--lora_rank", default=16, type=int, help="Rank of the LoRAs.")
parser.add_argument("--lora_alpha", default=64, type=int, help="The LoRA alpha parameter. None means alpha=rank.")

parser.add_argument(
    "--evaluate_every", default=5, type=int, help="Evaluate every n epochs. `-1` means never."
)
parser.add_argument(
    "--save_every", default=-1, type=int, help="Save checkpoint every n epochs. `-1` means never."
)
parser.add_argument(
    "--save_model", default=True, type=bool, help="Save model after training.", action=argparse.BooleanOptionalAction
)
parser.add_argument(
    "--save_logdir_name",
    default=False,
    action='store_true',
    help="Save this run's logdir path to logdir.txt",
)

args = parser.parse_args()

# Random seed
set_seed(args.seed)

# Saving starting time of computation
start = time.time()

args_dict = vars(args)
# Store the training arguments in the output file for future reference
with open(f"{args.output_dir}/training_args.json", "w") as f:
    f.write(json.dumps(args_dict, indent=4))

print(f"Training with the following arguments:\n\n{json.dumps(args_dict, indent=4)}")

model_path = f"/fp/projects01/ec403/IN5550/exam/unlearning/semeval25-unlearning-{args.model_size}-model"

# Loading model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print(f"Original model trainable parameters: {print_number_of_trainable_model_parameters(model)}")


# Dataset
processor = DatasetProcessor(data_dir=args.data, tokenizer=tokenizer, n_samples_per_task=None)
dataset = processor(split=args.split, task='all', split_tasks=False, split_retain=False)
data_collator = NPOPlusDescentDataCollator(tokenizer=tokenizer, padding='longest', pad_to_multiple_of=8)

# LoRA adapter
lora_config = LoraConfig(
    r=args.lora_rank, # Rank
    lora_alpha=args.lora_alpha,
    target_modules=["q_proj", "v_proj", "up_proj", "down_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

print(f"LoRA adapted model trainable parameters: {print_number_of_trainable_model_parameters(model)}")

# Training initialization
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=32,
    #eval_accumulation_steps=1,
    learning_rate=args.learning_rate,
    num_train_epochs=args.epochs,
    logging_steps=4,
    save_strategy="no",
    eval_strategy="no",
    fp16=True,
    report_to ="none",
    include_inputs_for_metrics=True
)

trainer = SequentialUnlearning(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    training_args=training_args,
    forget_dataset=dataset['forget'],
    retain_dataset=dataset['retain'],
    compute_metrics=make_compute_metrics(model, tokenizer, max_samples=32),
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    sequential=args.sequential,
    chunk_size=args.chunk_size,
    positive_ratio=args.positive_ratio,
    npo_beta=args.npo_beta,
    renyi_alpha=args.renyi_alpha,
    npo_mult=args.npo_mult,
    rt_mult=args.rt_mult,
    renyi_mult=args.renyi_mult,
)

trainer.train(split_retain=args.split_retain)

trainer.save_summary(args.output_dir, args.model_size)

if args.save_model == True:
    trainer.save_model(args.output_dir, args.model_size, args.learning_rate, args.split)

end = time.time()

print(f"Training time: {end-start}")
