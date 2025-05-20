#!/bin/bash
#SBATCH --job-name=unlearning
#SBATCH --account=ec403
#SBATCH --partition=accel    # To use the accelerator nodes
#SBATCH --gpus=a100:1
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8

source ${HOME}/.bashrc

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/

module load nlpl-transformers/4.47.1-foss-2022b-Python-3.10.8
module load nlpl-datasets/3.2.0-foss-2022b-Python-3.10.8
module load nlpl-accelerate/0.34.2-foss-2022b-Python-3.10.8
module load nlpl-nlptools/04-foss-2022b-Python-3.10.8
module load nlpl-trl/0.15.2-foss-2022b-Python-3.10.8

DATA=${1} # Path to data, e.g., /fp/projects01/ec403/IN5550/exam/unlearning/validation/
MODEL=${2} # The model you want to evaluate, e.g., /fp/projects01/ec403/IN5550/exam/unlearning/semeval25-unlearning-1B-model
MIA=${3} # Path to MIA data, e.g., /fp/projects01/ec403/IN5550/exam/unlearning/mia_data/
MMLU=${4} # Path to the MMLU metrics json file, e.g., /fp/projects01/ec403/IN5550/exam/unlearning/mmlu_results_1B.jsonl
TOKENS=${5} # Max length of the generated sequences, e.g., 256
BATCH=${6} # Batch size, e.g., 64
OUTPUT=${7} # model name as output directory

python3 evaluate_generations.py --data_path ${DATA} \
    --checkpoint_path ${MODEL} \
    --mia_data_path ${MIA} \
    --mmlu_metrics_file_path ${MMLU} \
    --max_new_tokens ${TOKENS} \
    --batch_size ${BATCH} \
    --output_dir output/${OUTPUT}
