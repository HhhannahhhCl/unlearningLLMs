#!/bin/bash
#SBATCH --job-name=unlearning
#SBATCH --account=ec403
#SBATCH --partition=accel    # To use the accelerator nodes
#SBATCH --gpus=a100:1
#SBATCH --time=1:00:00
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
module load nlpl-llmtools/06-foss-2022b-Python-3.10.8

MODEL=${1}  # Path to the model, e.g. /fp/projects01/ec403/IN5550/exam/unlearning/semeval25-unlearning-1B-model/
OUT=${2}  # Directory to save the results to.
BATCH=${3}  # Batch size (e.g., 64)

lm_eval --model hf \
    --model_args pretrained=${MODEL} \
    --tasks mmlu \
    --batch_size ${BATCH} \
    --output_path ${OUT}
