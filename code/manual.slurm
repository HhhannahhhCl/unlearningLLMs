#!/bin/bash

#SBATCH --job-name=IN9550
#SBATCH --account=ec403
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=accel --gpus=1
#SBATCH --exclude=gpu-14,gpu-4

# Increase this number when you really need parallel computing
#SBATCH --cpus-per-task=32

source ${HOME}/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge 
module use -a /fp/projects01/ec30/software/easybuild/modules/all/

module load nlpl-transformers/4.47.1-foss-2022b-Python-3.10.8
module load nlpl-llmtools/06-foss-2022b-Python-3.10.8
module load nlpl-datasets/3.2.0-foss-2022b-Python-3.10.8
module load nlpl-nlptools/04-foss-2022b-Python-3.10.8
module load nlpl-trl/0.15.2-foss-2022b-Python-3.10.8

echo $SUBMITDIR
echo "unlearning LLMs"

# We pass as arguments the path to the training corpus and (optionally) the number of CPU cores we want to use
python3 manual.py ${@}
