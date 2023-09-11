#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=40GB
#SBATCH --mail-user=<omitted>
#SBATCH --mail-type=FAIL,END
#SBATCH -o out/assignemnt_1a-%j
#SBATCH --export=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fall23_class

mkdir -p /scratch/general/vast/<omitted>/huggingface_cache
export TRANSFORMERS_CACHE="/scratch/general/vast/<omitted>/huggingface_cache"
export HF_DATASETS_CACHE="/scratch/general/vast/<omitted>/huggingface_cache"

OUT_DIR=/scratch/general/vast/<omitted>/fall23_cache/assignment1/models
python assignment_1.py --output_dir ${OUT_DIR} --do_test_eval