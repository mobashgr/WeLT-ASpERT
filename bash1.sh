#!/bin/bash
#SBATCH --job-name=yarab
#SBATCH --output=/hits/basement/sdbv/mobashgr/AspERT-WeLT/test/out-%j
#SBATCH --error=/hits/basement/sdbv/mobashgr/AspERT-WeLT/test/err-%j
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=skylake-deep.p

module load CUDA/11.0.2-GCC-9.3.0

. /home/mobashgr/miniconda3/etc/profile.d/conda.sh
conda activate base
python main.py train --config configs/exampl202.conf
