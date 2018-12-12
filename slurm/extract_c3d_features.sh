#!/bin/bash
#
#SBATCH --job-name=c3d_feature_extraction
#SBATCH --output=logs/c3d_feature_extraction
#
#SBATCH --mail-user=stroud@umich.edu
#SBATCH --mail-type=FAIL
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#
#SBATCH --workdir=/data/home/stroud/LifeQA

# launch run
python feature_extraction/extract_features.py --network c3d
