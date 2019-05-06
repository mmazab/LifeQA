#!/usr/bin/env bash

#PBS -N LQA-prep-resnet
#PBS -M sacastro@umich.edu
#PBS -m abe
#PBS -l nodes=1:gpus=1:titanv,pmem=16gb,walltime=40:00:00,qos=flux
#PBS -j oe
#PBS -V
#PBS -A mihalcea_fluxg
#PBS -q fluxg

# Change to the directory you submitted from
if [ -n "$PBS_O_WORKDIR" ]; then cd $PBS_O_WORKDIR; fi

source flux/flux_pre_steps.source
feature_extraction/extract_features.py resnet
