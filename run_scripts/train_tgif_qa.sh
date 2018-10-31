#!/usr/bin/env bash
model_folder=models/tgif_qa

rm -r "${model_folder}" 2> /dev/null
allennlp train lqa_framework/experiments/tgif_qa.json -s "${model_folder}" --include-package lqa_framework
