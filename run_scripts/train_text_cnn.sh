#!/usr/bin/env bash
model_folder=models/text_cnn

rm -r "${model_folder}" 2> /dev/null
allennlp train lqa_framework/experiments/text_cnn_baseline.json -s "${model_folder}" --include-package lqa_framework
