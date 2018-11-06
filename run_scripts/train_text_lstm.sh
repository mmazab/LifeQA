#!/usr/bin/env bash
model_folder=models/text_lstm

rm -r "${model_folder}" 2> /dev/null
allennlp train lqa_framework/experiments/text_lstm_baseline.jsonnet -s "${model_folder}" --include-package lqa_framework
