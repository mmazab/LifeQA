#!/usr/bin/env bash
rm -r models/cnn 2> /dev/null
allennlp train lqa_framework/experiments/lstm_baseline.json -s models/cnn --include-package lqa_framework
