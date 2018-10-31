#!/usr/bin/env bash
rm -r models/lstm 2> /dev/null
allennlp train lqa_framework/experiments/lstm_baseline.json -s models/lstm --include-package lqa_framework
