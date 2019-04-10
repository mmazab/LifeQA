#!/usr/bin/env bash
allennlp train \
    --force \
    --serialization-dir models/text_cnn \
    --include-package lqa_framework \
    lqa_framework/experiments/text_cnn_baseline.jsonnet
