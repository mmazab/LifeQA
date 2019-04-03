#!/usr/bin/env bash
allennlp train \
    --force \
    --serialization-dir models/attention_baseline \
    --include-package lqa_framework \
    lqa_framework/experiments/text_attention_baseline.jsonnet