#!/usr/bin/env bash
allennlp train \
    --force \
    --serialization-dir models/text_lstm \
    --include-package lqa_framework \
    lqa_framework/experiments/text_lstm_baseline.jsonnet
