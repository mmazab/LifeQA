#!/usr/bin/env bash
allennlp train \
    --force \
    --serialization-dir models/memn2n \
    --include-package lqa_framework \
    training_config/text_mem_lstm.jsonnet
