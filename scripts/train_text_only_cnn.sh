#!/usr/bin/env bash
allennlp train \
    --force \
    --serialization-dir models/text_only_cnn \
    --include-package lqa_framework \
    training_config/text_only_cnn.jsonnet
