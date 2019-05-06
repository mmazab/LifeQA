#!/usr/bin/env bash
allennlp train \
    --force \
    --serialization-dir models/tgif_qa \
    --include-package lqa_framework \
    training_config/tgif_qa.jsonnet
