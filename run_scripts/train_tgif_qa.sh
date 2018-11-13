#!/usr/bin/env bash
allennlp train \
    --force \
    --serialization-dir models/tgif_qa \
    --include-package lqa_framework \
    lqa_framework/experiments/tgif_qa.jsonnet
