#!/usr/bin/env bash
allennlp train \
    --force \
    --serialization-dir models/bidaf_text_only \
    --include-package lqa_framework \
    training_config/bidaf_text_only.jsonnet
