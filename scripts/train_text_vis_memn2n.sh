#!/usr/bin/env bash
allennlp train \
    --force \
    --serialization-dir models/memn2n_multimodal \
    --include-package lqa_framework \
    training_config/text_vis_memn2n.jsonnet
